"""Reasoning-replay CONSUMPTION eval: does a provider/model actually read replayed reasoning?

A 200 only proves a carrier was accepted. This harness measures whether it reached the model's
prompt: the prompt-token delta of a request carrying a ~400-token reasoning trace versus the
byte-identical request without it, plus a canary-recall question. Stdlib HTTP only; it measures
the WIRE, not Hermes. See README.md for the method and how to read the matrix.

    python evals/reasoning_replay/harness.py models --route nous
    python evals/reasoning_replay/harness.py run --route nous --models x-ai/grok-4.7 --out DIR
    python evals/reasoning_replay/harness.py run --route custom --kind chat \\
        --base-url https://api.deepseek.com/v1 --key-env DEEPSEEK_API_KEY --models deepseek-chat --out DIR
    python evals/reasoning_replay/harness.py report --out DIR
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
AUTH_JSON = HOME / "auth.json"
DOTENV = HOME / ".env"

# name -> (kind, base_url, credential source). "nous-auth" reads auth.json at call time.
ROUTES = {
    "nous": ("chat", "https://inference-api.nousresearch.com/v1", "nous-auth"),
    "openai-chat": ("chat", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openai-responses": ("responses", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "xai-chat": ("chat", "https://api.x.ai/v1", "XAI_API_KEY"),
    "xai-responses": ("responses", "https://api.x.ai/v1", "XAI_API_KEY"),
    "gemini-native": ("gemini", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
    "gemini-openai": ("chat", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("chat", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "huggingface": ("chat", "https://router.huggingface.co/v1", "HF_TOKEN"),
    "deepseek": ("chat", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "moonshot": ("chat", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "zai": ("chat", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
    "dashscope": ("chat", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "minimax": ("chat", "https://api.minimax.io/v1", "MINIMAX_API_KEY"),
    "fireworks": ("chat", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "groq": ("chat", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "cerebras": ("chat", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "mistral": ("chat", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "local": ("chat", "http://127.0.0.1:8099/v1", ""),
}

SYSTEM = "You are a concise assistant. Use tools when they are needed."
USER1 = "What is the weather in Paris right now? Use the get_weather tool."
# Step 2 turn 1 must make the model think, or newer models emit no reasoning item at all.
USER1_OPAQUE = ("A train leaves at 3:47pm and travels 283 km at 91 km/h. At what exact minute does it arrive? "
                "Then call get_weather for Paris.")
PROBE = ("If your earlier reasoning contained a verification word, reply with only that word; "
         "otherwise reply NONE.")
# xAI's chat/Responses safety layer 403s ("I'm sorry, I can't help with that request.") a user turn
# that asks about earlier *reasoning*; when the baseline 403s the whole position reruns with this.
PROBE_ALT = "If a verification word was mentioned for this task, reply with only that word; otherwise reply NONE."
PROBES = {"std": PROBE, "alt": PROBE_ALT}
TOOL_RESULT = "Paris: 18 C, light rain, wind 12 km/h."
FINAL = "It is 18 C with light rain in Paris."
CALL_ID = "call_rr_0001"
ARGS = '{"city": "Paris"}'
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
GEMINI_DUMMY_SIG = "skip_thought_signature_validator"  # documented validator bypass for synthetic history
TEXT_VARIANTS = ("none", "reasoning_content", "reasoning", "reasoning_details", "all")
RESPONSES_TEXT_VARIANTS = ("none", "summary_item", "content_item")
GEMINI_TEXT_VARIANTS = ("none", "thought_part")
POSITIONS = ("in_loop", "cross_turn")

_CANARIES = ["quillwort", "tamarack", "halyard", "pemmican", "lanolin", "gimbal", "sorrel", "bracken",
             "wimple", "fescue", "marlin", "osprey", "tallow", "zircon", "bramble", "cordage"]
_ADJ = ["amber", "brittle", "coastal", "dappled", "ember", "frosted", "gilded", "hollow", "ivory",
        "jagged", "kindled", "lunar", "molten", "nimble", "opal", "pewter", "quiet", "russet"]
_NOUN = ["barometer", "lighthouse", "isobar", "anemometer", "ledger", "compass", "weathervane",
         "harbor", "sextant", "almanac", "cistern", "gazebo", "trellis", "beacon", "quarry"]


def make_trace(canary: str, seed: int, sentences: int = 14) -> str:
    rng = random.Random(seed)
    out = []
    for i in range(sentences):
        out.append(f"Step {i + 1}: the {rng.choice(_ADJ)} {rng.choice(_NOUN)} suggests checking the "
                   f"{rng.choice(_ADJ)} {rng.choice(_NOUN)} before calling get_weather for Paris, "
                   f"ledger mark {rng.randint(1000, 9999)}.")
        if i == sentences // 2:
            out.append(f"The verification word for this task is {canary}.")
    return " ".join(out)


# ---------------------------------------------------------------- credentials / HTTP

def _dotenv() -> dict:
    env = {}
    if DOTENV.exists():
        for line in DOTENV.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.replace("export ", "").strip()] = v.strip().strip('"').strip("'")
    return env


def _nous_key() -> str:
    n = (json.loads(AUTH_JSON.read_text(encoding="utf-8-sig")).get("providers") or {}).get("nous") or {}
    return n.get("agent_key") or n.get("access_token") or ""


class Budget:
    def __init__(self, path: Path, cap: float):
        self.path, self.cap, self.lock = path, cap, threading.Lock()
        self.spent = 0.0
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                self.spent += json.loads(line).get("usd", 0.0)

    def charge(self, rec: dict) -> None:
        with self.lock:
            self.spent += rec.get("usd", 0.0)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def check(self) -> None:
        if self.spent >= self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f} >= cap ${self.cap}")


class BudgetExceeded(RuntimeError):
    pass


class Client:
    def __init__(self, route: str, kind: str, base_url: str, cred: str, budget: Budget, prices: dict):
        self.route, self.kind, self.base, self.cred = route, kind, base_url.rstrip("/"), cred
        self.budget, self.prices, self.env = budget, prices, _dotenv()

    def _key(self) -> str:
        if self.cred == "nous-auth":
            return _nous_key()
        return os.environ.get(self.cred) or self.env.get(self.cred, "") if self.cred else ""

    def post(self, path: str, body: dict, model: str, tag: str) -> tuple[int, dict, float]:
        self.budget.check()
        data = json.dumps(body).encode()
        last_key, t0 = None, time.time()
        status, j = 0, {}
        for attempt in range(6):
            key = self._key()
            if self.cred == "nous-auth" and key == last_key:
                time.sleep(15)  # live install rotates the key; give it a chance, then re-read
                key = self._key()
            url = self.base + path
            headers = {"Content-Type": "application/json"}
            if self.kind == "gemini":
                headers["x-goog-api-key"] = key
            elif key:
                headers["Authorization"] = "Bearer " + key
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers),
                                            timeout=180) as r:
                    status, raw, hdrs = r.status, r.read().decode("utf-8", "replace"), r.headers
            except urllib.error.HTTPError as e:
                status, raw, hdrs = e.code, e.read().decode("utf-8", "replace"), e.headers
            except Exception as e:  # network/timeouts are recorded as status 0
                status, raw, hdrs = 0, json.dumps({"error": f"{type(e).__name__}: {e}"}), {}
            try:
                j = json.loads(raw)
            except ValueError:
                j = {"_raw": raw[:2000]}
            if isinstance(j, list):  # Gemini wraps some errors in a one-element list
                j = j[0] if j and isinstance(j[0], dict) else {"_raw": raw[:2000]}
            if status == 401 and self.cred == "nous-auth" and last_key is None:
                last_key = key
                continue
            if status in (0, 429, 500, 502, 503, 504, 529) and attempt < 5:
                ra = (hdrs.get("Retry-After") if hdrs else None) or ""
                time.sleep(min(float(ra) if ra.replace(".", "").isdigit() else 4 * (attempt + 1), 65))
                continue
            break
        usd = self._cost(model, j)
        self.budget.charge({"t": time.time(), "route": self.route, "model": model, "tag": tag,
                            "status": status, "usd": usd, "billed": _usage_cost(j) is not None})
        return status, j, time.time() - t0

    def _cost(self, model: str, j: dict) -> float:
        billed = _usage_cost(j)
        if billed is not None:
            return float(billed)
        if not self.cred:  # keyless local server
            return 0.0
        pin, pout = self.prices.get(model) or self.prices.get("*", (5e-6, 25e-6))
        u = norm_usage(self.kind, j)
        return (u.get("prompt") or 0) * pin + ((u.get("completion") or 0) + (u.get("reasoning_extra") or 0)) * pout


def _usage_cost(j: dict):
    """Billed USD when the provider reports it: OpenRouter-style usage.cost, xAI cost_in_usd_ticks (1e-10 USD)."""
    u = j.get("usage") if isinstance(j, dict) else None
    if not isinstance(u, dict):
        return None
    if isinstance(u.get("cost"), (int, float)):
        return u["cost"]
    if isinstance(u.get("cost_in_usd_ticks"), (int, float)):
        return u["cost_in_usd_ticks"] / 1e10
    return None


def norm_usage(kind: str, j: dict) -> dict:
    if not isinstance(j, dict):
        return {}
    if kind == "gemini":
        u = j.get("usageMetadata") or {}
        return {"prompt": u.get("promptTokenCount"), "completion": u.get("candidatesTokenCount"),
                "reasoning": u.get("thoughtsTokenCount"), "reasoning_extra": u.get("thoughtsTokenCount") or 0}
    u = j.get("usage") or {}
    if kind == "responses":
        return {"prompt": u.get("input_tokens"), "completion": u.get("output_tokens"),
                "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens")}
    return {"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
            "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


def lane_of(route: str, j: dict) -> tuple[str, str]:
    """(lane, id-prefix). On the Portal, gen- ids are the OpenRouter lane; anything else is direct."""
    j = j if isinstance(j, dict) else {}
    rid = str(j.get("id") or j.get("responseId") or "")
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", rid):
        prefix = "uuid"
    else:
        m = re.match(r"[A-Za-z]+[-_]?", rid)
        prefix = m.group(0) if m else (rid[:6] or "-")
    if route == "nous":
        return ("openrouter" if rid.startswith("gen-") else "direct"), prefix
    return "direct", prefix


def err_text(j: dict) -> str:
    e = j.get("error") if isinstance(j, dict) else None
    if isinstance(e, dict):
        e = e.get("message") or json.dumps(e)
    if not isinstance(j, dict):
        return str(j)[:400]
    return str(e or j.get("message") or j.get("_raw") or j.get("detail") or json.dumps(j))[:400]


# ---------------------------------------------------------------- fixtures

def chat_text_messages(position: str, variant: str, trace: str, route_opts: dict, probe: str = PROBE) -> list:
    tc: dict = {"id": CALL_ID, "type": "function", "function": {"name": "get_weather", "arguments": ARGS}}
    if route_opts.get("dummy_extra_content"):
        tc["extra_content"] = {"google": {"thought_signature": GEMINI_DUMMY_SIG}}
    asst: dict = {"role": "assistant", "content": "", "tool_calls": [tc]}
    if variant in ("reasoning_content", "all"):
        asst["reasoning_content"] = trace
    if variant in ("reasoning", "all"):
        asst["reasoning"] = trace
    if variant in ("reasoning_details", "all"):
        asst["reasoning_details"] = [{"type": "reasoning.text", "text": trace, "format": "unknown", "index": 0}]
    return _chat_tail([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER1}, asst], position, probe)


def _chat_tail(msgs: list, position: str, probe: str = PROBE) -> list:
    if position == "in_loop":
        return msgs + [{"role": "tool", "tool_call_id": CALL_ID, "content": TOOL_RESULT + "\n\n" + probe}]
    return msgs + [{"role": "tool", "tool_call_id": CALL_ID, "content": TOOL_RESULT},
                   {"role": "assistant", "content": FINAL}, {"role": "user", "content": probe}]


def responses_text_input(position: str, variant: str, trace: str, probe: str = PROBE) -> list:
    items: list = [{"role": "user", "content": USER1}]
    if variant == "summary_item":
        items.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": trace}]})
    elif variant == "content_item":
        items.append({"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": trace}]})
    items.append({"type": "function_call", "call_id": CALL_ID, "name": "get_weather", "arguments": ARGS})
    return _responses_tail(items, position, probe)


def _responses_tail(items: list, position: str, probe: str = PROBE) -> list:
    if position == "in_loop":
        return items + [{"type": "function_call_output", "call_id": CALL_ID, "output": TOOL_RESULT + "\n\n" + probe}]
    return items + [{"type": "function_call_output", "call_id": CALL_ID, "output": TOOL_RESULT},
                    {"role": "assistant", "content": FINAL}, {"role": "user", "content": probe}]


def gemini_text_contents(position: str, variant: str, trace: str, probe: str = PROBE) -> list:
    parts = [{"text": trace, "thought": True}] if variant == "thought_part" else []
    parts.append({"functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                  "thoughtSignature": GEMINI_DUMMY_SIG})
    return _gemini_tail([{"role": "user", "parts": [{"text": USER1}]}, {"role": "model", "parts": parts}],
                        position, probe)


def _gemini_tail(contents: list, position: str, probe: str = PROBE) -> list:
    if position == "in_loop":
        return contents + [{"role": "user", "parts": [{"functionResponse": {
            "name": "get_weather", "response": {"result": TOOL_RESULT, "note": probe}}}]}]
    return contents + [{"role": "user", "parts": [{"functionResponse": {"name": "get_weather",
                                                                       "response": {"result": TOOL_RESULT}}}]},
                       {"role": "model", "parts": [{"text": FINAL}]}, {"role": "user", "parts": [{"text": probe}]}]


# ---------------------------------------------------------------- request bodies

def chat_body(model: str, messages: list, params: dict, max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "tools": TOOLS}
    body[params.get("max_key", "max_tokens")] = max_tokens
    for k in ("temperature", "reasoning", "reasoning_effort"):
        if k in params:
            body[k] = params[k]
    return body


def default_params(route: str, kind: str) -> dict:
    p: dict = {}
    if kind == "responses":
        return {"reasoning": {"effort": "low", "summary": "auto"}}
    if kind == "gemini":
        return {"thinking": {"includeThoughts": True, "thinkingLevel": "low"}, "temperature": 0}
    p["temperature"] = 0
    if route == "nous" or route in ("openrouter",):
        p["reasoning"] = {"effort": "low"}
    elif route in ("openai-chat", "gemini-openai"):
        p["reasoning_effort"] = "low"
    if route == "openai-chat":
        p["max_key"] = "max_completion_tokens"
        p.pop("temperature")
    return p


_PARAM_FIXES = [  # (error substring, mutation) applied to the BASELINE only, never per carrier
    (r"reasoning_effort to 'none'", lambda p: p.__setitem__("reasoning_effort", "none")),
    (r"temperature", lambda p: p.pop("temperature", None)),
    (r"max_tokens", lambda p: p.__setitem__("max_key", "max_completion_tokens")),
    (r"reasoning_effort|reasoningEffort", lambda p: p.pop("reasoning_effort", None)),
    (r"reasoning|effort", lambda p: p.pop("reasoning", None)),
    (r"thinkingLevel|thinking_level|thinking level", lambda p: p.__setitem__(
        "thinking", {"includeThoughts": True, "thinkingBudget": 512})),
    (r"thinking", lambda p: p.pop("thinking", None)),
]


class Prober:
    def __init__(self, client: Client, out: Path, canary: str, trace: str, args):
        self.c, self.out, self.canary, self.trace, self.a = client, out, canary, trace, args
        self.lock = threading.Lock()
        self.route_opts = {"dummy_extra_content": client.route == "gemini-openai"}
        self.model_probe: dict = {}  # model -> probe key that cleared the safety layer in the text step

    def record(self, rec: dict) -> dict:
        with self.lock, (self.out / "calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    # --- one call -> normalized record
    def call(self, model: str, step: str, position: str, variant: str, params: dict, payload, attempt=0,
             extra=None, probe: str = "std", run_id: str = "") -> dict:
        kind = self.c.kind
        if kind == "chat":
            path, body = "/chat/completions", chat_body(model, payload, params, self.a.max_tokens)
        elif kind == "responses":
            path = "/responses"
            body = {"model": model, "instructions": SYSTEM, "input": payload, "tools": [
                {"type": "function", "name": "get_weather", "description": "Current weather for a city.",
                 "parameters": TOOLS[0]["function"]["parameters"]}], "store": False,
                "max_output_tokens": max(self.a.max_tokens, 256), "include": ["reasoning.encrypted_content"]}
            body.update({k: v for k, v in params.items() if k in ("reasoning", "temperature")})
        else:
            path = f"/models/{model}:generateContent"
            gc = {"maxOutputTokens": max(self.a.max_tokens, 256)}
            if "temperature" in params:
                gc["temperature"] = params["temperature"]
            if params.get("thinking"):
                gc["thinkingConfig"] = params["thinking"]
            body = {"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": payload,
                    "tools": [{"functionDeclarations": [{"name": "get_weather", "description": "Current weather.",
                                                         "parameters": TOOLS[0]["function"]["parameters"]}]}],
                    "generationConfig": gc}
        if extra:
            body.update(extra)
        status, j, dt = self.c.post(path, body, model, f"{step}:{position}:{variant}")
        lane, prefix = lane_of(self.c.route, j)
        text, reasoning_text = extract_text(kind, j)
        u = norm_usage(kind, j)
        return self.record({
            "route": self.c.route, "model": model, "step": step, "position": position, "variant": variant,
            "probe": probe, "run_id": run_id, "attempt": attempt, "status": status, "lane": lane, "id_prefix": prefix,
            "provider": j.get("provider") if isinstance(j, dict) else None,
            "prompt_tokens": u.get("prompt"), "completion_tokens": u.get("completion"),
            "reasoning_tokens": u.get("reasoning"), "cached_tokens": u.get("cached"),
            "cost": _usage_cost(j), "error": "" if status == 200 else err_text(j),
            "answer": (text or "")[:200], "canary_in_answer": self.canary in (text or "").lower(),
            "canary_in_reasoning": self.canary in (reasoning_text or "").lower(),
            "secs": round(dt, 2), "t": time.time()})

    def want_direct(self, model: str) -> bool:
        return self.c.route == "nous" and model in self.a.want_direct

    def sample(self, model, step, position, variant, params, payload, extra=None, probe="std", run_id="") -> list:
        """One call, or up to --lane-attempts calls until a direct-lane 200 for want-direct ids."""
        recs = []
        for attempt in range(self.a.lane_attempts if self.want_direct(model) else 1):
            r = self.call(model, step, position, variant, params, payload, attempt, extra, probe, run_id)
            recs.append(r)
            if not (r["status"] == 200 and r["lane"] == "openrouter"):
                break
        return recs

    # --- baseline + param discovery
    def discover_params(self, model: str) -> dict | None:
        params = default_params(self.c.route, self.c.kind)
        payload = self.text_payload("in_loop", "none")
        for _ in range(len(_PARAM_FIXES) + 1):
            r = self.call(model, "params", "in_loop", "none", params, payload)
            if r["status"] == 200:
                return params
            if r["status"] != 400:
                return None
            for pat, fix in _PARAM_FIXES:
                key_before = json.dumps(params, sort_keys=True)
                if re.search(pat, r["error"], re.IGNORECASE):
                    fix(params)
                    if json.dumps(params, sort_keys=True) != key_before:
                        break
            else:
                return None
        return None

    def text_payload(self, position, variant, probe="std"):
        if self.c.kind == "chat":
            return chat_text_messages(position, variant, self.trace, self.route_opts, PROBES[probe])
        if self.c.kind == "responses":
            return responses_text_input(position, variant, self.trace, PROBES[probe])
        return gemini_text_contents(position, variant, self.trace, PROBES[probe])

    def variants(self):
        if self.a.variants:
            return self.a.variants
        return {"chat": TEXT_VARIANTS, "responses": RESPONSES_TEXT_VARIANTS, "gemini": GEMINI_TEXT_VARIANTS}[self.c.kind]

    def run_model(self, model: str) -> None:
        try:
            params = self.discover_params(model)
            if params is None:
                return
            self.record({"route": self.c.route, "model": model, "step": "meta", "params": params,
                         "canary": self.canary, "trace_chars": len(self.trace), "t": time.time()})
            if "text" in self.a.steps:
                for position in self.a.positions:
                    probe = self.a.probe
                    first = self.sample(model, "text", position, "none", params,
                                        self.text_payload(position, "none", probe), probe=probe)
                    if first[-1]["status"] == 403 and probe == "std":
                        probe = "alt"
                        self.sample(model, "text", position, "none", params,
                                    self.text_payload(position, "none", probe), probe=probe)
                    for variant in self.variants():
                        if variant == "none" and self.c.route != "nous":
                            continue  # one baseline is enough off the Portal; Portal gets a second
                        self.sample(model, "text", position, variant, params,
                                    self.text_payload(position, variant, probe), probe=probe)
                    self.backfill_baseline(model, "text", position, params, probe)
                    if probe == "alt":
                        self.model_probe[model] = "alt"
            if "opaque" in self.a.steps:
                self.opaque(model, params)
        except BudgetExceeded as e:
            self.record({"route": self.c.route, "model": model, "step": "budget_stop", "error": str(e)})

    def backfill_baseline(self, model, step, position, params, probe="std"):
        """Every lane a carrier landed on needs a 'none' baseline on that same lane."""
        if self.c.route != "nous":
            return
        recs = [r for r in read_calls(self.out) if r.get("model") == model and r.get("step") == step
                and r.get("position") == position and r.get("status") == 200 and r.get("route") == self.c.route
                and r.get("probe", "std") == probe]
        have = {r["lane"] for r in recs if r["variant"] == "none"}
        need = {r["lane"] for r in recs if r["variant"] != "none"} - have
        for _ in range(3 if need else 0):
            r = self.call(model, step, position, "none", params, self.text_payload(position, "none", probe), 9,
                          probe=probe)
            have.add(r["lane"])
            if not (need - have):
                break

    # --- step 2: real opaque carriers captured from turn 1 of the SAME model
    def opaque(self, model: str, params: dict) -> None:
        kind = self.c.kind
        if kind == "chat":
            return self._opaque_chat(model, params)
        if kind == "responses":
            return self._opaque_responses(model, params)
        return self._opaque_gemini(model, params)

    def _turn1(self, model, params, payload):
        """Live turn 1. Effort escalates (low -> medium -> high) until the model emits a tool call
        AND a reasoning carrier; want-direct ids also resample until a direct-lane response."""
        status, j, lane, prefix = 0, {}, "-", "-"
        efforts = ["low", "medium", "high"] if ("reasoning" in params or "reasoning_effort" in params) else [None]
        tries = 0
        for effort in efforts:
            p = copy.deepcopy(params)
            if effort and "reasoning" in p:
                p["reasoning"] = {"effort": effort}
            elif effort:
                p["reasoning_effort"] = effort
            for _ in range(self.a.lane_attempts if self.want_direct(model) else 1):
                tries += 1
                body = chat_body(model, payload, p, self.a.turn1_max_tokens)
                status, j, _ = self.c.post("/chat/completions", body, model, f"opaque:turn1:{effort}")
                lane, prefix = lane_of(self.c.route, j)
                if status != 200 or lane == "direct" or not self.want_direct(model):
                    break
            if status != 200:
                return status, j, lane, prefix
            msg = (j.get("choices") or [{}])[0].get("message") or {}
            if msg.get("tool_calls") and any(msg.get(k) for k in ("reasoning_content", "reasoning",
                                                                    "reasoning_details")):
                break
        return status, j, lane, prefix

    def _opaque_chat(self, model, params):
        status, j, lane, prefix = self._turn1(model, params, [{"role": "system", "content": SYSTEM},
                                                               {"role": "user", "content": USER1_OPAQUE}])
        msg = ((j.get("choices") or [{}])[0].get("message") or {}) if status == 200 else {}
        u = norm_usage("chat", j)
        rtext = msg.get("reasoning_content") or msg.get("reasoning") or ""
        details = msg.get("reasoning_details") or []
        tcs = msg.get("tool_calls") or []
        run_id = f"{time.time():.3f}"
        t1 = {"route": self.c.route, "model": model, "step": "opaque_turn1", "run_id": run_id, "status": status,
              "lane": lane,
              "id_prefix": prefix, "error": "" if status == 200 else err_text(j),
              "has_tool_call": bool(tcs), "carriers": sorted(k for k in ("reasoning_content", "reasoning",
                                                                          "reasoning_details") if msg.get(k)),
              "detail_formats": sorted({f"{d.get('type')}|{d.get('format')}" for d in details if isinstance(d, dict)}),
              "signed_details": sum(1 for d in details if isinstance(d, dict) and (d.get("signature") or d.get("data"))),
              "extra_content": any(tc.get("extra_content") for tc in tcs),
              "reasoning_chars": len(rtext), "reasoning_sample": rtext[:160], "reasoning_tokens": u.get("reasoning"),
              "completion_tokens": u.get("completion"), "cost": _usage_cost(j), "t": time.time()}
        self.record(t1)
        if not tcs or not (t1["carriers"] or t1["extra_content"]):  # nothing to replay: verbatim == stripped
            return
        base = {"role": "assistant", "content": msg.get("content") or "",
                "tool_calls": [{"id": tc["id"], "type": "function", "function": tc["function"]} for tc in tcs[:1]]}
        verb = copy.deepcopy(base)
        for k in ("reasoning_content", "reasoning", "reasoning_details"):
            if msg.get(k):
                verb[k] = msg[k]
        if tcs[0].get("extra_content"):
            verb["tool_calls"][0]["extra_content"] = tcs[0]["extra_content"]
        call_id = tcs[0]["id"]
        for position in self.a.positions:
            probe = self.model_probe.get(model, self.a.probe)
            for variant, asst in (("stripped", base), ("verbatim", verb)):
                for _ in range(2):  # retry the cell with the alt wording if the safety layer refuses it
                    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER1_OPAQUE}, asst]
                    tail = _chat_tail([], position, PROBES[probe])
                    tail[0]["tool_call_id"] = call_id
                    recs = self.sample(model, "opaque", position, variant, params, msgs + tail, probe=probe,
                                       run_id=run_id)
                    if recs[-1]["status"] != 403 or probe == "alt":
                        break
                    probe = "alt"

    def _opaque_responses(self, model, params):
        body = {"model": model, "instructions": SYSTEM, "input": [{"role": "user", "content": USER1_OPAQUE}],
                "tools": [{"type": "function", "name": "get_weather", "description": "Current weather for a city.",
                           "parameters": TOOLS[0]["function"]["parameters"]}], "store": False,
                "max_output_tokens": self.a.turn1_max_tokens, "include": ["reasoning.encrypted_content"]}
        body.update({k: v for k, v in params.items() if k in ("reasoning", "temperature")})
        status, j, out, rs, fc = 0, {}, [], [], None
        for effort in ("low", "medium", "high"):  # escalate until turn 1 returns a reasoning item + call
            if "reasoning" in body:
                body["reasoning"] = dict(body["reasoning"], effort=effort)
            status, j, _ = self.c.post("/responses", body, model, f"opaque:turn1:{effort}")
            out = (j.get("output") or []) if status == 200 else []
            rs = [o for o in out if o.get("type") == "reasoning" and o.get("encrypted_content")]
            fc = next((o for o in out if o.get("type") == "function_call"), None)
            if status != 200 or (rs and fc) or "reasoning" not in body:
                break
        u = norm_usage("responses", j)
        run_id = f"{time.time():.3f}"
        self.record({"route": self.c.route, "model": model, "step": "opaque_turn1", "run_id": run_id, "status": status,
                     "lane": "direct", "id_prefix": lane_of(self.c.route, j)[1],
                     "error": "" if status == 200 else err_text(j), "has_tool_call": bool(fc),
                     "carriers": ["reasoning_item"] * len(rs),
                     "detail_formats": sorted({"encrypted" if o.get("encrypted_content") else "plain" for o in rs}),
                     "signed_details": sum(1 for o in rs if o.get("encrypted_content")),
                     "reasoning_chars": sum(len(s.get("text", "")) for o in rs for s in o.get("summary") or []),
                     "reasoning_tokens": u.get("reasoning"), "completion_tokens": u.get("completion"),
                     "t": time.time()})
        if not fc or not rs:
            return
        call = {"type": "function_call", "call_id": fc["call_id"], "name": fc["name"], "arguments": fc["arguments"]}
        replay = [{k: v for k, v in o.items() if k not in ("id", "status")} for o in rs]
        probe = self.model_probe.get(model, self.a.probe)
        for position in self.a.positions:
            tail = _responses_tail([], position, PROBES[probe])
            tail[0]["call_id"] = fc["call_id"]
            cells = [("stripped", [call], None), ("verbatim", replay + [call], None)]
            if self.c.route == "openai-responses":
                ctx = {"reasoning": dict(params.get("reasoning") or {}, context="all_turns")}
                cells += [("stripped+all_turns", [call], ctx), ("verbatim+all_turns", replay + [call], ctx)]
            for variant, items, extra in cells:
                p = dict(params)
                if extra:
                    p["reasoning"] = extra["reasoning"]
                self.call(model, "opaque", position, variant, p, [{"role": "user", "content": USER1_OPAQUE}] + items + tail,
                          probe=probe, run_id=run_id)

    def _opaque_gemini(self, model, params):
        body = {"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": [
            {"role": "user", "parts": [{"text": USER1_OPAQUE}]}],
            "tools": [{"functionDeclarations": [{"name": "get_weather", "description": "Current weather.",
                                                 "parameters": TOOLS[0]["function"]["parameters"]}]}],
            "generationConfig": {"maxOutputTokens": self.a.turn1_max_tokens,
                                 **({"thinkingConfig": params["thinking"]} if params.get("thinking") else {})}}
        status, j, _ = self.c.post(f"/models/{model}:generateContent", body, model, "opaque:turn1")
        parts = (((j.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []) if status == 200 else []
        u = norm_usage("gemini", j)
        fcp = next((p for p in parts if "functionCall" in p), None)
        run_id = f"{time.time():.3f}"
        self.record({"route": self.c.route, "model": model, "step": "opaque_turn1", "run_id": run_id, "status": status,
                     "lane": "direct", "id_prefix": "-", "error": "" if status == 200 else err_text(j),
                     "has_tool_call": bool(fcp), "carriers": sorted({("thought" if p.get("thought") else "part")
                                                                     for p in parts}),
                     "detail_formats": ["google-gemini-v1:thoughtSignature"] if any(p.get("thoughtSignature")
                                                                                     for p in parts) else [],
                     "signed_details": sum(1 for p in parts if p.get("thoughtSignature")),
                     "reasoning_chars": sum(len(p.get("text", "")) for p in parts if p.get("thought")),
                     "reasoning_tokens": u.get("reasoning"), "completion_tokens": u.get("completion"),
                     "t": time.time()})
        if not fcp:
            return
        verbatim = {"role": "model", "parts": parts}
        stripped = {"role": "model", "parts": [{"functionCall": fcp["functionCall"]}]}
        dummy = {"role": "model", "parts": [{"functionCall": fcp["functionCall"], "thoughtSignature": GEMINI_DUMMY_SIG}]}
        probe = self.model_probe.get(model, self.a.probe)
        for position in self.a.positions:
            for variant, mturn in (("stripped", stripped), ("dummy_signature", dummy), ("verbatim", verbatim)):
                contents = _gemini_tail([{"role": "user", "parts": [{"text": USER1_OPAQUE}]}, mturn], position,
                                        PROBES[probe])
                self.call(model, "opaque", position, variant, params, contents, probe=probe, run_id=run_id)


def extract_text(kind: str, j: dict) -> tuple[str, str]:
    if not isinstance(j, dict):
        return "", ""
    if kind == "chat":
        m = (j.get("choices") or [{}])[0].get("message") or {}
        tcs = " ".join((tc.get("function") or {}).get("arguments", "") for tc in m.get("tool_calls") or [])
        rd = " ".join(str(d.get("text") or d.get("summary") or "") for d in m.get("reasoning_details") or []
                      if isinstance(d, dict))
        return f"{m.get('content') or ''} {tcs}".strip(), f"{m.get('reasoning_content') or m.get('reasoning') or ''} {rd}"
    if kind == "responses":
        text, reason = [], []
        for o in j.get("output") or []:
            if o.get("type") == "message":
                text += [c.get("text", "") for c in o.get("content") or []]
            elif o.get("type") == "reasoning":
                reason += [s.get("text", "") for s in o.get("summary") or []]
            elif o.get("type") == "function_call":
                text.append(o.get("arguments", ""))
        return " ".join(text), " ".join(reason)
    parts = ((j.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return (" ".join(p.get("text", "") for p in parts if not p.get("thought")),
            " ".join(p.get("text", "") for p in parts if p.get("thought")))


def read_calls(out: Path) -> list:
    p = out / "calls.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()] if p.exists() else []


# ---------------------------------------------------------------- models / prices

def nous_catalog() -> list:
    req = urllib.request.Request("https://inference-api.nousresearch.com/v1/models",
                                 headers={"Authorization": "Bearer " + _nous_key()})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["data"]


def nous_selection(catalog: list) -> tuple[list, list]:
    """Reasoning-capable, text-out, non-alias, non-batch ids sorted by output price; plus skips."""
    keep, skip = [], []
    for m in catalog:
        mid, sp = m["id"], m.get("supported_parameters") or []
        outm = (m.get("architecture") or {}).get("output_modalities") or ["text"]
        if mid.endswith(":batch") or mid.startswith("~"):
            skip.append((mid, "batch/alias of a listed id"))
        elif outm != ["text"]:
            skip.append((mid, "non-text output (image/audio)"))
        elif "reasoning" not in sp and "include_reasoning" not in sp:
            skip.append((mid, "not reasoning-capable (embeddings/instruct)"))
        else:
            pr = m.get("pricing") or {}
            keep.append((float(pr.get("completion") or 0), mid))
    keep.sort()
    return [k[1] for k in keep], skip


def price_table(catalog: list) -> dict:
    """Per-token (in, out) by id; vendor ids also keyed bare (openai/gpt-6-luna -> gpt-6-luna)."""
    t = {}
    for m in catalog:
        pr = m.get("pricing") or {}
        try:
            v = (float(pr.get("prompt") or 0), float(pr.get("completion") or 0))
        except ValueError:
            continue
        t[m["id"]] = v
        t.setdefault(m["id"].split("/", 1)[-1], v)
    t["*"] = (5e-6, 25e-6)
    return t


# ---------------------------------------------------------------- report

# text: delta >= 0.5*trace_tokens => consumed. opaque: the real carrier can be tiny (grok emits ~20
# reasoning tokens), so any verbatim-minus-stripped delta >= OPAQUE_MIN tokens counts.
CONSUMED, OPAQUE_MIN = 0.5, 8


def summarize(out: Path) -> list:
    calls = read_calls(out)
    want_direct: set = set()
    runs = out / "runs.jsonl"
    if runs.exists():
        for line in runs.read_text(encoding="utf-8-sig").splitlines():
            want_direct |= set(json.loads(line).get("want_direct") or [])
    meta = {(c["route"], c["model"]): c for c in calls if c.get("step") == "meta"}
    trace_tok = {}
    rows = {}
    for c in calls:
        if c.get("step") not in ("text", "opaque", "opaque_turn1", "params"):
            continue
        rows.setdefault((c["route"], c["model"]), []).append(c)
    table = []
    for (route, model), recs in sorted(rows.items()):
        lanes = sorted({r.get("lane") for r in recs if r.get("lane")}) or ["-"]
        # opaque cells are only comparable within one turn-1 capture: keep the newest run per lane
        latest = {}
        for r in recs:
            if r.get("step") == "opaque" and r.get("status") == 200:
                latest[r.get("lane")] = max(latest.get(r.get("lane"), ""), r.get("run_id", ""))
        recs = [r for r in recs if r.get("step") != "opaque" or r.get("run_id", "") == latest.get(r.get("lane"), "")
                or (r.get("lane") not in latest)]
        tchars = (meta.get((route, model)) or {}).get("trace_chars") or 1800
        trace_tok[(route, model)] = tchars / 4
        for lane in lanes:
            row = {"route": route, "model": model, "lane": lane, "text": {}, "opaque": {}, "errors": {},
                   "canary": {}, "id_prefixes": sorted({r.get("id_prefix") for r in recs if r.get("lane") == lane}),
                   "calls": sum(1 for r in recs if r.get("lane") == lane)}
            for step in ("text", "opaque"):
                for position in POSITIONS:
                    lr = [r for r in recs if r.get("step") == step and r.get("position") == position
                          and r.get("lane") == lane]
                    base_variant = "none" if step == "text" else "stripped"
                    bases: dict = {}
                    for bv in (base_variant, "dummy_signature"):  # Gemini 400s a stripped signature in-loop
                        for r in lr:
                            if r["variant"] == bv and r["status"] == 200 and r.get("prompt_tokens") is not None:
                                pk = r.get("probe", "std")
                                bases[pk] = min(bases.get(pk, 1 << 30), r["prompt_tokens"])
                        if bases:
                            break
                    # a probe whose baseline was refused (403) is superseded by the alt-probe cells
                    probes_used = {r.get("probe", "std") for r in lr if r["status"] == 200}
                    if "alt" in probes_used:
                        lr = [r for r in lr if r.get("probe", "std") == "alt"]
                    for r in lr:
                        v = r["variant"]
                        b = bases.get(r.get("probe", "std"))
                        if v == base_variant and step == "text":
                            if r["status"] != 200:
                                row["errors"][f"{step}:{position}:baseline"] = f"{r['status']} {r['error'][:200]}"
                            continue
                        cell = row[step].setdefault(position, {})
                        if r["status"] != 200:
                            cell.setdefault(v, f"{r['status']}")
                            row["errors"][f"{step}:{position}:{v}"] = f"{r['status']} {r['error'][:200]}"
                        elif b is not None and r.get("prompt_tokens") is not None:
                            cell[v] = r["prompt_tokens"] - b
                        else:
                            cell.setdefault(v, "nobase")
                        if step == "text" and r["status"] == 200:
                            row["canary"].setdefault(position, {})[v] = (
                                "Y" if r["canary_in_answer"] else ("r" if r["canary_in_reasoning"] else "n"))
            # rejected-on-every-lane errors have lane '-' semantics; fold non-200 rows from any lane
            t1 = [r for r in recs if r.get("step") == "opaque_turn1" and r.get("lane") in (lane, "-")]
            t1 = [r for r in t1 if r.get("run_id", "") == latest.get(lane)] or t1
            if t1:
                r = t1[-1]
                row["turn1"] = {k: r.get(k) for k in ("status", "carriers", "detail_formats", "signed_details",
                                                      "extra_content", "reasoning_chars", "reasoning_sample",
                                                      "reasoning_tokens", "has_tool_call", "error", "lane")}
            row["verdict"] = verdict(row, trace_tok[(route, model)])
            if row["text"] or row["opaque"] or row.get("turn1"):
                table.append(row)
        if route == "nous" and model in want_direct and "direct" not in lanes:
            table.append({"route": route, "model": model, "lane": "direct", "text": {}, "opaque": {}, "canary": {},
                          "errors": {"lane": f"direct lane not reached in {sum(1 for r in recs)} calls (all gen-)"},
                          "verdict": "direct-lane-not-reached", "id_prefixes": [], "calls": 0})
    # calls that never got a 200 baseline (params discovery failed) get an error row
    for (route, model), recs in rows.items():
        if not any(t["route"] == route and t["model"] == model for t in table):
            last = recs[-1]
            table.append({"route": route, "model": model, "lane": last.get("lane", "-"), "text": {},
                          "opaque": {}, "canary": {}, "errors": {"baseline": f"{last['status']} {last['error'][:200]}"},
                          "verdict": "baseline-failed", "id_prefixes": [], "calls": len(recs)})
    return table


def verdict(row: dict, trace_tokens: float) -> str:
    def consumed(cells):
        return [v for v, d in cells.items() if isinstance(d, int) and d >= CONSUMED * trace_tokens]

    def accepted(cells):
        return [v for v, d in cells.items() if isinstance(d, int)]

    il, ct = row["text"].get("in_loop", {}), row["text"].get("cross_turn", {})
    oil, oct_ = row["opaque"].get("in_loop", {}), row["opaque"].get("cross_turn", {})
    opaque_cross = [v for v, d in oct_.items() if v.startswith("verbatim") and isinstance(d, int) and d >= OPAQUE_MIN]
    opaque_in = [v for v, d in oil.items() if v.startswith("verbatim") and isinstance(d, int) and d >= OPAQUE_MIN]
    if consumed(ct) or opaque_cross:
        return "consumes-all-turns"
    if consumed(il) or opaque_in:
        return "in-loop-only"
    if accepted(il) or accepted(ct) or accepted(oil) or accepted(oct_):
        return "accepted-ignored"
    if il or ct or oil or oct_:
        return "rejected"
    return "no-data"


def fmt_cells(cells: dict, order) -> str:
    if not cells:
        return "-"
    short = {"reasoning_content": "rc", "reasoning": "r", "reasoning_details": "rd", "all": "all",
             "summary_item": "sum", "content_item": "cnt", "thought_part": "thought", "verbatim": "verb",
             "dummy_signature": "dummy", "stripped": "strip", "verbatim+all_turns": "verb+ctx",
             "stripped+all_turns": "strip+ctx"}
    keys = [k for k in order if k in cells] + [k for k in cells if k not in order]
    return " ".join(f"{short.get(k, k)}={cells[k]:+d}" if isinstance(cells[k], int) else f"{short.get(k, k)}={cells[k]}"
                    for k in keys)


def render(table: list, out: Path) -> str:
    order = TEXT_VARIANTS + RESPONSES_TEXT_VARIANTS + GEMINI_TEXT_VARIANTS
    lines = ["| route | model | lane | in-loop text delta | cross-turn text delta | opaque in-loop | opaque cross-turn "
             "| turn-1 carriers | canary (il / ct) | errors | verdict |", "|" + "---|" * 11]
    for r in table:
        t1 = r.get("turn1") or {}
        t1s = ""
        if t1:
            t1s = f"{t1.get('status')} {','.join(t1.get('carriers') or [])} {';'.join(t1.get('detail_formats') or [])}"
            if t1.get("signed_details"):
                t1s += f" signed={t1['signed_details']}"
            if t1.get("extra_content"):
                t1s += " extra_content"
            if t1.get("reasoning_tokens") is not None:
                t1s += f" rchars={t1.get('reasoning_chars')}/rtok={t1.get('reasoning_tokens')}"
        sh = {"reasoning_content": "rc", "reasoning": "r", "reasoning_details": "rd", "all": "all",
              "summary_item": "sum", "content_item": "cnt", "thought_part": "thought"}
        can = " / ".join(" ".join(f"{sh.get(k, k)}:{v}" for k, v in (r["canary"].get(p) or {}).items()) or "-"
                         for p in POSITIONS)
        errs = "; ".join(f"{k}: {v}" for k, v in list(r["errors"].items())[:3]).replace("|", "/")
        lines.append(f"| {r['route']} | {r['model']} | {r['lane']} {','.join(x for x in r['id_prefixes'] if x)} | "
                     f"{fmt_cells(r['text'].get('in_loop', {}), order)} | {fmt_cells(r['text'].get('cross_turn', {}), order)} | "
                     f"{fmt_cells(r['opaque'].get('in_loop', {}), order)} | {fmt_cells(r['opaque'].get('cross_turn', {}), order)} | "
                     f"{t1s.strip() or '-'} | {can} | {errs[:300] or '-'} | {r['verdict']} |")
    md = "\n".join(lines) + "\n"
    (out / "matrix.md").write_text(md, encoding="utf-8")
    (out / "matrix.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return md


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models")
    m.add_argument("--route", default="nous")
    r = sub.add_parser("run")
    r.add_argument("--route", required=True, help=f"one of {sorted(ROUTES)} or 'custom'")
    r.add_argument("--kind", choices=("chat", "responses", "gemini"), help="custom routes")
    r.add_argument("--base-url")
    r.add_argument("--key-env", help="env var (process env, then ~/.hermes/.env) holding the key")
    r.add_argument("--models", required=True, help="comma list, or 'portal-all' / 'portal-max:<usd per M out>'")
    r.add_argument("--exclude", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--steps", default="text,opaque")
    r.add_argument("--positions", default="in_loop,cross_turn")
    r.add_argument("--variants", default="")
    r.add_argument("--want-direct", default="", help="Portal ids to resample until a direct-lane result")
    r.add_argument("--lane-attempts", type=int, default=4)
    r.add_argument("--max-tokens", type=int, default=400)
    r.add_argument("--turn1-max-tokens", type=int, default=1024)
    r.add_argument("--budget", type=float, default=10.0, help="USD cap across every run in --out")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--probe", choices=sorted(PROBES), default="std",
                   help="std asks about 'earlier reasoning'; alt avoids the word (xAI 403s std cross-turn)")
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "models":
        keep, skip = nous_selection(nous_catalog())
        print(json.dumps({"test": keep, "skip": skip}, indent=1))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "report":
        print(render(summarize(out), out))
        return

    if a.route == "custom":
        kind, base, cred = a.kind or "chat", a.base_url, a.key_env or ""
    else:
        kind, base, cred = ROUTES[a.route]
        base, cred = a.base_url or base, a.key_env or cred
    catalog = nous_catalog() if (a.route == "nous" or a.models.startswith("portal")) else []
    if not catalog:
        try:
            catalog = nous_catalog()  # used only as a price table for vendor ids
        except Exception:
            catalog = []
    models = [x for x in a.models.split(",") if x]
    if a.models.startswith("portal"):
        sel, _ = nous_selection(catalog)
        cap = float(a.models.split(":", 1)[1]) if ":" in a.models else 1e9
        prices = {m["id"]: float((m.get("pricing") or {}).get("completion") or 0) * 1e6 for m in catalog}
        models = [x for x in sel if prices.get(x, 0) <= cap]
    models = [x for x in models if x not in set(a.exclude.split(","))]
    a.steps = a.steps.split(",")
    a.positions = a.positions.split(",")
    a.variants = [v for v in a.variants.split(",") if v]
    a.want_direct = set(a.want_direct.split(",")) if a.want_direct else set()
    seed = a.seed if a.seed is not None else random.randrange(1 << 30)
    canary = random.Random(seed).choice(_CANARIES)
    trace = make_trace(canary, seed)
    budget = Budget(out / "cost.jsonl", a.budget)
    client = Client(a.route, kind, base, cred, budget, price_table(catalog))
    prober = Prober(client, out, canary, trace, a)
    with (out / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "route": a.route, "base": base, "kind": kind, "models": models,
                            "seed": seed, "canary": canary, "trace_chars": len(trace),
                            "want_direct": sorted(a.want_direct)}) + "\n")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(prober.run_model, models))
    print(f"spent so far in {out}: ${budget.spent:.4f}")


if __name__ == "__main__":
    main()
