"""Shared machinery for the secrets-gated LIVE provider canary (``-m live``).

The live lane drives the REAL ``AIAgent`` + real provider adapters against real
vendor endpoints on cheap models. Only the tool is synthetic (a deterministic
``live_lookup`` registered in-process) so every assertion is about the wire:

* C9  wire-format drift: every inference request is accepted (no 4xx), tool calls
  come back parseable (valid JSON args), parallel tool calls and their results
  round-trip, history (incl. reasoning replay) is re-sent and accepted, and no
  reasoning/tool markup leaks into user-visible text.
* C17 prompt cache: on cache-capable routes the follow-up turn reads the cache and
  the outgoing request carries well-formed breakpoints.
* C11 real auth/routing: the credential resolved from the environment is the one
  sent, and it is only ever sent to the provider's own host; ``/models`` parses.

Credential values are snapshotted at import (the root conftest blanks every
credential-shaped env var per test) and are never printed, logged or written.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

# Snapshot BEFORE tests/conftest.py::_hermetic_environment strips credentials.
LIVE_KEY_VARS = (
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "NOUS_API_KEY",
    "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY",
)
_LIVE_KEYS: dict[str, str] = {
    name: os.environ[name].strip() for name in LIVE_KEY_VARS if os.environ.get(name, "").strip()
}

LOOKUP_VALUES = {"alpha": "ZEBRA-7731", "beta": "OTTER-2204", "gamma": "HERON-5918"}
TOOL_NAME = "live_lookup"
TOOLSET = "live_probe"

TURN1 = (
    f'Call the {TOOL_NAME} tool with key "alpha", then reply with only the value it returned.'
)
TURN2 = (
    f'Call {TOOL_NAME} for key "beta" AND for key "gamma". Issue BOTH calls in parallel in a '
    "single response (two tool calls in one message), then reply with only the two values."
)
TURN3 = (
    "Do not call any tools. From this conversation's history, list the three values you "
    "looked up, in the order alpha, beta, gamma, separated by spaces."
)

# Text that must never reach the user: reasoning tags, DeepSeek DSML, chat-template
# control tokens, raw tool-call JSON / XML.
LEAK_MARKERS = (
    "<think>", "</think>", "<thinking>", "</thinking>", "DSML", "<|", "|>",
    "<tool_call", "</tool_call>", "<function", "</function", '"tool_calls"',
    '"arguments"', '{"key"', "<invoke", "antml",
)

# Per-test spend guard (tokens, all buckets). A regression that loops or re-sends
# a huge prompt fails here instead of burning money.
MAX_TOKENS_PER_TEST = 90_000
MAX_USD_PER_TEST = 0.12


@dataclass(frozen=True)
class LiveCase:
    id: str
    provider: str  # hermes provider id passed to resolve_runtime_provider
    key_env: str
    model_prefs: tuple[str, ...]
    hosts: tuple[str, ...]  # the only hosts allowed to see the credential
    price_in: float  # $/M input tokens, list price (ceiling estimate)
    price_out: float  # $/M output tokens
    cache_expected: bool = False  # assert cache-read > 0 on the follow-up turn
    pad_tokens: int = 0  # grow the prefix past the vendor's minimum cacheable length
    native_anthropic: bool = False


# Cheapest tool-capable model per family (verified live Sep 2026). The first
# preference present in the provider's live /models listing is used, so a
# retired slug fails the listing test loudly instead of 404ing mid-conversation.
# Override with HERMES_LIVE_MODEL_<ID> (upper-cased, '-' -> '_').
LIVE_CASES: tuple[LiveCase, ...] = (
    LiveCase("openrouter-openai", "openrouter", "OPENROUTER_API_KEY",
             ("openai/gpt-4.1-nano", "openai/gpt-5-nano"), ("openrouter.ai",), 0.10, 0.40),
    LiveCase("openrouter-anthropic", "openrouter", "OPENROUTER_API_KEY",
             ("anthropic/claude-haiku-4.5", "anthropic/claude-3-haiku"), ("openrouter.ai",), 1.0, 5.0,
             cache_expected=True, pad_tokens=4600),
    LiveCase("openrouter-google", "openrouter", "OPENROUTER_API_KEY",
             ("google/gemini-2.5-flash-lite", "google/gemini-3.1-flash-lite"), ("openrouter.ai",), 0.10, 0.40),
    LiveCase("openrouter-xai", "openrouter", "OPENROUTER_API_KEY",
             ("x-ai/grok-4.3", "x-ai/grok-build-0.1", "x-ai/grok-4.20"), ("openrouter.ai",), 1.25, 2.50),
    LiveCase("openrouter-deepseek", "openrouter", "OPENROUTER_API_KEY",
             ("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-chat-v3.1"),
             ("openrouter.ai",), 0.10, 0.50),
    LiveCase("openrouter-qwen", "openrouter", "OPENROUTER_API_KEY",
             ("qwen/qwen3.7-flash", "qwen/qwen3.5-flash-02-23", "qwen/qwen3-235b-a22b-2507"),
             ("openrouter.ai",), 0.10, 0.40),
    LiveCase("nous-portal", "nous", "NOUS_API_KEY",
             ("deepseek/deepseek-v4-flash", "qwen/qwen3.7-flash", "google/gemini-2.5-flash-lite",
              "openai/gpt-4.1-nano"),
             ("inference-api.nousresearch.com", "portal.nousresearch.com"), 0.30, 1.20),
    LiveCase("anthropic-direct", "anthropic", "ANTHROPIC_API_KEY",
             ("claude-haiku-4-5", "claude-haiku-4-5-20251001"), ("api.anthropic.com",), 1.0, 5.0,
             cache_expected=True, pad_tokens=4600, native_anthropic=True),
    LiveCase("openai-direct", "openai", "OPENAI_API_KEY",
             ("gpt-5-nano", "gpt-5.4-nano", "gpt-4.1-nano"), ("api.openai.com",), 0.05, 0.40),
    LiveCase("gemini-direct", "gemini", "GEMINI_API_KEY",
             ("gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-lite-latest"),
             ("generativelanguage.googleapis.com",), 0.25, 1.50),
    LiveCase("xai-direct", "xai", "XAI_API_KEY",
             ("grok-4.3", "grok-4.20-0309-reasoning", "grok-build-0.1"), ("api.x.ai",), 1.25, 2.50),
)

LISTING_PROVIDERS: dict[str, str] = {c.provider: c.key_env for c in LIVE_CASES}


def live_key(env_name: str) -> str:
    """The snapshotted credential, or skip the test cleanly when it is absent."""
    value = _LIVE_KEYS.get(env_name, "")
    if not value:
        pytest.skip(f"{env_name} not set; live canary skipped")
    return value


def model_override(case: LiveCase) -> str | None:
    return os.environ.get("HERMES_LIVE_MODEL_" + case.id.upper().replace("-", "_")) or None


# HTTP wire recorder -----------------------------------------------------------

_INFERENCE_SUFFIXES = ("/chat/completions", "/responses", "/messages", ":streamGenerateContent",
                       ":generateContent")


@dataclass
class WireRecord:
    method: str
    host: str
    path: str
    status: int
    carries_key: bool
    body: Any = None
    error: str = ""  # redacted, truncated response body for status >= 400

    @property
    def is_inference(self) -> bool:
        return self.method == "POST" and self.path.endswith(_INFERENCE_SUFFIXES)


@dataclass
class WireLog:
    """Every HTTP exchange made through httpx (all SDKs Hermes uses sit on it)."""

    secret: str
    records: list[WireRecord] = field(default_factory=list)

    def inference(self, start: int = 0) -> list[WireRecord]:
        return [r for r in self.records[start:] if r.is_inference]

    def main_turn(self, start: int = 0) -> list[WireRecord]:
        """Inference requests of the agent loop itself (they carry the test tool),
        excluding auxiliary calls such as title generation."""
        return [r for r in self.inference(start)
                if isinstance(r.body, dict) and TOOL_NAME in json.dumps(r.body.get("tools") or "")]

    def describe(self, recs: list[WireRecord]) -> str:
        return ", ".join(f"{r.method} {r.host}{r.path} -> {r.status}" + (f" {r.error}" if r.error else "")
                         for r in recs)

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.inference():
            kind = "main" if r in self.main_turn() else "aux"
            counts[f"{kind}:{r.status}"] = counts.get(f"{kind}:{r.status}", 0) + 1
        return counts


def install_wire_recorder(monkeypatch: pytest.MonkeyPatch, secret: str) -> WireLog:
    import httpx

    log = WireLog(secret=secret)

    def _carries(request: "httpx.Request") -> bool:
        if secret in str(request.url):
            return True
        return any(secret in v for v in request.headers.values())

    def _body(request: "httpx.Request") -> Any:
        if request.method != "POST":
            return None
        try:
            return json.loads(request.content or b"null")
        except Exception:
            return None

    def _record(request: "httpx.Request", status: int, body: Any, error: str = "") -> None:
        log.records.append(WireRecord(request.method, request.url.host, request.url.path, status,
                                      _carries(request), body, error.replace(secret, "<redacted>")[:400]))

    def _error_text(resp: "httpx.Response") -> str:
        # Error bodies are small and read eagerly by every SDK anyway; cache them on
        # the response so the caller still sees the same content.
        if resp.status_code < 400:
            return ""
        try:
            return resp.read().decode("utf-8", "replace")
        except Exception as exc:
            return f"<unreadable: {type(exc).__name__}>"

    orig_sync = httpx.HTTPTransport.handle_request
    orig_async = httpx.AsyncHTTPTransport.handle_async_request

    def handle_request(self, request):  # noqa: ANN001
        body = _body(request)
        try:
            resp = orig_sync(self, request)
        except Exception:
            _record(request, -1, body)
            raise
        _record(request, resp.status_code, body, _error_text(resp))
        return resp

    async def handle_async_request(self, request):  # noqa: ANN001
        body = _body(request)
        try:
            resp = await orig_async(self, request)
        except Exception:
            _record(request, -1, body)
            raise
        err = ""
        if resp.status_code >= 400:
            try:
                err = (await resp.aread()).decode("utf-8", "replace")
            except Exception as exc:
                err = f"<unreadable: {type(exc).__name__}>"
        _record(request, resp.status_code, body, err)
        return resp

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request)
    return log


# Deterministic test tool --------------------------------------------------------

def register_lookup_tool() -> Callable[[], None]:
    from tools.registry import registry

    def handler(args: dict, **_kw: Any) -> str:
        key = str(args.get("key", ""))
        return json.dumps({"key": key, "value": LOOKUP_VALUES.get(key, "UNKNOWN")})

    registry.register(
        name=TOOL_NAME, toolset=TOOLSET,
        schema={
            "name": TOOL_NAME,
            "description": "Look up the secret value stored under a key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "alpha, beta or gamma"}},
                "required": ["key"],
            },
        },
        handler=handler,
    )
    return lambda: registry.deregister(TOOL_NAME)


# HERMES_HOME wiring -------------------------------------------------------------

def write_live_home(hermes_home: Path, provider: str, model: str) -> None:
    """Minimal real config: the provider under test, eager tools (no tool_search
    bridge), bounded retries. No credential is ever written to disk."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        f"  provider: {provider}\n"
        f"  default: {model}\n"
        "agent:\n"
        "  api_max_retries: 2\n"
        "tools:\n"
        "  tool_search:\n"
        "    enabled: \"off\"\n",
        encoding="utf-8",
    )


def pad_text(tokens: int) -> str:
    """~``tokens`` tokens of inert reference material (≈8 tokens per row)."""
    if tokens <= 0:
        return ""
    rows = [f"R{i:04d} amber quartz lantern {i * 7 % 1000:03d}" for i in range(tokens // 8)]
    return "\n\nReference table (ignore unless asked):\n" + "\n".join(rows)


# Assertions helpers ----------------------------------------------------------------

def assistant_tool_calls(messages: list[dict]) -> list[list[dict]]:
    """Tool-call groups (one list per assistant message) in order."""
    return [list(m["tool_calls"]) for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")]


def parse_tool_args(call: dict) -> dict:
    fn = call.get("function") or {}
    raw = fn.get("arguments")
    args = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(args, dict), f"tool args are not a JSON object: {raw!r}"
    return args


def leaked_markers(text: str) -> list[str]:
    return [m for m in LEAK_MARKERS if m in (text or "")]


def walk(obj: Any, path: str = "") -> Any:
    """Yield (path, key, value) for every dict entry in a nested JSON value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def conversation_part(body: Any) -> Any:
    """The history part of a request body across wire formats."""
    if not isinstance(body, dict):
        return None
    for key in ("messages", "input", "contents"):
        if key in body:
            return body[key]
    return None


REASONING_KEYS = {"reasoning_content", "reasoning", "reasoning_details", "thinking",
                  "thought_signature", "thoughtSignature", "encrypted_content", "signature"}


def reasoning_replay_keys(body: Any) -> set[str]:
    return {k for _p, k, _v in walk(conversation_part(body)) if k in REASONING_KEYS}


def cache_markers(body: Any) -> list[str]:
    return [p for p, k, _v in walk(body) if k == "cache_control"]


def usage_line(case: LiveCase, model: str, agent: Any, wire: WireLog) -> dict:
    inp = int(agent.session_input_tokens or 0)
    out = int(agent.session_output_tokens or 0)
    cr = int(agent.session_cache_read_tokens or 0)
    cw = int(agent.session_cache_write_tokens or 0)
    # Upper bound: cache reads at full input price, cache writes at 1.25x (Anthropic's rate).
    ceiling = ((inp + cr + 1.25 * cw) * case.price_in + out * case.price_out) / 1_000_000
    return {
        "case": case.id, "model": model, "api_calls": int(agent.session_api_calls or 0),
        "http_inference_calls": len(wire.inference()),
        "http_statuses": wire.status_counts(),
        "http_errors": sorted({f"{r.status} {r.error[:160]}" for r in wire.inference() if r.status >= 400})[:5],
        "input": inp, "output": out, "cache_read": cr, "cache_write": cw,
        "reasoning": int(getattr(agent, "session_reasoning_tokens", 0) or 0),
        "hermes_est_usd": round(float(agent.session_estimated_cost_usd or 0.0), 6),
        "hermes_cost_status": str(getattr(agent, "session_cost_status", "")),
        "list_price_ceiling_usd": round(ceiling, 6),
    }


def emit_usage(line: dict) -> None:
    print("LIVE-USAGE " + json.dumps(line, sort_keys=True), flush=True)
    target = os.environ.get("HERMES_LIVE_USAGE_FILE")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
