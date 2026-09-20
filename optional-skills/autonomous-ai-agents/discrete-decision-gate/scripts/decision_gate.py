#!/usr/bin/env python3
"""Discrete decision gate: typed single-choice verdicts for agent routing.

Asks a decision model exactly one typed question about program state and returns
one choice plus a confidence. It NEVER invents a verdict: if the backend is
missing, unreachable, or answers off-schema, the result is status="unavailable"
and the caller's own deterministic policy decides what happens next.

Backends
  typesafe   POST https://api.typesafe.ai/v1/systemone   (Jev 1P; TYPESAFE_API_KEY)
  jev        POST https://openrouter.ai/api/alpha/decisions (TypeSafe Jev via OpenRouter;
             OPENROUTER_API_KEY) - the local default gate backend
  openrouter POST https://openrouter.ai/api/v1/chat/completions (chat model; OPENROUTER_API_KEY)
  ollama     POST http://localhost:11434/api/chat        (offline; no key)

Usage
  decision_gate.py triage        --intent "rename a helper in one file"
  decision_gate.py blast-radius  --summary "touched 2 files: utils.py"
  decision_gate.py harness       --exit-code 1 --tail "<pytest output tail>"
  decision_gate.py ask --state "..." --question "..." --choices a,b,c [--describe 'a=x;b=y']
Common flags
  --backend auto|typesafe|jev|openrouter|ollama   (default auto)
  --model <id>      --threshold 0.8      --json      --quiet

Exit codes
  0 verdict returned        3 backend unavailable / off-schema answer
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen3:14b"
# TypeSafe Jev through OpenRouter's Decisions route: the local default against OPENROUTER_API_KEY.
# OpenRouter has no `typesafe/jev-latest` alias - `typesafe/jev-1.13` is the only Jev id there and
# every response names the snapshot it served (e.g. typesafe/jev-1.13-20260917).
JEV_MODEL = os.getenv("JEV_MODEL", "typesafe/jev-1.13")

# ---------------------------------------------------------------- task presets

PRESETS = {
    "triage": {
        "question": "Which engine should execute this Hermes task?",
        "choices": ["bash_direct", "crg_first", "direct_dsh"],
        "describe": {
            "bash_direct": "Single-file mechanical work with no cross-file callers to check: git status, read one file, rename a symbol inside one file, trivial shell check.",
            "crg_first": "Multi-file refactor, bug hunt, unknown blast radius, structural codebase change.",
            "direct_dsh": "Standalone single script or isolated text edit where AST mapping is unneeded.",
        },
        "unavailable_default": "bash_direct",
    },
    "blast-radius": {
        "question": "How risky is this change for auto-apply?",
        "choices": ["low", "high"],
        "describe": {
            "low": "Isolated leaf module, <= 3 files touched, safe to auto-apply without confirmation.",
            "high": "Touches core config, schema/auth models, shared state, or > 4 dependents; needs human signoff.",
        },
        # high-stakes gate: when the gate is unavailable, assume the risky answer
        "unavailable_default": "high",
    },
    "harness": {
        "question": "What is the next loop action for this coding-harness run?",
        "choices": ["complete", "retry", "abort"],
        "describe": {
            "complete": "Tests cleanly passed and the work is done.",
            "retry": "Test failure or syntax error the harness can fix in another pass.",
            "abort": "Catastrophic environment failure, missing credentials, or an infinite loop.",
        },
        "unavailable_default": "abort",
    },
}


# --------------------------------------------------------------------- backends

def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _parse_choice(text: str, choices: list[str]) -> str | None:
    """Accept the answer only if exactly one option is stated. No fuzzy guessing."""
    if text is None:
        return None
    low = text.strip().lower()
    exact = [c for c in choices if low == c.lower()]
    if len(exact) == 1:
        return exact[0]
    present = [c for c in choices if re.search(rf"\b{re.escape(c)}\b", low)]
    return present[0] if len(present) == 1 else None


def ask_typesafe(state: str, question: str, choices: list[str],
                 describe: dict, model: str, timeout: float) -> dict:
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        return {"status": "unavailable", "reason": "TYPESAFE_API_KEY not set"}
    criteria = {c: describe.get(c, c) for c in choices}
    payload = {
        "state": state,
        "model": model or "jev-latest",
        "questions": {"answer": {"type": "choice", "instructions": question,
                                 "criteria": criteria}},
    }
    data = _post(TYPESAFE_URL, payload,
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                 timeout)
    ans = data.get("answers", {}).get("answer", {})
    choice = ans.get("choice")
    if choice not in choices:
        matched = _parse_choice(str(choice), choices)
        if matched is None:
            return {"status": "unavailable",
                    "reason": f"off-schema choice from typesafe: {choice!r}"}
        choice = matched
    return {"status": "ok", "choice": choice,
            "confidence": ans.get("confidence"),
            "probabilities": ans.get("probabilities")}


def ask_jev(state: str, question: str, choices: list[str],
            describe: dict, model: str, timeout: float) -> dict:
    """TypeSafe Jev through OpenRouter's Decisions route (flat body, typed choice answer)."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return {"status": "unavailable", "reason": "OPENROUTER_API_KEY not set"}
    payload = {
        "model": model or JEV_MODEL,
        "state": state,
        "questions": {"answer": {"type": "choice", "instructions": question,
                                 "criteria": {c: describe.get(c, c) for c in choices}}},
    }
    data = _post(OPENROUTER_DECISIONS_URL, payload,
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                  "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                  "X-Title": "Hermes decision gate"},
                 timeout)
    if not isinstance(data.get("answers"), dict) or not data["answers"]:
        return {"status": "unavailable",
                "reason": f"no answers map in response: {json.dumps(data)[:200]}"}
    ans = data["answers"].get("answer") or {}
    choice = ans.get("choice")
    if choice not in choices:
        matched = _parse_choice(str(choice), choices)
        if matched is None:
            return {"status": "unavailable",
                    "reason": f"off-schema choice from jev: {choice!r}"}
        choice = matched
    return {"status": "ok", "choice": choice, "confidence": ans.get("confidence"),
            "probabilities": ans.get("probabilities"), "served_model": data.get("model"),
            "cost": (data.get("usage") or {}).get("cost")}


def ask_openrouter(state: str, question: str, choices: list[str],
                   describe: dict, model: str, timeout: float) -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return {"status": "unavailable", "reason": "OPENROUTER_API_KEY not set"}
    opts = "\n".join(f"- {c}: {describe.get(c, c)}" for c in choices)
    payload = {
        "model": model or DEFAULT_OPENROUTER_MODEL,
        "messages": [
            {"role": "system",
             "content": "You are a discrete classification gate. "
                        "Reply with exactly one of the allowed labels and nothing else."},
            {"role": "user",
             "content": f"State:\n{state}\n\nQuestion: {question}\n\n"
                        f"Allowed labels:\n{opts}\n\nLabel:"},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    data = _post(OPENROUTER_URL, payload,
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                  "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                  "X-Title": "Hermes decision gate"},
                 timeout)
    text = data["choices"][0]["message"]["content"]
    choice = _parse_choice(text, choices)
    if choice is None:
        return {"status": "unavailable",
                "reason": f"off-schema answer from openrouter: {text!r}"}
    return {"status": "ok", "choice": choice, "raw": text.strip()}


def ask_ollama(state: str, question: str, choices: list[str],
               describe: dict, model: str, timeout: float) -> dict:
    opts = "\n".join(f"- {c}: {describe.get(c, c)}" for c in choices)
    payload = {
        "model": model or DEFAULT_OLLAMA_MODEL,
        "stream": False,
        "think": False,  # qwen3-style reasoning models emit nothing useful otherwise
        "options": {"temperature": 0},
        "messages": [
            {"role": "system",
             "content": "You are a discrete classification gate. "
                        "Reply with exactly one of the allowed labels and nothing else."},
            {"role": "user",
             "content": f"State:\n{state}\n\nQuestion: {question}\n\n"
                        f"Allowed labels:\n{opts}\n\nLabel:"},
        ],
    }
    data = _post(OLLAMA_URL, payload, {"Content-Type": "application/json"}, timeout)
    text = data.get("message", {}).get("content", "")
    choice = _parse_choice(text, choices)
    if choice is None:
        return {"status": "unavailable",
                "reason": f"off-schema answer from ollama: {text[:200]!r}"}
    return {"status": "ok", "choice": choice, "raw": text.strip()}


BACKENDS = {"typesafe": ask_typesafe, "jev": ask_jev, "openrouter": ask_openrouter,
            "ollama": ask_ollama}
REQUIRED_ENV = {"typesafe": "TYPESAFE_API_KEY", "jev": "OPENROUTER_API_KEY",
                "openrouter": "OPENROUTER_API_KEY"}
# Resolution order for --backend auto: Jev via OpenRouter's Decisions route (the local default),
# then a chat model, then offline Ollama. TypeSafe 1P is opt-in only (`--backend typesafe`), so a
# stray TYPESAFE_API_KEY cannot take over the gate.
AUTO_ORDER = ("jev", "openrouter", "ollama")


def decide(state: str, question: str, choices: list[str], describe: dict | None = None,
           backend: str = "auto", model: str = "", timeout: float = 20.0,
           unavailable_default: str | None = None) -> dict:
    """Ask one typed question. Returns status ok|unavailable; never fabricates a choice."""
    describe = describe or {}
    order = ([backend] if backend != "auto"
             else [b for b in AUTO_ORDER
                   if b not in REQUIRED_ENV or os.getenv(REQUIRED_ENV[b])])
    if not order:
        return {"status": "unavailable", "backend": None,
                "reason": "no backend credentials found (TYPESAFE_API_KEY / OPENROUTER_API_KEY)"}

    # fail fast on a model id the backend does not actually serve
    last: dict = {}
    for name in order:
        started = time.monotonic()
        try:
            out = BACKENDS[name](state, question, choices, describe, model, timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            out = {"status": "unavailable",
                   "reason": f"{name} HTTP {e.code}: {body}"}
        except Exception as e:  # network, parse, timeout
            out = {"status": "unavailable", "reason": f"{name} {type(e).__name__}: {e}"}
        out["backend"] = name
        out["latency_ms"] = round((time.monotonic() - started) * 1000)
        if out.get("status") == "ok":
            return out
        last = out

    result = dict(last)
    if unavailable_default is not None:
        result["fallback"] = unavailable_default
        result["policy"] = "gate unavailable -> deterministic default, not a model verdict"
    return result


# ---------------------------------------------------------------- diagnostics

def _env_file_key(name: str) -> str | None:
    """Read a key from ~/.hermes/.env without mutating the process env."""
    if os.getenv(name):
        return os.environ[name]
    for path in (os.path.expanduser("~/.hermes/.env"), ".env"):
        try:
            with open(path) as fh:
                for line in fh:
                    if line.strip().startswith(name + "="):
                        return line.split("=", 1)[1].strip().strip("\"'")
        except OSError:
            continue
    return None


def doctor(backend: str = "auto", model: str = "", timeout: float = 20.0) -> dict:
    """Probe every backend: key present, endpoint reachable, model served, live verdict."""
    report: dict = {"backends": {}, "ok": []}
    names = ([*AUTO_ORDER, *(["typesafe"] if os.getenv("TYPESAFE_API_KEY") else [])]
             if backend == "auto" else [backend])
    for name in names:
        env_name = REQUIRED_ENV.get(name)
        key = _env_file_key(env_name) if env_name else "n/a"
        entry: dict = {"key": "present" if (key or env_name is None) else "MISSING"}
        if env_name and not os.getenv(env_name) and key:
            entry["key_source"] = "~/.hermes/.env (not in shell env - load it before calling)"
            os.environ[env_name] = key  # probe only; process-local
        if entry["key"] == "MISSING":
            entry["status"] = "unconfigured"
            report["backends"][name] = entry
            continue
        out = decide("Probe: the build script finished and all tests passed.",
                     "Did the tests pass?", ["pass", "fail"],
                     {"pass": "Tests passed.", "fail": "Tests failed."},
                     backend=name, model=model, timeout=timeout)
        entry.update({k: out.get(k) for k in ("status", "choice", "latency_ms", "reason")
                      if k in out})
        report["backends"][name] = entry
        if out.get("status") == "ok":
            report["ok"].append(name)
    report["ready"] = bool(report["ok"])
    return report


# ---------------------------------------------------------------------- CLI

def _emit(result: dict, as_json: bool, quiet: bool) -> int:
    if as_json:
        print(json.dumps(result, indent=2))
    elif not quiet:
        if result.get("status") == "ok":
            conf = result.get("confidence")
            conf = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
            print(f"{result['choice']}{conf} via {result['backend']} "
                  f"in {result['latency_ms']}ms")
            if result.get("probabilities"):
                print(json.dumps(result["probabilities"], sort_keys=True))
        else:
            print(f"UNAVAILABLE via {result.get('backend')}: {result.get('reason')}")
            if result.get("fallback"):
                print(f"policy: use deterministic default '{result['fallback']}'")
    return 0 if result.get("status") == "ok" else 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=[*PRESETS, "ask", "doctor"])
    p.add_argument("--intent", default="")
    p.add_argument("--summary", default="")
    p.add_argument("--exit-code", type=int, default=None)
    p.add_argument("--tail", default="")
    p.add_argument("--state", default="")
    p.add_argument("--question", default="")
    p.add_argument("--choices", default="")
    p.add_argument("--describe", default="", help="e.g. 'a=first;b=second'")
    p.add_argument("--backend", default="auto",
                   choices=["auto", *BACKENDS])
    p.add_argument("--model", default="")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--no-fallback-policy", action="store_true",
                   help="omit the deterministic default from the result")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    if a.task == "doctor":
        rep = doctor(backend=a.backend, model=a.model, timeout=a.timeout)
        print(json.dumps(rep, indent=2))
        for name, e in rep["backends"].items():
            line = f"{name:11} {e.get('status','?'):12} key={e['key']}"
            if e.get("latency_ms"):
                line += f" {e['latency_ms']}ms choice={e.get('choice')}"
            if e.get("reason"):
                line += f" reason={e['reason'][:120]}"
            print(line)
        return 0 if rep["ready"] else 3

    if a.task == "ask":
        if not (a.state and a.question and a.choices):
            p.error("ask requires --state, --question and --choices")
        state, question = a.state, a.question
        choices = [c.strip() for c in a.choices.split(",") if c.strip()]
        describe = dict(kv.split("=", 1) for kv in a.describe.split(";") if "=" in kv)
        fallback = choices[0] if choices else None
    else:
        preset = PRESETS[a.task]
        choices, question = preset["choices"], preset["question"]
        describe = preset["describe"]
        fallback = preset["unavailable_default"]
        if a.task == "triage":
            state = f"Task intent: {a.intent}"
        elif a.task == "blast-radius":
            state = f"Code-review-graph blast radius summary:\n{a.summary}"
        else:
            state = (f"Exit code: {a.exit_code}\n"
                     f"Terminal output tail:\n{a.tail[-1500:]}")

    if not state.strip():
        p.error("empty state - nothing to classify")

    result = decide(state, question, choices, describe, backend=a.backend,
                    model=a.model, timeout=a.timeout,
                    unavailable_default=None if a.no_fallback_policy else fallback)
    return _emit(result, a.json, a.quiet)


if __name__ == "__main__":
    sys.exit(main())
