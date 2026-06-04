#!/usr/bin/env python3
"""Compare OpenCode Go model behavior across Hermes and the native opencode CLI.

This is a local smoke harness for provider/plugin refinement, not a unit test.
It runs three probes per model:
  1) `opencode run` direct
  2) `hermes chat` with a realistic /goal-capable toolset (terminal,file)
  3) direct AIAgent with the provider's resolved api_mode and the same toolset

Use it to quickly spot provider-routing regressions and CLI/runtime mismatches.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUTH_JSON = Path.home() / ".local/share/opencode/auth.json"
PROMPT = "Respond with exactly: SMOKE_OK"
DEFAULT_MODELS = [
    "glm-5.1",
    "qwen3.7-plus",
    "qwen3.7-max",
    "kimi-k2.6",
    "minimax-m2.7",
]


def _stub_module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _ensure_repo_imports() -> None:
    sys.path.insert(0, str(ROOT))
    sys.modules.setdefault("fire", _stub_module("fire", Fire=lambda *a, **k: None))
    sys.modules.setdefault("firecrawl", _stub_module("firecrawl", Firecrawl=object))
    sys.modules.setdefault("fal_client", _stub_module("fal_client"))


def _load_key() -> str:
    data = json.loads(AUTH_JSON.read_text())
    return data["opencode-go"]["key"]


def _run(cmd: list[str], *, timeout: int = 180) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "exit_code": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "output": proc.stdout,
    }


def _summarize(output: str, ok_token: str = "SMOKE_OK") -> str:
    text = (output or "").strip()
    if (
        text.endswith(ok_token)
        or f"'final_response': '{ok_token}'" in text
        or f'"final_response": "{ok_token}"' in text
        or f"\n{ok_token}\n" in text
        or f"\n    {ok_token}\n" in text
    ):
        return "ok"
    lowered = text.lower()
    if "cannot specify both 'thinking' and 'reasoning_effort'" in lowered:
        return "reasoning_conflict"
    if "not supported for format oa-compat" in lowered:
        return "unsupported_oa_compat"
    if "404" in lowered and "not found" in lowered:
        return "not_found"
    if "api call failed after 3 retries" in lowered:
        return "retry_exhausted"
    if "error code:" in lowered or "error:" in lowered:
        return "error"
    return "unknown"


def probe_opencode(model: str) -> dict[str, Any]:
    result = _run(["opencode", "run", PROMPT, "--model", f"opencode-go/{model}"])
    result["status"] = _summarize(result["output"])
    return result


def probe_hermes_chat(model: str) -> dict[str, Any]:
    result = _run(
        [
            "hermes",
            "chat",
            "--provider",
            "opencode-go",
            "--toolsets",
            "terminal,file",
            "-m",
            f"opencode-go/{model}",
            "-q",
            PROMPT,
        ]
    )
    result["status"] = _summarize(result["output"])
    return result


def probe_aiagent(model: str) -> dict[str, Any]:
    _ensure_repo_imports()
    from run_agent import AIAgent
    from hermes_cli.models import opencode_model_api_mode

    started = time.time()
    api_mode = opencode_model_api_mode("opencode-go", model)
    key = _load_key()
    try:
        agent = AIAgent(
            api_key=key,
            base_url="https://opencode.ai/zen/go/v1",
            provider="opencode-go",
            api_mode=api_mode,
            model=model,
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=["terminal", "file"],
        )
        reply = str(agent.run_conversation(PROMPT))
        output = reply
        exit_code = 0
    except Exception as exc:  # pragma: no cover - smoke utility
        output = f"EXC: {type(exc).__name__}: {exc}"
        exit_code = 1
    return {
        "api_mode": api_mode,
        "exit_code": exit_code,
        "seconds": round(time.time() - started, 2),
        "output": output,
        "status": _summarize(output),
    }


def main(argv: list[str]) -> int:
    models = argv or DEFAULT_MODELS
    rows = []
    for model in models:
        row = {
            "model": model,
            "opencode": probe_opencode(model),
            "hermes_chat": probe_hermes_chat(model),
            "aiagent": probe_aiagent(model),
        }
        rows.append(row)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
