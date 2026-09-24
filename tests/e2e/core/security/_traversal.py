"""Private harness for the traversal sub-lane (store traversal + workspace escape).

``run_tool_calls`` drives the REAL agent tool path: one ``hermes chat -q`` process whose model (the
loopback fake) issues the given tool calls in order, then answers ``done``. Each tool result is read
back from the wire: the final request Hermes sent to the model carries every ``role: tool`` message,
keyed by the fake's sequential ``call_fake_<n>`` ids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tests.e2e.core.security import _helpers as H
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall


def digest(path: Path) -> str | None:
    """sha256 of a file's bytes (None when absent); symlinks are followed, as a reader would."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def tool_text(content: Any) -> str:
    """Flatten a chat-completions tool message ``content`` (str or content parts) to text."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def run_tool_calls(home: Path, calls: list[tuple[str, dict[str, Any]]], *, cwd: Path,
                   config: str = "", env_lines: dict[str, str] | None = None,
                   extra_env: dict[str, str] | None = None, timeout: float = 150.0) -> list[str]:
    """Run one ``hermes chat -q`` turn whose model issues ``calls`` sequentially; return each tool
    result text, in call order. Harness failures (non-zero exit, a lost tool result) are plain
    ``AssertionError`` so a KNOWN strict xfail never masks them."""
    key = H.canary("sk-traversal")
    script: list[Any] = [ToolCall(name, args) for name, args in calls] + [Text("done")]
    with FakeLLMServer(script, api_key=key) as srv:
        H.write_home(home / ".hermes", srv.base_url, api_key=key, config=config, env=env_lines)
        proc = H.run_hermes(["chat", "-q", "run the scripted tools", "-Q"], home, cwd=cwd,
                            extra_env=extra_env, timeout=timeout)
        reqs = srv.main_requests()
    assert proc.returncode == 0, f"hermes chat -q rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    assert len(reqs) == len(calls) + 1, f"expected {len(calls) + 1} model turns, saw {len(reqs)}\n{proc.stderr[-3000:]}"
    results = {m.get("tool_call_id"): tool_text(m.get("content"))
               for m in reqs[-1]["messages"] if m.get("role") == "tool"}
    ids = [f"call_fake_{i + 1}" for i in range(len(calls))]
    missing = [i for i in ids if i not in results]
    assert not missing, f"tool results missing for {missing}: got {sorted(results)}"
    return [results[i] for i in ids]


def result_json(text: str) -> dict[str, Any]:
    """Tool result as a dict (``{}`` when the result is not a JSON object)."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
