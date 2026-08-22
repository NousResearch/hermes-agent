from __future__ import annotations

import time
import sys
from types import SimpleNamespace

from agent.startup_coordinator import coordinate_session_start
from agent.shell_hooks import ShellHookSpec, _make_callback


def test_duplicate_and_oversize_context_is_bounded_with_immutable_receipt(tmp_path):
    agent = SimpleNamespace(
        session_id="session-1", model="MiniMax-M3", provider="litellm-local", platform="cli",
    )
    started = time.monotonic()
    outcome = coordinate_session_start(
        agent,
        [
            {"context": "x" * 5000, "_hermes_hook_identity": "primary"},
            {"context": "duplicate", "_hermes_hook_identity": "secondary"},
        ],
        elapsed_ms=1.0,
        config={"memory": {"enabled": False}, "mcp_servers": {}},
        home=tmp_path,
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    assert len(outcome.context.encode("utf-8")) <= 4096
    assert outcome.receipt["injector"] == {"identity": "primary", "count": 2, "accepted": 1}
    assert any("duplicate startup injector rejected" in row for row in outcome.receipt["degraded_reasons"])
    assert any("truncated" in row for row in outcome.receipt["degraded_reasons"])
    assert outcome.receipt_path is not None and outcome.receipt_path.is_file()
    assert elapsed_ms < 500


def test_real_shell_hook_coordinator_and_receipt_path_stay_below_hard_budget(tmp_path):
    hook = tmp_path / "startup_hook.py"
    hook.write_text(
        'print("{\\\"context\\\":\\\"STARTUP-PACKET\\\",'
        '\\"receipt_patch\\\":{\\\"adapter\\\":\\\"ready\\\"}}")\n',
        encoding="utf-8",
    )
    spec = ShellHookSpec(
        event="on_session_start", command=f"{sys.executable} {hook}", timeout=1,
    )
    callback = _make_callback(spec)
    agent = SimpleNamespace(
        session_id="session-shell", model="MiniMax-M3",
        provider="litellm-local", platform="cli",
    )
    started = time.monotonic()
    result = callback(session_id=agent.session_id, model=agent.model, platform="cli")
    outcome = coordinate_session_start(
        agent, [result], config={"memory": {"enabled": False}, "mcp_servers": {}},
        home=tmp_path / "home",
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    assert outcome.context == "STARTUP-PACKET"
    assert outcome.receipt["adapter"] == "ready"
    assert outcome.receipt_path is not None and outcome.receipt_path.is_file()
    assert elapsed_ms < 500
