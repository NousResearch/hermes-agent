"""Issue ingress control methods stay native-owner-only and disabled by default."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from gateway.session_issue22 import control_handlers, poll_enabled_source


def test_issue_ingress_methods_reject_non_owner_before_config_read(monkeypatch):
    monkeypatch.setenv("GITHUB_BRIDGE_CONFIG", "/missing/bridge-config.json")
    handlers = control_handlers(SimpleNamespace(native_owner=False))
    assert set(handlers) == {"issue22-admit", "issue22-poll"}
    result = asyncio.run(handlers["issue22-admit"](None, {"issue_number": 22, "comment_id": 1}))
    assert result == {"verdict": "REJECT", "reason": "OWNER_REQUIRED"}


def test_issue_ingress_methods_are_disabled_without_bridge_config(monkeypatch):
    monkeypatch.delenv("GITHUB_BRIDGE_CONFIG", raising=False)
    handlers = control_handlers(SimpleNamespace(native_owner=True))
    result = asyncio.run(handlers["issue22-poll"](None, {}))
    assert result == {"verdict": "REJECT", "reason": "INGRESS_DISABLED"}
    assert poll_enabled_source() is None


def test_issue22_poll_running_result_matches_canonical_gateway_contract():
    from tui_gateway.contracts.registry import CANONICAL_METHODS

    contract = CANONICAL_METHODS["issue22-poll"].result
    result = contract.model_validate({
        "verdict": "ACK", "issue_number": 22, "kind": "TASK", "state": "running",
        "task_id": "issue22-abc123", "comment_id": 9182,
    })
    assert result.state == "running"
    assert result.comment_id == 9182


def test_housekeeping_poll_invokes_only_the_enabled_source(monkeypatch):
    from gateway import session_issue22

    calls = []
    monkeypatch.setattr(session_issue22, "_source_handlers", lambda: (None, lambda params: calls.append(params) or {"verdict": "NOOP"}))
    session_issue22.poll_enabled_source()
    assert calls == [{}]
