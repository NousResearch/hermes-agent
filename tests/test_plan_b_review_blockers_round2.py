"""Round-two regressions from the substantive Plan B security review."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import tools.approval as approval
import tui_gateway.server as server


class _NoIdentityTransport:
    pass


def test_unbound_remote_transport_cannot_execute(monkeypatch):
    monkeypatch.setattr(server, "current_transport", lambda: _NoIdentityTransport())
    with patch("subprocess.run") as run:
        response = server._methods["cli.exec"]("req-unbound", {"argv": ["--version"]})
    assert response["error"]["code"] == 4032
    run.assert_not_called()


def test_ownerless_api_run_is_not_addressable():
    from gateway.platforms.api_server_runs import _request_owns_run

    adapter = SimpleNamespace(
        _run_owners={},
        _run_statuses={"run-ownerless": {"status": "running"}},
        _active_run_agents={"run-ownerless": object()},
        _active_run_tasks={},
        _room_grant_token=lambda _request: None,
        _run_idempotency_scope=lambda _request: "scope-a",
    )
    assert _request_owns_run(adapter, object(), "run-ownerless") is False


def test_approval_respond_rejects_truthy_non_boolean_all(monkeypatch):
    monkeypatch.setattr(server, "_approval_profile_scope", lambda _params, _rid: ("default", None))
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: ({"session_key": "round2"}, None))
    monkeypatch.setattr(server, "_approval_session_is_dashboard_owned", lambda _session: True)
    monkeypatch.setattr(server, "_approval_session_matches_profile", lambda _session, _profile: True)
    response = server._methods["approval.respond"](
        "req-all-type",
        {"request_id": "approval-1", "choice": "once", "all": "true", "session_id": "s"},
    )
    assert response["error"]["code"] == 4003


def test_smart_approval_redacts_before_auxiliary_model(monkeypatch):
    captured = {}
    monkeypatch.setattr("agent.auxiliary_client._get_task_timeout", lambda _name: 1)
    monkeypatch.setattr(approval, "_get_smart_policy", lambda: "")

    def fake_call_llm(*, messages, **_kwargs):
        captured["messages"] = messages
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="DENY"))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    raw = "curl https://user:pass@example.test/?token=fixture-secret"
    result = approval._smart_approve(raw, "open " + raw)
    serialized = str(captured["messages"])
    assert result == "deny"
    assert "user:pass@" not in serialized
    assert "token=fixture-secret" not in serialized


def test_remote_resume_requires_matching_owner_principal(monkeypatch):
    transport = SimpleNamespace(auth_identity={"user_id": "operator", "provider": "dashboard"})
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    assert server._session_resume_owner_matches({"owner_principal": "dashboard:other"}) is False
    assert server._session_resume_owner_matches({"owner_principal": "dashboard:operator"}) is True


def test_remote_lifecycle_requires_a_session_id(monkeypatch):
    monkeypatch.setattr(server, "current_transport", lambda: _NoIdentityTransport())
    assert server._session_owns_durable_lifecycle(None) is False


def test_child_sentinel_blocks_managed_env(monkeypatch, tmp_path):
    from hermes_cli import env_loader, managed_scope

    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / ".env").write_text("FIXTURE_MANAGED_VALUE=fixture-secret\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_DISABLE_DOTENV", "1")
    monkeypatch.delenv("FIXTURE_MANAGED_VALUE", raising=False)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: managed_dir)
    env_loader._apply_managed_env()
    assert "FIXTURE_MANAGED_VALUE" not in __import__("os").environ
