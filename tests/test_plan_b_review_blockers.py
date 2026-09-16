"""Regression tests for Plan B independent-review blockers.

These tests are intentionally narrow: each one proves a fail-closed boundary
without exercising a real provider, subprocess, or authenticated Dashboard.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.approval as approval
import tui_gateway.server as server


class _RemoteTransport:
    def __init__(self, identity):
        self.auth_identity = identity


def _remote_session(*, owner="dashboard:operator", profile="default", home=None):
    return {
        "session_key": "plan-b-session",
        "profile_name": profile,
        "profile_home": home or "/tmp/hermes-test/profiles/default",
        "owner_principal": owner,
        "cwd": ".",
        "history": [],
    }


def test_run_approval_gate_uses_explicit_raw_coalesce_target(monkeypatch):
    """Gateway approval coalescing must not reference an undefined command."""
    captured = {}
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda *_args: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "plan-b-session")
    monkeypatch.setattr(approval, "is_approved", lambda *_args: False)
    monkeypatch.setattr(approval, "_resolve_cli_approval_callback", lambda value=None: value)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_fire_approval_hook", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(approval, "_approval_data_coalesce_key", lambda data: captured.update(data) or "key")
    monkeypatch.setattr(
        approval,
        "_gateway_notify_cbs",
        {"plan-b-session": lambda _data: None},
    )
    monkeypatch.setattr(
        approval,
        "_await_gateway_decision",
        lambda *_args, **_kwargs: {"resolved": True, "choice": "once", "reason": None},
    )

    result = approval._run_approval_gate(
        pattern_key="dangerous",
        description="dangerous command",
        display_target="[REDACTED]",
        coalesce_target="rm -rf /secret",
        cron_deny_message="cron denied",
        single_query_deny_message="single query denied",
        autoapprove_log_prefix="test",
    )

    assert result["approved"] is True
    assert captured["command"] == "rm -rf /secret"


def test_execute_code_gateway_coalescing_uses_raw_script(monkeypatch):
    """The execute_code gate must pass a private key for the raw script."""
    captured = {}
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda *_args: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "plan-b-session")
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_should_skip_container_guards", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(approval, "is_approved", lambda *_args: False)
    monkeypatch.setattr(approval, "_resolve_cli_approval_callback", lambda value=None: value)
    monkeypatch.setattr(approval, "_present_with_selected_transport", lambda **_kwargs: {"selected": False})
    monkeypatch.setattr(approval, "_reset_denials", lambda *_args: None)
    monkeypatch.setattr(approval, "_fire_approval_hook", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {"plan-b-session": lambda _data: None})

    def fake_await(_session_key, _notify_cb, data, **kwargs):
        captured["data"] = dict(data)
        captured["coalesce_key"] = kwargs.get("coalesce_key")
        return {"resolved": True, "choice": "once", "reason": None}

    monkeypatch.setattr(approval, "_await_gateway_decision", fake_await)
    raw_code = "api_key = 'secret-a'\nprint(api_key)"
    result = approval.check_execute_code_guard(raw_code, "local")

    expected_command = "execute_code <<'PY'" + chr(10) + raw_code + chr(10) + "PY"
    expected_key = approval._approval_data_coalesce_key(
        {
            "command": expected_command,
            "description": result["description"],
            "pattern_key": "execute_code",
            "pattern_keys": ["execute_code"],
            "allow_permanent": True,
            "allow_session": True,
            "smart_denied": False,
        }
    )
    assert result["approved"] is True
    assert captured["coalesce_key"] == expected_key
    assert captured["data"]["command"] != expected_command


def test_notify_failure_releases_coalesced_followers(monkeypatch):
    """A failed notification must wake a follower instead of leaving it hung."""
    approval._gateway_queues.clear()
    approval._gateway_notify_cbs.clear()
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 5)
    notify_started = threading.Event()
    release_notify = threading.Event()
    results = {}

    def leader_notify(_data):
        notify_started.set()
        release_notify.wait(timeout=1)
        raise RuntimeError("transport down")

    data = {"command": "same", "description": "same", "pattern_key": "p"}

    def leader():
        results["leader"] = approval._await_gateway_decision(
            "plan-b-session", leader_notify, data, timeout_seconds=2
        )

    def follower():
        results["follower"] = approval._await_gateway_decision(
            "plan-b-session", lambda _data: None, data, timeout_seconds=2
        )

    leader_thread = threading.Thread(target=leader)
    follower_thread = threading.Thread(target=follower)
    leader_thread.start()
    assert notify_started.wait(timeout=5), "leader did not reach notification"
    follower_thread.start()
    for _ in range(100):
        if approval._gateway_queues.get("plan-b-session"):
            break
        time.sleep(0.005)
    release_notify.set()
    leader_thread.join(timeout=1)
    follower_thread.join(timeout=1)
    assert not leader_thread.is_alive()
    assert not follower_thread.is_alive(), "coalesced follower remained blocked after notify failure"
    assert results["leader"].get("notify_failed") is True
    assert results["follower"].get("resolved") is True
    assert results["follower"].get("choice") == "deny"


def test_per_request_timeout_is_reflected_in_queue_metadata(monkeypatch):
    """The UI expiry must match the timeout requested by the MCP call."""
    approval._gateway_queues.clear()
    approval._gateway_notify_cbs.clear()
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 300)
    notified = []
    result = {}
    approval.register_gateway_notify("plan-b-session", lambda data: notified.append(data))

    def wait():
        result["value"] = approval._await_gateway_decision(
            "plan-b-session",
            lambda data: notified.append(data),
            {"command": "mcp", "description": "mcp", "pattern_key": "mcp"},
            timeout_seconds=0.05,
        )

    thread = threading.Thread(target=wait)
    thread.start()
    for _ in range(100):
        queue = approval._gateway_queues.get("plan-b-session")
        if queue:
            break
        time.sleep(0.005)
    entry = approval._gateway_queues["plan-b-session"][0]
    assert entry.data["expires_at"] - entry.data["created_at"] == pytest.approx(0.05, abs=0.03)
    thread.join(timeout=1)
    assert result["value"]["resolved"] is False


def test_mcp_cli_choice_accepts_only_once(monkeypatch):
    """MCP elicitation must not turn session/always into per-call consent."""
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "plan-b-session")
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(
        approval,
        "prompt_dangerous_approval",
        lambda *_args, **_kwargs: "session",
    )
    assert approval.request_elicitation_consent("msg", "desc") == "decline"


def test_approval_payload_is_allowlisted_and_url_redacted(monkeypatch):
    """Pending approvals must not copy arbitrary private metadata to clients."""
    payload = server._approval_request_payload(
        {
            "request_id": "req-1",
            "command": "curl https://user:pass@example.test/a?token=secret",
            "description": "open https://user:pass@example.test/?key=secret",
            "choices": ["once", "session", "always", "unknown"],
            "allow_session": True,
            "allow_permanent": True,
            "created_at": 1,
            "expires_at": 2,
            "private_raw_command": "do-not-send",
            "metadata": {"token": "do-not-send"},
        }
    )
    assert set(payload) <= {
        "request_id",
        "command",
        "description",
        "title",
        "choices",
        "allow_session",
        "allow_permanent",
        "smart_denied",
        "created_at",
        "expires_at",
    }
    assert "private_raw_command" not in payload
    assert "metadata" not in payload
    assert "user:pass@" not in payload["command"]
    assert "user:pass@" not in payload["description"]


def test_api_approval_event_is_allowlisted_before_status_persistence():
    """API/SSE approval events must not persist arbitrary queue fields."""
    from gateway.platforms.api_server_runs import _safe_api_approval_event

    event = _safe_api_approval_event(
        {
            "request_id": "req-api",
            "command": "curl https://user:pass@example.test/?token=secret",
            "description": "open https://user:pass@example.test/?key=secret",
            "choices": ["once", "deny"],
            "private_raw_command": "do-not-send",
            "metadata": {"token": "do-not-send"},
        },
        run_id="run-api",
        timestamp=1.0,
    )
    assert event["event"] == "approval.request"
    assert event["run_id"] == "run-api"
    assert set(event) <= {
        "request_id",
        "command",
        "description",
        "title",
        "choices",
        "allow_session",
        "allow_permanent",
        "smart_denied",
        "created_at",
        "expires_at",
        "event",
        "run_id",
        "timestamp",
    }
    assert "metadata" not in event
    assert "private_raw_command" not in event
    assert "user:pass@" not in event["command"]
    assert "user:pass@" not in event["description"]


def test_cli_exec_child_sentinel_blocks_dotenv_and_reload_reads(tmp_path, monkeypatch):
    """A remote CLI child must not reload credentials from its profile files."""
    from hermes_cli import config, env_loader

    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "PLAN_B_FIXTURE_TOKEN=fixture-value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DISABLE_DOTENV", "1")
    monkeypatch.delenv("PLAN_B_FIXTURE_TOKEN", raising=False)
    config.invalidate_env_cache()
    env_loader.reset_secret_source_cache()
    try:
        assert env_loader.load_hermes_dotenv(hermes_home=home) == []
        assert "PLAN_B_FIXTURE_TOKEN" not in __import__("os").environ
        assert config.load_env().get("PLAN_B_FIXTURE_TOKEN") is None
        config.reload_env()
        assert "PLAN_B_FIXTURE_TOKEN" not in __import__("os").environ
    finally:
        config.invalidate_env_cache()
        monkeypatch.delenv("HERMES_DISABLE_DOTENV", raising=False)


def test_no_notifier_pending_and_hooks_redact_raw_target(monkeypatch):
    """Legacy pending fallback and approval hooks must not receive raw text."""
    approval._pending.clear()
    approval.clear_session("plan-b-session")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda *_args: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "plan-b-session")
    monkeypatch.setattr(approval, "is_approved", lambda *_args: False)
    monkeypatch.setattr(approval, "_resolve_cli_approval_callback", lambda value=None: value)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {})
    raw_target = "curl https://user:pass@example.test/?token=secret"
    try:
        result = approval._run_approval_gate(
            pattern_key="dangerous",
            description="open https://user:pass@example.test/?key=secret",
            display_target=raw_target,
            cron_deny_message="cron denied",
            single_query_deny_message="single query denied",
            autoapprove_log_prefix="test",
        )
        assert result["status"] == "approval_required"
        pending = approval.get_pending_gateway_approval("plan-b-session")
        assert pending is not None
        assert "user:pass@" not in str(result)
        assert "token=secret" not in str(result)
        assert "user:pass@" not in str(pending)
        assert "token=secret" not in str(pending)
        with patch("hermes_cli.lifecycle.invoke_hook") as invoke_hook:
            approval._fire_approval_hook(
                "pre_approval_request",
                command=raw_target,
                description="open https://user:pass@example.test/?key=secret",
            )
        payload = invoke_hook.call_args.kwargs
        assert "user:pass@" not in str(payload)
        assert "token=secret" not in str(payload)
    finally:
        approval.clear_session("plan-b-session")


def test_malformed_remote_identity_cannot_execute(monkeypatch):
    """A legacy/malformed WS identity must fail closed even with a known sid."""
    sid = "plan-b-malformed-identity"
    server._sessions[sid] = _remote_session()
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport(None))
    try:
        with patch("subprocess.run") as run:
            response = server._methods["shell.exec"](
                "req-identity",
                {"session_id": sid, "command": "echo should-not-run"},
            )
        assert response["error"]["code"] == 4032
        run.assert_not_called()
    finally:
        server._sessions.pop(sid, None)


def test_malformed_remote_identity_cannot_own_approval_session(monkeypatch):
    """Malformed WS identity cannot read or resolve a dashboard approval."""
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport(None))
    session = {
        "source": "dashboard",
        "owner_principal": "dashboard:operator",
        "profile_name": "default",
        "profile_home": "/tmp/hermes-test/profiles/default",
    }
    assert server._approval_session_is_dashboard_owned(session) is False


def test_legacy_approval_session_without_home_is_out_of_profile_scope(monkeypatch):
    """Approval scope must not infer current profile for a legacy row."""
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    assert server._approval_session_profile({"profile_name": "default"}) == ""
    assert server._approval_session_matches_profile(
        {"profile_name": "default"}, "default"
    ) is False


def test_remote_shell_exec_runs_policy_before_subprocess(monkeypatch):
    """Remote shell execution must not reach subprocess before the shared guard."""
    sid = "plan-b-guard-order"
    server._sessions[sid] = _remote_session()
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport({"user_id": "operator", "provider": "dashboard"}))
    monkeypatch.setattr(
        approval,
        "check_all_command_guards",
        lambda *_args, **_kwargs: {"approved": False, "message": "blocked"},
    )
    try:
        with patch("subprocess.run") as run:
            response = server._methods["shell.exec"](
                "req-guard",
                {"session_id": sid, "command": "echo guarded"},
            )
        assert response["error"]["code"] == 4005
        run.assert_not_called()
    finally:
        server._sessions.pop(sid, None)


def test_remote_shell_exec_uses_real_request_id_approval_queue(monkeypatch):
    """A dangerous remote shell command waits for its exact approval ID."""
    sid = "plan-b-real-remote-approval"
    session = _remote_session()
    server._sessions[sid] = session
    transport = _RemoteTransport({"user_id": "operator", "provider": "dashboard"})
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "_command_matches_permanent_allowlist", lambda *_args: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda *_args: False)
    notified = []
    approval.register_gateway_notify(session["session_key"], notified.append)
    completed = type(
        "Completed",
        (),
        {"stdout": "", "stderr": "", "returncode": 0},
    )()
    result = {}

    def invoke():
        result["response"] = server._methods["shell.exec"](
            "req-real-remote",
            {"session_id": sid, "command": "rm -rf /tmp/plan-b-approval"},
        )

    thread = threading.Thread(target=invoke)
    try:
        with patch("subprocess.run", return_value=completed) as run:
            thread.start()
            for _ in range(200):
                if notified:
                    break
                time.sleep(0.005)
            assert notified, "remote dangerous command did not emit approval"
            request_id = notified[0]["request_id"]
            assert request_id
            assert thread.is_alive(), "remote command executed before approval"
            assert approval.resolve_gateway_approval(
                session["session_key"], "once", request_id=request_id
            ) == 1
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert result["response"]["result"]["code"] == 0
            run.assert_called_once()
    finally:
        approval.unregister_gateway_notify(session["session_key"])
        server._sessions.pop(sid, None)


def test_remote_cli_exec_runs_policy_before_subprocess(monkeypatch):
    """Remote cli.exec must stop when the shared guard denies the argv."""
    sid = "plan-b-cli-guard-order"
    server._sessions[sid] = _remote_session()
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport({"user_id": "operator", "provider": "dashboard"}))
    monkeypatch.setattr(
        approval,
        "check_all_command_guards",
        lambda *_args, **_kwargs: {"approved": False, "message": "blocked"},
    )
    try:
        with patch("subprocess.run") as run:
            response = server._methods["cli.exec"](
                "req-cli-guard",
                {"session_id": sid, "argv": ["--version"]},
            )
        assert response["error"]["code"] == 4005
        run.assert_not_called()
    finally:
        server._sessions.pop(sid, None)


def test_remote_cli_exec_disables_child_dotenv(monkeypatch):
    """Authenticated cli.exec must pass the child dotenv opt-out."""
    sid = "plan-b-cli-dotenv"
    server._sessions[sid] = _remote_session()
    monkeypatch.setattr(
        server,
        "current_transport",
        lambda: _RemoteTransport({"user_id": "operator", "provider": "dashboard"}),
    )
    monkeypatch.setattr(
        approval,
        "check_all_command_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    completed = type(
        "Completed",
        (),
        {"stdout": "", "stderr": "", "returncode": 0},
    )()
    try:
        with patch("subprocess.run", return_value=completed) as run:
            response = server._methods["cli.exec"](
                "req-cli-dotenv",
                {"session_id": sid, "argv": ["--version"]},
            )
        assert response["result"]["blocked"] is False
        assert run.call_args.kwargs["env"]["HERMES_DISABLE_DOTENV"] == "1"
    finally:
        server._sessions.pop(sid, None)


def test_remote_quick_exec_requires_session_and_policy(monkeypatch):
    """Quick-command exec cannot bypass the remote session/policy boundary."""
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport({"user_id": "operator", "provider": "dashboard"}))
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"quick_commands": {"runcmd": {"type": "exec", "command": "echo hi"}}},
    )
    with patch("subprocess.run") as run:
        response = server._methods["command.dispatch"](
            "req-quick-missing",
            {"name": "runcmd", "arg": "", "session_id": ""},
        )
    assert response["error"]["code"] == 4032
    run.assert_not_called()


def test_remote_quick_exec_runs_policy_before_subprocess(monkeypatch):
    """A valid remote quick-command session still requires the shared guard."""
    sid = "plan-b-quick-guard-order"
    server._sessions[sid] = _remote_session()
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport({"user_id": "operator", "provider": "dashboard"}))
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"quick_commands": {"runcmd": {"type": "exec", "command": "echo hi"}}},
    )
    monkeypatch.setattr(
        approval,
        "check_all_command_guards",
        lambda *_args, **_kwargs: {"approved": False, "message": "blocked"},
    )
    try:
        with patch("subprocess.run") as run:
            response = server._methods["command.dispatch"](
                "req-quick-guard",
                {"name": "runcmd", "arg": "", "session_id": sid},
            )
        assert response["error"]["code"] == 4005
        run.assert_not_called()
    finally:
        server._sessions.pop(sid, None)
