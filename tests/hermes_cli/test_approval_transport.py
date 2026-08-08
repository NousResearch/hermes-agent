"""Approval transport plugin contract and fail-closed host routing."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
import yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _manifest(name: str = "fixture-approval") -> PluginManifest:
    return PluginManifest(
        name=name,
        version="1.0.0",
        description="fixture",
        source="user",
        key=name,
    )


def _context(manager: PluginManager, name: str = "fixture-approval") -> PluginContext:
    return PluginContext(_manifest(name), manager)


def _request():
    from hermes_cli.approval_transport import ApprovalRequest

    return ApprovalRequest.create(
        command="rm -rf /tmp/example",
        description="recursive delete",
        pattern_key="rm_recursive",
        pattern_keys=("rm_recursive",),
        session_key="session-a",
        surface="cli",
        allow_session=True,
        allow_permanent=True,
    )


def test_context_registers_owned_approval_transport():
    manager = PluginManager()
    callback = lambda request: request.respond("deny")

    _context(manager).register_approval_transport("phone", callback)

    registered = manager.get_approval_transport("phone")
    assert registered is not None
    assert registered.name == "phone"
    assert registered.plugin_id == "fixture-approval"
    assert registered.present is callback


def test_transport_names_are_unique_and_builtin_is_reserved():
    manager = PluginManager()
    _context(manager, "first").register_approval_transport("phone", lambda request: None)

    with pytest.raises(ValueError, match="already registered"):
        _context(manager, "second").register_approval_transport("phone", lambda request: None)
    with pytest.raises(ValueError, match="reserved"):
        _context(manager).register_approval_transport("builtin", lambda request: None)


def test_force_reload_clears_transport_registry(monkeypatch):
    manager = PluginManager()
    _context(manager).register_approval_transport("phone", lambda request: None)
    manager._discovered = True
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)

    manager.discover_and_load(force=True)

    assert manager.get_approval_transport("phone") is None


def test_transport_registry_is_manager_and_profile_isolated(monkeypatch, tmp_path):
    first = PluginManager()
    second = PluginManager()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "work"))
    _context(first).register_approval_transport("phone", lambda request: None)

    assert first.get_approval_transport("phone") is not None
    assert second.get_approval_transport("phone") is None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "personal"))
    assert first.get_approval_transport("phone") is None


def test_host_accepts_bound_sync_and_async_decisions():
    from hermes_cli.approval_transport import invoke_approval_transport

    request = _request()
    sync_result = invoke_approval_transport(
        lambda received: received.respond("session"), request, timeout_seconds=1
    )

    async def present(received):
        await asyncio.sleep(0)
        return received.respond("once")

    async_result = invoke_approval_transport(present, request, timeout_seconds=1)

    assert sync_result.choice == "session"
    assert sync_result.failure is None
    assert async_result.choice == "once"
    assert async_result.failure is None


def test_host_rejects_scope_not_offered_by_request():
    from hermes_cli.approval_transport import ApprovalRequest, invoke_approval_transport

    request = ApprovalRequest.create(
        command="dangerous",
        description="dangerous",
        pattern_key="danger",
        pattern_keys=("danger",),
        session_key="session-a",
        surface="cli",
        allow_session=False,
        allow_permanent=False,
    )

    result = invoke_approval_transport(
        lambda received: received.respond("always"), request, timeout_seconds=1
    )

    assert result.choice == "deny"
    assert result.failure == "invalid"


@pytest.mark.parametrize(
    ("present", "failure"),
    [
        (lambda request: {"choice": "once"}, "invalid"),
        (lambda request: request.respond("bogus"), "invalid"),
        (
            lambda request: type(request.respond("once"))(
                request_id="stale",
                request_digest=request.digest,
                choice="once",
            ),
            "stale",
        ),
        (
            lambda request: type(request.respond("once"))(
                request_id=request.request_id,
                request_digest="changed",
                choice="once",
            ),
            "stale",
        ),
    ],
)
def test_host_rejects_invalid_or_stale_decisions(present, failure):
    from hermes_cli.approval_transport import invoke_approval_transport

    result = invoke_approval_transport(present, _request(), timeout_seconds=1)

    assert result.choice == "deny"
    assert result.failure == failure


def test_host_timeout_and_exception_deny_without_waiting_forever():
    from hermes_cli.approval_transport import invoke_approval_transport

    def hangs(_request):
        time.sleep(1)

    def crashes(_request):
        raise RuntimeError("transport offline")

    started = time.monotonic()
    timeout = invoke_approval_transport(hangs, _request(), timeout_seconds=0.02)
    exception = invoke_approval_transport(crashes, _request(), timeout_seconds=1)

    assert time.monotonic() - started < 0.5
    assert (timeout.choice, timeout.failure) == ("deny", "timeout")
    assert (exception.choice, exception.failure) == ("deny", "error")


def test_host_transport_wait_is_interruptible_and_pollable():
    from hermes_cli.approval_transport import invoke_approval_transport

    polls = []

    def hangs(_request):
        time.sleep(1)

    result = invoke_approval_transport(
        hangs,
        _request(),
        timeout_seconds=1,
        poll_interval=0.01,
        on_poll=lambda: polls.append(1),
        is_interrupted=lambda: len(polls) >= 1,
    )

    assert result.choice == "deny"
    assert result.failure == "interrupted"
    assert polls


def test_host_caps_hung_transport_workers():
    from hermes_cli.approval_transport import (
        _MAX_ACTIVE_TRANSPORT_WORKERS,
        invoke_approval_transport,
    )

    release = threading.Event()

    def hangs(_request):
        release.wait()

    try:
        results = [
            invoke_approval_transport(hangs, _request(), timeout_seconds=0.001)
            for _ in range(_MAX_ACTIVE_TRANSPORT_WORKERS + 1)
        ]
    finally:
        release.set()

    assert results[-1].choice == "deny"
    assert results[-1].failure == "busy"


def _configure_manual_guard(monkeypatch, approval_module, manager, *, fallback=None):
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_module, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "detect_hardline_command", lambda command: (False, ""))
    monkeypatch.setattr(approval_module, "_check_sudo_stdin_guard", lambda command: (False, ""))
    monkeypatch.setattr(approval_module, "_match_user_deny_rule", lambda command: None)
    monkeypatch.setattr(approval_module, "_command_matches_permanent_allowlist", lambda command: False)
    monkeypatch.setattr(approval_module, "detect_dangerous_command", lambda command: (True, "danger", "dangerous"))
    monkeypatch.setattr(
        approval_module,
        "get_current_session_key",
        lambda *args, **kwargs: "session-a",
    )
    monkeypatch.setattr(approval_module, "is_approved", lambda *args: False)
    monkeypatch.setattr(approval_module, "get_plugin_manager", lambda: manager, raising=False)
    monkeypatch.setattr(
        approval_module,
        "_get_approval_transport_config",
        lambda: ("phone", fallback),
        raising=False,
    )
    monkeypatch.setattr("tools.tirith_security.check_command_security", lambda command: {"action": "allow"})


def test_cli_selected_transport_replaces_builtin_prompt(monkeypatch):
    from tools import approval

    manager = PluginManager()
    seen = []
    _context(manager).register_approval_transport(
        "phone", lambda request: seen.append(request) or request.respond("once")
    )
    _configure_manual_guard(monkeypatch, approval, manager)

    def builtin(*args, **kwargs):
        raise AssertionError("builtin prompt must not materialize")

    result = approval.check_all_command_guards(
        "rm -rf /tmp/example", "local", approval_callback=builtin
    )

    assert result["approved"] is True
    assert result["user_approved"] is True
    assert len(seen) == 1
    assert seen[0].command == "rm -rf /tmp/example"
    assert seen[0].allowed_choices == ("once", "session", "always", "deny")


def test_gateway_selected_transport_does_not_require_gateway_notifier(monkeypatch):
    from tools import approval

    manager = PluginManager()
    seen = []
    _context(manager).register_approval_transport(
        "phone", lambda request: seen.append(request) or request.respond("once")
    )
    _configure_manual_guard(monkeypatch, approval, manager)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {})

    result = approval.check_all_command_guards("rm -rf /tmp/example", "local")

    assert result["approved"] is True
    assert len(seen) == 1
    assert seen[0].surface == "gateway"


def test_execute_code_gateway_uses_selected_transport(monkeypatch):
    from tools import approval

    manager = PluginManager()
    seen = []
    _context(manager).register_approval_transport(
        "phone", lambda request: seen.append(request) or request.respond("once")
    )
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(
        approval, "get_current_session_key", lambda *args, **kwargs: "session-a"
    )
    monkeypatch.setattr(approval, "is_approved", lambda *args: False)
    monkeypatch.setattr(approval, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(
        approval, "_get_approval_transport_config", lambda: ("phone", None)
    )
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {})

    result = approval.check_execute_code_guard("print('ok')", "local")

    assert result["approved"] is True
    assert len(seen) == 1
    assert seen[0].pattern_key == "execute_code"
    assert seen[0].surface == "gateway"


def test_transport_failure_denies_without_builtin_fallback(monkeypatch):
    from tools import approval

    manager = PluginManager()
    _context(manager).register_approval_transport("phone", lambda request: {"choice": "once"})
    _configure_manual_guard(monkeypatch, approval, manager)
    builtin_calls = []

    result = approval.check_all_command_guards(
        "rm -rf /tmp/example",
        "local",
        approval_callback=lambda *args, **kwargs: builtin_calls.append(1) or "once",
    )

    assert result["approved"] is False
    assert result["outcome"] == "transport_invalid"
    assert builtin_calls == []


def test_redaction_failure_denies_before_transport_callback(monkeypatch):
    import agent.redact
    from tools import approval

    manager = PluginManager()
    calls = []
    _context(manager).register_approval_transport(
        "phone", lambda request: calls.append(request) or request.respond("once")
    )
    _configure_manual_guard(monkeypatch, approval, manager)

    def redaction_failed(text):
        if text in {"rm -rf /tmp/example", "dangerous"}:
            raise RuntimeError("redactor unavailable")
        return text

    monkeypatch.setattr(agent.redact, "redact_sensitive_text", redaction_failed)

    result = approval.check_all_command_guards("rm -rf /tmp/example", "local")

    assert result["approved"] is False
    assert result["outcome"] == "transport_error"
    assert calls == []


def test_explicit_builtin_fallback_uses_existing_surface(monkeypatch):
    from tools import approval

    manager = PluginManager()
    _context(manager).register_approval_transport("phone", lambda request: {"choice": "once"})
    _configure_manual_guard(monkeypatch, approval, manager, fallback="builtin")
    builtin_calls = []

    result = approval.check_all_command_guards(
        "rm -rf /tmp/example",
        "local",
        approval_callback=lambda *args, **kwargs: builtin_calls.append(1) or "once",
    )

    assert result["approved"] is True
    assert builtin_calls == [1]


def test_live_temp_home_fixture_plugin_routes_and_hardline_stays_core_owned(
    tmp_path, monkeypatch
):
    """Real discovery + config + guard path under an isolated HERMES_HOME."""
    import hermes_cli.plugins as plugins_module
    from tools import approval

    home = tmp_path / "hermes-home"
    plugin_dir = home / "plugins" / "fixture-approval"
    bundled = tmp_path / "empty-bundled"
    plugin_dir.mkdir(parents=True)
    bundled.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "fixture-approval",
                "version": "1.0.0",
                "description": "approval transport fixture",
            }
        )
    )
    (plugin_dir / "__init__.py").write_text(
        """import json
import os
from pathlib import Path


def present(request):
    output = Path(os.environ["HERMES_HOME"]) / "transport-invocations.jsonl"
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "request_id": request.request_id,
            "digest": request.digest,
            "command": request.command,
            "surface": request.surface,
            "timeout_seconds": request.timeout_seconds,
        }) + "\\n")
    return request.respond("once")


def register(ctx):
    ctx.register_approval_transport("fixture", present)
"""
    )
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": ["fixture-approval"]},
                "approvals": {"mode": "manual", "timeout": 2},
                "security": {
                    "tirith_enabled": False,
                    "approval": {"transport": "fixture"},
                },
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    manager = PluginManager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)
    manager.discover_and_load()
    token = approval.set_hermes_interactive_context(True)
    approval.clear_session("local")
    approval._permanent_approved.clear()
    try:
        routed = approval.check_all_command_guards(
            "rm -rf /tmp/hermes-approval-transport-fixture", "local"
        )
        manager.discover_and_load(force=True)
        reloaded = approval.check_all_command_guards(
            "rm -rf /tmp/hermes-approval-transport-fixture-reloaded", "local"
        )
        gateway_token = approval.set_hermes_interactive_context(False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        try:
            gateway_routed = approval.check_all_command_guards(
                "rm -rf /tmp/hermes-approval-transport-fixture-gateway", "local"
            )
        finally:
            approval.reset_hermes_interactive_context(gateway_token)
        hardline = approval.check_all_command_guards("rm -rf /", "local")
    finally:
        approval.reset_hermes_interactive_context(token)

    records = [
        json.loads(line)
        for line in (home / "transport-invocations.jsonl").read_text().splitlines()
    ]
    assert routed["approved"] is True
    assert reloaded["approved"] is True
    assert gateway_routed["approved"] is True
    assert records[0]["request_id"]
    assert records[0]["digest"]
    assert records[0]["surface"] == "cli"
    assert records[0]["timeout_seconds"] == 2
    assert records[2]["surface"] == "gateway"
    assert hardline["approved"] is False
    assert len(records) == 3


def test_hardline_blocks_before_selected_transport(monkeypatch):
    from tools import approval

    manager = PluginManager()
    calls = []
    _context(manager).register_approval_transport(
        "phone", lambda request: calls.append(request) or request.respond("once")
    )
    _configure_manual_guard(monkeypatch, approval, manager)
    monkeypatch.setattr(
        approval,
        "detect_hardline_command",
        lambda command: (True, "recursive delete of root filesystem"),
    )

    result = approval.check_all_command_guards("rm -rf /", "local")

    assert result["approved"] is False
    assert calls == []
