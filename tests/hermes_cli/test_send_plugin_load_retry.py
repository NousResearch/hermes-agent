"""Regression: ``hermes send`` retries plugin discovery once after a load-timeout.

Out-of-process send used to map a plugin-load miss to exit 1 and
``Is the gateway running with this platform connected?`` even when the
gateway was healthy. Attempt 1 stays the hang-guard (10s default, late
``register_platform`` discarded). The send path then does exactly one
fresh discovery with a longer per-call deadline.

These tests import only names that exist on unpatched v0.23.1+2.gf309630
so a miss is an assertion failure, not a collection error.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import types

import pytest

from hermes_cli import plugins_loader, send_cmd
from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

# Local copies of the contract (do not import the new helper on the red run).
_GATEWAY_BLAME = "Is the gateway running with this platform connected?"
_TIMEOUT_MARKER = "plugin_load_timeout platform="
_NO_ADAPTER = "No live adapter for platform '{platform}'. " + _GATEWAY_BLAME


def _parse(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    send_cmd.register_send_subparser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["send", *argv])


def _install_send_tool(monkeypatch, payloads):
    """ sequential JSON payloads for ``send_message_tool``; records each call. """
    calls = []

    def fake(args, **_kw):
        calls.append(dict(args))
        idx = min(len(calls) - 1, len(payloads) - 1)
        payload = payloads[idx]
        return payload if isinstance(payload, str) else json.dumps(payload)

    mod = types.ModuleType("tools.send_message_tool")
    mod.send_message_tool = fake
    monkeypatch.setitem(sys.modules, "tools.send_message_tool", mod)
    monkeypatch.setattr(send_cmd, "_load_hermes_env", lambda: None)
    return calls


def _plugin_row(*, name, error, kind="platform"):
    return {
        "name": name,
        "key": name,
        "kind": kind,
        "version": "1.0",
        "description": "",
        "source": "user",
        "enabled": False,
        "tools": 0,
        "hooks": 0,
        "middleware": 0,
        "commands": 0,
        "error": error,
    }


class _FakeManager:
    def __init__(self, rows):
        self._rows = list(rows)

    def list_plugins(self):
        return list(self._rows)


def test_load_timeout_default_is_ten_seconds():
    """Hang-guard default stays 10.0; this patch must not raise or delete it."""
    assert plugins_loader._LOAD_TIMEOUT_SECS == 10.0


def test_timeout_miss_retries_once_then_sends(monkeypatch, capsys):
    """(a) attempt 1 misses (b) exactly one retry (c) attempt 2 delivers."""
    no_adapter = {"error": _NO_ADAPTER.format(platform="discord")}
    sent = {"success": True, "message_id": "m-ok"}
    calls = _install_send_tool(monkeypatch, [no_adapter, sent])

    discover_calls = []
    resolved_during_retry = []

    def fake_discover(*, force=False):
        discover_calls.append({"force": force, "timeout": plugins_loader._resolve_plugin_load_timeout()})
        resolved_during_retry.append(plugins_loader._resolve_plugin_load_timeout())

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", fake_discover)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: _FakeManager([_plugin_row(name="discord-platform",
                                          error="load timed out after 10s (import + register() never returned)")]),
    )

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(["--to", "discord:#crew", "alert"]))
    out, err = capsys.readouterr()

    assert exc.value.code == 0
    assert len(calls) == 2, f"expected attempt1 miss + exactly one retry, got {len(calls)} sends"
    assert discover_calls == [{"force": True, "timeout": 30.0}]
    assert resolved_during_retry == [30.0]
    assert (out.strip() == "sent") or ("sent" in out)
    assert _GATEWAY_BLAME not in err
    assert _TIMEOUT_MARKER not in err


def test_double_miss_exits_2_with_timeout_marker(monkeypatch, capsys):
    """(d) both attempts miss → exit 2 and the plugin_load_timeout line; no gateway blame."""
    no_adapter = {"error": _NO_ADAPTER.format(platform="discord")}
    calls = _install_send_tool(monkeypatch, [no_adapter, no_adapter])
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda **_k: None)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: _FakeManager([_plugin_row(name="discord-platform",
                                          error="load timed out after 10s (import + register() never returned)")]),
    )

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(["--to", "discord:#crew", "alert"]))
    _out, err = capsys.readouterr()

    assert exc.value.code == 2
    assert len(calls) == 2
    assert err.strip() == "plugin_load_timeout platform=discord attempts=2"
    assert _GATEWAY_BLAME not in err


def test_late_register_platform_after_abandon_is_ignored(caplog):
    """(e) hang-guard: a late register_platform() from the abandoned load is discarded."""
    from gateway.platform_registry import platform_registry

    ctx = PluginContext(PluginManifest(name="retryprobe-platform", source="user", kind="platform"), PluginManager())
    ctx._abandon_load()
    handle = ctx.register_platform(
        name="retryprobe",
        label="Retry Probe",
        adapter_factory=lambda _cfg: None,
        check_fn=lambda: True,
    )
    assert handle is None
    assert platform_registry.get("retryprobe") is None
    assert any("after its load timed out" in rec.getMessage() and "register_platform" in rec.getMessage()
               for rec in caplog.records)


def test_hung_first_load_late_register_still_ignored(monkeypatch, caplog):
    """(e) via run_with_load_deadline: first pass times out; abandoned worker's late register is ignored."""
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 0.2)
    from gateway.platform_registry import platform_registry

    ctx = PluginContext(PluginManifest(name="retryhang-platform", source="user", kind="platform"), PluginManager())
    release = threading.Event()
    registered = []

    def hung_register():
        release.wait()
        ctx.register_platform(
            name="retryhang",
            label="Retry Hang",
            adapter_factory=lambda _cfg: None,
            check_fn=lambda: True,
        )
        registered.append("late")

    with pytest.raises(plugins_loader.PluginLoadTimeout):
        plugins_loader.run_with_load_deadline("retryhang-platform", ctx, hung_register)
    release.set()
    from tests.waiting import wait_until
    wait_until(lambda: bool(registered) or any(
        "after its load timed out" in rec.getMessage() for rec in caplog.records
    ), timeout=2.0)
    assert platform_registry.get("retryhang") is None
    assert getattr(ctx, "_load_abandoned", False) is True


def test_non_timeout_plugin_failure_does_not_retry(monkeypatch, capsys):
    """(f) a non-timeout miss keeps exit 1, original message, no second discovery."""
    err_payload = {"error": _NO_ADAPTER.format(platform="discord")}
    calls = _install_send_tool(monkeypatch, [err_payload])
    discover_calls = []
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda **_k: discover_calls.append(1))
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: _FakeManager([_plugin_row(name="discord-platform", error="no register() function")]),
    )

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(["--to", "discord:#crew", "alert"]))
    _out, err = capsys.readouterr()

    assert exc.value.code == 1
    assert len(calls) == 1
    assert discover_calls == []
    assert _GATEWAY_BLAME in err
    assert _TIMEOUT_MARKER not in err


def test_other_send_failure_does_not_retry(monkeypatch, capsys):
    """Any other failure keeps today's path: exit 1, original message, no retry."""
    payload = {"error": "Adapter send failed: boom"}
    calls = _install_send_tool(monkeypatch, [payload])
    discover_calls = []
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda **_k: discover_calls.append(1))
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: _FakeManager([_plugin_row(name="discord-platform",
                                          error="load timed out after 10s (import + register() never returned)")]),
    )

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(["--to", "discord:#crew", "alert"]))
    _out, err = capsys.readouterr()

    assert exc.value.code == 1
    assert len(calls) == 1
    assert discover_calls == []
    assert "Adapter send failed: boom" in err
    assert _TIMEOUT_MARKER not in err
