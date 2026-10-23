"""Copy to tests/gateway/test_activation_evidence.py in an isolated patched Hermes.

Run through scripts/run_tests.sh. Real runner/config classes, control registration,
process birth identity and native transport; no production state or external model call.
"""
import asyncio
import os
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.control_socket import query_gateway_control
from gateway.run import GatewayRunner
from gateway.run_bootstrap import _start_gateway_start_control_socket
from gateway.status import _get_process_hermes_home, get_process_start_time


@pytest.mark.platforms("windows")
@pytest.mark.asyncio
async def test_registered_export_projects_real_resolved_configs():
    runner = object.__new__(GatewayRunner)
    runner.session_runtime_descriptor = {}
    runner.session_ticket_store = None
    runner._running = True
    runner._primary_profile_name = "default"
    runner.config = GatewayConfig(multiplex_profiles=True, activation_evidence_export=True)
    runner.config.platforms = {Platform.WEBHOOK: PlatformConfig(enabled=True)}
    secondary = GatewayConfig()
    secondary.platforms = {Platform.WEBHOOK: PlatformConfig(enabled=False)}
    runner._profile_configs = {"secondary": secondary}
    runner._served_profile_homes = {"default": None, "secondary": None}
    runner._served_profile_signatures = {"secondary": ()}
    runner._profile_reconcile_lock = asyncio.Lock()
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    try:
        home = _get_process_hermes_home()
        reply = await asyncio.to_thread(query_gateway_control, home, "activation-evidence")
        assert reply is not None
        assert reply["runtimePid"] == os.getpid()
        assert reply["runtimeStartTime"] == get_process_start_time(os.getpid())
        assert Path(reply["installDirectory"]) == Path(__file__).resolve().parents[2]
        assert reply["profiles"] == {"default": {"webhookEnabled": True},
                                     "secondary": {"webhookEnabled": False}}
        secondary.platforms[Platform.WEBHOOK].enabled = True
        changed = await asyncio.to_thread(query_gateway_control, home, "activation-evidence")
        assert changed["profiles"]["secondary"]["webhookEnabled"] is True
        await runner._profile_reconcile_lock.acquire()
        assert await asyncio.to_thread(query_gateway_control, home, "activation-evidence") is None
        runner._profile_reconcile_lock.release()
    finally:
        await server.stop()


@pytest.mark.platforms("windows")
@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [{}, {"activation_evidence_export": False}])
async def test_export_is_absent_without_explicit_opt_in(settings):
    runner = object.__new__(GatewayRunner)
    runner.session_runtime_descriptor = {}
    runner.session_ticket_store = None
    runner.config = GatewayConfig(**settings)
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    try:
        home = _get_process_hermes_home()
        assert await asyncio.to_thread(query_gateway_control, home, "identify") is not None
        assert "activation-evidence" not in server._handlers
        assert await asyncio.to_thread(query_gateway_control, home, "activation-evidence") is None
    finally:
        await server.stop()


@pytest.mark.parametrize("value", [None, "false", "true", 0, 1, [], {}])
def test_invalid_explicit_opt_in_never_falls_back(value):
    with pytest.raises(ValueError, match="activation_evidence_export"):
        GatewayConfig.from_dict({"activation_evidence_export": value,
                                 "gateway": {"activation_evidence_export": True}})
    with pytest.raises(ValueError, match="activation_evidence_export"):
        GatewayConfig(activation_evidence_export=value)


@pytest.mark.parametrize("yaml_text, expected", [
    ("{}", False),
    ("gateway:\n  activation_evidence_export: true\n", True),
    ("gateway:\n  activation_evidence_export: false\n", False),
    ("activation_evidence_export: false\ngateway:\n  activation_evidence_export: true\n", False),
    ('gateway:\n  activation_evidence_export: "false"\n', None),
    ("activation_evidence_export: null\ngateway:\n  activation_evidence_export: true\n", None),
])
def test_real_yaml_opt_in_and_roundtrip(tmp_path, monkeypatch, yaml_text, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(yaml_text, encoding="utf-8")
    if expected is None:
        with pytest.raises(ValueError, match="activation_evidence_export"):
            load_gateway_config()
    else:
        config = load_gateway_config()
        assert config.activation_evidence_export is expected
        assert GatewayConfig.from_dict(config.to_dict()).activation_evidence_export is expected


def test_invalid_yaml_cannot_reenable_a_legacy_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "gateway.json").write_text('{"activation_evidence_export": true}', encoding="utf-8")
    (tmp_path / "config.yaml").write_text("gateway:\n  activation_evidence_export: [", encoding="utf-8")
    with pytest.raises(ValueError, match="activation_evidence_export"):
        load_gateway_config()
