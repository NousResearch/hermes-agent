"""A native gateway launch must respect a required plugin's unfinished update."""

import pytest


@pytest.mark.asyncio
async def test_manual_gateway_start_stops_before_replacing_owner_when_guard_pending(
        tmp_path, monkeypatch):
    from gateway import run_bootstrap
    from gateway.config import GatewayConfig
    from hermes_cli import update_plugin_guards

    marker = tmp_path / "pending.json"
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(update_plugin_guards, "pending_path", lambda source: marker)

    def premature_start():
        raise AssertionError("gateway touched startup state before the plugin check")

    monkeypatch.setattr(run_bootstrap, "_start_gateway_resolve_terminal_cwd", premature_start)
    assert await run_bootstrap.start_gateway(GatewayConfig(), replace=True) is False

    marker.unlink()
    assert update_plugin_guards.pending_reason(tmp_path) is None
