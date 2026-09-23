"""Regression for #120165: isolated backends must not own the shared host record."""

import asyncio
from types import SimpleNamespace

from gateway import host_rendezvous as hr
from hermes_cli import web_server


def test_isolated_start_leaves_host_rendezvous_for_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setattr(web_server, "_read_bound_port", lambda server, fallback: fallback)
    monkeypatch.setattr(web_server, "_start_parent_death_watchdog", lambda: None)
    monkeypatch.setattr(web_server, "_write_dashboard_ready_file", lambda port: None)
    monkeypatch.setattr(web_server, "_write_machine_sentinel_line", lambda line: None)
    monkeypatch.setattr(web_server, "_maybe_open_browser", lambda *args: None)
    monkeypatch.setattr(web_server, "_best_effort",
                        lambda what, fn: fn() if what == "host rendezvous publish" else None)

    async def started(isolated, port):
        web_server._on_server_started(
            SimpleNamespace(), host="127.0.0.1", port=port, headless=isolated,
            isolated=isolated, open_browser=False, initial_profile="",
            start_mcp_discovery_after_bind=False,
        )

    asyncio.run(started(True, 9120))
    assert hr.read_record(hr.ROLE_SERVE) is None
    assert not hr.owns_host_lock(hr.ROLE_SERVE)

    asyncio.run(started(False, 9119))
    record = hr.read_record(hr.ROLE_SERVE)
    assert record is not None
    assert record.port == 9119
    assert hr.owns_host_lock(hr.ROLE_SERVE)
    hr.clear_record(hr.ROLE_SERVE)
    hr.release_host_lock(hr.ROLE_SERVE)
