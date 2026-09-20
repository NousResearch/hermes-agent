"""Gateway-independent draining of restart-safe cron deliveries."""

from contextlib import contextmanager
from types import SimpleNamespace

import cron.scheduler as scheduler
from cron import scheduler_preflight as sched_preflight
import gateway.run as gateway_run


class _OneTickStopEvent:
    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_drains_cron_delivery_with_live_adapters(monkeypatch):
    adapters = {"discord": object()}
    loop = object()
    calls = []
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda live_adapters, live_loop: calls.append((live_adapters, live_loop)),
        raising=False,
    )

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(), adapters=adapters, loop=loop, interval=0
    )

    assert calls == [(adapters, loop)]


def test_gateway_housekeeping_drains_cron_delivery_without_connected_adapters(monkeypatch):
    adapters = {}
    loop = object()
    calls = []
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda live_adapters, live_loop: calls.append((live_adapters, live_loop)),
        raising=False,
    )

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(), adapters=adapters, loop=loop, interval=0
    )

    assert calls == [(adapters, loop)]


def test_multiplex_housekeeping_scopes_primary_and_drains_each_profile(
    tmp_path, monkeypatch
):
    root_adapters = {}
    secondary_adapters = {}
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        adapters=root_adapters,
        _profile_adapters={"secondary": secondary_adapters},
    )
    root_home = tmp_path / "root"
    secondary_home = tmp_path / "secondary"
    calls = []

    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: root_home)

    monkeypatch.setattr(
        gateway_run,
        "_handoff_watch_scopes",
        lambda _runner: [(None, None), ("secondary", secondary_home)],
    )

    @contextmanager
    def fake_scope(home):
        calls.append(("scope", home))
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    from gateway import run_profile_reconcile
    monkeypatch.setattr(run_profile_reconcile, "_mcp_config_reconciler", lambda runner: lambda: None)
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda adapters, loop: calls.append(("drain", adapters)),
    )

    gateway_run._start_gateway_housekeeping(
        _OneTickStopEvent(),
        adapters=root_adapters,
        loop=object(),
        interval=0,
        runner=runner,
    )

    assert calls == [
        ("scope", root_home),
        ("drain", root_adapters),
        ("scope", secondary_home),
        ("drain", secondary_adapters),
    ]


def test_multiplex_housekeeping_uses_primary_routes_for_credentialless_satellite(
    tmp_path, monkeypatch
):
    root_adapters = {"slack": object()}
    secondary_home = tmp_path / "secondary"
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        adapters=root_adapters,
        _profile_adapters={"secondary": {}},
    )
    calls = []
    routed = object()

    monkeypatch.setattr(
        gateway_run,
        "_handoff_watch_scopes",
        lambda _runner: [(None, None), ("secondary", secondary_home)],
    )

    @contextmanager
    def fake_scope(_home):
        yield

    class FakeSharedRouteAdapters:
        def __new__(cls, adapters, routes):
            calls.append(("routed", adapters, routes))
            return routed

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(sched_preflight, "SharedRouteAdapters", FakeSharedRouteAdapters)
    monkeypatch.setattr(sched_preflight, "_primary_profile_routes_for_current_home",
        lambda: ["route-to-secondary"],
    )
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda adapters, _loop: calls.append(("drain", adapters)),
    )

    gateway_run._drain_restart_safe_cron_deliveries(
        root_adapters, object(), runner
    )

    assert calls == [
        ("drain", root_adapters),
        ("routed", root_adapters, ["route-to-secondary"]),
        ("drain", routed),
    ]


def test_primary_drain_delivers_credentialless_satellite_queue_row_through_primary_adapter(
    tmp_path, monkeypatch,
):
    """Regression for #115656 (the chain #101113's tests skipped): an external restart-safe worker
    durably queues a routed satellite's result; the PRIMARY gateway PID claims the row under the
    satellite's scope and must send it through its live adapter, never the satellite's
    credentialless standalone lane. Before, ``_live_send_text`` re-resolved the transport from the
    plain adapter dict and reported ``No adapter configured for discord``.
    """
    import asyncio
    import threading
    from unittest.mock import patch

    import yaml

    from cron import delivery_queue
    from gateway.config import Platform
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / "root"
    satellite_home = root / "profiles" / "satellite"
    (satellite_home / "cron").mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"multiplex_profiles": True, "profile_routes": [
            {"platform": "discord", "guild_id": "G1", "chat_id": "C1", "profile": "satellite"}]},
    }), encoding="utf-8")
    # The satellite names its home channel but holds no token: block present, ``enabled`` False.
    (satellite_home / "config.yaml").write_text(
        yaml.safe_dump({"platforms": {"discord": {"home_channel": "C1"}}}), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)

    sent, standalone = [], []

    class PrimaryDiscord:
        async def send(self, chat_id, content, metadata=None):
            sent.append(chat_id)
            return {"success": True, "message_id": "m1"}

    async def _standalone(platform, pconfig, chat_id, text, **kwargs):
        standalone.append(chat_id)
        return {"success": False, "error": "DISCORD_BOT_TOKEN is not set"}

    # The external worker: no adapters, queues under the OWNING (satellite) home.
    token = set_hermes_home_override(str(satellite_home))
    try:
        delivery_queue.enqueue("exec-1", {"id": "job1", "name": "probe", "deliver": "discord:C1"}, "hello")
    finally:
        reset_hermes_home_override(token)

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True), _profile_adapters={"satellite": {}})
    monkeypatch.setattr(gateway_run, "_handoff_watch_scopes", lambda _r: [("satellite", satellite_home)])
    try:
        with patch("tools.send_message_tool._send_to_platform", _standalone), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            gateway_run._drain_restart_safe_cron_deliveries({Platform.DISCORD: PrimaryDiscord()}, loop, runner)
    finally:
        loop.call_soon_threadsafe(loop.stop)

    token = set_hermes_home_override(str(satellite_home))
    try:
        row = delivery_queue.get_status("exec-1")
    finally:
        reset_hermes_home_override(token)
    assert row["status"] == "delivered", row
    assert sent == ["C1"] and standalone == []
