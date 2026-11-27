"""A cron route naming a platform that left core (``deliver: <platform>:<chat>``, saved while core shipped
it) keeps delivering once the plugin registers the platform, and names the install when it does not."""

from unittest.mock import patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from hermes_cli.left_core_migration import LEFT_CORE

# Platforms that left core with no static ``Platform`` member: only the plugin can register them.
_MOVED = [f for f in LEFT_CORE if f.platform and f.platform not in Platform._value2member_map_]


@pytest.fixture(autouse=True)
def _no_discovery(monkeypatch):
    # The test home has no plugins; discovery must not register (or drop) the stand-in entry below.
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)


def _config(name):
    cfg = type("Cfg", (), {})()
    cfg.platforms = {Platform(name): PlatformConfig(enabled=True)}
    cfg.get_home_channel = lambda _platform: None
    return cfg


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_delivers_through_the_plugins_sender(feature):
    sent = []

    async def sender(pconfig, chat_id, message, **_kw):
        sent.append((chat_id, message))
        return {"success": True, "platform": feature.platform, "chat_id": chat_id, "message_id": "m1"}

    platform_registry.register(PlatformEntry(
        name=feature.platform, label=feature.label, adapter_factory=lambda cfg: None, check_fn=lambda: True,
        standalone_sender_fn=sender, cron_deliver_env_var=f"{feature.platform.upper()}_HOME_CHANNEL"))
    try:
        with patch("gateway.config.load_gateway_config", return_value=_config(feature.platform)):
            error = _deliver_result({"id": "j1", "name": "Report", "deliver": f"{feature.platform}:chat-1"},
                                    "the report", adapters=None, loop=None)
    finally:
        platform_registry.unregister(feature.platform)
    assert error is None
    assert [chat for chat, _ in sent] == ["chat-1"] and "the report" in sent[0][1]


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_without_the_plugin_names_the_install(feature):
    assert platform_registry.get(feature.platform) is None
    error = _deliver_result({"id": "j1", "name": "Report", "deliver": f"{feature.platform}:chat-1"},
                            "the report", adapters=None, loop=None)
    assert error and f"hermes plugins install {feature.plugin}" in error


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_webhook_route_stays_loaded_before_the_plugin_registers(feature):
    """Route load checks the launch registry; a secondary profile's plugin is not in it, and a skipped route is
    not re-read until its file changes."""
    from gateway.webhook_targets import is_dispatchable_target
    assert platform_registry.get(feature.platform) is None
    assert is_dispatchable_target(feature.platform) is True


@pytest.mark.parametrize("route", ["{p}:chat-1", "{p}"], ids=["chat", "home"])
@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_beside_a_delivered_one_still_names_the_install(feature, route):
    """One destination resolving must not hide the moved one it skipped: the run reports the install."""
    sent = []

    async def sender(pconfig, chat_id, message, **_kw):
        sent.append(chat_id)
        return {"success": True, "platform": "standin", "chat_id": chat_id, "message_id": "m1"}

    platform_registry.register(PlatformEntry(
        name="standin", label="Stand-in", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        standalone_sender_fn=sender, cron_deliver_env_var="STANDIN_HOME_CHANNEL"))
    try:
        with patch("gateway.config.load_gateway_config", return_value=_config("standin")):
            error = _deliver_result(
                {"id": "j1", "name": "Report", "deliver": "standin:chat-0," + route.format(p=feature.platform)},
                "the report", adapters=None, loop=None)
    finally:
        platform_registry.unregister("standin")
    assert sent == ["chat-0"]
    assert error and f"hermes plugins install {feature.plugin}" in error


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_origin_on_a_moved_platform_names_the_install(feature):
    error = _deliver_result(
        {"id": "j1", "name": "Report", "deliver": "origin",
         "origin": {"platform": feature.platform, "chat_id": "chat-1"}},
        "the report", adapters=None, loop=None)
    assert error and f"hermes plugins install {feature.plugin}" in error


def _satellite(tmp_path, monkeypatch, platform):
    """A credentialless ``fitness`` profile the primary routes ``<platform>:chat-1`` to; returns its home."""
    import hermes_yaml as yaml
    root = tmp_path / "root"
    home = root / "profiles" / "fitness"
    home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {"multiplex_profiles": True, "profile_routes": [
        {"platform": platform, "chat_id": "chat-1", "profile": "fitness"}]}}), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    return home


@pytest.mark.parametrize("deliver,reached", [("origin", True), ("{p}:chat-1", True), ("{p}:chat-2", False)],
                         ids=["origin", "explicit", "unrouted"])
@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_satellite_route_rides_the_primarys_bot_without_the_plugin(tmp_path, monkeypatch, feature, deliver, reached):
    """A secondary profile without the plugin whose chat the primary's bot serves delivers through that bot:
    the install hint is for a destination nothing can reach, not one a shared route reaches. A chat the
    primary does not route here stays closed and names the install."""
    from unittest.mock import MagicMock
    from cron.scheduler_preflight import _preflight_check_delivery, shared_route_adapters_for_current_home
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

    sent = []

    async def send(chat_id, content, metadata=None):
        sent.append(chat_id)
        return {"success": True, "message_id": "m1"}

    primary = MagicMock()
    primary.send = send
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_now(coro, _loop):
        import asyncio
        from concurrent.futures import Future
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    job = {"id": "j1", "name": "Report", "deliver": deliver.format(p=feature.platform),
           "origin": {"platform": feature.platform, "chat_id": "chat-1"}}
    home = _satellite(tmp_path, monkeypatch, feature.platform)
    # The primary's plugin registers the platform in the primary's scope only; its adapter map keys it.
    primary_scope = hermes_home_key(home.parent.parent)
    platform_registry.register(PlatformEntry(
        name=feature.platform, label=feature.label, adapter_factory=lambda cfg: None, check_fn=lambda: True),
        scope=primary_scope)
    token = set_hermes_home_override(str(home.parent.parent))
    primary_adapters = {Platform(feature.platform): primary}
    reset_hermes_home_override(token)
    token = set_hermes_home_override(str(home))
    try:
        assert platform_registry.get(feature.platform) is None
        assert (_preflight_check_delivery(job) is None) is reached
        adapters = shared_route_adapters_for_current_home(primary_adapters)
        with patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=run_now):
            error = _deliver_result(job, "the report", adapters=adapters, loop=loop)
    finally:
        reset_hermes_home_override(token)
        platform_registry.unregister(feature.platform, scope=primary_scope)
    if reached:
        assert error is None, error
        assert sent == ["chat-1"]
    else:
        assert error and f"plugins install {feature.plugin}" in error
        assert sent == []


@pytest.mark.parametrize("deliver", ["{p}", "{p}:chat-2"], ids=["bare", "unrouted"])
@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_satellite_preflight_blocks_a_target_no_route_serves(tmp_path, monkeypatch, feature, deliver):
    """The primary routing the platform here is not enough: a bare platform with no saved origin or home, or a
    chat no route maps here, reaches nothing, so preflight blocks it with the install instead of running."""
    from cron.scheduler_preflight import _preflight_check_delivery
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(_satellite(tmp_path, monkeypatch, feature.platform)))
    try:
        error = _preflight_check_delivery({"id": "j1", "name": "Report", "deliver": deliver.format(p=feature.platform)})
    finally:
        reset_hermes_home_override(token)
    assert error and f"plugins install {feature.plugin}" in error
