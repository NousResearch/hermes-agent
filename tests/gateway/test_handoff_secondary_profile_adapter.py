"""A secondary profile's handoff must deliver through ITS OWN adapter/config.

Regression guard for the third `/handoff` multi-profile bug. Even after the
watcher polls the right ``state.db`` and the session key carries the profile
namespace, ``_process_handoff`` still resolved delivery from ``self.adapters``
and ``self.config`` — which on a multiplexed gateway hold ONLY the primary
profile's adapters and home channel. A medicina handoff was therefore sent by
the default profile's bot, to the default profile's chat, while persisting a
``agent:medicina:...`` key and reporting ``handoff_state='completed'``: a
false positive that looks fine in the database and is wrong on the wire.

This was caught by an adversarial review reading gateway.log, not by the
end-to-end test — the log line showed ``hermes_plugins.telegram_platform``
(primary) instead of the secondary's ``..._home_<hash>`` adapter module.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_yaml as yaml
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner
from gateway.session import SessionEntry
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _adapter(tag):
    """A platform adapter stand-in that records which one was used."""
    a = MagicMock()
    a.tag = tag
    a.send = AsyncMock(return_value=SimpleNamespace(success=True))
    a.create_handoff_thread = AsyncMock(return_value=None)
    a._bot = None
    return a


def _config(chat_id):
    cfg = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    cfg.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=chat_id, name=f"home-{chat_id}",
    )
    return cfg


def _make_multiplex_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = _config("1111")          # primary/default home
    runner.config.multiplex_profiles = True
    runner.adapters = {Platform.TELEGRAM: _adapter("primary")}
    runner._profile_adapters = {
        "medicina": {Platform.TELEGRAM: _adapter("medicina")},
    }
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False,
    )

    captured = {}

    store = MagicMock()
    store.get_or_create_session = AsyncMock(return_value=SessionEntry(
        session_key="k", session_id="s",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    ))

    async def _switch(key, sid, *, preserve_prompt_pin=True):
        captured["session_key"] = key
        return SessionEntry(
            session_key=key, session_id=sid,
            created_at=datetime.now(), updated_at=datetime.now(),
            platform=Platform.TELEGRAM, chat_type="dm",
        )

    store.switch_session = AsyncMock(side_effect=_switch)
    # ``async_session_store`` is a derived property with no setter: it rebuilds
    # the facade whenever ``facade._store is not self.session_store``. Wiring
    # the mock as that ``_store`` is what makes the primed cache survive.
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=store.get_or_create_session,
        switch_session=store.switch_session,
    )
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._session_db = None

    async def _handle_message(event):
        captured["source"] = event.source
        return "ok"

    runner._handle_message = AsyncMock(side_effect=_handle_message)
    return runner, captured


def _spy_transport_factory(used):
    """Build a resolve_delivery_transport stand-in that records what it got.

    The real transport exposes ``.adapter`` AND an awaitable ``.send``; the
    handoff uses both, so the stand-in must too.
    """
    def _spy(platform, config, adapters):
        adapter = adapters[platform]
        used["adapter_tag"] = adapter.tag
        used["home_chat_id"] = config.get_home_channel(platform).chat_id

        async def _send(_platform, _chat_id, _text, _metadata=None):
            used["sent_via"] = adapter.tag
            return SimpleNamespace(success=True)

        return SimpleNamespace(adapter=adapter, send=_send)

    return _spy


@pytest.mark.asyncio
async def test_secondary_profile_handoff_uses_its_own_adapter(monkeypatch):
    """medicina's handoff must NOT be delivered by the primary's adapter."""
    runner, captured = _make_multiplex_runner()

    used = {}
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", _spy_transport_factory(used),
    )
    # The watcher would already be inside _profile_runtime_scope here, so a
    # fresh load resolves the secondary's config.
    monkeypatch.setattr("gateway.run.load_gateway_config", lambda: _config("2222"))

    await runner._process_handoff(
        {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
        profile_name="medicina",
    )

    assert used["adapter_tag"] == "medicina", (
        "delivery must use the secondary profile's own adapter, not the primary's"
    )
    assert used["sent_via"] == "medicina", "the message went out on the wrong bot"
    assert used["home_chat_id"] == "2222", (
        "delivery must use the secondary profile's own home channel"
    )
    assert captured["session_key"].startswith("agent:medicina:"), (
        f"session key must carry the profile namespace, got {captured['session_key']}"
    )
    assert captured["source"].profile == "medicina"


@pytest.mark.asyncio
async def test_default_profile_handoff_keeps_primary_adapter(monkeypatch):
    """The default/root path must behave exactly as before the fix."""
    runner, _captured = _make_multiplex_runner()

    used = {}
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", _spy_transport_factory(used),
    )

    await runner._process_handoff(
        {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
        profile_name=None,
    )

    assert used["adapter_tag"] == "primary"
    assert used["home_chat_id"] == "1111"


@pytest.mark.asyncio
async def test_secondary_profile_config_load_failure_fails_closed(monkeypatch):
    """A secondary profile whose config cannot load must fail the handoff.

    Falling back to the primary's config delivers through the right bot to
    the WRONG chat (the primary's home channel) and reports completed.
    """
    runner, _ = _make_multiplex_runner()
    used = {}

    def _boom():
        raise RuntimeError("config.yaml exploded")

    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", _spy_transport_factory(used),
    )
    monkeypatch.setattr("gateway.run.load_gateway_config", _boom)

    with pytest.raises(RuntimeError, match="could not load config"):
        await runner._process_handoff(
            {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
            profile_name="medicina",
        )
    assert used == {}, (
        "nothing may be delivered when the profile config fails to load"
    )


@pytest.mark.asyncio
async def test_secondary_profile_without_live_adapters_fails_loudly(monkeypatch):
    """Never silently fall back to the primary's bot — that ships to the wrong chat.

    Raising marks the row ``failed`` and the CLI reports it; delivering through
    another profile's bot would look like success.
    """
    runner, _ = _make_multiplex_runner()
    runner._profile_adapters = {}

    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", _spy_transport_factory({}),
    )

    with pytest.raises(RuntimeError, match="no live adapters"):
        await runner._process_handoff(
            {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
            profile_name="medicina",
        )


@pytest.mark.asyncio
async def test_shared_bot_satellite_handoff_drains_through_primary(monkeypatch):
    """A routed profile with NO own bot credential (token removed at multiplex migration)
    must hand off through the default bot, not fail with 'no live adapters'.

    Regression for the post-migration ``/handoff telegram`` breakage: butler is a shared-bot
    satellite — its ``_profile_adapters`` entry is the empty ``{}`` placeholder and a
    ``profile_routes`` entry targets it through the default bot. The raw ``_profile_adapters``
    lookup saw ``{}`` and raised, so the CLI always timed out. The canonical
    ``_adapters_for_profile`` resolver drains it through the primary's adapters.
    """
    from gateway.profile_routing import ProfileRoute

    runner, captured = _make_multiplex_runner()
    # Satellite: served, routed through the default bot, owns NO adapter of its own.
    runner._profile_adapters = {"butler": {}}
    runner._profile_failed_platforms = {}
    runner.config.profile_routes = [
        ProfileRoute(name="butler-dm", platform="telegram", profile="butler",
                     chat_id="6719571041", bot_profile=None),
    ]
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes", lambda cfg: [("butler", None)],
    )

    used = {}
    monkeypatch.setattr(
        "gateway.delivery.resolve_delivery_transport", _spy_transport_factory(used),
    )
    monkeypatch.setattr("gateway.run.load_gateway_config", lambda: _config("6719571041"))

    await runner._process_handoff(
        {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
        profile_name="butler",
    )

    assert used["adapter_tag"] == "primary", (
        "a shared-bot satellite must deliver through the default bot, not fail"
    )
    assert used["sent_via"] == "primary"
    assert captured["session_key"].startswith("agent:butler:"), (
        "the key must still carry the satellite's namespace"
    )


# ---------------------------------------------------------------------------
# The CONFIG half of a satellite handoff. The adapter-only regression above
# patched BOTH the config loader and the transport resolver, so a satellite whose
# own platform block is missing / not-enabled / disabled could not be observed to
# fail — and a satellite's config routinely has no usable block once its
# credential is removed. These keep the REAL loader and the REAL resolver.
# ---------------------------------------------------------------------------

_SATELLITE_CONFIG_SHAPES = {
    # a satellite that kept an enabled block (plain token removal)
    "block_enabled": {"telegram": {"enabled": True}},
    # no platform block at all
    "no_block": {},
    # block present, no explicit ``enabled`` (the platform default is off)
    "enabled_unset": {"telegram": {"reactions": False}},
    # block explicitly disabled
    "enabled_false": {"telegram": {"enabled": False}},
}


def _satellite_runner(monkeypatch):
    """A multiplex runner whose ``butler`` is a shared-bot satellite (no credential of its own)."""
    runner, captured = _make_multiplex_runner()
    runner.config.profile_routes = [
        ProfileRoute(name="butler-dm", platform="telegram", profile="butler",
                     chat_id="6719571041", bot_profile=None),
        # A catch-all that names no destination: it must never be taken as the home.
        ProfileRoute(name="telegram-fallback", platform="telegram", profile="butler"),
    ]
    runner._profile_adapters = {"butler": {}}   # the startup placeholder
    runner._profile_failed_platforms = {}
    monkeypatch.setattr("gateway.run._multiplex_profile_homes", lambda cfg: [("butler", None)])
    return runner, captured


def _write_satellite_home(tmp_path, body):
    home = tmp_path / "satellite"
    home.mkdir()
    config = {"gateway": {"multiplex_profiles": True}}
    config.update(body)
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return home


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(_SATELLITE_CONFIG_SHAPES))
async def test_satellite_handoff_resolves_route_target_across_config_shapes(
    monkeypatch, tmp_path, shape,
):
    """A satellite's handoff must not depend on its own platform block.

    Its transport is the primary's adapter (read from the primary's config, so an absent, unset or
    disabled satellite block cannot veto the delivery), and its destination is the
    ``profile_routes`` target — never the primary's home, never a destination-less catch-all.
    """
    sat_home = _write_satellite_home(tmp_path, _SATELLITE_CONFIG_SHAPES[shape])

    token = set_hermes_home_override(str(sat_home))
    try:
        runner, captured = _satellite_runner(monkeypatch)
        primary = runner.adapters[Platform.TELEGRAM]

        await runner._process_handoff(
            {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
            profile_name="butler",
        )

        primary.create_handoff_thread.assert_awaited_once()
        assert primary.create_handoff_thread.await_args.args[0] == "6719571041", (
            "the satellite must hand off to its ROUTE target, not the primary's home (1111)"
        )
        assert captured["session_key"].startswith("agent:butler:"), (
            "the key must still carry the satellite's namespace"
        )
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_satellite_without_a_concrete_route_fails_closed(monkeypatch, tmp_path):
    """A satellite with only a destination-less route must fail — even if it has a configured home.

    The satellite's own home channel is not a delivery grant: delivering there when no route
    authorises a concrete chat sends the CLI history to a stale/unauthorised target (the
    retained-home case).
    """
    sat_home = _write_satellite_home(tmp_path, {"telegram": {"enabled": True}})
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "2222")   # a configured home, unauthorised

    token = set_hermes_home_override(str(sat_home))
    try:
        runner, _ = _satellite_runner(monkeypatch)
        runner.config.profile_routes = [
            ProfileRoute(name="telegram-fallback", platform="telegram", profile="butler"),
        ]
        with pytest.raises(RuntimeError, match="no unambiguous route destination"):
            await runner._process_handoff(
                {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
                profile_name="butler",
            )
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_satellite_user_only_route_is_not_a_destination(monkeypatch, tmp_path):
    """``user_id`` is the inbound SENDER, not a chat — a user-only route grants no destination."""
    sat_home = _write_satellite_home(tmp_path, {"telegram": {"enabled": True}})

    token = set_hermes_home_override(str(sat_home))
    try:
        runner, _ = _satellite_runner(monkeypatch)
        runner.config.profile_routes = [
            ProfileRoute(name="telegram-owner", platform="telegram", profile="butler",
                         user_id="6719571041", bot_profile=None),
        ]
        with pytest.raises(RuntimeError, match="no unambiguous route destination"):
            await runner._process_handoff(
                {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
                profile_name="butler",
            )
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_satellite_disabled_route_is_not_a_destination(monkeypatch, tmp_path):
    """A disabled route grants nothing: with the sole concrete chat disabled the handoff fails.

    (The profile stays a satellite because an enabled catch-all still targets it; the disabled
    concrete route must not become the destination.)
    """
    sat_home = _write_satellite_home(tmp_path, {"telegram": {"enabled": True}})

    token = set_hermes_home_override(str(sat_home))
    try:
        runner, _ = _satellite_runner(monkeypatch)
        runner.config.profile_routes = [
            ProfileRoute(name="telegram-fallback", platform="telegram", profile="butler"),
            ProfileRoute(name="butler-dm", platform="telegram", profile="butler",
                         chat_id="6719571041", bot_profile=None, enabled=False),
        ]
        with pytest.raises(RuntimeError, match="no unambiguous route destination"):
            await runner._process_handoff(
                {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
                profile_name="butler",
            )
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_secondary_profile_keeps_its_own_home_with_real_resolver(monkeypatch, tmp_path):
    """A secondary that OWNS a credential keeps its own adapter and its own home.

    The satellite branch must not leak into a credential-owning secondary: this asserts the
    un-satellited path with the real resolver and a real (non-empty) platform block.
    """
    own_home = _write_satellite_home(tmp_path, {"telegram": {"enabled": True}})
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "2222")

    token = set_hermes_home_override(str(own_home))
    try:
        runner, captured = _make_multiplex_runner()
        # medicina owns an adapter, so it is NOT a satellite (no route for it).
        own = runner._profile_adapters["medicina"][Platform.TELEGRAM]

        await runner._process_handoff(
            {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
            profile_name="medicina",
        )

        own.create_handoff_thread.assert_awaited_once()
        assert own.create_handoff_thread.await_args.args[0] == "2222", (
            "an owning secondary must deliver to its own home, not the primary's"
        )
        assert captured["session_key"].startswith("agent:medicina:")
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_satellite_with_several_concrete_routes_fails_closed(monkeypatch, tmp_path):
    """Several concrete routes for the same profile+platform are ambiguous: fail, don't guess.

    Routes select a profile for inbound messages; they do not define a home, so when more than one
    names a chat the destination would depend on route order — the handoff must refuse instead.
    """
    sat_home = _write_satellite_home(tmp_path, {"telegram": {"enabled": True}})

    token = set_hermes_home_override(str(sat_home))
    try:
        runner, _ = _satellite_runner(monkeypatch)
        runner.config.profile_routes = [
            ProfileRoute(name="butler-guild", platform="telegram", profile="butler",
                         guild_id="g123", chat_id="999", bot_profile=None),
            ProfileRoute(name="butler-dm", platform="telegram", profile="butler",
                         chat_id="6719571041", bot_profile=None),
        ]
        with pytest.raises(RuntimeError, match="no unambiguous route destination"):
            await runner._process_handoff(
                {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
                profile_name="butler",
            )
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_satellite_route_picker_uses_the_sole_concrete_route(monkeypatch, tmp_path):
    """A guild-scoped route is the destination when it is the only concrete one (e.g. a Discord
    channel route, which always carries a guild)."""
    sat_home = _write_satellite_home(tmp_path, {"telegram": {"enabled": True}})

    token = set_hermes_home_override(str(sat_home))
    try:
        runner, _ = _satellite_runner(monkeypatch)
        runner.config.profile_routes = [
            ProfileRoute(name="butler-guild", platform="telegram", profile="butler",
                         guild_id="g123", chat_id="999", bot_profile=None),
        ]
        runner._profile_failed_platforms = {}
        primary = runner.adapters[Platform.TELEGRAM]

        await runner._process_handoff(
            {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
            profile_name="butler",
        )

        assert primary.create_handoff_thread.await_args.args[0] == "999", (
            "a guild-scoped route is the destination when it is the only one"
        )
    finally:
        reset_hermes_home_override(token)
