"""Multiplexed-profile authorization-policy propagation for typed commands.

Regression tests for the finding that normal typed slash commands on a
MULTIPLEXED SECONDARY profile were gated by ``self.config`` — the
multiplexer's OWN (primary) policy — while native picker callbacks correctly
captured the effective named-profile GatewayConfig.

Every test here drives the REAL ``_handle_message`` path through the
secondary adapter's production message handler (built by
``_configure_profile_adapter`` → ``_make_profile_message_handler``), so the
profile stamping, runtime scoping, and gate placement are the shipped code,
not a re-implementation.

The two profiles are configured with INTENTIONALLY DIVERGENT policies so any
wrong-profile fallback flips the result:

    primary ("default"): admins={"111"}, user_allowed={"status"}
    secondary ("work"):  admins={"222"}, user_allowed={}   (+ status in group)

Consequences the old bug got wrong:
  - 111 (primary admin) must be DENIED admin commands on the secondary bot
    (privilege-sensitive negative case).
  - 222 (secondary-only admin) must be ALLOWED them there.
"""

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource

PRIMARY_ADMIN = "111"
SECONDARY_ADMIN = "222"
PROFILE_NAME = "work"

PRIMARY_EXTRA = {
    "allow_admin_from": [PRIMARY_ADMIN],
    "group_allow_admin_from": [PRIMARY_ADMIN],
    "user_allowed_commands": ["status"],
}
SECONDARY_EXTRA = {
    "allow_admin_from": [SECONDARY_ADMIN],
    "group_allow_admin_from": [SECONDARY_ADMIN],
    "user_allowed_commands": [],
}

# Same quick-command name defined in BOTH profiles with different payloads:
# whichever profile's dict the lookup resolves proves itself in the output.
QC_NAME = "secrets"
PRIMARY_QC = {QC_NAME: {"type": "exec", "command": "printf primary-secret-payload"}}
SECONDARY_QC = {QC_NAME: {"type": "exec", "command": "printf secondary-secret-payload"}}


def _write_profile_scaffold(profile_home) -> None:
    """A minimal on-disk profile so get_profile_dir()/runtime scope resolve."""
    profile_home.mkdir(parents=True, exist_ok=True)
    (profile_home / ".env").write_text("", encoding="utf-8")


@pytest.fixture
def work_profile_home():
    """Create the secondary profile dir under the isolated HERMES_HOME root."""
    from hermes_cli.profiles import get_profile_dir

    home = get_profile_dir(PROFILE_NAME)
    _write_profile_scaffold(home)
    return home


def _make_multiplex_runner(
    *,
    primary_quick_commands=None,
    secondary_quick_commands=None,
):
    """A GatewayRunner-shaped object with divergent primary/second. policies."""
    runner = object.__new__(GatewayRunner)

    def _cfg(extra, quick_commands):
        cfg = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True, token="t", extra=dict(extra)
                )
            },
        )
        if quick_commands is not None:
            cfg.quick_commands = dict(quick_commands)
        return cfg

    runner.config = _cfg(PRIMARY_EXTRA, primary_quick_commands)
    secondary_config = _cfg(SECONDARY_EXTRA, secondary_quick_commands)
    runner.adapters = {Platform.TELEGRAM: MagicMock(send=AsyncMock())}
    runner._profile_gateway_configs = {}
    runner._profile_adapters = {}
    runner._busy_text_modes_by_profile = {}
    runner._busy_input_modes_by_profile = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    session_entry = SimpleNamespace(session_id="sess-1", session_key="k")
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.list_sessions_rich = AsyncMock(return_value=[])
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()

    # Wire the secondary adapter through the REAL production configuration
    # path and keep its installed message handler.
    secondary_adapter = MagicMock(send=AsyncMock())
    runner._configure_profile_adapter(
        secondary_adapter,
        PROFILE_NAME,
        Platform.TELEGRAM,
        gateway_config=secondary_config,
    )
    runner._secondary_adapter = secondary_adapter
    runner._secondary_config = secondary_config
    return runner


def _secondary_message_handler(runner):
    return runner._secondary_adapter.set_message_handler.call_args.args[0]


def _dm_source(user_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=f"chat-{user_id}",
        chat_type="dm",
        user_id=user_id,
        user_name=f"name-{user_id}",
    )


def _dm_event(text: str, user_id: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_dm_source(user_id),
        user_id=user_id,
        user_name=f"name-{user_id}",
        message_id="m1",
    )


# ---------------------------------------------------------------------------
# Registry + resolver contract
# ---------------------------------------------------------------------------


def test_configure_registers_effective_named_profile_config(work_profile_home):
    runner = _make_multiplex_runner()
    assert runner._profile_gateway_configs[PROFILE_NAME] is (
        runner._secondary_config
    )
    resolved, ok = runner._effective_gateway_config_for_source(
        SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="c",
            chat_type="dm",
            profile=PROFILE_NAME,
        )
    )
    assert ok is True
    assert resolved is runner._secondary_config


def test_resolver_fails_closed_for_unknown_named_profile(work_profile_home):
    runner = _make_multiplex_runner()
    resolved, ok = runner._effective_gateway_config_for_source(
        SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="c",
            chat_type="dm",
            profile="ghost",
        )
    )
    assert ok is False
    assert resolved is None


def test_resolver_unstamped_source_uses_primary_config(work_profile_home):
    runner = _make_multiplex_runner()
    resolved, ok = runner._effective_gateway_config_for_source(_dm_source("u"))
    assert ok is True
    assert resolved is runner.config


# ---------------------------------------------------------------------------
# Typed commands through the secondary adapter's REAL message handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_path_gates_secondary_by_its_own_policy(work_profile_home):
    """Cold-session typed command: the secondary-only admin may run an admin
    command on the secondary bot while the PRIMARY admin may not — proving
    neither grant nor deny is decided by the wrong profile."""
    runner = _make_multiplex_runner()
    runner._handle_stop_command = AsyncMock(return_value="stopped")
    handler = _secondary_message_handler(runner)

    granted = await handler(_dm_event("/stop", SECONDARY_ADMIN))
    assert granted == "stopped"
    runner._handle_stop_command.assert_awaited_once()

    denied = await handler(_dm_event("/stop", PRIMARY_ADMIN))
    assert denied is not None
    assert "⛔" in denied
    assert "/stop is admin-only here" in denied
    # Still exactly one dispatch attempt — the denial happened at the gate.
    runner._handle_stop_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_busy_fastpath_gate_uses_secondary_policy(work_profile_home):
    """Running-agent fast path: same divergence must hold while an agent is
    busy — the primary admin gets no bypass on the secondary bot."""
    runner = _make_multiplex_runner()
    handler = _secondary_message_handler(runner)
    runner._dispatch_busy_slash_command = AsyncMock(return_value="busy-ok")

    busy_src = _dm_source(SECONDARY_ADMIN)
    busy_src.profile = PROFILE_NAME
    busy_key = runner._session_key_for_source(busy_src)
    runner._running_agents[busy_key] = MagicMock()
    runner._running_agents_ts[busy_key] = 0

    granted = await handler(_dm_event("/restart", SECONDARY_ADMIN))
    assert granted == "busy-ok"

    denied = await handler(_dm_event("/restart", PRIMARY_ADMIN))
    assert denied is not None
    assert "⛔" in denied
    assert "/restart is admin-only here" in denied


@pytest.mark.asyncio
async def test_whoami_reports_the_serving_profile_tier(work_profile_home):
    """/whoami must answer with the effective profile's tier lists."""
    runner = _make_multiplex_runner()
    handler = _secondary_message_handler(runner)

    secondary_admin_view = await handler(_dm_event("/whoami", SECONDARY_ADMIN))
    assert "Tier: **admin**" in secondary_admin_view

    primary_admin_view = await handler(_dm_event("/whoami", PRIMARY_ADMIN))
    assert "Tier: user" in primary_admin_view
    assert "**admin**" not in primary_admin_view


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["help", "commands"])
@pytest.mark.parametrize("actor", [PRIMARY_ADMIN, SECONDARY_ADMIN, None])
@pytest.mark.parametrize("policy_storage", ["cached", "disk"])
async def test_shared_catalog_matches_actor_and_serving_policy(
    tmp_path, monkeypatch, command, actor, policy_storage,
):
    """Catalog visibility follows dispatch across A→B→A shared group routes."""
    import json
    import math
    from pathlib import Path

    from gateway.message_actor import source_for_event_actor
    from gateway.run import _profile_runtime_scope
    from hermes_cli.commands import gateway_help_lines
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    work = home / "profiles" / PROFILE_NAME
    for profile_home in (home, work):
        _write_profile_scaffold(profile_home)
    runner = _make_multiplex_runner()
    runner.config.platforms[Platform.TELEGRAM].extra["group_user_allowed_commands"] = ["commands", "model"]
    runner._secondary_config.platforms[Platform.TELEGRAM].extra["group_user_allowed_commands"] = ["commands", "status"]
    (work / "config.yaml").write_text(json.dumps({"platforms": {"telegram": {
        "enabled": True, "extra": runner._secondary_config.platforms[Platform.TELEGRAM].extra,
    }}}, ensure_ascii=False), encoding="utf-8")
    if policy_storage == "disk":
        runner._profile_gateway_configs.clear()

    for profile, profile_home in [(None, home), (PROFILE_NAME, work), (None, home)]:
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100",
                               chat_type="group", profile=profile, user_id=None)
        event = MessageEvent(text=f"/{command}", source=source, user_id=actor)
        with _profile_runtime_scope(profile_home):
            result = await runner._handle_message(event)
            if actor is None and command == "commands":
                assert "requires an identifiable user" in result
            # Exercise the real renderer even when the outer identity gate denies /commands.
            handler = getattr(runner, f"_handle_{command}_command")
            pages = math.ceil(len(gateway_help_lines()) / 15) if command == "commands" else 1
            rendered = []
            for page in range(1, pages + 1):
                event.text = f"/{command} {page}" if command == "commands" else "/help"
                rendered.append(await handler(event))
            catalog = "\n".join(rendered)
            for name in ("help", "whoami", "model", "status", "restart"):
                allowed = runner._check_slash_access(source_for_event_actor(event), name) is None
                assert (f"`/{name}" in catalog) is allowed, (profile, actor, name)
            if actor is not None or command == "help":
                assert result == rendered[0]
        assert source.user_id is None
        assert Path(get_hermes_home()) == home

    # An unresolved profile must not borrow primary policy, even at the renderer boundary.
    event.source.profile = "ghost"
    event.text = "/help"
    assert "policy context" in await runner._handle_message(event)
    assert "`/model" not in await handler(event)
    # Bare standalone runners retain disabled-policy semantics, but still need an actor.
    del runner.config
    for name in ("model", "status"):
        event.text = "/help"
        catalog = await runner._handle_help_command(event)
        allowed = runner._check_slash_access(source_for_event_actor(event), name) is None
        assert (f"`/{name}" in catalog) is allowed


@pytest.mark.asyncio
async def test_quick_command_exec_resolves_named_profile_dict(
    work_profile_home,
):
    """Privilege-sensitive negative case: the SAME quick-command name is
    defined in both profiles. The secondary bot must execute the SECONDARY
    payload only, and the primary's admin standing must not authorize it."""
    runner = _make_multiplex_runner(
        primary_quick_commands=PRIMARY_QC,
        secondary_quick_commands=SECONDARY_QC,
    )
    handler = _secondary_message_handler(runner)

    executed = await handler(_dm_event(f"/{QC_NAME}", SECONDARY_ADMIN))
    assert executed == "secondary-secret-payload"
    assert "primary-secret-payload" not in executed

    denied = await handler(_dm_event(f"/{QC_NAME}", PRIMARY_ADMIN))
    assert denied is not None
    assert "⛔" in denied
    assert "primary-secret-payload" not in (denied or "")
    assert "secondary-secret-payload" not in (denied or "")


@pytest.mark.asyncio
async def test_approvals_mutation_needs_serving_profile_admin(
    work_profile_home, monkeypatch
):
    """/approvals off mutates profile-wide security posture. With the
    command deliberately listed in the SECONDARY's user_allowed_commands,
    a non-admin passes the central gate but must still be stopped at the
    handler's admin boundary — judged by the SECONDARY list, not the
    multiplexer's (where 111 IS admin and the old bug let them through)."""
    runner = _make_multiplex_runner()
    runner._secondary_config.platforms[Platform.TELEGRAM].extra[
        "user_allowed_commands"
    ] = ["approvals"]
    handler = _secondary_message_handler(runner)
    run_mode = MagicMock(return_value=SimpleNamespace(message="approval-mode-set"))
    monkeypatch.setattr(
        "hermes_cli.approval_mode.run_approval_mode_command", run_mode
    )

    denied = await handler(_dm_event("/approvals off", PRIMARY_ADMIN))
    assert "Only gateway admins" in denied
    assert run_mode.call_count == 0

    allowed = await handler(_dm_event("/approvals off", SECONDARY_ADMIN))
    assert allowed == "approval-mode-set"
    assert run_mode.call_count == 1
    assert run_mode.call_args.args == ("off",)


@pytest.mark.asyncio
async def test_resume_cross_origin_widening_requires_serving_profile_admin(
    work_profile_home,
):
    """/resume --all widens the listing across origins — an explicitly
    configured admin OF THE SERVING PROFILE, nobody else. Driven through the
    real ``_handle_resume_command`` boundary (the central gate would already
    refuse /resume to a non-admin entirely; defense in depth means the
    cross-origin check inside the handler must not rely on that)."""
    runner = _make_multiplex_runner()

    stamped = _dm_source(PRIMARY_ADMIN)
    stamped.profile = PROFILE_NAME
    assert (
        runner._resume_caller_is_admin(stamped, PRIMARY_ADMIN) is False
    ), "primary-admin standing must not confer cross-origin access on secondary"
    stamped_admin = _dm_source(SECONDARY_ADMIN)
    stamped_admin.profile = PROFILE_NAME
    assert runner._resume_caller_is_admin(stamped_admin, SECONDARY_ADMIN) is True

    def _event(user_id):
        ev = _dm_event("/resume --all", user_id)
        ev.source.profile = PROFILE_NAME
        return ev

    await runner._handle_resume_command(_event(PRIMARY_ADMIN))
    primary_call = runner._session_db.list_sessions_rich.await_args
    assert primary_call.kwargs["session_key"] is not None, (
        "non-admin --all must stay scoped to their own session key"
    )

    await runner._handle_resume_command(_event(SECONDARY_ADMIN))
    admin_call = runner._session_db.list_sessions_rich.await_args
    assert admin_call.kwargs["session_key"] is None, (
        "serving-profile admin --all widens the listing"
    )


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unresolvable_named_profile_denies_instead_of_primary_fallback(
    work_profile_home,
):
    """A stamped profile whose policy context cannot be resolved must fail
    closed (even for floor commands) rather than silently judge by the
    multiplexer's own policy."""
    runner = _make_multiplex_runner()
    for cmd in ("/help", "/status", "/stop"):
        src = _dm_source(SECONDARY_ADMIN)
        src.profile = "ghost"
        result = await runner._handle_message(
            MessageEvent(text=cmd, source=src, user_id=SECONDARY_ADMIN, message_id="m2")
        )
        assert result is not None
        assert "unavailable without profile policy context" in result


@pytest.mark.asyncio
async def test_approvals_mutation_fails_closed_without_profile_policy(
    work_profile_home, monkeypatch
):
    """The handler-level mutation guard must not interpret an unresolved
    policy as the legacy disabled policy, whose is_admin() intentionally
    returns True for backward compatibility."""
    runner = _make_multiplex_runner()
    event = _dm_event("/approvals off", SECONDARY_ADMIN)
    event.source.profile = "ghost"
    run_mode = MagicMock(return_value=SimpleNamespace(message="mutated"))
    monkeypatch.setattr(
        "hermes_cli.approval_mode.run_approval_mode_command", run_mode
    )

    result = await runner._handle_approvals_command(event)

    assert "Only gateway admins" in result
    run_mode.assert_not_called()


# ---------------------------------------------------------------------------
# Backward-compat controls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiplex_off_stamped_profile_still_uses_process_config():
    """With multiplexing OFF, a stamped profile is ignored (documented
    semantics): gating stays byte-identical single-profile behavior."""
    runner = _make_multiplex_runner()
    runner.config.multiplex_profiles = False
    runner._handle_stop_command = AsyncMock(return_value="stopped")
    src = _dm_source(PRIMARY_ADMIN)
    src.profile = PROFILE_NAME
    result = await runner._handle_message(MessageEvent(text="/stop", source=src))
    assert result == "stopped"


@pytest.mark.asyncio
async def test_default_lane_under_multiplex_still_gated_by_primary(work_profile_home):
    """Unstamped (primary-lane) ingress under multiplex keeps judging by the
    multiplexer's own config — which IS that lane's effective policy."""
    runner = _make_multiplex_runner()
    runner._handle_stop_command = AsyncMock(return_value="stopped")

    granted = await runner._handle_message(
        MessageEvent(text="/stop", source=_dm_source(PRIMARY_ADMIN))
    )
    assert granted == "stopped"

    denied = await runner._handle_message(
        MessageEvent(text="/stop", source=_dm_source(SECONDARY_ADMIN))
    )
    assert denied is not None
    assert "⛔" in denied


@pytest.mark.asyncio
async def test_observed_dispatch_preserves_transport_and_shared_profile_sessions(tmp_path, monkeypatch):
    """Observed A→B→A commands retain provenance, actor policy and durable shared lanes."""
    from pathlib import Path
    import hermes_state
    from gateway.platforms.event import MessageType
    from gateway.session import SessionStore
    from gateway.session_identity import identity_of
    from hermes_constants import get_hermes_home
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    _write_profile_scaffold(home)
    _write_profile_scaffold(home / "profiles" / PROFILE_NAME)
    (home / "profiles" / PROFILE_NAME / "config.yaml").write_text("{}\n")
    runner = _make_multiplex_runner()
    runner._primary_profile_name = "default"
    runner.session_store = SessionStore(sessions_dir=home / "sessions", config=runner.config)
    adapters = {}
    handlers = {}
    for profile, cfg in [("default", runner.config), (PROFILE_NAME, runner._secondary_config)]:
        adapter = object.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter.config = cfg.platforms[Platform.TELEGRAM]
        adapter.config.extra.update(observe_unmentioned_group_messages=True, group_allowed_chats=["-100"])
        adapter.gateway_runner = runner
        adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
        adapter.set_owner_profile(profile)
        adapter.clear_session_status = AsyncMock()
        adapters[profile] = adapter
        handlers[profile] = (runner._make_default_profile_message_handler() if profile == "default"
                             else runner._make_profile_message_handler(profile))
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {PROFILE_NAME: {Platform.TELEGRAM: adapters[PROFILE_NAME]}}
    raw = SimpleNamespace(chat=SimpleNamespace(id=-100, type="supergroup"))
    observed = []
    for profile, admin in [("default", PRIMARY_ADMIN), (PROFILE_NAME, SECONDARY_ADMIN),
                           ("default", PRIMARY_ADMIN)]:
        adapter = adapters[profile]
        source = adapter.build_source(chat_id="-100", chat_type="group", user_id=admin)
        identity = adapter._canonicalize(source)
        assert identity is not None
        event = adapter._apply_telegram_group_observe_attribution(MessageEvent(
            text="/stop", source=source, user_id=admin, message_type=MessageType.COMMAND,
            raw_message=raw, message_id="command"))
        assert event.source.user_id is None
        assert identity_of(event.source) is identity
        assert runner._intake_adapter_for(event.source) is adapter
        response = await handlers[profile](event)
        assert response is not None and "⛔" not in response
        key = runner._session_key_for_source(event.source)
        entry = runner.session_store.get_or_create_session(event.source)
        observed.append((key, entry.session_id))
        ordinary = adapter._apply_telegram_group_observe_attribution(MessageEvent(
            text="hello", source=source, user_id=admin, raw_message=raw))
        assert runner._session_key_for_source(ordinary.source) == key
        assert runner.session_store.get_or_create_session(ordinary.source).session_id == entry.session_id
        assert runner.hooks.emit_collect.call_args.args[1]["user_id"] == admin
        before = runner.hooks.emit_collect.await_count
        for actor in [SECONDARY_ADMIN if admin == PRIMARY_ADMIN else PRIMARY_ADMIN, None]:
            denied = await handlers[profile](MessageEvent(
                text="/stop", source=event.source, user_id=actor, message_id="denied"))
            assert "⛔" in denied
        assert runner.hooks.emit_collect.await_count == before
        assert Path(get_hermes_home()) == home
    assert observed[0] == observed[2]
    assert observed[0][0] != observed[1][0]
    assert observed[0][1] != observed[1][1]


def test_secondary_picker_policy_follows_canonical_runtime_route(work_profile_home, monkeypatch):
    """A secondary bot routed to default must apply the destination's slash policy."""
    from pathlib import Path
    from gateway.profile_routing import ProfileRoute
    from gateway.session_identity import identity_of
    from hermes_constants import get_hermes_home

    runner = _make_multiplex_runner()
    runner._primary_profile_name = "default"
    default_home = Path(get_hermes_home())
    runner.config.profile_routes = [ProfileRoute(
        name="secondary-to-default", platform="telegram", chat_id="routed",
        bot_profile=PROFILE_NAME, profile="default")]
    monkeypatch.setattr("gateway.run._multiplex_profile_homes", lambda _cfg: [
        ("default", default_home), (PROFILE_NAME, work_profile_home)])
    checker = runner._make_profile_slash_access_check(
        PROFILE_NAME, gateway_config=runner._secondary_config)
    for chat_id, runtime, allowed, denied in [
        ("own", PROFILE_NAME, SECONDARY_ADMIN, PRIMARY_ADMIN),
        ("routed", "default", PRIMARY_ADMIN, SECONDARY_ADMIN),
        ("own", PROFILE_NAME, SECONDARY_ADMIN, PRIMARY_ADMIN),
    ]:
        for actor, expected_allowed in [(allowed, True), (denied, False)]:
            source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id,
                                   chat_type="group", user_id=actor)
            assert (checker(source, "model") is None) is expected_allowed
            if chat_id == "routed":
                assert identity_of(source).runtime_profile == runtime
        assert Path(get_hermes_home()) == default_home
