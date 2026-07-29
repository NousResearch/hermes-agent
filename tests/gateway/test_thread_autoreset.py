"""Per-thread automatic reset policy tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.platforms.base import MessageEvent
from gateway.session import AsyncSessionStore, SessionEntry, SessionSource, SessionStore
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.thread_reset_policy import ThreadResetPolicyStateError
from hermes_cli.commands import (
    resolve_command,
    slack_native_slashes,
    telegram_bot_commands,
)


def _source(
    thread_id: str | None = "10",
    *,
    chat_id: str = "-1001",
    chat_type: str | None = None,
    user_id: str = "user-a",
    platform: Platform = Platform.TELEGRAM,
    profile: str | None = None,
    scope_id: str | None = None,
) -> SessionSource:
    if chat_type is None:
        chat_type = {
            Platform.TELEGRAM: "forum",
            Platform.DISCORD: "thread",
            Platform.SLACK: "group",
        }.get(platform, "group")
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
        user_id=user_id,
        profile=profile,
        scope_id=scope_id,
    )


def _store(tmp_path, config: GatewayConfig | None = None) -> SessionStore:
    config = config or GatewayConfig()
    config.sessions_dir = tmp_path
    store = SessionStore(tmp_path, config)
    store._db = None
    return store


def _entry(
    source: SessionSource,
    updated_at: datetime,
    *,
    session_id: str | None = None,
) -> SessionEntry:
    return SessionEntry(
        session_key="route",
        session_id=session_id or f"session-{source.thread_id}",
        created_at=updated_at,
        updated_at=updated_at,
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _runner(store: SessionStore, *, slack_prefix: bool = False):
    runner = GatewaySlashCommandsMixin()
    runner.session_store = store
    runner.async_session_store = AsyncSessionStore(store)
    runner.adapters = (
        {Platform.SLACK: SimpleNamespace(typed_command_prefix="!")}
        if slack_prefix
        else {}
    )
    return runner


def test_thread_identity_isolates_profile_platform_chat_and_thread_not_user_or_session(
    tmp_path,
):
    store = _store(tmp_path, GatewayConfig(multiplex_profiles=True))
    source = _source(profile="work")
    store.set_thread_reset_policy(
        source,
        SessionResetPolicy(mode="daily", at_hour=6, at_minute=15),
    )

    same_route_other_user = _source(user_id="user-b", profile="work")
    same_route_other_session = _entry(
        same_route_other_user,
        datetime(2026, 7, 29),
        session_id="completely-unrelated-session-id",
    )
    isolated = (
        _source(thread_id="11", profile="work"),
        _source(chat_id="-1002", profile="work"),
        _source(profile="personal"),
        _source(platform=Platform.DISCORD, profile="work"),
        _source(platform=Platform.SLACK, profile="work"),
    )

    policy, resolution = store.get_effective_reset_policy(
        entry=same_route_other_session
    )
    assert (policy.at_hour, policy.at_minute, resolution) == (6, 15, "override")
    assert all(
        store.get_effective_reset_policy(source=other)[1] == "inherited"
        for other in isolated
    )


def test_slack_thread_overrides_are_isolated_by_workspace_scope(tmp_path):
    store = _store(tmp_path)
    workspace_a = _source(
        platform=Platform.SLACK,
        chat_id="C123",
        thread_id="1712345678.000001",
        scope_id="T-A",
    )
    workspace_b = _source(
        platform=Platform.SLACK,
        chat_id="C123",
        thread_id="1712345678.000001",
        scope_id="T-B",
    )

    store.set_thread_reset_policy(
        workspace_a,
        SessionResetPolicy(mode="daily", at_hour=6, at_minute=15),
    )
    store.set_thread_reset_policy(workspace_b, SessionResetPolicy(mode="none"))

    reloaded = _store(tmp_path)
    policy_a, resolution_a = reloaded.get_effective_reset_policy(source=workspace_a)
    policy_b, resolution_b = reloaded.get_effective_reset_policy(source=workspace_b)
    assert (policy_a.mode, policy_a.at_hour, policy_a.at_minute, resolution_a) == (
        "daily",
        6,
        15,
        "override",
    )
    assert (policy_b.mode, resolution_b) == ("none", "override")


def test_discord_thread_overrides_are_isolated_by_guild_scope(tmp_path):
    store = _store(tmp_path)
    guild_a = _source(
        platform=Platform.DISCORD,
        chat_id="1234567890",
        thread_id="9876543210",
        scope_id="guild:alpha/β",
    )
    guild_b = _source(
        platform=Platform.DISCORD,
        chat_id="1234567890",
        thread_id="9876543210",
        scope_id="guild:alpha%2Fβ",
    )

    store.set_thread_reset_policy(
        guild_a,
        SessionResetPolicy(mode="daily", at_hour=7, at_minute=45),
    )
    store.set_thread_reset_policy(guild_b, SessionResetPolicy(mode="none"))

    reloaded = _store(tmp_path)
    policy_a, resolution_a = reloaded.get_effective_reset_policy(source=guild_a)
    policy_b, resolution_b = reloaded.get_effective_reset_policy(source=guild_b)
    assert (policy_a.mode, policy_a.at_hour, policy_a.at_minute, resolution_a) == (
        "daily",
        7,
        45,
        "override",
    )
    assert (policy_b.mode, resolution_b) == ("none", "override")


def test_explicit_policy_persists_in_versioned_generic_state(tmp_path):
    source = _source()
    first = _store(tmp_path)
    first.set_thread_reset_policy(
        source,
        SessionResetPolicy(mode="daily", at_hour=23, at_minute=59),
    )

    state_path = tmp_path / "thread_reset_policies.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["version"] == 1
    assert list(state) == ["version", "threads"]
    assert list(state["threads"].values()) == [
        {"mode": "daily", "at_hour": 23, "at_minute": 59}
    ]
    assert not (tmp_path / "telegram_topic_reset_policies.json").exists()

    reloaded = _store(tmp_path)
    policy, resolution = reloaded.get_effective_reset_policy(source=source)
    assert (policy.mode, policy.at_hour, policy.at_minute, resolution) == (
        "daily",
        23,
        59,
        "override",
    )

    reloaded.set_thread_reset_policy(source, SessionResetPolicy(mode="none"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert list(state["threads"].values()) == [{"mode": "none"}]


def test_failed_atomic_write_keeps_last_known_policy(tmp_path, monkeypatch):
    source = _source()
    store = _store(tmp_path)
    store.set_thread_reset_policy(source, SessionResetPolicy(mode="none"))

    def fail_write(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        "gateway.thread_reset_policy.atomic_json_write",
        fail_write,
    )
    with pytest.raises(OSError, match="disk unavailable"):
        store.set_thread_reset_policy(
            source,
            SessionResetPolicy(mode="daily", at_hour=4, at_minute=30),
        )

    assert store.get_effective_reset_policy(source=source)[0].mode == "none"
    assert _store(tmp_path).get_effective_reset_policy(source=source)[0].mode == "none"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"version": 2, "threads": {}}',
        '{"version": 1, "threads": {}, "extra": true}',
        '{"version": 1, "threads": []}',
        '{"version": 1, "threads": {"route": "off"}}',
        '{"version": 1, "threads": {"route": {"mode": "none", "at_hour": 4}}}',
        '{"version": 1, "threads": {"route": {"mode": "daily", "at_hour": 4}}}',
        '{"version": 1, "threads": {"route": {"mode": "daily", "at_hour": true, "at_minute": 0}}}',
        '{"version": 1, "threads": {"route": {"mode": "daily", "at_hour": 4, "at_minute": 60}}}',
        '{"version": 1, "threads": {"route": {"mode": "idle"}}}',
    ],
)
def test_malformed_persistence_fails_closed_and_preserves_evidence(
    tmp_path, caplog, raw
):
    state_path = tmp_path / "thread_reset_policies.json"
    state_path.write_text(raw, encoding="utf-8")
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="daily", at_hour=4, at_minute=30)
    )

    with caplog.at_level("WARNING"):
        store = _store(tmp_path, config)

    policy, resolution = store.get_effective_reset_policy(source=_source())
    assert (policy.mode, resolution) == ("none", "invalid")
    assert "Malformed thread auto-reset state" in caplog.text
    with pytest.raises(ThreadResetPolicyStateError):
        store.set_thread_reset_policy(_source(), SessionResetPolicy(mode="none"))
    assert state_path.read_text(encoding="utf-8") == raw


def test_invalid_in_memory_policy_is_rejected_without_losing_last_known_good(tmp_path):
    source = _source()
    store = _store(tmp_path)
    store.set_thread_reset_policy(source, SessionResetPolicy(mode="none"))

    with pytest.raises(ValueError, match="mode"):
        store.set_thread_reset_policy(
            source,
            SessionResetPolicy(mode="idle", idle_minutes=10),
        )
    with pytest.raises(ValueError, match="at_minute"):
        store.set_thread_reset_policy(
            source,
            SessionResetPolicy(mode="daily", at_hour=4, at_minute=60),
        )

    assert store.get_effective_reset_policy(source=source)[0].mode == "none"


def test_thread_policy_precedes_platform_then_type_then_global(tmp_path):
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=30),
        reset_by_type={
            "forum": SessionResetPolicy(
                mode="both", at_hour=7, at_minute=20, idle_minutes=45
            ),
            "thread": SessionResetPolicy(mode="daily", at_hour=6, at_minute=25),
        },
        reset_by_platform={
            Platform.TELEGRAM: SessionResetPolicy(mode="daily", at_hour=9, at_minute=15)
        },
    )
    store = _store(tmp_path, config)
    source = _source()

    inherited, resolution = store.get_effective_reset_policy(source=source)
    assert (
        inherited.mode,
        inherited.at_hour,
        inherited.at_minute,
        resolution,
    ) == ("daily", 9, 15, "inherited")

    store.set_thread_reset_policy(
        source,
        SessionResetPolicy(mode="daily", at_hour=5, at_minute=40),
    )
    overridden, resolution = store.get_effective_reset_policy(source=source)
    assert (
        overridden.mode,
        overridden.at_hour,
        overridden.at_minute,
        resolution,
    ) == ("daily", 5, 40, "override")

    store.set_thread_reset_policy(source, None)
    restored, resolution = store.get_effective_reset_policy(source=source)
    assert (
        restored.mode,
        restored.at_hour,
        restored.at_minute,
        resolution,
    ) == ("daily", 9, 15, "inherited")

    discord = _source(platform=Platform.DISCORD)
    by_type, resolution = store.get_effective_reset_policy(source=discord)
    assert (
        by_type.mode,
        by_type.at_hour,
        by_type.at_minute,
        resolution,
    ) == ("daily", 6, 25, "inherited")


def test_two_threads_can_hold_different_daily_policies(tmp_path):
    store = _store(tmp_path)
    early = _source(thread_id="early")
    late = _source(thread_id="late")
    store.set_thread_reset_policy(
        early, SessionResetPolicy(mode="daily", at_hour=4, at_minute=5)
    )
    store.set_thread_reset_policy(
        late, SessionResetPolicy(mode="daily", at_hour=22, at_minute=55)
    )

    early_policy = store.get_effective_reset_policy(source=early)[0]
    late_policy = store.get_effective_reset_policy(source=late)[0]
    assert (early_policy.at_hour, early_policy.at_minute) == (4, 5)
    assert (late_policy.at_hour, late_policy.at_minute) == (22, 55)


@pytest.mark.asyncio
async def test_autoreset_command_on_daily_off_inherit_and_status(tmp_path):
    store = _store(tmp_path)
    runner = _runner(store)
    source = _source()

    assert resolve_command("autoreset").name == "autoreset"
    assert (
        await runner._handle_autoreset_command(
            MessageEvent(text="/autoreset", source=source)
        )
        == "Automatic reset: disabled (inherited policy)."
    )
    assert (
        await runner._handle_autoreset_command(
            MessageEvent(text="/autoreset on", source=source)
        )
        == "Automatic reset: daily at 04:00 (thread override)."
    )
    assert (
        await runner._handle_autoreset_command(
            MessageEvent(text="/autoreset daily 17:43", source=source)
        )
        == "Automatic reset: daily at 17:43 (thread override)."
    )
    assert (
        await runner._handle_autoreset_command(
            MessageEvent(text="/autoreset status", source=source)
        )
        == "Automatic reset: daily at 17:43 (thread override)."
    )
    assert (
        await runner._handle_autoreset_command(
            MessageEvent(text="/autoreset off", source=source)
        )
        == "Automatic reset: disabled (thread override)."
    )
    assert (
        await runner._handle_autoreset_command(
            MessageEvent(text="/autoreset inherit", source=source)
        )
        == "Automatic reset: disabled (inherited policy)."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        "daily",
        "daily ",
        "daily 4:00",
        "daily 04:0",
        "daily 24:00",
        "daily 23:60",
        "daily 12:30 extra",
        "maybe",
    ],
)
async def test_autoreset_command_rejects_invalid_times_without_rounding(tmp_path, args):
    store = _store(tmp_path)
    result = await _runner(store)._handle_autoreset_command(
        MessageEvent(text=f"/autoreset {args}", source=_source())
    )
    assert result == ("Usage: /autoreset [status|on|daily HH:MM|off|inherit]")
    assert not (tmp_path / "thread_reset_policies.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "chat_type"),
    [
        (Platform.TELEGRAM, "forum"),
        (Platform.DISCORD, "thread"),
        (Platform.SLACK, "group"),
        (Platform.TELEGRAM, "dm"),
        (Platform.DISCORD, "dm"),
        (Platform.SLACK, "dm"),
    ],
)
async def test_autoreset_accepts_supported_thread_sources(
    tmp_path, platform, chat_type
):
    store = _store(tmp_path)
    source = _source(platform=platform, chat_type=chat_type)
    result = await _runner(
        store, slack_prefix=platform == Platform.SLACK
    )._handle_autoreset_command(
        MessageEvent(
            text="/autoreset daily 08:07",
            source=source,
        )
    )

    assert result == "Automatic reset: daily at 08:07 (thread override)."
    assert store.get_effective_reset_policy(source=source)[1] == "override"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        _source(platform=Platform.TELEGRAM, thread_id=None, chat_type="dm"),
        _source(platform=Platform.DISCORD, thread_id=None, chat_type="group"),
        _source(platform=Platform.SLACK, thread_id=None),
        _source(platform=Platform.SLACK, chat_id=" "),
        _source(platform=Platform.SLACK, thread_id=" "),
    ],
)
async def test_autoreset_rejects_root_and_non_thread_sources(tmp_path, source):
    store = _store(tmp_path)
    runner = _runner(store, slack_prefix=source.platform == Platform.SLACK)
    result = await runner._handle_autoreset_command(
        MessageEvent(text="/autoreset on", source=source)
    )

    prefix = "!" if source.platform == Platform.SLACK else "/"
    assert result == (
        f"Run {prefix}autoreset inside the thread you want to configure "
        "(not a root conversation or parent channel)."
    )
    assert not (tmp_path / "thread_reset_policies.json").exists()


@pytest.mark.asyncio
async def test_autoreset_rejects_unsupported_platform(tmp_path):
    result = await _runner(_store(tmp_path))._handle_autoreset_command(
        MessageEvent(
            text="/autoreset on",
            source=_source(platform=Platform.MATRIX),
        )
    )
    assert result == (
        "/autoreset is only available in Telegram, Discord, and Slack threads."
    )


@pytest.mark.asyncio
async def test_malformed_state_command_fails_safely(tmp_path):
    (tmp_path / "thread_reset_policies.json").write_text(
        "not json",
        encoding="utf-8",
    )
    store = _store(
        tmp_path,
        GatewayConfig(default_reset_policy=SessionResetPolicy(mode="daily")),
    )
    runner = _runner(store)

    status = await runner._handle_autoreset_command(
        MessageEvent(text="/autoreset status", source=_source())
    )
    mutation = await runner._handle_autoreset_command(
        MessageEvent(text="/autoreset on", source=_source())
    )

    assert status == (
        "Automatic reset: disabled (safe fallback; thread policy state is invalid)."
    )
    assert mutation == ("Thread auto-reset state is invalid; no changes were made.")


def test_daily_reset_boundary_uses_minute_precision(tmp_path, monkeypatch):
    source = _source()
    store = _store(tmp_path)
    store.set_thread_reset_policy(
        source,
        SessionResetPolicy(mode="daily", at_hour=4, at_minute=30),
    )
    entry = _entry(source, datetime(2026, 7, 28, 4, 31))

    monkeypatch.setattr(
        "gateway.session._now", lambda: datetime(2026, 7, 29, 4, 29, 59)
    )
    assert store._is_session_expired(entry) is False
    assert store._should_reset(entry, source) is None

    monkeypatch.setattr("gateway.session._now", lambda: datetime(2026, 7, 29, 4, 30))
    assert store._is_session_expired(entry) is True
    assert store._should_reset(entry, source) == "daily"

    monkeypatch.setattr("gateway.session._now", lambda: datetime(2026, 7, 29, 4, 31))
    assert store._is_session_expired(entry) is True


def test_watcher_due_selection_and_active_process_safeguard_are_preserved(
    tmp_path, monkeypatch
):
    now = datetime(2026, 7, 29, 10, 0)
    monkeypatch.setattr("gateway.session._now", lambda: now)
    store = _store(tmp_path)
    due = _source(thread_id="due")
    later = _source(thread_id="later")
    store.set_thread_reset_policy(
        due, SessionResetPolicy(mode="daily", at_hour=9, at_minute=30)
    )
    store.set_thread_reset_policy(
        later, SessionResetPolicy(mode="daily", at_hour=10, at_minute=30)
    )
    entries = [
        _entry(due, now - timedelta(hours=2)),
        _entry(later, now - timedelta(hours=2)),
    ]

    selected = [
        entry.origin.thread_id for entry in entries if store._is_session_expired(entry)
    ]
    assert selected == ["due"]

    monkeypatch.setattr(store, "_has_active_processes_safe", lambda *_a, **_k: True)
    assert store._is_session_expired(entries[0]) is False
    assert store._should_reset(entries[0], due) is None


def test_manual_reset_rotates_session_and_preserves_thread_policy(tmp_path):
    store = _store(tmp_path)
    source = _source()
    store.set_thread_reset_policy(
        source,
        SessionResetPolicy(mode="daily", at_hour=12, at_minute=34),
    )
    original = store.get_or_create_session(source)

    reset = store.reset_session(original.session_key)

    assert reset is not None
    assert reset.session_id != original.session_id
    assert reset.is_fresh_reset is True
    policy, resolution = store.get_effective_reset_policy(source=source)
    assert (policy.at_hour, policy.at_minute, resolution) == (12, 34, "override")


def test_registry_surfaces_telegram_and_discord_but_not_native_slack():
    assert "autoreset" in {name for name, _description in telegram_bot_commands()}
    command = resolve_command("autoreset")
    assert command is not None and command.gateway_only is True
    assert "autoreset" not in {
        name for name, _description, _hint in slack_native_slashes()
    }
    assert "start" in {name for name, _description, _hint in slack_native_slashes()}
