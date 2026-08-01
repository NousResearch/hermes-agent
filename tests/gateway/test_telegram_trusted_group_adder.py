import asyncio
import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageType


def _adapter(*, enabled: object = True, trusted=None, extra=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    settings = dict(extra or {})
    if enabled is not None:
        settings["auto_allow_groups_from_trusted_adders"] = enabled
    if trusted is not None:
        settings["trusted_group_adders"] = trusted

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="not-written", extra=settings)
    adapter._bot = SimpleNamespace(id=999, token="not-written")
    adapter._active_telegram_auto_authorized_groups = set()
    return adapter


def _membership_update(
    *,
    chat_id=-100123,
    chat_type="supergroup",
    actor_id=111,
    old_status="member",
    new_status="administrator",
    member_user_id=999,
):
    member = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Private team content"),
        from_user=SimpleNamespace(id=actor_id, username="secret-operator"),
        old_chat_member=SimpleNamespace(
            status=old_status,
            user=SimpleNamespace(id=member_user_id),
        ),
        new_chat_member=SimpleNamespace(
            status=new_status,
            user=SimpleNamespace(id=member_user_id),
        ),
    )
    return SimpleNamespace(my_chat_member=member)


def _message(*, chat_id=-100123, chat_type="group", is_forum=False):
    return SimpleNamespace(
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
            title="Private team content",
            full_name=None,
            is_forum=is_forum,
        ),
        from_user=SimpleNamespace(
            id=111,
            full_name="Trusted stakeholder",
            is_bot=False,
        ),
        text="hello",
        message_id=42,
        message_thread_id=7 if is_forum else None,
        is_topic_message=is_forum,
        reply_to_message=None,
        date=None,
        forum_topic_created=None,
    )


@pytest.mark.asyncio
async def test_trusted_admin_promotion_persists_and_extends_effective_allowlists(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111], extra={
        "allowed_chats": ["-200"],
        "group_allowed_chats": ["-300"],
    })

    await adapter._handle_my_chat_member(_membership_update(), None)

    assert adapter._telegram_allowed_chats() == {"-200", "-100123"}
    assert adapter._telegram_group_allowed_chats() == {"-300", "-100123"}

    state_path = tmp_path / "state" / "telegram-auto-authorized-groups.json"
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "chat_ids": ["-100123"]
    }
    if os.name != "nt":
        assert stat.S_IMODE(state_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    raw = state_path.read_text(encoding="utf-8")
    assert "not-written" not in raw
    assert "Private team content" not in raw
    assert "secret-operator" not in raw


def test_adapter_captures_profile_home_for_durable_group_state(monkeypatch, tmp_path):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    first_home = tmp_path / "first-profile"
    second_home = tmp_path / "second-profile"
    unrelated_home = tmp_path / "later-runtime-scope"
    config = PlatformConfig(
        enabled=True,
        token="not-written",
        extra={"auto_allow_groups_from_trusted_adders": True},
    )

    monkeypatch.setenv("HERMES_HOME", str(first_home))
    first = TelegramAdapter(config)
    monkeypatch.setenv("HERMES_HOME", str(second_home))
    second = TelegramAdapter(config)

    # Multiplex dispatch runs after the construction-time profile scope exits.
    monkeypatch.setenv("HERMES_HOME", str(unrelated_home))
    first._write_telegram_auto_authorized_groups({"-100"})
    second._write_telegram_auto_authorized_groups({"-200"})

    assert first._telegram_auto_authorized_groups() == {"-100"}
    assert second._telegram_auto_authorized_groups() == {"-200"}
    assert json.loads(
        (first_home / "state" / "telegram-auto-authorized-groups.json").read_text(
            encoding="utf-8"
        )
    ) == {"chat_ids": ["-100"]}
    assert json.loads(
        (second_home / "state" / "telegram-auto-authorized-groups.json").read_text(
            encoding="utf-8"
        )
    ) == {"chat_ids": ["-200"]}
    assert not (unrelated_home / "state" / "telegram-auto-authorized-groups.json").exists()


@pytest.mark.parametrize(
    ("chat_type", "is_forum"),
    [("group", False), ("supergroup", True)],
)
def test_message_event_marks_only_active_dynamic_groups(chat_type, is_forum):
    adapter = _adapter(trusted=[111])
    adapter._active_telegram_auto_authorized_groups.add("-100123")

    event = adapter._build_message_event(
        _message(chat_type=chat_type, is_forum=is_forum),
        MessageType.TEXT,
    )

    assert event.metadata == {"telegram_auto_authorized_group_active": True}


def test_message_event_does_not_mark_unreconciled_persisted_candidate(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "telegram-auto-authorized-groups.json").write_text(
        json.dumps({"chat_ids": ["-100123"]}),
        encoding="utf-8",
    )
    adapter = _adapter(trusted=[111])

    assert adapter._telegram_auto_authorized_groups() == {"-100123"}
    event = adapter._build_message_event(_message(), MessageType.TEXT)

    assert "telegram_auto_authorized_group_active" not in event.metadata


@pytest.mark.parametrize(
    ("message", "active", "enabled", "explicit"),
    [
        (_message(), set(), True, ["-100123"]),
        (_message(), {"-100123"}, False, []),
        (_message(chat_id=111, chat_type="private"), {"111"}, True, []),
        (_message(chat_type="channel"), {"-100123"}, True, []),
        (_message(), None, True, []),
        (_message(), ["-100123"], True, []),
        (_message(chat_id="malformed"), {"malformed"}, True, []),
    ],
)
def test_message_event_dynamic_group_signal_fails_closed(
    message, active, enabled, explicit
):
    adapter = _adapter(
        enabled=enabled,
        trusted=[111],
        extra={"group_allowed_chats": explicit},
    )
    adapter._active_telegram_auto_authorized_groups = active

    event = adapter._build_message_event(message, MessageType.TEXT)

    assert "telegram_auto_authorized_group_active" not in event.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update,adapter",
    [
        (_membership_update(actor_id=222), _adapter(trusted=[111])),
        (_membership_update(new_status="member"), _adapter(trusted=[111])),
        (_membership_update(chat_type="channel"), _adapter(trusted=[111])),
        (_membership_update(chat_type="private", chat_id=123), _adapter(trusted=[111])),
        (_membership_update(member_user_id=12345), _adapter(trusted=[111])),
        (_membership_update(), _adapter(enabled=False, trusted=[111])),
        (_membership_update(), _adapter(trusted="111")),
        (_membership_update(), _adapter(trusted=[True, "not-an-id"])),
        (_membership_update(), _adapter(enabled="sometimes", trusted=[111])),
    ],
)
async def test_untrusted_or_malformed_updates_fail_closed(
    monkeypatch, tmp_path, update, adapter
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    await adapter._handle_my_chat_member(update, None)

    assert adapter._telegram_auto_authorized_groups() == set()
    assert not (tmp_path / "state" / "telegram-auto-authorized-groups.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("removed_status", ["member", "restricted", "left", "kicked"])
async def test_transition_away_from_administrator_revokes_regardless_of_actor(
    monkeypatch, tmp_path, removed_status
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])
    await adapter._handle_my_chat_member(_membership_update(), None)

    await adapter._handle_my_chat_member(
        _membership_update(
            actor_id=222,
            old_status="administrator",
            new_status=removed_status,
        ),
        None,
    )

    assert adapter._telegram_auto_authorized_groups() == set()
    assert adapter._telegram_allowed_chats() == set()
    assert adapter._telegram_group_allowed_chats() == set()


@pytest.mark.asyncio
async def test_malformed_demotion_without_target_user_still_revokes_known_chat(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])
    await adapter._handle_my_chat_member(_membership_update(), None)

    await adapter._handle_my_chat_member(
        _membership_update(
            actor_id=222,
            old_status="administrator",
            new_status="member",
            member_user_id=None,
        ),
        None,
    )

    assert adapter._telegram_auto_authorized_groups() == set()
    assert adapter._active_telegram_auto_authorized_group_ids() == set()


@pytest.mark.asyncio
async def test_gateway_authz_uses_only_reconciled_persisted_group_admission(
    monkeypatch, tmp_path
):
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "telegram-auto-authorized-groups.json").write_text(
        json.dumps({"chat_ids": ["-100123"]}),
        encoding="utf-8",
    )
    adapter = _adapter(trusted=[111], extra={"group_allowed_chats": ["-200"]})
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        user_id=None,
        user_name=None,
    )

    assert runner._is_user_authorized(source) is False

    adapter._bot = SimpleNamespace(
        id=999,
        get_chat_member=AsyncMock(
            return_value=SimpleNamespace(status="administrator")
        ),
    )
    await adapter._reconcile_telegram_auto_authorized_groups()

    assert runner._is_user_authorized(source) is True
    assert adapter.config.extra["group_allowed_chats"] == ["-200"]


def test_gateway_authz_strict_profile_ignores_legacy_group_environment(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100123")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(trusted=[111])
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        user_id="42",
        user_name=None,
    )

    assert adapter._telegram_group_allowed_chats() == set()
    assert runner._is_user_authorized(source) is False


@pytest.mark.asyncio
async def test_restart_reconciliation_activates_only_current_administrators(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "telegram-auto-authorized-groups.json").write_text(
        json.dumps({"chat_ids": ["-100", "-200", "-300"]}),
        encoding="utf-8",
    )
    adapter = _adapter(trusted=[111])

    async def get_chat_member(chat_id, bot_id):
        assert bot_id == 999
        if chat_id == "-100":
            return SimpleNamespace(status="administrator")
        if chat_id == "-200":
            return SimpleNamespace(status="member")
        raise RuntimeError("Bot API unavailable")

    adapter._bot = SimpleNamespace(id=999, get_chat_member=get_chat_member)

    assert adapter._telegram_allowed_chats() == set()
    await adapter._reconcile_telegram_auto_authorized_groups()

    assert adapter._telegram_allowed_chats() == {"-100"}
    assert adapter._telegram_group_allowed_chats() == {"-100"}
    assert adapter._telegram_auto_authorized_groups() == {"-100", "-200", "-300"}


@pytest.mark.asyncio
async def test_reconciliation_cannot_resurrect_group_after_concurrent_demotion(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "telegram-auto-authorized-groups.json").write_text(
        json.dumps({"chat_ids": ["-100123"]}),
        encoding="utf-8",
    )
    validation_started = asyncio.Event()
    finish_validation = asyncio.Event()

    async def get_chat_member(chat_id, bot_id):
        assert (chat_id, bot_id) == ("-100123", 999)
        validation_started.set()
        await finish_validation.wait()
        return SimpleNamespace(status="administrator")

    adapter = _adapter(trusted=[111])
    adapter._bot = SimpleNamespace(id=999, get_chat_member=get_chat_member)

    reconciliation = asyncio.create_task(
        adapter._reconcile_telegram_auto_authorized_groups()
    )
    await validation_started.wait()
    await adapter._handle_my_chat_member(
        _membership_update(
            actor_id=222,
            old_status="administrator",
            new_status="member",
        ),
        None,
    )
    finish_validation.set()
    await reconciliation

    assert adapter._telegram_auto_authorized_groups() == set()
    assert adapter._active_telegram_auto_authorized_group_ids() == set()


@pytest.mark.asyncio
async def test_failed_durable_revocation_tombstones_against_inflight_reconciliation(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "telegram-auto-authorized-groups.json"
    state_path.write_text(
        json.dumps({"chat_ids": ["-100123"]}),
        encoding="utf-8",
    )
    validation_started = asyncio.Event()
    finish_validation = asyncio.Event()

    async def get_chat_member(chat_id, bot_id):
        assert (chat_id, bot_id) == ("-100123", 999)
        validation_started.set()
        await finish_validation.wait()
        return SimpleNamespace(status="administrator")

    adapter = _adapter(trusted=[111])
    adapter._active_telegram_auto_authorized_groups.add("-100123")
    adapter._bot = SimpleNamespace(id=999, get_chat_member=get_chat_member)
    reconciliation = asyncio.create_task(
        adapter._reconcile_telegram_auto_authorized_groups()
    )
    await validation_started.wait()

    monkeypatch.setattr(
        adapter,
        "_write_telegram_auto_authorized_groups",
        Mock(side_effect=OSError("disk full")),
    )
    await adapter._handle_my_chat_member(
        _membership_update(old_status="administrator", new_status="member"),
        None,
    )
    finish_validation.set()
    await reconciliation

    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "chat_ids": ["-100123"]
    }
    assert adapter._active_telegram_auto_authorized_group_ids() == set()


@pytest.mark.asyncio
async def test_persistence_failure_never_creates_or_keeps_a_live_grant(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])

    def fail_write(_chat_ids):
        raise OSError("disk full")

    monkeypatch.setattr(adapter, "_write_telegram_auto_authorized_groups", fail_write)

    await adapter._handle_my_chat_member(_membership_update(), None)
    assert adapter._telegram_allowed_chats() == set()

    adapter._active_telegram_auto_authorized_groups.add("-100123")
    await adapter._handle_my_chat_member(
        _membership_update(
            actor_id=222,
            old_status="administrator",
            new_status="member",
        ),
        None,
    )
    assert adapter._telegram_allowed_chats() == set()


def test_strict_profile_allowlists_do_not_inherit_process_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHATS", "-100")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-200")

    profile_a_legacy = _adapter(enabled=False)
    profile_b_strict = _adapter(enabled=True)

    assert profile_a_legacy._telegram_allowed_chats() == {"-100"}
    assert profile_a_legacy._telegram_group_allowed_chats() == {"-200"}
    assert profile_b_strict._telegram_allowed_chats() == set()
    assert profile_b_strict._telegram_group_allowed_chats() == set()


@pytest.mark.asyncio
async def test_promotion_with_missing_old_status_is_denied(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])

    await adapter._handle_my_chat_member(
        _membership_update(old_status=None, new_status="administrator"),
        None,
    )

    assert adapter._telegram_auto_authorized_groups() == set()
    assert adapter._active_telegram_auto_authorized_group_ids() == set()


@pytest.mark.asyncio
async def test_recognized_non_admin_status_revokes_with_missing_old_status(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])
    await adapter._handle_my_chat_member(_membership_update(), None)

    await adapter._handle_my_chat_member(
        _membership_update(
            actor_id=None,
            old_status=None,
            new_status="member",
            member_user_id=None,
        ),
        None,
    )

    assert adapter._telegram_auto_authorized_groups() == set()
    assert adapter._active_telegram_auto_authorized_group_ids() == set()


@pytest.mark.asyncio
async def test_unknown_new_status_revokes_known_dynamic_group(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])
    await adapter._handle_my_chat_member(_membership_update(), None)

    await adapter._handle_my_chat_member(
        _membership_update(old_status=None, new_status=None, member_user_id=None),
        None,
    )

    assert adapter._telegram_auto_authorized_groups() == set()
    assert adapter._active_telegram_auto_authorized_group_ids() == set()


@pytest.mark.asyncio
async def test_malformed_state_fails_closed_without_being_overwritten(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    state_path = state_dir / "telegram-auto-authorized-groups.json"
    state_path.write_text('{"chat_ids": "-100123"}', encoding="utf-8")
    adapter = _adapter(trusted=[111])

    await adapter._handle_my_chat_member(_membership_update(), None)

    assert adapter._telegram_auto_authorized_groups() == set()
    assert state_path.read_text(encoding="utf-8") == '{"chat_ids": "-100123"}'


@pytest.mark.asyncio
async def test_symlinked_or_non_regular_state_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    external = tmp_path / "external.json"
    external.write_text(json.dumps({"chat_ids": ["-100123"]}), encoding="utf-8")
    state_path = state_dir / "telegram-auto-authorized-groups.json"
    state_path.symlink_to(external)
    adapter = _adapter(trusted=[111])

    assert adapter._telegram_auto_authorized_groups() == set()
    await adapter._handle_my_chat_member(_membership_update(), None)
    assert state_path.is_symlink()
    assert json.loads(external.read_text(encoding="utf-8")) == {
        "chat_ids": ["-100123"]
    }

    state_path.unlink()
    state_path.mkdir()
    assert adapter._telegram_auto_authorized_groups() == set()


@pytest.mark.parametrize("chat_id", ["0", "-0", "100", 100, True, "group"])
def test_persisted_schema_accepts_only_negative_numeric_chat_id_strings(chat_id):
    adapter = _adapter(trusted=[111])

    assert adapter._validated_auto_authorized_chat_ids({"chat_ids": [chat_id]}) is None


def test_trusted_adder_settings_bridge_from_profile_config(monkeypatch, tmp_path):
    hermes_home = tmp_path / "profile-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "telegram:\n"
        "  auto_allow_groups_from_trusted_adders: true\n"
        "  trusted_group_adders:\n"
        "    - 111\n"
        "    - \"222\"\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()
    telegram_config = config.platforms[Platform.TELEGRAM]
    adapter = _adapter(extra=telegram_config.extra)

    assert telegram_config.extra["auto_allow_groups_from_trusted_adders"] is True
    assert telegram_config.extra["trusted_group_adders"] == [111, "222"]
    assert adapter._telegram_auto_allow_groups_from_trusted_adders() is True
    assert adapter._telegram_trusted_group_adders() == {111, 222}


def test_windows_state_write_keeps_atomic_replace_without_posix_operations(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])
    monkeypatch.setattr(adapter, "_supports_posix_group_state_security", lambda: False)
    monkeypatch.setattr(os, "chmod", Mock(side_effect=AssertionError("POSIX chmod")))
    monkeypatch.setattr(os, "fchmod", Mock(side_effect=AssertionError("POSIX fchmod")))
    real_fsync = os.fsync
    fsync_calls = []

    def track_file_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", track_file_fsync)

    adapter._write_telegram_auto_authorized_groups({"-100123"})

    state_path = tmp_path / "state" / "telegram-auto-authorized-groups.json"
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "chat_ids": ["-100123"]
    }
    assert len(fsync_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["group", "supergroup"])
async def test_strict_admission_rejects_group_callback_before_any_state_mutation(
    monkeypatch, chat_type
):
    adapter = _adapter(trusted=[111])
    adapter._approval_state = {7: "session-key"}
    adapter._clarify_state = {"cid": "session-key"}
    adapter._model_picker_state = {"-100123": {"page": 1}}
    model_handler = AsyncMock()
    choice_handler = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_model_picker_callback", model_handler)
    monkeypatch.setattr(adapter, "_handle_choice_picker_callback", choice_handler)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")

    for data in ("mp:provider", "cp:reasoning", "ea:once:7", "cl:cid:0"):
        query = SimpleNamespace(
            data=data,
            message=SimpleNamespace(
                chat_id=-100123,
                chat=SimpleNamespace(type=chat_type),
                message_thread_id=8 if chat_type == "supergroup" else None,
            ),
            from_user=SimpleNamespace(id=111, first_name="Trusted"),
            answer=AsyncMock(),
        )
        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), SimpleNamespace()
        )
        assert "not authorized" in query.answer.call_args.kwargs["text"].lower()

    model_handler.assert_not_awaited()
    choice_handler.assert_not_awaited()
    assert adapter._approval_state == {7: "session-key"}
    assert adapter._clarify_state == {"cid": "session-key"}
    assert adapter._model_picker_state == {"-100123": {"page": 1}}


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", [None, "unknown"])
async def test_strict_admission_rejects_negative_callback_with_malformed_chat_type(
    monkeypatch, chat_type
):
    adapter = _adapter(trusted=[111])
    adapter._approval_state = {7: "session-key"}
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
    query = SimpleNamespace(
        data="ea:once:7",
        message=SimpleNamespace(
            chat_id=-100123,
            chat=SimpleNamespace(type=chat_type),
            message_thread_id=None,
        ),
        from_user=SimpleNamespace(id=111, first_name="Trusted"),
        answer=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    assert adapter._approval_state == {7: "session-key"}
    assert "not authorized" in query.answer.call_args.kwargs["text"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["explicit", "active"])
async def test_admitted_group_callback_retains_existing_callback_auth(
    monkeypatch, admission
):
    from tools import approval

    extra = {"allowed_chats": ["-100123"]} if admission == "explicit" else {}
    adapter = _adapter(trusted=[111], extra=extra)
    if admission == "active":
        adapter._active_telegram_auto_authorized_groups.add("-100123")
    adapter._approval_state = {7: "session-key"}
    resolved = Mock(return_value=0)
    monkeypatch.setattr(approval, "resolve_gateway_approval", resolved)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
    query = SimpleNamespace(
        data="ea:once:7",
        message=SimpleNamespace(
            chat_id=-100123,
            chat=SimpleNamespace(type="supergroup"),
            message_thread_id=8,
        ),
        from_user=SimpleNamespace(id=111, first_name="Trusted"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    resolved.assert_called_once_with("session-key", "once")
    assert adapter._approval_state == {}


@pytest.mark.asyncio
async def test_revoked_group_callback_is_denied_before_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
    adapter = _adapter(trusted=[111])
    await adapter._handle_my_chat_member(_membership_update(), None)
    await adapter._handle_my_chat_member(
        _membership_update(old_status="administrator", new_status="member"),
        None,
    )
    adapter._handle_choice_picker_callback = AsyncMock()
    query = SimpleNamespace(
        data="cp:reasoning",
        message=SimpleNamespace(
            chat_id=-100123,
            chat=SimpleNamespace(type="supergroup"),
            message_thread_id=8,
        ),
        from_user=SimpleNamespace(id=111, first_name="Trusted"),
        answer=AsyncMock(),
    )

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), SimpleNamespace()
    )

    adapter._handle_choice_picker_callback.assert_not_awaited()
    assert "not authorized" in query.answer.call_args.kwargs["text"].lower()


def test_my_chat_member_handler_is_registered_on_each_application(monkeypatch):
    import plugins.platforms.telegram.adapter as telegram_module

    callbacks = []

    class FakeChatMemberHandler:
        MY_CHAT_MEMBER = "my-chat-member"

        def __init__(self, callback, chat_member_types=None):
            callbacks.append((callback, chat_member_types))

    monkeypatch.setattr(telegram_module, "ChatMemberHandler", FakeChatMemberHandler, raising=False)
    monkeypatch.setattr(telegram_module, "TelegramMessageHandler", Mock())
    monkeypatch.setattr(telegram_module, "CallbackQueryHandler", Mock())
    monkeypatch.setattr(
        telegram_module,
        "filters",
        SimpleNamespace(
            TEXT=MagicMock(),
            COMMAND=MagicMock(),
            LOCATION=MagicMock(),
            VENUE=MagicMock(),
            PHOTO=MagicMock(),
            VIDEO=MagicMock(),
            AUDIO=MagicMock(),
            VOICE=MagicMock(),
            Document=SimpleNamespace(ALL=MagicMock()),
            Sticker=SimpleNamespace(ALL=MagicMock()),
            StatusUpdate=SimpleNamespace(MIGRATE=MagicMock()),
        ),
    )
    adapter = _adapter(trusted=[111])
    first_app = SimpleNamespace(add_handler=Mock())
    rebuilt_app = SimpleNamespace(add_handler=Mock())

    adapter._register_handlers(first_app)
    adapter._register_handlers(rebuilt_app)

    assert callbacks == [
        (adapter._handle_my_chat_member, FakeChatMemberHandler.MY_CHAT_MEMBER),
        (adapter._handle_my_chat_member, FakeChatMemberHandler.MY_CHAT_MEMBER),
    ]
    assert telegram_module.TelegramMessageHandler.call_args_list.count(
        call(telegram_module.filters.StatusUpdate.MIGRATE, adapter._handle_chat_migration)
    ) == 2
    assert first_app.add_handler.call_count == 7
    assert rebuilt_app.add_handler.call_count == 7


@pytest.mark.asyncio
async def test_basic_group_migration_atomically_moves_active_dynamic_grant(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(
        trusted=[111],
        extra={"allowed_chats": ["-999"], "group_allowed_chats": ["-998"]},
    )
    await adapter._handle_my_chat_member(
        _membership_update(chat_id=-100, chat_type="group"), None
    )
    update = SimpleNamespace(
        effective_message=SimpleNamespace(
            chat=SimpleNamespace(id=-100, type="group"),
            migrate_to_chat_id=-1000000000100,
        )
    )

    await adapter._handle_chat_migration(update, None)

    assert adapter._telegram_auto_authorized_groups() == {"-1000000000100"}
    assert "-100" not in adapter._telegram_allowed_chats()
    assert "-1000000000100" in adapter._telegram_allowed_chats()
    assert adapter.config.extra["allowed_chats"] == ["-999"]
    assert adapter.config.extra["group_allowed_chats"] == ["-998"]


@pytest.mark.asyncio
async def test_migration_does_not_copy_explicit_or_unvalidated_persisted_grants(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_path = state_dir / "telegram-auto-authorized-groups.json"
    state_path.write_text(json.dumps({"chat_ids": ["-200"]}), encoding="utf-8")
    adapter = _adapter(extra={"allowed_chats": ["-100"]})

    explicit_update = SimpleNamespace(
        effective_message=SimpleNamespace(
            chat=SimpleNamespace(id=-100, type="group"),
            migrate_to_chat_id=-1000000000100,
        )
    )
    await adapter._handle_chat_migration(explicit_update, None)
    assert adapter._telegram_auto_authorized_groups() == {"-200"}
    assert "-1000000000100" not in adapter._telegram_allowed_chats()

    persisted_update = SimpleNamespace(
        effective_message=SimpleNamespace(
            chat=SimpleNamespace(id=-200, type="group"),
            migrate_to_chat_id=-1000000000200,
        )
    )
    await adapter._handle_chat_migration(persisted_update, None)
    assert adapter._telegram_auto_authorized_groups() == {"-1000000000200"}
    assert "-1000000000200" not in adapter._telegram_allowed_chats()


@pytest.mark.asyncio
async def test_migration_persistence_failure_revokes_old_without_activating_new(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(trusted=[111])
    await adapter._handle_my_chat_member(
        _membership_update(chat_id=-100, chat_type="group"), None
    )
    state_path = tmp_path / "state" / "telegram-auto-authorized-groups.json"

    def fail_write(_chat_ids):
        raise OSError("disk full")

    monkeypatch.setattr(adapter, "_write_telegram_auto_authorized_groups", fail_write)
    update = SimpleNamespace(
        effective_message=SimpleNamespace(
            chat=SimpleNamespace(id=-100, type="group"),
            migrate_to_chat_id=-1000000000100,
        )
    )

    await adapter._handle_chat_migration(update, None)

    assert adapter._telegram_allowed_chats() == set()
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "chat_ids": ["-100"]
    }
