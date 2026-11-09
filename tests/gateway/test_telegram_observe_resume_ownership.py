"""Telegram observe mode: ``/resume`` in an observed group may not take over a co-member's per-user session.

Observe-mode commands carry ``shared_session=True`` so they key the shared group session (#151040).
The ``/resume`` ownership guards read that flag off the CALLER, so any member's same-chat per-user
session (pre-observe sessions, ``group_sessions_per_user`` rows) passed as "shared" (#151756). The
target's own key decides now; the shared group session itself still resumes for every member.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_state import AsyncSessionDB, SessionDB
from tests.gateway.test_telegram_group_gating import _bot_command_entity, _group_message, _make_adapter


def _observed_command_source(from_user_id, thread_id=None):
    """The source the real adapter gives a ``/resume`` sent in an observed group."""
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    seen = []
    adapter.handle_message = AsyncMock(side_effect=seen.append)
    adapter._enqueue_text_event = seen.append
    adapter._ensure_forum_commands = AsyncMock()
    text = "/resume@hermes_bot abc"
    msg = _group_message(text, from_user_id=from_user_id, thread_id=thread_id,
                         entities=[_bot_command_entity(text, "/resume@hermes_bot")])
    asyncio.run(adapter._handle_command(SimpleNamespace(update_id=1, message=msg, effective_message=None), None))
    (event,) = seen
    assert event.source.shared_session is True and event.source.user_id == str(from_user_id)
    return event.source


def _src(user_id=None, *, thread_id=None, shared_session=False):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id=user_id,
                         thread_id=thread_id, shared_session=shared_session)


def _runner(db, *, config=None, live=None):
    runner = object.__new__(GatewayRunner)
    runner.config = config or GatewayConfig()
    runner._session_db = AsyncSessionDB(db)
    runner._gateway_session_origin_for_id = lambda _session_id: live
    return runner


def _store(db, session_id, origin, config=None):
    config = config or GatewayConfig()
    key = build_session_key(origin, group_sessions_per_user=config.group_sessions_per_user,
                            thread_sessions_per_user=config.thread_sessions_per_user)
    db.create_session(session_id, "telegram", user_id=origin.user_id, chat_id="-100", chat_type="group",
                      thread_id=origin.thread_id, session_key=key, origin_json=json.dumps(origin.to_dict()))


def _allowed(runner, caller, session_id):
    return asyncio.run(runner._resume_target_allowed(caller, session_id))


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    database.close()


@pytest.mark.parametrize("live", [False, True], ids=["persisted", "live"])
def test_observed_command_cannot_resume_a_co_members_per_user_session(db, live):
    alice = _src("111")
    _store(db, "alice_row", alice)
    runner = _runner(db, live=alice if live else None)
    assert _allowed(runner, _observed_command_source(222), "alice_row") is False
    assert _allowed(runner, _observed_command_source(111), "alice_row") is True


@pytest.mark.parametrize("live", [False, True], ids=["persisted", "live"])
def test_observed_command_cannot_resume_a_co_members_per_user_thread(db, live):
    config = GatewayConfig(thread_sessions_per_user=True)
    alice = _src("111", thread_id="42")
    _store(db, "alice_thread", alice, config)
    runner = _runner(db, config=config, live=alice if live else None)
    assert _allowed(runner, _observed_command_source(222, thread_id=42), "alice_thread") is False


@pytest.mark.parametrize("live", [False, True], ids=["persisted", "live"])
def test_shared_targets_still_resume_for_every_member(db, live):
    shared = _src(shared_session=True)
    _store(db, "shared_row", shared)
    runner = _runner(db, live=shared if live else None)
    for caller in (_observed_command_source(111), _observed_command_source(222), _src("333")):
        assert _allowed(runner, caller, "shared_row") is True


@pytest.mark.parametrize("live", [False, True], ids=["persisted", "live"])
def test_pre_flag_participant_less_session_resumes_only_for_shared_callers(db, live):
    # Observe sessions stored before #151040 had no participant and no shared_session flag.
    legacy = _src()
    _store(db, "legacy_row", legacy)
    runner = _runner(db, live=legacy if live else None)
    assert _allowed(runner, _observed_command_source(222), "legacy_row") is True
    assert _allowed(runner, _src("222"), "legacy_row") is False


@pytest.mark.parametrize("live", [False, True], ids=["persisted", "live"])
def test_config_shared_thread_matches_the_session_key(db, live):
    # group_sessions_per_user=False keys every group and thread without a participant, even when
    # thread_sessions_per_user=True: the guard must agree with build_session_key.
    config = GatewayConfig(group_sessions_per_user=False, thread_sessions_per_user=True)
    alice = _src("111", thread_id="42")
    _store(db, "thread_row", alice, config)
    runner = _runner(db, config=config, live=alice if live else None)
    assert _allowed(runner, _src("222", thread_id="42"), "thread_row") is True
