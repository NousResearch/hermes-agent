"""Telegram observe mode: ``/resume`` in an observed group may not take over a co-member's per-user session.

Observe-mode commands carry ``shared_session=True`` so they key the shared group session (#151040).
The ``/resume`` ownership guard read that flag off the CALLER, so any member's same-chat per-user
row (pre-observe sessions, ``group_sessions_per_user`` rows) passed as "shared" (#151756). The
target's own key decides now; the shared group session itself still resumes for every member.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.gateway.test_telegram_group_gating import _bot_command_entity, _group_message, _make_adapter


def _observed_command_source(from_user_id, text="/resume@hermes_bot abc"):
    """The source the real adapter gives a ``/resume`` sent in an observed group."""
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    seen = []
    adapter.handle_message = AsyncMock(side_effect=seen.append)
    adapter._enqueue_text_event = seen.append
    adapter._ensure_forum_commands = AsyncMock()
    msg = _group_message(text, from_user_id=from_user_id,
                         entities=[_bot_command_entity(text, "/resume@hermes_bot")])
    asyncio.run(adapter._handle_command(SimpleNamespace(update_id=1, message=msg, effective_message=None), None))
    (event,) = seen
    assert event.source.shared_session is True and event.source.user_id == str(from_user_id)
    return event.source


def _runner(db=None):
    class _Runner(GatewayRunner):
        persistence = None  # the storage interface handle ``db``, set below

    runner = object.__new__(_Runner)
    runner.config = GatewayConfig()
    runner.persistence = db
    runner._gateway_session_origin_for_id = lambda _sid: None
    runner._resume_caller_is_admin = lambda _source: False
    return runner


@pytest.fixture
def db(tmp_path):
    return storage_handle(tmp_path)


def _alice_per_user_row(db):
    alice = SessionSource(platform=_observed_command_source(111).platform, chat_id="-100",
                          chat_type="group", user_id="111", user_name="Alice")
    seed_session(db, "alice_row", "telegram", user_id="111", chat_id="-100", chat_type="group",
                 session_key="agent:main:telegram:group:-100:111", origin_json=json.dumps(alice.to_dict()))
    return alice


def test_observed_command_cannot_resume_a_co_members_persisted_per_user_row(db):
    _alice_per_user_row(db)
    bob = _observed_command_source(222)
    runner = _runner(db)
    assert asyncio.run(runner._resume_target_allowed(bob, "alice_row")) is False
    # The owner keeps access to her own row.
    assert asyncio.run(runner._resume_target_allowed(_observed_command_source(111), "alice_row")) is True


def test_observed_command_cannot_resume_a_co_members_live_per_user_session(db):
    alice = _alice_per_user_row(db)
    runner = _runner(db)
    runner._gateway_session_origin_for_id = lambda _sid: alice
    assert asyncio.run(runner._resume_target_allowed(_observed_command_source(222), "alice_row")) is False


def test_observed_shared_group_session_still_resumes_for_every_member(db):
    shared = SessionSource(platform=_observed_command_source(111).platform, chat_id="-100",
                           chat_type="group", shared_session=True)
    seed_session(db, "shared_row", "telegram", user_id=None, chat_id="-100", chat_type="group",
                 session_key="agent:main:telegram:group:-100", origin_json=json.dumps(shared.to_dict()))
    runner = _runner(db)
    for member in (111, 222):
        assert asyncio.run(runner._resume_target_allowed(_observed_command_source(member), "shared_row")) is True
    runner._gateway_session_origin_for_id = lambda _sid: shared
    assert asyncio.run(runner._resume_target_allowed(_observed_command_source(222), "shared_row")) is True


def test_participant_less_row_still_fails_closed_for_a_per_user_caller(db):
    seed_session(db, "legacy_row", "telegram", user_id=None, chat_id="-100", chat_type="group",
                 session_key="agent:main:telegram:group:-100")
    caller = SessionSource(platform=_observed_command_source(111).platform, chat_id="-100",
                           chat_type="group", user_id="222")
    assert asyncio.run(_runner(db)._resume_target_allowed(caller, "legacy_row")) is False


def test_live_observe_session_stored_before_the_shared_flag_still_resumes_for_members(db):
    """An observe group's live session whose origin predates ``shared_session`` (no participant, no
    flag) stays the group's session for every observed member; a per-user caller is still refused."""
    legacy = SessionSource(platform=_observed_command_source(111).platform, chat_id="-100", chat_type="group")
    seed_session(db, "legacy_live", "telegram", user_id=None, chat_id="-100", chat_type="group",
                 session_key="agent:main:telegram:group:-100", origin_json=json.dumps(legacy.to_dict()))
    runner = _runner(db)
    runner._gateway_session_origin_for_id = lambda _sid: legacy
    for member in (111, 222):
        assert asyncio.run(runner._resume_target_allowed(_observed_command_source(member), "legacy_live")) is True
    per_user = SessionSource(platform=legacy.platform, chat_id="-100", chat_type="group", user_id="222")
    assert asyncio.run(runner._resume_target_allowed(per_user, "legacy_live")) is False
