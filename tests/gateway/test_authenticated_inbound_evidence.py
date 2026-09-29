"""Isolated ingress -> runner thread -> real SQLite canary; no remote memory writes."""
import asyncio
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.session_persistence import _db_flush_row
from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from hermes_inbound_evidence import read_authenticated_inbound, reset_admission
from hermes_state import SessionDB
from plugins.platforms.feishu.adapter import FeishuAdapter


async def incoming(text="canary preference", sender_type="user", message_id="m1"):
    adapter = object.__new__(FeishuAdapter)
    adapter.platform = Platform.FEISHU
    adapter.config = PlatformConfig(extra={})
    adapter.gateway_runner = None
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "bot", "", "Hermes"
    adapter._fetch_message_text = AsyncMock(return_value=None)
    adapter.get_chat_info = AsyncMock(return_value={"name": "canary", "type": "private"})
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_id": "u1", "user_name": "canary", "user_id_alt": None})
    adapter._dispatch_inbound_event = AsyncMock()
    sender_id = SimpleNamespace(open_id="u1")
    message = SimpleNamespace(content=json.dumps({"text": text}), message_type="text", mentions=[],
                              chat_id="c1", message_id=message_id, chat_type="p2p")
    data = SimpleNamespace(event=SimpleNamespace(message=message, sender=SimpleNamespace(sender_type=sender_type, sender_id=sender_id)))
    await adapter._process_inbound_message(data=data, message=message, sender_id=sender_id,
                                          chat_type="p2p", message_id=message_id, is_bot=sender_type == "bot")
    return adapter, adapter._dispatch_inbound_event.call_args.args[0]


class Ingress(GatewayInboundMixin):
    def __init__(self, home, adapter, authorized=True):
        self.home, self.adapter, self.authorized = home, adapter, authorized
    def _scale_to_zero_note_real_inbound(self): pass
    async def _hm_pre_gateway_dispatch_hook(self, event, source): return event
    def _is_user_authorized_for_source(self, source): return self.authorized
    def _admit_bot_message_for_source(self, source): return True
    def _intake_adapter_for(self, source): return self.adapter
    def _resolve_profile_home_for_source(self, source): return self.home
    def _session_key_for_source(self, source): return "agent:main:feishu:dm:c1"
    def _get_unauthorized_dm_behavior(self, *args, **kwargs): return "ignore"
    async def _hm_report_ignored_dm(self, source): pass
    def _get_executor(self): return None
    _run_in_executor_with_context = GatewayRunner._run_in_executor_with_context


def store(home):
    home.mkdir(exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    db.create_session("s1", "feishu", user_id="u1", chat_id="c1", chat_type="dm",
                      session_key="agent:main:feishu:dm:c1")
    return db


def persist_through_runner(db, event, *, row_change=None):
    runner = object.__new__(TurnRunner)
    runner._ctx = TurnContext(source=event.source, session_id="s1", session_key="agent:main:feishu:dm:c1",
                              message=event.text, inbound_message_id=event.message_id)
    runner._native_image_run_message = lambda: event.text
    agent = SimpleNamespace()
    def run(message, **kwargs):
        row = _db_flush_row(agent, {"role": "user", "content": message,
                                    "platform_message_id": kwargs["persist_user_platform_id"]}, True)
        if row_change:
            row.update(row_change)
        db.append_messages_batch("s1", [row])
        return row["_row_id"]
    agent.run_conversation = run
    return runner._run_conversation_with_approval(agent, [], None, None, None)


def test_committed_binding_survives_thread_hop_and_is_profile_scoped(tmp_path, monkeypatch):
    async def scenario():
        homes = [tmp_path / "a", tmp_path / "b"]
        stores = [store(home) for home in homes]
        set_multiplex_active(True)
        try:
            for n, index in enumerate((0, 1, 0)):
                home, db = homes[index], stores[index]
                with _profile_runtime_scope(home):
                    adapter, event = await incoming(message_id=f"m{n}")
                    ingress = Ingress(home, adapter)
                    assert await ingress._hm_admit_event(event)
                    # Failure after both row and binding were written rolls back both.
                    original = db._bump_session_counters
                    before = len(db.get_messages("s1"))
                    def fail(*args, **kwargs): raise RuntimeError("rollback canary")
                    with monkeypatch.context() as patch:
                        patch.setattr(db, "_bump_session_counters", fail)
                        with pytest.raises(RuntimeError, match="rollback canary"):
                            await ingress._run_in_executor_with_context(persist_through_runner, db, event)
                    assert db._bump_session_counters == original
                    assert len(db.get_messages("s1")) == before
                    row_id = await ingress._run_in_executor_with_context(persist_through_runner, db, event)
                    proof = read_authenticated_inbound(home, "s1", row_id)
                    assert proof and proof["content"] == event.text and proof["user_id"] == "u1"
                    assert proof["platform_message_id"] == f"m{n}"
                    db._write_sql("UPDATE messages SET content=? WHERE id=?", ("tampered", row_id))
                    assert read_authenticated_inbound(home, "s1", row_id) is None
                    db._write_sql("UPDATE messages SET content=? WHERE id=?", (event.text, row_id))
                    assert read_authenticated_inbound(home, "wrong-session", row_id) is None
                    # Same authenticated context cannot bind a row in the other profile.
                    foreign = await ingress._run_in_executor_with_context(persist_through_runner, stores[1-index], event)
                    assert read_authenticated_inbound(homes[1-index], "s1", foreign) is None
                    # Replayed event to a new row revokes its prior evidence too.
                    duplicate = await ingress._run_in_executor_with_context(persist_through_runner, db, event)
                    assert read_authenticated_inbound(home, "s1", duplicate) is None
                    assert read_authenticated_inbound(home, "s1", row_id) is None
        finally:
            reset_admission()
            set_multiplex_active(False)
            for db in stores: db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["unauthorized", "bot", "unknown-human", "internal", "rewritten", "forged", "row-content", "row-id", "assistant", "summary", "hidden"])
def test_untrusted_or_changed_input_never_attests(tmp_path, case):
    async def scenario():
        db = store(tmp_path / "home")
        try:
            adapter, event = await incoming(sender_type="bot" if case == "bot" else "unknown" if case == "unknown-human" else "user")
            if case == "internal": event.internal = True
            if case == "rewritten": event.text += " plugin rewrite"
            if case == "forged":
                event = MessageEvent(text=event.text, source=event.source, message_id=event.message_id,
                                     metadata={"_human_transport_receipt": {"human": True}})
            ingress = Ingress(tmp_path / "home", adapter, authorized=case != "unauthorized")
            await ingress._hm_admit_event(event)
            changes = {"row-content": {"content": "changed"}, "row-id": {"platform_message_id": "other"},
                       "assistant": {"role": "assistant"}, "summary": {"_compressed_summary": True},
                       "hidden": {"display_kind": "internal_notification"}}
            row_id = await ingress._run_in_executor_with_context(lambda: persist_through_runner(db, event, row_change=changes.get(case)))
            assert read_authenticated_inbound(tmp_path / "home", "s1", row_id) is None
        finally:
            reset_admission()
            db.close()
    asyncio.run(scenario())


def test_evidence_guard_serializes_revocation_and_releases_on_failure(tmp_path):
    from hermes_inbound_evidence import authenticated_inbound_guard
    async def scenario():
        home = tmp_path / 'guard'
        db = store(home)
        try:
            adapter, event = await incoming()
            ingress = Ingress(home, adapter)
            assert await ingress._hm_admit_event(event)
            row_id = await ingress._run_in_executor_with_context(persist_through_runner, db, event)
            with pytest.raises(RuntimeError, match='consumer failure'):
                with authenticated_inbound_guard(home, 's1', row_id) as proof:
                    assert proof['message_row_id'] == row_id
                    with closing(sqlite3.connect(home / 'state.db', timeout=0.05)) as rival:
                        with pytest.raises(sqlite3.OperationalError, match='locked'):
                            rival.execute('UPDATE authenticated_inbound_v1 SET revoked=1 WHERE message_row_id=?', (row_id,))
                    assert read_authenticated_inbound(home, 's1', row_id) is not None
                    raise RuntimeError('consumer failure')
            db._write_sql('UPDATE authenticated_inbound_v1 SET revoked=1 WHERE message_row_id=?', (row_id,))
            with pytest.raises(ValueError, match='revoked'):
                with authenticated_inbound_guard(home, 's1', row_id):
                    pytest.fail('Revoked source cannot enter consumer operation')
        finally:
            reset_admission()
            db.close()
    asyncio.run(scenario())


def test_current_event_lookup_requires_exact_admission_and_commit(tmp_path):
    from hermes_inbound_evidence import authenticated_turn, current_authenticated_inbound
    async def scenario():
        home = tmp_path / 'current'
        db = store(home)
        try:
            assert current_authenticated_inbound(home) is None
            adapter, event = await incoming()
            ingress = Ingress(home, adapter)
            await ingress._hm_admit_event(event)
            with authenticated_turn('s1', ingress._session_key_for_source(event.source)):
                assert current_authenticated_inbound(home) is None
                row_id = persist_through_runner(db, event)
                assert current_authenticated_inbound(home, 's1')['message_row_id'] == row_id
                assert current_authenticated_inbound(home, 'other') is None
                assert current_authenticated_inbound(tmp_path / 'other') is None
            assert current_authenticated_inbound(home) is None
        finally:
            reset_admission()
            db.close()
    asyncio.run(scenario())
