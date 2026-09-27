import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from plugins.subagent_progress import ProgressPlugin


def test_notice_uses_pinned_route_after_parent_turn_ends(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gateway.run", SimpleNamespace(_redact_gateway_user_facing_secrets=lambda s: s))
    plugin = ProgressPlugin(None, tmp_path)
    source = SimpleNamespace(chat_id="own-chat", thread_id="own-topic")
    adapter = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True, message_id="receipt-7")))
    runner = SimpleNamespace(
        async_session_store=SimpleNamespace(lookup_by_session_key=AsyncMock(return_value=SimpleNamespace(
            session_id="parent", origin=source))),
        _is_user_authorized_for_source=lambda source, **kw: True,
        _restored_source=lambda entry: entry.origin,
        _delivery_adapter_for=lambda source: adapter,
        _thread_metadata_for_source=lambda source: {"thread_id": source.thread_id})
    owner = {"route": "fixed-route", "parent": "parent"}
    payload = {"subagent_id": "s", "completed": "checked", "next_step": "test", "evidence": [],
               "blocker": "", "needs_decision": False}
    assert asyncio.run(plugin.deliver_notice(runner, owner, payload, 7))
    assert adapter.send.call_args.args[0] == "own-chat"
    assert adapter.send.call_args.kwargs["metadata"]["thread_id"] == "own-topic"
    with plugin.db() as db:
        receipt = json.loads(db.execute("SELECT receipt FROM deliveries WHERE report=7").fetchone()[0])
    assert receipt == {"success": True, "message_id": "receipt-7"}
    runner.async_session_store.lookup_by_session_key.return_value.session_id = "new-parent-after-reset"
    payload.update(needs_decision=True, blocker="choose")
    plugin.ctx = SimpleNamespace(inject_message=lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("reset route must never be awakened")))
    assert not asyncio.run(plugin.deliver_notice(runner, owner, payload, 8))
    assert adapter.send.call_count == 1
    # Compression remains the same conversation; a reset does not.
    runner._session_db = object()
    runner._resolve_compression_lineage_target = AsyncMock(return_value="compressed-parent")
    runner.async_session_store.lookup_by_session_key.return_value.session_id = "compressed-parent"
    wakes = []
    plugin.ctx = SimpleNamespace(inject_message=lambda *a, **kw: wakes.append(kw) or True)
    assert asyncio.run(plugin.deliver_notice(runner, owner, payload, 9))
    # Display and wake now have independent receipts; display never injects.
    assert wakes == []
