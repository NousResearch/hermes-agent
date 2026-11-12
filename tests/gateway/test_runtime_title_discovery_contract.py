"""Bad titles refuse through the actual authority without partial metadata writes."""
import pytest

from tests.gateway.test_messaging_session_picks import _compressed_chat, _mutate


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["rename", "sidebar"])
@pytest.mark.parametrize("length", [100, 101])
async def test_authority_title_discovery_keeps_validation_contract(tmp_path, monkeypatch, operation, length):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _source, _route, viewer, root, fences = await _compressed_chat(tmp_path)
    try:
        db = runner.session_store._db
        def snapshot():
            return {table: [tuple(row) for row in db._conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
                    for table in ("sessions", "messages", "state_meta", "gateway_routing")}
        before = snapshot()
        payload = {"title": "x" * length}
        if operation == "sidebar":
            payload["pinned"] = True
        reply = await viewer.dispatch(_mutate(root, fences, operation, payload, "title-boundary"))
        if length == 101:
            assert reply.get("error", {}).get("data", {}).get("reason") == "invalid_params", reply
            assert snapshot() == before
        else:
            assert reply.get("result", {}).get("title") == "x" * length, reply
            assert any(row[0] == "x" * length for row in db._conn.execute("SELECT title FROM sessions"))
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
