"""The managed worker's display history read paints a compacted tool result once, as the archived original, like the
owner's display reads; its model history keeps the stub."""

from persistence.sqlite.workers._compression import worker_history
from tests.persistence_sqlite.conversation.test_compacted_tool_paints_once import FULL, SID, STUB, compact_with_stub, seed


def _history(holder, *, display: bool) -> list:
    payload = {"target": SID, "include_ancestors": True, "include_inactive": False, "repair_alternation": False,
               "include_row_ids": True, "include_compacted": display}
    with holder._read_ctx() as conn:
        return worker_history(holder, conn, SID, payload)["messages"]


def test_worker_display_history_paints_a_compacted_tool_result_once(backend, scope_a):
    store = backend.open(scope_a)
    seed(store)
    compact_with_stub(store, 1)

    holder = store.holder()
    assert [m["content"] for m in _history(holder, display=True) if m["role"] == "tool"] == [FULL]
    assert [m["content"] for m in _history(holder, display=False) if m["role"] == "tool"] == [STUB]
