"""A stored receipt's lineage survives the receipt read and the reservation's exact replay unchanged, on SQLite.

The receipt is
stored as written (an empty lineage and a lineage with an ancestor before the creation id), ``receipt`` reads that
lineage back, an exact ``reserve`` replays the stored receipt without inserting a row, and the stored bytes stay as
they were.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from tests.contracts.authority.test_bindings_contract import _kwargs, make_binding


@pytest.mark.parametrize("lineage", [[], ["ancestor", "s-bind-1"]])
def test_binding_receipt_and_reservation_replay_keep_stored_lineage(backend, scope_a, lineage):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    binding = make_binding()
    stored = {"profile_id": "default", **_kwargs(binding), "lineage": lineage}
    encoded = json.dumps(stored)
    key = "gateway.local_policy.v1:" + binding.session_id
    store.state_meta.set(key, encoded)

    receipt = store.bindings.receipt(binding.session_id)
    replay = store.bindings.reserve(epoch=epoch, binding=binding, owner=None)

    assert receipt.lineage == tuple(lineage)
    assert replay.binding.lineage == tuple(lineage)
    assert replay.created is False
    assert store.state_meta.get(key) == encoded
    with sqlite3.connect(store.store_path) as conn:
        assert conn.execute("SELECT 1 FROM sessions WHERE id=?", (binding.session_id,)).fetchone() is None
