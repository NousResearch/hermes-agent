"""A pre-gateway chat adopted at its compression tip still answers to the id an editor holds.

``sessions adopt`` binds a pre-gateway chat at its compression TIP only — the receipt's lineage is
``[tip]``, because ``validate_local_lineage`` forces ``lineage[0] == legacy_lineage_root()`` and that
root is the tip itself for a legacy receipt. An editor that talked to the pre-gateway ACP server
holds the chain's CREATION id, so resolving the owner at the root alone answers ``not_found`` and
the session never reopens.
"""
import json

from acp_adapter.catalog import logical_session_id
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import policy_key, raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


def _adopted_chain(store, cwd):
    """root —compressed→ mid —compressed→ tip, with one receipt bound at the tip."""
    for sid, parent in (("chain-root", None), ("chain-mid", "chain-root"), ("chain-tip", "chain-mid")):
        extra = {} if parent is None else {
            "parent_session_id": parent, "model_config": {"_compressed_from": parent}}
        seed_session(store, sid, "acp", cwd=str(cwd), **extra)
        seed_messages(store, sid, {"role": "user", "content": sid})
        if sid != "chain-tip":
            store.sessions.end(sid, "compression")
    raw_holder(store)._execute_write(lambda conn: conn.execute(
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
        (policy_key("chain-tip"),
         json.dumps({"session_id": "chain-tip", "lineage": ["chain-tip"]}))))
    return "chain-root", "chain-mid", "chain-tip"


def test_every_id_in_an_adopted_chain_resolves_to_the_owner(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        root, mid, tip = _adopted_chain(store, tmp_path)
        # The editor's id really is the root of a real compression chain (not a reset fork).
        assert store.lineage.lineage(root) == (root, mid, tip)
    for held in (root, mid, tip):
        assert logical_session_id(tmp_path, held) == tip, held


def test_a_chain_nobody_adopted_resolves_to_its_creation_id(tmp_path):
    """Without a receipt the tip-first walk finds nothing and the compression root stands."""
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "orphan", "acp", cwd=str(tmp_path))
        seed_messages(store, "orphan", {"role": "user", "content": "hi"})
        store.sessions.end("orphan", "compression")
        seed_session(store, "orphan-tip", "acp", cwd=str(tmp_path),
                     parent_session_id="orphan", model_config={"_compressed_from": "orphan"})
    assert logical_session_id(tmp_path, "orphan") == "orphan"
    assert logical_session_id(tmp_path, "orphan-tip") == "orphan"
