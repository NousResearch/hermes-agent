"""Tests for ACP session-provenance derivation (issue #33617).

Exercises acp_adapter.provenance against a real SessionDB — no mocks — covering
the acceptance-criteria matrix: root session, compression-split continuation,
multi-depth chains, rotation flagging, and graceful handling of unknown ids.
"""

import time

import pytest

from acp_adapter.provenance import build_session_provenance, session_provenance_meta
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d


def _mk(db, sid, parent=None):
    db.create_session(session_id=sid, source="acp", parent_session_id=parent)


def test_root_session_no_compression(db):
    _mk(db, "root1")
    prov = build_session_provenance(db, "acp-1", "root1")
    assert prov["acpSessionId"] == "acp-1"
    assert prov["currentHermesSessionId"] == "root1"
    assert prov["rootHermesSessionId"] == "root1"
    assert prov["parentHermesSessionId"] is None
    assert prov["sessionKind"] == "root"
    assert prov["compressionDepth"] == 0
    assert "reason" not in prov  # no rotation signalled


def test_compression_split_continuation(db):
    # Parent ended with compression, child created afterwards.
    _mk(db, "old")
    db.end_session("old", "compression")
    time.sleep(0.001)
    _mk(db, "new", parent="old")

    prov = build_session_provenance(
        db, "acp-1", "new", previous_hermes_session_id="old"
    )
    assert prov["sessionKind"] == "continuation"
    assert prov["parentHermesSessionId"] == "old"
    assert prov["rootHermesSessionId"] == "old"
    assert prov["compressionDepth"] == 1
    assert prov["previousHermesSessionId"] == "old"
    # Head rotated this turn → reason/creatorKind flagged.
    assert prov["reason"] == "compression"
    assert prov["creatorKind"] == "compression"




def test_non_compression_parent_is_root_not_continuation(db):
    # A child with a parent that did NOT end via compression (e.g. delegate
    # or branch child) must not be reported as a compression continuation.
    _mk(db, "p")
    _mk(db, "c", parent="p")  # parent still live, no end_reason
    prov = build_session_provenance(db, "acp-1", "c")
    assert prov["sessionKind"] == "root"
    assert prov["compressionDepth"] == 0
    assert prov["rootHermesSessionId"] == "p"  # lineage root still walked






def test_meta_wrapper_shape(db):
    _mk(db, "root1")
    meta = session_provenance_meta(db, "acp-1", "root1")
    assert set(meta.keys()) == {"hermes"}
    assert "sessionProvenance" in meta["hermes"]
    assert meta["hermes"]["sessionProvenance"]["currentHermesSessionId"] == "root1"
