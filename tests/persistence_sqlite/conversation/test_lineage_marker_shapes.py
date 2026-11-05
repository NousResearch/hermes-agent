"""The SQLite record decoding of stored fork-marker shapes, without a database."""

import pytest

from persistence.conversation.sessions import SessionLineageKind
from persistence.sqlite.conversation.sessions import session_record


@pytest.mark.parametrize("marker, kind", [
    ("_delegate_from", SessionLineageKind.DELEGATE),
    ("_branched_from", SessionLineageKind.BRANCH),
    ("_reset_from", SessionLineageKind.RESET),
])
@pytest.mark.parametrize("value, origin", [
    (None, None), (False, None), (True, None), (42, None), (["parent"], None),
    ({"id": "parent"}, None), ("", None), (" \t\n", None), ("  parent \t", "  parent \t"),
])
def test_lineage_markers_require_nonblank_strings(marker, kind, value, origin):
    record = session_record({"id": "child", "started_at": 1.0, "parent_session_id": "parent",
                             "model_config": {marker: value, "temperature": 0.1}})
    assert record.lineage_kind is (kind if origin else SessionLineageKind.NONE)
    assert record.lineage_origin_id == origin
    assert dict(record.model_config) == {marker: value, "temperature": 0.1}


def test_invalid_higher_priority_marker_does_not_mask_a_valid_marker():
    record = session_record({"id": "child", "started_at": 1.0,
                             "model_config": {"_delegate_from": True, "_branched_from": " ", "_reset_from": " root "}})
    assert (record.lineage_kind, record.lineage_origin_id) == (SessionLineageKind.RESET, " root ")
