"""Contract cases for a damaged ``model_config`` cell: the record reads it as ``{}`` and says so with ``model_config_corrupt``, so a caller that must not overwrite a cell it cannot read can skip the write.

The corrupt rule is a strict JSON reader's: non-blank text that does not decode. ``NULL``, blank text and JSON that is not an object read ``{}`` and are not corrupt. Cells are planted through the backend's raw row seam, bypassing every typed write path.
"""

from __future__ import annotations

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession, is_corrupt_model_config

SID = SessionId("mc-cell")


@pytest.mark.parametrize(
    ("cell", "corrupt", "config"),
    [
        ("{not json", True, {}),
        ('{"model": "x"', True, {}),
        (None, False, {}),
        ("", False, {}),
        ("   ", False, {}),
        ("[1, 2]", False, {}),
        ("null", False, {}),
        ('{"model": "x"}', False, {"model": "x"}),
    ],
)
def test_the_record_says_whether_the_stored_cell_decodes(backend, scope_a, cell, corrupt, config):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SID, source="cli", model_config={"model": "before"}))
    backend.scope_store(scope_a)["sessions"][SID]["model_config"] = cell
    record = store.sessions.get(SID)
    # CONTRACT: a corrupt cell is flagged; model_config keeps its tolerant read.
    assert record.model_config_corrupt is corrupt
    assert dict(record.model_config) == config


def test_rows_written_through_the_verbs_are_not_corrupt(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SID, source="cli"))
    assert store.sessions.get(SID).model_config_corrupt is False
    store.sessions.patch_model_config(SID, {"model": "m"})
    record = store.sessions.get(SID)
    assert record.model_config_corrupt is False and dict(record.model_config) == {"model": "m"}


@pytest.mark.parametrize(
    ("raw", "corrupt"),
    [("{", True), ("nope", True), (None, False), ("", False), (" \n", False), ("[]", False), ('{"a": 1}', False), ({"a": 1}, False)],
)
def test_the_corrupt_rule(raw, corrupt):
    assert is_corrupt_model_config(raw) is corrupt
