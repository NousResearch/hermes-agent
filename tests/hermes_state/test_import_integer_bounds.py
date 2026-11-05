"""Invalid persisted integers are handled before a transcript import writes rows."""

import json

import pytest

from hermes_state import SessionDB
from hermes_state_portability import _IMPORT_INT_COLS


@pytest.mark.parametrize("number", ["1e309", str(2**63)])
def test_import_rejects_unstorable_message_tokens_without_partial_rows(tmp_path, number):
    payload = json.loads(
        '[{"id":"valid","messages":[{"role":"user","content":"keep"}]},'
        '{"id":"invalid","messages":[{"role":"user","content":"bad",'
        '"token_count":' + number + '}]}]'
    )
    with SessionDB(tmp_path / "state.db") as db:
        result = db.import_sessions(payload)
        assert not result["ok"]
        assert result["imported"] == 0
        assert result["errors"][0]["session_id"] == "invalid"
        assert "token_count" in result["errors"][0]["error"]
        assert db.get_session("valid") is None
        assert db.get_session("invalid") is None

        payload[1]["messages"][0]["token_count"] = 2**63 - 1
        assert db.import_sessions(payload)["imported"] == 2
        assert db.get_messages("invalid")[0]["token_count"] == 2**63 - 1


def test_import_defaults_unstorable_session_counters(tmp_path):
    payload = {"id": "recovered", **dict.fromkeys(_IMPORT_INT_COLS, 2**63)}
    payload["output_tokens"] = float("inf")
    with SessionDB(tmp_path / "state.db") as db:
        assert db.import_sessions([payload])["ok"]
        stored = db.get_session("recovered")
        assert all(stored[field] == 0 for field in _IMPORT_INT_COLS)
