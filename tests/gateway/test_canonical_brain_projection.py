import json
from gateway.canonical_brain_projection import fold_case_events
from tools import canonical_brain_tool as cbt


class _Sock:
    def close(self):
        pass


class _Helper:
    def __init__(self, rows):
        self.rows = rows
        self.sql = ""

    def get_secret_value(self):
        return "secret-handle-value"

    def connect(self, password):
        return _Sock()

    def sql_quote(self, value):
        return "'" + str(value).replace("'", "''") + "'"

    def query(self, sock, sql):
        self.sql = sql
        return {"rows": self.rows}


def _row(event_type, occurred_at, *, case_id="case:1", target=None, receipt=None):
    payload = {"summary": event_type}
    if event_type.startswith("route_back."):
        payload["route_back"] = {"target_ref": target or {}}
        if receipt:
            payload["receipt"] = receipt
    return {
        "event_id": event_type + occurred_at,
        "event_type": event_type,
        "case_id": case_id,
        "occurred_at": occurred_at,
        "source": {"source_refs": {"platform": "discord", "thread_id": "source", "message_id": occurred_at}},
        "status": {"state": event_type, "summary": event_type},
        "next_action": {},
        "payload": payload,
    }


def test_fold_uses_explicit_event_order_not_text_keywords():
    cases = fold_case_events([
        _row("route_back.required", "2026-01-01T00:00:00Z", target={"thread_id": "resolver"}),
        _row("case.note", "2026-01-01T00:01:00Z"),
        _row("route_back.sent", "2026-01-01T00:02:00Z", target={"thread_id": "requester"}, receipt={"message_id": "m1"}),
    ])
    assert cases[0]["latest_event_type"] == "route_back.sent"
    assert cases[0]["route_back"]["terminal"] is True
    assert cases[0]["route_back"]["target_ref"] == {"thread_id": "requester"}
    assert cases[0]["route_back"]["receipt"] == {"message_id": "m1"}


def test_fold_decodes_cloud_sql_jsonb_strings_mechanically():
    row = _row(
        "route_back.sent",
        "2026-01-01T00:02:00Z",
        target={"thread_id": "requester"},
        receipt={"message_id": "m1"},
    )
    for field in ("source", "status", "next_action", "payload"):
        row[field] = json.dumps(row[field])

    case = fold_case_events([row])[0]

    assert case["status"]["state"] == "route_back.sent"
    assert case["source_refs"][0]["thread_id"] == "source"
    assert case["route_back"]["target_ref"] == {"thread_id": "requester"}
    assert case["route_back"]["receipt"] == {"message_id": "m1"}


def test_query_requires_exact_case_or_thread(monkeypatch):
    helper = _Helper([_row("case.note", "2026-01-01T00:00:00Z")])
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)
    data = json.loads(cbt.canonical_brain_query_tool(case_id="case:1"))
    assert data["success"] is True
    assert data["case_count"] == 1
    assert "case_id = 'case:1'" in helper.sql

    invalid = json.loads(cbt.canonical_brain_query_tool(case_id="case:1", thread_id="thread"))
    assert "provide exactly one" in invalid["error"]
