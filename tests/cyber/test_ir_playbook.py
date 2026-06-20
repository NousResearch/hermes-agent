"""Unit tests for the IR playbook tool (no network required)."""
from __future__ import annotations

import json
import sys
import types

for mod in ("tools.registry",):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.registry = types.SimpleNamespace(register=lambda **_kw: None)
        stub.tool_error = lambda msg: msg
        sys.modules[mod] = stub

from tools.cyber_ir_playbook import IRStore, _handle  # noqa: E402


def _store() -> IRStore:
    return IRStore()


class TestIRStore:
    def test_create_returns_incident_id(self) -> None:
        store = _store()
        result = store.create("Test Incident", "P2")
        assert result["created"].startswith("INC-")

    def test_create_invalid_severity(self) -> None:
        store = _store()
        result = store.create("Bad sev", "P9")
        assert "error" in result

    def test_sequential_ids(self) -> None:
        store = _store()
        a = store.create("Inc A", "P1")
        b = store.create("Inc B", "P3")
        assert a["created"] != b["created"]

    def test_update_status(self) -> None:
        store = _store()
        inc_id = store.create("Test", "P2")["created"]
        result = store.update(inc_id, status="investigating")
        assert result["incident"]["status"] == "investigating"

    def test_update_invalid_status(self) -> None:
        store = _store()
        inc_id = store.create("Test", "P2")["created"]
        result = store.update(inc_id, status="hacked")
        assert "error" in result

    def test_update_unknown_incident(self) -> None:
        store = _store()
        result = store.update("INC-9999", status="closed")
        assert "error" in result

    def test_add_timeline(self) -> None:
        store = _store()
        inc_id = store.create("Test", "P2")["created"]
        result = store.add_timeline(inc_id, "Alert fired", actor="soc-analyst")
        assert result["timeline_length"] == 1

    def test_add_evidence_ioc(self) -> None:
        store = _store()
        inc_id = store.create("Test", "P1")["created"]
        result = store.add_evidence(inc_id, "ioc", "203.0.113.42", source="firewall")
        assert result["evidence_count"] == 1

    def test_add_evidence_unknown_type_coerced(self) -> None:
        store = _store()
        inc_id = store.create("Test", "P3")["created"]
        result = store.add_evidence(inc_id, "something_weird", "value")
        assert result["evidence_count"] == 1  # coerced to "other"

    def test_list_all_empty(self) -> None:
        store = _store()
        result = store.list_all()
        assert result["total"] == 0
        assert result["incidents"] == []

    def test_list_all_after_create(self) -> None:
        store = _store()
        store.create("Inc A", "P1")
        store.create("Inc B", "P2")
        result = store.list_all()
        assert result["total"] == 2

    def test_status_returns_full_incident(self) -> None:
        store = _store()
        inc_id = store.create("Full", "P2", description="desc")["created"]
        status = store.get_status(inc_id)
        assert status["title"] == "Full"
        assert status["description"] == "desc"
        assert "timeline" in status
        assert "evidence" in status

    def test_report_contains_markdown_header(self) -> None:
        store = _store()
        inc_id = store.create("Ransomware", "P1")["created"]
        store.add_timeline(inc_id, "Encrypted files detected", actor="EDR")
        store.add_evidence(inc_id, "ioc", "192.0.2.99", source="SIEM")
        result = store.report(inc_id)
        md = result["report_markdown"]
        assert "# Incident Report:" in md
        assert "192.0.2.99" in md
        assert "Encrypted files detected" in md

    def test_report_unknown_incident(self) -> None:
        store = _store()
        result = store.report("INC-9999")
        assert "error" in result

    def test_assignee_deduplication(self) -> None:
        store = _store()
        inc_id = store.create("Test", "P2")["created"]
        store.update(inc_id, assignee="alice")
        store.update(inc_id, assignee="alice")
        status = store.get_status(inc_id)
        assert status["assignees"].count("alice") == 1

    def test_tags_merged(self) -> None:
        store = _store()
        inc_id = store.create("Test", "P3")["created"]
        store.update(inc_id, tags=["phishing", "bec"])
        store.update(inc_id, tags=["bec", "finance"])
        status = store.get_status(inc_id)
        assert set(status["tags"]) == {"phishing", "bec", "finance"}


class TestIRHandler:
    def _make_store_kw(self) -> dict:
        class FakeAgent:
            pass
        agent = FakeAgent()
        agent._ir_store = IRStore()
        return {"store": agent._ir_store}

    def test_create_via_handler(self) -> None:
        kw = self._make_store_kw()
        out = json.loads(_handle({"action": "create", "title": "Test", "severity": "P2"}, **kw))
        assert "created" in out

    def test_list_via_handler(self) -> None:
        kw = self._make_store_kw()
        json.loads(_handle({"action": "create", "title": "Inc", "severity": "P1"}, **kw))
        out = json.loads(_handle({"action": "list"}, **kw))
        assert out["total"] == 1

    def test_unknown_action_returns_error(self) -> None:
        kw = self._make_store_kw()
        out = json.loads(_handle({"action": "explode"}, **kw))
        assert "error" in out
        assert "valid_actions" in out

    def test_missing_store_returns_error(self) -> None:
        out = json.loads(_handle({"action": "list"}))
        assert "error" in out
