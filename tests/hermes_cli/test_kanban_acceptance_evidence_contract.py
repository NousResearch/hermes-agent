"""Executable contract checks for the Kanban acceptance-evidence schema."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_acceptance_evidence import EvidenceValidationError, normalize_contract


_SCHEMA = Path(__file__).parents[2] / "website" / "docs" / "user-guide" / "features" / "acceptance-evidence.schema.json"


def _valid_contract() -> dict:
    return {
        "version": 1,
        "required": [
            {"id": "unit-tests", "kind": "check", "description": "Focused tests pass"},
            {"id": "canary", "kind": "canary", "description": "Live worker proof"},
        ],
        "observed": [
            {
                "id": "unit-tests",
                "status": "passed",
                "observed_at": 1789400001,
                "source": "pytest",
                "payload": {"command": "pytest tests/hermes_cli/test_kanban_acceptance_evidence_contract.py -q", "exit_code": 0},
            },
            {
                "id": "canary",
                "status": "passed",
                "observed_at": 1789400003,
                "source": "dispatcher",
                "payload": {"pid": 1234, "spawned_at": 1789400000, "heartbeat_at": 1789400002, "run_id": 44},
            },
        ],
    }


def test_acceptance_evidence_contract_round_trips_through_json() -> None:
    payload = _valid_contract()
    schema = json.loads(_SCHEMA.read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    restored = normalize_contract(json.loads(json.dumps(payload)))
    assert restored == payload


def test_legacy_card_without_evidence_contract_remains_representable() -> None:
    assert normalize_contract(None) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda contract: contract.pop("required"),
        lambda contract: contract["observed"].__setitem__(0, {"status": "passed", "observed_at": 1, "source": "pytest"}),
        lambda contract: contract["observed"].__setitem__(1, {"id": "canary", "status": "passed", "observed_at": 2, "source": "dispatcher", "payload": {"pid": 1234, "spawned_at": 2}}),
    ],
)
def test_schema_rejects_missing_or_unusable_completion_evidence(mutate) -> None:
    contract = _valid_contract()
    mutate(contract)
    with pytest.raises(EvidenceValidationError):
        normalize_contract(contract)


def test_completion_rejects_missing_declared_evidence_without_state_transition(tmp_path, monkeypatch) -> None:
    """A declared but unobserved requirement must keep a card in flight."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    contract = {
        "version": 1,
        "required": [{"id": "focused-tests", "kind": "check", "description": "Focused tests pass"}],
        "observed": [],
    }

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="evidence-gated", acceptance_evidence=contract)
        assert kb.complete_task(conn, task_id, summary="local tests ran") is False
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert task is not None
    assert task.status == "ready"
    assert any(event.kind == "acceptance_evidence_rejected" for event in events)
