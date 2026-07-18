import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / "plugins" / "truth-ledger"


def _load_plugin_module(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        PLUGIN_DIR / file_name,
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.truth_ledger"
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def contracts_mod():
    return _load_plugin_module("hermes_plugins.truth_ledger.contracts", "contracts.py")


@pytest.fixture
def schemas_mod():
    return _load_plugin_module("hermes_plugins.truth_ledger.schemas", "schemas.py")


def _sample_fact_candidate(*, operation: str = "assert", schema_version: int = 1) -> dict:
    return {
        "schema_version": schema_version,
        "operation": operation,
        "fact": {
            "scope": "user",
            "kind": "preference",
            "subject": "platform-user:slack:U123",
            "key": "communication.slack.reply_style",
            "value": "concise",
        },
        "evidence": {
            "type": "user_stated",
            "speaker_id": "U123",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "conversation_id": "C0B6M4X67ND",
            "thread_id": "1784318689.250609",
        },
        "confidence": 0.91,
    }


def _sample_ledger_event(*, operation: str = "assert", schema_version: int = 1) -> dict:
    event = {
        "schema_version": schema_version,
        "event_id": "evt_abcd1234",
        "occurred_at": "2026-07-17T21:00:00Z",
        "operation": operation,
        "fact_id": "fact_abcd1234",
        "supersedes": None,
        "fact": {
            "scope": "user",
            "kind": "preference",
            "subject": "platform-user:slack:U123",
            "key": "communication.slack.reply_style",
            "value": "concise",
        },
        "evidence": {
            "type": "user_stated",
            "profile": "default",
            "platform": "slack",
            "session_id": "session-1",
            "turn_id": "turn-1",
            "task_id": "",
            "speaker_id": "U123",
            "conversation_id": "C0B6M4X67ND",
            "thread_id": "1784318689.250609",
        },
        "extraction": {
            "schema_name": "truth-ledger.fact-candidates.v1",
            "provider": "openai-api",
            "model": "gpt-5",
            "prompt_version": 1,
        },
    }
    if operation == "supersede":
        event["supersedes"] = "fact_prev1234"
    return event


def test_contract_serialization_compact_utf8_and_newline(contracts_mod):
    payload = {
        "alpha": "A",
        "unicode": "café",
        "multiline": "line1\nline2",
    }

    encoded = contracts_mod.serialize_jsonl_record(payload)

    assert isinstance(encoded, bytes)
    assert encoded.endswith(b"\n")
    assert encoded[:3] != b"\xef\xbb\xbf"
    assert b"\\u00e9" not in encoded
    assert b"\nline2" not in encoded

    text = encoded.decode("utf-8")
    assert text.strip() == '{"alpha":"A","unicode":"café","multiline":"line1\\nline2"}'


def test_contract_raises_for_unknown_schema_version(contracts_mod):
    with pytest.raises(ValueError):
        contracts_mod.assert_schema_version(2)


def test_schema_inventory_exists(schemas_mod):
    names = set(schemas_mod.available_schema_names())
    assert names == {
        "source-envelope.v1",
        "fact-candidates.v1",
        "ledger-event.v1",
        "spool-record.v1",
        "dead-letter.v1",
        "current-projection.v1",
    }


def test_metaschema_validation_for_all_schemas(schemas_mod):
    result = schemas_mod.validate_all_metaschemas()
    assert result == {}


def test_fact_candidates_valid_and_invalid(schemas_mod):
    valid = {"schema_name": "truth-ledger.fact-candidates.v1", "facts": [_sample_fact_candidate()]}
    schemas_mod.validate_document("fact-candidates.v1", valid)

    bad_enum = {"schema_name": "truth-ledger.fact-candidates.v1", "facts": [_sample_fact_candidate(operation="delete")]}
    with pytest.raises(ValueError):
        schemas_mod.validate_document("fact-candidates.v1", bad_enum)

    unknown_version = {
        "schema_name": "truth-ledger.fact-candidates.v2",
        "facts": [_sample_fact_candidate(schema_version=2)],
    }
    with pytest.raises(ValueError):
        schemas_mod.validate_document("fact-candidates.v1", unknown_version)


def test_ledger_event_contracts_for_operations(schemas_mod):
    assert_event = _sample_ledger_event(operation="assert")
    supersede_event = _sample_ledger_event(operation="supersede")
    retract_event = _sample_ledger_event(operation="retract")

    schemas_mod.validate_document("ledger-event.v1", assert_event)
    schemas_mod.validate_document("ledger-event.v1", supersede_event)
    schemas_mod.validate_document("ledger-event.v1", retract_event)

    missing_supersedes = _sample_ledger_event(operation="supersede")
    missing_supersedes["supersedes"] = None
    with pytest.raises(ValueError):
        schemas_mod.validate_document("ledger-event.v1", missing_supersedes)


def test_source_envelope_and_dead_letter_size_limits(schemas_mod):
    envelope = {
        "schema_name": "truth-ledger.source-envelope.v1",
        "schema_version": 1,
        "captured_at": "2026-07-17T21:00:00Z",
        "profile": "default",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "origin": {
            "platform": "slack",
            "conversation_id": "C0B6M4X67ND",
            "thread_id": "1784318689.250609",
            "speaker_id": "U123",
        },
        "input": {"user_message": "keep replies short"},
        "output": {"assistant_response": "ok"},
    }
    schemas_mod.validate_document("source-envelope.v1", envelope)

    oversized = json.loads(json.dumps(envelope))
    oversized["output"]["assistant_response"] = "x" * 70000
    with pytest.raises(ValueError):
        schemas_mod.validate_document("source-envelope.v1", oversized)

    dead_letter = {
        "schema_name": "truth-ledger.dead-letter.v1",
        "schema_version": 1,
        "occurred_at": "2026-07-17T21:00:00Z",
        "reason_code": "schema_mismatch",
        "source_ref": {
            "profile": "default",
            "session_id": "session-1",
            "turn_id": "turn-1",
        },
        "attempt_count": 2,
        "last_error": "schema validation failed",
    }
    schemas_mod.validate_document("dead-letter.v1", dead_letter)


def test_current_projection_requires_explicit_null_or_value(schemas_mod):
    record = {
        "schema_name": "truth-ledger.current-projection.v1",
        "schema_version": 1,
        "logical_key": {
            "scope": "user",
            "subject": "platform-user:slack:U123",
            "key": "communication.slack.reply_style",
        },
        "state": "active",
        "fact_id": "fact_abcd1234",
        "value": "concise",
        "updated_at": "2026-07-17T21:00:00Z",
    }
    schemas_mod.validate_document("current-projection.v1", record)

    retracted = dict(record)
    retracted["state"] = "retracted"
    retracted["value"] = None
    schemas_mod.validate_document("current-projection.v1", retracted)

    invalid_missing_value = dict(record)
    invalid_missing_value.pop("value")
    with pytest.raises(ValueError):
        schemas_mod.validate_document("current-projection.v1", invalid_missing_value)
