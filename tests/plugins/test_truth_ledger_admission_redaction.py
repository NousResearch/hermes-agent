import importlib.util
import json
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def truth_ledger_modules():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "truth-ledger"

    if "hermes_plugins" not in __import__("sys").modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        __import__("sys").modules["hermes_plugins"] = ns

    pkg = types.ModuleType("hermes_plugins.truth_ledger")
    pkg.__path__ = [str(plugin_dir)]
    __import__("sys").modules["hermes_plugins.truth_ledger"] = pkg

    def _load(module_name: str):
        spec = importlib.util.spec_from_file_location(
            f"hermes_plugins.truth_ledger.{module_name}",
            plugin_dir / f"{module_name}.py",
            submodule_search_locations=[str(plugin_dir)],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "hermes_plugins.truth_ledger"
        mod.__path__ = [str(plugin_dir)]
        __import__("sys").modules[f"hermes_plugins.truth_ledger.{module_name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    identity = _load("identity")
    redaction = _load("redaction")
    admission = _load("admission")
    return {"admission": admission, "redaction": redaction, "identity": identity}


def _candidate(**overrides):
    data = {
        "scope": "user",
        "kind": "preference",
        "key": "response.style",
        "subject": "user:42",
        "value": "concise",
        "evidence_type": "user_stated",
        "operation": "assert",
    }
    data.update(overrides)
    return data


def test_valid_candidate_is_admitted(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(),
        metadata={"speaker_id": "user:42", "chat_id": "chat-1"},
    )
    assert decision["admit"] is True
    assert decision["status"] == "admit"


def test_unknown_speaker_blocks_user_scope(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(),
        metadata={"speaker_id": None, "chat_id": "chat-1"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "unknown_speaker"


@pytest.mark.parametrize("scope", ["local", "global", "session"])
def test_non_frozen_scope_rejected(truth_ledger_modules, scope):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(scope=scope),
        metadata={"speaker_id": "user:42"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "invalid_scope"


@pytest.mark.parametrize("kind", ["reminder", "todo", "habit"])
def test_non_frozen_kind_rejected(truth_ledger_modules, kind):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(kind=kind),
        metadata={"speaker_id": "user:42"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "invalid_kind"


def test_assistant_inferred_is_review_only(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(evidence_type="assistant_inferred"),
        metadata={"speaker_id": "user:42"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "inferred_evidence"


def test_do_not_remember_blocks_candidate(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(value="don't remember this: my API key is sk-abc1234567890"),
        metadata={"speaker_id": "user:42"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "do_not_remember"


def test_secret_value_is_rejected_and_redacted_in_reason(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(value="my token is ghp_1234567890abcdefghij"),
        metadata={"speaker_id": "user:42"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "sensitive_value"
    assert "ghp_1234567890abcdefghij" not in decision.get("detail", "")


def test_chatter_and_temporary_updates_are_rejected(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    decision = admission.evaluate_candidate(
        _candidate(value="running tests now, will update in a minute"),
        metadata={"speaker_id": "user:42"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "non_durable"


def test_oversized_candidate_is_rejected(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    huge = "x" * (admission.MAX_VALUE_BYTES + 1)
    decision = admission.evaluate_candidate(
        _candidate(value=huge),
        metadata={"speaker_id": "user:42"},
    )
    assert decision["admit"] is False
    assert decision["reason"] == "oversize"


def test_private_tool_and_private_doc_payloads_are_rejected(truth_ledger_modules):
    admission = truth_ledger_modules["admission"]
    private_tool = admission.evaluate_candidate(
        _candidate(),
        metadata={"speaker_id": "user:42", "source_channel": "private_tool_output"},
    )
    private_doc = admission.evaluate_candidate(
        _candidate(),
        metadata={"speaker_id": "user:42", "source_channel": "private_document"},
    )
    assert private_tool["admit"] is False
    assert private_tool["reason"] == "private_source"
    assert private_doc["admit"] is False
    assert private_doc["reason"] == "private_source"


def test_redaction_strips_conversation_history_and_sensitive_fields(truth_ledger_modules):
    redaction = truth_ledger_modules["redaction"]
    payload = {
        "conversation_history": [{"role": "user", "content": "secret"}],
        "tool_output": "Authorization: Bearer sk-1234567890abcdefghij",
        "value": "password=hunter2",
        "safe": "keep-me",
    }
    sanitized = redaction.sanitize_payload(payload)
    text = json.dumps(sanitized)
    assert "conversation_history" not in text
    assert "hunter2" not in text
    assert "sk-1234567890abcdefghij" not in text
    assert sanitized["safe"] == "keep-me"


def test_identity_gate_unknown_values(truth_ledger_modules):
    identity = truth_ledger_modules["identity"]
    assert identity.has_stable_speaker_id({"speaker_id": "user:42"}) is True
    assert identity.has_stable_speaker_id({"speaker_id": "unknown"}) is False
    assert identity.has_stable_speaker_id({"speaker_id": ""}) is False
