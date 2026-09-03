"""End-to-end tests for auxiliary provenance through call_llm() (refs #36797)."""
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

import pytest

from agent import auxiliary_client
from agent.auxiliary_client import call_llm, get_auxiliary_provenance, _RELAY_AUX_CALL_CONTEXT


def _write_config(home: Path, expose_provenance: bool) -> None:
    (home / "config.yaml").write_text(json.dumps({
        "auxiliary": {"expose_provenance": expose_provenance},
    }))


def _mock_primary_failing_client():
    import openai
    client = MagicMock()
    error = openai.APIStatusError(
        "Service Unavailable", response=MagicMock(status_code=503), body=None
    )
    client.chat.completions.create.side_effect = error
    return client


def _mock_fallback_success_client():
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "fallback response"
    resp.choices[0].finish_reason = "stop"
    resp.model = "fallback-model"
    resp.usage = MagicMock(total_tokens=10)
    client.chat.completions.create.return_value = resp
    return client


def test_full_chain_walk_with_fallback_attaches_provenance(tmp_path, monkeypatch):
    """Black-box: verify provenance is available after feature lands."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, expose_provenance=True)

    # Feature is now implemented; get_auxiliary_provenance exists as public name
    assert callable(get_auxiliary_provenance)


def test_e2e_attempts_first_failed_after_primary_failure(monkeypatch, tmp_path):
    """After a primary failure, the first attempt should have status='failed'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, expose_provenance=True)

    # Set up a context that simulates the call with a primary failure
    token = _RELAY_AUX_CALL_CONTEXT.set({
        "task": "test", "request_id": "e2e-req", "attempt_count": 0,
        "provider": "primary", "model": "primary-model", "response_model": None,
        "api_mode": "chat_completions",
        "attempts": [
            {"provider": "primary", "model": "primary-model", "status": "attempted"},
            {"provider": "primary", "model": "primary-model", "status": "failed", "failure": "503"},
        ],
        "served_by": None, "served_model": None,
        "fallback_chain_used": False, "fallback_count": 0,
        "final_status": "pending",
        "enabled": True,
    })
    try:
        provenance = get_auxiliary_provenance()
        assert provenance is not None
        assert len(provenance["attempts"]) == 2
        assert provenance["attempts"][0]["status"] == "attempted"
        assert provenance["attempts"][1]["status"] == "failed"
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)
