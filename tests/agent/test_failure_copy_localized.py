"""Failed-turn chat copy follows the active language; what the model and the gateway read stays English.

A truncated tool call ends the turn with curated copy: the user is shown it in their language, while the
stored ``error`` (gateway matchers, failed-turn metadata) and the assistant row that closes the tool tail
(replayed to the model next turn) keep the English rendering.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import i18n
from agent.turn_failure_copy import site_copy
from agent.turn_truncation import recover_from_truncation
from hermes_constants import FINISH_REASON_LENGTH, get_hermes_home

_DE_TRUNCATED = "Die Antwort des Modells wurde abgeschnitten; Hermes hat die unvollständige Aktion nicht ausgeführt."


@pytest.fixture
def german_overlay(monkeypatch):
    overlay = get_hermes_home() / "locales"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "de.yaml").write_text(f'"explainer.failure.truncated": "{_DE_TRUNCATED}"\n', encoding="utf-8")
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    i18n.reset_language_cache()
    yield
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()


def _agent_with_truncated_tool_call():
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.log_prefix = ""
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(id="c2")])
    agent._get_transport.return_value.normalize_response.return_value = message
    return agent


def test_truncated_tool_call_copy_is_localized_for_the_user_and_english_for_the_model(german_overlay):
    messages = [
        {"role": "user", "content": "write the report"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "data"},
    ]
    verdict = recover_from_truncation(
        _agent_with_truncated_tool_call(), SimpleNamespace(id="resp-1", usage=None), FINISH_REASON_LENGTH,
        MagicMock(), messages=messages, conversation_history=None, api_kwargs={}, api_call_count=2,
        effective_task_id=None, current_turn_user_idx=0, length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=4, retry_count=0, compression_attempts=0,
    )

    english = site_copy("truncated")
    assert english != _DE_TRUNCATED and "cut off" in english
    result = verdict.result
    assert result["final_response"] == _DE_TRUNCATED
    assert result["error"] == english
    assert result["failure_reason"] == "truncated"
    assert (messages[-1]["role"], messages[-1]["content"]) == ("assistant", english)
