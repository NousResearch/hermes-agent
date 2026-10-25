"""The route's model catalog decides the title lane's first request (no guaranteed 400 on mandatory routes).

OpenRouter's (and the Portal's) ``/v1/models`` flags ``reasoning.mandatory`` per model; such a route answers
a thinking-off aux call (``reasoning: {enabled: false}``) with 400 "Reasoning is mandatory for this endpoint
and cannot be disabled". The floor memo only learned that from the 400 itself, per process, so an aux-only
OpenRouter route — nothing else warms that catalog — paid the failed round-trip in every process.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.models as models_mod
from agent import auxiliary_reasoning_floor
from agent.auxiliary_client import call_llm
from hermes_cli import models_reasoning_caps

_CATALOG = [
    {"id": "openai/gpt-oss-20b", "supported_parameters": ["reasoning", "tools"],
     "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low"]}},
    {"id": "x-ai/grok-4.3", "supported_parameters": ["reasoning", "tools"],
     "reasoning": {"mandatory": False, "supported_efforts": ["high", "medium", "low", "none"]}},
]


@pytest.fixture
def fresh_process(monkeypatch):
    """Module state of a newly started process (catalog neither in memory nor memo); disk untouched."""
    def _reset():
        auxiliary_reasoning_floor._FLOORED_ROUTES.clear()
        for name in ("_openrouter_reasoning_caps_cache", "_openrouter_reasoning_caps_failed_at"):
            monkeypatch.setattr(models_mod, name, None)
        for name in ("_openrouter_caps_disk_checked", "_openrouter_caps_warm_started"):
            monkeypatch.setattr(models_mod, name, False)
    _reset()
    yield _reset
    auxiliary_reasoning_floor._FLOORED_ROUTES.clear()


def _title_request(model):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = {"ok": True}
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openrouter", model, None, "sk-or-x", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, model)),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=None),
    ):
        call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}],
                 reasoning_config={"enabled": False})
    return client.chat.completions.create.call_args_list[0].kwargs.get("extra_body", {}).get("reasoning")


def test_catalog_mandatory_model_gets_the_floor_on_the_first_request(fresh_process):
    """A mirror left by an earlier process: the mandatory model's first thinking-off request already
    carries the floor; an optional model on the same route keeps its disable."""
    models_reasoning_caps._seed_reasoning_caps(models_reasoning_caps._OPENROUTER_CATALOG_URL, _CATALOG)

    assert _title_request("openai/gpt-oss-20b") == {
        "enabled": True, "effort": auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT}
    assert _title_request("x-ai/grok-4.3") == {"enabled": False}


def test_cold_catalog_is_warmed_so_the_next_process_starts_at_the_floor(fresh_process, monkeypatch):
    """No mirror yet: the first process can only learn from the 400, but its lookup warms the catalog, so
    the next process never sends the disable."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)  # the warmer is a no-op under pytest
    warmed = []
    real_thread = threading.Thread

    def _joined_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        if kwargs.get("name") == "reasoning-caps-warm":
            warmed.append(thread)
        return thread

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"data": _CATALOG}).encode()

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", lambda req, timeout: _Resp())
    monkeypatch.setattr(models_reasoning_caps.threading, "Thread", _joined_thread)

    _title_request("openai/gpt-oss-20b")
    assert warmed, "a cold catalog lookup must start the background warm"
    for thread in warmed:
        thread.join(timeout=10)

    fresh_process()
    assert _title_request("openai/gpt-oss-20b") == {
        "enabled": True, "effort": auxiliary_reasoning_floor.REASONING_FLOOR_EFFORT}
