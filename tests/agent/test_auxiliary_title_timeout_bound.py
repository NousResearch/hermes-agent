"""Auto-title timeouts are bounded by ONE ``auxiliary.title_generation.timeout`` window and named as
timeouts (#89445, #66251).

On a slow local model the title request used to be retried on the same provider after every
full-budget timeout (three windows plus backoff ≈ 4x the configured deadline) and the failure was
logged at INFO as a ``connection error`` — indistinguishable from an unreachable endpoint — while
the only WARNING came from the fallback provider complaining about a model it never had."""

import logging
from unittest.mock import MagicMock, patch

from agent.auxiliary_client import call_llm


class _Timeout(Exception):
    pass


_Timeout.__name__ = "APITimeoutError"


def _route_patches(client):
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("mac-ollama", "qwen3.6:27b", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "qwen3.6:27b")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
    )


def test_title_timeout_hits_the_provider_once_and_names_the_deadline(caplog):
    primary = MagicMock()
    primary.base_url = "http://100.121.173.79:11434/v1"
    primary.chat.completions.create.side_effect = _Timeout("Request timed out.")
    p = _route_patches(primary)
    caplog.set_level(logging.INFO, logger="agent.auxiliary_client")
    with p[0], p[1], p[2], p[3], p[4]:
        try:
            call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}], timeout=30)
        except _Timeout:
            pass
        else:
            raise AssertionError("timeout must surface once every fallback is exhausted")
    # One request per deadline: the auto-title thread may not outlive the configured budget.
    assert primary.chat.completions.create.call_count == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    timed_out = [m for m in warnings if "timed out after 30s" in m]
    assert timed_out, warnings
    assert "http://100.121.173.79:11434/v1" in timed_out[0]
    assert "auxiliary.title_generation.timeout" in timed_out[0]
    assert not any("connection error on" in m for m in warnings), warnings


