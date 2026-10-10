"""Regression: the one-line cron failure notice must name the whole provider WALK, not the
config's claim that a backup provider exists.

The reported shape: a job pinned to ``nous``/``deepseek-v4.1-flash`` failed with a terminal
``Gemini HTTP 429`` after the chain had been walked, and the delivered alert said a backup
provider was configured — a statement about the CONFIG, which cannot say which hops were
tried or why each was passed over. ``_fallback_chain_phrase()`` reads the config; when the
error text carries the walk ``agent.fallback_trail`` renders, that walk is strictly better
evidence and replaces the config phrase.
"""
from agent import fallback_trail as ft
from cron.scheduler import _fallback_walk_clause, _summarize_cron_failure_for_delivery


def _walk_error() -> str:
    """The terminal error of the reported incident, walk attached the way turn_recovery does."""
    trail = ft.format_fallback_trail(_agent_with_incident_walk())
    return (
        "Gemini HTTP 429 (RESOURCE_EXHAUSTED): You exceeded your current quota"
        f"\n\n{trail}"
    )


def _agent_with_incident_walk():
    from unittest.mock import MagicMock
    agent = MagicMock()
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="stepfun/step-3.7-flash:free",
                          role="primary", reason="rate limit")
    ft.record_hop_failure(agent, provider="gemini", model="gemini-2.5-flash-lite",
                          role="fallback", reason="rate limit")
    return agent


def test_cron_alert_replaces_the_config_read_phrase_with_the_walk():
    message = _summarize_cron_failure_for_delivery({"name": "tasksheet-sync", "id": "ab12"}, _walk_error())

    # The walk names what the config read cannot: which hop failed and why.
    assert "Fallback walk:" in message
    assert "primary nous/stepfun/step-3.7-flash:free — failed: rate limit" in message
    # The config-derived clause only ever said a chain existed, which is not the reported cause.
    assert "No backup provider" not in message


def test_cron_alert_still_uses_the_config_phrase_without_a_walk():
    message = _summarize_cron_failure_for_delivery(
        {"name": "tasksheet-sync", "id": "ab12"}, "Error code: 429 - fair-share rate limit"
    )

    assert "rate-limited" in message
    assert "Fallback walk:" not in message


def test_walk_clause_is_empty_for_text_without_a_walk():
    assert _fallback_walk_clause("RuntimeError: Gemini HTTP 429") == ""
    assert _fallback_walk_clause("") == ""


def test_walk_clause_bounds_a_long_walk():
    text = "\n".join(f"  hop {i} fallback p{i}/m{i} — failed: rate limit" for i in range(1, 8))
    clause = _fallback_walk_clause(text)
    assert "(+3 more)" in clause
    assert clause.count("→") == 3
