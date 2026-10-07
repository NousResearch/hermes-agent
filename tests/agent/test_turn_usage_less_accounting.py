"""A provider response without a ``usage`` chunk is still a call that happened (#71578).

``record_response_usage`` increments ``session_api_calls`` and then returns early when the
response carries no usage. Everything after that return — the ``queue_token_counts`` write
that carries ``api_call_count``, every token counter and ``billing_provider`` — was skipped
entirely, so a session that demonstrably talked to the provider recorded
``api_call_count=0``, all tokens 0 and ``billing_provider`` NULL: indistinguishable from a
session that never ran.

``billing_provider`` NULL is the reporter's discriminator and it is a co-symptom — it rides
in the same skipped write, which is why the correlation was perfect (217/217) while the
cause sat one block away.

These tests assert on what reached the database, not on what was attempted.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Compressor:
    awaiting_real_usage_after_compression = False

    def update_from_response(self, _payload):
        pass

    def note_usage_less_response(self):
        pass


class _DB:
    """Records queue_token_counts calls without touching SQLite."""

    def __init__(self):
        self.queued = []

    def queue_token_counts(self, session_id, **kwargs):
        self.queued.append((session_id, kwargs))


class _Agent:
    def __init__(self, with_db=True):
        self.context_compressor = _Compressor()
        self.session_api_calls = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = None
        self.session_cost_source = None
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "anthropic/claude-opus-5"
        self.api_mode = "chat_completions"
        self.session_id = "sess-1"
        self._session_db = _DB() if with_db else None
        self._session_db_created = True
        self._api_latency_history = []
        self._api_output_history = []
        self.verbose_logging = False

    def _ensure_db_session(self):
        self.ensured = True


def _record(agent, response):
    from agent.turn_usage import record_response_usage

    return record_response_usage(
        agent, response, messages=[{"role": "user", "content": "hi"}],
        api_call_count=1, api_duration=0.1,
        compression_attempts=0, max_compression_attempts=3,
    )


@pytest.fixture(autouse=True)
def _stub_session_source(monkeypatch):
    """Keep ``_agent_session_source`` off the import chain.

    It late-imports ``run_agent`` -> ``hermes_cli.env_loader`` -> ``dotenv``, which a
    worktree venv may not have. Unit tests here are about what reaches the DB, not about
    the session-surface string, so stub the one function rather than inherit a missing
    transitive dependency.
    """
    import agent.turn_usage as turn_usage

    monkeypatch.setattr(turn_usage, "_agent_session_source", lambda _agent: "cli")


def test_usage_less_response_still_records_the_call():
    """The bug: no usage must not mean no record."""
    agent = _Agent()
    _record(agent, SimpleNamespace(usage=None))

    assert agent.session_api_calls == 1
    assert len(agent._session_db.queued) == 1, "the call never reached state.db"

    session_id, kwargs = agent._session_db.queued[0]
    assert session_id == "sess-1"
    assert kwargs["api_call_count"] == 1
    assert kwargs["billing_provider"] == "openrouter", "the route was known; record it"
    assert kwargs["model"] == "anthropic/claude-opus-5"
    # Unknowable quantities stay absent, not invented as zero-with-a-price.
    assert kwargs.get("input_tokens", 0) == 0
    assert kwargs.get("estimated_cost_usd") is None


def test_falsy_usage_takes_the_usage_less_path():
    """A falsy ``usage`` (``{}``, ``None``, ``0``) is usage-less.

    ``SimpleNamespace()`` is *truthy* — it would take the usage-bearing path — so it cannot
    stand in for "usage present but empty". A dict ``{}`` is what a provider adapter actually
    hands over, and it must record the call without inventing a price.
    """
    agent = _Agent()
    _record(agent, SimpleNamespace(usage={}))
    session_id, kwargs = agent._session_db.queued[0]
    assert kwargs["api_call_count"] == 1
    assert kwargs.get("estimated_cost_usd") is None
    assert kwargs.get("cost_status") is None


def test_absent_usage_attribute_is_usage_less():
    """A response with no ``usage`` attribute at all is the common streaming case."""
    agent = _Agent()
    _record(agent, SimpleNamespace())
    assert len(agent._session_db.queued) == 1
    _session_id, kwargs = agent._session_db.queued[0]
    assert kwargs["api_call_count"] == 1


def test_usage_bearing_response_is_unchanged():
    """The usage path must keep its full accounting — no regression on the common case."""
    agent = _Agent()
    usage = SimpleNamespace(
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    _record(agent, SimpleNamespace(usage=usage))

    assert agent.session_input_tokens == 100
    assert agent.session_output_tokens == 50

    session_id, kwargs = agent._session_db.queued[0]
    assert kwargs["input_tokens"] == 100
    assert kwargs["output_tokens"] == 50
    assert kwargs["api_call_count"] == 1
    # estimated_cost_usd is left to the pricing layer: it needs a live model-catalog lookup,
    # so it is None offline. Asserting on it would test the network, not this change.
    assert "estimated_cost_usd" in kwargs


def test_two_usage_less_calls_accumulate_to_two():
    """Each call records once — the counter must not collapse to a single write."""
    agent = _Agent()
    _record(agent, SimpleNamespace(usage=None))
    _record(agent, SimpleNamespace(usage=None))

    assert agent.session_api_calls == 2
    assert len(agent._session_db.queued) == 2
    assert all(kwargs["api_call_count"] == 1 for _, kwargs in agent._session_db.queued)


def test_no_session_db_is_a_no_op():
    """Sessions without a database (CLI one-shots) must not raise."""
    agent = _Agent(with_db=False)
    _record(agent, SimpleNamespace(usage=None))
    assert agent.session_api_calls == 1


def test_db_failure_does_not_kill_the_turn():
    """Accounting loss is logged, never raised into a turn."""

    class _Boom(_DB):
        def queue_token_counts(self, session_id, **kwargs):
            raise RuntimeError("db gone")

    agent = _Agent()
    agent._session_db = _Boom()
    _record(agent, SimpleNamespace(usage=None))
    assert agent.session_api_calls == 1


@pytest.mark.parametrize("missing_attr", ["_session_db", "session_id", "_session_db_created"])
def test_absent_attributes_are_tolerated(missing_attr):
    """A stripped-down agent must not AttributeError — the attribute is *absent*, not None.

    ``tui_gateway/synthetic_turn.py``'s ``SyntheticHeavyAgent`` has no ``_session_db`` at all,
    so reading it directly outside the try would raise mid-turn.
    """
    agent = _Agent()
    delattr(agent, missing_attr)
    _record(agent, SimpleNamespace(usage=None))
    assert agent.session_api_calls == 1


def test_absent_session_db_attribute_is_a_no_op_not_a_crash():
    """``SyntheticHeavyAgent`` (tui_gateway/synthetic_turn.py) has no ``_session_db`` at all.

    A direct ``agent._session_db`` read raises ``AttributeError``; the ``getattr`` default is
    what makes this a no-op. Asserting the observable difference: with the attribute gone,
    nothing is queued *and* the turn completes.
    """
    agent = _Agent()
    db = agent._session_db
    del agent._session_db
    _record(agent, SimpleNamespace(usage=None))
    assert agent.session_api_calls == 1
    assert db.queued == [], "nothing may be queued when there is no database"
    # Same for a missing _session_db_created on an agent that does have a database.
    agent2 = _Agent()
    del agent2._session_db_created
    _record(agent2, SimpleNamespace(usage=None))
    assert len(agent2._session_db.queued) == 1, "defaults to 'already created' and proceeds"


def test_route_is_not_claimed_by_a_usage_less_call():
    """The write must not win ``first_accounted_route``.

    create_session records the requested route before any API call; if that provider fails
    and a fallback answers, the session row must end up on the fallback — the one whose
    tokens are actually on the row — not on the provider that failed.
    """
    agent = _Agent()
    _record(agent, SimpleNamespace(usage=None))
    _session_id, kwargs = agent._session_db.queued[0]
    assert kwargs["route_authoritative"] is False