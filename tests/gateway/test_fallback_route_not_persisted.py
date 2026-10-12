"""Provider fallback must never become a session's durable route.

Regression coverage for the sticky-fallback incident: a transient Anthropic stream error activated
provider fallback, the gateway persisted the fallback route as the session's ``gateway_runtime``,
and every later turn was rebuilt on the fallback -- ``restore_primary_runtime()`` then "restored"
the fallback onto itself. Long-lived Matrix rooms stayed on the fallback model for days.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.run_turn import GatewayTurnMixin


class _Runner(GatewayTurnMixin):
    """Bare mixin host: _sync_session_model_from_agent only needs _session_db."""

    def __init__(self, row):
        self._db = MagicMock()
        self._db.get_session.return_value = row
        self._session_db = SimpleNamespace(_db=self._db)

    @property
    def written(self):
        """(session_id, parsed model_config, model kwarg) of the last write, or None."""
        if not self._db.update_session_meta.called:
            return None
        args, kwargs = self._db.update_session_meta.call_args
        return args[0], json.loads(args[1]), kwargs.get("model")


def _agent(model, provider, *, fallback, base_url="https://api.anthropic.com"):
    return SimpleNamespace(
        model=model, provider=provider, base_url=base_url, api_mode="anthropic_messages",
        _fallback_activated=fallback,
    )


def _primary_config():
    return json.dumps({"gateway_runtime": {
        "provider": "anthropic", "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages", "fallback_active": False,
    }})


def test_fallback_turn_does_not_overwrite_the_persisted_primary_route():
    """The whole bug in one assertion: a fallback turn must leave the routable identity alone."""
    runner = _Runner({"model": "claude-opus-5", "model_config": _primary_config()})

    runner._sync_session_model_from_agent("s1", _agent(
        "gpt-6-astra", "openai-codex", fallback=True, base_url="https://chatgpt.com/backend-api/codex/"))

    _sid, config, model_kwarg = runner.written
    runtime = config["gateway_runtime"]
    # Resume re-pins from gateway_runtime.provider -- it must still name the primary.
    assert runtime["provider"] == "anthropic"
    assert runtime["base_url"] == "https://api.anthropic.com"
    assert model_kwarg == "claude-opus-5"
    # ...but the fallback is still observable for billing/debugging.
    assert runtime["fallback_active"] is True


def test_successful_primary_turn_persists_normally():
    """The non-fallback path is unchanged: a real route change is still recorded."""
    runner = _Runner({"model": "claude-opus-5", "model_config": _primary_config()})

    runner._sync_session_model_from_agent("s1", _agent("claude-sonnet-5", "anthropic", fallback=False))

    _sid, config, model_kwarg = runner.written
    assert config["gateway_runtime"]["provider"] == "anthropic"
    assert config["gateway_runtime"]["fallback_active"] is False
    assert model_kwarg == "claude-sonnet-5"


def test_primary_route_is_restored_after_a_fallback_turn():
    """End-to-end ordering: fallback turn, then a recovered primary turn. The row must not be
    left describing the fallback -- that is what pinned real sessions for days."""
    row = {"model": "claude-opus-5", "model_config": _primary_config()}
    runner = _Runner(row)

    runner._sync_session_model_from_agent("s1", _agent("gpt-6-astra", "openai-codex", fallback=True))
    _sid, after_fallback, _m = runner.written
    row["model_config"] = json.dumps(after_fallback)  # feed the write back in, as the DB would

    runner._sync_session_model_from_agent("s1", _agent("claude-opus-5", "anthropic", fallback=False))
    _sid, after_primary, model_kwarg = runner.written

    assert after_primary["gateway_runtime"]["provider"] == "anthropic"
    assert after_primary["gateway_runtime"]["fallback_active"] is False
    assert model_kwarg == "claude-opus-5"


def test_repeated_fallback_turns_do_not_rewrite_every_turn():
    """Once flagged, further fallback turns are a no-op write (the row already says so)."""
    runner = _Runner({"model": "claude-opus-5", "model_config": json.dumps({"gateway_runtime": {
        "provider": "anthropic", "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages", "fallback_active": True,
    }})})

    runner._sync_session_model_from_agent("s1", _agent("gpt-6-astra", "openai-codex", fallback=True))

    assert runner.written is None


def test_first_ever_turn_falling_back_still_records_a_route():
    """No primary recorded yet: store the fallback route (flagged) rather than nothing, so the
    session is routable at all; a later primary turn overwrites it."""
    runner = _Runner({"model": None, "model_config": None})

    runner._sync_session_model_from_agent("s1", _agent("gpt-6-astra", "openai-codex", fallback=True))

    _sid, config, model_kwarg = runner.written
    assert config["gateway_runtime"]["provider"] == "openai-codex"
    assert config["gateway_runtime"]["fallback_active"] is True
    assert model_kwarg == "gpt-6-astra"
