"""Tests for run_agent.py per-API-call /log_generation emission.

The cornerstone fix for Hermes MC Live active-generation telemetry is:

1. At API-call start, ``run_agent.py`` captures the current DD identity
   (run_id, session_key, context) into LOCAL variables.
2. After the provider call returns, it emits ``/log_generation`` using
   those captured locals — not whatever ``self._dd_*`` happens to be at
   write time. This prevents turn N from writing under turn N+1's
   run_id when the cached AIAgent is reused across turns.
3. On a synchronous success response from obs-ingest, the per-turn DD
   context's ``per_call_generation_emitted`` flag flips to True so the
   gateway can skip the synthetic completion-time generation and avoid
   double-counting. CRITICAL: the helper must use the SYNC client
   (``log_generation_sync`` returning bool) — fire-and-forget would
   silently suppress the synthetic fallback when the underlying POST
   actually failed (registry misresolved, ingest down, schema reject).

These tests pin the contract on the helper that owns step 2+3
(``_emit_dd_per_call_generation``) so the bulk of run_agent.py's API
loop doesn't have to be stood up.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def _import_helper():
    from run_agent import _emit_dd_per_call_generation
    return _emit_dd_per_call_generation


def _make_canonical_usage(input_tokens=100, output_tokens=20, cache_read=5, cache_write=2):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _make_dd_obs(sync_returns: bool = True):
    """A dd_obs-shaped mock whose sync POST returns the given bool."""
    mod = MagicMock()
    mod.log_generation_sync.return_value = sync_returns
    return mod


class TestEmitDdPerCallGeneration:
    def test_posts_log_generation_with_captured_run_id(self):
        emit = _import_helper()
        dd_obs = _make_dd_obs(sync_returns=True)
        ctx = {
            "run_id": "hermes-sess-g1",
            "session_key": "agent:hermes:gateway:sess",
            "per_call_generation_emitted": False,
        }

        emit(
            dd_obs_module=dd_obs,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            session_id_fallback="sess",
            model="gpt-5.5",
            provider="openai-codex",
            requested_at="2026-04-30T07:48:07Z",
            completed_at="2026-04-30T07:48:09Z",
            latency_ms=2000,
            canonical_usage=_make_canonical_usage(),
            cost_amount_usd=0.0123,
            dd_context=ctx,
        )

        dd_obs.log_generation_sync.assert_called_once()
        kwargs = dd_obs.log_generation_sync.call_args.kwargs
        assert kwargs["run_id"] == "hermes-sess-g1"
        assert kwargs["session_id"] == "agent:hermes:gateway:sess"
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["provider"] == "openai-codex"
        assert kwargs["requested_at"] == "2026-04-30T07:48:07Z"
        assert kwargs["completed_at"] == "2026-04-30T07:48:09Z"
        assert kwargs["latency_ms"] == 2000
        assert kwargs["input_tokens"] == 100
        assert kwargs["output_tokens"] == 20
        assert kwargs["cache_read_tokens"] == 5
        assert kwargs["cache_creation_tokens"] == 2
        assert abs(kwargs["cost_total_usd"] - 0.0123) < 1e-9
        # Must use a unique generation_id (not a deterministic suffix
        # that would collide across calls in the same run).
        assert kwargs["generation_id"]
        assert kwargs["generation_id"] != "hermes-sess-g1:generation:1"

    def test_falls_back_to_session_id_when_session_key_blank(self):
        emit = _import_helper()
        dd_obs = _make_dd_obs(sync_returns=True)

        emit(
            dd_obs_module=dd_obs,
            run_id="hermes-sess-g1",
            session_key=None,
            session_id_fallback="legacy-session-id",
            model="gpt-5.5",
            provider="openai-codex",
            requested_at="2026-04-30T07:48:07Z",
            completed_at="2026-04-30T07:48:09Z",
            latency_ms=1000,
            canonical_usage=_make_canonical_usage(),
            cost_amount_usd=0.0,
            dd_context=None,
        )

        kwargs = dd_obs.log_generation_sync.call_args.kwargs
        assert kwargs["session_id"] == "legacy-session-id"

    def test_flips_per_call_emitted_flag_only_on_post_success(self):
        emit = _import_helper()
        dd_obs = _make_dd_obs(sync_returns=True)
        ctx = {
            "run_id": "hermes-sess-g1",
            "session_key": "agent:hermes:gateway:sess",
            "per_call_generation_emitted": False,
        }

        emit(
            dd_obs_module=dd_obs,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            session_id_fallback="sess",
            model="gpt-5.5",
            provider="openai-codex",
            requested_at="2026-04-30T07:48:07Z",
            completed_at="2026-04-30T07:48:09Z",
            latency_ms=1000,
            canonical_usage=_make_canonical_usage(),
            cost_amount_usd=0.0,
            dd_context=ctx,
        )

        assert ctx["per_call_generation_emitted"] is True

    def test_no_context_does_not_raise(self):
        emit = _import_helper()
        dd_obs = _make_dd_obs(sync_returns=True)

        emit(
            dd_obs_module=dd_obs,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            session_id_fallback="sess",
            model="gpt-5.5",
            provider="openai-codex",
            requested_at="2026-04-30T07:48:07Z",
            completed_at="2026-04-30T07:48:09Z",
            latency_ms=1000,
            canonical_usage=_make_canonical_usage(),
            cost_amount_usd=0.0,
            dd_context=None,
        )

        dd_obs.log_generation_sync.assert_called_once()

    def test_uses_sync_client_not_fire_and_forget(self):
        """The helper MUST use ``log_generation_sync`` (returns bool) —
        not the fire-and-forget ``log_generation`` — so the flag-flip
        decision is made on a real accepted/failed POST, not a daemon
        thread that returns immediately while the HTTP call is still
        in-flight or has already failed.
        """
        emit = _import_helper()
        dd_obs = _make_dd_obs(sync_returns=True)

        emit(
            dd_obs_module=dd_obs,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            session_id_fallback="sess",
            model="gpt-5.5",
            provider="openai-codex",
            requested_at="2026-04-30T07:48:07Z",
            completed_at="2026-04-30T07:48:09Z",
            latency_ms=1000,
            canonical_usage=_make_canonical_usage(),
            cost_amount_usd=0.0,
            dd_context=None,
        )

        dd_obs.log_generation_sync.assert_called_once()
        dd_obs.log_generation.assert_not_called()

    def test_post_failure_leaves_flag_false(self):
        """If obs-ingest is down / wrong URL / schema rejection, the
        sync POST returns False. The flag must stay False so the gateway
        still writes its synthetic completion-time fallback row.
        Otherwise MC Live ends up with NEITHER an active-generation row
        NOR a fallback — the worst possible outcome.
        """
        emit = _import_helper()
        dd_obs = _make_dd_obs(sync_returns=False)
        ctx = {
            "run_id": "hermes-sess-g1",
            "session_key": "agent:hermes:gateway:sess",
            "per_call_generation_emitted": False,
        }

        emit(
            dd_obs_module=dd_obs,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            session_id_fallback="sess",
            model="gpt-5.5",
            provider="openai-codex",
            requested_at="2026-04-30T07:48:07Z",
            completed_at="2026-04-30T07:48:09Z",
            latency_ms=1000,
            canonical_usage=_make_canonical_usage(),
            cost_amount_usd=0.0,
            dd_context=ctx,
        )

        # POST failed → flag stays False → synthetic fallback fires.
        assert ctx["per_call_generation_emitted"] is False

    def test_exception_in_dd_obs_is_swallowed(self):
        """Per-call generation logging must NEVER block or fail the
        agent loop — dd_obs failures are swallowed silently."""
        emit = _import_helper()
        dd_obs = MagicMock()
        dd_obs.log_generation_sync.side_effect = RuntimeError("ingest down")
        ctx = {
            "run_id": "hermes-sess-g1",
            "session_key": "agent:hermes:gateway:sess",
            "per_call_generation_emitted": False,
        }

        # Should not raise.
        emit(
            dd_obs_module=dd_obs,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            session_id_fallback="sess",
            model="gpt-5.5",
            provider="openai-codex",
            requested_at="2026-04-30T07:48:07Z",
            completed_at="2026-04-30T07:48:09Z",
            latency_ms=1000,
            canonical_usage=_make_canonical_usage(),
            cost_amount_usd=0.0,
            dd_context=ctx,
        )

        # Flag is NOT flipped if the post failed — gateway must still
        # write the synthetic summary as a fallback.
        assert ctx["per_call_generation_emitted"] is False


class TestCaptureAtCallStartInvariant:
    """The 'capture-at-API-call-start' invariant is what guarantees that
    a turn N per-call write never lands under turn N+1's run_id when the
    cached AIAgent is reused. The helper itself is the proof: it takes
    run_id/session_key/dd_context as explicit arguments rather than
    reading from a passed-in agent, so callers must capture into locals
    before the provider call.
    """

    def test_helper_signature_takes_explicit_identity_not_agent(self):
        """The helper must accept identity as kwargs — not as ``self``
        or an ``agent`` arg — so it cannot accidentally read mid-flight
        ``agent._dd_run_id`` mutations.
        """
        import inspect

        emit = _import_helper()
        sig = inspect.signature(emit)
        params = sig.parameters

        assert "run_id" in params
        assert "session_key" in params
        assert "dd_context" in params
        # No ``agent`` or ``self`` parameter — capture must happen in caller.
        assert "agent" not in params
        assert "self" not in params
