"""Rate-limit / credits header capture and low-credit notices for ``AIAgent``.

Parses provider response headers into ``_rate_limit_state`` / ``_credits_state`` and emits sticky notices.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import os
from typing import Any

from utils import is_truthy_value

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")



def _is_nous_agent(agent) -> bool:
    """True for the Nous provider (or a route to a nousresearch.com host)."""
    if (getattr(agent, "provider", "") or "").strip().lower() == "nous":
        return True
    from utils import base_url_host_matches
    base_url = str(getattr(agent, "base_url", "") or "")
    return bool(base_url) and base_url_host_matches(base_url, "nousresearch.com")

def _response_headers(http_response: Any):
    """Headers of a response, or None when there is nothing to parse."""
    return getattr(http_response, "headers", None) if http_response is not None else None


def _pct(fraction) -> str:
    return ("%.0f%%" % (fraction * 100)) if fraction is not None else "n/a"


def _adopt_credits_state(agent, state) -> None:
    """Retain-last-known: overwrite state and latch session-start remaining on the first header ever seen."""
    agent._credits_state = state
    if agent._credits_session_start_micros is None:
        agent._credits_session_start_micros = state.remaining_micros


class RateLimitCreditsMixin:
    """Rate-limit + credits header capture and notices (see module docstring)."""

    def _capture_rate_limits(self, http_response: Any) -> None:
        """Parse x-ratelimit-* headers from an HTTP response and cache the state (never raises)."""
        headers = _response_headers(http_response)
        if not headers:
            return
        try:
            from agent.rate_limit_tracker import parse_rate_limit_headers
            state = parse_rate_limit_headers(headers, provider=self.provider)
            if state is not None:
                self._rate_limit_state = state
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_rate_limit_state(self):
        """Return the last captured RateLimitState, or None."""
        return self._rate_limit_state

    def _capture_nous_model_switch(self, http_response: Any) -> None:
        """Record the Nous gateway's ``x-nous-model-switch`` header (a named account asked for the
        free tier's model; the gateway served its backing model and named it). Applied between
        calls by ``hermes_cli.anon_auth.apply_model_switch``. Fail-open."""
        headers = _response_headers(http_response)
        if not headers:
            return
        try:
            from hermes_cli.anon_auth import note_model_switch
            note_model_switch(self, headers)
        except Exception:
            pass  # Never let header parsing break the agent loop

    def _capture_anthropic_response_headers(self, http_response: Any) -> None:
        """Capture rate-limit + credits state from Anthropic Messages response headers (the SDK's
        aggregated ``Message`` drops them). Fail-open."""
        self._capture_rate_limits(http_response)
        self._capture_credits(http_response)

    def _capture_credits(self, http_response: Any, *, refused: bool = False) -> None:
        """Parse x-nous-credits-* headers, cache CreditsState, fire threshold notices.

        The PARSE is swallowed (miss → keep last-known); notice EVALUATION WARNS on failure so a
        depletion-notice bug cannot vanish silently. HERMES_DEV_CREDITS_FIXTURE injects a chosen state instead.
        """
        try:
            from agent.credits_tracker import dev_fixture_credits_state
            fixture = dev_fixture_credits_state()
        except Exception:
            fixture = None
        if fixture is not None:
            _adopt_credits_state(self, fixture)
            latch = getattr(self, "_credits_latch", None)
            if isinstance(latch, dict):
                # Only seen_below_90 — priming seen_grant_unspent would fire grant_spent on first observation.
                latch["seen_below_90"] = True  # let warn90 fire without a real crossing
            logger.info(
                "credits ▸ [FIXTURE] remaining=%d (%s) · paid=%s · denom=%s · used=%s "
                "(real headers bypassed — `echo clear` / unset HERMES_DEV_CREDITS_FIXTURE to restore)",
                fixture.remaining_micros, fixture.remaining_usd or "?", fixture.paid_access, fixture.denominator_kind,
                _pct(fixture.used_fraction))
            self._emit_credits_notices()
            return
        headers = _response_headers(http_response)
        if not headers:
            return
        dev = is_truthy_value(os.environ.get("HERMES_DEV_CREDITS"))

        # Parse: fail-open → miss; never overwrite good state with None.
        try:
            from agent.credits_tracker import parse_credits_headers
            state = parse_credits_headers(headers, provider=self.provider)
        except Exception:
            return
        if state is None:
            if dev:
                logger.info("credits ▸ response had no valid x-nous-credits-* headers "
                            "(miss — producer off / non-Nous path / >TTL stale)")
            return

        if state.is_usage_window and not _is_nous_agent(self):
            return  # window headers from any other upstream are ignored (no window UI, no Keep going button)
        if state.is_usage_window and not any(k.lower() == "x-nous-credits-purchased-micros" for k in headers):
            # A refused request's headers omit the top-up balance: keep the last known one, not "no credits".
            prior = getattr(self, "_credits_state", None)
            if prior is not None:
                import dataclasses
                state = dataclasses.replace(state, purchased_micros=prior.purchased_micros,
                                            purchased_usd=prior.purchased_usd)
        if state.is_usage_window and not refused and state.spend_source is None:
            import dataclasses
            state = dataclasses.replace(state, unmetered=True)  # admitted but not metered: no limit stop from it
        _adopt_credits_state(self, state)
        try:
            # A reply that contradicts the cached plan (upgrade, cancel, a limit filled or reset)
            # refreshes it once in the background, so pickers catch up without a new session.
            from agent.plan_snapshot_sync import sync_plan_snapshot
            sync_plan_snapshot(state)
        except Exception:
            logger.debug("Plan snapshot sync failed", exc_info=True)
        if dev:
            # HERMES_DEV_CREDITS streams each capture to agent.log (`hermes logs -f`, grep 'credits ▸').
            spent = self.get_credits_spent_micros()
            logger.info(
                "credits ▸ remaining=%d (%s) · paid=%s · denom=%s · used=%s · Δspent=%s · age=%s%s",
                state.remaining_micros, state.remaining_usd or "?", state.paid_access, state.denominator_kind,
                _pct(state.used_fraction), ("%.1f¢" % (spent / 10000)) if spent is not None else "n/a",
                (f"{state.age_seconds:.0f}s") if state.age_seconds != float("inf") else "n/a",
                (f" · disabled={state.disabled_reason}") if state.disabled_reason else "")
        self._emit_credits_notices()

    def _emit_credits_notices(self) -> None:
        """Run the threshold policy and emit notices (shared by the warm path and the cold-start seed).

        Runs only when a notice consumer is bound; WARNS on failure; clears FIRST so depleted lands last.
        """
        if getattr(self, "notice_callback", None) is None and getattr(self, "notice_clear_callback", None) is None:
            return
        state = getattr(self, "_credits_state", None)
        if not self._credits_notices_enabled() or state is None:
            return
        try:
            from agent.credits_tracker import (
                _remember_shown_band,
                evaluate_credits_notices, is_free_tier_model, new_credits_latch, rewarm_pricing_before_depleted_notice,
            )
            latch = getattr(self, "_credits_latch", None)
            if latch is None:
                latch = self._credits_latch = new_credits_latch()
            # Free-model gate: a depleted account can still inference on a free model. Local data only.
            model_is_free = is_free_tier_model(getattr(self, "model", "") or "", getattr(self, "base_url", "") or "")
            # Any blocking state (depleted, or a used-up window / large-model allowance) needs a warm catalog to
            # tell a model billed elsewhere apart once the 300s pricing cache lapses.
            blocking = state.depleted or state.window_exhausted or state.top_tier_exhausted
            if blocking and not model_is_free and rewarm_pricing_before_depleted_notice(self):
                return  # a cold catalog cannot say the model is billed elsewhere; the warm's re-run decides
            model_is_top_tier = False
            if state.is_usage_window and not model_is_free:
                # The top-tier approach warning only matters while on a top-tier model (gateway mark).
                from agent.plan_limits import cached_pricing, is_top_tier
                model_is_top_tier = is_top_tier(getattr(self, "model", "") or "",
                                                cached_pricing(getattr(self, "base_url", "") or ""))
            session_id = getattr(self, "session_id", None)
            if state.is_usage_window:
                from agent.credits_tracker import restore_window_notices
                restore_window_notices(session_id, latch)
            to_show, to_clear = evaluate_credits_notices(
                state, latch, model_is_free=model_is_free, model_is_top_tier=model_is_top_tier)
            if state.is_usage_window:
                from agent.credits_tracker import remember_window_notices
                remember_window_notices(session_id, latch)
            elif session_id:
                from agent.credits_tracker import forget_window_notices
                forget_window_notices(session_id)
            for key in to_clear:
                self._emit_notice_clear(key)
            for notice in to_show:
                self._emit_notice(notice)
            # Record the band this session last showed so a desktop reap/resume rebuild
            # (fresh agent + fresh latch, SAME session_id) restores it instead of
            # re-announcing the unchanged band as a fresh crossing (#101578). Both the
            # live-header path and the cold-start seed route through here, so this is the
            # single chokepoint.
            _remember_shown_band(getattr(self, "session_id", None), latch.get("usage_band"))
        except Exception:
            logger.warning("credits notice evaluation/emit failed", exc_info=True)

    def _credits_notices_enabled(self) -> bool:
        """``display.credits_notices``, read once per agent and cached (UI noise, not correctness); fail-open True."""
        cached = getattr(self, "_credits_notices_enabled_cache", None)
        if cached is not None:
            return cached
        enabled = True
        try:
            from hermes_cli.config import load_config
            display = (load_config() or {}).get("display")
            if isinstance(display, dict) and "credits_notices" in display:
                enabled = bool(display["credits_notices"])
        except Exception:
            pass
        self._credits_notices_enabled_cache = enabled
        return enabled

    def get_credits_state(self):
        """Return the last captured CreditsState, or None."""
        return self._credits_state

    def get_credits_spent_micros(self):
        """Session-cumulative micros spent = first_seen_remaining - current_remaining. None if no data."""
        if self._credits_session_start_micros is None or self._credits_state is None:
            return None
        return self._credits_session_start_micros - self._credits_state.remaining_micros

    def _check_openrouter_cache_status(self, http_response: Any) -> None:
        """Log X-OpenRouter-Cache-Status; HITs count in ``_or_cache_hits``. Never raises."""
        headers = _response_headers(http_response)
        if not headers:
            return
        try:
            status = headers.get("x-openrouter-cache-status")
            if not status:
                return
            if status.upper() == "HIT":
                self._or_cache_hits += 1
                logger.info("OpenRouter response cache HIT (total: %d)", self._or_cache_hits)
            else:
                logger.debug("OpenRouter response cache %s", status.upper())
        except Exception:
            pass  # Never let header parsing break the agent loop
