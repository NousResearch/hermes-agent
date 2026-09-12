"""Dialectic layer for HonchoMemoryProvider: multi-pass reasoning, cadence/backoff, liveness."""

from __future__ import annotations

import logging
import re
import threading
import time

from plugins.memory.honcho.client import spawn_context_thread

logger = logging.getLogger("plugins.memory.honcho")

_STRUCTURE_RE = re.compile(r"^(?:[*-] |\s*\d+\. )", re.MULTILINE)  # bullet / ordered-list line


class DialecticMixin:
    """Dialectic prefetch state machine. State attributes are initialised in
    HonchoMemoryProvider.__init__; this mixin only reads/writes them."""

    # Reasoning level per (depth, pass) when dialecticDepthLevels is not configured;
    # "base" means dialecticReasoningLevel. Early passes of deeper runs are lighter.
    _PROPORTIONAL_LEVELS: dict[tuple[int, int], str] = {
        (1, 0): "base", (2, 0): "minimal", (2, 1): "base", (3, 0): "minimal", (3, 1): "base", (3, 2): "low",
    }
    _LEVEL_ORDER = ("minimal", "low", "medium", "high", "max")
    # Char-count thresholds for the query-length reasoning heuristic.
    _HEURISTIC_LENGTH_MEDIUM, _HEURISTIC_LENGTH_HIGH = 120, 400
    # A thread older than timeout × multiplier is treated as dead so a hung Honcho call
    # can't block future fires indefinitely.
    _STALE_THREAD_MULTIPLIER = 2.0
    # A pending result fired more than cadence × multiplier turns ago is discarded on read,
    # so a stale conversational pivot isn't injected after trivial-prompt turns.
    _STALE_RESULT_MULTIPLIER = 2
    _BACKOFF_MAX = 8  # ceiling for the empty-streak backoff (× base cadence)
    # Turn-1 budgets: session init + base retrieval share the first; dialectic gets the second.
    _FIRST_TURN_BASE_TIMEOUT, _FIRST_TURN_DIALECTIC_CAP = 3.0, 2.0

    def _thread_is_live(self) -> bool:
        """Thread-alive guard that treats threads past the stale threshold as dead."""
        if not self._prefetch_thread or not self._prefetch_thread.is_alive():
            return False
        timeout = self._config.timeout if self._config and self._config.timeout else 8.0
        stale_after = timeout * self._STALE_THREAD_MULTIPLIER
        age = time.monotonic() - self._prefetch_thread_started_at
        if age > stale_after:
            logger.debug("Honcho prefetch thread age %.1fs exceeds stale threshold %.1fs — treating as dead", age, stale_after)
            return False
        return True

    def _effective_cadence(self) -> int:
        """Cadence plus empty-streak backoff, capped at _BACKOFF_MAX × base."""
        cadence, streak = self._dialectic_cadence, self._dialectic_empty_streak
        return min(cadence + streak, cadence * self._BACKOFF_MAX) if streak > 0 else cadence

    def _note_dialectic_failure(self, exc: BaseException) -> None:
        """Widen the empty-streak backoff after a failed cycle; auth failures are exempt
        because waiting cannot fix a dead token."""
        from plugins.memory.honcho.session import HonchoAuthError
        if isinstance(exc, HonchoAuthError):
            logger.warning("Honcho dialectic auth failure (not counted toward cadence backoff): %s", exc)
            return
        self._dialectic_empty_streak += 1

    def _spawn_dialectic(
        self, query: str, *, thread_name: str, fired_at: int, log_label: str, use_query_rewrite: bool = True,
    ) -> threading.Thread:
        """Start a background dialectic run that publishes into the pending-result slot. Only a
        non-empty result advances ``_last_dialectic_turn`` (so empty returns retry next turn) and
        resets the empty streak; failures widen the backoff."""
        def _run() -> None:
            try:
                r = self._run_dialectic_depth(query, use_query_rewrite=use_query_rewrite)
            except Exception as exc:
                logger.debug("Honcho %s failed: %s", log_label, exc)
                self._note_dialectic_failure(exc)
                return
            if r and r.strip():
                with self._prefetch_lock:
                    self._prefetch_result = r
                    self._prefetch_result_fired_at = fired_at
                self._last_dialectic_turn = fired_at
                self._dialectic_empty_streak = 0
            else:
                self._dialectic_empty_streak += 1

        self._prefetch_thread_started_at = time.monotonic()
        thread = spawn_context_thread(_run, name=thread_name)
        thread.start()
        self._prefetch_thread = thread
        return thread

    def _consume_pending_dialectic(self) -> str:
        """Pop the pending dialectic result, or "" when none is ready or it is stale."""
        with self._prefetch_lock:
            dialectic_result, fired_at = self._prefetch_result, self._prefetch_result_fired_at
            self._prefetch_result, self._prefetch_result_fired_at = "", -999
        stale_limit = self._dialectic_cadence * self._STALE_RESULT_MULTIPLIER
        if dialectic_result and fired_at >= 0 and (self._turn_count - fired_at) > stale_limit:
            logger.debug("Honcho pending dialectic discarded as stale: fired_at=%d, turn=%d, limit=%d",
                         fired_at, self._turn_count, stale_limit)
            return ""
        return dialectic_result if (dialectic_result and dialectic_result.strip()) else ""

    def _apply_reasoning_heuristic(self, base: str, query: str) -> str:
        """Scale ``base`` up by query length (+1 at >=120 chars, +2 at >=400), clamped at reasoning_level_cap."""
        if not self._reasoning_heuristic or not query or base not in self._LEVEL_ORDER:
            return base
        n = len(query)
        bump = 0 if n < self._HEURISTIC_LENGTH_MEDIUM else 1 if n < self._HEURISTIC_LENGTH_HIGH else 2
        cap_idx = self._LEVEL_ORDER.index(self._reasoning_level_cap)
        return self._LEVEL_ORDER[min(self._LEVEL_ORDER.index(base) + bump, cap_idx)]

    def _resolve_pass_level(self, pass_idx: int, query: str = "") -> str:
        """Reasoning level for a pass: explicit dialecticDepthLevels win, then the
        proportional table, then the base level scaled by the length heuristic."""
        if self._dialectic_depth_levels and pass_idx < len(self._dialectic_depth_levels):
            return self._dialectic_depth_levels[pass_idx]
        mapping = self._PROPORTIONAL_LEVELS.get((self._dialectic_depth, pass_idx))
        if mapping is None or mapping == "base":
            base = self._config.dialectic_reasoning_level if self._config else "low"
            return self._apply_reasoning_heuristic(base, query)
        return mapping

    def _build_dialectic_prompt(self, pass_idx: int, prior_results: list[str], is_cold: bool) -> str:
        """Pass 0: cold (general) or warm (session-scoped) query; pass 1: gap audit
        against the prior result; pass 2: reconciliation across prior passes."""
        if pass_idx == 0:
            if is_cold:
                return ("Who is this person? What are their preferences, goals, and working style? "
                        "Focus on facts that would help an AI assistant be immediately useful.")
            return ("Given what's been discussed in this session so far, what context about this user is most "
                    "relevant to the current conversation? Prioritize active context over biographical facts.")
        if pass_idx == 1:
            prior = prior_results[-1] if prior_results else ""
            return (f"Given this initial assessment:\n\n{prior}\n\n"
                    "What gaps remain in your understanding that would help going forward? Synthesize what you "
                    "actually know about the user's current state and immediate needs, grounded in evidence from "
                    "recent sessions.")
        p0, p1 = [prior_results[i] if len(prior_results) > i else "(empty)" for i in (0, 1)]
        return (f"Prior passes produced:\n\nPass 1:\n{p0}\n\nPass 2:\n{p1}\n\n"
                "Do these assessments cohere? Reconcile any contradictions and produce a final, concise "
                "synthesis of what matters most for the current conversation.")

    @staticmethod
    def _signal_sufficient(result: str) -> bool:
        """True when a pass returned enough signal to skip further passes: >100 chars
        with structure (headers/bullets/ordered list), or >300 chars regardless."""
        if not result or len(result.strip()) < 100:
            return False
        structured = "\n" in result and bool("##" in result or "•" in result or _STRUCTURE_RE.search(result))
        return structured or len(result.strip()) > 300

    def _run_dialectic_depth(self, query: str, *, use_query_rewrite: bool = True) -> str:
        """Run up to dialecticDepth .chat() passes, bailing early once a pass returns
        strong signal. Returns the last non-empty result."""
        if not self._manager or not self._session_key:
            return ""
        is_cold = not self._base_context_cache
        results: list[str] = []
        rewritten_query = ""
        if use_query_rewrite and self._query_rewrite_enabled and self._query_rewriter:
            try:
                rewritten_query = self._query_rewriter(query).strip()
            except Exception as exc:
                logger.debug("Honcho query rewriter failed: %s", exc)

        depth = self._dialectic_depth
        for i in range(depth):
            # Dependent prompts require a non-empty prior result; without one, later
            # passes retry the independent base prompt.
            prior_results = [r for r in results if r and r.strip()]
            if prior_results and self._signal_sufficient(prior_results[-1]):
                logger.debug("Honcho dialectic depth %d: pass %d skipped, prior signal sufficient", depth, i)
                break
            if prior_results:
                prompt = self._build_dialectic_prompt(i, prior_results, is_cold)
            else:
                if i:
                    logger.debug("Honcho dialectic depth %d: pass %d has no non-empty prior — "
                                 "falling back to base prompt", depth, i)
                prompt = rewritten_query or self._build_dialectic_prompt(0, [], is_cold)
            level = self._resolve_pass_level(i, query=query)
            logger.debug("Honcho dialectic depth %d: pass %d, level=%s, cold=%s", depth, i, level, is_cold)
            result = self._manager.dialectic_query(self._session_key, prompt, reasoning_level=level, peer="user")
            results.append(result or "")
        return next((r for r in reversed(results) if r and r.strip()), "")
