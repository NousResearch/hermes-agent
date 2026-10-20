"""
Composable output-guard pipeline for outbound gateway messages.

Every message the gateway sends to a user — an agent's final reply, a cron
delivery, a status line — can be threaded through an ordered chain of small,
independent validators before it leaves the process. Each validator inspects
(and optionally rewrites or drops) the text. The pipeline is the reusable
part; adding a new rule is one function plus one registry entry.

This generalizes three checks that previously lived as one-off code paths:

  * secret redaction         (``gateway/run.py:_redact_gateway_user_facing_secrets``)
  * provider-error rewriting (``gateway/run.py:_looks_like_gateway_provider_error``)
  * silence-narration drop   (``gateway/delivery.py:_is_silence_narration``)

New guards drop in beside them:

  * em-dash stripping        (opt-in; ``gateway.guards.strip_em_dashes``)
  * link verification        (opt-in, async; ``gateway.guards.verify_links``)

Design
------
A guard is a callable ``(text, ctx) -> GuardOutcome | None``. Returning
``None`` means "no change". Returning a :class:`GuardOutcome` can rewrite the
text (``text=...``) or suppress the message entirely (``drop=True``). Guards
run in registration order and the (possibly rewritten) text threads from one
to the next; a drop short-circuits the rest of the chain.

Guards may be sync or async. :func:`apply_output_guards` awaits async guards;
:func:`apply_output_guards_sync` runs only the sync ones (used on code paths
that are not inside an event loop). Async-only guards (network I/O such as
link checks) are simply skipped by the sync entry point.

Guards are cheap to reason about because each one owns a single concern and
sees the same :class:`GuardContext`. The pipeline never raises: a guard that
throws is logged and skipped so a buggy rule can never block delivery.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =========================================================================
# Data types
# =========================================================================

@dataclass
class GuardContext:
    """Everything a guard needs to decide what to do with a message.

    Attributes:
        platform:          Platform value string ("telegram", "discord", …).
        chat_id:           Destination chat id, when known.
        is_final_response: True for an agent's final reply (vs. a status line
                           or cron delivery). Some guards only apply here.
        metadata:          Free-form send metadata (thread id, job id, …).
    """
    platform: str = ""
    chat_id: Optional[str] = None
    is_final_response: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardOutcome:
    """Result of a single guard.

    ``text`` carries a rewrite (``None`` leaves the running text unchanged).
    ``drop`` suppresses the whole message and short-circuits the chain.
    ``reason`` is a short tag for logging/telemetry.
    """
    text: Optional[str] = None
    drop: bool = False
    reason: Optional[str] = None


GuardResult = Optional[GuardOutcome]
GuardFn = Callable[[str, GuardContext], Union[GuardResult, Awaitable[GuardResult]]]


@dataclass
class _Guard:
    name: str
    fn: GuardFn
    is_async: bool


# =========================================================================
# Pipeline
# =========================================================================

class OutputGuardPipeline:
    """An ordered, fault-isolated chain of output guards."""

    def __init__(self) -> None:
        self._guards: List[_Guard] = []

    def register(self, name: str, fn: GuardFn) -> None:
        """Append a guard. Order of registration is execution order."""
        self._guards.append(
            _Guard(name=name, fn=fn, is_async=inspect.iscoroutinefunction(fn))
        )

    def names(self) -> List[str]:
        return [g.name for g in self._guards]

    def _step(self, outcome: GuardResult, text: str, name: str) -> tuple[str, bool]:
        """Fold one guard's outcome into the running text.

        Returns ``(new_text, dropped)``.
        """
        if outcome is None:
            return text, False
        if outcome.drop:
            logger.info("[output-guard] %s dropped message (%s)", name, outcome.reason or "")
            return text, True
        if outcome.text is not None and outcome.text != text:
            logger.debug("[output-guard] %s rewrote message (%s)", name, outcome.reason or "")
            return outcome.text, False
        return text, False

    def apply_sync(self, text: str, ctx: GuardContext) -> Optional[str]:
        """Run the sync guards only. Returns the final text, or ``None`` to drop.

        Async guards are skipped — use :meth:`apply` on event-loop paths that
        need them (e.g. link verification).
        """
        current = str(text or "")
        for g in self._guards:
            if g.is_async:
                continue
            try:
                outcome = g.fn(current, ctx)  # type: ignore[assignment]
            except Exception as exc:  # never let a guard block delivery
                logger.warning("[output-guard] %s raised (skipped): %s", g.name, exc)
                continue
            current, dropped = self._step(outcome, current, g.name)  # type: ignore[arg-type]
            if dropped:
                return None
        return current

    async def apply(self, text: str, ctx: GuardContext) -> Optional[str]:
        """Run every guard (sync + async). Returns final text, or ``None`` to drop."""
        current = str(text or "")
        for g in self._guards:
            try:
                outcome = g.fn(current, ctx)
                if g.is_async or inspect.isawaitable(outcome):
                    outcome = await outcome  # type: ignore[assignment]
            except Exception as exc:
                logger.warning("[output-guard] %s raised (skipped): %s", g.name, exc)
                continue
            current, dropped = self._step(outcome, current, g.name)  # type: ignore[arg-type]
            if dropped:
                return None
        return current


# =========================================================================
# Config helpers
# =========================================================================

def _guard_flag(key: str, default: bool) -> bool:
    """Read a per-guard on/off flag.

    Env ``HERMES_GUARD_<KEY>`` overrides config; otherwise
    ``gateway.guards.<key>`` in config.yaml wins, falling back to ``default``.
    """
    env = os.getenv(f"HERMES_GUARD_{key.upper()}")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        guards = ((cfg.get("gateway", {}) or {}).get("guards", {}) or {})
        val = guards.get(key)
        if val is not None:
            return bool(val)
    except Exception:
        pass
    return default


# =========================================================================
# Built-in guards
# =========================================================================
# Each guard is a thin adapter around logic that already existed elsewhere in
# the gateway, re-expressed as a pipeline stage. Keeping the heavy lifting in
# the original modules (run.py, delivery.py) avoids duplicating regexes; the
# guards here delegate to those helpers and decide drop-vs-rewrite.


def _secret_redaction_guard(text: str, ctx: GuardContext) -> GuardResult:
    """Redact anything that looks like a credential before it leaves."""
    try:
        from gateway.run import _redact_gateway_user_facing_secrets
    except Exception:
        return None
    redacted = _redact_gateway_user_facing_secrets(text)
    if redacted != text:
        return GuardOutcome(text=redacted, reason="secret")
    return None


def _provider_error_guard(text: str, ctx: GuardContext) -> GuardResult:
    """Rewrite raw provider/API error envelopes into a short safe reply.

    Applies to every chat surface, not just Telegram. Upstream widened this
    invariant from Telegram (#28533) to all chat platforms (#39293): a provider
    error body can carry a leaked bearer token, request IDs, or policy text,
    and none of it should reach a chat user on any platform. The caller
    (``_sanitize_gateway_final_response``) has already excluded programmatic
    surfaces via ``_GATEWAY_RAW_TEXT_PLATFORMS``, so anything reaching this
    guard is a human-facing surface that should get the safe category instead
    of the raw envelope.
    """
    try:
        from gateway.run import (
            _looks_like_gateway_provider_error,
            _gateway_provider_error_reply,
        )
    except Exception:
        return None
    if _looks_like_gateway_provider_error(text):
        return GuardOutcome(text=_gateway_provider_error_reply(text), reason="provider-error")
    return None


def _silence_narration_guard(text: str, ctx: GuardContext) -> GuardResult:
    """Drop hallucinated silence tokens (*(silent)*, 🔇, a bare '.') entirely."""
    try:
        from gateway.delivery import _is_silence_narration
    except Exception:
        return None
    if _is_silence_narration(text):
        return GuardOutcome(drop=True, reason="silence-narration")
    return None


# --- em-dash stripping (opt-in) ------------------------------------------
# A single user-facing style rule expressed as a guard: replace em/en dashes
# with typographically safe substitutes. Off by default so it never surprises
# other deployments; enable with gateway.guards.strip_em_dashes: true.

_EM_DASH_RE = re.compile(r"\s*[\u2014\u2013]\s*")


def _em_dash_guard(text: str, ctx: GuardContext) -> GuardResult:
    if not _guard_flag("strip_em_dashes", default=False):
        return None
    if "\u2014" not in text and "\u2013" not in text:
        return None
    # Replace a dash flanked by spaces with ", " (clause break); a bare dash
    # between words becomes a plain hyphen-free comma too. Collapse doubles.
    rewritten = _EM_DASH_RE.sub(", ", text)
    rewritten = re.sub(r",\s*,", ", ", rewritten)
    if rewritten != text:
        return GuardOutcome(text=rewritten, reason="em-dash")
    return None


# --- link verification (opt-in, async) -----------------------------------
# Never emit a URL the gateway hasn't confirmed resolves. Off by default
# (adds latency + network I/O); enable with gateway.guards.verify_links: true.

_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+")
_LINK_TIMEOUT = 4.0


async def _verify_links_guard(text: str, ctx: GuardContext) -> GuardResult:
    if not _guard_flag("verify_links", default=False):
        return None
    urls = list(dict.fromkeys(_URL_RE.findall(text)))
    if not urls:
        return None
    try:
        import aiohttp
    except Exception:
        logger.debug("[output-guard] verify_links needs aiohttp; skipping")
        return None

    async def _resolves(session, url: str) -> bool:
        for method in (session.head, session.get):
            try:
                async with method(url, allow_redirects=True,
                                  timeout=aiohttp.ClientTimeout(total=_LINK_TIMEOUT)) as resp:
                    if resp.status < 400:
                        return True
                    if resp.status in (403, 405) and method is session.head:
                        continue  # some hosts reject HEAD; try GET
                    return False
            except Exception:
                continue
        return False

    dead: List[str] = []
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_resolves(session, u) for u in urls])
    dead = [u for u, ok in zip(urls, results) if not ok]
    if not dead:
        return None
    # Strip the dead links inline and flag them, rather than dropping the
    # whole message (the surrounding prose is usually still useful).
    rewritten = text
    for u in dead:
        rewritten = rewritten.replace(u, "[link removed: did not resolve]")
    logger.info("[output-guard] verify_links removed %d dead link(s)", len(dead))
    return GuardOutcome(text=rewritten, reason="dead-links")


# =========================================================================
# Default pipeline (singleton)
# =========================================================================

_default_pipeline: Optional[OutputGuardPipeline] = None


def get_default_pipeline() -> OutputGuardPipeline:
    """Return the process-wide default pipeline, building it on first use.

    Order matters: redact secrets first (so nothing downstream can leak a
    key), then rewrite provider errors, then style rules (em-dash), then
    network checks (links), and finally the silence drop last so a message
    that got rewritten to empty-ish still gets the drop check.
    """
    global _default_pipeline
    if _default_pipeline is None:
        p = OutputGuardPipeline()
        p.register("secret", _secret_redaction_guard)
        p.register("provider-error", _provider_error_guard)
        p.register("em-dash", _em_dash_guard)
        p.register("verify-links", _verify_links_guard)
        p.register("silence-narration", _silence_narration_guard)
        _default_pipeline = p
    return _default_pipeline


def reset_default_pipeline() -> None:
    """Drop the cached pipeline (tests toggle config between builds)."""
    global _default_pipeline
    _default_pipeline = None


def apply_output_guards_sync(text: str, ctx: GuardContext) -> Optional[str]:
    """Convenience wrapper: run sync guards of the default pipeline."""
    return get_default_pipeline().apply_sync(text, ctx)


async def apply_output_guards(text: str, ctx: GuardContext) -> Optional[str]:
    """Convenience wrapper: run all guards of the default pipeline."""
    return await get_default_pipeline().apply(text, ctx)
