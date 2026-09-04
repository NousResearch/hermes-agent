"""One owner for "the primary's credentials are all benched — route around it".

``resolve_runtime_provider`` reports a fully rate-limited credential pool by
annotating the runtime it hands back (``CREDENTIALS_COOLING_DOWN_KEY``); it
deliberately does not raise, because the same function also answers status
probes, model pickers and readiness checks, for which a cooling-down provider
is still a configured one.  Acting on that report is the caller's decision.

Four callers now own a fallback chain and need the identical three answers --
the gateway, the interactive CLI (which is also the kanban worker's entry
point, since a card dispatches as ``hermes chat -q``), cron jobs, and one-shot
runs.  Duplicating the walk four times is how the same entry starts resolving
differently depending on who asked, so the policy lives here and each caller
supplies only its own chain.

The policy itself, in one line: a cooldown DEMOTES a provider, it does not
disqualify it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def runtime_is_rate_limited(runtime: Optional[dict[str, Any]]) -> bool:
    """Whether *runtime* draws on a credential pool that is serving a 429.

    Fail-open: an unreadable pool must never veto a usable fallback.
    """
    if not isinstance(runtime, dict):
        return False
    try:
        from hermes_cli.runtime_provider import (
            runtime_credentials_cooling_down_until,
        )

        return runtime_credentials_cooling_down_until(runtime) is not None
    except Exception:
        logger.debug(
            "Could not probe %s's pool for a cooldown",
            runtime.get("provider") or "?",
            exc_info=True,
        )
        return False


def resolve_fallback_entry_runtime(
    entry: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Resolve one fallback-chain entry, or ``None`` when it is unusable.

    ``target_model`` matters: without it the entry's api_mode is derived from
    the *primary's* model, which can pick the wrong wire protocol for the
    fallback's endpoint.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.fallback_config import resolve_entry_api_key

    try:
        return resolve_runtime_provider(
            requested=entry.get("provider"),
            target_model=(entry.get("model") or "").strip() or None,
            explicit_base_url=entry.get("base_url"),
            explicit_api_key=resolve_entry_api_key(entry),
        )
    except Exception as exc:
        # A chain entry that is simply unconfigured is an ordinary outcome, so
        # this stays at debug -- but the catch is broad enough to swallow a
        # genuine resolver fault, and losing that traceback is how "the
        # fallback just never fires" becomes unexplainable.
        logger.debug(
            "Fallback entry %s failed: %s", entry.get("provider"), exc,
            exc_info=True,
        )
        return None


def resolve_non_cooling_fallback_runtime(
    chain: Optional[list[dict[str, Any]]],
    *,
    is_rate_limited: Optional[Callable[[dict[str, Any]], bool]] = None,
    resolve_entry: Optional[
        Callable[[dict[str, Any]], Optional[dict[str, Any]]]
    ] = None,
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[dict[str, Any]]]:
    """First entry in *chain* that resolves AND is not itself rate-limited.

    Used when the primary's pooled credentials are all benched by a 429.
    Picking a fallback whose own pool is cooling down would just move the
    doomed request one hop down the chain, so those entries are passed over
    in favour of a later, healthy one.

    Returns ``(runtime, model, entry)`` for the chosen entry, or a triple of
    ``None`` when nothing in the chain is usable — the caller then keeps the
    primary, because a cooldown demotes a provider rather than disqualifying
    it.

    The *entry* comes back because resolving it once is not enough for a
    long-lived caller: an entry defined by an inline ``base_url`` rather than
    a registered provider name cannot be re-resolved from its provider name
    alone, so a caller that will resolve again next turn needs the entry to
    pin the same endpoint.

    ``is_rate_limited`` / ``resolve_entry`` exist so a caller that already
    owns these decisions (the gateway) keeps its own module-level names as
    the seam, rather than having two spellings of the same predicate that
    can be patched or evolved apart.
    """
    _is_rate_limited = is_rate_limited or runtime_is_rate_limited
    _resolve_entry = resolve_entry or resolve_fallback_entry_runtime

    cooling_but_resolvable: Optional[
        tuple[dict[str, Any], str, dict[str, Any]]
    ] = None
    for entry in chain or []:
        if not isinstance(entry, dict):
            continue
        model = (entry.get("model") or "").strip()
        if not model:
            # Swapping the provider while keeping the primary's model name
            # would send e.g. a Gemini model id to OpenRouter.
            logger.warning(
                "Fallback entry %s has no model — skipping",
                entry.get("provider") or "?",
            )
            continue
        runtime = _resolve_entry(entry)
        if runtime is None:
            continue
        if _is_rate_limited(runtime):
            logger.info(
                "Fallback %s is itself rate-limited — looking further down "
                "the chain", entry.get("provider") or "?",
            )
            if cooling_but_resolvable is None:
                cooling_but_resolvable = (runtime, model, entry)
            continue
        logger.info(
            "Fallback provider resolved: %s model=%s",
            entry.get("provider") or runtime.get("provider"), model,
        )
        return runtime, model or None, entry

    # Everything left is cooling too. A benched fallback still beats a benched
    # primary that has no chance of a different quota bucket.
    if cooling_but_resolvable is not None:
        runtime, model, entry = cooling_but_resolvable
        logger.warning(
            "Every fallback is rate-limited too — using the first one anyway"
        )
        return runtime, model or None, entry
    return None, None, None


def cooldown_label(until: float) -> str:
    """Local wall-clock spelling of a cooldown reset, for the operator.

    A corrupt persisted timestamp must not take down the turn it is only being
    printed in, so an unrepresentable value degrades to the raw epoch.
    """
    from datetime import datetime

    try:
        return datetime.fromtimestamp(until).strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return f"epoch {until}"


@dataclass(frozen=True)
class Demotion:
    """What :func:`demote_if_rate_limited` decided.

    ``switched`` is the one question most callers ask; ``entry`` matters only
    to a caller that will resolve again on a later turn and must land on the
    same endpoint.
    """

    runtime: dict[str, Any]
    model: Optional[str] = None
    cooling_until: Optional[float] = None
    entry: Optional[dict[str, Any]] = None

    @property
    def switched(self) -> bool:
        """Whether a fallback actually took over.

        Keyed on ``entry`` because that is the thing the walk sets last and
        only on success. ``model`` is guaranteed non-empty alongside it: the
        walk refuses a model-less entry outright, since swapping the provider
        while keeping the primary's model name would send e.g. a Gemini model
        id to OpenRouter.
        """
        return self.entry is not None

    def explicit_pins(self) -> tuple[Optional[str], Optional[str]]:
        """``(base_url, api_key)`` a caller should pin to re-reach this route.

        A caller that resolves again on a later turn cannot get back to a
        chain entry defined by an inline ``base_url`` from its provider name
        alone. Reading that off the entry is this object's job, not the
        caller's -- the entry is chain-config shape, and only this module
        should have to know it.
        """
        from hermes_cli.fallback_config import resolve_entry_api_key

        entry = self.entry or {}
        return (entry.get("base_url") or None, resolve_entry_api_key(entry) or None)


def demote_if_rate_limited(
    runtime: dict[str, Any],
    chain: Optional[list[dict[str, Any]]],
    *,
    subject: str = "Primary provider",
    is_rate_limited: Optional[Callable[[dict[str, Any]], bool]] = None,
    resolve_entry: Optional[
        Callable[[dict[str, Any]], Optional[dict[str, Any]]]
    ] = None,
) -> "Demotion":
    """Swap *runtime* for a healthy fallback when its own pool is serving a 429.

    The decision every caller shares, so the walk is not the only part with one
    owner: ask whether this runtime is benched, and if it is, take the first
    entry in *chain* that is not benched too.

    Returns a :class:`Demotion`.

    * ``runtime`` is the fallback when one was usable, otherwise the one passed
      in -- a cooldown DEMOTES a provider, it does not disqualify it, and the
      real upstream 429 beats refusing to run.
    * ``model`` is the fallback's model, or ``None`` when nothing was swapped.
      Provider and model move together: swapping one while keeping the other
      would send e.g. a Gemini model id to OpenRouter.
    * ``cooling_until`` is the reset time whenever the runtime WAS benched,
      including the case where nothing in the chain could take over. A caller
      that owes a return (a long-lived session) needs that time even on the
      turn it had to stay put.
    * ``entry`` is the chain entry that took over, for a caller that must pin
      the same endpoint when it resolves again.

    *subject* names the demoted party in the log line, so a cron job can say
    which job it was.
    """
    from hermes_cli.runtime_provider import runtime_credentials_cooling_down_until

    cooling_until = runtime_credentials_cooling_down_until(runtime)
    if not cooling_until:
        return Demotion(runtime)

    fb_runtime, fb_model, fb_entry = resolve_non_cooling_fallback_runtime(
        chain, is_rate_limited=is_rate_limited, resolve_entry=resolve_entry
    )
    if fb_runtime is None:
        logger.warning(
            "%s: no usable fallback while %s cools down — spending the "
            "rate-limited credential as a last resort",
            subject,
            runtime.get("provider") or "?",
        )
        return Demotion(runtime, cooling_until=cooling_until)

    logger.warning(
        "%s: %s is rate-limited until %s — using fallback %s/%s until it lifts",
        subject,
        runtime.get("provider") or "?",
        cooldown_label(cooling_until),
        fb_runtime.get("provider") or "?",
        fb_model,
    )
    return Demotion(fb_runtime, fb_model, cooling_until, fb_entry)
