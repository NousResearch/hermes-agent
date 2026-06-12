"""Credits tracking for Nous inference API responses.

Parses x-nous-credits-* (and optional x-nous-tool-pool-*) headers from
inference responses into a validated CreditsState dataclass.  Provides
depletion detection (paid_access), subscription-cap used_fraction, and
warn-once schema-version gating.  This is the hardened parser used by all
live consumers (run_agent, tui_gateway) — not a dev-only shim.

Header schema (x-nous-credits-* family):
    x-nous-credits-version                    contract/schema version
    x-nous-credits-remaining-micros           total remaining balance (micros)
    x-nous-credits-remaining-usd              same, formatted USD string
    x-nous-credits-subscription-micros        subscription balance (SIGNED; may be negative/debt)
    x-nous-credits-subscription-usd           same, formatted USD string
    x-nous-credits-subscription-limit-micros  subscription cap (PAIRED/optional)
    x-nous-credits-subscription-limit-usd     same, formatted USD string (PAIRED/optional)
    x-nous-credits-rollover-micros            rolled-over balance (micros)
    x-nous-credits-purchased-micros           purchased balance (micros)
    x-nous-credits-purchased-usd              same, formatted USD string
    x-nous-credits-denominator-kind           "subscription_cap" | "none"
    x-nous-credits-paid-access                "true" | "false" (STRING!)
    x-nous-credits-disabled-reason            reason string (header omitted when null)
    x-nous-credits-as-of-ms                   server-side timestamp (ms epoch)

Tool-pool headers use a SEPARATE prefix:
    x-nous-tool-pool-micros                   tool-pool balance (micros)
    x-nous-tool-pool-gated-off                "true" | "false" (STRING!)

Money is handled as micros ints only; *_usd values are preserved verbatim as
the raw strings the server sent (never re-parsed to float).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from utils import is_truthy_value

logger = logging.getLogger(__name__)

# Warn-once latch: emit the version-unsupported warning at most once per process.
_version_warning_emitted: bool = False

# Valid denominator kinds (exhaustive set from the API contract).
_VALID_DENOMINATOR_KINDS = frozenset({"subscription_cap", "none"})

# USD format: optional leading minus, one-or-more digits, dot, exactly 2 digits.
_USD_RE = re.compile(r"^-?\d+\.\d{2}$")


# ── Internal helpers ─────────────────────────────────────────────────────────


_SENTINEL = object()  # singleton sentinel for "parse failed"


def _safe_int(value: Any) -> Any:
    """Parse a header value to an exact int (money-safe).

    The contract guarantees every ``*_micros`` field is an integer string —
    we parse with ``int()`` directly, NOT ``int(float(...))``, to avoid float-
    precision loss above 2**53 that would silently corrupt large money values.

    Returns the parsed int, or ``_SENTINEL`` if the value is not a valid integer
    string (including float-shaped strings like "1.5").  The sentinel lets callers
    detect the failure and return None from the overall parse (fail-hard-on-bad-
    input, not silently coerce).
    """
    if value is None:
        return _SENTINEL
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return _SENTINEL



def _validate_usd(value: Optional[str]) -> bool:
    """Return True iff value is a non-None string matching ^-?\\d+\\.\\d{2}$."""
    if value is None:
        return False
    return bool(_USD_RE.match(value))


# ── CreditsState dataclass ───────────────────────────────────────────────────


@dataclass
class CreditsState:
    """Full credits state parsed from x-nous-credits-* response headers."""

    version: int = 0
    remaining_micros: int = 0
    remaining_usd: str = ""
    subscription_micros: int = 0  # SIGNED — may be negative (debt). ONLY field allowed negative.
    subscription_usd: str = ""
    subscription_limit_micros: Optional[int] = None  # PAIRED + OPTIONAL (only when subscription_cap)
    subscription_limit_usd: Optional[str] = None
    rollover_micros: int = 0
    purchased_micros: int = 0
    purchased_usd: str = ""
    tool_pool_micros: int = 0
    tool_pool_gated_off: bool = False
    denominator_kind: str = "none"  # "subscription_cap" | "none"
    paid_access: bool = True  # depletion keys off THIS == False, NEVER remaining==0
    disabled_reason: Optional[str] = None  # header omitted entirely when null
    as_of_ms: int = 0
    captured_at: float = 0.0  # time.time() when this was captured
    from_header: bool = False  # True only when populated by parse_credits_headers()

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        if not self.has_data:
            return float("inf")
        return time.time() - self.captured_at

    @property
    def depleted(self) -> bool:
        """True when the account has lost paid access.

        Keyed off ``paid_access == False`` ONLY — never ``remaining_micros == 0``,
        which would give a false positive whenever the balance is zero but access
        is still live (e.g. subscription renewal pending).
        """
        return not self.paid_access

    @property
    def used_fraction(self) -> Optional[float]:
        """Fraction of the subscription cap consumed, in [0.0, 1.0].

        Computable only when ``subscription_limit_micros`` is a truthy (non-zero,
        non-None) int.  Guarded on the LIMIT FIELD, not ``denominator_kind`` —
        the limit field is the real denominator; ``denominator_kind`` is metadata.
        Returns None when there is no computable denominator (no limit, or limit==0).
        """
        if not isinstance(self.subscription_limit_micros, int):
            return None
        if self.subscription_limit_micros <= 0:
            return None
        used = self.subscription_limit_micros - self.subscription_micros
        return max(0.0, min(1.0, used / self.subscription_limit_micros))


# ── Credits policy constants ─────────────────────────────────────────────────
# Switching credits notices from sticky→TTL later would also require wiring a
# paired *_TTL_MS companion for each notice kind — the field exists on AgentNotice
# but is not yet plumbed through the policy loop.

CREDITS_NOTICE_KIND = "sticky"      # v1: credits notices are sticky
CREDITS_RESTORED_TTL_MS = 8000     # the only TTL notice in v1 (depletion-recovery confirmation)

# Usage-gauge bands (ascending). Each is (threshold_fraction, level, label_pct).
# The notice shows the HIGHEST band the current used_fraction has reached — a single
# escalating status-bar line (50 → 75 → 90), not three stacked notices. Crossing the
# next band up replaces the line; recovering below a band steps it back down. Edit
# this list to retune the bands; the policy derives everything from it.
CREDITS_USAGE_BANDS: tuple[tuple[float, str, int], ...] = (
    (0.50, "info", 50),
    (0.75, "warn", 75),
    (0.90, "warn", 90),
)
CREDITS_USAGE_KEY = "credits.usage"  # single key for the escalating usage notice


# ── AgentNotice (out-of-band notice payload; driver-agnostic) ────────────────


@dataclass
class AgentNotice:
    """A structured, driver-agnostic out-of-band notice.

    The agent fires these via ``AIAgent.notice_callback`` (and clears them via
    ``notice_clear_callback``); each driver renders it its own way — the TUI as a
    status-bar override, the CLI as a console line, etc. v1 credits notices are all
    ``kind="sticky"``; ``kind``/``ttl_ms`` are kept fully expressive so a future
    config/slash-command can switch them to TTL without touching the policy (a
    single default seam — see L4).
    """

    text: str
    level: str = "info"            # info | warn | error | success
    kind: str = "sticky"           # sticky | ttl
    ttl_ms: Optional[int] = None   # honored only when kind == "ttl"
    key: Optional[str] = None      # dedupe / fired-once-latch / clear key
    id: Optional[str] = None


# ── is_free_tier_model (local-data-only free-model check) ────────────────────


def is_free_tier_model(model: str, base_url: str = "") -> bool:
    """Return True when *model* is a Nous free-tier model, using ONLY local data.

    Two signals, both zero-network:

    1. The ``:free`` suffix — the canonical Nous free SKU marker (e.g.
       ``nvidia/nemotron-3-ultra:free``). Free by construction on the API side
       (spend is forced to 0 for ``:free`` ids).
    2. A peek into the in-process pricing cache in ``hermes_cli.models``
       (populated when the model picker fetched ``/v1/models`` pricing for
       *base_url*). PEEK ONLY — a cache miss never triggers a fetch. This is
       CLI/TUI-session best-effort: gateway sessions never run the picker's
       pricing fetch, so suppression there rests entirely on the ``:free``
       suffix (which all Nous free SKUs carry).

    Fail-open to False (the depleted notice still shows) on any error: wrongly
    showing the warning is recoverable noise; wrongly hiding it on a paid model
    would mask a real billing block.
    """
    if not model:
        return False
    if model.endswith(":free"):
        return True
    if not base_url:
        return False
    try:
        from hermes_cli.models import _is_model_free, _pricing_cache

        # Mirror get_pricing_for_provider's key normalization: the agent's
        # Nous base_url is /v1-suffixed (https://inference-api.nousresearch.com/v1)
        # but the picker keys _pricing_cache on the pre-/v1 root.
        key = base_url.rstrip("/")
        if key.endswith("/v1"):
            key = key[:-3].rstrip("/")
        pricing = _pricing_cache.get(key)
        if not pricing:
            return False
        return _is_model_free(model, pricing)
    except Exception:
        return False


# ── evaluate_credits_notices (pure reconciliation function) ──────────────────


def evaluate_credits_notices(
    state: CreditsState,
    latch: dict,
    *,
    model_is_free: bool = False,
) -> tuple[list[AgentNotice], list[str]]:
    """Reconcile credits notices against the latch. Mutates ``latch`` IN PLACE.

    latch = {"active": set[str], "seen_below_90": bool, "usage_band": Optional[int]}.

    ``model_is_free``: True when the session's active model is a Nous free-tier
    model (see :func:`is_free_tier_model`). Suppresses the ``credits.depleted``
    notice — a depleted account on a free model can keep inferencing, so the
    error banner is noise (and confuses free-tier users who never had credits).
    Suppression does NOT emit the "restored" success notice; that fires only on
    a genuine ``paid_access`` flip back to True.

    Returns ``(to_show: list[AgentNotice], to_clear: list[str])``.
    Caller emits to_clear FIRST, then to_show.

    Pure function — no I/O, no agent/run_agent imports.
    """
    to_show: list[AgentNotice] = []
    to_clear: list[str] = []

    uf = state.used_fraction

    # Crossing latch: once we've observed uf below the LOWEST band, escalating
    # usage notices may fire. This prevents a brand-new session that opens
    # mid-range from firing spuriously on the first observation (the cold-start
    # seed primes this explicitly when it WANTS an open-high warning).
    _lowest_band = CREDITS_USAGE_BANDS[0][0]
    if uf is not None and uf < _lowest_band:
        latch["seen_below_90"] = True  # gate opened: usage-band notices may now fire

    active = latch["active"]

    # ── Conditions ───────────────────────────────────────────────────────────
    # Highest band whose threshold the current usage has reached (None below all).
    current_band: Optional[tuple[float, str, int]] = None
    if uf is not None:
        for band in CREDITS_USAGE_BANDS:  # ascending → last match wins = highest
            if uf >= band[0]:
                current_band = band
    # Top-up suppression: when the account holds purchased (top-up) credits,
    # the subscription-cap gauge is the wrong denominator — warning "90% used"
    # at a user sitting on $50 of top-up is noise (and it previously stuck
    # PERMANENTLY alongside grant_spent at >=100%). Suppress the usage band
    # entirely; the cap-reached case is covered by the grant_spent info notice
    # below, which already names the remaining top-up balance. A top-up landing
    # mid-session flips current_band → None and the clear path below removes
    # any showing band line.
    if state.purchased_micros > 0:
        current_band = None
    grant_cond = (
        state.denominator_kind == "subscription_cap"
        and uf is not None
        and uf >= 1.0
        and state.purchased_micros > 0
    )
    depleted_cond = not state.paid_access

    # ── usage gauge (escalating single notice: 50 → 75 → 90) ──────────────────
    # Show only the highest crossed band; replace the line when the band changes
    # (climb or step-down on recovery); clear entirely when usage drops below the
    # lowest band or the denominator disappears (uf is None).
    shown_band = latch.get("usage_band")  # the pct label currently displayed, or None
    target_band = current_band[2] if (current_band and latch["seen_below_90"]) else None
    if target_band != shown_band:
        if CREDITS_USAGE_KEY in active:
            to_clear.append(CREDITS_USAGE_KEY)
            active.discard(CREDITS_USAGE_KEY)
        if target_band is not None:
            # Belt-and-suspenders: a producer could set subscription_limit_micros
            # without subscription_limit_usd. Render "$? cap" rather than "$None cap".
            _cap_usd = state.subscription_limit_usd or "?"
            _level = current_band[1]  # type: ignore[index]  (current_band set when target_band set)
            to_show.append(
                AgentNotice(
                    text=f"{'⚠' if _level == 'warn' else '•'} Credits {target_band}% used · ${_cap_usd} cap",
                    level=_level,
                    kind=CREDITS_NOTICE_KIND,
                    key=CREDITS_USAGE_KEY,
                    id=CREDITS_USAGE_KEY,
                )
            )
            active.add(CREDITS_USAGE_KEY)
        latch["usage_band"] = target_band

    # ── grant_spent ──────────────────────────────────────────────────────────
    if grant_cond and "credits.grant_spent" not in active:
        to_show.append(
            AgentNotice(
                text=f"• Grant spent · ${state.purchased_usd} top-up left",
                level="info",
                kind=CREDITS_NOTICE_KIND,
                key="credits.grant_spent",
                id="credits.grant_spent",
            )
        )
        active.add("credits.grant_spent")
    elif "credits.grant_spent" in active and not grant_cond:
        to_clear.append("credits.grant_spent")
        active.discard("credits.grant_spent")

    # ── depleted ─────────────────────────────────────────────────────────────
    # Suppressed while the active model is free: inference still works there,
    # so the error banner would just alarm users (free-tier users especially,
    # who never had paid credits to "lose").
    show_depleted = depleted_cond and not model_is_free
    if show_depleted and "credits.depleted" not in active:
        to_show.append(
            AgentNotice(
                text="✕ Credit access paused · run /usage for balance",
                level="error",
                kind=CREDITS_NOTICE_KIND,
                key="credits.depleted",
                id="credits.depleted",
            )
        )
        active.add("credits.depleted")
    elif "credits.depleted" in active and not show_depleted:
        to_clear.append("credits.depleted")
        active.discard("credits.depleted")
        if not depleted_cond:
            # Genuine recovery (paid_access flipped back True): also emit the
            # success notice. A clear caused by switching to a free model while
            # still depleted must NOT claim access was restored.
            to_show.append(
                AgentNotice(
                    text="✓ Credit access restored",
                    level="success",
                    kind="ttl",
                    ttl_ms=CREDITS_RESTORED_TTL_MS,
                    key="credits.restored",
                    id="credits.restored",
                )
            )

    return (to_show, to_clear)


# ── parse_credits_headers ────────────────────────────────────────────────────


def parse_credits_headers(
    headers: Mapping[str, str],
    provider: str = "",
) -> Optional[CreditsState]:
    """Parse x-nous-credits-* (and x-nous-tool-pool-*) headers into a CreditsState.

    Returns None (miss) on ANY of:
    - No ``x-nous-credits-version`` header present.
    - Version != 1 (> 1 also emits a one-time logger.warning).
    - Any ``*_micros`` field is non-integer, or negative for a non-subscription field.
    - Any ``*_usd`` field doesn't match ``^-?\\d+\\.\\d{2}$``.
    - ``denominator_kind`` is not in {"subscription_cap", "none"}.
    - ``paid_access`` / ``tool_pool_gated_off`` is not exactly "true"/"false".
    - ``as_of_ms`` is not a valid integer.
    - Any unexpected exception.

    Fail-open on the subscription_limit pair: a half-pair (only -micros or only
    -usd present) is treated as both-absent; the overall parse STILL SUCCEEDS
    but with subscription_limit_micros/usd both None.
    """
    global _version_warning_emitted

    try:
        # Cheap probe before the full lowercase copy: bail when the version
        # sentinel header is absent (the common case for non-Nous providers, on
        # every API call) — skips allocating a dict over the whole response's
        # headers on the hot path, while preserving case-insensitivity. Behaviour
        # is identical: a missing version header was already a None return below.
        if not any(k.lower() == "x-nous-credits-version" for k in headers):
            return None
        # Normalize to lowercase so lookups work regardless of how the server
        # capitalises headers (HTTP header names are case-insensitive per RFC 7230).
        lowered = {k.lower(): v for k, v in headers.items()}

        # ── Version check ────────────────────────────────────────────────────
        # Must be present and exactly 1; > 1 warns once then returns None.
        version_raw = lowered.get("x-nous-credits-version")
        if version_raw is None:
            return None
        version_val = _safe_int(version_raw)
        if version_val is _SENTINEL:
            return None
        if version_val != 1:
            if version_val > 1 and not _version_warning_emitted:
                _version_warning_emitted = True
                logger.warning(
                    "credits header version %d unsupported, ignoring — update Hermes",
                    version_val,
                )
            return None

        # ── Helper: parse a required non-negative int field (fail → None) ───
        def _req_nonneg(key: str) -> Any:
            raw = lowered.get(key)
            val = _safe_int(raw)
            if val is _SENTINEL:
                return _SENTINEL
            if val < 0:
                return _SENTINEL
            return val

        # ── Helper: parse a required int field that may be negative (subscription only) ─
        def _req_int(key: str) -> Any:
            raw = lowered.get(key)
            val = _safe_int(raw)
            if val is _SENTINEL:
                return _SENTINEL
            return val

        # ── Parse micros fields ──────────────────────────────────────────────
        remaining_micros = _req_nonneg("x-nous-credits-remaining-micros")
        if remaining_micros is _SENTINEL:
            return None

        subscription_micros = _req_int("x-nous-credits-subscription-micros")
        if subscription_micros is _SENTINEL:
            return None

        rollover_micros = _req_nonneg("x-nous-credits-rollover-micros")
        if rollover_micros is _SENTINEL:
            return None

        purchased_micros = _req_nonneg("x-nous-credits-purchased-micros")
        if purchased_micros is _SENTINEL:
            return None

        # tool_pool_micros is OPTIONAL: absent → 0 (default); present-but-invalid → None (miss).
        _tp_raw = lowered.get("x-nous-tool-pool-micros")
        if _tp_raw is None:
            tool_pool_micros = 0
        else:
            _tp_val = _safe_int(_tp_raw)
            if _tp_val is _SENTINEL or _tp_val < 0:
                return None
            tool_pool_micros = _tp_val

        as_of_ms = _req_nonneg("x-nous-credits-as-of-ms")
        if as_of_ms is _SENTINEL:
            return None

        # ── Validate USD strings ─────────────────────────────────────────────
        remaining_usd = lowered.get("x-nous-credits-remaining-usd", "")
        if not _validate_usd(remaining_usd):
            return None

        subscription_usd = lowered.get("x-nous-credits-subscription-usd", "")
        if not _validate_usd(subscription_usd):
            return None

        purchased_usd = lowered.get("x-nous-credits-purchased-usd", "")
        if not _validate_usd(purchased_usd):
            return None

        # ── subscription_limit_* PAIRED + OPTIONAL ───────────────────────────
        # Both present → validate both; half-pair → treat BOTH as absent (parse
        # still succeeds, just with no limit pair).
        sub_limit_micros_raw = lowered.get("x-nous-credits-subscription-limit-micros")
        sub_limit_usd_raw = lowered.get("x-nous-credits-subscription-limit-usd")

        subscription_limit_micros: Optional[int] = None
        subscription_limit_usd: Optional[str] = None

        if sub_limit_micros_raw is not None and sub_limit_usd_raw is not None:
            # Both present — validate both; any invalid → return None (bad data)
            lm = _safe_int(sub_limit_micros_raw)
            if lm is _SENTINEL:
                return None
            if lm < 0:
                return None
            if not _validate_usd(sub_limit_usd_raw):
                return None
            subscription_limit_micros = lm
            subscription_limit_usd = sub_limit_usd_raw
        # else: half-pair or both absent → leave both None, parse continues

        # ── denominator_kind ─────────────────────────────────────────────────
        denominator_kind = lowered.get("x-nous-credits-denominator-kind", "none")
        if denominator_kind not in _VALID_DENOMINATOR_KINDS:
            return None

        # ── paid_access / tool_pool_gated_off ────────────────────────────────
        # Both must be exactly "true" or "false" (case-insensitive).  An absent
        # paid_access header → fail-open (assume access); absent tool_pool_gated_off
        # → default False.  Present but invalid → return None.
        if "x-nous-credits-paid-access" in lowered:
            pa_raw = lowered["x-nous-credits-paid-access"].strip().lower()
            if pa_raw not in ("true", "false"):
                return None
            paid_access = pa_raw == "true"
        else:
            paid_access = True  # fail-open

        if "x-nous-tool-pool-gated-off" in lowered:
            tpgo_raw = lowered["x-nous-tool-pool-gated-off"].strip().lower()
            if tpgo_raw not in ("true", "false"):
                return None
            tool_pool_gated_off = tpgo_raw == "true"
        else:
            tool_pool_gated_off = False

        # ── disabled_reason: header omitted when null ────────────────────────
        disabled_reason = lowered.get("x-nous-credits-disabled-reason")  # None if absent

        return CreditsState(
            version=version_val,
            remaining_micros=remaining_micros,
            remaining_usd=remaining_usd,
            subscription_micros=subscription_micros,
            subscription_usd=subscription_usd,
            subscription_limit_micros=subscription_limit_micros,
            subscription_limit_usd=subscription_limit_usd,
            rollover_micros=rollover_micros,
            purchased_micros=purchased_micros,
            purchased_usd=purchased_usd,
            tool_pool_micros=tool_pool_micros,
            tool_pool_gated_off=tool_pool_gated_off,
            denominator_kind=denominator_kind,
            paid_access=paid_access,
            disabled_reason=disabled_reason,
            as_of_ms=as_of_ms,
            captured_at=time.time(),
            from_header=True,
        )

    except Exception:
        # Fail-open → miss, but leave a breadcrumb so a parser/import regression
        # (feature silently dead) is distinguishable from a legitimate no-headers
        # response in agent.log, without needing a dev flag.
        logger.debug("credits ▸ parse_credits_headers raised (fail-open miss)", exc_info=True)
        return None


# ── Dev test fixtures (HERMES_DEV_CREDITS_FIXTURE) ───────────────────────────
# Throwaway dev scaffolding: trigger any notice state on demand for testing,
# without real spend or Redis seeding. Set HERMES_DEV_CREDITS_FIXTURE to either a
# state NAME (fixed for the session) or a FILE PATH whose contents are a state
# name (re-read every turn → flip states live: `echo depleted > /tmp/cf`, take a
# turn; `echo healthy > /tmp/cf`, take a turn → recovery).
#
# A fixture drives THREE surfaces uniformly, so the whole credits UX is testable
# offline: (1) the per-turn capture/notice path (_capture_credits), (2) the
# cold-start seed at session open (conversation_loop → depletion/warn90 hydrate
# immediately), and (3) the /usage view (nous_credits_lines renders the fixture).
# `clear` / `none` / unset → real behaviour. Delete with the rest of the
# HERMES_DEV_CREDITS scaffolding.
_DEV_FIXTURES: dict[str, dict] = {
    "healthy": dict(  # used_fraction ~0.1, paid → no notice (recovery target)
        remaining_micros=30_340_000, remaining_usd="30.34",
        subscription_micros=18_000_000, subscription_usd="18.00",
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        purchased_micros=12_340_000, purchased_usd="12.34",
        denominator_kind="subscription_cap", paid_access=True,
    ),
    "sub_50pct": dict(  # used_fraction == 0.5 → credits.usage band 50 (info)
        remaining_micros=10_000_000, remaining_usd="10.00",
        subscription_micros=10_000_000, subscription_usd="10.00",
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=True,
    ),
    "sub_75pct": dict(  # used_fraction == 0.75 → credits.usage band 75 (warn)
        remaining_micros=5_000_000, remaining_usd="5.00",
        subscription_micros=5_000_000, subscription_usd="5.00",
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=True,
    ),
    "sub_90pct": dict(  # used_fraction == 0.9 → credits.usage band 90 (warn)
        remaining_micros=2_000_000, remaining_usd="2.00",
        subscription_micros=2_000_000, subscription_usd="2.00",
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        denominator_kind="subscription_cap", paid_access=True,
    ),
    "grant_exhausted": dict(  # used_fraction == 1.0 + purchased>0 → credits.grant_spent
        remaining_micros=12_340_000, remaining_usd="12.34",
        subscription_micros=0, subscription_usd="0.00",
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        purchased_micros=12_340_000, purchased_usd="12.34",
        denominator_kind="subscription_cap", paid_access=True,
    ),
    "depleted": dict(  # paid_access False → credits.depleted (sticky)
        remaining_micros=0, remaining_usd="0.00",
        subscription_micros=0, subscription_usd="0.00",
        purchased_micros=0, purchased_usd="0.00",
        paid_access=False, disabled_reason="out_of_credits",
    ),
    "debt": dict(  # subscription in debt (negative, the only signed field) → depleted
        remaining_micros=0, remaining_usd="0.00",
        subscription_micros=-5_000_000, subscription_usd="-5.00",
        subscription_limit_micros=20_000_000, subscription_limit_usd="20.00",
        purchased_micros=0, purchased_usd="0.00",
        denominator_kind="subscription_cap", paid_access=False,
        disabled_reason="out_of_credits",
    ),
}


def dev_fixture_credits_state() -> Optional[CreditsState]:
    """Return a fixture CreditsState for HERMES_DEV_CREDITS_FIXTURE, or None.

    The env value is a state name, OR a path to a file whose contents are a state
    name (re-read each call → flip states live without a restart). Unknown name /
    "clear" / "none" / unset → None (normal behaviour). Throwaway test scaffolding.

    Hard prod-leak guard: a fixture applies ONLY when the dev flag HERMES_DEV_CREDITS
    is also on, so a stray HERMES_DEV_CREDITS_FIXTURE (leaked into a shell profile, a
    container env, a launch plist, …) can never surface fabricated balances/notices
    on a real account.
    """
    if not is_truthy_value(os.environ.get("HERMES_DEV_CREDITS")):
        return None
    raw = os.environ.get("HERMES_DEV_CREDITS_FIXTURE", "").strip()
    if not raw:
        return None
    name = raw
    if os.path.sep in raw or "/" in raw:  # looks like a path → read the name from the file
        try:
            with open(raw, "r", encoding="utf-8") as fh:
                name = fh.read().strip()
        except OSError:
            return None
    spec = _DEV_FIXTURES.get(name.lower())
    if not spec:
        return None
    # Stamp the fields the REAL parser always guarantees, so a fixture state is
    # field-identical to a parse_credits_headers() result from equivalent headers
    # (verified by the differential test): version is always 1, and purchased_usd
    # is always a valid usd string (the parser rejects a missing/empty one, so a
    # real zero-top-up account still carries "0.00"). Specs may override these.
    merged = {"version": 1, "purchased_usd": "0.00", **spec}
    return CreditsState(**merged, from_header=True, captured_at=time.time())


def _credits_state_from_account(info) -> Optional[CreditsState]:
    """Map a NousPortalAccountInfo into a header-shaped CreditsState for the seed.

    Float account dollars → micros (plus a DISPLAY *_usd string — allowed, since
    we're formatting account floats, NOT parsing a server-provided *_usd). Returns
    None if the account can't yield a usable state (fail-open)."""
    try:
        _acc = getattr(info, "paid_service_access_info", None)
        _sub = getattr(info, "subscription", None)

        def _to_micros(dollars):
            return int(round(dollars * 1_000_000)) if isinstance(dollars, (int, float)) else 0

        def _to_usd(dollars):
            # DISPLAY formatting of an account float (not a server *_usd string);
            # "" when absent so render/notice copy falls back gracefully.
            return f"{dollars:.2f}" if isinstance(dollars, (int, float)) else ""

        _monthly = getattr(_sub, "monthly_credits", None)
        _has_cap = isinstance(_monthly, (int, float)) and _monthly > 0
        _paid = getattr(info, "paid_service_access", None)
        return CreditsState(
            remaining_micros=_to_micros(getattr(_acc, "total_usable_credits", None)),
            remaining_usd=_to_usd(getattr(_acc, "total_usable_credits", None)),
            subscription_micros=_to_micros(getattr(_acc, "subscription_credits_remaining", None)),
            subscription_usd=_to_usd(getattr(_acc, "subscription_credits_remaining", None)),
            subscription_limit_micros=_to_micros(_monthly) if _has_cap else None,
            subscription_limit_usd=_to_usd(_monthly) if _has_cap else None,
            purchased_micros=_to_micros(getattr(_acc, "purchased_credits_remaining", None)),
            purchased_usd=_to_usd(getattr(_acc, "purchased_credits_remaining", None)),
            rollover_micros=_to_micros(getattr(_sub, "rollover_credits", None)),
            denominator_kind="subscription_cap" if _has_cap else "none",
            paid_access=_paid if isinstance(_paid, bool) else True,
            from_header=False,
            captured_at=time.time(),
        )
    except Exception:
        logger.debug("credits ▸ seed account→state mapping failed", exc_info=True)
        return None


def _hydrate_seed_state(agent, state) -> None:
    """Install a seed CreditsState on the agent and fire the notice policy once.

    Sets _credits_state, latches session-start remaining, and primes the crossing
    gate (the cold-start snapshot IS the first observation, so a session that opens
    already in a band warns immediately — the live header path keeps true crossing
    semantics), then emits. Safe to call from a worker thread: emit already runs
    off-thread in the TUI build path."""
    agent._credits_state = state
    if getattr(agent, "_credits_session_start_micros", None) is None:
        agent._credits_session_start_micros = state.remaining_micros
    _latch = getattr(agent, "_credits_latch", None)
    if isinstance(_latch, dict) and state.used_fraction is not None:
        _latch["seen_below_90"] = True
    emit = getattr(agent, "_emit_credits_notices", None)
    if callable(emit):
        emit()


def seed_credits_at_session_start(agent) -> bool:
    """Hydrate agent._credits_state from /api/oauth/account (or a dev fixture) and
    fire the notice policy, so depletion / usage-band warnings show at session OPEN.

    Shared by (a) the TUI/desktop agent build (fires at "ready", before any message)
    and (b) the first-turn conversation setup (fallback for plain CLI / when the
    build path didn't seed). Idempotent: a second call is a no-op once a seed or a
    real header has already populated _credits_state.

    Returns True if it seeded this call, False otherwise (not nous / already seeded /
    fail-open error). Never raises — credits must never block session startup.
    """
    try:
        if getattr(agent, "provider", "") != "nous":
            return False
        # Idempotent: don't re-seed if state already exists (seed or live header).
        if getattr(agent, "_credits_state", None) is not None:
            return False
        fixture = None
        try:
            fixture = dev_fixture_credits_state()
        except Exception:
            fixture = None
        if fixture is not None:
            # Synchronous: a fixture is instant (no network), and tests rely on the
            # state + notice landing before this returns.
            _hydrate_seed_state(agent, fixture)
            return True

        # Real portal fetch is FIRE-AND-FORGET: a slow/unreachable portal must never
        # delay session "ready". A daemon thread hydrates + emits when it resolves,
        # re-checking idempotency first (a live inference header may land before it).
        import threading

        def _bg_seed() -> None:
            try:
                from hermes_cli.nous_account import get_nous_portal_account_info
                info = get_nous_portal_account_info(force_fresh=True)
                if getattr(agent, "_credits_state", None) is not None:
                    return  # a live inference header beat us — don't clobber it
                state = _credits_state_from_account(info)
                if state is not None:
                    _hydrate_seed_state(agent, state)
            except Exception:
                logger.debug("credits ▸ session-start seed (background) failed", exc_info=True)

        threading.Thread(target=_bg_seed, name="credits-seed", daemon=True).start()
        return True
    except Exception:
        # Fail-open: any auth/portal hiccup leaves _credits_state as-is, never blocks.
        # Innermost log across all four call sites (TUI build / CLI build / first
        # turn / desktop), so a dead session-open seed is diagnosable in agent.log.
        logger.debug("credits ▸ session-start seed failed (fail-open)", exc_info=True)
        return False
