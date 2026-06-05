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
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

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
        if version_val > 1:
            if not _version_warning_emitted:
                _version_warning_emitted = True
                logger.warning(
                    "credits header version %d unsupported, ignoring — update Hermes",
                    version_val,
                )
            return None
        if version_val != 1:
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
        return None
