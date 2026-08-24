"""Nous credits: parse ``x-nous-credits-*`` / ``x-nous-tool-pool-*`` response
headers into a validated CreditsState (depletion = paid_access, subscription-cap
used_fraction, warn-once schema-version gating) and drive the notice policy.
Header contract: see ``_HEADER_FIELDS``. Money is micros ints only; ``*_usd``
strings are preserved verbatim (never re-parsed to float)."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional

from utils import is_truthy_value

logger = logging.getLogger(__name__)

_version_warning_emitted: bool = False  # warn-once latch (per process)
_VALID_DENOMINATOR_KINDS = frozenset({"subscription_cap", "none"})
_USD_RE = re.compile(r"^-?\d+\.\d{2}$")  # optional minus, digits, exactly 2 decimals
_SENTINEL = object()  # "parse failed"


def _safe_int(value: Any) -> Any:
    """Exact int (money-safe) or ``_SENTINEL``. ``int()`` directly, NOT ``int(float())``
    (precision loss above 2**53 corrupts money); float-shaped strings fail."""
    try:
        return _SENTINEL if value is None else int(str(value))
    except (TypeError, ValueError):
        return _SENTINEL


@dataclass
class CreditsState:
    """Credits state parsed from x-nous-credits-* response headers."""

    version: int = 0
    remaining_micros: int = 0
    remaining_usd: str = ""
    subscription_micros: int = 0  # SIGNED — the ONLY field allowed negative (debt)
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
    captured_at: float = 0.0  # time.time() when captured
    from_header: bool = False  # True only when populated by parse_credits_headers()

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at if self.has_data else float("inf")

    @property
    def depleted(self) -> bool:
        """``paid_access == False`` ONLY — ``remaining_micros == 0`` is a false positive
        when the balance is zero but access is live (renewal pending)."""
        return not self.paid_access

    @property
    def used_fraction(self) -> Optional[float]:
        """Fraction of the subscription cap consumed in [0.0, 1.0]; None without a computable
        denominator. Guarded on the LIMIT FIELD (the real denominator), not ``denominator_kind``."""
        lim = self.subscription_limit_micros
        if not isinstance(lim, int) or lim <= 0:
            return None
        return max(0.0, min(1.0, (lim - self.subscription_micros) / lim))


# ── Credits policy constants. Switching notices sticky→TTL later also needs a
# paired *_TTL_MS per notice kind (AgentNotice has the field; not plumbed yet).
CREDITS_NOTICE_KIND = "sticky"      # v1: credits notices are sticky
CREDITS_RESTORED_TTL_MS = 8000     # the only TTL notice in v1 (depletion-recovery confirmation)
# Usage-gauge bands (ascending): (threshold_fraction, level, label_pct). One
# escalating line shows the HIGHEST band reached; climbing replaces it, recovery steps down.
CREDITS_USAGE_BANDS: tuple[tuple[float, str, int], ...] = ((0.50, "info", 50), (0.75, "warn", 75), (0.90, "warn", 90))
CREDITS_USAGE_KEY = "credits.usage"
# Min subscription balance counting as "grant not yet spent" for the grant_spent
# gate. 1¢: portal-seeded states (float dollars → micros) can carry sub-cent residue
# where headers report 0 — without the floor a seed opens the gate and the first
# header re-creates the at-open nag.
GRANT_UNSPENT_MIN_MICROS = 10_000


def new_credits_latch() -> dict:
    """Fresh notice latch for :func:`evaluate_credits_notices`; every producer builds it here so a new gate key lands everywhere."""
    return {"active": set(), "seen_below_90": False, "usage_band": None, "seen_grant_unspent": False}


@dataclass
class AgentNotice:
    """Driver-agnostic out-of-band notice (``AIAgent.notice_callback`` / ``notice_clear_callback``); each driver
    renders its own way. ``kind``/``ttl_ms`` stay expressive so a future config can switch v1's sticky notices to TTL."""

    text: str
    level: str = "info"            # info | warn | error | success
    kind: str = "sticky"           # sticky | ttl
    ttl_ms: Optional[int] = None   # honored only when kind == "ttl"
    key: Optional[str] = None      # dedupe / fired-once-latch / clear key
    id: Optional[str] = None


def _sticky_notice(text: str, level: str, key: str) -> AgentNotice:
    return AgentNotice(text=text, level=level, kind=CREDITS_NOTICE_KIND, key=key, id=key)


def is_free_tier_model(model: str, base_url: str = "") -> bool:
    """True when *model* is a Nous free-tier model, using ONLY local data: (1) ``:free`` suffix — canonical
    Nous free SKU marker; (2) ``stealth/`` prefix — stealth-preview SKUs are free without the suffix
    (naming-convention trust: a PAID ``stealth/`` model would wrongly suppress the banner); (3) a PEEK into
    ``hermes_cli.models``' pricing cache (filled by the model picker; a miss never fetches). Fail-open to
    False (depleted notice still shows): a wrong warning is recoverable noise; hiding it masks a real block."""
    if not model:
        return False
    if model.endswith(":free") or model.startswith("stealth/"):
        return True
    if not base_url:
        return False
    try:
        from hermes_cli.models import _is_model_free
        from hermes_cli.models_pricing import peek_cached_pricing

        pricing = peek_cached_pricing(base_url)  # owns the /v1-suffix and auth-state key details
        return bool(pricing) and _is_model_free(model, pricing)
    except Exception:
        return False


def evaluate_credits_notices(state: CreditsState, latch: dict, *, model_is_free: bool = False) -> tuple[list[AgentNotice], list[str]]:
    """Reconcile credits notices against the latch (see :func:`new_credits_latch`); mutates ``latch`` IN
    PLACE. Pure — no I/O, no agent/run_agent imports. ``model_is_free`` suppresses ``credits.depleted`` (a
    depleted account on a free model keeps inferencing) WITHOUT emitting "restored" — that fires only on a
    genuine ``paid_access`` flip back to True. Returns ``(to_show, to_clear)``; caller emits to_clear FIRST."""
    to_show: list[AgentNotice] = []
    to_clear: list[str] = []
    uf, active = state.used_fraction, latch["active"]
    # Crossing latch: band notices fire only once uf was observed below the LOWEST
    # band, so a session opening mid-range doesn't fire on its first observation
    # (the cold-start seed primes this when it WANTS an open-high warning).
    if uf is not None and uf < CREDITS_USAGE_BANDS[0][0]:
        latch["seen_below_90"] = True
    # Grant-spent gate: fires only after this session OBSERVED the grant unspent
    # (≥1¢). Opening at grant-spent is a steady STATE (/usage carries it), not an
    # event. Unlike seen_below_90, seeds must NOT prime this gate.
    if uf is not None and uf < 1.0 and state.subscription_micros >= GRANT_UNSPENT_MIN_MICROS:
        latch["seen_grant_unspent"] = True

    # Highest band reached (ascending → last match wins); None below all. Top-up
    # suppression: with purchased credits the cap gauge is the wrong denominator
    # ("90% used" on $50 of top-up is noise; it used to stick PERMANENTLY beside
    # grant_spent at >=100%) — grant_spent covers the cap-reached case, and a
    # mid-session top-up flips current_band → None so the clear path removes the line.
    current_band: Optional[tuple[float, str, int]] = None
    if uf is not None and state.purchased_micros <= 0:
        current_band = next((b for b in reversed(CREDITS_USAGE_BANDS) if uf >= b[0]), None)

    # ── usage gauge: highest crossed band only; replace on band change (climb or
    # step-down); clear below the lowest band or when the denominator vanishes.
    target_band = current_band[2] if (current_band and latch["seen_below_90"]) else None
    if target_band != latch.get("usage_band"):
        if CREDITS_USAGE_KEY in active:
            to_clear.append(CREDITS_USAGE_KEY)
            active.discard(CREDITS_USAGE_KEY)
        if target_band is not None:
            # Absolute dollars used (a bare "N%" is only meaningful against a Nous cap): cap − remaining,
            # clamped [0, cap]; "$?" if a producer set the limit without its *_usd. Re-emits on band change only.
            level = current_band[1]  # type: ignore[index]  (current_band set when target_band set)
            lim = state.subscription_limit_micros or 0
            used_usd = f"{max(0, min(lim, lim - state.subscription_micros)) / 1_000_000:.2f}" if lim else "?"
            text = f"{'⚠' if level == 'warn' else '•'} You've used ${used_usd} of your ${state.subscription_limit_usd or '?'} cap"
            to_show.append(_sticky_notice(text, level, CREDITS_USAGE_KEY))
            active.add(CREDITS_USAGE_KEY)
        latch["usage_band"] = target_band

    # ── grant_spent: the gate guards only the SHOW and is CONSUMED by it — one
    # announcement per crossing. A header flicker (uf → None → 1.0) clears the
    # line but cannot re-announce; only a renewal re-opening the gate (fresh ≥1¢
    # observation) arms the next. .get(): default closed for hand-built latches.
    grant_cond = (
        state.denominator_kind == "subscription_cap" and uf is not None and uf >= 1.0 and state.purchased_micros > 0
    )
    if grant_cond and "credits.grant_spent" not in active and latch.get("seen_grant_unspent", False):
        to_show.append(_sticky_notice(f"• Grant spent · ${state.purchased_usd} top-up left", "info", "credits.grant_spent"))
        active.add("credits.grant_spent")
        latch["seen_grant_unspent"] = False
    elif "credits.grant_spent" in active and not grant_cond:
        to_clear.append("credits.grant_spent")
        active.discard("credits.grant_spent")

    # ── depleted: suppressed while the model is free (inference still works).
    depleted_cond = not state.paid_access
    show_depleted = depleted_cond and not model_is_free
    if show_depleted and "credits.depleted" not in active:
        to_show.append(_sticky_notice("✕ Credit access paused · run /topup to top up", "error", "credits.depleted"))
        active.add("credits.depleted")
    elif "credits.depleted" in active and not show_depleted:
        to_clear.append("credits.depleted")
        active.discard("credits.depleted")
        if not depleted_cond:  # genuine recovery only — a free-model switch while depleted is NOT "restored"
            to_show.append(AgentNotice(
                text="✓ Credit access restored", level="success", kind="ttl",
                ttl_ms=CREDITS_RESTORED_TTL_MS, key="credits.restored", id="credits.restored",
            ))
    return (to_show, to_clear)


# Header contract: (field, kind[, default-when-absent]); a field is REQUIRED unless it has a default.
# Header name = ``x-nous-credits-<field>`` (``x-nous-<field>`` for tool_pool_*), underscores → dashes.
# micros: int >= 0 ("signed": may be negative); usd: the server's formatted string ^-?\d+\.\d{2}$
# (never re-parsed); bool: "true"/"false" STRING. Handled inline: subscription-limit-* (PAIRED/optional),
# denominator-kind ("subscription_cap" | "none"), disabled-reason (omitted when null).
_HEADER_FIELDS: tuple[tuple, ...] = (
    ("remaining_micros", "micros"), ("subscription_micros", "signed"), ("rollover_micros", "micros"),
    ("purchased_micros", "micros"), ("as_of_ms", "micros"), ("tool_pool_micros", "micros", 0),
    ("remaining_usd", "usd"), ("subscription_usd", "usd"), ("purchased_usd", "usd"),
    ("paid_access", "bool", True),  # absent → fail-open (assume access)
    ("tool_pool_gated_off", "bool", False),
)


def _header_name(field: str) -> str:
    return "x-nous-" + ("" if field.startswith("tool_pool_") else "credits-") + field.replace("_", "-")


def _parse_field(kind: str, raw: Optional[str], default: Any = _SENTINEL) -> Any:
    """One header value → field value; ``default`` when absent, ``_SENTINEL`` on a contract violation."""
    if raw is None:
        return default
    if kind in ("micros", "signed"):
        val = _safe_int(raw)
        return _SENTINEL if val is _SENTINEL or (kind == "micros" and val < 0) else val
    if kind == "usd":
        return raw if _USD_RE.match(raw) else _SENTINEL
    flag = raw.strip().lower()
    return _SENTINEL if flag not in ("true", "false") else flag == "true"


def parse_credits_headers(headers: Mapping[str, str], provider: str = "") -> Optional[CreditsState]:
    """Parse x-nous-credits-* (and x-nous-tool-pool-*) headers into a CreditsState.
    None (miss) on ANY of: no version header; version != 1 (> 1 also warns once);
    a required field violating ``_HEADER_FIELDS``; unknown ``denominator_kind``;
    any unexpected exception. Fail-open on the subscription_limit pair: a
    half-pair (only -micros or only -usd) parses as both-absent (both None)."""
    global _version_warning_emitted
    try:
        # Cheap probe before the lowercase copy (header names are case-insensitive): bail when the
        # version header is absent — the hot path for non-Nous providers.
        if not any(k.lower() == "x-nous-credits-version" for k in headers):
            return None
        lowered = {k.lower(): v for k, v in headers.items()}
        version_val = _safe_int(lowered.get("x-nous-credits-version"))
        if version_val is _SENTINEL:
            return None
        if version_val != 1:
            if version_val > 1 and not _version_warning_emitted:
                _version_warning_emitted = True
                logger.warning("credits header version %d unsupported, ignoring — update Hermes", version_val)
            return None
        fields: dict[str, Any] = {
            name: _parse_field(kind, lowered.get(_header_name(name)), *default) for name, kind, *default in _HEADER_FIELDS
        }
        lim_micros_raw = lowered.get("x-nous-credits-subscription-limit-micros")
        lim_usd_raw = lowered.get("x-nous-credits-subscription-limit-usd")
        if lim_micros_raw is not None and lim_usd_raw is not None:
            fields["subscription_limit_micros"] = _parse_field("micros", lim_micros_raw)
            fields["subscription_limit_usd"] = _parse_field("usd", lim_usd_raw)
        denominator_kind = lowered.get("x-nous-credits-denominator-kind", "none")
        if _SENTINEL in fields.values() or denominator_kind not in _VALID_DENOMINATOR_KINDS:
            return None
        disabled_reason = lowered.get("x-nous-credits-disabled-reason")  # None if absent (omitted when null)
        return CreditsState(version=version_val, denominator_kind=denominator_kind, disabled_reason=disabled_reason,
                            captured_at=time.time(), from_header=True, **fields)
    except Exception:  # fail-open → miss; the breadcrumb distinguishes a parser regression from a no-headers response
        logger.debug("credits ▸ parse_credits_headers raised (fail-open miss)", exc_info=True)
        return None


# ── Dev fixtures (HERMES_DEV_CREDITS_FIXTURE): throwaway scaffolding to trigger any notice state
# without real spend. Value is a state NAME or a FILE PATH whose contents are a name (re-read every
# turn → `echo depleted > /tmp/cf` flips live). Drives per-turn notices, the cold-start seed, and /usage.
def _fixture(remaining: str, subscription: str, limit: Optional[str] = None, purchased: Optional[str] = None,
             *, paid: bool = True, reason: Optional[str] = None) -> dict:
    """Fixture spec from *_usd strings; micros derived exactly (Decimal)."""
    d: dict = {}
    for field, usd in (("remaining", remaining), ("subscription", subscription), ("subscription_limit", limit), ("purchased", purchased)):
        if usd is not None:
            d[f"{field}_micros"], d[f"{field}_usd"] = int(Decimal(usd) * 1_000_000), usd
    if limit is not None:
        d["denominator_kind"] = "subscription_cap"
    d["paid_access"] = paid
    return d if reason is None else {**d, "disabled_reason": reason}


_DEV_FIXTURES: dict[str, dict] = {
    "healthy": _fixture("30.34", "18.00", "20.00", "12.34"),  # used_fraction ~0.1, paid → no notice (recovery target)
    "sub_50pct": _fixture("10.00", "10.00", "20.00"),  # used_fraction == 0.5 → credits.usage band 50 (info)
    "sub_75pct": _fixture("5.00", "5.00", "20.00"),  # used_fraction == 0.75 → band 75 (warn)
    "sub_90pct": _fixture("2.00", "2.00", "20.00"),  # used_fraction == 0.9 → band 90 (warn)
    # uf == 1.0 + purchased>0 → SILENT at open (crossing-gated); flip healthy →
    # grant_exhausted via the fixture-file path to see credits.grant_spent
    "grant_exhausted": _fixture("12.34", "0.00", "20.00", "12.34"),
    "depleted": _fixture("0.00", "0.00", None, "0.00", paid=False, reason="out_of_credits"),  # → credits.depleted
    # subscription in debt (negative, the only signed field) → depleted
    "debt": _fixture("0.00", "-5.00", "20.00", "0.00", paid=False, reason="out_of_credits"),
}


def dev_fixture_credits_state() -> Optional[CreditsState]:
    """Fixture CreditsState for HERMES_DEV_CREDITS_FIXTURE, or None (unknown name / unset). Prod-leak guard:
    applies ONLY when HERMES_DEV_CREDITS is also on, so a stray fixture env var never surfaces fabricated balances."""
    name = os.environ.get("HERMES_DEV_CREDITS_FIXTURE", "").strip()
    if not name or not is_truthy_value(os.environ.get("HERMES_DEV_CREDITS")):
        return None
    if os.path.sep in name or "/" in name:  # looks like a path → read the name from the file
        try:
            with open(name, "r", encoding="utf-8") as fh:
                name = fh.read().strip()
        except OSError:
            return None
    if not (spec := _DEV_FIXTURES.get(name.lower())):
        return None
    # Stamp what the REAL parser always guarantees so a fixture is field-identical to a
    # parse_credits_headers() result: version 1 and a valid purchased_usd (zero top-up = "0.00").
    return CreditsState(**{"version": 1, "purchased_usd": "0.00", **spec}, from_header=True, captured_at=time.time())


def _credits_state_from_account(info) -> Optional[CreditsState]:
    """Map a NousPortalAccountInfo into a header-shaped CreditsState for the seed. Float account dollars →
    micros plus a DISPLAY *_usd (formatting account floats is allowed; parsing a server *_usd is not). Fail-open → None."""
    try:
        acc = getattr(info, "paid_service_access_info", None)
        sub = getattr(info, "subscription", None)

        def _money(dollars) -> tuple[int, str]:  # (micros, display usd); (0, "") when absent
            return (int(round(dollars * 1_000_000)), f"{dollars:.2f}") if isinstance(dollars, (int, float)) else (0, "")
        fields: dict[str, Any] = {}
        for prefix, attr in (("remaining", "total_usable_credits"), ("subscription", "subscription_credits_remaining"),
                             ("purchased", "purchased_credits_remaining")):
            fields[f"{prefix}_micros"], fields[f"{prefix}_usd"] = _money(getattr(acc, attr, None))
        monthly = getattr(sub, "monthly_credits", None)
        cap = _money(monthly) if isinstance(monthly, (int, float)) and monthly > 0 else (None, None)
        paid = getattr(info, "paid_service_access", None)
        return CreditsState(
            **fields, subscription_limit_micros=cap[0], subscription_limit_usd=cap[1], from_header=False, captured_at=time.time(),
            rollover_micros=_money(getattr(sub, "rollover_credits", None))[0], paid_access=paid if isinstance(paid, bool) else True,
            denominator_kind="subscription_cap" if cap[0] is not None else "none",
        )
    except Exception:
        logger.debug("credits ▸ seed account→state mapping failed", exc_info=True)
        return None


def _hydrate_seed_state(agent, state) -> None:
    """Install a seed CreditsState on the agent and fire the notice policy once. Primes the crossing gate:
    the cold-start snapshot IS the first observation, so a session opening in a band warns immediately."""
    agent._credits_state = state
    if getattr(agent, "_credits_session_start_micros", None) is None:
        agent._credits_session_start_micros = state.remaining_micros
    latch = getattr(agent, "_credits_latch", None)
    if isinstance(latch, dict) and state.used_fraction is not None:
        latch["seen_below_90"] = True  # ONLY this gate — priming seen_grant_unspent would revive the steady-state nag
    if callable(emit := getattr(agent, "_emit_credits_notices", None)):
        emit()


def seed_credits_at_session_start(agent) -> bool:
    """Hydrate agent._credits_state from the portal account (or a dev fixture) and fire the notice policy so
    warnings show at session OPEN (TUI/desktop "ready" and plain-CLI first-turn setup). Idempotent once a seed
    or real header populated _credits_state. Returns True iff it seeded this call. Never raises."""
    try:
        if getattr(agent, "provider", "") != "nous" or getattr(agent, "_credits_state", None) is not None:
            return False
        try:
            fixture = dev_fixture_credits_state()
        except Exception:
            fixture = None
        if fixture is not None:  # synchronous: instant, and tests rely on state + notice landing before return
            _hydrate_seed_state(agent, fixture)
            return True

        def _bg_seed() -> None:  # FIRE-AND-FORGET: a slow portal must never delay "ready"
            try:
                from hermes_cli.nous_account import get_nous_portal_account_info
                info = get_nous_portal_account_info(force_fresh=True)
                if getattr(agent, "_credits_state", None) is not None:
                    return  # a live inference header beat us — don't clobber it
                if (state := _credits_state_from_account(info)) is not None:
                    _hydrate_seed_state(agent, state)
            except Exception:
                logger.debug("credits ▸ session-start seed (background) failed", exc_info=True)
        threading.Thread(target=_bg_seed, name="credits-seed", daemon=True).start()
        return True
    except Exception:
        logger.debug("credits ▸ session-start seed failed (fail-open)", exc_info=True)  # innermost log: diagnosable dead seed
        return False
