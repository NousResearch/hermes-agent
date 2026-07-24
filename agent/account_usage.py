from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import httpx

from agent.anthropic_adapter import _is_oauth_token, resolve_anthropic_token
from hermes_cli.auth import AuthError, _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.runtime_provider import resolve_runtime_provider

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AccountUsageWindow:
    label: str
    used_percent: Optional[float] = None
    reset_at: Optional[datetime] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class AccountUsageSnapshot:
    provider: str
    source: str
    fetched_at: datetime
    title: str = "Account limits"
    plan: Optional[str] = None
    windows: tuple[AccountUsageWindow, ...] = ()
    details: tuple[str, ...] = ()
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.windows or self.details) and not self.unavailable_reason


def _title_case_slug(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return cleaned.replace("_", " ").replace("-", " ").title()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return None
    return None


def _format_reset(dt: Optional[datetime]) -> str:
    if not dt:
        return "unknown"
    local_dt = dt.astimezone()
    delta = dt - _utc_now()
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return f"now ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        rel = f"in {days}d {hours}h"
    elif hours > 0:
        rel = f"in {hours}h {minutes}m"
    else:
        rel = f"in {minutes}m"
    return f"{rel} ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"


def render_account_usage_lines(snapshot: Optional[AccountUsageSnapshot], *, markdown: bool = False) -> list[str]:
    if not snapshot:
        return []
    header = f"📈 {'**' if markdown else ''}{snapshot.title}{'**' if markdown else ''}"
    lines = [header]
    if snapshot.plan:
        lines.append(f"Provider: {snapshot.provider} ({snapshot.plan})")
    else:
        lines.append(f"Provider: {snapshot.provider}")
    for window in snapshot.windows:
        if window.used_percent is None:
            base = f"{window.label}: unavailable"
        else:
            remaining = max(0, round(100 - float(window.used_percent)))
            used = max(0, round(float(window.used_percent)))
            base = f"{window.label}: {remaining}% remaining ({used}% used)"
        if window.reset_at:
            base += f" • resets {_format_reset(window.reset_at)}"
        elif window.detail:
            base += f" • {window.detail}"
        lines.append(base)
    for detail in snapshot.details:
        lines.append(detail)
    if snapshot.unavailable_reason:
        lines.append(f"Unavailable: {snapshot.unavailable_reason}")
    return lines


_QUOTA_PROVIDER_PRESENTATION = {
    "anthropic": ("🟠", "Anthropic"),
    "deepseek": ("🔵", "DeepSeek"),
    "google-antigravity": ("🧠", "Google Antigravity"),
    "openai-codex": ("🤖", "OpenAI Codex"),
    "opencode-go": ("⚡", "OpenCode Go"),
    "openrouter": ("🌐", "OpenRouter"),
}


def _quota_status_icon(remaining_percent: int) -> str:
    if remaining_percent >= 50:
        return "🟢"
    if remaining_percent >= 20:
        return "🟡"
    return "🔴"


def _remaining_percent_from_detail(value: str) -> Optional[int]:
    text = value.strip()
    suffix = "% left"
    if not text.lower().endswith(suffix):
        return None
    try:
        numeric = float(text[: -len(suffix)].strip())
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    percent = round(numeric)
    return max(0, min(100, percent))


def _render_quota_detail_lines(details: tuple[str, ...]) -> list[str]:
    raw_details = [str(detail) for detail in details if str(detail).strip()]
    lines: list[str] = []
    index = 0
    while index < len(raw_details):
        raw = raw_details[index]
        text = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        next_raw = raw_details[index + 1] if index + 1 < len(raw_details) else ""
        next_text = next_raw.strip()
        next_indent = len(next_raw) - len(next_raw.lstrip()) if next_raw else 0

        if indent == 0:
            label, separator, value = text.partition(":")
            if separator:
                detail_icon = "💳" if "balance" in label.lower() else "🧾"
                lines.append(f"{detail_icon} **{label}** · {value.strip()}")
            elif next_indent > 0:
                lines.append(f"**{text}**")
            else:
                lines.append(text)
            index += 1
            continue

        remaining = _remaining_percent_from_detail(text)
        if remaining is not None:
            lines.append(f"  {_quota_status_icon(remaining)} **{remaining}% left**")
            index += 1
            continue

        if text.lower().startswith(("reset ", "resets ")):
            lines.append(f"  ↳ {text}")
            index += 1
            continue

        next_remaining = _remaining_percent_from_detail(next_text) if next_indent > indent else None
        if next_remaining is not None:
            lines.append(
                f"• **{text}** · {_quota_status_icon(next_remaining)} "
                f"**{next_remaining}% left**"
            )
            index += 2
            if index < len(raw_details):
                reset_raw = raw_details[index]
                reset_text = reset_raw.strip()
                reset_indent = len(reset_raw) - len(reset_raw.lstrip())
                if reset_indent > indent and reset_text.lower().startswith(("reset ", "resets ")):
                    lines.append(f"  ↳ {reset_text}")
                    index += 1
            continue

        lines.append(f"• {text}")
        index += 1

    return lines


def render_account_quota_card_lines(snapshot: AccountUsageSnapshot) -> list[str]:
    """Render one compact Markdown card for the gateway `/quota` command."""
    provider_id = snapshot.provider.strip().lower()
    icon, provider_name = _QUOTA_PROVIDER_PRESENTATION.get(
        provider_id,
        ("🔹", _title_case_slug(snapshot.provider) or snapshot.provider),
    )
    plan = f" · {snapshot.plan}" if snapshot.plan else ""
    lines = [f"{icon} **{provider_name}**{plan}"]

    for window in snapshot.windows:
        try:
            used_percent = float(window.used_percent) if window.used_percent is not None else None
        except (TypeError, ValueError, OverflowError):
            used_percent = None
        if used_percent is None or not math.isfinite(used_percent):
            lines.append(f"⚪ **Unavailable** · {window.label}")
        else:
            remaining = max(0, min(100, round(100 - used_percent)))
            lines.append(f"{_quota_status_icon(remaining)} **{remaining}% left** · {window.label}")
        if window.reset_at:
            lines.append(f"   ↳ Resets {_format_reset(window.reset_at)}")
        elif window.detail:
            lines.append(f"   ↳ {window.detail}")

    lines.extend(_render_quota_detail_lines(snapshot.details))

    if snapshot.unavailable_reason:
        lines.append(f"⚠️ {snapshot.unavailable_reason}")
    return lines



def _fmt_usd(d: float) -> str:
    return f"${d:,.2f}"


def _is_finite_num(v: Any) -> TypeGuard[float]:
    """True iff v is a real numeric value (int or float, not bool, not NaN/Inf).

    Typed as a ``TypeGuard[float]`` so the type checker narrows ``v`` to a real
    number in the positive branch — callers can then do arithmetic / pass it to
    ``_fmt_usd`` without a None-operand warning.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def build_nous_credits_snapshot(account_info) -> Optional[AccountUsageSnapshot]:
    """Map a NousPortalAccountInfo into an AccountUsageSnapshot for /usage.

    Shows dollar magnitudes (subscription / top-up / total) + renewal date + a
    portal CTA. When the portal supplies a subscription denominator
    (``monthly_credits``), also emits a subscription-usage window so the renderer
    shows a real ``% used`` gauge; when it's absent (older portals) the view
    gracefully degrades to magnitudes-only. Returns None when there's no usable
    account info to show (fail-open: caller just shows nothing).
    """
    try:
        from hermes_cli.nous_account import nous_portal_topup_url

        if account_info is None or not getattr(account_info, "logged_in", False):
            return None

        access = getattr(account_info, "paid_service_access_info", None)
        sub = getattr(account_info, "subscription", None)

        windows: list[AccountUsageWindow] = []
        details: list[str] = []

        # Subscription usage gauge — only when the portal supplies a positive
        # monthly_credits denominator AND a finite remaining balance that does
        # not exceed the cap. Money math is on float dollars (allowed: numeric
        # account fields, NOT a server-provided *_usd string). used = cap -
        # remaining; clamp [0,100] so a debt balance (remaining < 0) reads 100%.
        # Excluded on purpose:
        #   - non-finite values (NaN/Infinity slip past isinstance and json.loads
        #     parses bare NaN/Infinity by default) → would render "$nan"/"$inf"
        #     and a falsely-confident gauge;
        #   - remaining > cap (rollover balance spanning the period) → monthly_credits
        #     is no longer a meaningful denominator, and "$X of $Y left" with X>Y
        #     reads as a contradiction. Both fall back to the magnitudes lines.
        if sub is not None:
            monthly_credits = getattr(sub, "monthly_credits", None)
            sub_remaining = getattr(sub, "credits_remaining", None)
            if (
                _is_finite_num(monthly_credits)
                and monthly_credits > 0
                and _is_finite_num(sub_remaining)
                and sub_remaining <= monthly_credits
            ):
                used = monthly_credits - sub_remaining
                used_pct = max(0.0, min(100.0, used / monthly_credits * 100.0))
                windows.append(
                    AccountUsageWindow(
                        label="Subscription",
                        used_percent=used_pct,
                        detail=f"{_fmt_usd(sub_remaining)} of {_fmt_usd(monthly_credits)} left",
                    )
                )

        if access is not None:
            sub_credits = getattr(access, "subscription_credits_remaining", None)
            if _is_finite_num(sub_credits):
                details.append(f"Subscription credits: {_fmt_usd(sub_credits)}")
            purchased = getattr(access, "purchased_credits_remaining", None)
            if _is_finite_num(purchased):
                details.append(f"Top-up credits: {_fmt_usd(purchased)}")
            total_usable = getattr(access, "total_usable_credits", None)
            if _is_finite_num(total_usable):
                details.append(f"Total usable: {_fmt_usd(total_usable)}")

        if sub is not None:
            rollover = getattr(sub, "rollover_credits", None)
            if _is_finite_num(rollover) and rollover > 0:
                details.append(f"Rollover: {_fmt_usd(rollover)}")
            period_end = getattr(sub, "current_period_end", None)
            if period_end:
                details.append(f"Renews: {period_end}")

        paid = getattr(account_info, "paid_service_access", None)
        if paid is False:
            details.append("Status: access depleted — top up to restore")

        if not windows and not details:
            return None

        details.append(f"Top up: {nous_portal_topup_url(account_info)}")
        details.append("(or run /topup)")

        plan = getattr(sub, "plan", None) if sub is not None else None
        return AccountUsageSnapshot(
            provider="nous",
            source="portal-account",
            fetched_at=_utc_now(),
            title="Nous credits",
            plan=plan,
            windows=tuple(windows),
            details=tuple(details),
        )
    except (AttributeError, TypeError):
        return None


def nous_credits_lines(*, markdown: bool = False, timeout: float = 10.0) -> list[str]:
    """Return rendered Nous-credits /usage lines, or [] when there's nothing to show.

    Account-independent of any live agent: gated on "a Nous account is logged in"
    (a cheap local auth-state check), then a wall-clock-bounded portal fetch. Shared
    by the CLI ``_show_usage`` and the TUI ``session.usage`` RPC so both surfaces show
    the same block regardless of session API-call count or resume state. Fail-open:
    any auth/portal hiccup or timeout returns [] (the caller shows nothing).

    Dev override: when HERMES_DEV_CREDITS_FIXTURE selects a fixture state, /usage
    renders from that fixture instead of the real portal (so the block + gauge are
    testable without a live account). Throwaway scaffolding.
    """
    # Dev fixture short-circuit — render /usage from the injected state, no portal.
    try:
        from agent.credits_tracker import dev_fixture_credits_state

        fixture = dev_fixture_credits_state()
    except Exception:
        fixture = None
    if fixture is not None:
        snapshot = _snapshot_from_credits_state(fixture)
        return render_account_usage_lines(snapshot, markdown=markdown)

    try:
        from hermes_cli.auth import get_provider_auth_state

        tok = (get_provider_auth_state("nous") or {}).get("access_token")
        if not (isinstance(tok, str) and tok.strip()):
            return []
    except Exception:
        return []
    try:
        import concurrent.futures

        from hermes_cli.nous_account import get_nous_portal_account_info

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            account = pool.submit(
                get_nous_portal_account_info, force_fresh=True
            ).result(timeout=timeout)
        snapshot = build_nous_credits_snapshot(account)
        return render_account_usage_lines(snapshot, markdown=markdown)
    except Exception:
        # Fail-open (caller shows nothing), but leave a breadcrumb so a dead
        # /usage credits block is diagnosable in agent.log without a dev flag.
        logger.debug("credits ▸ /usage portal fetch/render failed (fail-open)", exc_info=True)
        return []


def _snapshot_from_credits_state(state) -> Optional[AccountUsageSnapshot]:
    """Map a header-shaped CreditsState (e.g. a dev fixture) to the /usage snapshot.

    Renders the same magnitudes + monthly-grant % window the portal path produces,
    so HERMES_DEV_CREDITS_FIXTURE can exercise /usage without a live account. The
    *_usd strings are mock display values here (not server balance to compute on);
    the % comes from CreditsState.used_fraction (micros math). Fail-open → None.
    """
    try:
        if state is None:
            return None

        windows: list[AccountUsageWindow] = []
        details: list[str] = []

        uf = getattr(state, "used_fraction", None)
        if isinstance(uf, (int, float)) and math.isfinite(uf):
            cap_usd = getattr(state, "subscription_limit_usd", None)
            sub_usd = getattr(state, "subscription_usd", None)
            detail = None
            if sub_usd and cap_usd:
                detail = f"${sub_usd} of ${cap_usd} left"
            windows.append(
                AccountUsageWindow(
                    label="Subscription",
                    used_percent=max(0.0, min(100.0, uf * 100.0)),
                    detail=detail,
                )
            )

        sub_usd = getattr(state, "subscription_usd", None)
        if sub_usd:
            details.append(f"Subscription credits: ${sub_usd}")
        purchased_usd = getattr(state, "purchased_usd", None)
        if purchased_usd:
            details.append(f"Top-up credits: ${purchased_usd}")
        remaining_usd = getattr(state, "remaining_usd", None)
        if remaining_usd:
            details.append(f"Total usable: ${remaining_usd}")
        if getattr(state, "paid_access", True) is False:
            details.append("Status: access depleted — top up to restore")

        if not windows and not details:
            return None

        details.append("(dev fixture — HERMES_DEV_CREDITS_FIXTURE)")
        return AccountUsageSnapshot(
            provider="nous",
            source="dev-fixture",
            fetched_at=_utc_now(),
            title="Nous credits",
            windows=tuple(windows),
            details=tuple(details),
        )
    except (AttributeError, TypeError):
        return None


@dataclass(frozen=True)
class CreditsView:
    """Surface-agnostic data for the ``/topup`` balance view.

    One portal fetch, one parse — consumed identically by the CLI panel, the
    gateway button, and any other money surface. Fail-open: when not logged in
    or the portal is unreachable, ``logged_in`` is False / ``topup_url`` is None
    and callers degrade gracefully.
    """

    logged_in: bool
    balance_lines: tuple[str, ...] = ()
    identity_line: Optional[str] = None
    topup_url: Optional[str] = None
    depleted: bool = False


def build_credits_view(*, markdown: bool = False, timeout: float = 10.0) -> CreditsView:
    """Build the /topup balance view: balance block + identity line + top-up URL.

    Reuses the same account fetch + snapshot + URL builder as the /usage credits
    block, so the numbers always match. The balance block is the rendered
    snapshot MINUS its trailing top-up/command-hint lines (the /topup surface
    supplies its own affordance). Fail-open → ``CreditsView(logged_in=False)``.
    """
    not_logged_in = CreditsView(logged_in=False)
    try:
        from hermes_cli.auth import get_provider_auth_state

        tok = (get_provider_auth_state("nous") or {}).get("access_token")
        if not (isinstance(tok, str) and tok.strip()):
            return not_logged_in
    except Exception:
        return not_logged_in

    try:
        import concurrent.futures

        from hermes_cli.nous_account import (
            get_nous_portal_account_info,
            nous_portal_topup_url,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            account = pool.submit(get_nous_portal_account_info, force_fresh=True).result(
                timeout=timeout
            )
    except Exception:
        logger.debug("credits ▸ /topup portal fetch failed (fail-open)", exc_info=True)
        return not_logged_in

    if account is None or not getattr(account, "logged_in", False):
        return not_logged_in

    snapshot = build_nous_credits_snapshot(account)
    # Balance lines = the snapshot block minus the two trailing affordance lines
    # ("Top up: <url>" + "(or run /topup)") that build_nous_credits_snapshot
    # appends for the /usage surface. /topup renders its own button/panel.
    balance_lines: list[str] = []
    if snapshot is not None:
        rendered = render_account_usage_lines(snapshot, markdown=markdown)
        balance_lines = [
            line
            for line in rendered
            if not line.lstrip().startswith("Top up:")
            and not line.lstrip().startswith("(or run")
        ]

    # Identity line — shown before any open (roadmap §4.4).
    email = getattr(account, "email", None)
    org_name = getattr(account, "org_name", None)
    who: list[str] = []
    if email:
        who.append(str(email))
    if org_name:
        who.append(f"org {org_name}")
    identity_line = ("Topping up as " + " / ".join(who)) if who else None

    return CreditsView(
        logged_in=True,
        balance_lines=tuple(balance_lines),
        identity_line=identity_line,
        topup_url=nous_portal_topup_url(account),
        depleted=getattr(account, "paid_service_access", None) is False,
    )


def _codex_backend_urls(base_url: str) -> tuple[str, str, str]:
    """Resolve the Codex backend endpoints (usage, reset-credits list, consume).

    Mirrors the Codex CLI's PathStyle split (codex-rs backend-client): base URLs
    containing ``/backend-api`` use the ChatGPT ``/wham/...`` paths; everything
    else uses ``/api/codex/...``.
    """
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://chatgpt.com/backend-api/codex"
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    prefix = normalized + ("/wham" if "/backend-api" in normalized else "/api/codex")
    return (
        prefix + "/usage",
        prefix + "/rate-limit-reset-credits",
        prefix + "/rate-limit-reset-credits/consume",
    )


def _resolve_codex_usage_url(base_url: str) -> str:
    return _codex_backend_urls(base_url)[0]


def _resolve_codex_usage_credentials(
    base_url: Optional[str],
    api_key: Optional[str],
) -> tuple[str, str, Optional[str]]:
    """Resolve Codex quota credentials from the native runtime path.

    Prefer explicit live-agent credentials, then the legacy singleton OAuth
    state, then the credential pool.  Hermes's native OAuth setup now stores
    device-code logins in the pool, so quota diagnostics must not depend only
    on the older singleton store.
    """
    explicit_key = str(api_key or "").strip()
    if explicit_key:
        return explicit_key, str(base_url or "").strip(), None

    # Tier 2: the native runtime resolver. It ALREADY falls back to the
    # credential pool when the singleton is empty (see
    # ``resolve_codex_runtime_credentials`` — issue #32992), so in a pool-only
    # setup this returns a usable ``source="credential_pool"`` token.
    #
    # Only ``AuthError`` ("no creds" / rate-limited) is caught so tier 3 can
    # run: a broad ``except Exception`` would (a) mask a transient refresh /
    # network failure and silently hand back a DIFFERENT pool account's usage,
    # and (b) hide genuine programming errors. A refresh/network error must
    # propagate — the outer ``fetch_account_usage`` guard fails open (shows
    # nothing this turn) rather than reporting the wrong account.
    #
    # The ``account_id`` (for the ``ChatGPT-Account-Id`` header) is read
    # best-effort: a partial/missing singleton token store must not sink an
    # otherwise-usable resolver credential and force a header-less pool fallback.
    try:
        creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        account_id: Optional[str] = None
        try:
            token_data = _read_codex_tokens()
            tokens = token_data.get("tokens") or {}
            account_id = str(tokens.get("account_id", "") or "").strip() or None
        except AuthError:
            # Pool-only creds carry no singleton account_id; header is optional.
            logger.debug("codex ▸ /usage account_id read failed (best-effort)", exc_info=True)
        return creds["api_key"], str(creds.get("base_url", "") or "").strip(), account_id
    except AuthError:
        logger.debug("codex ▸ /usage runtime resolver returned no creds; trying pool", exc_info=True)

    # Tier 3: direct pool select. Reached only when the resolver itself raises
    # AuthError (e.g. singleton missing AND its own pool read found nothing at
    # resolve time, but a pool entry is usable now). Pool credentials have no
    # account_id concept, so the ChatGPT-Account-Id header is intentionally
    # omitted here.
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = pool.select()
    if entry is None:
        raise RuntimeError("No available openai-codex credential in credential pool")
    return entry.runtime_api_key, str(entry.runtime_base_url or base_url or "").strip(), None


def _fetch_codex_account_usage(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    token, resolved_base_url, account_id = _resolve_codex_usage_credentials(base_url, api_key)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    with httpx.Client(timeout=15.0) as client:
        response = client.get(_resolve_codex_usage_url(resolved_base_url), headers=headers)
        response.raise_for_status()
    payload = response.json() or {}
    rate_limit = payload.get("rate_limit") or {}
    windows: list[AccountUsageWindow] = []
    for key, label in (("primary_window", "Session"), ("secondary_window", "Weekly")):
        window = rate_limit.get(key) or {}
        used = window.get("used_percent")
        if used is None:
            continue
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=float(used),
                reset_at=_parse_dt(window.get("reset_at")),
            )
        )
    details: list[str] = []
    reset_credits = payload.get("rate_limit_reset_credits") or {}
    banked = reset_credits.get("available_count")
    if isinstance(banked, (int, float)) and int(banked) > 0:
        count = int(banked)
        plural = "s" if count != 1 else ""
        details.append(
            f"You have {count} reset{plural} banked - use /usage reset to activate"
        )
    credits = payload.get("credits") or {}
    if credits.get("has_credits"):
        balance = credits.get("balance")
        if isinstance(balance, (int, float)):
            details.append(f"Credits balance: ${float(balance):.2f}")
        elif credits.get("unlimited"):
            details.append("Credits balance: unlimited")
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=_utc_now(),
        plan=_title_case_slug(payload.get("plan_type")),
        windows=tuple(windows),
        details=tuple(details),
    )


@dataclass(frozen=True)
class CodexResetRedeemResult:
    """Outcome of a `/usage reset` attempt against the Codex backend."""

    status: str  # reset | nothing_to_reset | no_credit | already_redeemed |
    #              not_exhausted | no_credits_banked | unavailable
    message: str
    available_count: int = 0
    windows_reset: int = 0

    @property
    def redeemed(self) -> bool:
        return self.status == "reset"


# Client-side guard threshold: a rate-limit window only counts as exhausted
# when it is fully used. Below this, redeeming a banked reset wastes most of
# its value, so we block and point at --force instead.
_CODEX_WINDOW_EXHAUSTED_PERCENT = 100.0


def redeem_codex_reset_credit(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    force: bool = False,
) -> CodexResetRedeemResult:
    """Redeem one banked Codex rate-limit reset credit (`/usage reset`).

    Flow (mirrors the Codex CLI's reset-credits picker, codex-rs
    ``backend-client``):

    1. ``GET .../usage`` — read the current windows + banked credit count.
    2. Guard: zero banked credits → refuse. No window fully used and not
       ``force`` → refuse with a warning (a banked reset restores the WHOLE
       5h + weekly allowance; burning it early wastes it). The backend has
       the same protection (``nothing_to_reset`` doesn't consume the
       credit), but failing fast client-side gives a clearer message.
    3. ``POST .../rate-limit-reset-credits/consume`` with a fresh UUID
       idempotency key (``redeem_request_id``). No ``credit_id`` — the
       backend picks the next available credit, exactly like the CLI's
       default "Full reset" option.

    Never raises: every failure mode returns a ``CodexResetRedeemResult``
    with a user-renderable message.
    """
    import uuid

    try:
        token, resolved_base_url, account_id = _resolve_codex_usage_credentials(base_url, api_key)
    except Exception:
        return CodexResetRedeemResult(
            status="unavailable",
            message="No Codex credentials available. Run `hermes auth` to sign in with your ChatGPT account.",
        )
    usage_url, _credits_url, consume_url = _codex_backend_urls(resolved_base_url)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id

    try:
        with httpx.Client(timeout=15.0) as client:
            usage_resp = client.get(usage_url, headers=headers)
            usage_resp.raise_for_status()
            payload = usage_resp.json() or {}

            reset_credits = payload.get("rate_limit_reset_credits") or {}
            raw_count = reset_credits.get("available_count")
            available = int(raw_count) if isinstance(raw_count, (int, float)) else 0
            if available <= 0:
                return CodexResetRedeemResult(
                    status="no_credits_banked",
                    message="No banked reset credits on this account — nothing to redeem.",
                )

            rate_limit = payload.get("rate_limit") or {}
            worst_used: Optional[float] = None
            for key in ("primary_window", "secondary_window"):
                used = (rate_limit.get(key) or {}).get("used_percent")
                if isinstance(used, (int, float)):
                    worst_used = max(worst_used or 0.0, float(used))
            exhausted = worst_used is not None and worst_used >= _CODEX_WINDOW_EXHAUSTED_PERCENT
            if not exhausted and not force:
                usage_note = (
                    f"your busiest window is only {worst_used:.0f}% used"
                    if worst_used is not None
                    else "your current usage could not be confirmed as exhausted"
                )
                plural = "s" if available != 1 else ""
                return CodexResetRedeemResult(
                    status="not_exhausted",
                    message=(
                        f"⚠️ Not redeeming: {usage_note}. A banked reset restores your FULL "
                        f"5h + weekly limits, so spending it now would waste most of it. "
                        f"You have {available} reset{plural} banked. "
                        f"Use `/usage reset --force` to redeem anyway."
                    ),
                    available_count=available,
                )

            consume_resp = client.post(
                consume_url,
                headers={**headers, "Content-Type": "application/json"},
                json={"redeem_request_id": str(uuid.uuid4())},
            )
            consume_resp.raise_for_status()
            body = consume_resp.json() or {}
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            return CodexResetRedeemResult(
                status="unavailable",
                message=(
                    "Codex backend rejected the request (HTTP "
                    f"{code}). Reset credits require ChatGPT-account (OAuth) auth — "
                    "run `hermes auth` and sign in with your ChatGPT account."
                ),
            )
        return CodexResetRedeemResult(
            status="unavailable",
            message=f"Codex backend error (HTTP {code}) — try again shortly.",
        )
    except Exception as exc:
        return CodexResetRedeemResult(
            status="unavailable",
            message=f"Could not reach the Codex backend: {exc}",
        )

    code = str(body.get("code", "") or "").strip().lower()
    windows_reset = body.get("windows_reset")
    windows_reset = int(windows_reset) if isinstance(windows_reset, (int, float)) else 0
    remaining = max(0, available - 1)
    plural = "s" if remaining != 1 else ""
    if code == "reset":
        return CodexResetRedeemResult(
            status="reset",
            message=(
                f"✅ Reset redeemed — your usage limits have been reset. "
                f"{remaining} banked reset{plural} remaining."
            ),
            available_count=remaining,
            windows_reset=windows_reset,
        )
    if code == "nothing_to_reset":
        return CodexResetRedeemResult(
            status="nothing_to_reset",
            message=(
                "Backend reports nothing to reset — your limits aren't exhausted. "
                "The credit was NOT spent."
            ),
            available_count=available,
        )
    if code == "no_credit":
        return CodexResetRedeemResult(
            status="no_credit",
            message="Backend reports no available reset credit on this account.",
        )
    if code == "already_redeemed":
        return CodexResetRedeemResult(
            status="already_redeemed",
            message="This redemption was already processed — no additional credit was spent.",
            available_count=remaining,
        )
    return CodexResetRedeemResult(
        status="unavailable",
        message=f"Unexpected response from the Codex backend: {body!r}",
    )


def _fetch_anthropic_account_usage() -> Optional[AccountUsageSnapshot]:
    token = (resolve_anthropic_token() or "").strip()
    if not token:
        return None
    if not _is_oauth_token(token):
        return AccountUsageSnapshot(
            provider="anthropic",
            source="oauth_usage_api",
            fetched_at=_utc_now(),
            unavailable_reason="Anthropic account limits are only available for OAuth-backed Claude accounts.",
        )
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "claude-code/2.1.0",
        }
        with httpx.Client(timeout=15.0) as client:
            response = client.get("https://api.anthropic.com/api/oauth/usage", headers=headers)
            response.raise_for_status()
        payload = response.json() or {}
        windows: list[AccountUsageWindow] = []
        mapping = (
            ("five_hour", "Current session"),
            ("seven_day", "Current week"),
            ("seven_day_opus", "Opus week"),
            ("seven_day_sonnet", "Sonnet week"),
        )
        for key, label in mapping:
            window = payload.get(key) or {}
            util = window.get("utilization")
            if util is None:
                continue
            used = float(util) * 100 if float(util) <= 1 else float(util)
            windows.append(
                AccountUsageWindow(
                    label=label,
                    used_percent=used,
                    reset_at=_parse_dt(window.get("resets_at")),
                )
            )
        details: list[str] = []
        extra = payload.get("extra_usage") or {}
        if extra.get("is_enabled"):
            used_credits = extra.get("used_credits")
            monthly_limit = extra.get("monthly_limit")
            currency = extra.get("currency") or "USD"
            if isinstance(used_credits, (int, float)) and isinstance(monthly_limit, (int, float)):
                details.append(
                    f"Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {currency}"
                )
        return AccountUsageSnapshot(
            provider="anthropic",
            source="oauth_usage_api",
            fetched_at=_utc_now(),
            windows=tuple(windows),
            details=tuple(details),
        )
    except Exception as exc:
        logger.debug("quota ▸ Anthropic lookup failed", exc_info=True)
        return AccountUsageSnapshot(
            provider="anthropic",
            source="oauth_usage_api",
            fetched_at=_utc_now(),
            unavailable_reason="Quota lookup failed.",
        )


def _fetch_openrouter_account_usage(base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    runtime = resolve_runtime_provider(
        requested="openrouter",
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None
    try:
        normalized = str(runtime.get("base_url", "") or "").rstrip("/")
        credits_url = f"{normalized}/credits"
        key_url = f"{normalized}/key"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        key_data = {}
        credits_data = {}
        with httpx.Client(timeout=10.0) as client:
            key_resp = client.get(key_url, headers=headers)
            key_resp.raise_for_status()
            key_data = (key_resp.json() or {}).get("data") or {}
            try:
                credits_resp = client.get(credits_url, headers=headers)
                credits_resp.raise_for_status()
                credits_data = (credits_resp.json() or {}).get("data") or {}
            except Exception:
                logger.debug("quota ▸ OpenRouter /credits lookup failed (optional)", exc_info=True)

        details = []
        if credits_data:
            total_credits = float(credits_data.get("total_credits") or 0.0)
            total_usage = float(credits_data.get("total_usage") or 0.0)
            details.append(f"Credits balance: ${max(0.0, total_credits - total_usage):.2f}")
        windows: list[AccountUsageWindow] = []
        limit = key_data.get("limit")
        limit_remaining = key_data.get("limit_remaining")
        limit_reset = str(key_data.get("limit_reset") or "").strip()
        usage = key_data.get("usage")
        if (
            isinstance(limit, (int, float))
            and float(limit) > 0
            and isinstance(limit_remaining, (int, float))
            and 0 <= float(limit_remaining) <= float(limit)
        ):
            limit_value = float(limit)
            remaining_value = float(limit_remaining)
            used_percent = ((limit_value - remaining_value) / limit_value) * 100
            detail_parts = [f"${remaining_value:.2f} of ${limit_value:.2f} remaining"]
            if limit_reset:
                detail_parts.append(f"resets {limit_reset}")
            windows.append(
                AccountUsageWindow(
                    label="API key quota",
                    used_percent=used_percent,
                    detail=" • ".join(detail_parts),
                )
            )
        if isinstance(usage, (int, float)):
            usage_parts = [f"API key usage: ${float(usage):.2f} total"]
            for value, label in (
                (key_data.get("usage_daily"), "today"),
                (key_data.get("usage_weekly"), "this week"),
                (key_data.get("usage_monthly"), "this month"),
            ):
                if isinstance(value, (int, float)) and float(value) > 0:
                    usage_parts.append(f"${float(value):.2f} {label}")
            details.append(" • ".join(usage_parts))
        return AccountUsageSnapshot(
            provider="openrouter",
            source="credits_api",
            fetched_at=_utc_now(),
            windows=tuple(windows),
            details=tuple(details),
        )
    except Exception as exc:
        logger.debug("quota ▸ OpenRouter lookup failed", exc_info=True)
        return AccountUsageSnapshot(
            provider="openrouter",
            source="credits_api",
            fetched_at=_utc_now(),
            unavailable_reason="Quota lookup failed.",
        )
    windows: list[AccountUsageWindow] = []
    limit = key_data.get("limit")
    limit_remaining = key_data.get("limit_remaining")
    limit_reset = str(key_data.get("limit_reset") or "").strip()
    usage = key_data.get("usage")
    if (
        isinstance(limit, (int, float))
        and float(limit) > 0
        and isinstance(limit_remaining, (int, float))
        and 0 <= float(limit_remaining) <= float(limit)
    ):
        limit_value = float(limit)
        remaining_value = float(limit_remaining)
        used_percent = ((limit_value - remaining_value) / limit_value) * 100
        detail_parts = [f"${remaining_value:.2f} of ${limit_value:.2f} remaining"]
        if limit_reset:
            detail_parts.append(f"resets {limit_reset}")
        windows.append(
            AccountUsageWindow(
                label="API key quota",
                used_percent=used_percent,
                detail=" • ".join(detail_parts),
            )
        )
    if isinstance(usage, (int, float)):
        usage_parts = [f"API key usage: ${float(usage):.2f} total"]
        for value, label in (
            (key_data.get("usage_daily"), "today"),
            (key_data.get("usage_weekly"), "this week"),
            (key_data.get("usage_monthly"), "this month"),
        ):
            if isinstance(value, (int, float)) and float(value) > 0:
                usage_parts.append(f"${float(value):.2f} {label}")
        details.append(" • ".join(usage_parts))
    return AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_deepseek_account_usage(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    """Fetch DeepSeek balance using the configured runtime credential."""
    runtime = resolve_runtime_provider(
        requested="deepseek",
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None
    try:
        runtime_base_url = str(runtime.get("base_url", "") or "").strip().rstrip("/")
        if runtime_base_url:
            for _ in range(2):
                if runtime_base_url.lower().endswith("/v1"):
                    runtime_base_url = runtime_base_url[:-3].rstrip("/")
                elif runtime_base_url.lower().endswith("/anthropic"):
                    runtime_base_url = runtime_base_url[:-10].rstrip("/")
            balance_url = f"{runtime_base_url}/user/balance" if runtime_base_url else "https://api.deepseek.com/user/balance"
        else:
            balance_url = "https://api.deepseek.com/user/balance"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        with httpx.Client(timeout=10.0) as client:
            response = client.get(balance_url, headers=headers)
            response.raise_for_status()
        payload = response.json() or {}
        details: list[str] = []
        for balance in payload.get("balance_infos") or ():
            if not isinstance(balance, dict):
                continue
            currency = str(balance.get("currency") or "").strip().upper()
            total = balance.get("total_balance")
            if not currency or total is None:
                continue
            symbol = "¥" if currency == "CNY" else "$"
            details.append(f"Balance ({currency}): {symbol}{total}")
        if not details and payload.get("is_available") is not None:
            status = "Available" if payload.get("is_available") else "Low balance"
            details.append(f"Status: {status}")
        if not details:
            raise ValueError("No balance details returned by DeepSeek API")
        return AccountUsageSnapshot(
            provider="deepseek",
            source="balance_api",
            fetched_at=_utc_now(),
            details=tuple(details),
        )
    except Exception as exc:
        logger.debug("quota ▸ DeepSeek lookup failed", exc_info=True)
        return AccountUsageSnapshot(
            provider="deepseek",
            source="balance_api",
            fetched_at=_utc_now(),
            unavailable_reason="Quota lookup failed.",
        )
def _antigravity_details_from_quota_summary(payload: dict[str, Any]) -> list[str]:
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        return []

    details: list[str] = []
    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        group_name = str(group.get("displayName") or "").strip()
        buckets = group.get("buckets")
        if not group_name or not isinstance(buckets, list):
            continue

        group_details: list[str] = []
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            remaining = bucket.get("remainingFraction")
            if not _is_finite_num(remaining):
                continue
            label = str(bucket.get("displayName") or bucket.get("window") or "Quota").strip()
            percent = round(max(0.0, min(1.0, float(remaining))) * 100)
            group_details.extend((f"  {label}", f"    {percent}% left"))
            reset_at = _parse_dt(bucket.get("resetTime"))
            if reset_at is not None:
                group_details.append(f"    Resets {_format_reset(reset_at)}")

        if group_details:
            details.append(group_name)
            details.extend(group_details)
    return details


def _fetch_antigravity_account_usage() -> Optional[AccountUsageSnapshot]:
    runtime = resolve_runtime_provider(requested="google-antigravity")
    access_token = str(runtime.get("api_key", "") or "").strip()
    if not access_token:
        return None
    project_id = str(runtime.get("project_id", "") or "").strip()

    try:
        from agent.antigravity_code_assist import retrieve_user_quota_summary_antigravity

        payload = retrieve_user_quota_summary_antigravity(
            access_token,
            project_id=project_id,
        )
    except Exception:
        logger.debug("quota ▸ Google Antigravity lookup failed", exc_info=True)
        return AccountUsageSnapshot(
            provider="google-antigravity",
            source="oauth_quota_summary_api",
            fetched_at=_utc_now(),
            unavailable_reason="Quota lookup failed.",
        )

    details = _antigravity_details_from_quota_summary(payload)
    if not details:
        return AccountUsageSnapshot(
            provider="google-antigravity",
            source="oauth_quota_summary_api",
            fetched_at=_utc_now(),
            unavailable_reason="No quota groups reported.",
        )
    return AccountUsageSnapshot(
        provider="google-antigravity",
        source="oauth_quota_summary_api",
        fetched_at=_utc_now(),
        details=tuple(details),
    )


def _parse_opencode_go_window(html: str, field: str) -> Optional[tuple[float, float]]:
    """Extract usage percentage and reset seconds from the Go dashboard payload."""
    import re

    patterns = (
        rf"{field}:\$R\[\d+\]=\{{[^}}]*usagePercent:(-?\d+(?:\.\d+)?)[^}}]*resetInSec:(-?\d+(?:\.\d+)?)[^}}]*\}}",
        rf"{field}:\$R\[\d+\]=\{{[^}}]*resetInSec:(-?\d+(?:\.\d+)?)[^}}]*usagePercent:(-?\d+(?:\.\d+)?)[^}}]*\}}",
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, html)
        if not match:
            continue
        first, second = (max(0.0, float(value)) for value in match.groups())
        return (first, second) if index == 0 else (second, first)
    return None


def _fetch_opencode_go_quota() -> Optional[AccountUsageSnapshot]:
    """Read local OpenCode Go subscription limits from its authenticated dashboard."""
    workspace_id = os.getenv("OPENCODE_GO_WORKSPACE_ID", "").strip()
    auth_cookie = os.getenv("OPENCODE_GO_AUTH_COOKIE", "").strip()
    if not workspace_id or not auth_cookie:
        return None

    from urllib.parse import quote

    url = f"https://opencode.ai/workspace/{quote(workspace_id)}/go"
    headers = {
        "User-Agent": "hermes-agent/1.0",
        "Accept": "text/html",
        "Cookie": f"auth={auth_cookie}",
    }
    try:
        response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
    except Exception:
        logger.debug("quota ▸ OpenCode Go dashboard lookup failed", exc_info=True)
        return None

    fetched_at = _utc_now()
    windows: list[AccountUsageWindow] = []
    for field, label in (("rollingUsage", "5h"), ("weeklyUsage", "Weekly"), ("monthlyUsage", "Monthly")):
        parsed = _parse_opencode_go_window(response.text, field)
        if parsed is None:
            continue
        used_percent, reset_seconds = parsed
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=min(100.0, used_percent),
                reset_at=fetched_at + timedelta(seconds=reset_seconds),
            )
        )
    if not windows:
        return None
    return AccountUsageSnapshot(
        provider="opencode-go",
        source="dashboard_scrape",
        fetched_at=fetched_at,
        windows=tuple(windows),
    )


def fetch_account_usage(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    normalized = str(provider or "").strip().lower()
    if normalized in {"", "auto", "custom"}:
        return None
    try:
        if normalized == "openai-codex":
            return _fetch_codex_account_usage(base_url=base_url, api_key=api_key)
        if normalized == "anthropic":
            return _fetch_anthropic_account_usage()
        if normalized == "openrouter":
            return _fetch_openrouter_account_usage(base_url, api_key)
        if normalized == "deepseek":
            return _fetch_deepseek_account_usage(base_url, api_key)
        if normalized == "google-antigravity":
            return _fetch_antigravity_account_usage()
        if normalized == "opencode-go":
            return _fetch_opencode_go_quota()
    except Exception:
        return None
    return None


_QUOTA_PROVIDER_IDS = (
    "openai-codex",
    "anthropic",
    "openrouter",
    "google-antigravity",
    "deepseek",
    "opencode-go",
)


def fetch_all_providers_quota() -> list[AccountUsageSnapshot]:
    """Fetch quota data for supported providers with configured credentials.

    Each provider's existing runtime/auth resolution is authoritative. This keeps
    aggregate discovery aligned with normal provider configuration, including
    OpenRouter credentials that are not exposed as environment variables.
    """
    snapshots: list[AccountUsageSnapshot] = []
    for provider in _QUOTA_PROVIDER_IDS:
        try:
            snapshot = fetch_account_usage(provider)
        except Exception:
            logger.debug("quota ▸ %s lookup failed", provider, exc_info=True)
            continue
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots
