from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
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
        details.append("(or run /credits)")

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
    """Surface-agnostic data for the ``/credits`` command.

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
    """Build the /credits view: balance block + identity line + top-up URL.

    Reuses the same account fetch + snapshot + URL builder as the /usage credits
    block, so the numbers always match. The balance block is the rendered
    snapshot MINUS its trailing top-up/command-hint lines (the /credits surface
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
        logger.debug("credits ▸ /credits portal fetch failed (fail-open)", exc_info=True)
        return not_logged_in

    if account is None or not getattr(account, "logged_in", False):
        return not_logged_in

    snapshot = build_nous_credits_snapshot(account)
    # Balance lines = the snapshot block minus the two trailing affordance lines
    # ("Top up: <url>" + "(or run /credits)") that build_nous_credits_snapshot
    # appends for the /usage surface. /credits renders its own button/panel.
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


def _resolve_codex_usage_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://chatgpt.com/backend-api/codex"
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    if "/backend-api" in normalized:
        return normalized + "/wham/usage"
    return normalized + "/api/codex/usage"


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


def _fetch_openrouter_account_usage(base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    runtime = resolve_runtime_provider(
        requested="openrouter",
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None
    normalized = str(runtime.get("base_url", "") or "").rstrip("/")
    credits_url = f"{normalized}/credits"
    key_url = f"{normalized}/key"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=10.0) as client:
        credits_resp = client.get(credits_url, headers=headers)
        credits_resp.raise_for_status()
        credits = (credits_resp.json() or {}).get("data") or {}
        try:
            key_resp = client.get(key_url, headers=headers)
            key_resp.raise_for_status()
            key_data = (key_resp.json() or {}).get("data") or {}
        except Exception:
            key_data = {}
    total_credits = float(credits.get("total_credits") or 0.0)
    total_usage = float(credits.get("total_usage") or 0.0)
    details = [f"Credits balance: ${max(0.0, total_credits - total_usage):.2f}"]
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


def _fetch_zai_account_usage(api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    """Fetch Z.AI (Zhipu) token quota via undocumented monitoring endpoint."""
    token = str(api_key or "").strip()
    if not token:
        return None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(
            "https://api.z.ai/api/monitor/usage/quota/limit",
            headers=headers,
        )
        resp.raise_for_status()
    payload = resp.json() or {}
    data = payload.get("data") or {}
    windows: list[AccountUsageWindow] = []
    details: list[str] = []

    limits = data.get("limits") or []
    for lim in limits:
        lim_type = str(lim.get("type") or "")
        pct = lim.get("percentage")
        next_reset_ms = lim.get("nextResetTime")
        remaining = lim.get("remaining")

        if lim_type == "TOKENS_LIMIT" and pct is not None:
            unit_h = int(lim.get("unit") or 0)
            num_w = int(lim.get("number") or 0)
            label = f"Tokens ({num_w * unit_h}h window)"
            reset_dt = _parse_dt(next_reset_ms / 1000) if next_reset_ms else None
            windows.append(
                AccountUsageWindow(
                    label=label,
                    used_percent=float(pct),
                    reset_at=reset_dt,
                )
            )
        elif lim_type == "TIME_LIMIT" and pct is not None:
            reset_dt = _parse_dt(next_reset_ms / 1000) if next_reset_ms else None
            windows.append(
                AccountUsageWindow(
                    label="Requests",
                    used_percent=float(pct),
                    reset_at=reset_dt,
                    detail=f"{remaining} remaining" if remaining else None,
                )
            )

    level = data.get("level")
    if level:
        details.append(f"Plan: {level}")

    if not windows and not details:
        return None

    return AccountUsageSnapshot(
        provider="zai",
        source="quota_monitor_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_cloudflare_account_usage(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    """Fetch Cloudflare Workers AI neuron usage via GraphQL Analytics API.

    Requires an API Token (not URL token ``cfut_``) with
    ``Account > Workers AI Analytics > Read`` scope.
    """
    token = str(api_key or "").strip()
    if not token:
        return None

    # Extract account_id from base_url
    # e.g. https://api.cloudflare.com/client/v4/accounts/{id}/ai/v1
    account_id: Optional[str] = None
    if base_url:
        import re

        m = re.search(r"/accounts/([0-9a-f]+)", base_url)
        if m:
            account_id = m.group(1)
    if not account_id:
        return None

    # Free tier: 10,000 neurons/day
    FREE_TIER_DAILY = 10_000

    # Query today's neuron usage
    now = _utc_now()
    today_start = now.strftime("%Y-%m-%dT00:00:00Z")
    tomorrow_start = (now.replace(hour=0, minute=0, second=0, microsecond=0))
    from datetime import timedelta as _td

    tomorrow_start = (tomorrow_start + _td(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    gql_query = {
        "query": (
            "{ viewer { accounts(filter: { accountTag: \""
            + account_id
            + "\" }) { aiInferenceAdaptiveGroups("
            "filter: { datetime_geq: \"" + today_start + "\", "
            "datetime_leq: \"" + tomorrow_start + "\" }, limit: 1000"
            ") { sum { totalNeurons totalInputTokens totalOutputTokens } "
            "dimensions { modelId } } } } }"
        ),
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                "https://api.cloudflare.com/client/v4/graphql",
                json=gql_query,
                headers=headers,
            )
            resp.raise_for_status()
    except Exception:
        return None

    payload = resp.json() or {}
    if payload.get("errors"):
        # Token lacks analytics scope (common with cfut_ URL tokens)
        return AccountUsageSnapshot(
            provider="cloudflare",
            source="graphql_analytics",
            fetched_at=_utc_now(),
            unavailable_reason=(
                "Token lacks Workers AI Analytics scope. "
                "Create an API Token with "
                "Account > Workers AI Analytics > Read."
            ),
        )

    accounts_data = (payload.get("data") or {}).get("viewer", {}).get("accounts", [])
    if not accounts_data:
        return None

    groups = accounts_data[0].get("aiInferenceAdaptiveGroups") or []
    total_neurons = 0
    total_input_tokens = 0
    total_output_tokens = 0
    model_details: list[str] = []
    for g in groups:
        s = g.get("sum") or {}
        total_neurons += int(s.get("totalNeurons") or 0)
        total_input_tokens += int(s.get("totalInputTokens") or 0)
        total_output_tokens += int(s.get("totalOutputTokens") or 0)
        model = (g.get("dimensions") or {}).get("modelId", "?")
        model_neurons = int(s.get("totalNeurons") or 0)
        if model_neurons > 0:
            model_details.append(f"{model}: {model_neurons} neurons")

    if total_neurons == 0 and not model_details:
        return AccountUsageSnapshot(
            provider="cloudflare",
            source="graphql_analytics",
            fetched_at=_utc_now(),
            windows=(
                AccountUsageWindow(
                    label="Neurons (daily)",
                    used_percent=0.0,
                    detail=f"0 / {FREE_TIER_DAILY} neurons used today",
                ),
            ),
            details=("Free tier: 10,000 neurons/day",),
        )

    used_pct = (total_neurons / FREE_TIER_DAILY) * 100 if FREE_TIER_DAILY > 0 else 0
    windows = [
        AccountUsageWindow(
            label="Neurons (daily)",
            used_percent=min(used_pct, 100.0),
            detail=f"{total_neurons} / {FREE_TIER_DAILY} neurons used today",
        ),
    ]
    details = [
        f"Tokens today: {total_input_tokens:,} in / {total_output_tokens:,} out",
        f"Free tier: {FREE_TIER_DAILY:,} neurons/day",
    ]
    # Show top 3 models by neuron usage
    if model_details:
        details.append("Models: " + ", ".join(model_details[:3]))

    return AccountUsageSnapshot(
        provider="cloudflare",
        source="graphql_analytics",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_google_account_usage(api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    """Fetch Google Gemini quota via Cloud Code Assist internal API.

    Uses the same ``v1internal:fetchAvailableModels`` endpoint discovered in
    the Antigravity Cockpit project. Requires a Google OAuth token (not an API
    key) with ``cloud-platform`` scope.
    """
    token = str(api_key or "").strip()
    if not token:
        return None

    # OAuth tokens contain dots (eyJ...), API keys (AIza...) do not
    is_oauth = token.startswith("ya29.") or token.startswith("4/") or "." in token[:20]
    if not is_oauth:
        return AccountUsageSnapshot(
            provider="google",
            source="cloud_code_assist",
            fetched_at=_utc_now(),
            unavailable_reason=(
                "Google quota requires an OAuth token (Antigravity/Gemini Code "
                "Assist credential), not an API key. API keys have no quota "
                "endpoint."
            ),
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    base_url = "https://cloudcode-pa.googleapis.com"

    try:
        with httpx.Client(timeout=10.0) as client:
            # Step 1: loadCodeAssist to get project ID
            resp1 = client.post(
                f"{base_url}/v1internal:loadCodeAssist",
                json={"metadata": {"ideType": "ANTIGRAVITY"}},
                headers=headers,
            )
            resp1.raise_for_status()
            data1 = resp1.json() or {}
            project_id = data1.get("cloudaicompanionProject")
            current_tier = data1.get("currentTier") or data1.get("paidTier")

            if not project_id:
                return AccountUsageSnapshot(
                    provider="google",
                    source="cloud_code_assist",
                    fetched_at=_utc_now(),
                    unavailable_reason="No Cloud Code Assist project found",
                )

            # Step 2: retrieveUserQuota — returns per-model quota buckets
            resp2 = client.post(
                f"{base_url}/v1internal:retrieveUserQuota",
                json={"project": project_id} if project_id else {},
                headers=headers,
            )
            resp2.raise_for_status()
            data2 = resp2.json() or {}
    except Exception:
        return None

    # retrieveUserQuota returns { buckets: [ {modelId, tokenType, remainingFraction, resetTime} ] }
    raw_buckets = data2.get("buckets") or []
    windows: list[AccountUsageWindow] = []
    details: list[str] = []

    if current_tier:
        details.append(f"Tier: {current_tier}")

    _USER_FACING_PREFIXES = ("gemini-", "claude-", "gpt-")
    for b in raw_buckets:
        if not isinstance(b, dict):
            continue
        model_id = str(b.get("modelId") or "")
        # Filter out internal infrastructure buckets
        if model_id and not model_id.startswith(_USER_FACING_PREFIXES):
            continue
        remaining_frac = float(b.get("remainingFraction") or 0.0)
        used_pct = round((1.0 - remaining_frac) * 100.0, 1)
        token_type = str(b.get("tokenType") or "")
        reset_time = str(b.get("resetTime") or "")

        label = model_id or "quota"
        if token_type:
            label += f" [{token_type}]"

        detail_parts = []
        if remaining_frac > 0:
            detail_parts.append(f"{remaining_frac*100:.0f}% remaining")
        if reset_time:
            detail_parts.append(f"resets {reset_time}")
        detail = " | ".join(detail_parts) if detail_parts else None

        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=used_pct,
                detail=detail,
            )
        )

    if not windows:
        return AccountUsageSnapshot(
            provider="google",
            source="cloud_code_assist",
            fetched_at=_utc_now(),
            unavailable_reason="No model quota information available",
        )

    return AccountUsageSnapshot(
        provider="google",
        source="cloud_code_assist",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_nvidia_account_usage(api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    """Fetch NVIDIA NGC account info as a minimal usage indicator.

    NVIDIA does not expose a consumption/quota API endpoint. This function
    retrieves the NGC subscription tier and expiry, which is the closest
    available signal.
    """
    token = str(api_key or "").strip()
    if not token:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            # /v2/orgs returns the org list with enablements
            resp = client.get(
                "https://api.ngc.nvidia.com/v2/orgs",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json() or {}

            # /v2/users/me/subscriptions shows active products
            sub_resp = client.get(
                "https://api.ngc.nvidia.com/v2/users/me/subscriptions",
                headers=headers,
            )
            sub_data = {}
            if sub_resp.status_code == 200:
                sub_data = sub_resp.json() or {}
    except Exception:
        return None

    orgs = data.get("organizations") or []
    details: list[str] = []

    if orgs:
        org = orgs[0]
        display_name = org.get("displayName")
        org_type = org.get("type")
        if display_name:
            label = f"Org: {display_name}"
            if org_type:
                label += f" ({org_type})"
            details.append(label)

        enablements = org.get("productEnablements") or []
        for pe in enablements:
            pe_type = str(pe.get("type") or "")
            product = str(pe.get("productName") or "")
            expiry = str(pe.get("expirationDate") or "")

            if product:
                label = product
                if pe_type:
                    label += f" ({pe_type})"
                if expiry:
                    label += f" -- expires {expiry}"
                details.append(label)

    subs = sub_data.get("subscriptions") or []
    for sub in subs:
        products = sub.get("products") or []
        pending = sub.get("pendingProducts") or []
        expired = sub.get("expiredProducts") or []
        if products:
            details.append(f"Active products: {', '.join(products)}")
        if pending:
            details.append(f"Pending: {', '.join(pending)}")
        if expired:
            details.append(f"Expired: {', '.join(expired)}")

    if not details:
        return None

    return AccountUsageSnapshot(
        provider="nvidia",
        source="ngc_account_api",
        fetched_at=_utc_now(),
        details=tuple(details),
        unavailable_reason=(
            "NVIDIA does not expose a consumption/quota API. "
            "Showing subscription info instead."
        ),
    )


def fetch_account_usage(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    normalized = str(provider or "").strip().lower()
    # Heuristic: resolve real provider from base_url when the name is not
    # directly recognised (e.g. "xai-oauth" hitting api.z.ai for glm models).
    if normalized not in {"openai-codex", "anthropic", "openrouter", "zai", "nous", "cloudflare", "google", "nvidia"}:
        _host = (base_url or "").lower()
        if "api.z.ai" in _host or "open.bigmodel.cn" in _host:
            normalized = "zai"
            # xai-oauth OAuth tokens won't work on the Z.AI quota endpoint;
            # resolve the explicit zai provider key from config.
            if not api_key:
                try:
                    for _name in ("zai", "custom:zai"):
                        _rt = resolve_runtime_provider(
                            requested=_name, explicit_base_url=None, explicit_api_key=None,
                        )
                        api_key = str(_rt.get("api_key", "") or "").strip()
                        if api_key:
                            break
                except Exception:
                    pass
    if normalized in {"", "auto", "custom"}:
        return None
    try:
        if normalized == "openai-codex":
            return _fetch_codex_account_usage(base_url=base_url, api_key=api_key)
        if normalized == "anthropic":
            return _fetch_anthropic_account_usage()
        if normalized == "openrouter":
            return _fetch_openrouter_account_usage(base_url, api_key)
        if normalized == "zai":
            return _fetch_zai_account_usage(api_key)
        if normalized == "cloudflare":
            return _fetch_cloudflare_account_usage(base_url, api_key)
        if normalized == "google":
            return _fetch_google_account_usage(api_key)
        if normalized == "nvidia":
            return _fetch_nvidia_account_usage(api_key)
    except Exception:
        return None
    return None
