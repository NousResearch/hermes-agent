from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpx

from agent.anthropic_credentials import _is_oauth_token, resolve_anthropic_token
from hermes_cli.auth import AuthError, _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.runtime_provider import resolve_runtime_provider

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger(__name__)

_DEPLETED_LINE = "Status: access depleted — top up to restore"


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


def _snapshot(provider: str, source: str, windows: list, details: list, **kw: Any) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(provider=provider, source=source, fetched_at=_utc_now(), windows=tuple(windows), details=tuple(details), **kw)


def _title_case_slug(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    return cleaned.replace("_", " ").replace("-", " ").title() if cleaned else None


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if not isinstance(value, str) or not (text := value.strip()):
        return None
    text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_reset(dt: Optional[datetime]) -> str:
    if not dt:
        return "unknown"
    stamp = dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    total_seconds = int((dt - _utc_now()).total_seconds())
    if total_seconds <= 0:
        return f"now ({stamp})"
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"in {days}d {hours}h ({stamp})"
    return f"in {hours}h {minutes}m ({stamp})" if hours else f"in {minutes}m ({stamp})"


def render_account_usage_lines(snapshot: Optional[AccountUsageSnapshot], *, markdown: bool = False) -> list[str]:
    if not snapshot:
        return []
    bold = "**" if markdown else ""
    plan = f" ({snapshot.plan})" if snapshot.plan else ""
    lines = [f"📈 {bold}{snapshot.title}{bold}", f"Provider: {snapshot.provider}{plan}"]
    for window in snapshot.windows:
        if window.used_percent is None:
            base = f"{window.label}: unavailable"
        else:
            used = float(window.used_percent)
            base = f"{window.label}: {max(0, round(100 - used))}% remaining ({max(0, round(used))}% used)"
        if window.reset_at:
            base += f" • resets {_format_reset(window.reset_at)}"
        elif window.detail:
            base += f" • {window.detail}"
        lines.append(base)
    lines.extend(snapshot.details)
    if snapshot.unavailable_reason:
        lines.append(f"Unavailable: {snapshot.unavailable_reason}")
    return lines


def _fmt_usd(d: float) -> str:
    return f"${d:,.2f}"


def _is_num(v: Any) -> TypeGuard[float]:
    return isinstance(v, (int, float))


def _is_finite_num(v: Any) -> TypeGuard[float]:
    """True iff v is a real number (int/float, not bool, not NaN/Inf); TypeGuard so callers can do arithmetic."""
    return _is_num(v) and not isinstance(v, bool) and math.isfinite(v)


def _nous_snapshot(windows: list, details: list, tail: list, *, source: str, plan: Optional[str] = None) -> Optional[AccountUsageSnapshot]:
    """Nous snapshot with *tail* lines appended, or None when there is nothing to show."""
    if not windows and not details:
        return None
    return _snapshot("nous", source, windows, details + tail, title="Nous credits", plan=plan)


def build_nous_credits_snapshot(account_info) -> Optional[AccountUsageSnapshot]:
    """NousPortalAccountInfo → /usage snapshot: dollar magnitudes + renewal date + portal CTA, plus a ``% used``
    gauge when the portal supplies ``monthly_credits``. Fail-open → None."""
    try:
        from hermes_cli.nous_account import nous_portal_topup_url
        if account_info is None or not getattr(account_info, "logged_in", False):
            return None
        access = getattr(account_info, "paid_service_access_info", None)
        sub = getattr(account_info, "subscription", None)
        windows: list[AccountUsageWindow] = []
        details: list[str] = []
        # Gauge needs a positive cap AND a finite remaining <= cap (numeric fields, NOT a server *_usd); used =
        # cap - remaining clamped [0,100] so debt reads 100%. NaN/Inf (json.loads accepts bare NaN → "$nan") and
        # remaining > cap (rollover makes the cap a meaningless denominator) fall back to the magnitudes lines.
        if sub is not None:
            cap = getattr(sub, "monthly_credits", None)
            sub_remaining = getattr(sub, "credits_remaining", None)
            if _is_finite_num(cap) and cap > 0 and _is_finite_num(sub_remaining) and sub_remaining <= cap:
                windows.append(AccountUsageWindow(
                    label="Subscription", used_percent=max(0.0, min(100.0, (cap - sub_remaining) / cap * 100.0)),
                    detail=f"{_fmt_usd(sub_remaining)} of {_fmt_usd(cap)} left",
                ))
        if access is not None:
            for attr, label in (("subscription_credits_remaining", "Subscription credits"),
                                ("purchased_credits_remaining", "Top-up credits"), ("total_usable_credits", "Total usable")):
                value = getattr(access, attr, None)
                if _is_finite_num(value):
                    details.append(f"{label}: {_fmt_usd(value)}")
        if sub is not None:
            rollover = getattr(sub, "rollover_credits", None)
            if _is_finite_num(rollover) and rollover > 0:
                details.append(f"Rollover: {_fmt_usd(rollover)}")
            period_end = getattr(sub, "current_period_end", None)
            if period_end:
                details.append(f"Renews: {period_end}")
        if getattr(account_info, "paid_service_access", None) is False:
            details.append(_DEPLETED_LINE)
        return _nous_snapshot(windows, details, [f"Top up: {nous_portal_topup_url(account_info)}", "(or run /topup)"],
                              source="portal-account", plan=getattr(sub, "plan", None) if sub is not None else None)
    except (AttributeError, TypeError):
        return None


def _nous_logged_in() -> bool:
    """Cheap local auth-state check: a Nous access token is present. Fail-open False."""
    try:
        from hermes_cli.auth import get_provider_auth_state
        tok = (get_provider_auth_state("nous") or {}).get("access_token")
        return isinstance(tok, str) and bool(tok.strip())
    except Exception:
        return False


def _fetch_portal_account(timeout: float):
    """Wall-clock-bounded fresh portal account fetch (raises on any failure/timeout)."""
    import concurrent.futures
    from hermes_cli.nous_account import get_nous_portal_account_info
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(get_nous_portal_account_info, force_fresh=True).result(timeout=timeout)


def nous_credits_lines(*, markdown: bool = False, timeout: float = 10.0) -> list[str]:
    """Rendered Nous-credits /usage lines, or [] when there's nothing to show. Independent of any live agent
    (logged-in gate, then a bounded portal fetch); shared by CLI ``_show_usage`` and the TUI ``session.usage`` RPC.
    Fail-open: any hiccup or timeout → []. HERMES_DEV_CREDITS_FIXTURE renders from the fixture instead of the portal."""
    try:
        from agent.credits_tracker import dev_fixture_credits_state
        fixture = dev_fixture_credits_state()
    except Exception:
        fixture = None
    if fixture is not None:
        return render_account_usage_lines(_snapshot_from_credits_state(fixture), markdown=markdown)
    if not _nous_logged_in():
        return []
    try:
        snapshot = build_nous_credits_snapshot(_fetch_portal_account(timeout))
        return render_account_usage_lines(snapshot, markdown=markdown)
    except Exception:
        # Fail-open; breadcrumb so a dead /usage credits block is diagnosable.
        logger.debug("credits ▸ /usage portal fetch/render failed (fail-open)", exc_info=True)
        return []


def _snapshot_from_credits_state(state) -> Optional[AccountUsageSnapshot]:
    """Header-shaped CreditsState (dev fixture) → /usage snapshot, same shape as the portal path. *_usd strings
    are display-only; the % comes from CreditsState.used_fraction. Fail-open → None."""
    try:
        if state is None:
            return None
        windows: list[AccountUsageWindow] = []
        details: list[str] = []
        uf = getattr(state, "used_fraction", None)
        sub_usd = getattr(state, "subscription_usd", None)
        cap_usd = getattr(state, "subscription_limit_usd", None)
        if _is_num(uf) and math.isfinite(uf):
            windows.append(AccountUsageWindow(
                label="Subscription", used_percent=max(0.0, min(100.0, uf * 100.0)),
                detail=f"${sub_usd} of ${cap_usd} left" if sub_usd and cap_usd else None,
            ))
        for value, label in ((sub_usd, "Subscription credits"), (getattr(state, "purchased_usd", None), "Top-up credits"),
                             (getattr(state, "remaining_usd", None), "Total usable")):
            if value:
                details.append(f"{label}: ${value}")
        if getattr(state, "paid_access", True) is False:
            details.append(_DEPLETED_LINE)
        return _nous_snapshot(windows, details, ["(dev fixture — HERMES_DEV_CREDITS_FIXTURE)"], source="dev-fixture")
    except (AttributeError, TypeError):
        return None


@dataclass(frozen=True)
class CreditsView:
    """Surface-agnostic ``/topup`` balance view: one portal fetch, consumed identically by every money surface.
    Fail-open: not logged in / portal unreachable → ``logged_in`` False, ``topup_url`` None."""

    logged_in: bool
    balance_lines: tuple[str, ...] = ()
    identity_line: Optional[str] = None
    topup_url: Optional[str] = None
    depleted: bool = False


def build_credits_view(*, markdown: bool = False, timeout: float = 10.0) -> CreditsView:
    """/topup view: balance block + identity line + top-up URL. Reuses the /usage fetch + snapshot so numbers
    match; the balance block drops the trailing top-up/hint lines (/topup has its own affordance).
    Fail-open → ``CreditsView(logged_in=False)``."""
    not_logged_in = CreditsView(logged_in=False)
    if not _nous_logged_in():
        return not_logged_in
    try:
        account = _fetch_portal_account(timeout)
    except Exception:
        logger.debug("credits ▸ /topup portal fetch failed (fail-open)", exc_info=True)
        return not_logged_in
    if account is None or not getattr(account, "logged_in", False):
        return not_logged_in
    from hermes_cli.nous_account import nous_portal_topup_url
    balance_lines = [
        line
        for line in render_account_usage_lines(build_nous_credits_snapshot(account), markdown=markdown)
        if not line.lstrip().startswith(("Top up:", "(or run"))
    ]
    who = [str(v) for v in (getattr(account, "email", None),) if v]
    org_name = getattr(account, "org_name", None)
    if org_name:
        who.append(f"org {org_name}")
    return CreditsView(
        logged_in=True, balance_lines=tuple(balance_lines),
        identity_line=("Topping up as " + " / ".join(who)) if who else None, topup_url=nous_portal_topup_url(account),
        depleted=getattr(account, "paid_service_access", None) is False,
    )


def _codex_backend_urls(base_url: str) -> tuple[str, str, str]:
    """Codex backend endpoints (usage, reset-credits list, consume). Mirrors the Codex CLI's PathStyle
    split: ``/backend-api`` bases use the ChatGPT ``/wham/`` paths; everything else ``/api/codex/``."""
    normalized = (base_url or "").strip().rstrip("/") or "https://chatgpt.com/backend-api/codex"
    normalized = normalized.removesuffix("/codex")
    prefix = normalized + ("/wham" if "/backend-api" in normalized else "/api/codex")
    return (prefix + "/usage", prefix + "/rate-limit-reset-credits", prefix + "/rate-limit-reset-credits/consume")


def _resolve_codex_usage_credentials(
    base_url: Optional[str], api_key: Optional[str],
) -> tuple[str, str, Optional[str]]:
    """Codex quota credentials: explicit live-agent creds → native runtime resolver (itself pool-aware) → direct
    pool select. Native OAuth stores device-code logins in the pool, so the singleton store alone is not enough."""
    explicit_key = str(api_key or "").strip()
    if explicit_key:
        return explicit_key, str(base_url or "").strip(), None
    # Only AuthError is caught so tier 3 can run: a broad except would mask a transient refresh/network failure
    # and hand back a DIFFERENT pool account's usage; such errors must propagate to the fail-open outer guard.
    # account_id is best-effort: a partial singleton store must not sink a usable credential.
    try:
        # Tier 2: the native runtime resolver. It ALREADY falls back to the credential pool when the
        # singleton is empty (see ``resolve_codex_runtime_credentials`` — issue #32992), so in a pool-only
        # setup this returns a usable ``source="credential_pool"`` token. A refresh/network error must
        # propagate — the outer ``fetch_account_usage`` guard fails open (shows nothing this turn) rather
        # than reporting the wrong account.
        creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        account_id: Optional[str] = None
        try:
            tokens = _read_codex_tokens().get("tokens") or {}
            account_id = str(tokens.get("account_id", "") or "").strip() or None
        except AuthError:
            # Pool-only creds carry no singleton account_id; header is optional.
            logger.debug("codex ▸ /usage account_id read failed (best-effort)", exc_info=True)
        return creds["api_key"], str(creds.get("base_url", "") or "").strip(), account_id
    except AuthError:
        logger.debug("codex ▸ /usage runtime resolver returned no creds; trying pool", exc_info=True)
    # Tier 3: pool credentials have no account_id concept → header omitted.
    from agent.credential_pool import load_pool
    entry = load_pool("openai-codex").select()
    if entry is None:
        raise RuntimeError("No available openai-codex credential in credential pool")
    return entry.runtime_api_key, str(entry.runtime_base_url or base_url or "").strip(), None


def _codex_banked_resets(payload: dict) -> int:
    raw = (payload.get("rate_limit_reset_credits") or {}).get("available_count")
    return int(raw) if _is_num(raw) else 0


def _codex_headers(token: str, account_id: Optional[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": "codex-cli",
            **({"ChatGPT-Account-Id": account_id} if account_id else {})}


def _get_json(url: str, headers: dict[str, str], *, timeout: float) -> dict:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
    return response.json() or {}


def _usage_windows(
    source: dict, mapping: tuple[tuple[str, str], ...], used_key: str, reset_key: str, *, fraction: bool = False
) -> list[AccountUsageWindow]:
    """Build windows from ``source[key][used_key]``; ``fraction`` scales values <= 1 to percent."""
    windows: list[AccountUsageWindow] = []
    for key, label in mapping:
        window = source.get(key) or {}
        used = window.get(used_key)
        if used is None:
            continue
        used = float(used)
        if fraction and used <= 1:
            used *= 100
        windows.append(AccountUsageWindow(label=label, used_percent=used, reset_at=_parse_dt(window.get(reset_key))))
    return windows


def _plural(count: int) -> str:
    return "s" if count != 1 else ""


def _fetch_codex_account_usage(
    base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    token, resolved_base_url, account_id = _resolve_codex_usage_credentials(base_url, api_key)
    payload = _get_json(_codex_backend_urls(resolved_base_url)[0], _codex_headers(token, account_id), timeout=15.0)
    windows = _usage_windows(payload.get("rate_limit") or {}, (("primary_window", "Session"), ("secondary_window", "Weekly")),
                             "used_percent", "reset_at")
    details: list[str] = []
    count = _codex_banked_resets(payload)
    if count > 0:
        details.append(f"You have {count} reset{_plural(count)} banked - use /usage reset to activate")
    credits, balance = payload.get("credits") or {}, (payload.get("credits") or {}).get("balance")
    if credits.get("has_credits") and _is_num(balance):
        details.append(f"Credits balance: ${float(balance):.2f}")
    elif credits.get("has_credits") and credits.get("unlimited"):
        details.append("Credits balance: unlimited")
    return _snapshot("openai-codex", "usage_api", windows, details, plan=_title_case_slug(payload.get("plan_type")))


@dataclass(frozen=True)
class CodexResetRedeemResult:
    """Outcome of a `/usage reset` attempt against the Codex backend."""

    status: str  # reset|nothing_to_reset|no_credit|already_redeemed|not_exhausted|no_credits_banked|unavailable
    message: str
    available_count: int = 0
    windows_reset: int = 0

    @property
    def redeemed(self) -> bool:
        return self.status == "reset"


# Client-side guard: a window only counts as exhausted when fully used; below this, redeeming a banked reset
# wastes most of its value → block, point at --force.
_CODEX_WINDOW_EXHAUSTED_PERCENT = 100.0


def _unavailable(message: str) -> CodexResetRedeemResult:
    return CodexResetRedeemResult(status="unavailable", message=message)


def _codex_reset_guard(payload: dict, available: int, force: bool) -> Optional[CodexResetRedeemResult]:
    """Refuse a redemption that would be wasted (no banked credits, or no window fully used and not ``force``)."""
    if available <= 0:
        return CodexResetRedeemResult(status="no_credits_banked", message="No banked reset credits on this account — nothing to redeem.")
    rate_limit = payload.get("rate_limit") or {}
    used_pcts = [float(u) for u in ((rate_limit.get(k) or {}).get("used_percent") for k in ("primary_window", "secondary_window"))
                 if _is_num(u)]
    worst_used: Optional[float] = max(0.0, *used_pcts) if used_pcts else None
    if force or (worst_used is not None and worst_used >= _CODEX_WINDOW_EXHAUSTED_PERCENT):
        return None
    usage_note = (f"your busiest window is only {worst_used:.0f}% used" if worst_used is not None
                  else "your current usage could not be confirmed as exhausted")
    return CodexResetRedeemResult(
        status="not_exhausted", available_count=available,
        message=(f"⚠️ Not redeeming: {usage_note}. A banked reset restores your FULL 5h + weekly limits, so spending it "
                 f"now would waste most of it. You have {available} reset{_plural(available)} banked. "
                 f"Use `/usage reset --force` to redeem anyway."),
    )


def _codex_reset_outcome(body: dict, available: int) -> CodexResetRedeemResult:
    """Map the consume response ``code`` to a result (``reset`` also lifts persisted pool cooldowns)."""
    code = str(body.get("code", "") or "").strip().lower()
    remaining = max(0, available - 1)
    outcomes: dict[str, tuple[str, int]] = {
        "reset": (f"✅ Reset redeemed — your usage limits have been reset. {remaining} banked reset{_plural(remaining)} remaining.",
                  remaining),
        "nothing_to_reset": ("Backend reports nothing to reset — your limits aren't exhausted. The credit was NOT spent.", available),
        "no_credit": ("Backend reports no available reset credit on this account.", 0),
        "already_redeemed": ("This redemption was already processed — no additional credit was spent.", remaining),
    }
    if code not in outcomes:
        return _unavailable(f"Unexpected response from the Codex backend: {body!r}")
    windows_reset = 0
    if code == "reset":
        # Quota is restored upstream — lift persisted pool cooldowns so the credential isn't frozen behind a
        # stale ``last_error_reset_at``.
        try:
            from hermes_cli.auth import clear_codex_pool_quota_cooldowns
            clear_codex_pool_quota_cooldowns()
        except Exception:
            logger.debug("Failed to clear Codex pool cooldowns after reset redemption", exc_info=True)
        raw = body.get("windows_reset")
        windows_reset = int(raw) if _is_num(raw) else 0
    message, count = outcomes[code]
    return CodexResetRedeemResult(status=code, message=message, available_count=count, windows_reset=windows_reset)


def redeem_codex_reset_credit(
    *, base_url: Optional[str] = None, api_key: Optional[str] = None, force: bool = False,
) -> CodexResetRedeemResult:
    """Redeem one banked Codex rate-limit reset credit (`/usage reset`), mirroring the Codex CLI picker: GET usage →
    guard (a reset restores the WHOLE 5h + weekly allowance, and the backend's own ``nothing_to_reset`` guard is
    less clear) → POST consume with a fresh UUID ``redeem_request_id`` and no ``credit_id`` (the backend picks the
    next credit). Never raises: every failure returns a result."""
    import uuid
    try:
        token, resolved_base_url, account_id = _resolve_codex_usage_credentials(base_url, api_key)
    except Exception:
        return _unavailable("No Codex credentials available. Run `hermes auth` to sign in with your ChatGPT account.")
    usage_url, _credits_url, consume_url = _codex_backend_urls(resolved_base_url)
    headers = _codex_headers(token, account_id)
    try:
        with httpx.Client(timeout=15.0) as client:
            usage_resp = client.get(usage_url, headers=headers)
            usage_resp.raise_for_status()
            payload = usage_resp.json() or {}
            available = _codex_banked_resets(payload)
            refused = _codex_reset_guard(payload, available, force)
            if refused is not None:
                return refused
            consume_resp = client.post(
                consume_url, headers={**headers, "Content-Type": "application/json"},
                json={"redeem_request_id": str(uuid.uuid4())},
            )
            consume_resp.raise_for_status()
            body = consume_resp.json() or {}
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            return _unavailable(f"Codex backend rejected the request (HTTP {code}). Reset credits require ChatGPT-account "
                                "(OAuth) auth — run `hermes auth` and sign in with your ChatGPT account.")
        return _unavailable(f"Codex backend error (HTTP {code}) — try again shortly.")
    except Exception as exc:
        return _unavailable(f"Could not reach the Codex backend: {exc}")
    return _codex_reset_outcome(body, available)


def _fetch_anthropic_account_usage(
    base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[AccountUsageSnapshot]:
    token = (resolve_anthropic_token() or "").strip()
    if not token:
        return None
    if not _is_oauth_token(token):
        return _snapshot("anthropic", "oauth_usage_api", [], [],
                         unavailable_reason="Anthropic account limits are only available for OAuth-backed Claude accounts.")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json",
               "anthropic-beta": "oauth-2025-04-20", "User-Agent": "claude-code/2.1.0"}
    payload = _get_json("https://api.anthropic.com/api/oauth/usage", headers, timeout=15.0)
    windows = _usage_windows(
        payload, (("five_hour", "Current session"), ("seven_day", "Current week"), ("seven_day_opus", "Opus week"),
                  ("seven_day_sonnet", "Sonnet week")), "utilization", "resets_at", fraction=True,
    )
    details: list[str] = []
    extra = payload.get("extra_usage") or {}
    used_credits, monthly_limit = extra.get("used_credits"), extra.get("monthly_limit")
    if extra.get("is_enabled") and _is_num(used_credits) and _is_num(monthly_limit):
        details.append(f"Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {extra.get('currency') or 'USD'}")
    return _snapshot("anthropic", "oauth_usage_api", windows, details)


def _fetch_openrouter_account_usage(base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    runtime = resolve_runtime_provider(requested="openrouter", explicit_base_url=base_url, explicit_api_key=api_key)
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None
    normalized = str(runtime.get("base_url", "") or "").rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    with httpx.Client(timeout=10.0) as client:
        def _data(path: str) -> dict:
            resp = client.get(f"{normalized}/{path}", headers=headers)
            resp.raise_for_status()
            return (resp.json() or {}).get("data") or {}
        credits = _data("credits")
        try:
            key_data = _data("key")
        except Exception:
            key_data = {}
    balance = float(credits.get("total_credits") or 0.0) - float(credits.get("total_usage") or 0.0)
    details = [f"Credits balance: ${max(0.0, balance):.2f}"]
    windows: list[AccountUsageWindow] = []
    limit, limit_remaining, usage = key_data.get("limit"), key_data.get("limit_remaining"), key_data.get("usage")
    limit_reset = str(key_data.get("limit_reset") or "").strip()
    if _is_num(limit) and float(limit) > 0 and _is_num(limit_remaining) and 0 <= float(limit_remaining) <= float(limit):
        limit_value, remaining_value = float(limit), float(limit_remaining)
        detail_parts = [f"${remaining_value:.2f} of ${limit_value:.2f} remaining", *([f"resets {limit_reset}"] if limit_reset else [])]
        windows.append(AccountUsageWindow(label="API key quota", used_percent=((limit_value - remaining_value) / limit_value) * 100,
                                          detail=" • ".join(detail_parts)))
    if _is_num(usage):
        usage_parts = [f"API key usage: ${float(usage):.2f} total"]
        for key, label in (("usage_daily", "today"), ("usage_weekly", "this week"), ("usage_monthly", "this month")):
            value = key_data.get(key)
            if _is_num(value) and float(value) > 0:
                usage_parts.append(f"${float(value):.2f} {label}")
        details.append(" • ".join(usage_parts))
    return _snapshot("openrouter", "credits_api", windows, details)


_USAGE_FETCHERS: dict[str, Callable[[Optional[str], Optional[str]], Optional[AccountUsageSnapshot]]] = {
    "openai-codex": _fetch_codex_account_usage, "anthropic": _fetch_anthropic_account_usage,
    "openrouter": _fetch_openrouter_account_usage,
}


def fetch_account_usage(
    provider: Optional[str], *, base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    fetcher = _USAGE_FETCHERS.get(str(provider or "").strip().lower())
    try:
        return fetcher(base_url, api_key) if fetcher else None
    except Exception:
        return None
