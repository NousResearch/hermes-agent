from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpx

from agent.anthropic_credentials import _is_oauth_token, resolve_anthropic_token
from hermes_cli.auth import AuthError, _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.auth_codex import _codex_pool_route_base_url
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_time import safe_strftime

from agent.usage_gauges import BP_FULL

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger(__name__)

_DEPLETED_LINE = "Status: access depleted — top up to restore"
# Usage-window plans: allowances are shown as percent used.
_WINDOW_TOP_TIER_USED_UP = "Monthly large-model usage is used up."
_WINDOW_PAUSED_LINE = ("Status: included usage paused. Your subscription payment failed. Update your card to restore it. "
                       "Until then, Hermes uses your credits.")
_WINDOW_PAUSED_OTHER_LINE = "Status: included usage paused. See Billing in the Portal. Until then, Hermes uses your credits."


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class AccountUsageWindow:
    label: str
    used_percent: Optional[float] = None
    reset_at: Optional[datetime] = None
    detail: Optional[str] = None
    # ``account``: exhausting this window exhausts the whole login (Codex session/weekly).
    # ``model``: the window caps only one model family (Anthropic Opus/Sonnet weekly) and can
    # never imply the account itself is out of quota.
    scope: str = "account"


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
    # Stable account identity of the credential the snapshot was fetched with (e.g. a decoded
    # Codex JWT principal), when the provider supports one. Never a secret; ``None`` = the
    # fetcher cannot tell accounts apart, so the snapshot belongs to the provider's legacy slot.
    identity: Optional[str] = None
    # Exact decoded provider response body (no headers/credentials) for integrations that need
    # fields Hermes does not normalize yet. Only populated by providers that fetch a JSON body.
    raw: Optional[dict] = None
    # Usage-window plans only: the stacked bar layout. Every other snapshot
    # renders exactly as before.
    usage_window: bool = False
    # Account namespace (``agent.account_usage_cache._identity_id_for``) of the
    # credential this snapshot was actually fetched with, when core selected one
    # by identity: a decoded Codex principal, the cache's existing one-way token
    # fingerprint, or ``entry:<pool id>``. This is the exact id the delegation
    # roster publishes as ``provider_account_id`` and the inventory publishes as
    # ``identity_id``; never the token itself. ``None`` means core could not
    # prove which credential was selected (for example an env/config key);
    # callers must render that account as unknown rather than guess.
    identity_id: Optional[str] = None
    # Stable account id when the provider exposes one (today the decoded Codex
    # JWT principal, ``codex:<account>:<sub>``). Never a token.
    account_id: Optional[str] = None
    # Canonical, non-path profile scope (``home:<16 hex>``) so two Hermes
    # profiles' identical provider rows can never be correlated. The hash is
    # over the canonical home path, not a credential.
    profile_scope: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.windows or self.details) and not self.unavailable_reason


def _snapshot(provider: str, source: str, windows: list, details: list, **kw: Any) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(provider=provider, source=source, fetched_at=_utc_now(), windows=tuple(windows), details=tuple(details), **kw)


def _profile_scope() -> Optional[str]:
    """Canonical, non-secret profile scope string (``home:<16 hex>``).

    Same value the Desktop Provider Usage plugin derives for its own home
    scope: a hash of the canonical Hermes home path, never the path itself and
    never a credential. Two profiles therefore produce different scopes even
    when they select the same provider/account ids.
    """
    try:
        import hashlib

        from hermes_constants import get_hermes_home, hermes_home_key

        key = hermes_home_key(get_hermes_home())
        return "home:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


# Credential-shaped keys are dropped from ``snapshot.raw`` recursively. Provider
# usage bodies never need them, and a custom gateway body must not smuggle a
# token into a plugin pane that persists rows.
_RAW_DROP_KEYS = frozenset({
    "authorization", "apikey", "accesstoken", "refreshtoken", "agentkey",
    "clientsecret", "secret", "password", "cookie", "setcookie", "headers",
    "credential", "credentials", "authtoken", "xapikey",
})


def _sanitized_raw(value: Any) -> Any:
    """Decoded provider JSON with credential-shaped keys removed, recursively."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("_", "").replace("-", "")
            if normalized in _RAW_DROP_KEYS:
                continue
            cleaned[key] = _sanitized_raw(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitized_raw(item) for item in value]
    return value


def _redacted_text(value: Any, token: str = "") -> str:
    text = str(value or "")
    secret = str(token or "").strip()
    return text.replace(secret, "redacted") if secret and secret in text else text


def _safe_base_url(value: Any, *, token: str = "") -> Optional[str]:
    """A pool entry's base URL without userinfo, query, fragment or the token.

    A custom gateway may carry ``user:password@`` or a key in a query/path; the
    inventory is non-secret, so only scheme, host, port and path survive.
    """
    from urllib.parse import urlsplit, urlunsplit

    text = _redacted_text(value, token).strip()
    if not text:
        return None
    try:
        parts = urlsplit(text)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not host:
        return None
    netloc = f"[{host}]" if ":" in host and not host.startswith("[") else host
    if port:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _credential_identity(provider: str, entry: Any) -> Optional[str]:
    """The account namespace the cache/metadata footer already uses (``_identity_id_for``).

    Codex resolves to the trusted decoded principal (``codex:<account>:<sub>``);
    other credentials use the cache's existing ``fp:<16 hex>`` one-way token
    fingerprint, and a metadata-only row uses ``entry:<pool id>``. This is the
    exact id the delegation roster publishes as ``provider_account_id``; it is
    a correlation namespace, never the token, and is not reversible. The pool
    row's own stable id is published separately as ``credential_id``.
    """
    if entry is None:
        return None
    try:
        from agent.account_usage_cache import _identity_id_for

        value = str(_identity_id_for(provider, entry) or "").strip()
    except Exception:
        entry_id = str(getattr(entry, "id", "") or "").strip()
        return f"entry:{entry_id}" if entry_id else None
    if not value or value == "entry:":
        return None
    return value


def _credential_id(entry: Any) -> Optional[str]:
    """The pool row's existing stable id (non-secret), independent of token material."""
    value = str(getattr(entry, "id", "") or "").strip()
    return value or None


def _credential_account_id(provider: str, entry: Any) -> Optional[str]:
    """Stable non-secret account id for providers that expose one (Codex JWT)."""
    if str(provider or "").strip().lower() != "openai-codex" or entry is None:
        return None
    try:
        from agent.credential_pool import _codex_principal_identity

        token = str(getattr(entry, "runtime_api_key", "") or "").strip()
        principal = _codex_principal_identity(token)
        return f"codex:{principal[0]}:{principal[1]}" if principal else None
    except Exception:
        return None


def _title_case_slug(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    return cleaned.replace("_", " ").replace("-", " ").title() if cleaned else None


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if not isinstance(value, str) or not (text := value.strip()):
        return None
    text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


def _format_reset(dt: Optional[datetime]) -> str:
    if not dt:
        return "unknown"
    stamp = safe_strftime(dt.astimezone(), "%Y-%m-%d %H:%M %Z")
    total_seconds = int((dt - _utc_now()).total_seconds())
    if total_seconds <= 0:
        return f"now ({stamp})"
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"in {days}d {hours}h ({stamp})"
    return f"in {hours}h {minutes}m ({stamp})" if hours else f"in {minutes}m ({stamp})"


def _reset_phrase(dt: Optional[datetime]) -> str:
    """``resets in 23h`` — no wall-clock stamp (those wrap and collide in the TUI)."""
    if not dt:
        return "resets soon"
    try:
        from hermes_cli.model_picker_usage import format_resets_in
        return format_resets_in(int(dt.timestamp() * 1000))
    except Exception:
        logger.debug("Reset phrase formatting failed", exc_info=True)
        return f"resets {_format_reset(dt)}"


def render_account_usage_lines(snapshot: Optional[AccountUsageSnapshot], *, markdown: bool = False) -> list[str]:
    if not snapshot:
        return []
    if snapshot.usage_window:
        return _render_usage_window_lines(snapshot)
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


def _render_usage_window_lines(snapshot: AccountUsageSnapshot) -> list[str]:
    """Usage-window plans: plain stacked bar lines for CLI ``/usage`` and the TUI panel. No emoji,
    no ANSI, no markdown — Ink ``<Text>`` would otherwise count escape bytes as width and smear the wrap."""
    if snapshot.plan:
        lines = [snapshot.plan]
    else:
        lines = [snapshot.title, f"Provider: {snapshot.provider}"]
    label_w = max((len(window.label) for window in snapshot.windows), default=0)
    for window in snapshot.windows:
        label = f"{window.label:<{label_w}}" if label_w else window.label
        if window.used_percent is None:
            base = f"{label}  unavailable"
        else:
            used = max(0.0, min(100.0, float(window.used_percent))) / 100.0
            try:
                from hermes_cli.model_picker_usage import format_percent_used, render_bar
                bar, pct = render_bar(used), format_percent_used(used)
            except Exception:
                logger.debug("Usage bar rendering failed", exc_info=True)
                bar, pct = "", f"{int(used * 100)}%"
            base = f"{label}  {bar}  {pct} used" if bar else f"{label}  {pct} used"
        if window.reset_at:
            base += f"  ·  {_reset_phrase(window.reset_at)}"
        elif window.detail:
            base += f"  ·  {window.detail}"
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


def _nous_snapshot(windows: list, details: list, tail: list, *, source: str, plan: Optional[str] = None,
                   usage_window: bool = False) -> Optional[AccountUsageSnapshot]:
    """Nous snapshot with *tail* lines appended, or None when there is nothing to show."""
    if not windows and not details:
        return None
    return _snapshot("nous", source, windows, details + tail, title="Nous credits", plan=plan, usage_window=usage_window)


def _bp_window(label: str, used_bp: Any, resets_at_ms: Any) -> Optional[AccountUsageWindow]:
    """One usage-window gauge row from basis points + an epoch-ms reset (already bounded by
    ``usage_gauges.parse_gauges``); None when the percent is not an int. An unknown reset keeps the row,
    shown without a reset."""
    if isinstance(used_bp, bool) or not isinstance(used_bp, int):
        return None
    known = isinstance(resets_at_ms, int) and not isinstance(resets_at_ms, bool)
    reset_at = datetime.fromtimestamp(resets_at_ms / 1000, tz=UTC) if known else None
    return AccountUsageWindow(label=label, used_percent=max(0.0, min(100.0, used_bp / BP_FULL * 100.0)),
                              reset_at=reset_at)


def _window_top_tier_line(reset_at: Optional[datetime], keep_going_on_credits: Any = None) -> str:
    """``Monthly large-model usage is used up. Resets Nov 3. Past this, large models use your credits.`` (local
    date). With Keep going on credits off, credits don't unlock it, so the line points at the other included models."""
    resets = ""
    if isinstance(reset_at, datetime):
        local = reset_at.astimezone()
        resets = f" Resets {local:%b} {local.day}."
    tail = " Your other included usage is still available." if keep_going_on_credits is False else " Past this, large models use your credits."
    return f"{_WINDOW_TOP_TIER_USED_UP}{resets}{tail}"


_WINDOW_KEEP_GOING_OFF_LINE = "Keep going on credits: off. Hermes waits for the reset (turn on in /topup)."
_WINDOW_KEEP_GOING_ON_LINE = "Keep going on credits: on. Uses your credits when included usage runs out."


def _usage_window_rows(gauges: Any, *, paused_reason: Any,
                       keep_going_on_credits: Any = None) -> tuple[list[AccountUsageWindow], list[str]]:
    """One row per server allowance, in the server's order, plus the top-tier, payment-paused and top-up-off
    lines. Empty unless the list has an allowance that stops every model. ``keep_going_on_credits``: the
    account setting (None: an older server)."""
    from agent.usage_gauges import SCOPE_TOP_TIER, all_scope, gauge_title, parse_gauges
    parsed = parse_gauges(gauges)
    if not parsed or not all_scope(parsed):
        return [], []
    out: list[AccountUsageWindow] = []
    top_row: Optional[AccountUsageWindow] = None
    top_full = False
    for g in parsed:
        row = _bp_window(gauge_title(g), g.used_bp, g.resets_at_ms)
        if row is None:
            continue
        out.append(row)
        if g.scope == SCOPE_TOP_TIER:
            top_row, top_full = row, g.full
    lines: list[str] = []
    if top_full:
        lines.append(_window_top_tier_line(top_row.reset_at if top_row else None, keep_going_on_credits))
    if isinstance(paused_reason, str) and paused_reason:
        from agent.usage_gauges import paused_for_payment
        lines.append(_WINDOW_PAUSED_LINE if paused_for_payment(paused_reason) else _WINDOW_PAUSED_OTHER_LINE)
    elif keep_going_on_credits is False:
        lines.append(_WINDOW_KEEP_GOING_OFF_LINE)
    elif keep_going_on_credits is True:
        lines.append(_WINDOW_KEEP_GOING_ON_LINE)
    return out, lines


def build_nous_credits_snapshot(account_info) -> Optional[AccountUsageSnapshot]:
    """NousPortalAccountInfo → /usage snapshot: dollar magnitudes + renewal date + portal CTA, plus a ``% used``
    gauge when the portal supplies ``monthly_credits``. A usage-window account (``usage_window`` present) shows
    its allowances as percent used, the large-model line and its top-up balance instead. Fail-open → None."""
    try:
        from hermes_cli.nous_account import nous_portal_topup_url
        if account_info is None or not getattr(account_info, "logged_in", False):
            return None
        access = getattr(account_info, "paid_service_access_info", None)
        sub = getattr(account_info, "subscription", None)
        windows: list[AccountUsageWindow] = []
        details: list[str] = []
        uw = getattr(account_info, "usage_window", None)
        if uw is not None:
            windows, window_lines = _usage_window_rows(
                getattr(uw, "gauges", None), paused_reason=getattr(uw, "paused_reason", None),
                keep_going_on_credits=getattr(uw, "keep_going_on_credits", None),
            )
        if windows:  # usage-window account: percents, plus the top-up balance
            details.extend(window_lines)
            topup = getattr(access, "purchased_credits_remaining", None) if access is not None else None
            if _is_finite_num(topup):
                details.append(f"Top-up credits: {_fmt_usd(topup)}")
            if getattr(account_info, "paid_service_access", None) is False and not window_lines:
                details.append(_DEPLETED_LINE)
            if sub is not None and (period_end := getattr(sub, "current_period_end", None)):
                from agent.billing_usage import format_renews
                details.append(f"Renews: {format_renews(period_end) or period_end}")
            return _nous_snapshot(windows, details, ["Top up via /topup"], source="portal-account",
                                  plan=getattr(sub, "plan", None) if sub is not None else None, usage_window=True)
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
    """Wall-clock-bounded fresh portal account fetch (raises on any failure/timeout).

    No ``with`` block on purpose: ``Executor.__exit__`` joins the worker via
    ``shutdown(wait=True)``, so a portal that accepts the connection but never
    answers would hold the caller until the provider's own timeout instead of
    ``timeout``. The abandoned daemon worker runs on to its own network timeout
    and never blocks the caller or process exit; its eventual exception is
    drained so GC never logs "exception was never retrieved"."""
    import contextvars
    from hermes_cli.nous_account import get_nous_portal_account_info
    from tools.daemon_pool import DaemonThreadPoolExecutor

    context = contextvars.copy_context()
    pool = DaemonThreadPoolExecutor(max_workers=1)
    future = pool.submit(context.run, get_nous_portal_account_info, force_fresh=True)
    try:
        return future.result(timeout=timeout)
    except BaseException:
        future.add_done_callback(lambda f: f.exception())
        raise
    finally:
        pool.shutdown(wait=False)


def _header_gauge_percent(bp: Any) -> Optional[float]:
    if isinstance(bp, bool) or not isinstance(bp, int):
        return None
    return max(0.0, min(100.0, bp / BP_FULL * 100.0))


def apply_header_gauge(snapshot: Optional[AccountUsageSnapshot], credits_state: Any) -> Optional[AccountUsageSnapshot]:
    """Raise window percents to the last response headers when those are ahead of the portal.

    The portal counter waits on spend-log ingestion. The headers already include finished included
    calls, and a large-model call that drew top-up reports that allowance full."""
    if snapshot is None or not snapshot.usage_window or credits_state is None:
        return snapshot
    if not getattr(credits_state, "from_header", False) or not getattr(credits_state, "is_usage_window", False):
        return snapshot
    # Row title → (header used bp, that gauge's reset). A header from before its gauge reset describes the
    # old window: the portal's fresh number stands.
    from agent.usage_gauges import gauge_title
    mapping = {gauge_title(g): (g.used_bp, g.resets_at_ms) for g in getattr(credits_state, "gauges", ()) or ()}
    now_ms = int(time.time() * 1000)
    windows: list[AccountUsageWindow] = []
    changed = False
    for window in snapshot.windows:
        bp, resets_at_ms = mapping.get(window.label, (None, None))
        stale = isinstance(resets_at_ms, int) and not isinstance(resets_at_ms, bool) and now_ms >= resets_at_ms
        header = None if stale else _header_gauge_percent(bp)
        current = window.used_percent
        if header is None or current is None or header <= float(current):
            windows.append(window)
            continue
        changed = True
        windows.append(replace(window, used_percent=header))
    if not changed:
        return snapshot
    details = snapshot.details
    top_gauge = getattr(credits_state, "top_tier_gauge", None)
    top = next((w for w in windows if top_gauge is not None and w.label == gauge_title(top_gauge)), None)
    if top is not None and (top.used_percent or 0) >= 100:
        if not any(line.startswith(_WINDOW_TOP_TIER_USED_UP) for line in details):
            details = details + (_window_top_tier_line(top.reset_at, getattr(credits_state, "keep_going_on_credits", None)),)
    return replace(snapshot, windows=tuple(windows), details=details)


def nous_credits_lines(*, markdown: bool = False, timeout: float = 10.0, credits_state: Any = None) -> list[str]:
    """Rendered Nous-credits /usage lines, or [] when there's nothing to show. Independent of any live agent
    (logged-in gate, then a bounded portal fetch); shared by CLI ``_show_usage`` and the TUI ``session.usage`` RPC.
    Fail-open: any hiccup or timeout → []. HERMES_DEV_CREDITS_FIXTURE renders from the fixture instead of the portal.
    ``credits_state`` is the last response's header gauge; when it is ahead of the portal, the bars use it."""
    return nous_credits_view(markdown=markdown, timeout=timeout, credits_state=credits_state)[0]


def nous_credits_view(*, markdown: bool = False, timeout: float = 10.0, credits_state: Any = None) -> tuple[list[str], bool]:
    """``nous_credits_lines`` plus whether they are the usage-window layout (callers keep today's
    presentation for everyone else). Fail-open → ``([], False)``."""
    try:
        from agent.credits_tracker import dev_fixture_credits_state
        fixture = dev_fixture_credits_state()
    except Exception:
        logger.debug("Dev credits fixture unavailable", exc_info=True)
        fixture = None
    if fixture is not None:
        snapshot = _snapshot_from_credits_state(fixture)
        return render_account_usage_lines(snapshot, markdown=markdown), bool(snapshot and snapshot.usage_window)
    if not _nous_logged_in():
        return [], False
    try:
        snapshot = apply_header_gauge(
            build_nous_credits_snapshot(_fetch_portal_account(timeout)), credits_state,
        )
        return render_account_usage_lines(snapshot, markdown=markdown), bool(snapshot and snapshot.usage_window)
    except Exception:
        # Fail-open; breadcrumb so a dead /usage credits block is diagnosable.
        logger.debug("credits ▸ /usage portal fetch/render failed (fail-open)", exc_info=True)
        return [], False


def nous_credits_usage_fields(agent: Any = None) -> dict[str, Any]:
    """The ``session.usage`` fields for the Nous credits block: ``credits_lines`` (when there are any) and
    ``credits_usage_window`` for the usage-window layout. Reads *agent*'s live credits headers. Never raises."""
    try:
        lines, is_window = nous_credits_view(credits_state=getattr(agent, "_credits_state", None) if agent else None)
    except Exception:
        logger.debug("credits ▸ /usage fields unavailable", exc_info=True)
        return {}
    return ({"credits_lines": lines} | ({"credits_usage_window": True} if is_window else {})) if lines else {}


def _snapshot_from_credits_state(state) -> Optional[AccountUsageSnapshot]:
    """Header-shaped CreditsState (dev fixture) → /usage snapshot, same shape as the portal path. *_usd strings
    are display-only; the % comes from CreditsState.used_fraction. Fail-open → None."""
    try:
        if state is None:
            return None
        windows: list[AccountUsageWindow] = []
        details: list[str] = []
        if getattr(state, "is_usage_window", False):
            windows, window_lines = _usage_window_rows(
                state.gauges, paused_reason=state.paused_reason,
                keep_going_on_credits=getattr(state, "keep_going_on_credits", None),
            )
            details.extend(window_lines)
            if getattr(state, "purchased_usd", None):
                details.append(f"Top-up credits: ${state.purchased_usd}")
            if getattr(state, "paid_access", True) is False and not window_lines:
                details.append(_DEPLETED_LINE)
            return _nous_snapshot(windows, details, ["(dev fixture — HERMES_DEV_CREDITS_FIXTURE)"], source="dev-fixture",
                                  usage_window=True)
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
        if not line.lstrip().startswith(("Top up:", "Top up via", "(or run"))
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
    base_url: Optional[str], api_key: Optional[str], *, force_refresh: bool = False,
) -> tuple[str, str, Optional[str]]:
    """Codex quota credentials: explicit live-agent creds → native runtime resolver (itself pool-aware) → direct
    pool select. Native OAuth stores device-code logins in the pool, so the singleton store alone is not enough."""
    explicit_key = str(api_key or "").strip()
    if explicit_key and not force_refresh:
        return explicit_key, str(base_url or "").strip(), None
    if explicit_key:
        # Forced retry for a live agent's own credential: refresh THAT credential (singleton or the
        # pool entry that issued it), never re-resolve — that would render another pool account's usage.
        try:
            singleton_key = str((_read_codex_tokens().get("tokens") or {}).get("access_token", "") or "").strip()
        except AuthError:
            singleton_key = ""
        if singleton_key != explicit_key:
            from agent.credential_pool import load_pool
            entry = load_pool("openai-codex").try_refresh_matching(api_key_hint=explicit_key)
            if entry is None:
                raise RuntimeError("Could not refresh the Codex credential this session runs on")
            return entry.runtime_api_key, _codex_pool_route_base_url(entry.runtime_base_url or base_url), None
    # Only AuthError is caught so tier 3 can run: a broad except would mask a transient refresh/network failure
    # and hand back a DIFFERENT pool account's usage; such errors must propagate to the fail-open outer guard.
    # account_id is best-effort: a partial singleton store must not sink a usable credential.
    try:
        # Tier 2: the native runtime resolver. It ALREADY falls back to the credential pool when the
        # singleton is empty (see ``resolve_codex_runtime_credentials`` — issue #32992), so in a pool-only
        # setup this returns a usable ``source="credential_pool"`` token. A refresh/network error must
        # propagate — the outer ``fetch_account_usage`` guard fails open (shows nothing this turn) rather
        # than reporting the wrong account.
        resolve_kwargs = {"refresh_if_expiring": True}
        if force_refresh:
            resolve_kwargs["force_refresh"] = True
        creds = resolve_codex_runtime_credentials(**resolve_kwargs)
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
    # Pool rows keep the canonical URL; a gateway key must go to its route host, not chatgpt.com (#121486).
    return entry.runtime_api_key, _codex_pool_route_base_url(entry.runtime_base_url or base_url), None


def _codex_banked_resets(payload: dict) -> int:
    raw = (payload.get("rate_limit_reset_credits") or {}).get("available_count")
    return int(raw) if _is_num(raw) else 0


def _codex_headers(token: str, account_id: Optional[str]) -> dict[str, str]:
    """auth.json's ``account_id`` wins over the JWT claim; the JWT still supplies the residency header."""
    from agent.codex_headers import codex_account_headers
    return {"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": "codex-cli",
            **codex_account_headers(token), **({"ChatGPT-Account-ID": account_id} if account_id else {})}


def _get_json(url: str, headers: dict[str, str], *, timeout: float) -> dict:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
    return response.json() or {}


def _usage_windows(
    source: dict, mapping: tuple[tuple[str, str], ...], used_key: str, reset_key: str, *, fraction: bool = False,
    model_scoped: frozenset[str] | set[str] = frozenset(),
) -> list[AccountUsageWindow]:
    """Build windows from ``source[key][used_key]``; ``fraction`` scales values <= 1 to percent.
    ``model_scoped`` keys build windows that cap only a model family, never the account."""
    windows: list[AccountUsageWindow] = []
    for key, label in mapping:
        window = source.get(key) or {}
        used = window.get(used_key)
        if used is None:
            continue
        used = float(used)
        if fraction and used <= 1:
            used *= 100
        windows.append(AccountUsageWindow(
            label=label, used_percent=used, reset_at=_parse_dt(window.get(reset_key)),
            scope="model" if key in model_scoped else "account",
        ))
    return windows


# Published Codex quota windows by ``limit_window_seconds``: 5h session and 7-day weekly.
_CODEX_WINDOW_LABELS_BY_SECONDS = {18000: "Session", 604800: "Weekly"}
_CODEX_WINDOW_POSITIONAL_LABELS = (("primary_window", "Session"), ("secondary_window", "Weekly"))


def _codex_window_labels(rate_limit: dict) -> tuple[tuple[str, str], ...]:
    """Label Codex windows by their published duration, not response position (#65387).

    The usage API keys windows ``primary_window``/``secondary_window`` by position; when only the
    weekly limit is returned it occupies ``primary_window`` and the positional mapping mislabeled it
    ``Session``. Windows whose ``limit_window_seconds`` is missing or unrecognized keep the legacy
    positional label so duration-less payloads render exactly as before.
    """
    labels = []
    for key, fallback in _CODEX_WINDOW_POSITIONAL_LABELS:
        window = rate_limit.get(key) or {}
        seconds = window.get("limit_window_seconds") if isinstance(window, dict) else None
        label = _CODEX_WINDOW_LABELS_BY_SECONDS.get(int(seconds), fallback) if _is_num(seconds) else fallback
        labels.append((key, label))
    return tuple(labels)


def _plural(count: int) -> str:
    return "s" if count != 1 else ""


def _fetch_codex_account_usage(
    base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    return _fetch_codex_account_usage_impl(base_url, api_key, read_only=False)


def _fetch_codex_account_usage_read_only(
    base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    """Picker-cache variant: a 401 is reported, never repaired. The live-agent path
    (``_fetch_codex_account_usage``) retries via a forced credential refresh because the
    session it serves owns that credential; a per-credential read of the pool must NOT
    rotate, refresh or mutate anything — it just marks that account unknown."""
    return _fetch_codex_account_usage_impl(base_url, api_key, read_only=True)


def _fetch_codex_account_usage_impl(
    base_url: Optional[str] = None, api_key: Optional[str] = None, *, read_only: bool = False,
) -> Optional[AccountUsageSnapshot]:
    token, resolved_base_url, account_id = _resolve_codex_usage_credentials(base_url, api_key)
    try:
        payload = _get_json(
            _codex_backend_urls(resolved_base_url)[0], _codex_headers(token, account_id), timeout=15.0,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 401:
            raise
        if read_only:
            # Never refresh/rotate on the picker's read-only path — surface the failure.
            raise
        token, resolved_base_url, account_id = _resolve_codex_usage_credentials(
            base_url, api_key, force_refresh=True,
        )
        payload = _get_json(
            _codex_backend_urls(resolved_base_url)[0], _codex_headers(token, account_id), timeout=15.0,
        )
    rate_limit = payload.get("rate_limit") or {}
    windows = _usage_windows(rate_limit, _codex_window_labels(rate_limit), "used_percent", "reset_at")
    details: list[str] = []
    count = _codex_banked_resets(payload)
    if count > 0:
        details.append(f"You have {count} reset{_plural(count)} banked - use /usage reset to activate")
    credits, balance = payload.get("credits") or {}, (payload.get("credits") or {}).get("balance")
    if credits.get("has_credits") and _is_num(balance):
        details.append(f"Credits balance: ${float(balance):.2f}")
    elif credits.get("has_credits") and credits.get("unlimited"):
        details.append("Credits balance: unlimited")
    identity = _codex_snapshot_identity(token)
    return _snapshot("openai-codex", "usage_api", windows, details, plan=_title_case_slug(payload.get("plan_type")),
                     identity=identity, account_id=identity,
                     raw=_sanitized_raw(payload), profile_scope=_profile_scope())


def _codex_snapshot_identity(token: str) -> Optional[str]:
    """Trusted per-account identity of a Codex credential: the decoded JWT principal
    (``chatgpt_account_id`` + ``sub``), never the token itself. Two credentials for one
    workspace member share it, so pool accounts are counted once, honestly."""
    from agent.credential_pool import _codex_principal_identity

    principal = _codex_principal_identity(token)
    return f"codex:{principal[0]}:{principal[1]}" if principal else None


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
    redeem_request_id = str(uuid.uuid4())
    try:
        for attempt in range(2):
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
                        json={"redeem_request_id": redeem_request_id},
                    )
                    consume_resp.raise_for_status()
                    body = consume_resp.json() or {}
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 401 or attempt > 0:
                    raise
                try:
                    token, resolved_base_url, account_id = _resolve_codex_usage_credentials(
                        base_url, api_key, force_refresh=True,
                    )
                except Exception:
                    # Refresh token dead too: the 401 hint (re-login) is the actionable message.
                    raise exc from None
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
    base_url: Optional[str] = None, api_key: Optional[str] = None, *, read_only: bool = False
) -> Optional[AccountUsageSnapshot]:
    # An explicit api_key (live agent / per-credential pool read) must not be shadowed by
    # the ambient OAuth singleton: fetch_account_usage(base_url, api_key) promises to
    # report THAT credential's usage. An API key answers the OAuth usage endpoint with
    # 401, so only an OAuth-shaped token reaches the fetch.
    explicit = str(api_key or "").strip()
    token = explicit or (resolve_anthropic_token(refresh=not read_only) or "").strip()
    if not token:
        return None
    if not _is_oauth_token(token):
        return _snapshot("anthropic", "oauth_usage_api", [], [],
                         unavailable_reason="Anthropic account limits are only available for OAuth-backed Claude accounts.",
                         profile_scope=_profile_scope())
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json",
               "anthropic-beta": "oauth-2025-04-20", "User-Agent": "claude-code/2.1.0"}
    payload = _get_json("https://api.anthropic.com/api/oauth/usage", headers, timeout=15.0)
    windows = _usage_windows(
        payload, (("five_hour", "Current session"), ("seven_day", "Current week"), ("seven_day_opus", "Opus week"),
                  ("seven_day_sonnet", "Sonnet week")), "utilization", "resets_at", fraction=True,
        model_scoped={"seven_day_opus", "seven_day_sonnet"},
    )
    details: list[str] = []
    extra = payload.get("extra_usage") or {}
    used_credits, monthly_limit = extra.get("used_credits"), extra.get("monthly_limit")
    if extra.get("is_enabled") and _is_num(used_credits) and _is_num(monthly_limit):
        details.append(f"Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {extra.get('currency') or 'USD'}")
    return _snapshot("anthropic", "oauth_usage_api", windows, details,
                     raw=_sanitized_raw(payload), profile_scope=_profile_scope())


def _fetch_anthropic_account_usage_read_only(
    base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[AccountUsageSnapshot]:
    """Read-only picker variant: never refresh a Claude Code grant on the way to a read."""
    return _fetch_anthropic_account_usage(base_url, api_key, read_only=True)


def _fetch_openrouter_account_usage(base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    token, resolved_base = _usage_credentials("openrouter", base_url, api_key)
    if not token:
        return None
    normalized = (resolved_base or "https://openrouter.ai/api/v1").rstrip("/")
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
    # Exact decoded bodies, no headers or credentials: the plugin pane keeps
    # OpenRouter's free-model daily quota, per-period key spend and BYOK total,
    # none of which the normalized details above can reconstruct.
    return _snapshot("openrouter", "credits_api", windows, details,
                     raw=_sanitized_raw({"credits": credits, "key": key_data}),
                     profile_scope=_profile_scope())


def _fetch_openrouter_account_usage_read_only(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    """Read-only picker variant: this fetcher never rotates or refreshes a credential."""
    return _fetch_openrouter_account_usage(base_url, api_key)


def _usage_credentials(provider: str, base_url: Optional[str], api_key: Optional[str]) -> tuple[str, str]:
    """``(token, base_url)`` for a usage read, resolved inside core.

    An explicit credential pair (the per-identity path, which core resolved
    from the pool itself) is used as-is. Otherwise the same profile-aware
    runtime resolver inference uses supplies env, config, OAuth or a pooled
    credential, so a usage read never collapses to env-only. No ``force_refresh``
    is ever requested: a healthy token is read, not rotated.
    """
    explicit_key = str(api_key or "").strip()
    explicit_base = str(base_url or "").strip()
    if explicit_key and explicit_base:
        return explicit_key, explicit_base
    runtime = resolve_runtime_provider(
        requested=provider,
        explicit_base_url=explicit_base or None,
        explicit_api_key=explicit_key or None,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    resolved_base = str(runtime.get("base_url", "") or explicit_base or "").strip()
    return token, resolved_base


def _json_object(response: Any) -> dict:
    payload = response.json() or {}
    return payload if isinstance(payload, dict) else {}


def _deepseek_root(base_url: str) -> str:
    value = (base_url or "https://api.deepseek.com/v1").strip().rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3]
    return value.rstrip("/")


def _fetch_deepseek_account_usage(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    """DeepSeek ``/user/balance``; every ``balance_infos`` row stays on ``raw``."""
    token, resolved_base = _usage_credentials("deepseek", base_url, api_key)
    if not token:
        return None
    payload = _get_json(
        f"{_deepseek_root(resolved_base)}/user/balance",
        {"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=12.0,
    )
    infos = payload.get("balance_infos") or payload.get("balances") or []
    if isinstance(infos, dict):
        infos = [infos]
    if not infos and payload.get("currency"):
        infos = [payload]
    details: list[str] = []
    has_balance = False
    for info in infos if isinstance(infos, list) else []:
        if not isinstance(info, dict):
            continue
        # The exact per-currency rows still ride on raw so the pane keeps
        # granted/topped-up components and the provider's own currency; this
        # line only keeps core's own /usage surface useful.
        total = info.get("total_balance")
        if total is None:
            continue
        has_balance = True
        details.append(f"{str(info.get('currency') or 'USD').upper()} balance: {total}")
    if payload.get("is_available") is False:
        details.insert(0, "Account available: no")
    return _snapshot("deepseek", "user_balance_api", [], details, title="DeepSeek balance",
                     unavailable_reason=None if has_balance else "DeepSeek returned no balance entries.",
                     raw=_sanitized_raw({"balance": payload}), profile_scope=_profile_scope())


def _fetch_deepseek_account_usage_read_only(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    return _fetch_deepseek_account_usage(base_url, api_key)


def _ollama_cloud_usage_windows(payload: dict) -> tuple[list[AccountUsageWindow], dict]:
    raw_limits = payload.get("limits")
    limits: dict = raw_limits if isinstance(raw_limits, dict) else {}
    windows: list[AccountUsageWindow] = []
    for key, label in (("session", "5-hour"), ("weekly", "Weekly"), ("monthly", "Monthly")):
        entry = limits.get(key)
        if not isinstance(entry, dict):
            continue
        fraction = entry.get("usage")
        if not _is_num(fraction):
            continue
        used = float(fraction) * 100.0 if 0.0 <= float(fraction) <= 1.0 else float(fraction)
        windows.append(AccountUsageWindow(
            label=label, used_percent=round(max(0.0, min(100.0, used)), 2)))
    return windows, limits


def _ollama_cloud_details(payload: dict, limits: dict) -> list[str]:
    details: list[str] = []
    models = None
    for key in ("monthly", "weekly", "session"):
        entry = limits.get(key)
        if isinstance(entry, dict) and entry.get("models"):
            models = entry.get("models")
            break
    if isinstance(models, dict):
        entries = [
            dict(value, name=value.get("name") or name) if isinstance(value, dict)
            else {"name": name, "request_count": value}
            for name, value in models.items()
        ]
    elif isinstance(models, list):
        entries = [item for item in models if isinstance(item, dict)]
    else:
        entries = []
    for item in entries[:6]:
        name = str(item.get("name") or "").strip()
        count = item.get("request_count")
        if name and _is_num(count):
            count_int = int(count)
            details.append(f"{name}: {count_int} request{'s' if count_int != 1 else ''}")
    raw_activity = payload.get("activity")
    activity: dict = raw_activity if isinstance(raw_activity, dict) else {}
    cost = str(activity.get("cost") or "").strip()
    if cost:
        try:
            cost_text = f"${float(cost):.2f}"
        except ValueError:
            cost_text = cost
        raw_period = activity.get("period")
        period: dict = raw_period if isinstance(raw_period, dict) else {}
        period_text = str(period.get("type") or "").replace("_", " ").strip()
        details.append(f"Spend in the {period_text or 'recent period'}: {cost_text}")
    return details


def _ollama_cloud_credit_window(balance: dict) -> Optional[AccountUsageWindow]:
    included = balance.get("included") if isinstance(balance.get("included"), dict) else {}
    allowance = included.get("allowance_usd")
    remaining = included.get("balance_usd")
    if not _is_num(allowance) or float(allowance) <= 0 or not _is_num(remaining):
        return None
    allowance_value = float(allowance)
    remaining_value = max(0.0, min(allowance_value, float(remaining)))
    spent = max(0.0, allowance_value - remaining_value)
    period = included.get("period") if isinstance(included.get("period"), dict) else {}
    allowance_text = f"${allowance_value:.0f}" if allowance_value == int(allowance_value) else f"${allowance_value:.2f}"
    return AccountUsageWindow(
        label="Monthly included",
        used_percent=(spent / allowance_value) * 100.0,
        reset_at=_parse_dt(period.get("until")),
        detail=f"${spent:.2f} spent · ${remaining_value:.2f} left of {allowance_text}",
    )


def _fetch_ollama_cloud_account_usage(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    """Ollama Cloud ``/api/usage`` + ``/api/balance``; both bodies ride on raw."""
    token, resolved_base = _usage_credentials("ollama-cloud", base_url, api_key)
    if not token:
        return None
    base = (resolved_base or "https://ollama.com/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    with httpx.Client(timeout=15.0) as client:
        response = client.get(f"{base}/api/usage", headers=headers)
        response.raise_for_status()
        usage = _json_object(response)
        balance: dict = {}
        try:
            balance_response = client.get(f"{base}/api/balance", headers=headers)
            if balance_response.status_code == 200:
                parsed = balance_response.json() or {}
                if isinstance(parsed, dict):
                    balance = parsed
        except httpx.HTTPError:
            logger.debug("Ollama Cloud balance endpoint unavailable", exc_info=True)
    windows, limits = _ollama_cloud_usage_windows(usage)
    details = _ollama_cloud_details(usage, limits)
    credit = _ollama_cloud_credit_window(balance)
    balance_windows: list[AccountUsageWindow] = []
    if credit is not None:
        balance_windows.append(credit)
    else:
        # Older accounts publish the included pool as session/weekly
        # ``remaining_percent`` instead of allowance_usd/balance_usd.
        raw_included = balance.get("included")
        included: dict = raw_included if isinstance(raw_included, dict) else {}
        for key, label in (("session", "5-hour"), ("weekly", "Weekly")):
            entry = included.get(key)
            if not isinstance(entry, dict) or not _is_num(entry.get("remaining_percent")):
                continue
            remaining = max(0.0, min(100.0, float(entry["remaining_percent"])))
            balance_windows.append(AccountUsageWindow(
                label=label, used_percent=100.0 - remaining,
                reset_at=_parse_dt(entry.get("resets_at")),
            ))
    if balance_windows:
        windows = [*balance_windows, *windows]
    else:
        details.append("Reset times are not published on the usage report; they stay unavailable rather than estimated.")
    return _snapshot("ollama-cloud", "usage_api", windows, details, title="Ollama Cloud",
                     raw=_sanitized_raw({"usage": usage, "balance": balance}),
                     profile_scope=_profile_scope())


def _fetch_ollama_cloud_account_usage_read_only(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    return _fetch_ollama_cloud_account_usage(base_url, api_key)


def _fetch_opencode_zen_account_usage(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    """Verify Zen API access without scraping its browser-only wallet."""
    token, resolved_base = _usage_credentials("opencode-zen", base_url, api_key)
    if not token:
        return None
    base = (resolved_base or "https://opencode.ai/zen/v1").rstrip("/")
    payload = _get_json(
        f"{base}/models",
        {"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=15.0,
    )
    detail = "Zen API access is authenticated. Credit balance and spend are not exposed through OpenCode API keys."
    return _snapshot("opencode-zen", "models_api", [], [detail], title="OpenCode account",
                     plan="Zen API credits", raw=_sanitized_raw({"models": payload}),
                     profile_scope=_profile_scope())


def _fetch_opencode_zen_account_usage_read_only(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    return _fetch_opencode_zen_account_usage(base_url, api_key)


def _fetch_opencode_go_account_usage(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    """Fallback for OpenCode Go when the bundled profile hook returns nothing.

    The literal endpoint is intentional: the runtime base URL loses ``/v1`` in
    anthropic_messages mode, and ``/usage`` exists only under ``/v1``.
    """
    token, _resolved_base = _usage_credentials("opencode-go", base_url, api_key)
    if not token:
        return None
    payload = _get_json(
        "https://opencode.ai/zen/go/v1/usage",
        {"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=10.0,
    )
    raw_usage = payload.get("usage")
    usage: dict = raw_usage if isinstance(raw_usage, dict) else {}
    windows: list[AccountUsageWindow] = []
    for key, label in (("rolling", "Rolling window"), ("weekly", "Weekly"), ("monthly", "Monthly")):
        window = usage.get(key)
        if not isinstance(window, dict) or not _is_num(window.get("percent")):
            continue
        windows.append(AccountUsageWindow(
            label=label, used_percent=float(window["percent"]),
            reset_at=_parse_dt(window.get("resetsAt")),
        ))
    return _snapshot("opencode-go", "go_usage_api", windows, [], title="OpenCode usage",
                     plan="Go subscription", raw=_sanitized_raw(payload),
                     profile_scope=_profile_scope())


def _fetch_nous_account_usage(
    base_url: Optional[str], api_key: Optional[str], *, read_only: bool = False
) -> Optional[AccountUsageSnapshot]:
    """Nous credits from core's portal account helper.

    ``read_only`` never forces a portal refresh: the cached/JWT snapshot is
    used. A normal /usage read keeps the pane's existing fresh fetch.
    """
    del base_url, api_key  # Nous credentials are resolved by the portal helper.
    from hermes_cli.nous_account import get_nous_portal_account_info

    account = get_nous_portal_account_info(force_fresh=not read_only)
    snapshot = build_nous_credits_snapshot(account)
    if snapshot is None:
        return None
    return replace(snapshot, profile_scope=snapshot.profile_scope or _profile_scope())


def _fetch_nous_account_usage_read_only(
    base_url: Optional[str], api_key: Optional[str]
) -> Optional[AccountUsageSnapshot]:
    return _fetch_nous_account_usage(base_url, api_key, read_only=True)


def fetch_nous_account_usage(
    *, read_only: bool = False,
    base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    """Public Nous credits read: a fresh portal snapshot, or the cached/JWT one.

    ``read_only=True`` never forces a portal refresh. Kept as a named entry
    point because core's existing ``/usage`` surfaces already render the Nous
    credits block through ``nous_credits_lines``; registering Nous in the
    provider dispatch would print the same block twice.
    """
    return _fetch_nous_account_usage(base_url, api_key, read_only=read_only)


_USAGE_FETCHERS: dict[str, Callable[[Optional[str], Optional[str]], Optional[AccountUsageSnapshot]]] = {
    "openai-codex": _fetch_codex_account_usage, "anthropic": _fetch_anthropic_account_usage,
    "openrouter": _fetch_openrouter_account_usage,
    "deepseek": _fetch_deepseek_account_usage,
    "ollama-cloud": _fetch_ollama_cloud_account_usage,
    "opencode-zen": _fetch_opencode_zen_account_usage,
}

# Picker per-credential variants: same parsers/fetch shape, but a failure is never repaired by
# rotating/refreshing a credential (see ``fetch_account_usage(read_only=True)``). Nous lives only
# here: its normal /usage block is owned by ``nous_credits_lines``, so a provider dispatch would
# duplicate it, while the plugin's read-only panel needs the same snapshot.
_READ_ONLY_USAGE_FETCHERS: dict[str, Callable[[Optional[str], Optional[str]], Optional[AccountUsageSnapshot]]] = {
    "openai-codex": _fetch_codex_account_usage_read_only,
    "anthropic": _fetch_anthropic_account_usage_read_only,
    "openrouter": _fetch_openrouter_account_usage_read_only,
    "deepseek": _fetch_deepseek_account_usage_read_only,
    "ollama-cloud": _fetch_ollama_cloud_account_usage_read_only,
    "opencode-zen": _fetch_opencode_zen_account_usage_read_only,
    "nous": _fetch_nous_account_usage_read_only,
}

# Providers whose plugin profile hook is preferred but can be backed by a core
# read-only fetch when the hook has no answer. OpenCode Go's hook returns None
# without a key or when its endpoint call fails; the pane then still says why.
_USAGE_FALLBACK_FETCHERS: dict[str, Callable[[Optional[str], Optional[str]], Optional[AccountUsageSnapshot]]] = {
    "opencode-go": _fetch_opencode_go_account_usage,
}


# Wall-clock bound on a plugin profile's ``fetch_account_usage`` hook. The built-in fetchers above carry
# their own httpx timeouts; a plugin hook is arbitrary code, and the gateway/TUI ``/usage`` paths await
# this function with no deadline of their own (only the CLI wraps it in a 10 s future), so the bound
# lives here where every surface shares it.
PLUGIN_USAGE_HOOK_DEADLINE_S = 10.0


def _call_plugin_usage_hook(profile, base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    """Run the profile hook under the shared deadline; past it → None. Exceptions re-raise in the
    caller so ``fetch_account_usage`` fails open without a worker-thread traceback on ``/usage``."""
    from agent.deadline import run_bounded_sync
    from providers.base import ProviderProfile

    if type(profile).fetch_account_usage is ProviderProfile.fetch_account_usage:
        return None  # base no-op: no thread to spawn
    bounded = run_bounded_sync(
        lambda: profile.fetch_account_usage(base_url=base_url, api_key=api_key),
        PLUGIN_USAGE_HOOK_DEADLINE_S, label="plugin-account-usage")
    return None if bounded.timed_out else bounded.value


def list_account_usage_identities(provider: str, *, base_url: Optional[str] = None) -> list[dict[str, Any]]:
    """Non-secret inventory of this profile's pooled credentials for *provider*.

    Each row is ``{provider, identity_id, credential_id, account_id, profile_scope,
    label, auth_type, source, available, status, request_count, last_status_at,
    base_url}``.  ``identity_id`` is the same account namespace the usage cache
    and delegation roster use (``_identity_id_for``): ``codex:<account>:<sub>``,
    the cache's existing ``fp:<16 hex>`` token fingerprint, or ``entry:<pool
    id>`` for a metadata-only row.  It is a non-secret correlation id, never a
    token, and is the key :func:`fetch_account_usage` accepts.  ``credential_id``
    is the pool row's own stable id (``PooledCredential.id``).  ``account_id``
    is the provider's stable account id when it exposes one (today a decoded
    Codex JWT principal).  No access token, refresh token, agent key or
    credential-store value is returned.  ``base_url`` is stripped of userinfo,
    query and fragment.  A provider with no pool rows returns an empty list
    (the caller's config/env presence check stays separate).  A broken
    credential store fails closed to an empty list.
    """
    slug = str(provider or "").strip().lower()
    if not slug:
        return []
    try:
        from agent.credential_pool import load_pool

        entries = load_pool(slug).entries()
    except Exception:
        return []

    accounts: list[dict[str, Any]] = []
    seen: set[str] = set()
    scope = _profile_scope()
    for entry in entries:
        identity = _credential_identity(slug, entry)
        if not identity or identity in seen:
            # No account namespace at all: the footer must render this row as
            # unknown instead of receiving a made-up key. Rows for the same
            # Codex account collapse to one account-level row.
            continue
        seen.add(identity)
        token = str(getattr(entry, "runtime_api_key", "") or "").strip()
        credential_id = _credential_id(entry)
        status = str(getattr(entry, "last_status", None) or "ok")
        try:
            request_count = int(getattr(entry, "request_count", 0) or 0)
        except (TypeError, ValueError):
            request_count = 0
        accounts.append({
            "provider": slug,
            "identity_id": identity,
            "credential_id": _redacted_text(credential_id, token) if credential_id else None,
            "account_id": _credential_account_id(slug, entry),
            "profile_scope": scope,
            "label": _redacted_text(
                getattr(entry, "label", None) or getattr(entry, "source", None) or identity, token),
            "auth_type": _redacted_text(getattr(entry, "auth_type", None), token),
            "source": _redacted_text(getattr(entry, "source", None), token),
            "available": status not in {"dead", "exhausted"},
            "status": status,
            "request_count": request_count,
            "last_status_at": getattr(entry, "last_status_at", None),
            "base_url": _safe_base_url(
                getattr(entry, "runtime_base_url", None) or base_url, token=token),
        })
    return accounts


def provider_credential_summary(provider: str) -> dict[str, Any]:
    """Non-secret credential presence for one provider, pool-aware.

    Returns ``{provider, configured, account_count, source, profile_scope}``
    where ``source`` is ``pool`` / ``oauth`` / ``env`` / ``config`` / ``none``.
    A pool-only provider (no env var, no OAuth store) is configured because its
    pool rows are counted.  No value, header, path or base URL is returned.
    """
    slug = str(provider or "").strip().lower()
    scope = _profile_scope()
    if not slug:
        return {"provider": "", "configured": False, "account_count": 0, "source": "none",
                "profile_scope": scope}
    accounts = list_account_usage_identities(slug)
    if accounts:
        return {
            "provider": slug,
            "configured": True,
            "account_count": len(accounts),
            "source": "pool",
            "profile_scope": scope,
        }
    status: dict[str, Any] = {}
    try:
        from hermes_cli.auth import get_auth_status

        value = get_auth_status(slug)
        status = value if isinstance(value, dict) else {}
    except Exception:
        status = {}
    key_source = str(status.get("key_source") or "").strip()
    upstream = str(status.get("source") or "").strip().lower()
    configured = bool(status.get("logged_in") or status.get("configured"))
    if configured:
        if "pool" in key_source.lower() or upstream.startswith("pool"):
            source = "pool"
        elif key_source and key_source.upper() == key_source and any(ch.isalpha() for ch in key_source):
            source = "env"
        elif status.get("logged_in"):
            source = "oauth"
        else:
            source = "config"
    else:
        source = "none"
    return {
        "provider": slug,
        "configured": configured,
        "account_count": 1 if configured else 0,
        "source": source,
        "profile_scope": scope,
    }


def _entry_identity_matches(provider: str, entry: Any, wanted: str) -> bool:
    """Match the public account namespace plus the pool id/legacy ids the picker may hold."""
    identity = _credential_identity(provider, entry)
    if identity and wanted == identity:
        return True
    entry_id = _credential_id(entry)
    if entry_id and wanted == entry_id:
        return True
    if wanted.startswith("entry:") and entry_id and wanted[6:] == entry_id:
        return True
    account_id = _credential_account_id(provider, entry)
    if account_id and wanted == account_id:
        return True
    return False


def _account_usage_entry(provider: str, identity_id: str):
    """Pool row for *identity_id*, resolved inside core; the token stays here."""
    wanted = str(identity_id or "").strip()
    if not wanted:
        return None
    from agent.credential_pool import load_pool

    for entry in load_pool(provider).entries():
        try:
            if _entry_identity_matches(provider, entry, wanted):
                return entry
        except Exception:
            continue
    return None


def fetch_account_usage(
    provider: Optional[str], *, base_url: Optional[str] = None, api_key: Optional[str] = None,
    read_only: bool = False, identity_id: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    from agent.account_usage_cache import remember_account_usage

    slug = str(provider or "").strip().lower()
    resolved_entry = None
    # A per-account request names only the public identity from
    # ``list_account_usage_identities``; core resolves the pool row and its
    # token here, so plugin code never reads the credential store.
    if identity_id and not api_key:
        try:
            resolved_entry = _account_usage_entry(slug, identity_id)
        except Exception:
            resolved_entry = None
        if resolved_entry is None:
            return None
        api_key = str(getattr(resolved_entry, "runtime_api_key", "") or "").strip() or None
        base_url = str(getattr(resolved_entry, "runtime_base_url", "") or "").strip() or base_url
        if not api_key:
            return None
    # The picker's per-credential path resolves variants that never rotate/refresh a credential
    # on failure (Codex 401), so a read cannot repair or bench a pool entry it does not own.
    fetcher = _READ_ONLY_USAGE_FETCHERS.get(slug) if read_only else None
    if fetcher is None:
        fetcher = _USAGE_FETCHERS.get(slug)
    import time as _time

    started = _time.monotonic()
    try:
        if fetcher:
            snapshot = fetcher(base_url, api_key)
        else:
            from providers import get_provider_profile

            profile = get_provider_profile(slug)
            snapshot = None
            if profile:
                try:
                    snapshot = _call_plugin_usage_hook(profile, base_url, api_key)
                except Exception:
                    if slug not in _USAGE_FALLBACK_FETCHERS:
                        raise
            if snapshot is None and slug in _USAGE_FALLBACK_FETCHERS:
                snapshot = _USAGE_FALLBACK_FETCHERS[slug](base_url, api_key)
    except Exception:
        logger.debug("account usage fetch failed for %s (fail-open)", slug, exc_info=True)
        return None
    # A per-identity read knows exactly which credential it used; stamp that
    # stable id (never the token) so the footer cannot map the row to another
    # pooled account. Any other fetch stays unknown rather than guessing.
    # Nous portal credits are account-global (the fetcher never accepts a
    # credential), so stamping a pool row there would be a false correspondence.
    if snapshot is not None and resolved_entry is not None and slug != "nous":
        snapshot = replace(
            snapshot,
            identity_id=_credential_identity(slug, resolved_entry),
            account_id=_credential_account_id(slug, resolved_entry),
            profile_scope=snapshot.profile_scope or _profile_scope(),
        )
    # Every fetch (``/usage``, the per-turn ``session.usage``) keeps the picker's cache current.
    # A per-credential fetch is remembered under its account slot only; the provider-wide fetch
    # keeps both the legacy gauge slot and (when the fetcher identifies the account) that
    # account's slot current.
    remember_account_usage(
        provider, snapshot,
        identity_id=_credential_identity(slug, resolved_entry) if slug != "nous" else None,
        base_url=base_url, started_monotonic=started,
    )
    return snapshot
