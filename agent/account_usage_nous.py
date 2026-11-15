"""``/usage``'s Nous account views: the portal fetch, the credits and usage-window snapshot (window bars,
Keep going and billing hints), the off-route section and the ``/topup`` balance view.

Kept apart from ``agent.account_usage`` (snapshot types and the other providers' fetchers): only Nous
reads the portal account. Late-imports the facade's snapshot types.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

from agent.usage_gauges import BP_FULL

if TYPE_CHECKING:
    from typing import TypeGuard

    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

logger = logging.getLogger(__name__)

_DEPLETED_LINE = "Status: access depleted — top up to restore"
# Usage-window plans: allowances are shown as percent used.
_WINDOW_TOP_TIER_USED_UP = "Monthly large-model usage is used up."
_WINDOW_PAUSED_LINE = ("Status: included usage paused. Your subscription payment failed. Update your card to restore it. "
                       "Until then, Hermes uses your credits.")
_WINDOW_PAUSED_OTHER_LINE = "Status: included usage paused. See Billing in the Portal. Until then, Hermes uses your credits."


def _fmt_usd(d: float) -> str:
    return f"${d:,.2f}"


def _is_finite_num(v: Any) -> TypeGuard[float]:
    """True iff v is a real number (int/float, not bool, not NaN/Inf); TypeGuard so callers can do arithmetic."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _nous_snapshot(windows: list, details: list, tail: list, *, source: str, plan: Optional[str] = None,
                   usage_window: bool = False) -> Optional[AccountUsageSnapshot]:
    """Nous snapshot with *tail* lines appended, or None when there is nothing to show."""
    from agent.account_usage import _snapshot
    if not windows and not details:
        return None
    return _snapshot("nous", source, windows, details + tail, title="Nous credits", plan=plan, usage_window=usage_window)


def _bp_window(label: str, used_bp: Any, resets_at_ms: Any) -> Optional[AccountUsageWindow]:
    """One usage-window gauge row from basis points + an epoch-ms reset (already bounded by
    ``usage_gauges.parse_gauges``); None when the percent is not an int. An unknown reset keeps the row,
    shown without a reset."""
    from agent.account_usage import AccountUsageWindow
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


_WINDOW_KEEP_GOING_OFF_STATUS = "Keep going on credits: off. Nous sessions wait for the reset."
_WINDOW_KEEP_GOING_OFF_LINE = "Keep going on credits: off. Hermes waits for the reset (turn on in /topup)."
_WINDOW_KEEP_GOING_ON_LINE = "Keep going on credits: on. Uses your credits when included usage runs out."


def _usage_window_rows(gauges: Any, *, paused_reason: Any, keep_going_on_credits: Any = None,
                       cta: bool = True) -> tuple[list[AccountUsageWindow], list[str]]:
    """One row per server allowance, in the server's order, plus the top-tier, payment-paused and top-up-off
    lines. Empty unless the list has an allowance that stops every model. ``keep_going_on_credits``: the
    account setting (None: an older server). ``cta=False``: the setting's status without its /topup action."""
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
        lines.append(_WINDOW_KEEP_GOING_OFF_LINE if cta else _WINDOW_KEEP_GOING_OFF_STATUS)
    elif keep_going_on_credits is True:
        lines.append(_WINDOW_KEEP_GOING_ON_LINE)
    return out, lines


def build_nous_credits_snapshot(account_info, *, cta: bool = True) -> Optional[AccountUsageSnapshot]:
    """NousPortalAccountInfo → /usage snapshot: dollar magnitudes + renewal date + portal CTA, plus a ``% used``
    gauge when the portal supplies ``monthly_credits``. A usage-window account (``usage_window`` present) shows
    its allowances as percent used, the large-model line and its top-up balance instead. Fail-open → None."""
    from agent.account_usage import AccountUsageWindow
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
                keep_going_on_credits=getattr(uw, "keep_going_on_credits", None), cta=cta,
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
            return _nous_snapshot(windows, details, ["Top up via /topup"] if cta else [], source="portal-account",
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
        tail = [f"Top up: {nous_portal_topup_url(account_info)}", "(or run /topup)"] if cta else []
        return _nous_snapshot(windows, details, tail,
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


def _fetch_portal_account(timeout: float, *, stored_token_only: bool = False):
    """Wall-clock-bounded fresh portal account fetch (raises on any failure/timeout).

    No ``with`` block on purpose: ``Executor.__exit__`` joins the worker via
    ``shutdown(wait=True)``, so a portal that accepts the connection but never
    answers would hold the caller until the provider's own timeout instead of
    ``timeout``. The abandoned daemon worker runs on to its own network timeout
    and never blocks the caller or process exit; its eventual exception is
    drained so GC never logs "exception was never retrieved". ``stored_token_only``: read with the stored
    token, never refreshing it (``stored_token_account_info``)."""
    import contextvars
    from functools import partial
    from hermes_cli.nous_account import get_nous_portal_account_info, stored_token_account_info
    from tools.daemon_pool import DaemonThreadPoolExecutor

    context = contextvars.copy_context()
    pool = DaemonThreadPoolExecutor(max_workers=1)
    read = stored_token_account_info if stored_token_only else partial(get_nous_portal_account_info, force_fresh=True)
    future = pool.submit(context.run, read)
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


def nous_credits_lines(*, markdown: bool = False, timeout: float = 10.0, credits_state: Any = None,
                       nous_route: bool = True) -> list[str]:
    """Rendered Nous-credits /usage lines, or [] when there's nothing to show. Independent of any live agent
    (logged-in gate, then a bounded portal fetch); shared by CLI ``_show_usage`` and the TUI ``session.usage`` RPC.
    Fail-open: any hiccup or timeout → []. HERMES_DEV_CREDITS_FIXTURE renders from the fixture instead of the portal.
    ``credits_state`` is the last response's header gauge; when it is ahead of the portal, the bars use it."""
    return nous_credits_view(markdown=markdown, timeout=timeout, credits_state=credits_state, nous_route=nous_route)[0]


def nous_credits_view(*, markdown: bool = False, timeout: float = 10.0, credits_state: Any = None,
                      nous_route: bool = True) -> tuple[list[str], bool]:
    """``nous_credits_lines`` plus whether they are the usage-window layout (callers keep today's
    presentation for everyone else). ``nous_route=False`` (a session on another provider): the stored
    token only, never refreshed, and no top-up lines. Fail-open → ``([], False)``."""
    lines, is_window, _account = _nous_account_view(markdown=markdown, timeout=timeout, credits_state=credits_state,
                                                    nous_route=nous_route)
    return lines, is_window


def _nous_account_view(*, markdown: bool, timeout: float, credits_state: Any,
                       nous_route: bool) -> tuple[list[str], bool, Any]:
    """``nous_credits_view`` plus the account it read (None: the dev fixture, signed out or a failed read)."""
    from agent.account_usage import render_account_usage_lines
    try:
        from agent.credits_tracker import dev_fixture_credits_state
        fixture = dev_fixture_credits_state()
    except Exception:
        logger.debug("Dev credits fixture unavailable", exc_info=True)
        fixture = None
    if fixture is not None:
        snapshot = _snapshot_from_credits_state(fixture)
        return render_account_usage_lines(snapshot, markdown=markdown), bool(snapshot and snapshot.usage_window), None
    if not _nous_logged_in():
        return [], False, None
    try:
        account = _fetch_portal_account(timeout, stored_token_only=not nous_route)
        snapshot = apply_header_gauge(build_nous_credits_snapshot(account, cta=nous_route), credits_state)
        return render_account_usage_lines(snapshot, markdown=markdown), bool(snapshot and snapshot.usage_window), account
    except Exception:
        # Fail-open; breadcrumb so a dead /usage credits block is diagnosable.
        logger.debug("credits ▸ /usage portal fetch/render failed (fail-open)", exc_info=True)
        return [], False, None


def nous_credits_usage_fields(agent: Any = None, *, nous_route: bool = True) -> dict[str, Any]:
    """The ``session.usage`` fields for the Nous credits block: ``credits_lines`` (when there are any) and
    ``credits_usage_window`` for the usage-window layout. Reads *agent*'s live credits headers; off a Nous
    route the lines are :func:`nous_usage_lines`' separate section. Never raises."""
    try:
        lines, is_window = nous_credits_view(credits_state=getattr(agent, "_credits_state", None) if agent else None,
                                             nous_route=nous_route)
    except Exception:
        logger.debug("credits ▸ /usage fields unavailable", exc_info=True)
        return {}
    lines = lines if nous_route or not lines else [NOUS_OTHER_ROUTE_HEADING, *lines]
    return ({"credits_lines": lines} | ({"credits_usage_window": True} if is_window else {})) if lines else {}


# Heads the Nous section of a session another provider runs, so it never reads as that provider's balance.
NOUS_OTHER_ROUTE_HEADING = "Nous Portal balance (a separate account: this session runs on another provider)"


def nous_usage_lines(*, nous_route: bool, markdown: bool = False, credits_state: Any = None) -> list[str]:
    """``/usage``'s Nous section, after the active provider's block: the balance whenever Nous is signed
    in on this profile, whatever provider runs the session. Off a Nous route it carries its own heading,
    reads with the stored token only and has no top-up lines: that session spends another provider's
    quota, so a top-up nudge would point at the wrong account (a depleted balance keeps its status line)."""
    lines = nous_credits_lines(markdown=markdown, credits_state=credits_state, nous_route=nous_route)
    if nous_route or not lines:
        return lines
    return [f"**{NOUS_OTHER_ROUTE_HEADING}**" if markdown else NOUS_OTHER_ROUTE_HEADING, *lines]


def nous_usage_view(*, nous_route: bool, credits_state: Any = None) -> tuple[list[str], Any]:
    """:func:`nous_usage_lines` and, on a Nous route, the same read's dollar model
    (``agent.billing_usage.UsageModel``: the classic CLI's two-bar block), so ``/usage`` reads the account once."""
    lines, _window, account = _nous_account_view(markdown=False, timeout=10.0, credits_state=credits_state,
                                                 nous_route=nous_route)
    if not nous_route:
        return ([NOUS_OTHER_ROUTE_HEADING, *lines] if lines else []), None
    from agent.billing_usage import _dev_fixture_usage_model, usage_model_from_account
    return lines, _dev_fixture_usage_model() or usage_model_from_account(account)


def _snapshot_from_credits_state(state) -> Optional[AccountUsageSnapshot]:
    """Header-shaped CreditsState (dev fixture) → /usage snapshot, same shape as the portal path. *_usd strings
    are display-only; the % comes from CreditsState.used_fraction. Fail-open → None."""
    from agent.account_usage import AccountUsageWindow
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
        if isinstance(uf, (int, float)) and math.isfinite(uf):
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
    from agent.account_usage import render_account_usage_lines
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
