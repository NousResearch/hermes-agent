from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpx

from agent.anthropic_credentials import _is_oauth_token, resolve_anthropic_token
from hermes_cli.auth import AuthError, _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.auth_codex import _codex_pool_route_base_url
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_time import safe_strftime

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger(__name__)


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


def account_limit_lines(provider: Optional[str], *, base_url: Optional[str] = None, api_key: Optional[str] = None,
                        markdown: bool = False) -> list[str]:
    """``/usage``'s provider plan-limits block on every surface: the built-in fetcher, else the plugin
    profile's ``fetch_account_usage`` hook (Claude plan limits on DirectSDK). [] when there is none."""
    if not provider:
        return []
    return render_account_usage_lines(fetch_account_usage(provider, base_url=base_url, api_key=api_key),
                                      markdown=markdown)


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


def _is_num(v: Any) -> TypeGuard[float]:
    return isinstance(v, (int, float))


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
    source: dict, mapping: tuple[tuple[str, str], ...], used_key: str, reset_key: str, *,
    model_scoped: frozenset[str] | set[str] = frozenset(),
) -> list[AccountUsageWindow]:
    """Build windows from ``source[key][used_key]``.

    ``used_key`` values are PERCENTAGES (0-100) on every provider that reaches this helper —
    never 0-1 fractions. An earlier version scaled any value <= 1 by 100 to accommodate a
    fraction shape; that heuristic silently turned Anthropic's ``seven_day.utilization`` of 1.0
    ("1% of the weekly limit used") into 100% used. Do not reintroduce it.

    ``model_scoped`` keys build windows that cap only a model family, never the account.
    """
    windows: list[AccountUsageWindow] = []
    for key, label in mapping:
        window = source.get(key) or {}
        used = window.get(used_key)
        if used is None:
            continue
        used = float(used)
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
    return _snapshot("openai-codex", "usage_api", windows, details, plan=_title_case_slug(payload.get("plan_type")),
                     identity=_codex_snapshot_identity(token), raw=payload)


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
    base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[AccountUsageSnapshot]:
    # An explicit api_key (live agent / per-credential pool read) must not be shadowed by
    # the ambient OAuth singleton: fetch_account_usage(base_url, api_key) promises to
    # report THAT credential's usage. An API key answers the OAuth usage endpoint with
    # 401, so only an OAuth-shaped token reaches the fetch.
    explicit = str(api_key or "").strip()
    token = explicit or (resolve_anthropic_token() or "").strip()
    if not token:
        return None
    if not _is_oauth_token(token):
        return _snapshot("anthropic", "oauth_usage_api", [], [],
                         unavailable_reason="Anthropic account limits are only available for OAuth-backed Claude accounts.")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json",
               "anthropic-beta": "oauth-2025-04-20", "User-Agent": "claude-code/2.1.0"}
    payload = _get_json("https://api.anthropic.com/api/oauth/usage", headers, timeout=15.0)
    # Anthropic's ``utilization`` is already a PERCENT (0-100), never a 0-1 fraction — it mirrors
    # the ``limits[].percent`` field for the same window (live check 2026-11-03: five_hour 5.0 ->
    # "5% used", seven_day 1.0 -> "1% used" on the Claude dashboard). The ``fraction`` heuristic
    # therefore mis-scaled a window that reports exactly 1.0: "1% used" was multiplied by 100 and
    # rendered as "100% used / 0% remaining". Do NOT pass fraction=True here.
    windows = _usage_windows(
        payload, (("five_hour", "Current session"), ("seven_day", "Current week"), ("seven_day_opus", "Opus week"),
                  ("seven_day_sonnet", "Sonnet week")), "utilization", "resets_at",
        model_scoped={"seven_day_opus", "seven_day_sonnet"},
    )
    details: list[str] = []
    extra = payload.get("extra_usage") or {}
    used_credits, monthly_limit = extra.get("used_credits"), extra.get("monthly_limit")
    if extra.get("is_enabled") and _is_num(used_credits) and _is_num(monthly_limit):
        details.append(f"Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {extra.get('currency') or 'USD'}")
    return _snapshot("anthropic", "oauth_usage_api", windows, details)


def _fetch_minimax_account_usage(
    base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[AccountUsageSnapshot]:
    """MiniMax M-Plan usage windows from ``/v1/token_plan/remains``.

    This endpoint accepts the SAME OAuth bearer used for inference (scope
    ``group_id profile model.completion``) — no admin API key is needed. The separate
    ``/v1/account/balance`` endpoint is a different thing: it answers HTTP 403
    ``token_type_mismatch`` ("api key is not an admin key") for an inference-scoped token.

    Each ``model_remains`` entry is one model family (``general``, ``video``) with its own
    5-hour and weekly window. ``current_*_remaining_percent`` is the PERCENT REMAINING (0-100),
    so the used percent is ``100 - remaining``. Window bounds are epoch MILLISECONDS.
    """
    token = str(api_key or "").strip()
    if not token:
        try:
            from hermes_cli.auth_minimax import resolve_minimax_oauth_runtime_credentials
            token = str(resolve_minimax_oauth_runtime_credentials().get("api_key") or "").strip()
        except Exception:
            return None
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = _get_json("https://api.minimax.io/v1/token_plan/remains", headers, timeout=15.0)
    windows: list[AccountUsageWindow] = []
    details: list[str] = []
    for entry in payload.get("model_remains") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("model_name") or "model").strip() or "model"
        for pct_key, reset_key, label in (
            ("current_interval_remaining_percent", "end_time", f"5-hour ({name})"),
            ("current_weekly_remaining_percent", "weekly_end_time", f"Weekly ({name})"),
        ):
            remaining = entry.get(pct_key)
            if not _is_finite_num(remaining):
                continue
            reset_ms = entry.get(reset_key)
            reset_at = _parse_dt(float(reset_ms) / 1000.0) if _is_finite_num(reset_ms) else None
            # ``scope="model"``: each ``model_remains`` entry carries its OWN allowance, so an
            # exhausted ``video`` family says nothing about the ``general`` (text) allowance.
            # An account-scoped window here made the picker mark the whole credential
            # "limited" (``hermes_cli/inventory.py``: any exhausted account-scoped window
            # benches the account) while its text quota was untouched.
            windows.append(AccountUsageWindow(
                label=label, used_percent=max(0.0, 100.0 - float(remaining)),
                reset_at=reset_at, scope="model",
            ))
    return _snapshot("minimax-oauth", "token_plan_remains", windows, details, raw=payload)


def _fetch_xai_account_usage(
    base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[AccountUsageSnapshot]:
    """The live-agent fetch: it owns the credential, so it may refresh it.

    See ``_fetch_xai_account_usage_read_only`` for the picker's per-credential read.
    """
    return _fetch_xai_account_usage_impl(base_url, api_key, read_only=False)


def _fetch_xai_account_usage_read_only(
    base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[AccountUsageSnapshot]:
    """Picker-cache variant: a usage READ never rotates or refreshes the credential.

    The live path may refresh an expiring token because the session it serves owns that
    credential; a per-credential read of the pool must NOT rotate it -- it just marks
    that account unknown. So this path resolves the token with
    ``refresh_if_expiring=False`` (no auth-store lock, no refresh) and lets a 401
    surface as a failure.
    """
    return _fetch_xai_account_usage_impl(base_url, api_key, read_only=True)


def _fetch_xai_account_usage_impl(
    base_url: Optional[str] = None, api_key: Optional[str] = None, *, read_only: bool = False
) -> Optional[AccountUsageSnapshot]:
    """Grok / X Premium+ subscription credit usage from the Grok CLI chat proxy.

    ``GET https://cli-chat-proxy.grok.com/v1/billing?format=credits`` is the CLI's own
    credits surface — the same one the Grok Build TUI's ``/usage`` reads. **Without the
    ``format=credits`` query** the endpoint answers the METERED-API billing shape
    (``monthlyLimit``/``used``/``onDemandCap``, all zero for a subscription account),
    which says nothing about the weekly subscription pool.

    Needs the CLI's identity header ``x-xai-token-auth: xai-grok-cli`` alongside the
    bearer. ``config.creditUsagePercent`` is the PERCENT of the weekly pool USED and
    ``config.currentPeriod.end`` is the reset. ``config.productUsage`` breaks the pool
    down by product (GrokChat / GrokBuild / GrokImagine) as shares of the same pool.

    The separate developer-platform prepaid balance lives on the xAI Management API
    (``management-api.x.ai``), which needs a management key and does not apply to a
    consumer subscription account — xAI answers that team as
    ``team:personal-team-immutable``.
    """
    token = str(api_key or "").strip()
    if not token:
        try:
            from hermes_cli.auth_xai import resolve_xai_oauth_runtime_credentials
            # A read must not take the auth-store lock or rotate the xAI refresh token:
            # ``refresh_if_expiring`` defaults to True, so the default would repair (and
            # mutate) the credential a read-only pick merely wants to look at.
            token = str(resolve_xai_oauth_runtime_credentials(
                refresh_if_expiring=not read_only).get("api_key") or "").strip()
        except Exception:
            return None
    if not token:
        return None
    headers = {
        "Authorization": f"Bearer {token}",
        "x-xai-token-auth": "xai-grok-cli",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = _get_json(
        "https://cli-chat-proxy.grok.com/v1/billing?format=credits", headers, timeout=20.0)
    config = payload.get("config") or {}
    windows: list[AccountUsageWindow] = []
    details: list[str] = []
    used_percent = config.get("creditUsagePercent")
    if _is_finite_num(used_percent):
        period = config.get("currentPeriod") or {}
        reset_at = _parse_dt(period.get("end")) or _parse_dt(config.get("billingPeriodEnd"))
        label = "Weekly" if str(period.get("type") or "").endswith("WEEKLY") else "Credits"
        windows.append(AccountUsageWindow(
            label=label, used_percent=float(used_percent), reset_at=reset_at, scope="account",
        ))
    for product in config.get("productUsage") or []:
        if not isinstance(product, dict):
            continue
        name = str(product.get("product") or "").strip()
        share = product.get("usagePercent")
        if name and _is_finite_num(share):
            details.append(f"{name}: {float(share):g}% of the pool")
    plan = None
    try:
        settings = _get_json("https://cli-chat-proxy.grok.com/v1/settings", headers, timeout=10.0)
        plan = _title_case_slug(settings.get("subscription_tier_display"))
    except Exception:
        pass
    return _snapshot("xai-oauth", "cli_proxy_credits", windows, details, plan=plan, raw=payload)


def _fetch_openrouter_account_usage(base_url: Optional[str] = None, api_key: Optional[str] = None) -> Optional[AccountUsageSnapshot]:
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
    "openrouter": _fetch_openrouter_account_usage, "minimax-oauth": _fetch_minimax_account_usage,
    "xai-oauth": _fetch_xai_account_usage,
}

# Picker per-credential variants: same parsers/fetch shape, but a failure is never repaired by
# rotating/refreshing a credential (see ``fetch_account_usage(read_only=True)``).
_READ_ONLY_USAGE_FETCHERS: dict[str, Callable[[Optional[str], Optional[str]], Optional[AccountUsageSnapshot]]] = {
    "openai-codex": _fetch_codex_account_usage_read_only,
    "xai-oauth": _fetch_xai_account_usage_read_only,
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


def fetch_account_usage(
    provider: Optional[str], *, base_url: Optional[str] = None, api_key: Optional[str] = None,
    read_only: bool = False, identity_id: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    from agent.account_usage_cache import remember_account_usage

    slug = str(provider or "").strip().lower()
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
            snapshot = _call_plugin_usage_hook(profile, base_url, api_key) if profile else None
    except Exception:
        return None
    # Every fetch (``/usage``, the per-turn ``session.usage``) keeps the picker's cache current.
    # A per-credential fetch is remembered under its account slot only; the provider-wide fetch
    # keeps both the legacy gauge slot and (when the fetcher identifies the account) that
    # account's slot current.
    remember_account_usage(provider, snapshot, identity_id=identity_id, base_url=base_url,
                           started_monotonic=started)
    return snapshot


# Published plugin API (hermes_cli/data/plugin_api.json lists this whole module): names that moved to the Nous
# sibling stay importable here; catalog plugins import build_credits_view and build_nous_credits_snapshot.
# Bottom of the file because the sibling late-imports this facade.
from agent.usage_gauges import BP_FULL
from agent.account_usage_nous import (
    CreditsView,
    apply_header_gauge,
    build_credits_view,
    build_nous_credits_snapshot,
    nous_credits_lines,
    nous_credits_usage_fields,
    nous_credits_view,
)
