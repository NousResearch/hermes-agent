"""Hermes-native Claude/Codex plan quota snapshots for lightweight UI display.

The status bar must stay cheap: no blocking network I/O or credential work in the
render path.  This module reads small JSON snapshots under ``$HERMES_HOME/state``
and, when the active provider is Claude Code or OpenAI Codex, starts a throttled
background refresh that writes the same Hermes-owned snapshot format.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional

from hermes_constants import get_hermes_home

NativeQuotaProvider = Literal["claude-cli", "openai-codex"]
NativeQuotaWindowKey = Literal["five_hour", "seven_day"]

DEFAULT_STALE_AFTER_SECONDS = 10 * 60
DEFAULT_CACHE_TTL_SECONDS = 4.0
DEFAULT_REFRESH_INTERVAL_SECONDS = 10 * 60
DEFAULT_REFRESH_TIMEOUT_SECONDS = 8.0

RequestJson = Callable[[str, Mapping[str, str], float], Any]


@dataclass(frozen=True)
class NativeQuotaWindow:
    key: NativeQuotaWindowKey
    used_percentage: float
    resets_at: str | None = None

    @property
    def label(self) -> str:
        return "5h" if self.key == "five_hour" else "7d"


@dataclass(frozen=True)
class NativeQuotaSummary:
    provider: NativeQuotaProvider
    short_label: str
    fetched_at: str
    age_seconds: float
    stale: bool
    windows: tuple[NativeQuotaWindow, ...]
    source_path: str


@dataclass
class _CacheEntry:
    summary: NativeQuotaSummary | None
    cached_at: float
    source_paths: tuple[str, ...]
    provider: NativeQuotaProvider


_cache: dict[NativeQuotaProvider, _CacheEntry] = {}
_refresh_lock = threading.Lock()
_refresh_inflight: set[NativeQuotaProvider] = set()
_refresh_attempted_at: dict[NativeQuotaProvider, float] = {}


def active_native_quota_provider(
    provider: str | None,
    model: str | None,
    base_url: str | None = None,
) -> NativeQuotaProvider | None:
    """Return the plan-quota provider relevant to the active model, if any."""
    provider_l = (provider or "").lower()
    model_l = (model or "").lower()
    base_l = (base_url or "").lower()

    if "openai-codex" in provider_l or provider_l == "codex" or "/backend-api/codex" in base_l:
        return "openai-codex"
    if "claude-cli" in provider_l:
        return "claude-cli"

    # Hermes/Codex sessions commonly show the model (for example ``gpt-5.5``)
    # while the provider carries the plan identity.  Keep model-only inference
    # conservative: Claude model IDs are plan-relevant; GPT model IDs are not
    # assumed to be Codex unless the provider/base_url says so.
    if "claude" in model_l:
        return "claude-cli"
    return None


def get_native_quota_statusbar_for_model(
    provider: str | None,
    model: str | None,
    base_url: str | None = None,
    *,
    runtime_api_key: str | None = None,
    runtime_account_id: str | None = None,
    state_dirs: Iterable[str | os.PathLike[str]] | None = None,
    now: float | None = None,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    auto_refresh: bool = True,
) -> str:
    """Return a compact status-bar string for the active Claude/Codex plan.

    Empty string means no relevant provider or no usable snapshot.  The output is
    intentionally compact, e.g. ``cdx 5h 12%↻3h 7d 4%↻5d``.  When using the
    default Hermes state directory, this call may kick off a non-blocking,
    throttled refresh thread; it never waits for provider network I/O.
    """
    quota_provider = active_native_quota_provider(provider, model, base_url)
    if quota_provider is None:
        return ""
    if auto_refresh and state_dirs is None:
        runtime_credentials: dict[str, Any] | None = None
        if quota_provider == "openai-codex" and runtime_api_key:
            runtime_credentials = {
                "api_key": runtime_api_key,
                "base_url": base_url or "https://chatgpt.com/backend-api/codex",
            }
            if runtime_account_id:
                runtime_credentials["account_id"] = runtime_account_id
        ensure_native_quota_refresh_async(
            quota_provider,
            now=now,
            stale_after_seconds=stale_after_seconds,
            codex_credentials=runtime_credentials,
        )
    summary = read_native_quota_summary(
        quota_provider,
        state_dirs=state_dirs,
        now=now,
        stale_after_seconds=stale_after_seconds,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    if summary is None:
        return ""
    return format_native_quota_statusbar(summary, now=now)


def read_native_quota_summary(
    provider: NativeQuotaProvider,
    *,
    state_dirs: Iterable[str | os.PathLike[str]] | None = None,
    now: float | None = None,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> NativeQuotaSummary | None:
    """Read the freshest compatible Hermes native quota snapshot."""
    now_ts = time.time() if now is None else float(now)
    dirs = tuple(str(p) for p in _quota_state_dirs(state_dirs))

    cached = _cache.get(provider)
    if cached and cached.source_paths == dirs and now_ts - cached.cached_at < cache_ttl_seconds:
        return cached.summary

    path = _freshest_snapshot_path(provider, dirs)
    summary = _read_snapshot(path, provider, now_ts, stale_after_seconds) if path else None
    _cache[provider] = _CacheEntry(summary=summary, cached_at=now_ts, source_paths=dirs, provider=provider)
    return summary


def clear_native_quota_cache() -> None:
    """Clear the small in-process quota snapshot cache (mainly for tests)."""
    _cache.clear()


def format_native_quota_statusbar(summary: NativeQuotaSummary, *, now: float | None = None) -> str:
    """Format a snapshot summary for the single-line CLI status bar."""
    parts = [f"{summary.short_label}{'~' if summary.stale else ''}"]
    now_ts = time.time() if now is None else float(now)
    for window in summary.windows:
        pct = _format_pct(window.used_percentage)
        reset = _format_reset_countdown(window.resets_at, now_ts)
        parts.append(f"{window.label} {pct}{reset or ''}")
    return " ".join(parts) if len(parts) > 1 else ""


def ensure_native_quota_refresh_async(
    provider: NativeQuotaProvider,
    *,
    now: float | None = None,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    refresh_interval_seconds: float = DEFAULT_REFRESH_INTERVAL_SECONDS,
    codex_credentials: Mapping[str, Any] | None = None,
    claude_credentials: Mapping[str, Any] | None = None,
) -> bool:
    """Start a throttled background refresh for ``provider`` if needed.

    Returns True when a new daemon thread was started.  The function is safe to
    call from the status bar render path: it does only cheap timestamp checks and
    disk reads before returning.
    """
    now_ts = time.time() if now is None else float(now)
    with _refresh_lock:
        if provider in _refresh_inflight:
            return False
        last = _refresh_attempted_at.get(provider)
        if last is not None and now_ts - last < refresh_interval_seconds:
            return False

    summary = read_native_quota_summary(
        provider,
        now=now_ts,
        stale_after_seconds=stale_after_seconds,
        cache_ttl_seconds=0,
    )
    if summary is not None and not summary.stale:
        return False

    with _refresh_lock:
        if provider in _refresh_inflight:
            return False
        last = _refresh_attempted_at.get(provider)
        if last is not None and now_ts - last < refresh_interval_seconds:
            return False
        _refresh_attempted_at[provider] = now_ts
        _refresh_inflight.add(provider)

    def _worker() -> None:
        try:
            refresh_native_quota_snapshot(
                provider,
                codex_credentials=codex_credentials,
                claude_credentials=claude_credentials,
            )
        finally:
            with _refresh_lock:
                _refresh_inflight.discard(provider)

    thread = threading.Thread(target=_worker, name=f"native-quota-refresh-{provider}", daemon=True)
    thread.start()
    return True


def refresh_native_quota_snapshot(
    provider: NativeQuotaProvider,
    *,
    now: datetime | None = None,
    request_json: RequestJson | None = None,
    codex_credentials: Mapping[str, Any] | None = None,
    claude_credentials: Mapping[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_REFRESH_TIMEOUT_SECONDS,
) -> Path | None:
    """Fetch native plan quota for ``provider`` and write a Hermes snapshot.

    This is the standalone Hermes refresh path.  It uses Hermes's own Codex auth
    state and Claude Code OAuth credentials; it never reads Pi runtime files.
    """
    now_dt = now or datetime.now(timezone.utc)
    fetcher = request_json or _request_json
    if provider == "openai-codex":
        snapshot = _refresh_codex_snapshot(
            now_dt,
            fetcher,
            codex_credentials=codex_credentials,
            timeout_seconds=timeout_seconds,
        )
    else:
        snapshot = _refresh_claude_snapshot(
            now_dt,
            fetcher,
            claude_credentials=claude_credentials,
            timeout_seconds=timeout_seconds,
        )
    if not snapshot:
        return None
    path = _native_quota_snapshot_path(provider)
    _write_snapshot_json(path, snapshot)
    clear_native_quota_cache()
    return path


def _refresh_codex_snapshot(
    now_dt: datetime,
    request_json: RequestJson,
    *,
    codex_credentials: Mapping[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    creds: Mapping[str, Any]
    if codex_credentials is not None:
        creds = codex_credentials
    else:
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials
            creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        except Exception:
            creds = _codex_credentials_from_pool() or {}

    access_token = str(creds.get("api_key") or creds.get("access_token") or "").strip()
    if not access_token and codex_credentials is None:
        creds = _codex_credentials_from_pool() or {}
        access_token = str(creds.get("api_key") or creds.get("access_token") or "").strip()
    if not access_token:
        return None
    base_url = str(creds.get("base_url") or "https://chatgpt.com/backend-api/codex").rstrip("/")
    endpoint = f"{base_url}/usage" if "/backend-api" in base_url else f"{base_url}/api/codex/usage"
    explicit_account_id = str(creds.get("account_id") or "").strip()
    derived_account_id = _codex_account_id_from_token(access_token) or ""
    account_id = explicit_account_id or derived_account_id
    headers = {
        "authorization": f"Bearer {access_token}",
        "user-agent": "codex-cli",
        "accept": "application/json",
    }
    if account_id:
        headers["chatgpt-account-id"] = account_id

    payload = request_json(endpoint, headers, timeout_seconds)
    if not isinstance(payload, Mapping) and account_id and not explicit_account_id:
        # Some ChatGPT accounts reject a derived JWT account id even though the
        # bearer token can resolve the correct default account on its own.  Keep
        # explicit stored/runtime account ids authoritative, but retry without a
        # best-effort derived header before giving up.
        headers.pop("chatgpt-account-id", None)
        payload = request_json(endpoint, headers, timeout_seconds)
    if not isinstance(payload, Mapping):
        return None
    return _sanitize_codex_usage_payload(payload, now_dt)


def _codex_credentials_from_pool() -> dict[str, Any] | None:
    """Return the active profile's Codex pool credential for quota refresh.

    Profiles often keep OAuth credentials in the credential pool instead of the
    legacy ``providers.openai-codex`` slot.  The status bar refresh path should
    be able to use the same profile-scoped pool entry as inference without
    leaking token values into the render path.
    """
    try:
        from agent.credential_pool import load_pool
        pool = load_pool("openai-codex")
        if not pool or not pool.has_credentials():
            return None
        entry = pool.peek()
    except Exception:
        return None
    if not entry:
        return None
    access_token = str(getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", None) or "").strip()
    if not access_token:
        return None
    creds: dict[str, Any] = {"api_key": access_token}
    base_url = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None)
    if base_url:
        creds["base_url"] = str(base_url)
    account_id = getattr(entry, "account_id", None)
    if account_id:
        creds["account_id"] = str(account_id)
    return creds


def _refresh_claude_snapshot(
    now_dt: datetime,
    request_json: RequestJson,
    *,
    claude_credentials: Mapping[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    creds: Mapping[str, Any]
    if claude_credentials is not None:
        creds = claude_credentials
    else:
        try:
            from agent.anthropic_adapter import read_claude_code_credentials
            found = read_claude_code_credentials()
        except Exception:
            found = None
        if not isinstance(found, Mapping):
            return None
        creds = found

    access_token = str(creds.get("accessToken") or creds.get("access_token") or "").strip()
    if not access_token:
        return None
    headers = {
        "authorization": f"Bearer {access_token}",
        "user-agent": "claude-code",
        "accept": "application/json",
    }
    payload = request_json("https://api.anthropic.com/api/oauth/usage", headers, timeout_seconds)
    if not isinstance(payload, Mapping):
        return None
    return _sanitize_claude_usage_payload(
        payload,
        now_dt,
        auth_profile=str(creds.get("source") or "") or None,
    )


def _request_json(url: str, headers: Mapping[str, str], timeout_seconds: float) -> Any:
    req = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=max(2.0, float(timeout_seconds))) as resp:
            status = getattr(resp, "status", 200)
            if status < 200 or status >= 300:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None


def _sanitize_codex_usage_payload(payload: Mapping[str, Any], now_dt: datetime) -> dict[str, Any] | None:
    snapshot: dict[str, Any] = {
        "version": 1,
        "provider": "openai-codex",
        "source": "codex-usage-endpoint",
        "status": "exact",
        "fetched_at": _iso(now_dt),
        "windows": _windows_from_codex_rate_limit(payload.get("rate_limit")),
    }
    plan_type = payload.get("plan_type")
    if isinstance(plan_type, str) and plan_type:
        snapshot["plan_type"] = plan_type
    additional_limits = []
    raw_limits = payload.get("additional_rate_limits")
    if isinstance(raw_limits, list):
        for item in raw_limits:
            if not isinstance(item, Mapping):
                continue
            windows = _windows_from_codex_rate_limit(item.get("rate_limit"))
            if not windows:
                continue
            limit_id = item.get("metered_feature") or item.get("limit_name") or "unknown"
            limit: dict[str, Any] = {"limit_id": str(limit_id), "windows": windows}
            if item.get("limit_name") is not None:
                limit["limit_name"] = str(item.get("limit_name"))
            additional_limits.append(limit)
    if additional_limits:
        snapshot["additional_limits"] = additional_limits
    return _prune_snapshot_dict(snapshot, now_dt)


def _windows_from_codex_rate_limit(rate_limit: Any) -> dict[str, Any]:
    if not isinstance(rate_limit, Mapping):
        return {}
    windows: dict[str, Any] = {}
    primary = _window_from_codex(rate_limit.get("primary_window"))
    secondary = _window_from_codex(rate_limit.get("secondary_window"))
    if primary:
        windows["five_hour"] = primary
    if secondary:
        windows["seven_day"] = secondary
    return windows


def _window_from_codex(window: Any) -> dict[str, Any] | None:
    if not isinstance(window, Mapping):
        return None
    pct = _safe_float(window.get("used_percent"))
    if pct is None:
        return None
    reset_at = _safe_float(window.get("reset_at"))
    return {
        "used_percentage": _clamp_pct(pct),
        "resets_at": _iso(datetime.fromtimestamp(reset_at, tz=timezone.utc)) if reset_at and reset_at > 0 else None,
    }


def _sanitize_claude_usage_payload(
    payload: Mapping[str, Any],
    now_dt: datetime,
    *,
    auth_profile: str | None = None,
) -> dict[str, Any] | None:
    five_hour = _window_from_claude(payload.get("five_hour"))
    seven_day = _window_from_claude(payload.get("seven_day"))
    windows: dict[str, Any] = {}
    if five_hour:
        windows["five_hour"] = five_hour
    if seven_day:
        windows["seven_day"] = seven_day

    additional_limits = []
    for limit_id, limit_name in (
        ("seven_day_opus", "Opus 7d"),
        ("seven_day_sonnet", "Sonnet 7d"),
        ("seven_day_oauth_apps", "OAuth apps 7d"),
        ("seven_day_cowork", "Cowork 7d"),
        ("seven_day_omelette", "Claude Code 7d"),
    ):
        window = _window_from_claude(payload.get(limit_id))
        if window:
            additional_limits.append({
                "limit_id": limit_id,
                "limit_name": limit_name,
                "windows": {"seven_day": window},
            })

    snapshot: dict[str, Any] = {
        "version": 1,
        "provider": "claude-cli",
        "source": "claude-oauth-usage",
        "status": "exact",
        "fetched_at": _iso(now_dt),
        "windows": windows,
    }
    if auth_profile:
        snapshot["auth_profile"] = auth_profile
    if additional_limits:
        snapshot["additional_limits"] = additional_limits
    return _prune_snapshot_dict(snapshot, now_dt)


def _window_from_claude(window: Any) -> dict[str, Any] | None:
    if not isinstance(window, Mapping):
        return None
    pct = _safe_float(window.get("utilization"))
    if pct is None:
        return None
    reset = window.get("resets_at")
    return {
        "used_percentage": _clamp_pct(pct),
        "resets_at": reset if isinstance(reset, str) and reset else None,
    }


def _prune_snapshot_dict(snapshot: dict[str, Any], now_dt: datetime) -> dict[str, Any] | None:
    now_ts = now_dt.timestamp()
    snapshot = dict(snapshot)
    snapshot["windows"] = _prune_window_dict(snapshot.get("windows"), now_ts)
    additional = []
    for limit in snapshot.get("additional_limits") or []:
        if not isinstance(limit, Mapping):
            continue
        pruned = dict(limit)
        pruned["windows"] = _prune_window_dict(limit.get("windows"), now_ts)
        if pruned["windows"]:
            additional.append(pruned)
    if additional:
        snapshot["additional_limits"] = additional
    else:
        snapshot.pop("additional_limits", None)
    if not snapshot["windows"] and not additional:
        return None
    return snapshot


def _prune_window_dict(windows: Any, now_ts: float) -> dict[str, Any]:
    if not isinstance(windows, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in ("five_hour", "seven_day"):
        window = windows.get(key)
        if not isinstance(window, Mapping):
            continue
        pct = _safe_float(window.get("used_percentage"))
        if pct is None:
            continue
        resets_at = window.get("resets_at") if isinstance(window.get("resets_at"), str) else None
        reset_ts = _parse_iso_timestamp(resets_at)
        if reset_ts is not None and reset_ts <= now_ts:
            continue
        out[key] = {"used_percentage": _clamp_pct(pct), "resets_at": resets_at}
    return out


def _quota_state_dirs(state_dirs: Iterable[str | os.PathLike[str]] | None) -> list[Path]:
    if state_dirs is not None:
        return _dedupe_paths(Path(p).expanduser() for p in state_dirs)

    paths: list[Path] = []
    env_dir = os.getenv("HERMES_NATIVE_QUOTA_STATE_DIR")
    if env_dir:
        paths.extend(Path(p).expanduser() for p in env_dir.split(os.pathsep) if p.strip())

    # Native Hermes location.  Deliberately do not read Pi runtime state: Hermes
    # quota display is standalone and snapshots are owned by this profile.
    try:
        paths.append(Path(get_hermes_home()) / "state")
    except Exception:
        pass
    return _dedupe_paths(paths)


def _native_quota_snapshot_path(provider: NativeQuotaProvider) -> Path:
    state_dirs = _quota_state_dirs(None)
    state_dir = state_dirs[0] if state_dirs else Path(get_hermes_home()) / "state"
    return state_dir / f"provider-native-quota-{provider}.json"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _freshest_snapshot_path(provider: NativeQuotaProvider, state_dirs: Iterable[str]) -> Path | None:
    prefix = f"provider-native-quota-{provider}"
    best: tuple[float, Path] | None = None
    for state_dir_raw in state_dirs:
        state_dir = Path(state_dir_raw)
        try:
            entries = list(state_dir.iterdir())
        except OSError:
            continue
        for path in entries:
            name = path.name
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue
            tail = name[len(prefix):-len(".json")]
            # Accept default file plus provider-specific fingerprinted variants.
            if tail and not tail.startswith("-"):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, path)
    return best[1] if best else None


def _read_snapshot(
    path: Path,
    provider: NativeQuotaProvider,
    now_ts: float,
    stale_after_seconds: float,
) -> NativeQuotaSummary | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("version") != 1 or raw.get("provider") != provider:
        return None

    fetched_at_value = raw.get("fetched_at")
    if not isinstance(fetched_at_value, str):
        return None
    fetched_ts = _parse_iso_timestamp(fetched_at_value)
    if fetched_ts is None:
        return None
    age_seconds = max(0.0, now_ts - fetched_ts)
    windows = _collect_windows(raw, now_ts)
    if not windows:
        return None

    return NativeQuotaSummary(
        provider=provider,
        short_label="cla" if provider == "claude-cli" else "cdx",
        fetched_at=fetched_at_value,
        age_seconds=age_seconds,
        stale=age_seconds > stale_after_seconds,
        windows=tuple(windows),
        source_path=str(path),
    )


def _collect_windows(raw: dict[str, Any], now_ts: float) -> list[NativeQuotaWindow]:
    """Pick the most constrained 5h/7d window across primary + extra limits."""
    best: dict[NativeQuotaWindowKey, NativeQuotaWindow] = {}

    def visit(container: Any) -> None:
        if not isinstance(container, dict):
            return
        windows = container.get("windows")
        if not isinstance(windows, dict):
            return
        for key in ("five_hour", "seven_day"):
            raw_window = windows.get(key)
            if not isinstance(raw_window, dict):
                continue
            pct = _safe_float(raw_window.get("used_percentage"))
            if pct is None:
                continue
            resets_at = raw_window.get("resets_at") if isinstance(raw_window.get("resets_at"), str) else None
            reset_ts = _parse_iso_timestamp(resets_at)
            # Expired windows are not actionable and are pruned so the status
            # bar does not advertise stale reset times.
            if reset_ts is not None and reset_ts <= now_ts:
                continue
            candidate = NativeQuotaWindow(key, _clamp_pct(pct), resets_at)  # type: ignore[arg-type]
            existing = best.get(key)  # type: ignore[arg-type]
            if existing is None or candidate.used_percentage > existing.used_percentage:
                best[key] = candidate  # type: ignore[index]

    visit(raw)
    additional = raw.get("additional_limits")
    if isinstance(additional, list):
        for item in additional:
            visit(item)

    return [best[key] for key in ("five_hour", "seven_day") if key in best]


def _write_snapshot_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, round(float(value), 1)))


def _parse_iso_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _codex_account_id_from_token(token: str) -> str | None:
    claims = _decode_jwt_payload(token)
    nested = claims.get("https://api.openai.com/auth")
    if isinstance(nested, Mapping):
        account_id = nested.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    account_id = claims.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def _format_pct(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def _format_reset_countdown(resets_at: str | None, now_ts: float) -> str | None:
    reset_ts = _parse_iso_timestamp(resets_at)
    if reset_ts is None:
        return None
    seconds = reset_ts - now_ts
    if seconds <= 0:
        return "↻now"
    if seconds < 60:
        return f"↻{max(1, round(seconds))}s"
    if seconds < 3600:
        return f"↻{round(seconds / 60)}m"
    if seconds < 86400:
        hours = int(seconds // 3600)
        minutes = round((seconds - hours * 3600) / 60)
        if minutes >= 60:
            hours += 1
            minutes = 0
        if hours >= 24:
            return "↻1d"
        return f"↻{hours}h{minutes}m" if minutes > 0 and hours < 10 else f"↻{hours}h"
    return f"↻{round(seconds / 86400)}d"
