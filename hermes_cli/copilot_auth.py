"""GitHub Copilot authentication utilities (credential order matches the Copilot CLI:
COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN, then ``gh auth token``)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from hermes_cli._subprocess_compat import IS_WINDOWS, windows_hide_flags

logger = logging.getLogger(__name__)

# VS Code's GitHub App client ID: mints ghu_* tokens exchangeable for Copilot API JWTs (needed for
# internal-only models / enterprise endpoints). The opencode App ID mints gho_* tokens that 404.
COPILOT_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"
_CLASSIC_PAT_PREFIX = "ghp_"  # rejected by the Copilot API (gho_ / github_pat_ / ghu_ work)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


def validate_copilot_token(token: str) -> tuple[bool, str]:
    """Validate that a token is usable with the Copilot API."""
    token = token.strip()
    if not token:
        return False, "Empty token"
    if token.startswith(_CLASSIC_PAT_PREFIX):
        return False, (
            "Classic Personal Access Tokens (ghp_*) are not supported by the "
            "Copilot API. Use one of:\n"
            "  → `copilot login` or `hermes model` to authenticate via OAuth\n"
            "  → A fine-grained PAT (github_pat_*) with Copilot Requests permission\n"
            "  → `gh auth login` with the default device code flow (produces gho_* tokens)")
    return True, "OK"


def resolve_copilot_token() -> tuple[str, str]:
    """Resolve a GitHub token suitable for Copilot API use → (token, source); ("", "") if none.

    Raises ValueError if only a classic PAT is available.
    """
    any_env_var_set = False
    for env_var in COPILOT_ENV_VARS:
        val = os.getenv(env_var, "").strip()
        if not val:
            continue
        any_env_var_set = True
        valid, msg = validate_copilot_token(val)
        if valid:
            return val, env_var
        logger.warning("Token from %s is not supported: %s", env_var, msg)
    # `gh auth token` fallback ONLY when no Copilot env var was set: an exported GITHUB_TOKEN
    # (even a classic PAT) means the user intends *that* token; skipping also avoids a slow
    # subprocess (up to 5s on Windows) on every cold start.
    if any_env_var_set:
        logger.debug("Copilot env var(s) set but none held a supported token; skipping `gh auth "
                     "token` fallback to honor explicit env-var intent (and avoid the subprocess "
                     "cost on cold start, #60800).")
        return "", ""
    token = _try_gh_cli_token()
    if token:
        valid, msg = validate_copilot_token(token)
        if not valid:
            raise ValueError(f"Token from `gh auth token` is a classic PAT (ghp_*). {msg}")
        return token, "gh auth token"
    return "", ""


def _gh_cli_candidates() -> list[str]:
    """Candidate ``gh`` binary paths, including common Homebrew installs."""
    candidates: list[str] = [c for c in (shutil.which("gh"),) if c]
    candidates += [
        c for c in ("/opt/homebrew/bin/gh", "/usr/local/bin/gh", str(Path.home() / ".local/bin/gh"))
        if c not in candidates and os.path.isfile(c) and os.access(c, os.X_OK)]
    return candidates


# ``gh auth token`` cache (misses too). With no credential store the probe blocks its full 5s on
# keyring / D-Bus, and provider inventory probes Copilot several times per request — an uncached
# miss made one settings page a 4×5s stall past Desktop's 15s IPC budget. Short TTL keeps a
# fresh ``gh auth login`` discoverable.
_GH_CLI_TOKEN_CACHE_TTL_SECONDS = 300.0
_gh_cli_token_cache: tuple[float, Optional[str]] | None = None


def _invalidate_gh_cli_token_cache() -> None:
    """Reset the ``gh auth token`` probe cache (used by tests and re-auth flows)."""
    global _gh_cli_token_cache
    _gh_cli_token_cache = None


def _try_gh_cli_token() -> Optional[str]:
    """Token from ``gh auth token`` when available; the result (incl. a miss) is cached per TTL."""
    global _gh_cli_token_cache
    now = time.monotonic()
    cache = _gh_cli_token_cache
    if cache is not None and now - cache[0] < _GH_CLI_TOKEN_CACHE_TTL_SECONDS:
        return cache[1]
    token = _probe_gh_cli_token()
    _gh_cli_token_cache = (now, token)
    return token


def _probe_gh_cli_token() -> Optional[str]:
    """Uncached ``gh auth token`` subprocess probe (see ``_try_gh_cli_token``)."""
    hostname = os.getenv("COPILOT_GH_HOST", "").strip()
    # gh must not short-circuit on GITHUB_TOKEN / GH_TOKEN, nor prompt from a backend process.
    clean_env = {k: v for k, v in os.environ.items() if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}
    clean_env.setdefault("GH_PROMPT_DISABLED", "1")
    clean_env.setdefault("GH_NO_UPDATE_NOTIFIER", "1")
    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    host_args = ["--hostname", hostname] if hostname else []
    for gh_path in _gh_cli_candidates():
        cmd = [gh_path, "auth", "token", *host_args]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=5, env=clean_env,
                                    stdin=subprocess.DEVNULL, **_popen_kwargs)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("gh CLI token lookup failed (%s): %s", gh_path, exc)
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


_DEVICE_CODE_TERMINAL_ERRORS = {"expired_token": "  ✗ Device code expired. Please try again.",
                                "access_denied": "  ✗ Authorization was denied."}


def _post_form(url: str, fields: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode(fields).encode(),
        headers={"Accept": "application/json", "User-Agent": "HermesAgent/1.0",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def copilot_device_code_login(
    *, host: str = "github.com", timeout_seconds: float = 300) -> Optional[str]:
    """Run the GitHub OAuth device code flow for Copilot."""
    domain = host.rstrip("/")
    try:
        device_data = _post_form(f"https://{domain}/login/device/code",
                                 {"client_id": COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"}, 15)
    except Exception as exc:
        logger.error("Failed to initiate device authorization: %s", exc)
        print(f"  ✗ Failed to start device authorization: {exc}")
        return None

    verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _DEVICE_CODE_POLL_INTERVAL), 1)
    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")
        return None
    print(f"\n  Open this URL in your browser: {verification_uri}\n"
          f"  Enter this code: {user_code}\n")
    print("  Waiting for authorization...", end="", flush=True)
    poll_fields = {"client_id": COPILOT_OAUTH_CLIENT_ID, "device_code": device_code,
                   "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)
        try:
            result = _post_form(f"https://{domain}/login/oauth/access_token", poll_fields, 10)
        except Exception:
            print(".", end="", flush=True)
            continue
        if result.get("access_token"):
            print(" ✓")
            return result["access_token"]
        error = result.get("error", "")
        if error == "slow_down":
            # RFC 8628: add 5 seconds to polling interval (or honor a server-supplied one)
            server_interval = result.get("interval")
            is_num = isinstance(server_interval, (int, float)) and server_interval > 0
            interval = int(server_interval) if is_num else interval + 5
        if error in ("authorization_pending", "slow_down"):
            print(".", end="", flush=True)
            continue
        if error:
            print("\n" + _DEVICE_CODE_TERMINAL_ERRORS.get(error,
                                                       f"  ✗ Authorization failed: {error}"))
            return None
    print("\n  ✗ Timed out waiting for authorization.")
    return None


# In-process cache: raw_token_fingerprint -> (api_token, expires_at_epoch, base_url).
_jwt_cache: dict[str, tuple[str, float, Optional[str]]] = {}
_JWT_REFRESH_MARGIN_SECONDS = 120  # refresh 2 min before expiry
# Exchange endpoint and headers (matching VS Code / Copilot CLI)
_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_EDITOR_VERSION = "vscode/1.104.1"
_EXCHANGE_USER_AGENT = "GitHubCopilotChat/0.26.7"

# Transient-failure hardening: gateway startup races network readiness, and a single-shot
# exchange failing there silently degrades to the RAW GitHub token, whose integrator allowlist
# omits enterprise-only models → HTTP 400 every turn until restart. Retry, and persist the last
# good JWT so a restart during a blip reuses the still-valid ~30-min token.
_EXCHANGE_MAX_ATTEMPTS = 3
_EXCHANGE_BACKOFF_BASE_SECONDS = 1.5  # sleeps ~1.5s, ~3.0s between attempts
_JWT_DISK_FILENAME = ".copilot_jwt.json"
_JWT_DISK_MAX_BYTES = 1_048_576  # 1 MiB cap on the persisted JWT store read
# Negative cache: fingerprint -> epoch until which attempts raise immediately (success clears
# it). Without it a permanently-rejected token burned ~4.5s of retry backoff on EVERY
# provider-discovery pass (/model picker, delegation spawns, dashboard).
_exchange_failure_cache: dict[str, float] = {}
# Single-flight per fingerprint: concurrent callers (dashboard polls every few seconds) wait on
# the ONE in-flight exchange instead of each spawning a hung resolver thread during a DNS outage.
_exchange_locks: dict[str, threading.Lock] = {}
_exchange_locks_guard = threading.Lock()


def _exchange_lock_for(fp: str) -> threading.Lock:
    with _exchange_locks_guard:
        lock = _exchange_locks.get(fp)
        if lock is None:
            lock = _exchange_locks[fp] = threading.Lock()
        return lock


_EXCHANGE_FAILURE_TTL_TRANSIENT_SECONDS = 60.0     # network blips: retry soon
_EXCHANGE_FAILURE_TTL_PERMANENT_SECONDS = 1800.0   # 401/403/404: won't heal
# The token itself is rejected — retrying with backoff just blocks the caller.
_EXCHANGE_PERMANENT_HTTP_STATUSES = frozenset({401, 403, 404})


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


def _read_jwt_store(path: Path) -> Optional[dict]:
    """Bounded read of the on-disk JWT store → dict, or None if missing/unusable (a store over
    the 1 MiB cap or non-dict can't balloon memory or get rewritten back out)."""
    if not path.exists():
        return None
    try:
        if path.stat().st_size > _JWT_DISK_MAX_BYTES:
            logger.debug("Persisted Copilot JWT store exceeds %d bytes; ignoring",
                         _JWT_DISK_MAX_BYTES)
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except Exception as exc:
        logger.debug("Failed to read persisted Copilot JWT store: %s", exc)
        return None


def _write_jwt_store(path: Path, store: dict) -> None:
    """Atomically write the JWT store (tmp + os.replace), best-effort 0o600."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store), encoding="utf-8")
    with contextlib.suppress(Exception):
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _jwt_disk_path() -> Optional[Path]:
    """Path to the on-disk exchanged-JWT cache (profile-aware), or None."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / _JWT_DISK_FILENAME
    except Exception:
        return None


def _with_jwt_store(verb: str, op):
    """Run ``op(path, store_or_None)`` against the disk store; failures are logged, never raised."""
    path = _jwt_disk_path()
    if not path:
        return None
    try:
        return op(path, _read_jwt_store(path))
    except Exception as exc:
        logger.debug("Failed to %s Copilot JWT: %s", verb, exc)
        return None


def evict_cached_exchanged_token(raw_token: str) -> None:
    """Drop any cached exchanged JWT for ``raw_token`` (in-process + on-disk) — the runtime
    stale-credential recovery path for ``model_not_available_for_integrator`` 400s."""
    if not raw_token:
        return
    fp = _token_fingerprint(raw_token)
    _jwt_cache.pop(fp, None)
    # Eviction = "force a fresh exchange": the negative-cache entry must go too.
    _exchange_failure_cache.pop(fp, None)

    def _evict(path, store):
        if store is not None and fp in store:
            del store[fp]
            _write_jwt_store(path, store)

    _with_jwt_store("evict cached", _evict)


def _load_jwt_from_disk(fp: str) -> Optional[tuple[str, float, Optional[str]]]:
    """Persisted exchanged JWT for ``fp`` → (api_token, expires_at, base_url), or None."""
    def _load(path, store):
        entry = (store or {}).get(fp)
        if not isinstance(entry, dict):
            return None
        api_token = entry.get("api_token", "")
        expires_at = float(entry.get("expires_at", 0) or 0)
        return (api_token, expires_at, entry.get("base_url")) if api_token and expires_at else None

    return _with_jwt_store("load persisted", _load)


def _save_jwt_to_disk(fp: str, api_token: str, expires_at: float, base_url: Optional[str]) -> None:
    """Persist an exchanged JWT (0o600), pruning expired entries."""
    def _save(path, store):
        now = time.time()
        kept = {
            k: v for k, v in (store or {}).items()
            if isinstance(v, dict) and float(v.get("expires_at", 0) or 0) > now}
        kept[fp] = {"api_token": api_token, "expires_at": expires_at, "base_url": base_url}
        _write_jwt_store(path, kept)

    _with_jwt_store("persist", _save)


# urllib's ``timeout`` only bounds socket ops AFTER DNS; getaddrinfo ignores it, so a networkless
# Windows host can hang for minutes (observed: a 17-minute event-loop stall).
_DNS_GRACE_SECONDS = 5.0


def _urlopen_bounded(req, timeout: float):
    """urlopen() on a daemon thread, abandoned after timeout + _DNS_GRACE_SECONDS so a
    DNS/getaddrinfo hang cannot block the caller. Raises the worker's exception or TimeoutError."""
    box: dict = {}
    abandoned = threading.Event()

    def _worker() -> None:
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except BaseException as exc:  # re-raised on the caller's thread
            box["exc"] = exc
            return
        if abandoned.is_set():  # caller already timed out — release the socket
            with contextlib.suppress(Exception):
                resp.close()
            return
        box["resp"] = resp

    t = threading.Thread(target=_worker, name="copilot-token-exchange", daemon=True)
    t.start()
    t.join(timeout + _DNS_GRACE_SECONDS)
    if t.is_alive():
        abandoned.set()
        raise TimeoutError("copilot token exchange exceeded hard cap of "
                           f"{timeout + _DNS_GRACE_SECONDS:.0f}s (DNS/getaddrinfo hang?)")
    if "exc" in box:
        raise box["exc"]
    if "resp" not in box:
        raise TimeoutError("copilot token exchange worker died without result")
    return box["resp"]


def _fetch_exchange_with_retry(req, timeout: float, fp: str) -> dict:
    """GET the exchange with backoff for startup network races; raises ValueError on failure.

    Permanent rejections (401/403/404) skip the retry loop. Failures populate the negative
    cache (long TTL for permanent, short for transient); success clears it.
    """
    last_exc: Optional[Exception] = None
    permanent_failure = False
    for attempt in range(1, _EXCHANGE_MAX_ATTEMPTS + 1):
        try:
            with _urlopen_bounded(req, timeout) as resp:
                data = json.loads(resp.read().decode())
            _exchange_failure_cache.pop(fp, None)
            return data
        except Exception as exc:  # noqa: BLE001 — retry all, re-raise below
            last_exc = exc
            status = getattr(exc, "code", None) or getattr(exc, "status", None)
            permanent_failure = status in _EXCHANGE_PERMANENT_HTTP_STATUSES
            if permanent_failure:
                logger.debug("Copilot token exchange rejected (HTTP %s); not retrying", status)
                break
            if attempt < _EXCHANGE_MAX_ATTEMPTS:
                sleep_s = _EXCHANGE_BACKOFF_BASE_SECONDS * attempt
                logger.debug("Copilot token exchange attempt %d/%d failed (%s); retrying in %.1fs",
                             attempt, _EXCHANGE_MAX_ATTEMPTS, exc, sleep_s)
                time.sleep(sleep_s)
    _exchange_failure_cache[fp] = time.time() + (
        _EXCHANGE_FAILURE_TTL_PERMANENT_SECONDS if permanent_failure
        else _EXCHANGE_FAILURE_TTL_TRANSIENT_SECONDS)
    raise ValueError(f"Copilot token exchange failed after {_EXCHANGE_MAX_ATTEMPTS} attempts: "
                     f"{last_exc}") from last_exc


def _cache_entry_fresh(cached) -> bool:
    return bool(cached) and time.time() < cached[1] - _JWT_REFRESH_MARGIN_SECONDS


def exchange_copilot_token(
    raw_token: str, *, timeout: float = 10.0) -> tuple[str, float, Optional[str]]:
    """Exchange a raw GitHub token for a Copilot API token → (token, expires_at, base_url).

    The token is a semicolon-separated string (not a JWT) used as a Bearer token. ``base_url``
    is the account-specific host: the exchange's ``endpoints.api`` (enterprise/proxied
    accounts), else derived from the token's ``proxy-ep``; individual accounts have neither,
    so it is None. Cached in-process until close to expiry. Raises ``ValueError`` on failure.
    """
    fp = _token_fingerprint(raw_token)
    # Fast paths outside the lock: a valid in-process JWT needs no exchange, and a recent failure
    # means queueing behind the in-flight holder (up to ~50 s) would only park an executor thread
    # to learn the same answer.
    cached = _jwt_cache.get(fp)
    if _cache_entry_fresh(cached):
        return cached
    _fail_until = _exchange_failure_cache.get(fp, 0.0)
    if time.time() < _fail_until:
        raise ValueError("Copilot token exchange recently failed; skipping re-attempt "
                         f"for another {int(_fail_until - time.time())}s")
    # Note: a waiter's own ``timeout`` is not honoured across the lock wait — by design of
    # single-flight, it observes the holder's outcome instead.
    with _exchange_lock_for(fp):
        return _exchange_copilot_token_locked(raw_token, fp, timeout=timeout)


def _exchange_copilot_token_locked(
    raw_token: str, fp: str, *, timeout: float) -> tuple[str, float, Optional[str]]:
    # Re-check in-process under the lock (a queued-behind caller may have just exchanged), then
    # on-disk: a fresh process may hold a still-valid persisted JWT, avoiding a network
    # round-trip precisely when the network is most likely flaky.
    for lookup in (_jwt_cache.get, _load_jwt_from_disk):
        cached = lookup(fp)
        if _cache_entry_fresh(cached):
            _jwt_cache[fp] = cached
            return cached
    # Negative cache: fail fast so provider discovery / picker opens don't block.
    _fail_until = _exchange_failure_cache.get(fp, 0.0)
    if time.time() < _fail_until:
        raise ValueError("Copilot token exchange recently failed; skipping re-attempt "
                         f"for another {int(_fail_until - time.time())}s")
    req = urllib.request.Request(
        _TOKEN_EXCHANGE_URL, method="GET",
        headers={"Authorization": f"token {raw_token}", "User-Agent": _EXCHANGE_USER_AGENT,
                 "Accept": "application/json", "Editor-Version": _EDITOR_VERSION})
    data = _fetch_exchange_with_retry(req, timeout, fp)
    api_token = data.get("token", "")
    if not api_token:
        raise ValueError("Copilot token exchange returned empty token")
    expires_at = float(data.get("expires_at") or 0) or time.time() + 1800
    # ``endpoints.api`` is authoritative (Copilot Enterprise / proxied accounts); else derive from
    # the token's ``proxy-ep``. Individual accounts have neither → None (registry default).
    endpoints = data.get("endpoints")
    base_url: Optional[str] = (
        str(endpoints.get("api") or "").strip().rstrip("/") if isinstance(endpoints, dict) else ""
    ) or _derive_base_url_from_proxy_ep(api_token)
    _jwt_cache[fp] = (api_token, expires_at, base_url)
    _save_jwt_to_disk(fp, api_token, expires_at, base_url)
    logger.debug("Copilot token exchanged, expires_at=%s, base_url=%s", expires_at, base_url)
    return api_token, expires_at, base_url


def _derive_base_url_from_proxy_ep(token: str) -> Optional[str]:
    """Copilot API base URL from the token's ``proxy-ep=proxy.<host>`` field (→ ``api.``)."""
    m = re.search(r'(?:^|;)\s*proxy-ep=([^;\s]+)', token)
    if not m:
        return None
    proxy_ep = re.sub(r"^https?://", "", m.group(1), count=1).rstrip("/")
    proxy_ep = re.sub(r"^proxy\.", "api.", proxy_ep, count=1)
    return f"https://{proxy_ep}"


def get_copilot_api_token(raw_token: str) -> tuple[str, Optional[str]]:
    """``(api_token, base_url)`` from the exchange, or ``(raw_token, None)`` when it fails
    (accounts that don't need exchange keep working)."""
    if not raw_token:
        return raw_token, None
    try:
        api_token, _, base_url = exchange_copilot_token(raw_token)
        return api_token, base_url
    except Exception as exc:
        logger.debug("Copilot token exchange failed, using raw token: %s", exc)
        return raw_token, None


def copilot_request_headers(
    *, is_agent_turn: bool = True, is_vision: bool = False) -> dict[str, str]:
    """Build the standard headers for Copilot API requests."""
    headers: dict[str, str] = {"Editor-Version": _EDITOR_VERSION, "User-Agent": "HermesAgent/1.0",
                               "Copilot-Integration-Id": "vscode-chat",
                               "Openai-Intent": "conversation-edits",
                               "x-initiator": "agent" if is_agent_turn else "user"}
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"
    return headers
