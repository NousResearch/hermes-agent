"""Mint a provider API key by running a command (``key_cmd``).

Enterprise gateways (SSO/OIDC brokers, cloud IAM, auth proxies) issue SHORT-LIVED bearers; a key
copied into ``.env`` goes stale within the hour. ``key_cmd`` names a command that PRINTS a token
(the ``apiKeyHelper`` / ``gcloud auth print-access-token`` idiom). Both wire clients accept a
callable API key and invoke it per request; the token is cached until shortly before expiry.
Output contract: ONLY the token on stdout, bare or as JSON with an ``access_token`` field
(``expires_in`` / ISO ``expiry`` honoured). Precedence: explicit ``--api-key`` wins (one-off
recovery escape hatch); otherwise ``key_cmd`` beats a static ``api_key`` / ``key_env``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Treat a token as spent slightly before expiry so a request can't be signed with one that dies in
# flight (60s = usual OAuth cache leeway).
_TOKEN_REFRESH_LEEWAY_SECONDS = 60.0
# Helpers answer from a local cache in milliseconds; this long means hung.
_MINT_TIMEOUT_SECONDS = 15
# No advertised expiry: nothing in the request path re-mints on 401 (the SDK retries 429/5xx only), so
# a process-lifetime cache would 401 forever once the token died. Re-mint on a bounded window instead.
_NO_TTL_REFRESH_SECONDS = 900.0


class CommandTokenError(RuntimeError):
    """A ``key_cmd`` failed to produce a usable token."""


def materialize_probe_api_key(api_key: object) -> str:
    """Best-effort probe credential; never send a callable's repr or log mint errors."""
    try:
        token = api_key() if callable(api_key) else api_key
    except Exception:
        return ""
    return token.strip() if isinstance(token, str) else ""


def normalize_key_cmd_env(value: object) -> list[str]:
    """Declared ``key_cmd_env`` names: the Tier 1 entries one helper may still see."""
    items = [value] if isinstance(value, str) else (value if isinstance(value, (list, tuple)) else [])
    return [name.strip() for name in items if isinstance(name, str) and name.strip()]


def _mint(command: str, label: str, key_cmd_env: object = None) -> tuple[str, Optional[float]]:
    """Run *command*, returning ``(token, ttl_seconds_or_None)``. The helper runs FOR the profile whose
    provider is being minted: it gets that profile's own env (secrets + HERMES_HOME), never the multiplexer's
    launch environ — an ``op read`` / ``vault kv get`` helper must sign in as the served profile. The served
    profile's ``.env`` overlay re-adds Tier 1 adapter secrets with no second pass, so scrub again before
    spawning (#126902, the openviking double scrub from #125755); only names the provider declares in
    ``key_cmd_env`` (e.g. ``GH_TOKEN`` for a ``gh auth token``-style helper) are restored."""
    try:
        from tools.environments.local import hermes_subprocess_env, served_profile_child_env

        served = served_profile_child_env(inherit_credentials=True)
        child_env = hermes_subprocess_env(inherit_credentials=True, base_env=served)
        wanted = normalize_key_cmd_env(key_cmd_env)
        if wanted:
            by_fold = {}
            for key, val in served.items():
                by_fold.setdefault(key.upper(), (key, val))
            for name in wanted:
                if name.upper() in by_fold:
                    key, val = by_fold[name.upper()]
                    child_env[key] = val
    except Exception as exc:
        # Never fall back to implicit full-environment inheritance when the
        # credential-scoped environment cannot be constructed.
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not prepare a credential-scoped environment"
        ) from exc

    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, errors="replace", timeout=_MINT_TIMEOUT_SECONDS,
            env=child_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} timed out after {_MINT_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise CommandTokenError(f"key_cmd for provider {label!r} could not be executed: {exc}") from exc

    if completed.returncode != 0:
        # NEVER include stdout/stderr (may hold a token) or the command string (may embed
        # `--client-secret=…`); name the provider instead.
        raise CommandTokenError(
            f"key_cmd for provider {label!r} exited {completed.returncode}. "
            f"Run that provider's key_cmd manually to see why "
            f"(e.g. `databricks auth login` if its OAuth session expired)."
        )

    stdout = completed.stdout or ""
    if not stdout.strip():
        raise CommandTokenError(f"key_cmd for provider {label!r} produced no output")

    # JSON payload — the shape `databricks auth token --output json` prints.
    if stdout.lstrip().startswith("{"):
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            token = str(payload.get("access_token") or "").strip()
            if not token:
                raise CommandTokenError(
                    f"key_cmd for provider {label!r} returned JSON without an 'access_token' field"
                )
            ttl = payload.get("expires_in")
            if isinstance(ttl, (int, float)) and ttl > 0:
                return token, float(ttl)
            # CLI helpers often print an absolute ISO 8601 deadline instead of OAuth's relative
            # lifetime; honour it or the token 401s once past. Lazy import: hermes_cli.auth imports agent.*.
            from hermes_cli.auth import _parse_iso_timestamp

            for field in ("expiry", "expiresOn"):
                deadline = _parse_iso_timestamp(payload.get(field))
                remaining = deadline - time.time() if deadline is not None else 0
                if remaining > 0:
                    return token, remaining
            return token, None

    # Bare token: stdout carries the token and nothing else. Do NOT keep one line of several — that
    # turns a misconfigured helper (banner, warning) into a corrupt-key 401 far harder to diagnose.
    token = stdout.strip()
    if "\n" in token:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} printed multiple lines; it must "
            "print only the token (or JSON with an 'access_token' field)"
        )
    return token, None


class CommandTokenSource:
    """Callable returning a bearer token, cached until shortly before expiry."""

    def __init__(self, command: str, label: str = "custom", key_cmd_env: object = None) -> None:
        self._command = command
        self._label = label or "custom"
        self._key_cmd_env = normalize_key_cmd_env(key_cmd_env)
        self._lock = threading.Lock()
        self._token = ""
        self._expires_at: float = 0.0

    @property
    def cache_identity(self) -> str:
        """Stable catalog identity; token rotation must not mint on cache reads."""
        return f"cmd:{self._command}"

    def __call__(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            token, ttl = _mint(self._command, self._label, self._key_cmd_env)
            self._token = token
            self._expires_at = time.monotonic() + (
                max(ttl - _TOKEN_REFRESH_LEEWAY_SECONDS, 5.0) if ttl else _NO_TTL_REFRESH_SECONDS
            )
            logger.debug(
                "Minted key_cmd token for provider %s (ttl=%s)",
                self._label, f"{int(ttl)}s" if ttl else "unknown",
            )
            return token


def build_command_token_provider(key_cmd: str, provider_label: str = "custom",
                                   key_cmd_env: object = None) -> Optional[CommandTokenSource]:
    """A per-request token provider for *key_cmd*, or ``None`` when unset."""
    command = str(key_cmd or "").strip()
    return CommandTokenSource(command, provider_label, key_cmd_env) if command else None
