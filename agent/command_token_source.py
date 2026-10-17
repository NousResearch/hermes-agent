"""Mint a provider API key by running a command (``key_cmd``).

Enterprise gateways (SSO/OIDC brokers, cloud IAM, auth proxies) issue SHORT-LIVED bearers; a key
copied into ``.env`` goes stale within the hour. ``key_cmd`` names a command that PRINTS a token
(the ``apiKeyHelper`` / ``gcloud auth print-access-token`` idiom). Both wire clients accept a
callable API key and invoke it per request; the token is cached until shortly before expiry.
Output contract: ONLY the token on stdout, bare or as JSON with an ``access_token`` field
(the shortest of ``expires_in`` / ISO ``expiry`` honoured). Precedence: explicit ``--api-key`` wins (one-off
recovery escape hatch); otherwise ``key_cmd`` beats a static ``api_key`` / ``key_env``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import Optional

from agent.credential_source import KEY_CMD

logger = logging.getLogger(__name__)

# Treat a token as spent slightly before expiry so a request can't be signed with one that dies in
# flight (60s = usual OAuth cache leeway).
_TOKEN_REFRESH_LEEWAY_SECONDS = 60.0
# Helpers answer from a local cache in milliseconds; this long means hung.
_MINT_TIMEOUT_SECONDS = 15
# No advertised expiry: a process-lifetime cache would 401 forever once the token died (a pool-less
# provider gets one forced re-mint per token, see remint_after_auth_failure). Re-mint on a bounded window.
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


def _mint(command: str, label: str) -> tuple[str, Optional[float]]:
    """Run *command*, returning ``(token, ttl_seconds_or_None)``. The helper runs FOR the profile whose
    provider is being minted: it gets that profile's own env (secrets + HERMES_HOME), never the multiplexer's
    launch environ — an ``op read`` / ``vault kv get`` helper must sign in as the served profile."""
    from tools.environments.local import served_profile_child_env

    try:
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, shell=True, capture_output=True, text=True, errors="replace", timeout=_MINT_TIMEOUT_SECONDS,
            env=served_profile_child_env(inherit_credentials=True),
            check=False,
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
            # CLI helpers often print an absolute ISO 8601 deadline instead of (or beside) OAuth's
            # relative lifetime. Honour the SHORTEST: a helper answering from its own cache reprints the
            # original ``expires_in`` while the deadline counts down. Lazy import: hermes_cli.auth imports agent.*.
            from hermes_cli.auth import _parse_iso_timestamp

            relative = payload.get("expires_in")
            lifetimes = [float(relative)] if isinstance(relative, (int, float)) else []
            for field in ("expiry", "expiresOn"):
                deadline = _parse_iso_timestamp(payload.get(field))
                if deadline is not None:
                    lifetimes.append(deadline - time.time())
            usable = [ttl for ttl in lifetimes if ttl > 0]
            return token, (min(usable) if usable else None)

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

    hermes_credential_source = KEY_CMD  # display label source; read without calling (agent.credential_source)

    def __init__(self, command: str, label: str = "custom") -> None:
        self._command = command
        self._label = label or "custom"
        self._lock = threading.Lock()
        self._token = ""
        self._expires_at: float = 0.0
        self._remint_pending = False  # the next mint answers a rejection
        self._reminted = False  # the cached token answered a rejection

    @property
    def cache_identity(self) -> str:
        """Stable catalog identity; token rotation must not mint on cache reads."""
        return f"cmd:{self._command}"

    def __call__(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            token, ttl = _mint(self._command, self._label)
            self._token = token
            self._reminted, self._remint_pending = self._remint_pending, False
            self._expires_at = time.monotonic() + (
                max(ttl - _TOKEN_REFRESH_LEEWAY_SECONDS, 5.0) if ttl else _NO_TTL_REFRESH_SECONDS
            )
            logger.debug(
                "Minted key_cmd token for provider %s (ttl=%s)",
                self._label, f"{int(ttl)}s" if ttl else "unknown",
            )
            return token

    def invalidate(self) -> bool:
        """Drop a live cached token the provider rejected, so the next call re-runs the helper.

        The advertised TTL can outlive the credential (a cached helper reprints a stale lifetime; a
        broker can revoke early), so a rejection is the authoritative staleness signal. Refused for a
        token that itself answered a rejection: a revoked credential mints the same dead token, and
        retrying it would spin. The next scheduled mint re-arms recovery. Also refused when nothing
        live is cached (the next call mints anyway, and a failed mint will fail again)."""
        with self._lock:
            if self._reminted or not self._token or time.monotonic() >= self._expires_at:
                return False
            self._token, self._expires_at, self._remint_pending = "", 0.0, True
            return True


def remint_after_auth_failure(api_key: object, status_code: Optional[int], classified_reason: object) -> bool:
    """Auth recovery for a provider without a credential pool: True when a rejected ``key_cmd``
    token was dropped so the retry re-mints (#91119). A bare status counts only as 401; 403 is
    often quota or permission, which a new token cannot fix."""
    from agent.error_classifier import FailoverReason

    if classified_reason is None:
        is_auth = status_code == 401
    else:
        is_auth = classified_reason in (FailoverReason.auth, FailoverReason.auth_permanent)
    if not (is_auth and isinstance(api_key, CommandTokenSource) and api_key.invalidate()):
        return False
    logger.info("Auth failure on key_cmd provider %s: dropped the cached token, re-minting for one retry", api_key._label)
    return True


def build_command_token_provider(key_cmd: str, provider_label: str = "custom") -> Optional[CommandTokenSource]:
    """A per-request token provider for *key_cmd*, or ``None`` when unset."""
    command = str(key_cmd or "").strip()
    return CommandTokenSource(command, provider_label) if command else None
