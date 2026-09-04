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
from typing import Callable, Optional

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


def _mint(command: str, label: str) -> tuple[str, Optional[float]]:
    """Run *command*, returning ``(token, ttl_seconds_or_None)``."""
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=_MINT_TIMEOUT_SECONDS,
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

    def __init__(self, command: str, label: str = "custom") -> None:
        self._command = command
        self._label = label or "custom"
        self._lock = threading.Lock()
        self._token = ""
        self._expires_at: float = 0.0

    def __call__(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._expires_at:
                return self._token
            token, ttl = _mint(self._command, self._label)
            self._token = token
            self._expires_at = time.monotonic() + (
                max(ttl - _TOKEN_REFRESH_LEEWAY_SECONDS, 5.0) if ttl else _NO_TTL_REFRESH_SECONDS
            )
            logger.debug(
                "Minted key_cmd token for provider %s (ttl=%s)",
                self._label, f"{int(ttl)}s" if ttl else "unknown",
            )
            return token


def build_command_token_provider(key_cmd: str, provider_label: str = "custom") -> Optional[Callable[[], str]]:
    """A per-request token provider for *key_cmd*, or ``None`` when unset."""
    command = str(key_cmd or "").strip()
    return CommandTokenSource(command, provider_label) if command else None
