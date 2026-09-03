"""Mint a provider API key by running a command (``key_cmd``).

Static API keys are the exception at enterprise gateways: SSO/OIDC brokers,
cloud IAM, and internal auth proxies all issue SHORT-LIVED bearers instead.
A key copied into ``.env`` (``key_env``) is stale within the hour, so every
request after that 401s and the user has to restart the session.

``key_cmd`` names a command that PRINTS a token, so the credential is derived
rather than stored::

    providers:
      my-gateway:
        base_url: https://gateway.internal.example.com/v1
        api_mode: chat_completions
        key_cmd: my-auth-cli print-token --profile prod

This is the established pattern for agent tooling — Claude Code's
``apiKeyHelper``, the ``gcloud auth print-access-token`` / ``aws ecr
get-login-password`` idiom, and vendor helpers such as ``databricks auth
token`` all expose exactly this contract. Hermes already accepts a callable
API key on both wire clients (the Entra ID / Azure identity path) and invokes
it per request, so nothing downstream changes: the token is simply always
fresh. It is cached until shortly before expiry, so the command runs about
once per token lifetime rather than once per request.

Output contract: print ONLY the token on stdout, either bare or as JSON with
an ``access_token`` field (``expires_in`` is honoured when present) — the
shape OAuth 2.0 token endpoints and the helpers above already emit.

Precedence: an explicit ``--api-key`` still wins (the one-off recovery escape
hatch); otherwise ``key_cmd`` is preferred over a static ``api_key`` /
``key_env`` on the same entry.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Treat a cached token as spent slightly before its stated expiry, so a request
# can't be signed with a token that dies in flight. 60s matches the leeway used
# by comparable OAuth token caches.
_TOKEN_REFRESH_LEEWAY_SECONDS = 60.0
# A token helper reads a local credential cache and should answer in
# milliseconds; anything approaching this budget is hung, not slow.
_MINT_TIMEOUT_SECONDS = 15
# When a helper advertises NO expiry, the token cannot be cached for the life
# of the process: nothing in the request path re-mints on 401 (the SDK retries
# 429/5xx only), so an expired no-TTL token would 401 every request until
# restart. Re-mint on a bounded window instead — the helper answers from a
# local credential cache in milliseconds, so a periodic re-run is cheap, and a
# helper that wants a longer cache can simply advertise its real expiry.
_NO_TTL_REFRESH_SECONDS = 900.0


class CommandTokenError(RuntimeError):
    """A ``key_cmd`` failed to produce a usable token."""


def _mint(command: str, label: str) -> tuple[str, Optional[float]]:
    """Run *command*, returning ``(token, ttl_seconds_or_None)``."""
    # A configured helper is an external child process, not part of the
    # trusted Hermes process. Give it the credential-scoped environment for
    # non-terminal subprocesses. Import lazily to avoid adding environment
    # machinery to the static-provider import path.
    try:
        from tools.environments.local import hermes_subprocess_env

        child_env = hermes_subprocess_env(inherit_credentials=False)
    except Exception as exc:
        # Never fall back to implicit full-environment inheritance when the
        # credential-scoped environment cannot be constructed.
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not prepare a "
            "credential-scoped environment"
        ) from exc

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_MINT_TIMEOUT_SECONDS,
            env=child_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} timed out after "
            f"{_MINT_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise CommandTokenError(
            f"key_cmd for provider {label!r} could not be executed: {exc}"
        ) from exc

    if completed.returncode != 0:
        # NEVER include stdout/stderr: a partially-successful auth helper can
        # print a token or refresh secret there. The command STRING is also
        # withheld — a key_cmd can legitimately embed a secret
        # (`print-token --client-secret=…`), so echoing it back would leak the
        # very credential this module exists to protect. Name the provider so
        # the user knows which config entry to run by hand.
        raise CommandTokenError(
            f"key_cmd for provider {label!r} exited {completed.returncode}. "
            f"Run that provider's key_cmd manually to see why "
            f"(e.g. `databricks auth login` if its OAuth session expired)."
        )

    stdout = completed.stdout or ""
    if not stdout.strip():
        raise CommandTokenError(f"key_cmd for provider {label!r} produced no output")

    # JSON payload — the shape `databricks auth token --output json` prints.
    # Token extraction mirrors databricks/ucode's get_databricks_token:
    #   json.loads(result.stdout or "{}").get("access_token", "")
    if stdout.lstrip().startswith("{"):
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            token = str(payload.get("access_token") or "").strip()
            if not token:
                raise CommandTokenError(
                    f"key_cmd for provider {label!r} returned JSON without an "
                    "'access_token' field"
                )
            ttl = payload.get("expires_in")
            if isinstance(ttl, (int, float)) and ttl > 0:
                return token, float(ttl)
            # A relative lifetime is the OAuth 2.0 field, but CLI token helpers
            # commonly print an absolute ISO 8601 deadline instead. Treating
            # that as "no TTL advertised" caches the token for the life of the
            # process, so every request 401s once the deadline passes.
            # Imported lazily: hermes_cli.auth imports from agent.* at module
            # level, so a top-level import here would risk a cycle.
            from hermes_cli.auth import _parse_iso_timestamp

            for field in ("expiry", "expiresOn"):
                deadline = _parse_iso_timestamp(payload.get(field))
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining > 0:
                        return token, remaining
            return token, None

    # Bare token. The contract every comparable helper documents is "stdout
    # carries the token and nothing else" — extra output would be consumed as
    # part of the credential. Strip surrounding whitespace and take the rest
    # verbatim; do NOT silently keep one line of several, which converts a
    # misconfigured helper (banner, warning, two tokens) into a corrupt-key 401
    # that is far harder to diagnose than an explicit refusal.
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
            self._expires_at = (
                time.monotonic() + max(ttl - _TOKEN_REFRESH_LEEWAY_SECONDS, 5.0)
                if ttl
                # No advertised TTL: bounded cache (see _NO_TTL_REFRESH_SECONDS)
                # — there is no 401-driven re-mint hook to fall back on.
                else time.monotonic() + _NO_TTL_REFRESH_SECONDS
            )
            logger.debug(
                "Minted key_cmd token for provider %s (ttl=%s)",
                self._label, f"{int(ttl)}s" if ttl else "unknown",
            )
            return token


def build_command_token_provider(
    key_cmd: str,
    provider_label: str = "custom",
) -> Optional[Callable[[], str]]:
    """A per-request token provider for *key_cmd*, or ``None`` when unset."""
    command = str(key_cmd or "").strip()
    if not command:
        return None
    return CommandTokenSource(command, provider_label)
