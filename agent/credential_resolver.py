"""CredentialResolver — resolves credential ref files to live values.

Plan 002-C: MCP Connection Pool (MCPGateway).

Security contract:
  - Ref files in users/{id}/credentials/{server}.ref contain only URIs
    pointing to the real secret source (Keychain, Secrets Manager, or env var).
  - Resolved values are cached in-process memory ONLY for the session lifetime.
  - Resolved values are NEVER written to disk.
  - Call clear() on session close to shred the in-memory cache.

Ref file format (JSON):
  {"type": "keychain",       "service": "hermes/blake/gmail", "account": "blake"}
  {"type": "env",            "var": "GMAIL_TOKEN"}
  {"type": "secrets-manager","arn": "arn:aws:secretsmanager:us-east-1:..."}
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_credentials_root

logger = logging.getLogger(__name__)


class CredentialResolver:
    """Resolve credential ref files to live secret values.

    Values are cached per (user_id, server_name) for the session lifetime.
    Thread-safe.

    Usage::

        resolver = CredentialResolver()
        token = resolver.resolve("blake", "gmail")   # None if no ref file
        resolver.clear()   # call on session close
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Optional[str]] = {}
        self._lock = threading.Lock()

    def resolve(self, user_id: str, server_name: str) -> Optional[str]:
        """Return the credential value for (user_id, server_name), or None.

        Returns None in these cases:
          - No ref file at users/{id}/credentials/{server}.ref
          - Ref file is malformed
          - Ref type is unknown
          - Underlying secret lookup failed (Keychain, Secrets Manager, env)

        All failure modes are logged at WARNING level, not raised.  A missing
        credential is NOT an error here — the gateway decides whether a missing
        credential is a blocking failure (MCPAccessDenied) or a soft pass.
        """
        cache_key = f"{user_id}:{server_name}"
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        # Resolve outside the lock so slow I/O (Keychain, network) doesn't
        # block other threads.
        value = self._load(user_id, server_name)
        with self._lock:
            self._cache[cache_key] = value
        return value

    def refresh(self, user_id: str, server_name: str) -> Optional[str]:
        """Force re-resolve, evicting any cached value first.

        Call this when an MCP tool call returns 401/Unauthorized so the
        gateway gets a fresh token before retrying.
        """
        cache_key = f"{user_id}:{server_name}"
        with self._lock:
            self._cache.pop(cache_key, None)
        return self.resolve(user_id, server_name)

    def clear(self) -> None:
        """Shred all cached credential values.

        Call on session close to ensure no values linger in memory after the
        session ends.
        """
        with self._lock:
            self._cache.clear()
        logger.debug("CredentialResolver: cache cleared")

    # ------------------------------------------------------------------
    # Private resolution methods
    # ------------------------------------------------------------------

    def _load(self, user_id: str, server_name: str) -> Optional[str]:
        """Read ref file and dispatch to the appropriate resolver."""
        creds_root = get_credentials_root(user_id)
        ref_file = creds_root / f"{server_name}.ref"

        if not ref_file.exists():
            return None

        try:
            ref = json.loads(ref_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "CredentialResolver: cannot parse ref file %s: %s", ref_file, exc
            )
            return None

        ref_type = ref.get("type", "")
        if ref_type == "keychain":
            return self._resolve_keychain(
                service=ref.get("service", ""),
                account=ref.get("account", ""),
            )
        if ref_type == "env":
            return self._resolve_env(var=ref.get("var", ""))
        if ref_type == "secrets-manager":
            return self._resolve_secrets_manager(arn=ref.get("arn", ""))

        logger.warning(
            "CredentialResolver: unknown ref type %r in %s — valid types: "
            "keychain, env, secrets-manager",
            ref_type,
            ref_file,
        )
        return None

    def _resolve_keychain(self, service: str, account: str) -> Optional[str]:
        """Resolve a macOS Keychain entry via the `security` CLI."""
        if not service:
            logger.warning("CredentialResolver: keychain ref missing 'service'")
            return None
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s", service,
                    "-a", account,
                    "-w",          # write only the password to stdout
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
            logger.debug(
                "CredentialResolver: keychain lookup failed for service=%r "
                "account=%r (returncode=%d)",
                service,
                account,
                result.returncode,
            )
            return None
        except FileNotFoundError:
            # `security` CLI not available (Linux / CI)
            logger.debug("CredentialResolver: 'security' CLI not found — not macOS?")
            return None
        except Exception as exc:
            logger.warning("CredentialResolver: keychain lookup error: %s", exc)
            return None

    def _resolve_env(self, var: str) -> Optional[str]:
        """Resolve an environment variable."""
        if not var:
            logger.warning("CredentialResolver: env ref missing 'var'")
            return None
        return os.environ.get(var) or None

    def _resolve_secrets_manager(self, arn: str) -> Optional[str]:
        """Resolve an AWS Secrets Manager secret by ARN."""
        if not arn:
            logger.warning("CredentialResolver: secrets-manager ref missing 'arn'")
            return None
        try:
            import boto3  # type: ignore
        except ImportError:
            logger.warning(
                "CredentialResolver: boto3 not installed — "
                "cannot resolve secrets-manager ARN %r",
                arn,
            )
            return None
        try:
            client = boto3.client("secretsmanager")
            resp = client.get_secret_value(SecretId=arn)
            return resp.get("SecretString") or None
        except Exception as exc:
            logger.warning(
                "CredentialResolver: SecretsManager lookup failed for ARN %r: %s",
                arn,
                exc,
            )
            return None
