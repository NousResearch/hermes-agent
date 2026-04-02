"""Credential store backed by the system keyring.

All secrets are stored under service 'hermes-cred-proxy'.
A JSON index of names is maintained at service 'hermes-cred-proxy-index',
key 'names', so that list() can enumerate them without reading every value.

Public API: set(), list(), delete()
Internal:   _get()  — used only by the substitutor, never exposed to callers.
"""

import json

import keyring
import keyring.errors

_SERVICE = "hermes-cred-proxy"
_INDEX_SERVICE = "hermes-cred-proxy-index"
_INDEX_KEY = "names"


class CredStore:
    def _read_index(self) -> list[str]:
        raw = keyring.get_password(_INDEX_SERVICE, _INDEX_KEY)
        if raw is None:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    def _write_index(self, names: list[str]) -> None:
        keyring.set_password(_INDEX_SERVICE, _INDEX_KEY, json.dumps(names))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        """Store a credential under *name*."""
        keyring.set_password(_SERVICE, name, value)
        names = self._read_index()
        if name not in names:
            names.append(name)
            self._write_index(names)

    def list(self) -> list[str]:
        """Return sorted list of stored credential names (no values)."""
        return sorted(self._read_index())

    def delete(self, name: str) -> None:
        """Remove credential *name* from the store.

        Raises KeyError if the name does not exist.
        """
        names = self._read_index()
        if name not in names:
            raise KeyError(f"Credential {name!r} not found")
        try:
            keyring.delete_password(_SERVICE, name)
        except Exception:
            pass  # Already gone from keyring; index is source of truth for existence
        names.remove(name)
        self._write_index(names)

    # ------------------------------------------------------------------
    # Internal-only access (used by substitutor — NOT part of public API)
    # ------------------------------------------------------------------

    def _get(self, name: str) -> str:
        """Return the value for *name*.

        Intentionally private: agent processes must not be able to call
        this through any public interface.  Raises KeyError if not found.
        """
        value = keyring.get_password(_SERVICE, name)
        if value is None:
            raise KeyError(f"Credential {name!r} not found")
        return value
