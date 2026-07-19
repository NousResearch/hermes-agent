"""Immutable, secret-safe state-store specifications."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Optional


StateStoreBackend = Literal["sqlite", "postgres"]


@dataclass(frozen=True)
class StateStoreSpec:
    """Resolved state-store identity for one Hermes home/profile.

    ``postgres_dsn_env`` is the name of an environment variable, never its
    value. This makes the spec safe to include in diagnostics and cache keys.
    """

    home: Path
    profile: str
    backend: StateStoreBackend
    sqlite_path: Path
    postgres_dsn_env: str
    postgres_schema: Optional[str]
    read_only: bool

    @property
    def store_key(self) -> str:
        """Stable, non-secret key for store-local caches and diagnostics."""
        identity = "\0".join(
            (
                str(self.home),
                self.profile,
                self.backend,
                str(self.sqlite_path),
                self.postgres_dsn_env,
                self.postgres_schema or "",
            )
        )
        fingerprint = sha256(identity.encode("utf-8")).hexdigest()[:24]
        return f"{self.backend}:{fingerprint}"
