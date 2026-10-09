"""Application-supplied authentication scope and policy; no configuration I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hermes_constants import hermes_home_key


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be an explicit non-empty identifier")


@dataclass(frozen=True)
class CredentialScope:
    """Capture the owning home, not the launch environment or a profile label.

    execution_id optionally distinguishes requests/leases within one home.
    This value does not install a secret scope or replace existing scope binding.
    """

    profile_home: Path = field(compare=False)
    execution_id: str | None = None
    home_key: str = field(init=False)

    def __post_init__(self) -> None:
        home = Path(self.profile_home)
        if not home.is_absolute():
            raise ValueError("profile_home must be an explicit absolute path")
        if self.execution_id is not None:
            _require_identifier(self.execution_id, "execution_id")
        key = hermes_home_key(home)
        object.__setattr__(self, "profile_home", Path(key))
        object.__setattr__(self, "home_key", key)


@dataclass(frozen=True)
class AuthSettings:
    """Already-resolved policy for one provider in one profile.

    The existing configuration owner supplies defaults and validates strategy
    names. Authentication must not load config.yaml or duplicate its parser.
    """

    adopt_external_logins: bool
    pool_strategy: str

    def __post_init__(self) -> None:
        if not isinstance(self.adopt_external_logins, bool):
            raise ValueError("adopt_external_logins must be a resolved boolean")
        _require_identifier(self.pool_strategy, "pool_strategy")


@dataclass(frozen=True)
class AuthContext:
    """Per-operation settings and a scope-aware application secret reader.

    Construct again when settings change; do not install a process-wide
    startup snapshot. read_secret receives the explicit scope even on a miss:
    the application must use the existing secret-scope infrastructure and
    must not fall back to another profile's process environment.
    Config/.env writes remain with their existing owner in the storage cut.
    """

    scope: CredentialScope
    settings: AuthSettings
    read_secret: Callable[[CredentialScope, str], str | None] = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.scope, CredentialScope):
            raise TypeError("scope must be a CredentialScope")
        if not isinstance(self.settings, AuthSettings):
            raise TypeError("settings must be AuthSettings")
        if not callable(self.read_secret):
            raise TypeError("read_secret must be a scope-aware callable")
