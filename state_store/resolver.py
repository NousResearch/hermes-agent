"""Profile-explicit state-store resolution.

The resolver deliberately does not read Hermes ambient profile state or load
configuration. Callers provide the home and already-resolved config they want
to use, which keeps profile selection deterministic.
"""

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any, Optional

from state_store.spec import StateStoreSpec

_POSTGRES_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class StateStoreConfigurationError(ValueError):
    """Raised for invalid state-store settings without echoing their values."""


class StateStoreBackendNotActivatedError(RuntimeError):
    """Raised when a configured backend has no production implementation yet."""


def _profile_for_home(home: Path) -> str:
    if home.parent.name == "profiles":
        return home.name
    return "default"


def _optional_mapping(
    parent: Mapping[str, Any], key: str, *, setting: str
) -> Mapping[str, Any]:
    if key not in parent:
        return {}
    value = parent[key]
    if not isinstance(value, Mapping):
        raise StateStoreConfigurationError(f"{setting} must be a mapping")
    return value


def _string_setting(value: Any, *, name: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise StateStoreConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _env_var_name(value: Any) -> str:
    name = _string_setting(
        value,
        name="sessions.state.postgres.dsn_env",
        default="HERMES_STATE_POSTGRES_DSN",
    )
    if not (name[0].isalpha() or name[0] == "_") or not all(
        char.isalnum() or char == "_" for char in name
    ):
        raise StateStoreConfigurationError(
            "sessions.state.postgres.dsn_env must name an environment variable"
        )
    return name


def _postgres_schema(value: Any, *, backend: str) -> Optional[str]:
    if value is None:
        if backend == "postgres":
            raise StateStoreConfigurationError(
                "sessions.state.postgres.schema must be explicitly configured for PostgreSQL"
            )
        return None
    if not isinstance(value, str):
        raise StateStoreConfigurationError(
            "sessions.state.postgres.schema must be a PostgreSQL identifier"
        )
    schema = value.strip()
    if not schema or not _POSTGRES_IDENTIFIER_RE.fullmatch(schema):
        raise StateStoreConfigurationError(
            "sessions.state.postgres.schema must be a lowercase PostgreSQL identifier"
        )
    return schema


def resolve_state_store(
    home: Path,
    config: Optional[Mapping[str, Any]] = None,
    *,
    profile: Optional[str] = None,
    read_only: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> StateStoreSpec:
    """Resolve a state-store spec without consulting ambient profile state.

    ``environ`` is accepted for the future backend factory API. This
    behavior-neutral resolver intentionally records only the configured DSN
    environment-variable *name* and never reads its secret value.
    """

    del environ
    resolved_home = Path(home)
    if config is None:
        root: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        root = config
    else:
        raise StateStoreConfigurationError("state-store config must be a mapping")

    sessions = _optional_mapping(root, "sessions", setting="sessions")
    state = _optional_mapping(sessions, "state", setting="sessions.state")
    backend = _string_setting(
        state.get("backend"), name="sessions.state.backend", default="sqlite"
    ).lower()
    if backend not in {"sqlite", "postgres"}:
        raise StateStoreConfigurationError(
            "sessions.state.backend must be 'sqlite' or 'postgres'"
        )

    sqlite_path_setting = _string_setting(
        state.get("sqlite_path"),
        name="sessions.state.sqlite_path",
        default="state.db",
    )
    sqlite_path = Path(sqlite_path_setting).expanduser()
    if not sqlite_path.is_absolute():
        sqlite_path = resolved_home / sqlite_path

    resolved_profile = profile or _profile_for_home(resolved_home)
    if not isinstance(resolved_profile, str) or not resolved_profile.strip():
        raise StateStoreConfigurationError("state-store profile must be non-empty")
    resolved_profile = resolved_profile.strip()

    postgres = _optional_mapping(
        state, "postgres", setting="sessions.state.postgres"
    )
    postgres_dsn_env = _env_var_name(postgres.get("dsn_env"))
    postgres_schema = _postgres_schema(
        postgres.get("schema"),
        backend=backend,
    )

    return StateStoreSpec(
        home=resolved_home,
        profile=resolved_profile,
        backend=backend,
        sqlite_path=sqlite_path,
        postgres_dsn_env=postgres_dsn_env,
        postgres_schema=postgres_schema,
        read_only=bool(read_only),
    )
