"""Selected-state-store boundary for legacy SQLite maintenance operations.

Maintenance commands historically operate on ``state.db`` files.  This facade
resolves the selected store before callers inspect or open those files, and
makes every unported PostgreSQL operation an explicit refusal instead of a
SQLite fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from state_store import StateStoreConfigurationError, resolve_state_store_config
from state_store_runtime_readiness import inspect_runtime_activation


@dataclass(frozen=True)
class StateStoreMaintenanceOperations:
    """Backend-selected maintenance capability registry.

    Only SQLite file maintenance is implemented today.  PostgreSQL operations
    must be added here with a native implementation before a command may route
    to it; a capability name alone never authorizes a fallback.
    """

    selected_backend: str
    profile_home: Path
    profile_name: str | None = None
    tenant_schema: str | None = None
    _sqlite_capabilities = frozenset({
        "sessions-repair", "sessions-recover", "sessions-import", "sessions-repair-profiles",
        "backup", "backup-quick", "archive-import", "snapshot-list", "snapshot-create", "snapshot-restore",
        "snapshot-prune", "update-snapshot", "claw-snapshot", "approvals-suggest", "doctor",
    })

    _postgresql_capabilities = frozenset({"sessions-prune", "sessions-archive", "sessions-clean-markers", "sessions-optimize"})

    @classmethod
    def resolve(
        cls, config: Mapping[str, Any] | None = None, *, home: Path | None = None,
    ) -> "StateStoreMaintenanceOperations":
        try:
            report = inspect_runtime_activation(config, home=home or _current_home(), include_inventory=False)
        except StateStoreConfigurationError as exc:
            # Preserve legacy SQLite maintenance for malformed unrelated config
            # fixtures.  A config that actually declares state_store is an
            # attempted backend selection and must fail closed instead.
            if _declares_state_store(config, home or _current_home()):
                raise StateStoreMaintenanceConfigurationError(
                    "PostgreSQL state-store configuration could not be resolved safely; "
                    f"no SQLite fallback is permitted: {exc}"
                ) from exc
            return cls("sqlite", (home or _current_home()).expanduser().resolve())
        return cls(report.selected_backend, Path(report.profile_home), report.profile_name, report.tenant_schema)

    def require(self, capability: str) -> None:
        supported = self._sqlite_capabilities if self.selected_backend == "sqlite" else self._postgresql_capabilities
        if capability not in supported:
            if capability in self._sqlite_capabilities or capability in self._postgresql_capabilities:
                raise StateStoreMaintenanceCapabilityError(capability, self.profile_home)
            raise ValueError(f"Unknown state-store maintenance capability: {capability}")

    def doctor(self, config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Observe the selected PostgreSQL tenant without SQLite access or mutation."""
        return self._postgresql_operations(config).doctor()

    def logical_backup(
        self, output_directory: Path, *, quiesced: bool, config: Mapping[str, Any] | None = None,
    ) -> Any:
        """Create and verify a schema-only PG logical backup via the native primitive."""
        output_directory = output_directory.expanduser().resolve()
        if not output_directory.is_dir():
            raise StateStoreMaintenanceConfigurationError(
                "PostgreSQL logical-backup requires an existing user-selected output directory"
            )
        if hasattr(os, "geteuid") and output_directory.stat().st_uid != os.geteuid():
            raise StateStoreMaintenanceConfigurationError(
                "PostgreSQL logical-backup output directory must be owned by the active user"
            )
        operations = self._postgresql_operations(config)
        backup = operations.backup(output_directory, quiesced=quiesced)
        verification = operations.restore_and_verify(backup.backup_directory)
        return backup, verification

    def optimize(self, config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Run the selected tenant's bounded native PostgreSQL maintenance pass."""
        return self._postgresql_operations(config).optimize()

    def _postgresql_operations(self, config: Mapping[str, Any] | None) -> Any:
        if self.selected_backend != "postgresql" or not self.tenant_schema:
            raise StateStoreMaintenanceCapabilityError("state-store doctor", self.profile_home)
        try:
            from state_store import _profile_secret_lookup
            from postgresql_state_store_operations import PostgreSQLSandboxOperations

            from state_store_runtime_readiness import _read_profile_config

            raw_config = config if config is not None else _read_profile_config(self.profile_home)
            resolved = resolve_state_store_config(
                raw_config, secret_lookup=lambda name: _profile_secret_lookup(self.profile_home, name),
            )
            if resolved.backend != "postgresql" or resolved.postgresql is None:
                raise StateStoreMaintenanceConfigurationError("PostgreSQL state-store selection changed during maintenance")
            dsn = _profile_secret_lookup(self.profile_home, resolved.postgresql.dsn_env)
            if not str(dsn or "").strip():
                raise StateStoreMaintenanceConfigurationError("PostgreSQL state-store secret is unavailable; no SQLite fallback is permitted")
            return PostgreSQLSandboxOperations(
                resolved.postgresql, str(dsn), schema=self.tenant_schema,
                profile_identity={"home": str(self.profile_home), "name": str(self.profile_name or "default")},
            )
        except StateStoreMaintenanceError:
            raise
        except Exception as exc:
            raise StateStoreMaintenanceConfigurationError(
                "PostgreSQL state-store maintenance could not resolve the active trusted tenant; "
                "no SQLite fallback is permitted"
            ) from exc


class StateStoreMaintenanceError(RuntimeError):
    """Maintenance cannot safely use the selected state store."""


class StateStoreMaintenanceConfigurationError(StateStoreMaintenanceError):
    """Selection itself was invalid, so falling through to SQLite is unsafe."""


class StateStoreMaintenanceCapabilityError(StateStoreMaintenanceError):
    """The selected PostgreSQL store lacks this native maintenance capability."""

    def __init__(self, capability: str, home: Path) -> None:
        self.capability = capability
        self.home = home
        command = capability.replace("-", " ")
        super().__init__(
            f"PostgreSQL state store does not support {command} yet; "
            "no SQLite fallback is permitted."
        )


def require_state_store_maintenance(
    capability: str, config: Mapping[str, Any] | None = None, *, home: Path | None = None,
) -> StateStoreMaintenanceOperations:
    """Resolve selection and refuse unported maintenance before ``state.db`` access."""
    operations = StateStoreMaintenanceOperations.resolve(config, home=home)
    operations.require(capability)
    return operations


def _current_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _declares_state_store(config: Mapping[str, Any] | None, home: Path) -> bool:
    if isinstance(config, Mapping):
        return "state_store" in config
    try:
        import yaml
    except ImportError:
        return False
    try:
        raw = yaml.safe_load((home.expanduser() / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return False
    return isinstance(raw, Mapping) and "state_store" in raw
