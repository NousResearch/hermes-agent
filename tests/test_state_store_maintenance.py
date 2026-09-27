"""Tenant-capability boundaries for PostgreSQL maintenance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_manually_constructed_maintenance_rejects_raw_tenant_before_operations_or_sql(monkeypatch, tmp_path):
    """A matching raw identifier cannot retarget maintenance to another tenant."""
    import state_store_maintenance as maintenance

    raw_tenant = "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"
    operations = maintenance.StateStoreMaintenanceOperations(
        "postgresql", tmp_path, "default", raw_tenant,
    )
    reached: list[str] = []

    def unexpected_config_resolution(*_args, **_kwargs):
        reached.append("configuration")
        raise AssertionError("raw schema reached configuration")

    monkeypatch.setattr(
        maintenance,
        "_canonical_profile_tenant_schema",
        lambda _home: (_ for _ in ()).throw(AssertionError("raw schema reached canonical resolution")),
    )
    monkeypatch.setattr(
        maintenance,
        "resolve_state_store_config",
        unexpected_config_resolution,
    )

    with pytest.raises(maintenance.StateStoreMaintenanceConfigurationError, match="trusted tenant"):
        operations.doctor({})

    assert reached == []


def test_manually_constructed_maintenance_rejects_another_profile_capability_before_sql(monkeypatch, tmp_path):
    """Even an issued capability is confined to its canonical profile tenant."""
    import state_store_maintenance as maintenance
    from state_store_alembic.runner import _runtime_state_store_schema

    operations = maintenance.StateStoreMaintenanceOperations(
        "postgresql", tmp_path, "default",
        _runtime_state_store_schema("hermes_state_store_tenant_0123456789abcdef0123456789abcdef"),
    )
    reached: list[str] = []

    monkeypatch.setattr(
        maintenance,
        "_canonical_profile_tenant_schema",
        lambda _home: _runtime_state_store_schema("hermes_state_store_tenant_fedcba9876543210fedcba9876543210"),
    )
    monkeypatch.setattr(
        maintenance,
        "resolve_state_store_config",
        lambda *_args, **_kwargs: reached.append("configuration"),
    )

    with pytest.raises(maintenance.StateStoreMaintenanceConfigurationError, match="canonical profile tenant"):
        operations.doctor({})

    assert reached == []


def test_resolved_maintenance_passes_issued_profile_capability_to_postgresql_operations(monkeypatch, tmp_path):
    import state_store_maintenance as maintenance
    from state_store import PostgreSQLStateStoreConfig, ResolvedStateStoreConfig
    from state_store_alembic.runner import _runtime_state_store_schema

    tenant = _runtime_state_store_schema("hermes_state_store_tenant_0123456789abcdef0123456789abcdef")
    report = SimpleNamespace(
        selected_backend="postgresql",
        profile_home=str(tmp_path),
        profile_name="default",
        tenant_schema=tenant.name,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(maintenance, "inspect_runtime_activation", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(maintenance, "_canonical_profile_tenant_schema", lambda home: tenant)
    resolved = maintenance.StateStoreMaintenanceOperations.resolve({}, home=tmp_path)

    assert resolved.tenant_schema is tenant

    class Operations:
        def __init__(self, _settings, _dsn, *, schema, profile_identity):
            captured.update(schema=schema, profile_identity=profile_identity)

        def doctor(self):
            return {"ok": True}

    monkeypatch.setattr(
        maintenance,
        "resolve_state_store_config",
        lambda *_args, **_kwargs: ResolvedStateStoreConfig(
            "postgresql", PostgreSQLStateStoreConfig("TEST_DSN", 1, 1),
        ),
    )
    monkeypatch.setattr("state_store._profile_secret_lookup", lambda *_args: "postgresql://unused")
    monkeypatch.setattr("postgresql_state_store_operations.PostgreSQLSandboxOperations", Operations)

    assert resolved.doctor({}) == {"ok": True}
    assert captured == {
        "schema": tenant,
        "profile_identity": {"home": str(tmp_path), "name": "default"},
    }
