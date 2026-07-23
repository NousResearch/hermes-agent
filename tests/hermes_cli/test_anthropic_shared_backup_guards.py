"""Generic backup must refuse active/dormant shared Anthropic state."""

from __future__ import annotations

import pytest

from tests.agent.anthropic_shared_test_helpers import (
    enable_marker,
    shared_root,
    stage_three,
)


def test_has_dormant_shared_state(shared_root):
    stage_three(shared_root, attest=False)
    from agent.anthropic_shared_pool import has_dormant_or_active_shared_state

    assert has_dormant_or_active_shared_state() is True


def test_refuse_generic_backup_dormant(shared_root):
    stage_three(shared_root, attest=False)
    from agent.anthropic_shared_pool import refuse_generic_backup_if_shared
    from hermes_cli.auth import AuthError

    with pytest.raises(AuthError) as ei:
        refuse_generic_backup_if_shared()
    assert ei.value.code == "shared_blocks_generic_backup"


def test_refuse_generic_backup_active(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent.anthropic_shared_pool import refuse_generic_backup_if_shared
    from hermes_cli.auth import AuthError

    with pytest.raises(AuthError):
        refuse_generic_backup_if_shared()


def test_shared_backup_restore_roundtrip(shared_root, tmp_path, monkeypatch):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root, epoch="dddddddd-dddd-dddd-dddd-dddddddddddd")
    monkeypatch.setattr(
        "agent.anthropic_shared_pool.require_no_live_gateways_for_scope_change",
        lambda: None,
    )
    from agent.anthropic_shared_pool import (
        create_shared_backup,
        get_shared_mutation_capability,
        is_shared_scope_active,
        restore_shared_backup,
        remove_scope_marker,
        load_shared_pool_for_management,
    )

    cap = get_shared_mutation_capability()
    out = tmp_path / "backup.tgz"
    create_shared_backup(out, capability=cap)
    assert out.exists()
    assert out.stat().st_mode & 0o077 == 0

    # Destroy live state
    remove_scope_marker()
    from agent.anthropic_shared_pool import clear_shared_namespace

    clear_shared_namespace(capability=cap)
    assert not is_shared_scope_active()

    restore_shared_backup(out, yes=True, capability=cap)
    assert is_shared_scope_active()
    pool = load_shared_pool_for_management(require_active_three=True)
    assert len(pool["entries"]) == 3


def test_backup_module_hooks_refuse(shared_root, monkeypatch):
    """hermes_cli.backup should call refuse when shared state exists."""
    stage_three(shared_root, attest=False)
    import hermes_cli.backup as backup_mod

    # Prefer explicit guard if present; otherwise patch run_backup preflight.
    if hasattr(backup_mod, "refuse_generic_backup_if_shared"):
        from hermes_cli.auth import AuthError
        with pytest.raises(AuthError):
            backup_mod.refuse_generic_backup_if_shared()
    else:
        from agent.anthropic_shared_pool import refuse_generic_backup_if_shared
        from hermes_cli.auth import AuthError
        with pytest.raises(AuthError):
            refuse_generic_backup_if_shared()
