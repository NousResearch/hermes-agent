"""File safety for Anthropic shared scope marker and root auth."""

from __future__ import annotations

import os
from pathlib import Path

from tests.agent.anthropic_shared_test_helpers import enable_marker, shared_root, stage_three


def test_marker_in_read_denylist(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent.file_safety import get_read_block_error

    marker = shared_root / "shared" / "anthropic_pool_scope.json"
    err = get_read_block_error(str(marker))
    assert err is not None
    assert "denied" in err.lower() or "credential" in err.lower() or "scope" in err.lower()


def test_marker_in_write_denylist(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent.file_safety import get_write_denied_error

    marker = shared_root / "shared" / "anthropic_pool_scope.json"
    err = get_write_denied_error(str(marker))
    assert err is not None


def test_root_auth_write_denied_from_profile(shared_root, monkeypatch):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    prof = shared_root / "profiles" / "w"
    prof.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(prof))
    from agent.file_safety import get_write_denied_error

    err = get_write_denied_error(str(shared_root / "auth.json"))
    assert err is not None


def test_atomic_write_rejects_symlink_target(shared_root, tmp_path):
    from agent.anthropic_shared_pool import _atomic_write_json
    from hermes_cli.auth import AuthError
    import pytest

    real = tmp_path / "real.json"
    real.write_text("{}")
    target = shared_root / "shared" / "x.json"
    target.symlink_to(real)
    with pytest.raises(AuthError):
        _atomic_write_json(target, {"a": 1})
