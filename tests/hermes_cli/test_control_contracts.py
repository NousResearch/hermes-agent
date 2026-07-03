from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.control_contracts import ContractError, make_child_payload, validate_statute_dispatch_v1


def payload(root: Path, **overrides):
    data = {
        "schema": "statute_dispatch_v1",
        "silo": "statute",
        "repo_root": str(root),
        "allowed_paths": [str(root)],
        "task_type": "generic",
        "task_permissions": ["read", "write", "test"],
        "parent_dispatch_id": None,
        "instructions": "work",
        "constraints": {"no_live_db_mutation": True, "no_push": True},
    }
    data.update(overrides)
    return data


def test_statute_dispatch_v1_accepts_valid_temp_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert validate_statute_dispatch_v1(payload(root))["repo_root"] == str(root)


def test_statute_dispatch_v1_rejects_path_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ContractError):
        validate_statute_dispatch_v1(payload(root, allowed_paths=[str(tmp_path)]))


def test_child_scope_is_monotonic(tmp_path):
    root = tmp_path / "repo"
    child_dir = root / "sub"
    child_dir.mkdir(parents=True)
    parent = payload(root, allowed_paths=[str(root)], task_permissions=["read", "test"])
    child = make_child_payload(parent, parent_dispatch_id="disp_parent")
    assert child["parent_dispatch_id"] == "disp_parent"
    with pytest.raises(ContractError):
        validate_statute_dispatch_v1(payload(root, allowed_paths=[str(root)], task_permissions=["read", "write"], parent_dispatch_id="disp_parent"), parent=parent, require_parent=True)
    with pytest.raises(ContractError):
        validate_statute_dispatch_v1(payload(root, constraints={"no_live_db_mutation": True, "no_push": False}, parent_dispatch_id="disp_parent"), parent=parent, require_parent=True)


def test_push_at_successful_wave_closeout_can_replace_no_push(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    wave = validate_statute_dispatch_v1(payload(root, constraints={"no_live_db_mutation": True, "push_at_successful_wave_closeout": True}))
    assert wave["constraints"]["push_at_successful_wave_closeout"] is True


def test_invalid_task_type_and_permissions_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ContractError):
        validate_statute_dispatch_v1(payload(root, task_type="escape"))
    with pytest.raises(ContractError):
        validate_statute_dispatch_v1(payload(root, task_permissions=["read", "sudo"]))
