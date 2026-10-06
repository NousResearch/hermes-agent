"""A dependency generation's staged workspace must never masquerade as the install (#130116)."""
from __future__ import annotations

from pathlib import Path

from pm.environments import install_key, install_root_for_tree, is_staged_workspace


def _staged_layout(tmp_path: Path, install: Path, *, record: bool) -> Path:
    state_dir = tmp_path / "installs" / install_key(install)
    (state_dir / "inputs").mkdir(parents=True)
    if record:
        (state_dir / "inputs" / ".project-root").write_text(str(install.resolve()), encoding="utf-8")
    workspace = state_dir / "environments" / "0123abcd" / "workspace"
    (workspace / "hermes_cli").mkdir(parents=True)
    return workspace


def test_staged_workspace_resolves_to_its_recorded_install(tmp_path):
    install = tmp_path / "hermes-agent"
    (install / "hermes_cli").mkdir(parents=True)
    workspace = _staged_layout(tmp_path, install, record=True)

    assert is_staged_workspace(workspace)
    assert install_root_for_tree(workspace) == install.resolve()
    assert not is_staged_workspace(install)
    assert install_root_for_tree(install) == install.resolve()


def test_unrecorded_or_mismatched_record_keeps_the_workspace_and_stays_staged(tmp_path):
    install = tmp_path / "hermes-agent"
    (install / "hermes_cli").mkdir(parents=True)
    unrecorded = _staged_layout(tmp_path, install, record=False)
    assert install_root_for_tree(unrecorded) == unrecorded.resolve()
    assert is_staged_workspace(unrecorded)

    other = tmp_path / "elsewhere"
    (other / "hermes_cli").mkdir(parents=True)
    mismatched = _staged_layout(tmp_path / "second", install, record=True)
    (mismatched.parent.parent.parent / "inputs" / ".project-root").write_text(str(other), encoding="utf-8")
    assert install_root_for_tree(mismatched) == mismatched.resolve()
