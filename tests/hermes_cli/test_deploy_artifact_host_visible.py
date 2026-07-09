from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_zip(path: Path, entries: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)


def test_deploy_artifact_contract_accepts_host_visible_canonical_zip(tmp_path):
    package = tmp_path / "deploy-package.zip"
    _write_zip(
        package,
        {
            "manifest.json": '{"ok": true}',
            "dist/index.html": "<html>ready</html>",
        },
    )
    expected_sha256 = hashlib.sha256(package.read_bytes()).hexdigest()

    result = kb.verify_deploy_artifact_contract(
        artifact_path=str(package),
        expected_sha256=expected_sha256,
        required_entries=["manifest.json", "dist/index.html"],
    )

    assert result["ok"] is True
    assert result["checked"]["canonical_path"] == str(package.resolve())
    assert result["checked"]["byte_length"] == package.stat().st_size
    assert result["checked"]["sha256"] == expected_sha256
    assert result["checked"]["runtime_can_read"] is True


def test_deploy_artifact_contract_rejects_noncanonical_host_invisible_path():
    result = kb.verify_deploy_artifact_contract(
        artifact_path=r"\\d\\deploy\\package.zip",
        required_entries=["manifest.json"],
    )

    codes = {issue["code"] for issue in result["issues"]}
    assert result["ok"] is False
    assert "noncanonical_path" in codes
    assert "host_visible_missing" in codes


def test_kanban_complete_blocks_wrapper_success_without_declared_artifact(kanban_home, tmp_path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="package deploy", assignee="worker")
        kb.claim_task(conn, task_id)
        missing = tmp_path / "missing-deploy-package.zip"

        with pytest.raises(kb.ArtifactVerificationError):
            kb.complete_task(
                conn,
                task_id,
                summary="Wrapper said deploy package ready.",
                metadata={"artifacts": [str(missing)]},
            )

        assert kb.get_task(conn, task_id).status == "running"
