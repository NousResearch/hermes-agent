"""Version transition reporting after ``hermes update``.

Ported from PrimeIntellect-ai/prime-agent#630: a successful self-update
reports both versions (``v0.19.4 → v0.20.0``). pyproject.toml is an inert
0.0.0 on source checkouts, so the reported versions are the checkout's
runtime identity -- the same one the completion publishes in its stamp.
"""

import os
from pathlib import Path
import subprocess

import pytest

from hermes_cli import update_cmd
from hermes_cli.source_stamp import write_source_stamp


def _write_pyproject(root: Path, version: str) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "hermes-agent"\nversion = "{version}"\n',
        encoding="utf-8",
    )


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    class _FakeMain:
        PROJECT_ROOT = tmp_path

    monkeypatch.setattr(update_cmd, "_m", lambda: _FakeMain)
    return tmp_path


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


class TestReadProjectVersion:
    def test_reads_version(self, fake_root):
        _write_pyproject(fake_root, "0.20.0")
        assert update_cmd._read_project_version() == "0.20.0"

    def test_missing_file_returns_none(self, fake_root):
        assert update_cmd._read_project_version() is None

    def test_malformed_toml_returns_none(self, fake_root):
        (fake_root / "pyproject.toml").write_text("not [ toml", encoding="utf-8")
        assert update_cmd._read_project_version() is None


class TestUpdateCompleteMessage:
    def test_transition_reports_the_identity_the_stamp_publishes(self, fake_root):
        _git(fake_root, "init", "-q")
        _write_pyproject(fake_root, "0.0.0")
        _git(fake_root, "add", "pyproject.toml")
        _git(fake_root, "commit", "-qm", "release")
        _git(fake_root, "tag", "v0.21.4")
        pre = update_cmd._checkout_version()
        _git(fake_root, "commit", "-q", "--allow-empty", "-m", "update")

        message = update_cmd._update_complete_message(pre)
        stamp = write_source_stamp(fake_root)
        assert stamp is not None

        assert message == f"✓ Update complete! (v{pre} → v{stamp['displayVersion']})"

    def test_tagless_checkout_reports_its_commit_identity(self, fake_root):
        _git(fake_root, "init", "-q")
        _git(fake_root, "commit", "-q", "--allow-empty", "-m", "only")

        message = update_cmd._update_complete_message("0.21.4")
        stamp = write_source_stamp(fake_root)
        assert stamp is not None

        assert message == f"✓ Update complete! (v0.21.4 → {stamp['displayVersion']})"
