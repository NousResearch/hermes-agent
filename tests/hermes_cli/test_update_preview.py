"""Tests for update preview helpers and interactive CLI confirmation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys


def test_get_update_preview_summarizes_commits_and_release_notes(tmp_path):
    from hermes_cli.update_preview import get_update_preview

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    def fake_run(cmd, **kwargs):
        joined = " ".join(str(part) for part in cmd)
        if "rev-list --count" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="3\n", stderr="")
        if "log --format=%s --no-merges" in joined:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "fix(gateway): break compression loop (#9893)\n"
                    "feat: add gateway restart flag (#10043)\n"
                    "fix: strip non-ascii API keys (#6843)\n"
                ),
                stderr="",
            )
        if f"show origin/main:hermes_cli/__init__.py" in joined:
            return subprocess.CompletedProcess(
                cmd, 0, stdout='__version__ = "0.10.0"\n', stderr=""
            )
        if "diff --name-only" in joined:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="RELEASE_v0.10.0.md\n", stderr=""
            )
        if "show origin/main:RELEASE_v0.10.0.md" in joined:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "# Hermes Agent v0.10.0\n\n"
                    "## ✨ Highlights\n\n"
                    "- **Nous Tool Gateway** — managed tools for subscribers.\n"
                    "- **Safer updates** — preview changes before applying them.\n"
                ),
                stderr="",
            )
        raise AssertionError(f"Unexpected git command: {cmd}")

    with patch("hermes_cli.update_preview.subprocess.run", side_effect=fake_run):
        preview = get_update_preview(repo_dir, git_cmd=["git"], base_ref="main")

    assert preview is not None
    assert preview.commit_count == 3
    assert preview.target_version == "0.10.0"
    assert preview.commits[:2] == [
        "fix(gateway): break compression loop (#9893)",
        "feat: add gateway restart flag (#10043)",
    ]
    assert preview.releases[0].version == "v0.10.0"
    assert preview.releases[0].highlights[0].startswith("Nous Tool Gateway")


def test_cmd_update_interactive_preview_can_cancel(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(load_dotenv=lambda *args, **kwargs: False),
    )
    import hermes_cli.main as hermes_main
    from hermes_cli.update_preview import ReleaseSummary, UpdatePreview

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_get_origin_url", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(hermes_main, "_is_fork", lambda *_args, **_kwargs: False)

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "fetch", "origin"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if cmd == ["git", "rev-list", "HEAD..origin/main", "--count"]:
            return SimpleNamespace(returncode=0, stdout="3\n", stderr="")
        raise AssertionError(f"Unexpected subprocess call: {cmd}")

    preview = UpdatePreview(
        current_version="0.9.0",
        target_version="0.10.0",
        commit_count=3,
        base_ref="HEAD",
        remote_ref="origin/main",
        commits=["fix(gateway): break compression loop (#9893)"],
        releases=[ReleaseSummary(version="v0.10.0", highlights=["Nous Tool Gateway"])],
    )

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with patch("hermes_cli.update_preview.resolve_update_base_ref", return_value="HEAD"), \
         patch("hermes_cli.update_preview.get_update_preview", return_value=preview), \
         patch("builtins.input", return_value="n"), \
         patch.object(hermes_main.sys.stdin, "isatty", return_value=True), \
         patch.object(hermes_main.sys.stdout, "isatty", return_value=True):
        hermes_main.cmd_update(SimpleNamespace(yes=False))

    output = capsys.readouterr().out
    assert "Hermes update preview" in output
    assert "hermes update --yes" in output
    assert "Update cancelled." in output
