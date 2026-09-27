"""Regression test for #125112: `hermes update --check` on a narrow clone."""

import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _mk_narrow_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Origin with a main branch + a tag; clone pinned to the tag only.

    Reproduces the older installer's shape: ``--depth 1 --single-branch
    --branch <tag>`` leaves ``remote.origin.fetch`` mapping only the tag, with
    no ``refs/heads/*`` line, so fetching main by name writes ``FETCH_HEAD``
    without ever materialising ``origin/main``.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.com")
    _git(origin, "config", "user.name", "t")
    _git(origin, "commit", "--allow-empty", "-q", "-m", "c0")
    _git(origin, "tag", "v9.9.9")
    _git(origin, "commit", "--allow-empty", "-q", "-m", "c1")

    clone = tmp_path / "clone"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            "v9.9.9",
            f"file://{origin}",
            str(clone),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return clone, origin


def test_check_fetch_materialises_tracking_ref_on_narrow_clone(
    tmp_path, monkeypatch, capsys
):
    import hermes_cli.update_cmd as update_cmd

    clone, origin = _mk_narrow_clone(tmp_path)
    assert "refs/heads/*" not in _git(
        clone, "config", "--get-all", "remote.origin.fetch"
    )
    tip_sha = _git(origin, "rev-parse", "main")
    # The defect: a by-name fetch leaves origin/main unresolvable.
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "rev-parse",
                "--verify",
                "--quiet",
                "origin/main",
            ],
            capture_output=True,
        ).returncode
        != 0
    )

    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", clone)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda root: None
    )
    monkeypatch.setattr(
        "hermes_cli.source_check._github_compare_behind", lambda *a, **k: None
    )

    update_cmd._cmd_update_check("main")

    out = capsys.readouterr().out
    # The fetch now materialises the tracking ref instead of failing.
    assert _git(clone, "rev-parse", "origin/main") == tip_sha
    assert "not found on origin" not in out
    assert "Update available" in out
    # The fix fetches by refspec and never rewrites the remote's fetch config.
    assert any(
        line.strip().startswith("+refs/tags/v9.9.9")
        for line in _git(
            clone, "config", "--get-all", "remote.origin.fetch"
        ).splitlines()
    )
