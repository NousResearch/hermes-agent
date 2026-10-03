"""Read-only inventory for parked ``hermes update`` autostashes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd
from hermes_cli import update_cmd_stash


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check,
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "init")
    return repo


def _make_autostash(repo: Path, *, name: str = "hermes-update-autostash-20260920-120000") -> str:
    (repo / "tracked.txt").write_text("local\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")
    _git(repo, "stash", "push", "--include-untracked", "-m", name)
    return _git(repo, "rev-parse", "refs/stash").stdout.strip()


def _receipt(home: Path, stash_ref: str, detail: str, *, outcome: str = "success") -> None:
    directory = home / "logs" / "update_receipts"
    directory.mkdir(parents=True)
    payload = {
        "started_at": "2026-09-20T12:01:00+00:00",
        "outcome": outcome,
        "steps": [
            {
                "name": "local_changes_stash",
                "ok": not detail.startswith("parked"),
                "detail": f"{detail}: {stash_ref}" if detail in {"restored", "discarded"}
                else f"parked: {stash_ref} ({detail})",
            }
        ],
    }
    (directory / "update_20260920_120100_1.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def test_inventory_lists_contents_and_never_changes_stash_refs(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    stash_ref = _make_autostash(repo)
    before = _git(repo, "stash", "list", "--format=%H").stdout
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])

    count = update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    after = _git(repo, "stash", "list", "--format=%H").stdout
    out = capsys.readouterr().out
    assert count == 1
    assert before == after
    assert "stash@{0}" in out
    assert stash_ref in out
    assert "Files: 1 tracked, 1 untracked" in out
    assert 'tracked: "tracked.txt"' in out
    assert 'untracked: "scratch.txt"' in out
    assert "Reason: unknown — legacy stash or receipt unavailable" in out
    assert f"git stash show --stat {stash_ref}" in out
    assert f"git stash apply {stash_ref}" in out
    assert "git stash drop 0" in out


def test_inventory_ignores_non_hermes_stashes(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("user work\n", encoding="utf-8")
    _git(repo, "stash", "push", "-m", "my work")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])

    count = update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    assert count == 0
    assert "No Hermes update autostashes found." in capsys.readouterr().out


def test_inventory_ignores_near_collision_stash_name(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("user work\n", encoding="utf-8")
    _git(
        repo, "stash", "push", "-m",
        "my hermes-update-autostash-20260920-120000 notes",
    )
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])

    count = update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    assert count == 0
    assert "No Hermes update autostashes found." in capsys.readouterr().out


def test_inventory_handles_tracked_only_stash(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("tracked only\n", encoding="utf-8")
    _git(repo, "stash", "push", "-m", "hermes-update-autostash-20260920-120000")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])

    update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    assert "Files: 1 tracked, 0 untracked" in capsys.readouterr().out


def test_inventory_quotes_unusual_valid_filename(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "notes μ [draft].txt").write_text("notes\n", encoding="utf-8")
    _git(repo, "stash", "push", "--include-untracked", "-m", "hermes-update-autostash-20260920-120000")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])

    update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    assert 'untracked: "notes \\u03bc [draft].txt"' in capsys.readouterr().out


def test_inventory_uses_matching_receipt_reason(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    stash_ref = _make_autostash(repo)
    home = tmp_path / "home"
    _receipt(home, stash_ref, "restore hit conflicts", outcome="success")
    monkeypatch.setattr(
        "hermes_cli.update_receipt._profile_homes", lambda: [("default", home)],
    )

    update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    out = capsys.readouterr().out
    assert "Reason: parked — restore hit conflicts" in out
    assert "Receipt: default, 2026-09-20T12:01:00+00:00, outcome success" in out


def test_inventory_does_not_guess_from_nonmatching_receipt(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    _make_autostash(repo)
    home = tmp_path / "home"
    _receipt(home, "0" * 40, "restore hit conflicts", outcome="success")
    monkeypatch.setattr(
        "hermes_cli.update_receipt._profile_homes", lambda: [("default", home)],
    )

    update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    out = capsys.readouterr().out
    assert "Reason: unknown — legacy stash or receipt unavailable" in out
    assert "Receipt:" not in out


def test_inventory_marks_restored_stash_as_incomplete_cleanup(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    stash_ref = _make_autostash(repo)
    home = tmp_path / "profile"
    _receipt(home, stash_ref, "restored")
    monkeypatch.setattr(
        "hermes_cli.update_receipt._profile_homes", lambda: [("writer", home)],
    )

    update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    out = capsys.readouterr().out
    assert "Reason: restored — stash cleanup incomplete" in out
    assert "Receipt: writer" in out


def test_inventory_path_preview_is_bounded(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    for index in range(7):
        (repo / f"file-{index}.txt").write_text(str(index), encoding="utf-8")
    _git(repo, "stash", "push", "--include-untracked", "-m", "hermes-update-autostash-20260920-120000")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])

    update_cmd_stash._print_update_autostash_inventory(["git"], repo)

    out = capsys.readouterr().out
    assert "Files: 0 tracked, 7 untracked" in out
    assert "... 2 more" in out


def test_update_parser_accepts_list_autostashes():
    from hermes_cli.subcommands.update import build_update_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_update_parser(subparsers, cmd_update=lambda args: None)

    assert parser.parse_args(["update", "--list-autostashes"]).list_autostashes is True
    assert parser.parse_args(["update"]).list_autostashes is False


def test_list_autostashes_exits_before_update_lock(tmp_path, monkeypatch):
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        update_cmd, "_print_update_autostash_inventory",
        lambda git_cmd, cwd: calls.append((git_cmd, cwd)) or 0,
        raising=False,
    )
    monkeypatch.setattr(
        hermes_main, "_install_hangup_protection",
        lambda **kwargs: pytest.fail("update runtime must not start"),
    )
    args = SimpleNamespace(
        list_autostashes=True, plan=False, list_venv_holders=False, check=False,
    )

    hermes_main.cmd_update(args)

    assert calls == [(["git"], tmp_path)]
