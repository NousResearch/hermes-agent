"""Tests for `hermes curator report` CLI behavior."""

from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture
def curator_report_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    from hermes_cli import curator as curator_cli
    importlib.reload(curator_cli)

    return {
        "home": home,
        "curator": curator,
        "curator_cli": curator_cli,
    }


def test_report_prints_latest_report_from_state_path(curator_report_env, capsys):
    curator = curator_report_env["curator"]
    curator_cli = curator_report_env["curator_cli"]

    run_dir = curator._reports_root() / "20260511-010203"
    run_dir.mkdir(parents=True)
    expected = "# Curator run\nlatest\n"
    (run_dir / "REPORT.md").write_text(expected, encoding="utf-8")

    state = curator.load_state()
    state["last_report_path"] = str(run_dir)
    curator.save_state(state)

    assert curator_cli._cmd_report(Namespace(path_only=False)) == 0
    assert capsys.readouterr().out == expected


def test_report_path_flag_prints_resolved_report_md(curator_report_env, capsys):
    curator = curator_report_env["curator"]
    curator_cli = curator_report_env["curator_cli"]

    run_dir = curator._reports_root() / "20260511-010203"
    run_dir.mkdir(parents=True)
    report_path = run_dir / "REPORT.md"
    report_path.write_text("# Curator run\n", encoding="utf-8")

    state = curator.load_state()
    state["last_report_path"] = str(run_dir)
    curator.save_state(state)

    assert curator_cli._cmd_report(Namespace(path_only=True)) == 0
    assert capsys.readouterr().out.strip() == str(report_path)


def test_report_falls_back_when_state_path_is_missing(curator_report_env, capsys):
    curator = curator_report_env["curator"]
    curator_cli = curator_report_env["curator_cli"]

    root = curator._reports_root()
    older = root / "20260511-010203"
    newer = root / "20260511-040506"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "REPORT.md").write_text("# older\n", encoding="utf-8")
    (newer / "REPORT.md").write_text("# newer\n", encoding="utf-8")

    state = curator.load_state()
    state["last_report_path"] = str(root / "missing-run")
    curator.save_state(state)

    assert curator_cli._cmd_report(Namespace(path_only=False)) == 0
    assert capsys.readouterr().out == "# newer\n"


def test_report_without_any_run_is_graceful(curator_report_env, capsys):
    curator_cli = curator_report_env["curator_cli"]

    assert curator_cli._cmd_report(Namespace(path_only=False)) == 1
    err = capsys.readouterr().err
    assert "curator: no report found yet" in err

