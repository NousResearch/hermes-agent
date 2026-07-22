"""Regression tests for #33488 (CLI max_in_progress / max_spawn / per-profile
config passthrough) and #29415 (kanban_swarm humanizer skill ref).

These two fixes are bundled because they're both small, both touch the
kanban dispatcher's CLI surface, and they each guard against a silent
operator footgun that only manifests in long-running setups.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_cli_passthrough_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    def is_hermes_module(name):
        return (
            name.startswith("hermes_cli")
            or name.startswith("hermes_state")
            or name == "hermes_constants"
        )

    # These tests need a fresh import after HERMES_HOME changes, but removing
    # shared modules without restoring them leaves later collected test files
    # holding an older module object than ``import hermes_cli.kanban_db``
    # returns. Snapshot and restore the exact objects to keep cross-file runs
    # deterministic.
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if is_hermes_module(name)
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    try:
        yield test_home
    finally:
        for name in list(sys.modules):
            if is_hermes_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def test_cli_dispatch_passes_max_in_progress_from_config(isolated_kanban_home, monkeypatch):
    """#33488: hermes kanban dispatch must pass kanban.max_in_progress from
    config to dispatch_once. Without this, the global concurrency cap is
    unreachable from the CLI even though it works from the gateway."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    # Configure max_in_progress in the loaded config.
    fake_config = {
        "kanban": {
            "max_in_progress": 3,
            "max_spawn": 5,
            "default_assignee": "default",
            "max_in_progress_per_profile": 2,
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: fake_config
    )

    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    monkeypatch.setattr(kanban_db, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    # Every config value must have reached dispatch_once.
    assert captured.get("max_in_progress") == 3, (
        f"CLI must pass kanban.max_in_progress from config; got {captured.get('max_in_progress')!r}"
    )
    assert captured.get("max_spawn") == 5, (
        f"CLI must pass kanban.max_spawn from config when --max is not provided; got {captured.get('max_spawn')!r}"
    )
    assert captured.get("default_assignee") == "default"
    assert captured.get("max_in_progress_per_profile") == 2


def test_dispatch_parser_exposes_explicit_run_and_optional_assignee_scope():
    from hermes_cli import kanban as kb_cli

    parser = argparse.ArgumentParser()
    kb_cli.build_parser(parser.add_subparsers(dest="command"))

    preview = parser.parse_args(["kanban", "dispatch"])
    assert preview.run is False
    assert preview.assignee is None

    scoped_run = parser.parse_args(
        ["kanban", "dispatch", "--run", "--assignee", "alice"]
    )
    assert scoped_run.run is True
    assert scoped_run.assignee == "alice"


def test_cli_dispatch_defaults_to_preview_until_explicit_run(
    isolated_kanban_home, monkeypatch, capsys
):
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult(spawned=[("t_preview", "alice", "")])

    monkeypatch.setattr(kanban_db, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(
        dry_run=False, run=False, max=None, assignee=None, failure_limit=2, json=False
    )
    kb_cli._cmd_dispatch(args)

    assert captured["dry_run"] is True
    out = capsys.readouterr().out
    assert "Dispatch preview only" in out
    assert "--run" in out
    assert "t_preview" in out
    assert "(preview)" in out


def test_cli_dispatch_run_passes_explicit_scope_and_emits_json_receipt(
    isolated_kanban_home, monkeypatch, capsys
):
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    captured = {}
    receipt = {
        "task_id": "t_run",
        "assignee": "alice",
        "workspace": "/tmp/work",
        "worker_pid": 1234,
        "status": "running",
    }

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult(
            spawned=[("t_run", "alice", "/tmp/work")], receipts=[receipt]
        )

    monkeypatch.setattr(kanban_db, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(
        dry_run=False, run=True, max=None, assignee="alice", failure_limit=2, json=True
    )
    kb_cli._cmd_dispatch(args)

    assert captured["dry_run"] is False
    assert captured["assignee_filter"] == "alice"
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "run"
    assert payload["preview_only"] is False
    assert payload["assignee_filter"] == "alice"
    assert payload["receipts"] == [receipt]


def test_cli_dispatch_run_prints_non_json_spawn_receipt(
    isolated_kanban_home, monkeypatch, capsys
):
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    receipt = {
        "task_id": "t_run",
        "assignee": "alice",
        "workspace": "/tmp/work",
        "worker_pid": 1234,
        "status": "running",
    }

    def fake_dispatch_once(conn, **kwargs):
        result = kanban_db.DispatchResult(
            spawned=[("t_run", "alice", "/tmp/work")]
        )
        setattr(result, "receipts", [receipt])
        return result

    monkeypatch.setattr(kanban_db, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(
        dry_run=False,
        run=True,
        max=None,
        assignee=None,
        failure_limit=2,
        json=False,
    )
    kb_cli._cmd_dispatch(args)

    out = capsys.readouterr().out
    assert "Dispatch run complete." in out
    assert "Receipts:" in out
    assert "t_run" in out
    assert "pid=1234" in out
    assert "assignee=alice" in out
    assert "workspace=/tmp/work" in out


def test_cli_max_flag_overrides_config_max_spawn(isolated_kanban_home, monkeypatch):
    """--max on the CLI takes precedence over kanban.max_spawn in config.
    The CLI flag is the explicit operator signal; config is the default."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    fake_config = {"kanban": {"max_spawn": 10}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}
    monkeypatch.setattr(
        kanban_db, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )

    args = argparse.Namespace(dry_run=True, max=2, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    assert captured.get("max_spawn") == 2, (
        f"CLI --max=2 must override config kanban.max_spawn=10; got {captured.get('max_spawn')!r}"
    )


def test_cli_invalid_max_in_progress_silently_disables(isolated_kanban_home, monkeypatch):
    """Invalid kanban.max_in_progress values (0, negative, non-int) should
    silently fall through to None — no crash, no surprise behavior."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    for bad_val in (0, -1, "abc", "1.5"):
        fake_config = {"kanban": {"max_in_progress": bad_val}}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)
        captured = {}
        monkeypatch.setattr(
            kanban_db, "dispatch_once",
            lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
        )
        args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
        kb_cli._cmd_dispatch(args)
        assert captured.get("max_in_progress") is None, (
            f"invalid max_in_progress={bad_val!r} should fall through to None, "
            f"got {captured.get('max_in_progress')!r}"
        )


def test_kanban_swarm_uses_existing_humanizer_skill():
    """#29415: kanban_swarm.py used to hardcode skills=['avoid-ai-writing'],
    a skill that doesn't exist in any registry — synthesizer workers
    crashed with 'Unknown skill(s): avoid-ai-writing' on every retry.

    Verify the synthesizer card now uses the bundled 'humanizer' skill
    which actually exists at skills/creative/humanizer/SKILL.md."""
    import pathlib

    swarm_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "hermes_cli" / "kanban_swarm.py"
    )
    src = swarm_path.read_text()
    assert "avoid-ai-writing" not in src, (
        "kanban_swarm.py must not reference 'avoid-ai-writing' — that "
        "skill doesn't exist in any registry, crashing synthesizers (#29415)"
    )
    assert '"humanizer"' in src, (
        "kanban_swarm.py should use the bundled 'humanizer' skill for "
        "synthesizer cards (the original intent of 'avoid-ai-writing')"
    )

    # And the replacement skill must actually exist on disk.
    skills_root = (
        pathlib.Path(__file__).resolve().parent.parent.parent / "skills"
    )
    humanizer_path = skills_root / "creative" / "humanizer" / "SKILL.md"
    assert humanizer_path.is_file(), (
        f"humanizer skill missing at {humanizer_path}; the kanban_swarm fix "
        "for #29415 requires this bundled skill to exist"
    )
