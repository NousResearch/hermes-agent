"""Regression tests for #33488 (CLI max_in_progress / max_spawn / per-profile
config passthrough) and #29415 (kanban_swarm humanizer skill ref).

These two fixes are bundled because they're both small, both touch the
kanban dispatcher's CLI surface, and they each guard against a silent
operator footgun that only manifests in long-running setups.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_cli_passthrough_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    yield test_home


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
