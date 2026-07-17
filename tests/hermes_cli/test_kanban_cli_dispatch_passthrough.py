"""Regression tests for #33488 (CLI max_in_progress / max_spawn / per-profile
config passthrough) and #29415 (kanban_swarm humanizer skill ref).

These two fixes are bundled because they're both small, both touch the
kanban dispatcher's CLI surface, and they each guard against a silent
operator footgun that only manifests in long-running setups.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_console_entrypoint_propagates_kanban_exit_codes(tmp_path):
    """The installed console wrapper must preserve subcommand return codes."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n  dispatch_in_gateway: true\n", encoding="utf-8",
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for name in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        env.pop(name, None)
    entrypoint = [
        sys.executable,
        "-c",
        "from hermes_cli.main import main; raise SystemExit(main())",
    ]

    rejected = subprocess.run(
        entrypoint + ["kanban", "dispatch", "--dry-run"],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert rejected.returncode == 2
    assert "manual dispatch is disabled" in rejected.stderr
    assert not (home / "kanban.db").exists()

    help_result = subprocess.run(
        entrypoint + ["kanban", "--help"],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert help_result.returncode == 0


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
            "dispatch_in_gateway": False,
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


def test_cli_dispatch_refuses_mutation_when_gateway_owns_dispatch(
    isolated_kanban_home, monkeypatch, capsys,
):
    """Manual dispatch must not race the gateway-owned dispatcher."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )
    dispatch_once = MagicMock()
    monkeypatch.setattr(kanban_db, "dispatch_once", dispatch_once)

    rc = kb_cli.kanban_command(argparse.Namespace(
        kanban_action="dispatch", dry_run=False, max=None,
        failure_limit=2, json=False,
    ))

    assert rc == 2
    assert capsys.readouterr().err == (
        "kanban: manual dispatch is disabled while "
        "kanban.dispatch_in_gateway=true; use the singleton gateway "
        "(`hermes gateway start`) or set it to false first.\n"
    )
    dispatch_once.assert_not_called()


@pytest.mark.parametrize("error", [ValueError("bad dispatch"), RuntimeError("bad dispatch")])
def test_cli_dispatch_preserves_handler_error_exit_without_auto_init(
    isolated_kanban_home, monkeypatch, capsys, error,
):
    """Early dispatch keeps the normal handler error contract before DB init."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        kb_cli, "_cmd_dispatch", lambda args: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        kanban_db, "init_db", lambda: pytest.fail("dispatch must not auto-initialize"),
    )

    rc = kb_cli.kanban_command(argparse.Namespace(kanban_action="dispatch"))

    assert rc == 1
    assert capsys.readouterr().err == f"kanban: {error}\n"


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"kanban": {"dispatch_in_gateway": True}},
        {"kanban": {"dispatch_in_gateway": None}},
        {"kanban": {"dispatch_in_gateway": 0}},
        {"kanban": {"dispatch_in_gateway": ""}},
        {"kanban": {"dispatch_in_gateway": "false"}},
        {"kanban": {"dispatch_in_gateway": []}},
        {"kanban": {"dispatch_in_gateway": {}}},
    ],
)
def test_cli_dispatch_refuses_without_db_access_until_literal_false_admission(
    isolated_kanban_home, monkeypatch, capsys, dry_run, config,
):
    """Only literal false allows either CLI dispatch mode to touch the DB."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: config,
    )
    for name in ("init_db", "connect_closing", "dispatch_once"):
        monkeypatch.setattr(
            kanban_db, name, lambda *args, _name=name, **kwargs: pytest.fail(
                f"rejected admission must not call {_name}"
            ),
        )

    rc = kb_cli.kanban_command(argparse.Namespace(
        kanban_action="dispatch", dry_run=dry_run, max=None,
        failure_limit=2, json=False,
    ))

    assert rc == 2
    assert "manual dispatch is disabled" in capsys.readouterr().err


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_dispatch_standalone_holds_singleton_lock(
    isolated_kanban_home, monkeypatch, dry_run,
):
    """Every standalone manual-dispatch mode uses the gateway lock."""
    from gateway import kanban_watchers
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    handle = object()
    acquired = []
    released = []
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    monkeypatch.setattr(
        kanban_watchers,
        "_acquire_singleton_lock",
        lambda path: acquired.append(path) or (handle, "held"),
    )
    monkeypatch.setattr(kanban_watchers, "_release_singleton_lock", released.append)
    captured = {}
    monkeypatch.setattr(
        kanban_db,
        "dispatch_once",
        lambda conn, **kwargs: (captured.update(kwargs), kanban_db.DispatchResult())[1],
    )

    rc = kb_cli.kanban_command(argparse.Namespace(
        kanban_action="dispatch", dry_run=dry_run, max=None,
        failure_limit=2, json=False,
    ))

    assert rc == 0
    assert acquired == [kanban_db.kanban_home() / "kanban" / ".dispatcher.lock"]
    assert captured["dry_run"] is dry_run
    assert released == [handle]


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_dispatch_refuses_when_config_admission_fails(
    isolated_kanban_home, monkeypatch, capsys, dry_run,
):
    """A failed config read must not make either CLI mode permissive."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: (_ for _ in ()).throw(RuntimeError("config unavailable")),
    )
    for name in ("init_db", "connect_closing", "dispatch_once"):
        monkeypatch.setattr(
            kanban_db, name, lambda *args, _name=name, **kwargs: pytest.fail(
                f"failed admission must not call {_name}"
            ),
        )

    rc = kb_cli.kanban_command(argparse.Namespace(
        kanban_action="dispatch", dry_run=dry_run, max=None,
        failure_limit=2, json=False,
    ))

    assert rc == 2
    assert "manual dispatch is disabled" in capsys.readouterr().err


def test_cli_dispatch_admits_before_init_and_releases_after_dispatch(
    isolated_kanban_home, monkeypatch,
):
    """Standalone dispatch keeps board scope across admission, lock, and DB work."""
    from gateway import kanban_watchers
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    events = []
    handle = object()
    kanban_db.create_board("beta", name="Beta")

    def record(event):
        assert kanban_db.get_current_board() == "beta"
        events.append(event)

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: record("admit") or {"kanban": {"dispatch_in_gateway": False}},
    )
    monkeypatch.setattr(
        kanban_watchers, "_acquire_singleton_lock",
        lambda path: record("lock") or (handle, "held"),
    )
    monkeypatch.setattr(
        kanban_watchers, "_release_singleton_lock",
        lambda released: record("release"),
    )
    monkeypatch.setattr(kanban_db, "init_db", lambda: record("init"))

    @contextlib.contextmanager
    def connect():
        record("connect")
        yield object()

    monkeypatch.setattr(kanban_db, "connect_closing", connect)
    monkeypatch.setattr(
        kanban_db, "dispatch_once",
        lambda conn, **kwargs: record("dispatch") or kanban_db.DispatchResult(),
    )

    rc = kb_cli.kanban_command(argparse.Namespace(
        kanban_action="dispatch", board="beta", dry_run=False, max=None,
        failure_limit=2, json=False,
    ))

    assert rc == 0
    assert events == ["admit", "lock", "init", "connect", "dispatch", "release"]


def test_cli_dispatch_releases_lock_when_init_fails(
    isolated_kanban_home, monkeypatch, capsys,
):
    """A standalone init failure releases the singleton lock and returns an error."""
    from gateway import kanban_watchers
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    handle = object()
    released = []
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    monkeypatch.setattr(
        kanban_watchers, "_acquire_singleton_lock", lambda path: (handle, "held"),
    )
    monkeypatch.setattr(kanban_watchers, "_release_singleton_lock", released.append)
    monkeypatch.setattr(
        kanban_db, "init_db", lambda: (_ for _ in ()).throw(RuntimeError("broken DB")),
    )
    monkeypatch.setattr(
        kanban_db, "connect_closing",
        lambda: pytest.fail("failed initialization must not connect"),
    )

    rc = kb_cli.kanban_command(argparse.Namespace(
        kanban_action="dispatch", dry_run=False, max=None, failure_limit=2, json=False,
    ))

    assert rc == 1
    assert "could not initialize database: broken DB" in capsys.readouterr().err
    assert released == [handle]


def test_cli_max_flag_overrides_config_max_spawn(isolated_kanban_home, monkeypatch):
    """--max on the CLI takes precedence over kanban.max_spawn in config.
    The CLI flag is the explicit operator signal; config is the default."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    fake_config = {"kanban": {"dispatch_in_gateway": False, "max_spawn": 10}}
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
        fake_config = {"kanban": {"dispatch_in_gateway": False, "max_in_progress": bad_val}}
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
