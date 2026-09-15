"""Integration tests for Wave2/2c: the deterministic gate precheck wired
into kanban_db_dispatch.py's per-task dispatch path.

Independent of the pure-function unit tests in
``tests/hermes_cli/test_kanban_gate_precheck.py`` — these exercise the REAL
``dispatch_once()`` -> ``_dispatch_lane_task()`` path against a real sqlite
DB, proving ``kanban.gate_precheck_enabled`` (config.yaml, NOT an env var)
actually gates dispatch and is a true no-op when unset/False, matching this
repo's default-safe precedent for opt-in dispatch guards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _config_with(monkeypatch, gate_precheck_enabled):
    import hermes_cli.config as cfgmod

    cfg = {"kanban": {"gate_precheck_enabled": gate_precheck_enabled}}
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: cfg)


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


class TestGatePrecheckDefaultOff:
    def test_flag_absent_does_not_block_a_task_that_would_fail_precheck(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        """Default False (unset config.yaml): even a task whose
        ears_sentence is garbage must dispatch normally — the precheck is a
        true no-op until an operator opts in via config.yaml."""
        import hermes_cli.config as cfgmod
        monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})

        spawns: list = []
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title="t", assignee="alice",
                ears_sentence="not even close to an EARS sentence",
            )
            result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

        assert spawns == [task_id]
        assert result.skipped_gate_precheck == []

    def test_explicit_false_does_not_block_dispatch(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        _config_with(monkeypatch, False)

        spawns: list = []
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title="t", assignee="alice",
                ears_sentence="not even close to an EARS sentence",
                scope_paths=[f"file{i}.py" for i in range(9)],
            )
            result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

        assert spawns == [task_id]
        assert result.skipped_gate_precheck == []


class TestGatePrecheckEnabled:
    def test_malformed_ears_sentence_blocks_dispatch_when_enabled(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        _config_with(monkeypatch, True)

        spawns: list = []
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title="t", assignee="alice",
                ears_sentence="not even close to an EARS sentence",
            )
            result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

        assert spawns == []
        assert result.skipped_gate_precheck == [
            (task_id, "ears_sentence does not look EARS-shaped: missing the "
                       "mandatory 'shall' response keyword"),
        ]

    def test_wellformed_ears_sentence_dispatches_when_enabled(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        _config_with(monkeypatch, True)

        spawns: list = []
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title="t", assignee="alice",
                ears_sentence="When the user clicks submit, the system shall save the form.",
            )
            result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

        assert spawns == [task_id]
        assert result.skipped_gate_precheck == []

    def test_unset_ears_sentence_is_not_gated_and_dispatches(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        """not_gated is a DISPATCHABLE status — this precheck validates
        shape when evidence is present, it never forces ears_sentence
        population as a hard gate."""
        _config_with(monkeypatch, True)

        spawns: list = []
        with kbc.connect() as conn:
            task_id = kb.create_task(conn, title="t", assignee="alice")
            result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

        assert spawns == [task_id]
        assert result.skipped_gate_precheck == []

    def test_fast_lane_bypasses_ears_check_when_enabled(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        """A single-file doc-glob task with no behavior-change language
        fast-lanes even when its (malformed) ears_sentence would otherwise
        block it — a README typo must not be forced through the EARS gate."""
        _config_with(monkeypatch, True)

        spawns: list = []
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title="update README typo", assignee="alice",
                ears_sentence="not an EARS sentence at all",
                scope_paths=["README.md"],
            )
            result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

        assert spawns == [task_id]
        assert result.skipped_gate_precheck == []

    def test_oversized_scope_blocks_dispatch_when_enabled(
        self, kanban_home, all_assignees_spawnable, monkeypatch,
    ):
        _config_with(monkeypatch, True)

        spawns: list = []
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn, title="t", assignee="alice",
                ears_sentence="The system shall do the thing.",
                scope_paths=[f"file{i}.py" for i in range(6)],
            )
            result = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))

        assert spawns == []
        assert result.skipped_gate_precheck[0][0] == task_id
        assert "likely decomposable" in result.skipped_gate_precheck[0][1]
