"""Dispatcher quota preflight: provider quota=0 시 ready 상태 유지 TDD.

검증 항목:
  1. quota=0 (용량 부족)일 때 task를 claim하지 않고 ready 유지
  2. usage lookup 오류(None) 시 ready 유지
  3. 회복 후 다음 tick에서 정상 spawn
  4. claim이 preflight 이전에 발생하지 않는 것 확인 (claim-before-preflight 방지)
  5. quota 부족 skip 시 failure counter 불변
  6. provider_meter_remediation 비활성 시 preflight 생략
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_cli.kanban_db as kb
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from tests.hermes_cli._kanban_isolation import pin_kanban_db


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _snapshot(provider: str, used: float | None) -> AccountUsageSnapshot:
    """Used percent = 0 means full headroom; 100 means no headroom."""
    return AccountUsageSnapshot(
        provider=provider,
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="window", used_percent=used),),
    )


def _exhausted_snapshot(provider: str) -> AccountUsageSnapshot:
    """Snapshot with 100% used — no headroom."""
    return _snapshot(provider, 100.0)


def _healthy_snapshot(provider: str) -> AccountUsageSnapshot:
    """Snapshot with 50% used — plenty of headroom."""
    return _snapshot(provider, 50.0)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated kanban DB fixture.

    Pins HERMES_KANBAN_DB directly so the fixture is safe even when the
    test file is executed outside the ``tests/`` tree (no conftest.py
    autouse guard) or when a dispatcher has injected HERMES_KANBAN_DB
    into the worker environment.  ``pin_kanban_db()`` strips ALL
    HERMES_KANBAN_* vars before setting its own pin, which is the
    highest-priority path in ``kanban_db_path()``.
    """
    pin_kanban_db(tmp_path, monkeypatch)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # HERMES_KANBAN_DB is pinned to tmp_path/kanban.db — verify before init.
    assert kb.kanban_db_path().resolve() == (tmp_path / "kanban.db").resolve(), (
        "kanban_db_path() must resolve inside tmp_path after pin_kanban_db()"
    )
    kb.init_db()
    return home


def _make_task(conn, title="test-task", assignee="myprofile"):
    t = kb.create_task(conn, title=title, assignee=assignee)
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (t,))
    conn.commit()
    return t


def _failure_count(conn, task_id: str) -> int:
    row = conn.execute(
        "SELECT consecutive_failures FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _status(conn, task_id: str) -> str:
    row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    return row[0] if row else "unknown"


def _meter_settings(
    *,
    enabled: bool = True,
    claude_model: str = "claude-sonnet-4-6",
    codex_model: str = "gpt-5.6-luna",
    minimum_remaining_percent: int = 10,
) -> dict:
    return {
        "kanban": {
            "provider_meter_remediation": {
                "enabled": enabled,
                "claude_model": claude_model,
                "codex_model": codex_model,
                "minimum_remaining_percent": minimum_remaining_percent,
            }
        }
    }


# ---------------------------------------------------------------------------
# RED tests: These must FAIL before the preflight feature is implemented
# ---------------------------------------------------------------------------

class TestQuotaPreflightReadyRetention:
    """quota=0(고갈) 상태에서 task가 ready 상태를 유지하는지 검증."""

    def test_quota_exhausted_task_stays_ready_not_claimed(self, kanban_home):
        """두 provider 모두 quota 고갈 시 task를 claim하지 않고 ready 유지."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        spawn_calls = []

        def fake_spawn(task, workspace, **_):
            spawn_calls.append(task.id)
            return 1234

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    side_effect=lambda p, **_: _exhausted_snapshot(p),
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        # task는 spawn되지 않아야 한다
        assert tid not in [t for t, _, _ in result.spawned], (
            "quota 고갈 시 task가 spawn되면 안 됨"
        )
        # spawn_fn이 호출되지 않아야 한다
        assert tid not in spawn_calls, "quota 고갈 시 spawn_fn이 호출되면 안 됨"

        # task는 ready 상태를 유지해야 한다
        with kb.connect() as conn:
            assert _status(conn, tid) == "ready", (
                "quota 고갈 시 task가 ready 상태를 유지해야 함"
            )

    def test_quota_exhausted_does_not_count_as_failure(self, kanban_home):
        """quota 고갈로 skip된 경우 failure counter가 증가하면 안 됨."""
        with kb.connect() as conn:
            tid = _make_task(conn)
            initial_failures = _failure_count(conn, tid)

        def fake_spawn(task, workspace, **_):
            return 1234

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    side_effect=lambda p, **_: _exhausted_snapshot(p),
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                kb.dispatch_once(conn, spawn_fn=fake_spawn)

        with kb.connect() as conn:
            after_failures = _failure_count(conn, tid)

        assert after_failures == initial_failures, (
            f"quota 고갈 skip 시 failure counter 불변이어야 함: "
            f"{initial_failures} -> {after_failures}"
        )

    def test_quota_exhausted_does_not_auto_block(self, kanban_home):
        """quota 고갈로 skip된 경우 auto-block되면 안 됨."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        def fake_spawn(task, workspace, **_):
            return 1234

        # failure_limit=1로 설정해도 quota skip은 auto-block 안 됨
        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    side_effect=lambda p, **_: _exhausted_snapshot(p),
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(conn, spawn_fn=fake_spawn, failure_limit=1)

        assert tid not in result.auto_blocked, "quota skip은 auto-block 유발하면 안 됨"

        with kb.connect() as conn:
            assert _status(conn, tid) == "ready", "task는 여전히 ready 상태여야 함"

    def test_usage_lookup_error_keeps_task_ready(self, kanban_home):
        """usage lookup이 None(오류) 반환 시 task를 claim하지 않고 ready 유지."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        def fake_spawn(task, workspace, **_):
            return 1234

        # fetch_account_usage가 None 반환 (lookup 실패 시뮬레이션)
        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    return_value=None,
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert tid not in [t for t, _, _ in result.spawned]

        with kb.connect() as conn:
            assert _status(conn, tid) == "ready"
            # failure counter도 불변
            assert _failure_count(conn, tid) == 0

    def test_claim_does_not_happen_before_preflight(self, kanban_home):
        """claim이 preflight 이전에 발생하지 않음: preflight 실패 시 claim_task 미호출."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        claimed_tasks = []
        original_claim = kb.claim_task

        def tracking_claim(c, task_id, **kw):
            claimed_tasks.append(task_id)
            return original_claim(c, task_id, **kw)

        def fake_spawn(task, workspace, **_):
            return 1234

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    side_effect=lambda p, **_: _exhausted_snapshot(p),
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
                patch("hermes_cli.kanban_db.claim_task", side_effect=tracking_claim),
            ):
                kb.dispatch_once(conn, spawn_fn=fake_spawn)

        # quota preflight 실패 시 claim_task를 아예 호출하지 않아야 한다
        assert tid not in claimed_tasks, (
            "quota preflight 실패 시 claim_task가 호출되면 안 됨 (claim-before-preflight)"
        )


class TestQuotaPreflightRecovery:
    """quota 회복 후 다음 tick에서 정상 spawn."""

    def test_recovery_next_tick_spawns_task(self, kanban_home):
        """첫 tick: quota 고갈 → skip. 두 번째 tick: quota 회복 → spawn."""
        # Provide credentials for the assignee profile so the strict
        # credential gate in _read_assignee_credentials() passes.
        # Without this the gate raises RuntimeError on both ticks and
        # the recovery assertion fails.
        profile_dir = kanban_home / "profiles" / "myprofile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / ".env").write_text("ANTHROPIC_TOKEN=test-sentinel-token\n")

        with kb.connect() as conn:
            tid = _make_task(conn)

        spawn_calls = []

        def fake_spawn(task, workspace, **_):
            spawn_calls.append(task.id)
            return 1234

        # 첫 tick: 고갈
        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    side_effect=lambda p, **_: _exhausted_snapshot(p),
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                r1 = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert tid not in [t for t, _, _ in r1.spawned]
        assert tid not in spawn_calls

        with kb.connect() as conn:
            assert _status(conn, tid) == "ready"

        # 두 번째 tick: 회복
        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    side_effect=lambda p, **_: _healthy_snapshot(p),
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                r2 = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert tid in [t for t, _, _ in r2.spawned], (
            "quota 회복 후 다음 tick에서 task가 spawn되어야 함"
        )
        assert tid in spawn_calls

    def test_multiple_quota_skips_never_accumulate_failures(self, kanban_home):
        """여러 번 quota skip해도 failure counter가 올라가지 않음."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        def fake_spawn(task, workspace, **_):
            return 1234

        for _ in range(5):
            with kb.connect() as conn:
                with (
                    patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                    patch(
                        "agent.account_usage.fetch_account_usage",
                        side_effect=lambda p, **_: _exhausted_snapshot(p),
                    ),
                    patch("hermes_cli.profiles.profile_exists", return_value=True),
                ):
                    kb.dispatch_once(conn, spawn_fn=fake_spawn)

        with kb.connect() as conn:
            assert _failure_count(conn, tid) == 0, (
                "5번 quota skip 후에도 failure counter가 0이어야 함"
            )
            assert _status(conn, tid) == "ready"


class TestQuotaPreflightDisabled:
    """provider_meter_remediation 비활성 시 preflight 생략 → 기존 동작."""

    def test_disabled_remediation_does_not_affect_spawn(self, kanban_home):
        """remediation disabled → preflight 없음, 정상 spawn."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        spawn_calls = []

        def fake_spawn(task, workspace, **_):
            spawn_calls.append(task.id)
            return 1234

        disabled_settings = _meter_settings(enabled=False)
        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=disabled_settings),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        # remediation 비활성이면 quota check 없이 spawn
        assert tid in [t for t, _, _ in result.spawned]
        assert tid in spawn_calls

    def test_no_config_does_not_affect_spawn(self, kanban_home):
        """config 없음 → preflight 생략, 정상 spawn."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        spawn_calls = []

        def fake_spawn(task, workspace, **_):
            spawn_calls.append(task.id)
            return 1234

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value={}),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert tid in [t for t, _, _ in result.spawned]
        assert tid in spawn_calls


class TestQuotaPreflightDispatchResult:
    """DispatchResult에 quota preflight skip이 노출되는지 확인."""

    def test_skipped_quota_preflight_field_exists_in_dispatch_result(self):
        """DispatchResult에 skipped_quota_preflight 필드가 있어야 함."""
        result = kb.DispatchResult()
        assert hasattr(result, "skipped_quota_preflight"), (
            "DispatchResult에 skipped_quota_preflight 필드가 없음 — 구현 필요"
        )
        assert isinstance(result.skipped_quota_preflight, list), (
            "skipped_quota_preflight는 list여야 함"
        )

    def test_skipped_quota_preflight_populated_on_quota_exhausted(self, kanban_home):
        """quota 고갈 시 result.skipped_quota_preflight에 task_id가 포함되어야 함."""
        with kb.connect() as conn:
            tid = _make_task(conn)

        def fake_spawn(task, workspace, **_):
            return 1234

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch(
                    "agent.account_usage.fetch_account_usage",
                    side_effect=lambda p, **_: _exhausted_snapshot(p),
                ),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert tid in result.skipped_quota_preflight, (
            f"quota 고갈 시 task {tid}가 skipped_quota_preflight에 있어야 함, "
            f"실제: {result.skipped_quota_preflight}"
        )


# ---------------------------------------------------------------------------
# Isolation regression tests
# ---------------------------------------------------------------------------
# These two tests verify the fix itself so future regressions are caught
# immediately rather than being discovered by auditing the real kanban.db.


class TestKanbanDBIsolation:
    """DB isolation invariant tests — catch regression before real DB leak."""

    def test_fixture_db_is_inside_tmp_path(self, kanban_home, tmp_path):
        """After fixture setup, kanban_db_path() must resolve inside tmp_path.

        This is the in-tree sanity check: it would have caught the original
        accident where HERMES_KANBAN_HOME-only isolation let a dispatcher-
        injected HERMES_KANBAN_DB override the fixture.
        """
        db = kb.kanban_db_path().resolve()
        assert str(db).startswith(str(tmp_path.resolve())), (
            f"kanban_db_path() {db} is NOT under tmp_path {tmp_path} — "
            "the fixture isolation is broken and tests may write to the real DB"
        )
        # The DB file must also actually exist under tmp_path (init_db ran).
        assert db.exists(), "kanban.db was not created inside tmp_path"

    def test_conftest_bypass_does_not_write_to_copy_db(self, tmp_path):
        """Simulate the dispatcher 'conftest bypass' scenario end-to-end.

        The dispatcher runs the test file from outside tests/ so conftest.py
        autouse fixtures never fire.  With the old fixture, dispatcher env
        vars (HERMES_KANBAN_DB) bled into the test process and wrote to the
        real DB.  This test reproduces that exact path using a subprocess.

        Procedure:
        1. Create a fresh copy DB (stands in for the real ~/.hermes/kanban.db).
        2. Copy this test file to a temp directory (so conftest.py is absent).
        3. Run pytest on the copy with HERMES_KANBAN_DB pointing at the copy DB
           (simulating dispatcher env injection).
        4. Assert the copy DB task count is unchanged after the run.
        """
        # 1. Copy DB: empty kanban DB acts as stand-in for real DB.
        copy_db = tmp_path / "copy_kanban.db"
        import hermes_cli.kanban_db as _kb_fresh  # noqa: PLC0415
        env_before = dict(os.environ)
        env_before["HERMES_KANBAN_DB"] = str(copy_db)
        # init the copy DB so we can count tasks
        import sqlite3 as _sqlite3  # noqa: PLC0415
        with _sqlite3.connect(str(copy_db)) as _c:
            # minimal schema: just enough for SELECT count(*) FROM tasks
            _c.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                "id TEXT PRIMARY KEY, title TEXT, status TEXT, assignee TEXT,"
                " created_at INTEGER, consecutive_failures INTEGER DEFAULT 0)"
            )
            _c.commit()
        count_before = _sqlite3.connect(
            f"file:{copy_db}?mode=ro", uri=True
        ).execute("SELECT count(*) FROM tasks").fetchone()[0]

        # 2. Write a minimal self-contained test file to a temp directory that
        #    contains no conftest.py.  It calls kb.init_db() + kb.connect() so
        #    it would write to whichever DB kanban_db_path() resolves to.
        isolated_dir = tmp_path / "isolated_run"
        isolated_dir.mkdir()
        test_file = isolated_dir / "test_isolated_quota.py"
        test_file.write_text(
            textwrap.dedent("""\
                \"\"\"Minimal isolation check: runs outside tests/ with no conftest.py.\"\"\"
                import os, sys
                from pathlib import Path
                sys.path.insert(0, os.environ["HERMES_PROJECT_ROOT"])

                import hermes_cli.kanban_db as kb

                def test_db_inside_tmp(tmp_path, monkeypatch):
                    from tests.hermes_cli._kanban_isolation import pin_kanban_db
                    pin_kanban_db(tmp_path, monkeypatch)
                    home = tmp_path / ".hermes"
                    home.mkdir()
                    monkeypatch.setenv("HERMES_HOME", str(home))
                    monkeypatch.setattr(Path, "home", lambda: tmp_path)
                    kb.init_db()
                    with kb.connect() as conn:
                        kb.create_task(conn, title="isolated-test-task", assignee="x")
                    db = kb.kanban_db_path()
                    assert str(db.resolve()).startswith(str(tmp_path.resolve())), (
                        f"DB {db} is outside tmp_path {tmp_path}"
                    )
            """)
        )

        # 3. Run subprocess with dispatcher-style env (HERMES_KANBAN_DB pointing
        #    at copy_db — this is what a real dispatcher worker would inject).
        project_root = str(
            Path(__file__).parent.parent.parent.resolve()
        )
        sub_env = {
            k: v for k, v in os.environ.items()
            # Strip conftest-stripped vars to simulate a clean worker env,
            # then inject the dispatcher vars.
            if not k.startswith("HERMES_KANBAN_")
        }
        sub_env["HERMES_KANBAN_DB"] = str(copy_db)  # dispatcher injection
        sub_env["HERMES_PROJECT_ROOT"] = project_root
        sub_env["PYTHONPATH"] = project_root

        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-q",
                "-p", "no:cacheprovider",
            ],
            cwd=str(isolated_dir),
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # 4. Count tasks in the copy DB after the run.
        count_after = _sqlite3.connect(
            f"file:{copy_db}?mode=ro", uri=True
        ).execute("SELECT count(*) FROM tasks").fetchone()[0]

        # The subprocess must have passed (the fix works).
        assert result.returncode == 0, (
            f"Isolated pytest returned {result.returncode}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        # The copy DB must be untouched (no tasks leaked into it).
        assert count_after == count_before, (
            f"Isolated run wrote to the copy DB! "
            f"tasks: {count_before} → {count_after}.\n"
            "This means the DB isolation fix is broken and real DB leaks "
            "would occur in dispatcher-spawned pytest sessions."
        )
