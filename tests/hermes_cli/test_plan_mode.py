"""Tests for hermes_cli/plan_mode.py — code-enforced plan mode."""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import plan_mode

    plan_mode._DB_CACHE.clear()
    yield home
    plan_mode._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# PlanState + persistence
# ──────────────────────────────────────────────────────────────────────


class TestPlanState:
    def test_roundtrip_json(self):
        from hermes_cli.plan_mode import PlanState

        s = PlanState(status="planning", plan_path=".hermes/plans/p.md", entered_at=1.0,
                      entered_by="user", approval_clarify_id="c1")
        s2 = PlanState.from_json(s.to_json())
        assert s2 == s

    def test_from_json_defaults(self):
        from hermes_cli.plan_mode import PlanState, STATUS_PLANNING

        s = PlanState.from_json("{}")
        assert s.status == STATUS_PLANNING
        assert s.plan_path is None

    def test_is_restricted(self):
        from hermes_cli.plan_mode import PlanState

        assert PlanState(status="planning").is_restricted()
        assert PlanState(status="pending_approval").is_restricted()
        assert not PlanState(status="approved").is_restricted()
        assert not PlanState(status="off").is_restricted()


class TestPersistence:
    def test_save_load_roundtrip(self, hermes_home):
        from hermes_cli.plan_mode import PlanState, save_plan, load_plan

        save_plan("sid-1", PlanState(status="planning", plan_path="x.md"))
        got = load_plan("sid-1")
        assert got is not None
        assert got.status == "planning"
        assert got.plan_path == "x.md"

    def test_load_missing_is_none(self, hermes_home):
        from hermes_cli.plan_mode import load_plan

        assert load_plan("nope") is None

    def test_persistence_across_managers(self, hermes_home):
        """A second PlanManager for the same session sees the saved state."""
        from hermes_cli.plan_mode import PlanManager

        mgr1 = PlanManager("persist-sid")
        mgr1.enter()
        mgr1.set_plan_path(".hermes/plans/p.md")

        mgr2 = PlanManager("persist-sid")
        assert mgr2.state is not None
        assert mgr2.state.status == "planning"
        assert mgr2.state.plan_path == ".hermes/plans/p.md"
        assert mgr2.is_active()


# ──────────────────────────────────────────────────────────────────────
# Manager transitions
# ──────────────────────────────────────────────────────────────────────


class TestPlanManager:
    def test_enter_then_approve_lifts_restriction(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, load_plan

        mgr = PlanManager("sid-a")
        mgr.enter()
        assert mgr.is_active()
        mgr.approve()
        assert not mgr.is_active()
        assert load_plan("sid-a").status == "approved"

    def test_request_approval_then_keep_planning(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager

        mgr = PlanManager("sid-b")
        mgr.enter()
        mgr.request_approval(clarify_id="c9")
        assert mgr.state.status == "pending_approval"
        assert mgr.state.approval_clarify_id == "c9"
        mgr.keep_planning()
        assert mgr.state.status == "planning"
        assert mgr.state.approval_clarify_id is None
        assert mgr.is_active()

    def test_exit_discards_and_never_approves(self, hermes_home):
        """/plan exit writes the `off` tombstone — it must NEVER approve."""
        from hermes_cli.plan_mode import PlanManager, load_plan, STATUS_APPROVED

        mgr = PlanManager("sid-c")
        mgr.enter()
        mgr.request_approval()
        had = mgr.exit()
        assert had is True
        state = load_plan("sid-c")
        assert state is not None
        assert state.status != STATUS_APPROVED
        assert state.status == "off"
        assert not PlanManager("sid-c").is_active()


# ──────────────────────────────────────────────────────────────────────
# Dispatch guard matrix (layer b)
# ──────────────────────────────────────────────────────────────────────


class TestToolBlockReason:
    def test_mutating_blocked_in_planning(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, tool_block_reason

        PlanManager("g1").enter()
        assert tool_block_reason("g1", "terminal", {"command": "ls"}) is not None
        assert tool_block_reason("g1", "delegate_task", {}) is not None

    def test_mutating_blocked_in_pending_approval(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, tool_block_reason

        mgr = PlanManager("g2")
        mgr.enter()
        mgr.request_approval()
        assert tool_block_reason("g2", "terminal", {"command": "ls"}) is not None

    def test_read_only_allowed_in_planning(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, tool_block_reason

        PlanManager("g3").enter()
        assert tool_block_reason("g3", "read_file", {"path": "a.py"}) is None
        assert tool_block_reason("g3", "search_files", {}) is None

    def test_plan_file_write_allowed(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, tool_block_reason

        PlanManager("g4").enter()
        assert tool_block_reason("g4", "write_file", {"path": ".hermes/plans/p.md"}) is None
        # Nested UNDER the task's plans dir is still allowed.
        assert tool_block_reason("g4", "patch", {"path": ".hermes/plans/feature/p.md"}) is None

    def test_non_plan_file_write_blocked(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, tool_block_reason

        PlanManager("g5").enter()
        assert tool_block_reason("g5", "write_file", {"path": "src/app.py"}) is not None

    def test_not_in_plan_mode_allows_all(self, hermes_home):
        from hermes_cli.plan_mode import tool_block_reason

        assert tool_block_reason("nosession-xyz", "terminal", {"command": "ls"}) is None

    def test_approved_allows_all(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, tool_block_reason

        mgr = PlanManager("g6")
        mgr.enter()
        mgr.approve()
        assert tool_block_reason("g6", "terminal", {"command": "ls"}) is None

    def test_unreadable_state_fails_closed(self, hermes_home, monkeypatch):
        """If a plan row exists but cannot be read, mutating tools fail closed."""
        from hermes_cli import plan_mode

        monkeypatch.setattr(
            plan_mode, "_read_plan", lambda sid: (plan_mode._READ_UNREADABLE, None)
        )
        assert plan_mode.tool_block_reason("gx", "terminal", {"command": "ls"}) is not None
        # Read-only tools are never blocked, even fail-closed.
        assert plan_mode.tool_block_reason("gx", "read_file", {"path": "a"}) is None


class TestIsPlanPath:
    """Resolved-path containment — traversal and symlink escapes must fail.

    ``file_tools`` resolves relative paths against the TASK's base dir (its live
    terminal cwd / registered workspace override / ``TERMINAL_CWD``) and follows
    symlinks before writing. ``is_plan_path`` reuses that same task-aware
    resolver for the candidate AND the plans-dir root, so the guard proves
    containment for exactly the path the write will produce — the base and its
    symlink resolution cannot diverge from the write's. Containment is required
    against the task's real ``<task_base>/.hermes/plans``.
    """

    def test_legit_nested_plan_path_passes(self, tmp_path, monkeypatch):
        from hermes_cli.plan_mode import is_plan_path

        monkeypatch.chdir(tmp_path)
        # Relative and nested forms that genuinely live under the plans dir.
        assert is_plan_path(".hermes/plans/p.md") is True
        assert is_plan_path(".hermes/plans/feature/step-1/plan.md") is True

    def test_plans_segment_not_at_task_root_is_rejected(self, tmp_path, monkeypatch):
        """A ``.hermes/plans`` nested under an arbitrary subdir is NOT the task's
        plans dir. The guard now anchors to ``<task_base>/.hermes/plans``, so
        writes into ``work/.hermes/plans`` (a bypass the old string-derived-root
        matcher allowed) are blocked."""
        from hermes_cli.plan_mode import is_plan_path

        monkeypatch.chdir(tmp_path)
        assert is_plan_path("work/.hermes/plans/p.md") is False
        assert is_plan_path("src/module/.hermes/plans/payload.py") is False

    def test_dotdot_traversal_escapes_and_is_rejected(self, tmp_path, monkeypatch):
        from hermes_cli.plan_mode import is_plan_path

        monkeypatch.chdir(tmp_path)
        # Contains the ".hermes/plans" segment but resolves OUTSIDE it.
        assert is_plan_path(".hermes/plans/../../src/app.py") is False
        assert is_plan_path(".hermes/plans/../plans-evil/app.py") is False

    def test_absolute_path_outside_is_rejected(self, tmp_path, monkeypatch):
        from hermes_cli.plan_mode import is_plan_path

        monkeypatch.chdir(tmp_path)
        # No ".hermes/plans" segment at all → not under the task's plans dir.
        assert is_plan_path("/etc/passwd") is False
        assert is_plan_path("/tmp/src/app.py") is False

    def test_symlink_inside_plans_dir_pointing_outside_is_rejected(self, tmp_path, monkeypatch):
        from hermes_cli.plan_mode import is_plan_path

        # Anchor the task base to tmp_path so the resolved root is tmp_path's
        # plans dir (matching where the symlink structure is built).
        monkeypatch.chdir(tmp_path)
        plans = tmp_path / ".hermes" / "plans"
        plans.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        # A symlink placed INSIDE the plans dir that points outside it: a write
        # "under" the plans dir would land in `outside/` after resolution.
        link = plans / "escape"
        link.symlink_to(outside, target_is_directory=True)

        # The literal string sits under ".hermes/plans", but resolves outside.
        escaped = str(link / "app.py")
        assert ".hermes/plans" in escaped.replace("\\", "/")
        assert is_plan_path(escaped) is False

        # A real (non-symlinked) sibling under the plans dir still passes.
        legit = plans / "real" / "plan.md"
        assert is_plan_path(str(legit)) is True

    def test_guard_anchors_to_task_base_not_process_cwd(self, tmp_path, monkeypatch):
        """Core base-divergence regression.

        The write anchors relative paths to the TASK base (``TERMINAL_CWD`` /
        worktree), not the agent PROCESS cwd. If the guard resolved against the
        process cwd it would prove containment in the WRONG tree: here a
        ``.hermes/plans/escape`` symlink exists only at the TASK base, so a guard
        that anchored to the process cwd (where that path is a benign lexical
        non-existent) would wrongly ALLOW a write that actually escapes at the
        task base. The guard must anchor to the task base and block it.
        """
        from hermes_cli.plan_mode import is_plan_path

        process_cwd = tmp_path / "proc"
        task_cwd = tmp_path / "task"
        # A real, benign plans dir at the PROCESS cwd.
        (process_cwd / ".hermes" / "plans").mkdir(parents=True)
        # At the TASK base: a plans dir whose `escape` entry symlinks OUT of it.
        task_plans = task_cwd / ".hermes" / "plans"
        task_plans.mkdir(parents=True)
        (task_cwd / "outside").mkdir()
        (task_plans / "escape").symlink_to(task_cwd / "outside", target_is_directory=True)

        monkeypatch.chdir(process_cwd)
        monkeypatch.setenv("TERMINAL_CWD", str(task_cwd))

        # A genuine plan write resolves inside the TASK's plans dir → allowed.
        assert is_plan_path(".hermes/plans/plan.md") is True
        # The escape only exists at the task base; a process-cwd-anchored guard
        # would MISS it. Anchored to the task base, it is correctly blocked.
        assert is_plan_path(".hermes/plans/escape/app.py") is False

    def test_symlinked_plans_dir_at_task_base_stays_consistent(self, tmp_path, monkeypatch):
        """When the plans dir itself is a symlink at the task base, the guard and
        the write resolve it identically: genuine plan writes are contained and
        traversal out is still blocked."""
        from hermes_cli.plan_mode import is_plan_path

        task_cwd = tmp_path / "task"
        (task_cwd / ".hermes").mkdir(parents=True)
        real_plans = task_cwd / "real_plans_elsewhere"
        real_plans.mkdir()
        # `.hermes/plans` is a symlink to a dir elsewhere under the task base.
        (task_cwd / ".hermes" / "plans").symlink_to(real_plans, target_is_directory=True)

        monkeypatch.chdir(task_cwd)
        monkeypatch.setenv("TERMINAL_CWD", str(task_cwd))

        # Root and candidate both follow the symlink → contained → allowed.
        assert is_plan_path(".hermes/plans/plan.md") is True
        # Traversal out of the (symlinked) plans dir is still rejected.
        assert is_plan_path(".hermes/plans/../secret.py") is False

    def test_unresolvable_path_fails_closed(self, tmp_path, monkeypatch):
        """A resolver failure must fail closed (block), never allow."""
        import hermes_cli.plan_mode as plan_mode

        def _boom(*_a, **_k):
            raise OSError("symlink loop")

        monkeypatch.setattr(
            "tools.file_tools._resolve_path_for_task", _boom, raising=True
        )
        assert plan_mode.is_plan_path(".hermes/plans/p.md") is False


# ──────────────────────────────────────────────────────────────────────
# Toolset policy (layer a)
# ──────────────────────────────────────────────────────────────────────


class TestToolsetPolicy:
    def test_session_disabled_toolsets_are_mutating_only(self):
        from hermes_cli.plan_mode import session_disabled_toolsets

        disabled = session_disabled_toolsets()
        assert "terminal" in disabled
        assert "code_execution" in disabled
        assert "delegation" in disabled
        # file/skills/browser mix read-only tools and must stay enabled.
        assert "file" not in disabled
        assert "skills" not in disabled
        assert "browser" not in disabled

    def test_policy_pass_through_when_inactive(self, hermes_home):
        from hermes_cli.plan_mode import apply_session_toolset_policy

        en, dis = apply_session_toolset_policy("inactive-sid", ["file", "terminal"], None)
        assert en == ["file", "terminal"]
        assert dis is None

    def test_policy_restricts_when_active(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, apply_session_toolset_policy, PLAN_TOOLSET

        PlanManager("act-sid").enter()
        en, dis = apply_session_toolset_policy("act-sid", ["file", "terminal"], None)
        assert PLAN_TOOLSET in en
        assert "terminal" in dis
        assert "code_execution" in dis

    def test_policy_none_enabled_stays_none(self, hermes_home):
        from hermes_cli.plan_mode import PlanManager, apply_session_toolset_policy

        PlanManager("act2").enter()
        en, dis = apply_session_toolset_policy("act2", None, None)
        assert en is None  # None means "all" — don't collapse to only plan
        assert "terminal" in dis


# ──────────────────────────────────────────────────────────────────────
# config default
# ──────────────────────────────────────────────────────────────────────


class TestConfigDefault:
    def test_always_materialises_planning(self, hermes_home, monkeypatch):
        from hermes_cli import plan_mode

        monkeypatch.setattr(plan_mode, "_config_plan_mode", lambda: "always")
        state = plan_mode.ensure_default_for_session("fresh-sid")
        assert state is not None
        assert state.status == "planning"
        assert state.entered_by == "config"
        assert plan_mode.PlanManager("fresh-sid").is_active()

    def test_default_off_no_row(self, hermes_home, monkeypatch):
        from hermes_cli import plan_mode

        monkeypatch.setattr(plan_mode, "_config_plan_mode", lambda: None)
        assert plan_mode.ensure_default_for_session("fresh-2") is None
        assert plan_mode.load_plan("fresh-2") is None

    def test_always_does_not_override_existing(self, hermes_home, monkeypatch):
        from hermes_cli import plan_mode

        # Session already exited plan mode; `always` must not re-enter it.
        plan_mode.PlanManager("exited-sid").exit()
        monkeypatch.setattr(plan_mode, "_config_plan_mode", lambda: "always")
        assert plan_mode.ensure_default_for_session("exited-sid") is None
        assert not plan_mode.PlanManager("exited-sid").is_active()
