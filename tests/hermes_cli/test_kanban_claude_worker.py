"""Tests for the direct Claude Code worker lane (kanban_claude_worker).

Covers the handoff's required test surface:

1. Assignee routing — ``claude-code`` routes to the external lane and is NOT
   bucketed nonspawnable; a bogus non-profile assignee still is.
2. Prompt builder — includes task id/title/body/workspace + no-commit/push/
   deploy/secrets constraints.
3. Spawn command — uses ``claude -p`` / bounded ``--max-turns`` / JSON output
   and never ``--dangerously-skip-permissions``.
4. Failure handling — Claude nonzero exit / verification failure record a
   retryable failure (requeue, then auto-block at the limit).
5. Success handling — completes only when verification ran AND passed against
   a real diff; otherwise blocks for manual review.

All Claude / shell / git interactions are injected stubs — no real ``claude``
binary or subprocess is touched.
"""
from __future__ import annotations

import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB (mirrors the suite's pattern)."""
    test_home = tempfile.mkdtemp(prefix="kanban_claude_worker_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    from hermes_cli import kanban_claude_worker
    yield kanban_db, kanban_claude_worker, test_home


def _fake_spawn(*args, **kwargs):
    return 4242


# ---------------------------------------------------------------------------
# 1. Routing
# ---------------------------------------------------------------------------


def test_is_external_claude_worker_recognises_sentinel():
    from hermes_cli import kanban_claude_worker as w
    assert w.is_external_claude_worker("claude-code") is True
    assert w.is_external_claude_worker("Claude-Code") is True   # case-insensitive
    assert w.is_external_claude_worker(" claude-code ") is True  # trims
    assert w.is_external_claude_worker("coder") is False
    assert w.is_external_claude_worker(None) is False
    assert w.is_external_claude_worker("") is False


def test_dispatcher_routes_claude_code_assignee(isolated_kanban_home):
    """A ``claude-code`` card is claimed + spawned (bypasses profile_exists),
    while a genuinely non-profile assignee is still bucketed nonspawnable."""
    kb, _w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        claude_id = kb.create_task(conn, title="claude task", assignee="claude-code")
        bogus_id = kb.create_task(conn, title="bogus task", assignee="not-a-real-profile")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)

    spawned_ids = [t[0] for t in res.spawned]
    assert claude_id in spawned_ids, "claude-code card must route to the external lane"
    assert claude_id not in res.skipped_nonspawnable
    # The bogus profile assignee remains rejected (existing behaviour intact).
    assert bogus_id in res.skipped_nonspawnable
    assert bogus_id not in spawned_ids


# ---------------------------------------------------------------------------
# 2. Prompt builder
# ---------------------------------------------------------------------------


class _Task:
    """Minimal stand-in carrying only the fields the worker reads."""
    def __init__(self, id="t_abc", title="Fix bug", body="do the thing",
                 current_run_id=None, max_runtime_seconds=None):
        self.id = id
        self.title = title
        self.body = body
        self.current_run_id = current_run_id
        self.max_runtime_seconds = max_runtime_seconds


def test_build_claude_code_prompt_includes_context_and_constraints():
    from hermes_cli import kanban_claude_worker as w
    task = _Task(id="t_xyz", title="Add docstring", body="## Acceptance\n- [ ] docstring")
    prompt = w.build_claude_code_prompt(task, "/abs/workspace")

    assert "t_xyz" in prompt
    assert "Add docstring" in prompt
    assert "## Acceptance" in prompt
    assert "/abs/workspace" in prompt
    low = prompt.lower()
    for forbidden in ("commit", "push", "deploy", "secrets"):
        assert forbidden in low, f"prompt must forbid {forbidden}"
    # The runner owns verification: Claude must be told NOT to run it itself,
    # NOT to use Bash to verify, and to leave verification to the runner.
    assert "do not run the verification command yourself" in low
    assert "do not use bash just to verify" in low
    assert "leave verification to it" in low


def test_extract_verification_command():
    from hermes_cli import kanban_claude_worker as w
    body = "## Repo\n/x\n\n## Verification\npython -m py_compile main.py\n\n## Constraints\nno commit"
    assert w.extract_verification_command(body) == "python -m py_compile main.py"
    # fenced block: keep contents, drop the fences
    fenced = "## Verification\n```\npytest -q\n```\n"
    assert w.extract_verification_command(fenced) == "pytest -q"
    assert w.extract_verification_command("just a body, no section") is None
    assert w.extract_verification_command(None) is None


# ---------------------------------------------------------------------------
# 3. Spawn command safety
# ---------------------------------------------------------------------------


def test_build_claude_cmd_is_bounded_and_safe():
    from hermes_cli import kanban_claude_worker as w
    cmd = w.build_claude_cmd("PROMPT", {"max_turns": 7, "model": "claude-opus-4-8"})
    assert cmd[0] == "claude"
    assert "-p" in cmd and "PROMPT" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--max-turns" in cmd and cmd[cmd.index("--max-turns") + 1] == "7"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "claude-opus-4-8"
    # Hard safety guard.
    assert "--dangerously-skip-permissions" not in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"


def test_build_claude_cmd_downgrades_bypass_permission_mode():
    from hermes_cli import kanban_claude_worker as w
    for bad in ("bypassPermissions", "bypass", "dangerously-skip-permissions"):
        cmd = w.build_claude_cmd("P", {"permission_mode": bad})
        assert "--dangerously-skip-permissions" not in cmd
        assert cmd[cmd.index("--permission-mode") + 1] == w.DEFAULT_PERMISSION_MODE


def test_build_claude_cmd_defaults_invalid_max_turns():
    from hermes_cli import kanban_claude_worker as w
    cmd = w.build_claude_cmd("P", {"max_turns": "garbage"})
    assert cmd[cmd.index("--max-turns") + 1] == str(w.DEFAULT_MAX_TURNS)
    cmd2 = w.build_claude_cmd("P", {"max_turns": 0})
    assert cmd2[cmd2.index("--max-turns") + 1] == str(w.DEFAULT_MAX_TURNS)


def test_parse_claude_json_tolerates_noise():
    from hermes_cli import kanban_claude_worker as w
    assert w.parse_claude_json('{"is_error": false, "result": "ok"}')["result"] == "ok"
    assert w.parse_claude_json('warning\n{"result": "x"}')["result"] == "x"
    assert w.parse_claude_json("not json at all") == {}
    assert w.parse_claude_json("") == {}


# ---------------------------------------------------------------------------
# 4 + 5. run_worker outcomes (real DB, injected claude/verify/diff)
# ---------------------------------------------------------------------------


def _claimed_task(kb, conn, **kw):
    """Create a ready task and claim it -> running with an open run."""
    body = kw.pop("body", "## Verification\npython -m py_compile main.py")
    tid = kb.create_task(conn, title=kw.pop("title", "t"), assignee="claude-code",
                         body=body, **kw)
    return kb.claim_task(conn, tid)


def _ok_claude(cmd=None, **_kw):
    from hermes_cli import kanban_claude_worker as w
    return w.ClaudeOutcome(returncode=0, stdout='{"is_error": false, "result": "done"}',
                           result_text="done", is_error=False, subtype="success", num_turns=3)


def _fail_claude(cmd=None, **_kw):
    from hermes_cli import kanban_claude_worker as w
    return w.ClaudeOutcome(returncode=1, stdout="", stderr="boom", is_error=True,
                           subtype="error_during_execution")


def _maxturns_claude(cmd=None, **_kw):
    """Reproduces the real-smoke abnormal exit: Claude burned its turns being
    denied Bash for a verification it should never have run."""
    from hermes_cli import kanban_claude_worker as w
    return w.ClaudeOutcome(
        returncode=1,
        stdout='{"is_error": true, "subtype": "error_max_turns", "num_turns": 6, "result": "hit max turns"}',
        stderr="permission denied: Bash", result_text="hit max turns",
        is_error=True, subtype="error_max_turns", num_turns=6,
    )


def test_run_worker_success_completes(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)
        assert task is not None

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_ok_claude,
            run_verification=lambda command, **_k: w.VerificationResult(
                ran=True, command=command, returncode=0, output="compiled"),
            diff_summary=lambda _ws: ("main.py | 1 +", ["main.py"]),
        )
        assert res.status == "completed"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task.id,)).fetchone()
        assert row["status"] == "done"
        comments = kb.list_comments(conn, task.id)
        assert any("COMPLETE" in c.body for c in comments)


def test_run_worker_verification_failure_requeues(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_ok_claude,
            run_verification=lambda command, **_k: w.VerificationResult(
                ran=True, command=command, returncode=1, output="FAILED"),
            diff_summary=lambda _ws: ("main.py | 1 +", ["main.py"]),
        )
        assert res.status == "failed"
        assert res.auto_blocked is False
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (task.id,)
        ).fetchone()
        # Below the limit -> requeued to ready with the counter bumped.
        assert row["status"] == "ready"
        assert row["consecutive_failures"] == 1
        assert any("verification FAILED" in c.body for c in kb.list_comments(conn, task.id))


def test_run_worker_verification_failure_autoblocks_at_limit(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=1,
            run_claude=_ok_claude,
            run_verification=lambda command, **_k: w.VerificationResult(
                ran=True, command=command, returncode=2, output="nope"),
            diff_summary=lambda _ws: ("main.py | 1 +", ["main.py"]),
        )
        assert res.status == "failed"
        assert res.auto_blocked is True
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task.id,)).fetchone()
        assert row["status"] == "blocked"


def test_run_worker_claude_failure_records_failure(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)

        verify_calls = []
        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_fail_claude,
            run_verification=lambda command, **_k: verify_calls.append(command),
            diff_summary=lambda _ws: ("", []),
        )
        assert res.status == "failed"
        assert verify_calls == [], "verification must NOT run after a Claude failure"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task.id,)).fetchone()
        assert row["status"] == "ready"  # retryable, under the limit
        assert any("FAILED" in c.body for c in kb.list_comments(conn, task.id))


def test_run_worker_no_verification_blocks_for_review(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn, body="## Acceptance\n- [ ] do it (no verification section)")

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_ok_claude,
            run_verification=lambda *a, **k: pytest.fail("verification should not run"),
            diff_summary=lambda _ws: ("main.py | 1 +", ["main.py"]),
        )
        assert res.status == "blocked_manual_review"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task.id,)).fetchone()
        assert row["status"] == "blocked"
        assert any("needs review" in c.body for c in kb.list_comments(conn, task.id))


def test_run_worker_empty_diff_blocks_for_review(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_ok_claude,
            run_verification=lambda command, **_k: w.VerificationResult(
                ran=True, command=command, returncode=0, output="ok"),
            diff_summary=lambda _ws: ("", []),  # Claude claimed success but no diff
        )
        assert res.status == "blocked_manual_review"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task.id,)).fetchone()
        assert row["status"] == "blocked"


# ---------------------------------------------------------------------------
# Abnormal-Claude-exit policy (the real-smoke regression: error_max_turns from
# Claude trying to run verification it should never have run).
# ---------------------------------------------------------------------------


def test_abnormal_claude_but_verification_passes_blocks_for_review(isolated_kanban_home):
    """THE regression: Claude hit error_max_turns, but it left a real diff and
    the runner's own verification passes. Must NOT retry/gave_up and must NOT
    auto-complete — block for manual review."""
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_maxturns_claude,
            run_verification=lambda command, **_k: w.VerificationResult(
                ran=True, command=command, returncode=0, output="compiled"),
            diff_summary=lambda _ws: ("main.py | 1 +", ["main.py"]),
        )
        assert res.status == "blocked_manual_review"
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (task.id,)
        ).fetchone()
        assert row["status"] == "blocked"
        # block_task is terminal-for-review, NOT a recorded failure: the
        # failure counter must stay at 0 so this never counts toward gave_up.
        assert row["consecutive_failures"] == 0
        body = "\n".join(c.body for c in kb.list_comments(conn, task.id))
        assert "manual review" in body.lower()
        assert "runner verification passed against a real diff" in body.lower()


def test_abnormal_claude_no_diff_retries(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)

        verify_calls = []
        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_maxturns_claude,
            run_verification=lambda command, **_k: verify_calls.append(command),
            diff_summary=lambda _ws: ("", []),  # abnormal AND nothing produced
        )
        assert res.status == "failed"
        assert verify_calls == [], "no diff -> nothing to verify"
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (task.id,)
        ).fetchone()
        assert row["status"] == "ready"  # retryable under the limit
        assert row["consecutive_failures"] == 1


def test_abnormal_claude_diff_verification_fails_retries(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn)

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_maxturns_claude,
            run_verification=lambda command, **_k: w.VerificationResult(
                ran=True, command=command, returncode=1, output="syntax error"),
            diff_summary=lambda _ws: ("main.py | 1 +", ["main.py"]),
        )
        assert res.status == "failed"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task.id,)).fetchone()
        assert row["status"] == "ready"  # retryable: real verification failed


def test_abnormal_claude_diff_no_verification_blocks_for_review(isolated_kanban_home):
    kb, w, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task = _claimed_task(kb, conn, body="## Acceptance\n- [ ] do it (no verification section)")

        res = w.run_worker(
            conn, task, "/tmp/ws", board="default", failure_limit=5,
            run_claude=_maxturns_claude,
            run_verification=lambda *a, **k: pytest.fail("no verification command -> must not run"),
            diff_summary=lambda _ws: ("main.py | 1 +", ["main.py"]),
        )
        assert res.status == "blocked_manual_review"
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task.id,)).fetchone()
        assert row["status"] == "blocked"
