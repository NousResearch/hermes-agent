"""Review-lane skill guard + crash-output extractor anchoring.

Incident (board ``skill-ownership-plugin``, card t_bbaf8adc, investigated in
t_bc9a316f): every review-lane claim on a profile whose home lacks the
force-loaded ``sdlc-review`` skill died in CLI boot — ``Error: Unknown
skill(s): sdlc-review`` — and crash-looped until the failure budget tripped.
The crash-output extractor (``_worker_final_output``) then attributed the
PREVIOUS cleanly-exited run's final chat to the crashed runs: it cut the
shared append-mode log at the predecessor's ``Resume this session with:``
marker and discarded the crash stanzas after it, manufacturing a phantom
"reviewer re-authored a conclusion" narrative.

These tests pin the two repairs:

* the review lane resolves the forced skill under the assignee profile's home
  (same loader as CLI boot) and auto-blocks the card instead of spawning a
  guaranteed crash loop;
* the extractor anchors its window to the END of the exit summary's last line
  (``Messages:``), so a pre-chat crash reports its own stanzas, not the
  predecessor's chat — while clean-exit protocol-violation diagnostics keep
  the worker's own final message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def review_gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}}
    )


def _make_profile(home: Path, name: str, *, with_skill: bool, disabled: bool = False) -> Path:
    """Real live profile dir (identity marker) under the isolated home."""
    profile = home / "profiles" / name
    profile.mkdir(parents=True)
    cfg = "skills:\n"
    if disabled:
        cfg += "  disabled:\n    - sdlc-review\n"
    (profile / "config.yaml").write_text(cfg, encoding="utf-8")
    if with_skill:
        skill_dir = profile / "skills" / "devops" / "sdlc-review"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: sdlc-review\n"
            "description: Review Kanban handoffs and route verified outcomes.\n"
            "---\n"
            "Review guidance.\n",
            encoding="utf-8",
        )
    return profile


def _review_task(conn, assignee: str = "reviewer") -> str:
    tid = kb.create_task(conn, title="needs review", assignee=assignee)
    implementation = kb.claim_task(conn, tid)
    assert implementation is not None
    assert kb.request_review(
        conn, tid, summary="ready", expected_run_id=implementation.current_run_id,
    )
    return tid


class _SpawnRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, task, workspace):  # back-compat 2-arg signature
        self.calls.append(task.id)
        return None


# ---------------------------------------------------------------------------
# Fix 1 — claim-time review-skill validation
# ---------------------------------------------------------------------------


def test_review_dispatch_blocks_when_profile_lacks_sdlc_review(
    kanban_home: Path, review_gate_on, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance (a): a review-lane claim on a profile whose home cannot load
    ``sdlc-review`` must NOT spawn a worker (it would die pre-boot, every
    attempt, until the breaker trips). The card is auto-blocked on the FIRST
    failure with a reason naming the profile and the skill."""
    _make_profile(kanban_home, "reviewer", with_skill=False)
    spawn = _SpawnRecorder()

    with kbc.connect() as conn:
        tid = _review_task(conn)
        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        assert tid not in [s[0] for s in result.spawned]
        assert tid in result.auto_blocked
        assert spawn.calls == []
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert "sdlc-review" in (task.last_failure_error or "")
        assert "reviewer" in (task.last_failure_error or "")
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert "gave_up" in [row["kind"] for row in events]


def test_review_dispatch_spawns_when_profile_has_sdlc_review(
    kanban_home: Path, review_gate_on
) -> None:
    """The guard must not over-block: a profile that actually ships the skill
    spawns the reviewer with the skill still force-loaded."""
    _make_profile(kanban_home, "reviewer", with_skill=True)
    captured: list[list[str]] = []

    def spawn(task, workspace):
        captured.append(list(task.skills or []))
        return None

    with kbc.connect() as conn:
        tid = _review_task(conn)
        result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert tid in [s[0] for s in result.spawned]
    assert captured == [["sdlc-review"]]


def test_review_dispatch_blocks_when_skill_disabled(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI parity: an operator-disabled skill is as unloadable as a missing one
    (``--skills`` preloading treats disabled as missing), so the guard blocks.

    No ``review_gate_on`` here: its ``load_config`` monkeypatch would hide the
    profile's ``skills.disabled`` from the very config read under test."""
    _make_profile(kanban_home, "reviewer", with_skill=True, disabled=True)
    spawn = _SpawnRecorder()

    with kbc.connect() as conn:
        tid = _review_task(conn)
        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        assert tid not in [s[0] for s in result.spawned]
        assert tid in result.auto_blocked


def test_ready_lane_unaffected_by_review_skill_guard(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is review-lane only: a ready-lane task on the same skill-less
    profile spawns exactly as before."""
    _make_profile(kanban_home, "reviewer", with_skill=False)
    spawn = _SpawnRecorder()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="reviewer")
        result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert tid in [s[0] for s in result.spawned]
    assert spawn.calls == [tid]


# ---------------------------------------------------------------------------
# Fix 2 — crash-output extractor anchoring
# ---------------------------------------------------------------------------

_PREDECESSOR_CHAT = "Fixed and verified: gates 291/0, zero new failures."

_EXIT_SUMMARY_BLOCK = (
    "\n[kanban-worker-exit] rc=0\n"
    "\n"
    "Resume this session with:\n"
    "  hermes --resume 20260919_144801_f92ffd -p developer\n"
    '  hermes -c "continue the card" -p developer\n'
    "\n"
    "Session:        20260919_144801_f92ffd\n"
    "Title:          Fix v0.2.0 blockers\n"
    "Duration:       1h 7m 19s\n"
    "Messages:       331 (2 user, 328 tool calls)\n"
)

_CRASH_STANZA = (
    "Query: work kanban task t_example\n"
    "Initializing agent...\n"
    "Error: Unknown skill(s): sdlc-review\n"
)


def _install_log(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    logs: dict[str, str] = {"t_example": text}

    def fake_read(task_id, *, tail_bytes=None, board=None):
        return logs.get(task_id)

    monkeypatch.setattr(kb, "read_worker_log", fake_read)


def test_final_output_reports_crash_stanza_not_predecessor_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance (b): predecessor exit-summary block followed by the crashed
    run's pre-boot stanza — the extractor must return the crash stanza, not the
    predecessor's chat (the incident's phantom 're-authored conclusion')."""
    _install_log(
        monkeypatch,
        _PREDECESSOR_CHAT + "\n" + "─" * 60 + "\n" + _EXIT_SUMMARY_BLOCK + _CRASH_STANZA * 6,
    )
    out = kbd._worker_final_output("t_example")
    assert "Unknown skill(s): sdlc-review" in out
    assert out.endswith("Error: Unknown skill(s): sdlc-review")
    assert "291/0" not in out          # predecessor chat must not leak
    assert "--resume" not in out       # predecessor exit block must not leak
    assert "Messages:" not in out


def test_final_output_preserves_clean_run_own_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No regression on the clean-exit protocol-violation shape: when nothing
    follows the exit summary, the worker's own final chat is still reported."""
    _install_log(monkeypatch, _PREDECESSOR_CHAT + "\n" + _EXIT_SUMMARY_BLOCK)
    out = kbd._worker_final_output("t_example")
    assert "291/0" in out
    assert "--resume" not in out
    assert "Messages:" not in out


def test_final_output_pre_chat_crash_without_query_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash that printed only an error line (no Query echo) after the
    predecessor's summary still reports its own error line."""
    _install_log(
        monkeypatch, _EXIT_SUMMARY_BLOCK + "Error: provider deadline exceeded\n",
    )
    out = kbd._worker_final_output("t_example")
    assert "provider deadline exceeded" in out
    assert "291/0" not in out
    assert "--resume" not in out


def test_final_output_missing_and_empty_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing log → None → ''; empty log → ''. Never raises."""
    assert kbd._worker_final_output("t_nolog") == ""
    _install_log(monkeypatch, "")
    assert kbd._worker_final_output("t_example") == ""
