"""Tests for the git-verified completion guardrail (post-mortem 2026-09-07).

Covers the two dispatch/completion levers that stop a repo-editing card from
reaching `done` with uncommitted work or from sharing a checkout:

  1. Worktree isolation (`_maybe_upgrade_repo_scratch`): a card dispatched as
     `scratch` whose body references a git repo is auto-upgraded to an isolated
     per-task worktree, so two concurrent cards never share one working tree.
  2. Mandatory skill wiring (`_default_spawn`): every default-spawned worker
     force-loads `kanban-git-verified-completion`, so it commits + pushes before
     `kanban_complete` (avoiding the kernel-level gate rejection).

The dispatch machinery moved to ``hermes_cli.kanban_db_dispatch`` in the
Sep 2026 decomposition, so this file imports it as ``kbd`` (and the connect
layer as ``kbc``) rather than reaching them through ``kanban_db``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _make_git_repo(base: Path, name: str = "repo") -> str:
    repo = base / name
    repo.mkdir(parents=True, exist_ok=True)

    def git(*args, **kw):
        return subprocess.run(
            ["git"] + list(args), cwd=repo, capture_output=True, text=True, **kw
        )

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    return str(repo)


def _mk(conn, *, title="card", body=None, workspace_kind="scratch", **kw):
    return kb.create_task(
        conn, title=title, body=body,
        workspace_kind=workspace_kind, assignee="conan", **kw,
    )


def test_repo_referencing_scratch_card_upgraded_to_worktree(kanban_home, tmp_path):
    """Acceptance: a card that edits a repo must never be dispatched as a shared
    `scratch` checkout. `_maybe_upgrade_repo_scratch` auto-upgrades it to an
    isolated per-task worktree."""
    repo = _make_git_repo(tmp_path)
    conn = kbc.connect()
    try:
        tid = _mk(conn, body=f"implement the fix in {repo}/src/main.py")
        task = kb.get_task(conn, tid)
        assert task.workspace_kind == "scratch"

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert upgraded.workspace_kind == "worktree"
        # The worktree must anchor on the actual repo the card edits.
        assert Path(upgraded.workspace_path).resolve() == Path(repo).resolve()
        assert upgraded.branch_name == f"wt/{tid}"

        # The upgrade is observable in the event log.
        evs = kb.list_events(conn, tid)
        kinds = [e.kind for e in evs]
        assert "workspace_upgraded_to_worktree" in kinds
    finally:
        conn.close()


def test_non_repo_scratch_card_left_alone_with_warning(kanban_home, tmp_path):
    """A scratch card that references no git repo stays scratch (we cannot
    isolate on a repo that doesn't exist), but the hazard is surfaced as an
    event so an operator can pin it to a project if it edits a repo."""
    conn = kbc.connect()
    try:
        tid = _mk(conn, body="update the SSOT rules per the incident")
        task = kb.get_task(conn, tid)

        result = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert result.workspace_kind == "scratch"
        evs = kb.list_events(conn, tid)
        kinds = [e.kind for e in evs]
        assert "scratch_repo_reference_warning" in kinds
    finally:
        conn.close()


def test_default_spawn_force_loads_git_verify_skill(kanban_home, monkeypatch):
    """Acceptance: the mandatory git-verified-completion skill is wired into the
    default dispatch path — even a worker with no task.skills gets it."""
    repo = _make_git_repo(kanban_home.parent)
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="spawn", assignee="conan",
            body=f"do it in {repo}", workspace_kind="scratch",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    # Task was upgraded by dispatch policy? For the spawn test we just need a
    # task object; the skill wiring is independent of workspace_kind.
    assert task.id == tid

    captured = {}

    class _FakeProc:
        pid = 4242

    real_popen = subprocess.Popen

    def fake_popen(args, **kw):
        captured["args"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # Resolve the assignee's profile without touching a real one: these are
    # imported inside `_default_spawn` from hermes_cli.profiles.
    monkeypatch.setattr(
        "hermes_cli.profiles.normalize_profile_name", lambda name: name
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env", lambda name: str(kanban_home)
    )

    try:
        pid = kbd._default_spawn(task, workspace=repo, board=None)
    finally:
        subprocess.Popen = real_popen

    assert pid == 4242
    args = captured["args"]
    assert "--skills" in args
    i = args.index("--skills")
    assert i + 1 < len(args)
    assert args[i + 1] == kbd.DEFAULT_GIT_VERIFY_SKILL


# ---------------------------------------------------------------------------
# Anchor source: a board-default anchor is a guess, and must say so
# (card t_31f36f49 — a card targeting gamified-plant was anchored, via the
# board default_workdir fallback, to the sibling repo sams-app).
# ---------------------------------------------------------------------------


def _payloads(conn, tid, kind):
    return [e.payload for e in kb.list_events(conn, tid) if e.kind == kind]


def test_prose_anchor_records_anchor_source(kanban_home, tmp_path):
    """Invariant: the upgrade event names the lever that picked the repo."""
    repo = _make_git_repo(tmp_path)
    conn = kbc.connect()
    try:
        tid = _mk(conn, body=f"implement the fix in {repo}/src/main.py")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert len(payloads) == 1
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert payloads[0]["repo"] == str(Path(repo).resolve())
        # The prose itself named the repo — nothing to flag.
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_board_default_anchor_without_prose_confirmation_is_flagged(kanban_home, tmp_path):
    """The regression card t_31f36f49 measured: prose names no repo path and no
    repo name, so the board's default_workdir decides — the worktree lands on a
    sibling repo the card never mentions. That guess must be visible on the
    board, not just in the workspace path."""
    repo = _make_git_repo(tmp_path)
    kb.create_board("sibling-probe", default_workdir=repo)
    conn = kbc.connect(board="sibling-probe")
    try:
        tid = _mk(conn, body="Apply the same .worktrees/ ignore rule to gamified-plant")
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board="sibling-probe")

        assert upgraded.workspace_kind == "worktree"
        assert Path(upgraded.workspace_path).resolve() == Path(repo).resolve()
        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_BOARD_DEFAULT

        flagged = _payloads(conn, tid, "workspace_anchor_fallback")
        assert len(flagged) == 1
        assert flagged[0]["repo"] == str(Path(repo).resolve())
        assert flagged[0]["anchor_source"] == kbd.ANCHOR_SOURCE_BOARD_DEFAULT
        assert flagged[0]["board"] == "sibling-probe"
        assert flagged[0]["card_names_repo_in_prose"] is False
        assert flagged[0]["sibling_repo_named_in_prose"] is None
    finally:
        conn.close()


def test_sibling_repo_named_in_prose_is_flagged(kanban_home, tmp_path):
    """The measured shape of t_ef94e814: the card is about gamified-plant and
    mentions the board repo in passing (``sams-app.sh``). Anchoring on sams-app
    must still be flagged — a sibling repo is what the card is actually about,
    and the passing mention is not confirmation."""
    ws = tmp_path / "workspace"
    sams = _make_git_repo(ws, "sams-app")
    _make_git_repo(ws, "gamified-plant")
    kb.create_board("sams-probe", default_workdir=sams)
    conn = kbc.connect(board="sams-probe")
    try:
        tid = _mk(
            conn,
            body="Port the ignore rules to gamified-plant, same sweep as sams-app.sh",
        )
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="sams-probe")

        flagged = _payloads(conn, tid, "workspace_anchor_fallback")
        assert len(flagged) == 1
        assert flagged[0]["repo"] == str(Path(sams).resolve())
        assert flagged[0]["sibling_repo_named_in_prose"] == "gamified-plant"
        # "sams-app.sh" is not the repo name — the prose never names the anchor.
        assert flagged[0]["card_names_repo_in_prose"] is False
    finally:
        conn.close()


def test_prose_path_gone_from_disk_is_flagged(kanban_home, tmp_path):
    """A card naming a *deleted* sibling worktree inside a repo resolves through
    the enclosing repo, so the worktree lands on a repo the card never meant.
    Measured on t_31f36f49 itself, whose prose pointed at
    ``.../sams-app/.worktrees/t_ef94e814`` after that worktree was gone."""
    repo = _make_git_repo(tmp_path)
    (Path(repo) / ".worktrees").mkdir(exist_ok=True)
    gone = Path(repo) / ".worktrees" / "t_ef94e814"
    assert not gone.exists()
    conn = kbc.connect(board="default")
    try:
        tid = _mk(conn, body=f"Continue the audit in {gone}/docs (checkout was there)")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="default")

        # The anchor is still the enclosing repo — but it says why it is a guess.
        upgraded = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert upgraded[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert upgraded[0]["repo"] == str(Path(repo).resolve())
        unverified = _payloads(conn, tid, "workspace_anchor_unverified")
        assert len(unverified) == 1
        assert unverified[0]["prose_worktree"] == str(gone)
        assert unverified[0]["repo"] == str(Path(repo).resolve())
    finally:
        conn.close()


def test_live_prose_path_is_not_flagged(kanban_home, tmp_path):
    """A file the card is about to create is a normal prose path, not a hazard —
    flagging it would be noise an operator learns to ignore."""
    repo = _make_git_repo(tmp_path)
    (Path(repo) / "docs").mkdir(exist_ok=True)
    conn = kbc.connect(board="default")
    try:
        tid = _mk(conn, body=f"Edit {repo}/docs/new-file.md in place")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="default")

        assert _payloads(conn, tid, "workspace_anchor_unverified") == []
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_existing_worktree_path_is_not_flagged(kanban_home, tmp_path):
    """A card that names a worktree which *is* on disk resolves cleanly."""
    repo = _make_git_repo(tmp_path)
    live = Path(repo) / ".worktrees" / "t_alive"
    live.mkdir(parents=True)
    conn = kbc.connect(board="default")
    try:
        tid = _mk(conn, body=f"The checkout is at {live} — keep going")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="default")

        assert _payloads(conn, tid, "workspace_anchor_unverified") == []
    finally:
        conn.close()


def test_board_default_anchor_confirmed_by_prose_is_not_flagged(kanban_home, tmp_path):
    """A board-default anchor is fine when the card's own prose names that repo;
    flagging it would be noise an operator learns to ignore."""
    repo = _make_git_repo(tmp_path)
    kb.create_board("self-probe", default_workdir=repo)
    conn = kbc.connect(board="self-probe")
    try:
        tid = _mk(conn, body="Update the repo README and .gitignore")
        task = kb.get_task(conn, tid)

        kbd._maybe_upgrade_repo_scratch(conn, task, board="self-probe")

        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_BOARD_DEFAULT
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_prose_names_repo_is_word_bounded(tmp_path):
    """Basename matching must not fire on a longer or dotted name."""
    repo = Path(tmp_path) / "sams-app"
    assert kbd._prose_names_repo("fix sams-app now", None, repo)
    assert kbd._prose_names_repo(None, "SAMS-APP deploy notes", repo)
    assert not kbd._prose_names_repo("fix sams-app-old now", None, repo)
    assert not kbd._prose_names_repo("see x.sams-app for detail", None, repo)


# ---------------------------------------------------------------------------
# Two path shapes the prose anchor used to miss (card t_7ad7badc):
#   A. the token names an *existing file* — the commonest card shape
#      ("edit <repo>/.gitignore"); `git -C <file>` fails, so the card fell
#      through to the board default and anchored a worktree on a sibling repo.
#   B. the token is markdown inline code — the closing backtick was captured
#      into the path, so the repo path in the body resolved to nothing.
# ---------------------------------------------------------------------------


def test_prose_path_naming_an_existing_file_anchors_its_repo(kanban_home, tmp_path):
    """Invariant: a path token naming a file that exists resolves through the
    file's directory to the repo that holds it, and beats the board default."""
    ws = tmp_path / "workspace"
    plant = _make_git_repo(ws, "gamified-plant")
    sams = _make_git_repo(ws, "sams-app")
    (Path(plant) / ".gitignore").write_text("node_modules\n")
    kb.create_board("file-probe", default_workdir=sams)
    conn = kbc.connect(board="file-probe")
    try:
        tid = _mk(
            conn,
            body=f"Apply the same .worktrees/ ignore rule to {plant}/.gitignore",
        )
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board="file-probe")

        assert Path(upgraded.workspace_path).resolve() == Path(plant).resolve()
        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert payloads[0]["repo"] == str(Path(plant).resolve())
        # The prose itself named the repo — the board default never decided.
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()


def test_prose_path_naming_a_file_that_does_not_exist_still_anchors(kanban_home, tmp_path):
    """Regression: a path the card is about to *create* resolves through its
    nearest existing ancestor to the same repo, as it always did."""
    repo = _make_git_repo(tmp_path)
    anchor = kbd._card_repo_anchor("t", f"create {repo}/src/main.py", board=None)
    assert anchor is not None
    assert anchor.source == kbd.ANCHOR_SOURCE_PROSE
    assert Path(anchor.repo).resolve() == Path(repo).resolve()


def test_backtick_wrapped_prose_path_anchors_its_repo(kanban_home, tmp_path):
    """Invariant: markdown inline code must not capture its closing backtick.
    t_c7131f54 was warned as "card body references no resolvable git repo"
    while the repo path sat in the body wrapped in backticks."""
    repo = _make_git_repo(tmp_path)
    anchor = kbd._card_repo_anchor(
        "t", f"- Repo = `{repo}`, has origin but no worktrees", board=None,
    )
    assert anchor is not None
    assert anchor.source == kbd.ANCHOR_SOURCE_PROSE
    assert Path(anchor.repo).resolve() == Path(repo).resolve()
    assert anchor.prose_path == str(repo)


def test_backtick_wrapped_path_beats_the_board_default(kanban_home, tmp_path):
    """End-to-end: a backtick-wrapped path naming a sibling repo wins over the
    board default, and is not reported as an unconfirmed guess."""
    ws = tmp_path / "workspace"
    sams = _make_git_repo(ws, "sams-app")
    littlepanda = _make_git_repo(ws, "littlepanda")
    kb.create_board("sams-probe", default_workdir=sams)
    conn = kbc.connect(board="sams-probe")
    try:
        tid = _mk(conn, body=f"- Repo = `{littlepanda}`, has origin but no worktrees")
        task = kb.get_task(conn, tid)

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board="sams-probe")

        assert Path(upgraded.workspace_path).resolve() == Path(littlepanda).resolve()
        payloads = _payloads(conn, tid, "workspace_upgraded_to_worktree")
        assert payloads[0]["anchor_source"] == kbd.ANCHOR_SOURCE_PROSE
        assert payloads[0]["repo"] == str(Path(littlepanda).resolve())
        assert _payloads(conn, tid, "workspace_anchor_fallback") == []
    finally:
        conn.close()

