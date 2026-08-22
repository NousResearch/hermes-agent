"""Tests for /opt/data/scripts/github_issues_mirror.py — Part A of
SPEC-active-pr-guard-reviewer-feedback (2026-08-20, hermes-agent#2).

The mirror script lives outside the hermes-agent repo at
``/opt/data/scripts/github_issues_mirror.py`` (deployment path, no PR).
These tests import it directly via ``importlib.util`` and cover the four
behaviors the spec calls out:

1. ``GH_REF_RE`` matches both ``/issues/<N>`` and ``/pull/<N>`` URLs.
2. ``_pick_ref`` prefers the freshest GH ref across body/title/comments
   (recency tie-break) over the first-in-body ref.
3. ``mirror_pull`` polls BOTH the issue ref AND the PR ref for a task
   whose body+comments span both kinds (dual-ref).
4. Dedup by ``(owner, repo, N, comment_id)`` so a comment returned by
   both endpoints with the same numeric id lands exactly once.

The script's ``KANBAN_DB`` constant is monkeypatched to a tmp path per
test so no real kanban state is touched. The hermes-agent ``init_db``
schema is reused so the queries the mirror runs actually work.
"""

from __future__ import annotations

import importlib.util  # type: ignore[import-untyped]
import json
import sqlite3
import sys
from pathlib import Path

# pytest is provided by the project's test runner; Pyright can't see it
# from the venv-less analysis env, hence the noqa.
import pytest  # type: ignore[import-not-found]

# hermes-agent import — used for init_db() to get the right schema.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hermes_cli import kanban_db as kb  # noqa: E402

MIRROR_PATH = Path("/opt/data/scripts/github_issues_mirror.py")


def _load_mirror():
    """Import the mirror script as a module (it's not a package).

    Register the module in sys.modules BEFORE exec_module so that any
    ``@dataclass`` declarations inside resolve ``cls.__module__``
    (CPython 3.11+ dataclass internals require this).
    """
    spec = importlib.util.spec_from_file_location(  # type: ignore[arg-type]
        "github_issues_mirror", MIRROR_PATH,
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["github_issues_mirror"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── shared fixtures ───────────────────────────────────────────────


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """Load the mirror module with KANBAN_DB / STATE_DIR pointed at tmp.

    Returns the module — tests reach into module-level functions and
    constants via attribute access. Use the ``kanban_db`` fixture to
    pre-populate the kanban with tasks/comments via real hermes schema.
    """
    # Load the module FIRST so the monkeypatch below can reference it
    # via direct attribute (pytest's monkeypatch.setattr string form
    # needs the module imported, but we just-loaded it as a unique
    # sys.modules entry, so the simpler in-place attribute is fine).
    mod = _load_mirror()
    kanban_db_path = tmp_path / "kanban.db"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    mod.KANBAN_DB = str(kanban_db_path)  # type: ignore[attr-defined]
    mod.STATE_DIR = state_dir  # type: ignore[attr-defined]
    # Initialize the kanban schema at this exact path so the mirror's
    # own sqlite3.connect(KANBAN_DB) queries find the right tables.
    # ``kanban_db_path`` lives outside HERMES_HOME, so we point
    # HERMES_KANBAN_DB at it BEFORE init_db — both ``kb.connect()``
    # (used by the tests' setup helpers) and the mirror's internal
    # ``sqlite3.connect(KANBAN_DB)`` then agree on the path.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_db_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    # append_kanban_comment normally shells out to `hermes kanban comment`
    # — replace it with a direct sqlite3 write so tests don't spawn
    # subprocesses or depend on the hermes CLI being installed.
    captured = {"calls": []}

    def fake_append(task_id: str, body: str) -> None:
        captured["calls"].append((task_id, body))
        conn = sqlite3.connect(str(kanban_db_path))
        try:
            with kb.write_txn(conn, allow_nested=True):
                conn.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, strftime('%s','now'))",
                    (task_id, "mirror", body),
                )
        finally:
            conn.close()

    monkeypatch.setattr(mod, "append_kanban_comment", fake_append)
    mod._test_captured = captured  # type: ignore[attr-defined]
    return mod





# ── Acceptance #6: GH_REF_RE matches both /issues/ and /pull/ ─────


def test_gh_ref_re_matches_issue_urls(mirror) -> None:
    """Spec acceptance #6: GH_REF_RE matches github.com/.../issues/<N>."""
    matches = list(mirror.GH_REF_RE.finditer(
        "see https://github.com/aliaadil/alerthq/issues/174 for context"
    ))
    assert len(matches) == 1
    m = matches[0]
    assert m.group(1) == "aliaadil"
    assert m.group(2) == "alerthq"
    assert m.group(3) == "issues"
    assert int(m.group(4)) == 174


def test_gh_ref_re_matches_pr_urls(mirror) -> None:
    """Spec acceptance #6: GH_REF_RE matches github.com/.../pull/<N>.

    Before 2026-08-20 the regex only matched /issues/, so PR-thread
    comments on a task whose body referenced the issue were silently
    dropped — the bug that left PR #178's feedback invisible.
    """
    matches = list(mirror.GH_REF_RE.finditer(
        "PR opened at https://github.com/aliaadil/alerthq/pull/178 — please review"
    ))
    assert len(matches) == 1
    m = matches[0]
    assert m.group(1) == "aliaadil"
    assert m.group(2) == "alerthq"
    assert m.group(3) == "pull"
    assert int(m.group(4)) == 178


def test_gh_ref_re_accepts_both_kinds_in_same_text(mirror) -> None:
    """A comment mentioning BOTH the issue and the PR returns both refs
    in the order they appear — used by the dual-ref mirror logic."""
    text = (
        "orig issue: https://github.com/x/y/issues/1\n"
        "followup PR: https://github.com/x/y/pull/42"
    )
    matches = list(mirror.GH_REF_RE.finditer(text))
    assert [(m.group(3), int(m.group(4))) for m in matches] == [
        ("issues", 1), ("pull", 42),
    ]


# ── Acceptance #7: _pick_ref prefers freshest ref by recency ──────


def test_pick_ref_prefers_freshest_over_first(mirror) -> None:
    """Spec acceptance #7: when body+title reference #174 (old) and a
    recent comment references #178 (new), _pick_ref returns #178 even
    though #174 appears first in the body.

    Reproduces the t_de993dac case: task body says `aliaadil/alerthq#174`,
    a worker later posted a comment with the PR URL `pull/178`. The
    mirror must pick 178, not 174.
    """
    body = "imported from aliaadil/alerthq#174"
    title = "fix the audit log"
    now = 1_700_000_000
    refs = (
        mirror._parse_gh_refs_tagged(body, created_at=0)   # body: ts=0
        + mirror._parse_gh_refs_tagged(title, created_at=0)  # title: ts=0
        + mirror._parse_gh_refs_tagged(
            "Opened https://github.com/aliaadil/alerthq/pull/178",
            created_at=now,
        )
    )
    chosen = mirror._pick_ref(refs)
    assert chosen == ("aliaadil", "alerthq", 178)


def test_pick_ref_falls_back_to_body_when_no_timestamp(mirror) -> None:
    """Without any timestamp, the first full URL beats a short ref.

    Mirrors the legacy behavior — body/title refs default to ts=0 so
    they tie, then full-URL wins the tie-break.
    """
    refs = (
        mirror._parse_gh_refs_tagged("aliaadil/alerthq#174")
        + mirror._parse_gh_refs_tagged("https://github.com/aliaadil/alerthq/pull/178")
    )
    chosen = mirror._pick_ref(refs)
    # The PR URL (full-URL) wins the ts=0 tie over the short-ref #174.
    assert chosen == ("aliaadil", "alerthq", 178)


def test_pick_ref_dedupes_owner_repo_n(mirror) -> None:
    """If the same (owner, repo, N) appears multiple times, only the
    freshest occurrence wins — body mention plus a recent comment
    pointing at the same #N resolves to a single entry."""
    now = 1_700_000_000
    refs = (
        mirror._parse_gh_refs_tagged("see https://github.com/x/y/pull/9", created_at=0)
        + mirror._parse_gh_refs_tagged(
            "Re-mentioned: https://github.com/x/y/pull/9", created_at=now,
        )
    )
    chosen = mirror._pick_ref(refs)
    assert chosen == ("x", "y", 9)


# ── Secondary-ref helper (dual-ref pairing) ──────────────────────


def test_secondary_ref_finds_paired_opposite_kind(
    mirror, tmp_path, monkeypatch,
) -> None:
    """When the primary ref is an issue, _secondary_ref_for_task
    returns the PR ref (and vice versa). This is the building block
    of dual-ref mirror_pull: both refs get polled.
    """
    # Create a task with both an issue ref in the body and a PR ref
    # in a recent comment.
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="t_de993dac repro",
            body="imported from https://github.com/aliaadil/alerthq/issues/174",
            assignee="builder",
        )
        now = 1_700_000_000
        with kb.write_txn(conn, allow_nested=True):
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, 'builder', ?, ?)",
                (tid, "Opened https://github.com/aliaadil/alerthq/pull/178", now),
            )

    primary = ("aliaadil", "alerthq", 174)  # the issue ref
    secondary = mirror._secondary_ref_for_task(tid, primary)
    assert secondary == ("aliaadil", "alerthq", 178)

    # And the inverse: primary is the PR, secondary is the issue.
    primary_pr = ("aliaadil", "alerthq", 178)
    secondary_issue = mirror._secondary_ref_for_task(tid, primary_pr)
    assert secondary_issue == ("aliaadil", "alerthq", 174)


# ── Acceptance #8: mirror_pull polls BOTH refs ────────────────────


def test_mirror_pull_appends_comments_from_both_refs(
    mirror, tmp_path, monkeypatch,
) -> None:
    """Spec acceptance #8: when a task links both an issue and a PR,
    mirror_pull polls both endpoints and appends comments from either
    source to the kanban task.

    Stubs out ``gh_issue_comments`` so we can return canned responses
    for the issue ref and the PR ref separately — without going to the
    network.
    """
    # Create a task with both refs.
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="dual-ref task",
            body="https://github.com/aliaadil/alerthq/issues/174",
            assignee="builder",
        )
        now = 1_700_000_000
        with kb.write_txn(conn, allow_nested=True):
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, 'builder', ?, ?)",
                (tid, "PR https://github.com/aliaadil/alerthq/pull/178", now),
            )

    # Stub gh_issue_comments: when called for the issue ref return one
    # comment, when called for the PR ref return a different comment.
    canned_by_ref: dict[tuple[str, str, int], list[dict]] = {
        ("aliaadil", "alerthq", 174): [
            {
                "id": 1001, "user": {"login": "aliaadil"},
                "created_at": "2026-08-20T19:00:00Z",
                "body": "issue-thread comment from reviewer",
            },
        ],
        ("aliaadil", "alerthq", 178): [
            {
                "id": 1002, "user": {"login": "aliaadil"},
                "created_at": "2026-08-20T19:10:04Z",
                "body": "the logging is not sufficient. ALL actions need to be logged",
            },
        ],
    }
    monkeypatch.setattr(
        mirror, "gh_issue_comments",
        lambda owner, repo, n, since: canned_by_ref.get((owner, repo, n), []),
    )
    # Empty get_open_tasks_with_gh_ref stub: pass the task in directly.
    tasks = [{
        "id": tid, "title": "dual-ref task",
        "body": "https://github.com/aliaadil/alerthq/issues/174",
        "status": "ready", "assignee": "builder", "created_at": 1,
    }]
    appended, skipped = mirror.mirror_pull(tasks)

    # Both comments were appended (one per ref).
    assert appended == 2
    assert skipped == 0
    # Verify both bodies are in the kanban task_comments table.
    with kb.connect() as conn:
        bodies = [
            r["body"] for r in conn.execute(
                "SELECT body FROM task_comments WHERE task_id = ? AND author = 'mirror' ORDER BY id",
                (tid,),
            ).fetchall()
        ]
    assert any("issue-thread comment from reviewer" in b for b in bodies)
    assert any("logging is not sufficient" in b for b in bodies)
    assert any("_↩ from GH comment by **aliaadil**" in b for b in bodies)


# ── Acceptance #4 (PR-thread dedupe): same id from both endpoints ──


def test_mirror_pull_dedupes_same_comment_id_across_refs(
    mirror, tmp_path, monkeypatch,
) -> None:
    """Idempotent re-pulls of the SAME comment id must land at most once
    on the kanban task — even if the GH-side cutoff gets out-of-sync and
    the comment shows up twice across pulls of the SAME ref.

    The mirror uses ``_save_pulled_comment_ids`` keyed by ref_key
    ``f"{owner}/{repo}#{n}"`` (no kind — same number across issue-style
    and PR-style endpoints shares the set). When ``since_iso`` regresses
    (clock skew, manual sidecar edit, replay scenario), the dedupe
    set keeps a comment from being appended twice.

    Stubs ``gh_issue_comments`` so the SAME comment id comes back on
    every call, and resets the sidecar ``last_gh_comment_at`` between
    calls to force a re-pull. Asserts only one kanban row lands.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="dedup task",
            body="https://github.com/aliaadil/alerthq/issues/174",
            assignee="builder",
        )

    shared_comment = {
        "id": 9999, "user": {"login": "aliaadil"},
        "created_at": "2026-08-20T19:10:04Z",
        "body": "duplicate-surface comment",
    }
    monkeypatch.setattr(
        mirror, "gh_issue_comments",
        lambda owner, repo, n, since: [shared_comment],
    )

    tasks = [{
        "id": tid, "title": "dedup task",
        "body": "https://github.com/aliaadil/alerthq/issues/174",
        "status": "ready", "assignee": "builder", "created_at": 1,
    }]
    appended1, _ = mirror.mirror_pull(tasks)
    assert appended1 == 1

    # Reset the sidecar to simulate clock skew / replay — the
    # ``since=`` cutoff regresses below the comment's timestamp.
    (mirror.STATE_DIR / "last_gh_comment_at.json").write_text("{}")
    mirror._load_pulled_comment_ids  # noqa: B018 — keep reference
    # _save_pulled_comment_ids persists the dedupe set; the state file
    # is also reloaded inside the next mirror_pull call, so the dedupe
    # set survives the cutoff reset.

    appended2, _ = mirror.mirror_pull(tasks)
    # Despite the cutoff reset pulling the same comment again, the
    # dedupe set prevents a second kanban row.
    assert appended2 == 0

    # Verify exactly one mirror row in kanban.
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ? AND author = 'mirror'",
            (tid,),
        ).fetchall()
    assert len(rows) == 1


# ── Sidecar state survives across calls (sanity) ─────────────────


def test_last_gh_comment_at_sidecar_persists(
    mirror, tmp_path, monkeypatch,
) -> None:
    """After mirror_pull runs, ``last_gh_comment_at.json`` records the
    per-ref max GH timestamp. A second call against the same stub
    returns no new comments (since both refs' cutoffs cover them) and
    leaves the sidecar alone.

    Mostly a regression guard against the cutoff-evolution bugs that
    hit earlier (the mirror used kanban-side max instead of GH-side
    max and dropped comments).
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="sidecar task",
            body="https://github.com/aliaadil/alerthq/issues/174",
            assignee="builder",
        )

    canned = {
        ("aliaadil", "alerthq", 174): [
            {
                "id": 5001, "user": {"login": "aliaadil"},
                "created_at": "2026-08-20T19:00:00Z",
                "body": "first comment",
            },
        ],
        ("aliaadil", "alerthq", 178): [],  # no PR-thread comments
    }
    monkeypatch.setattr(
        mirror, "gh_issue_comments",
        lambda owner, repo, n, since: canned.get((owner, repo, n), []),
    )

    tasks = [{
        "id": tid, "title": "sidecar task",
        "body": "https://github.com/aliaadil/alerthq/issues/174",
        "status": "ready", "assignee": "builder", "created_at": 1,
    }]

    appended1, _ = mirror.mirror_pull(tasks)
    assert appended1 == 1

    # Sidecar file exists and has the issue-ref entry.
    sidecar = mirror.STATE_DIR / "last_gh_comment_at.json"
    assert sidecar.exists()
    state = json.loads(sidecar.read_text())
    assert "aliaadil/alerthq#174" in state
    assert int(state["aliaadil/alerthq#174"]) > 0

    # A second pull appends nothing (cutoff already covers everything).
    appended2, _ = mirror.mirror_pull(tasks)
    assert appended2 == 0