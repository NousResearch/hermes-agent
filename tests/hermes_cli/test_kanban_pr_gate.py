"""PR-gate re-evaluator: a card blocked on "merge PR #N" resumes when #N merges.

Context (2026-09-21 board sweep): six ``needs_input`` cards sat blocked for up
to 12 hours on gates whose stated blocker was "merge PR #N then unblock me" —
every referenced PR had already merged. Nothing in the dispatcher tick
re-evaluated a block whose premise is an EXTERNAL object, so the class was only
ever caught by a human board sweep.

These tests pin:

* the reference parser truth table (bare ``#N`` with/without repo context,
  ``owner/repo#N``, full URL, multiple refs, zero refs, ambiguous context),
* the state machine (all-merged -> unblock; any-open -> hold; closed-unmerged ->
  comment only; lookup error -> no-op),
* the per-tick lookup cap and the cross-tick result cache,
* scope fencing (only ``needs_input``/``capability``/``dependency`` blocks; never
  a block whose reason names no PR),
* and one end-to-end pass against a real sqlite board with a stubbed ``gh``.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect
from hermes_cli import kanban_db_dispatch
from hermes_cli import kanban_pr_gate as prg


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB.

    ``HERMES_KANBAN_SANDBOX=1`` is the POSITIVE opt-in :func:`prg._db_is_sandboxed`
    requires: containment alone is not proof of isolation (the fleet's own
    ``HERMES_HOME=~/.hermes`` contains the live board), so a harness running a
    stubbed oracle has to declare itself.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_SANDBOX", "1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _clear_pr_cache() -> None:
    prg.clear_cache()
    yield
    prg.clear_cache()


def _merged(sha: str = "abcdef1234567890", at: str = "2026-09-20T13:05:00Z") -> dict:
    return {"state": "MERGED", "mergedAt": at, "mergeCommitSha": sha}


def _open_pr() -> dict:
    return {"state": "OPEN", "mergedAt": None, "mergeCommitSha": None}


def _closed() -> dict:
    return {"state": "CLOSED", "mergedAt": None, "mergeCommitSha": None}


def _stub(mapping, *, calls=None):
    """Build a ``query_fn`` over ``{(repo, number): payload-or-None}``."""

    def query_fn(repo: str, number: int):
        if calls is not None:
            calls.append((repo, number))
        return mapping.get((repo, number))

    return query_fn


# ---------------------------------------------------------------------------
# Parser truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,default_repo,expected",
    [
        # Bare #N with repo context resolves against it.
        ("merge PR #787 then unblock me", "ANG-Ventures/hermes-agent",
         [("ANG-Ventures/hermes-agent", 787)]),
        # Bare #N with NO repo context is unresolvable -> dropped.
        ("merge PR #787 then unblock me", None, []),
        # owner/repo#N carries its own context and wins over the default.
        ("blocked on NousResearch/hermes-agent#4211", "ANG-Ventures/hermes-agent",
         [("NousResearch/hermes-agent", 4211)]),
        # Full URL.
        ("waiting on https://github.com/ANG-Ventures/hermes-agent/pull/790", None,
         [("ANG-Ventures/hermes-agent", 790)]),
        # URL with trailing path/punctuation (JSON handbacks abut a quote+comma).
        ('{"pr": "https://github.com/o/r/pull/12/files",}', None, [("o/r", 12)]),
        # bare pull/N against context.
        ("gate: pull/55 must land", "o/r", [("o/r", 55)]),
        # Multiple refs, de-duplicated and ordered.
        ("needs #5 and o/r#9 and #5 again", "o/r",
         [("o/r", 5), ("o/r", 9)]),
        # Zero PR references.
        ("blocked: need Ace to decide the retention window", "o/r", []),
        # A task id is not a PR reference.
        ("blocked on t_3c0420ad finishing", "o/r", []),
        # Case-insensitive repo, normalized.
        ("ANG-Ventures/Hermes-Agent#1", None, [("ANG-Ventures/Hermes-Agent", 1)]),
        # #0 is not a valid PR number.
        ("merge PR #0", "o/r", []),
    ],
)
def test_parse_pr_refs_truth_table(text, default_repo, expected) -> None:
    refs = prg.parse_pr_refs(text, default_repo=default_repo)
    assert [(r.repo, r.number) for r in refs] == expected


def test_parse_pr_refs_ignores_non_string() -> None:
    assert prg.parse_pr_refs(None, default_repo="o/r") == []
    assert prg.parse_pr_refs("", default_repo="o/r") == []


def test_repo_context_prefers_single_workspace_remote(tmp_path: Path) -> None:
    repo = tmp_path / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:o/r.git"],
        cwd=repo, check=True,
    )
    assert prg.repo_context(workspace_path=str(repo), body=None) == "o/r"


def test_repo_context_falls_back_to_body_mention(tmp_path: Path) -> None:
    assert prg.repo_context(
        workspace_path=str(tmp_path / "missing"),
        body="land it on ANG-Ventures/hermes-agent first",
    ) == "ANG-Ventures/hermes-agent"


@pytest.mark.parametrize(
    "body",
    [
        # Reviewer round-3 case 1: two-segment source path before the repo.
        (
            "BUILD: edit hermes_cli/kanban.py. "
            "Fork-first ANG-Ventures/hermes-agent. Merge PR #808."
        ),
        # Reviewer round-3 case 2: nested test path before the repo.
        (
            "Tests in tests/hermes_cli/test_x.py; "
            "repo ANG-Ventures/hermes-agent; merge #808."
        ),
        # THIS card's own body: a GLOB path, whose extension a suffix
        # denylist never sees because ``*`` truncates the match.
        (
            "BUILD: in the kanban dispatcher tick (hermes-agent, "
            "hermes_cli/kanban*.py - find the tick that runs "
            "recompute_ready). Fork-first ANG-Ventures/hermes-agent."
        ),
        # Underscored module dir with no extension at all.
        (
            "patch hermes_cli/kanban_db then land on "
            "ANG-Ventures/hermes-agent; merge #808."
        ),
    ],
)
def test_repo_context_ignores_source_paths_before_body_repo(
    tmp_path: Path, body: str
) -> None:
    """A GitHub *owner* never contains ``_`` or ``.``; a module path does.

    The discriminator must be a positive property of the slug, not a
    denylist of file extensions: ``hermes_cli/kanban*.py`` truncates to
    ``hermes_cli/kanban`` (no suffix to deny) and ``hermes_cli/kanban_db``
    never had one.
    """
    assert prg.repo_context(
        workspace_path=str(tmp_path / "missing"), body=body
    ) == "ANG-Ventures/hermes-agent"


def test_repo_context_is_none_when_body_has_two_plausible_slugs(
    tmp_path: Path,
) -> None:
    """``src/utils`` is a *legal* repo slug, so it cannot be ruled out.

    Two uncorroborated candidates is the same fail-safe as two disagreeing
    remotes: take no action rather than query a coin-flip repo.
    """
    assert prg.repo_context(
        workspace_path=str(tmp_path / "missing"),
        body="see src/utils then repo ANG-Ventures/hermes-agent merge #5",
    ) is None


def test_repo_context_prefers_the_corroborated_slug(tmp_path: Path) -> None:
    """A slug also seen in a PR URL / qualified ref wins over a bare one."""
    assert prg.repo_context(
        workspace_path=str(tmp_path / "missing"),
        body=(
            "see src/utils then merge "
            "https://github.com/ANG-Ventures/hermes-agent/pull/808 "
            "and also #809"
        ),
    ) == "ANG-Ventures/hermes-agent"
    assert prg.repo_context(
        workspace_path=str(tmp_path / "missing"),
        body="see src/utils then merge ANG-Ventures/hermes-agent#808 and #809",
    ) == "ANG-Ventures/hermes-agent"


def test_repo_context_is_none_when_remotes_disagree_and_body_is_silent(
    tmp_path: Path,
) -> None:
    """Fail-safe: never GUESS which of two remotes a bare ``#N`` means.

    The hermes-agent checkout has ``origin`` = NousResearch and ``fork`` =
    ANG-Ventures, so a bare ``#787`` is genuinely ambiguous. Returning None
    makes the re-evaluator take no action rather than unblock on a coin flip.
    """
    repo = tmp_path / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:a/one.git"],
        cwd=repo, check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "fork", "git@github.com:b/two.git"],
        cwd=repo, check=True,
    )
    assert prg.repo_context(workspace_path=str(repo), body=None) is None
    # ... but an explicit body mention disambiguates it.
    assert prg.repo_context(
        workspace_path=str(repo), body="PR is on b/two"
    ) == "b/two"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def _blocked_card(conn, *, reason: str, kind: str = "needs_input", body=None):
    tid = kb.create_task(conn, title="gated card", body=body)
    kb.claim_task(conn, tid)
    assert kb.block_task(
        conn, tid, reason=reason, kind=kind,
        expected_run_id=kb.get_task(conn, tid).current_run_id,
    )
    return tid


def test_all_merged_unblocks_and_records_one_event(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(
            conn,
            reason="merge o/r#7 and o/r#8 then unblock me",
        )
        assert kb.get_task(conn, tid).status == "blocked"

        outcomes = prg.reevaluate_pr_gates(
            conn,
            query_fn=_stub({("o/r", 7): _merged("1111111122222222"),
                            ("o/r", 8): _merged("3333333344444444")}),
        )

    assert [o.action for o in outcomes] == ["unblocked"]
    with kanban_db_connect.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"
        events = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            )
        ]
        assert events.count("gate_auto_resolved") == 1
        payload = json.loads(conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'gate_auto_resolved'",
            (tid,),
        ).fetchone()["payload"])
        assert payload["prs"] == ["o/r#7", "o/r#8"]
        comments = [
            r["body"] for r in conn.execute(
                "SELECT body FROM task_comments WHERE task_id = ?", (tid,)
            )
        ]
        assert len(comments) == 1
        assert "gate satisfied" in comments[0]
        assert "o/r#7 merged 11111111" in comments[0]
        # The unblock reason and the comment are the same sentence.
        assert comments[0] == outcomes[0].detail


def test_one_open_pr_holds_the_card(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 and o/r#8 then unblock me")
        outcomes = prg.reevaluate_pr_gates(
            conn,
            query_fn=_stub({("o/r", 7): _merged(), ("o/r", 8): _open_pr()}),
        )
        assert [o.action for o in outcomes] == ["held"]
        assert kb.get_task(conn, tid).status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) c FROM task_comments WHERE task_id = ?", (tid,)
        ).fetchone()["c"] == 0


def test_closed_unmerged_comments_but_never_unblocks(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#9 then unblock me")
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({("o/r", 9): _closed()}),
        )
        assert [o.action for o in outcomes] == ["closed_unmerged"]
        assert kb.get_task(conn, tid).status == "blocked"
        body = conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ?", (tid,)
        ).fetchone()["body"]
        assert "closed without merge" in body
        assert "needs a human" in body
        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            )
        ]
        assert "gate_auto_resolved" not in kinds


def test_closed_unmerged_comments_only_once_across_ticks(kanban_home: Path) -> None:
    """The advisory comment must not be re-posted every 60-second tick."""
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#9 then unblock me")
        for _ in range(3):
            prg.reevaluate_pr_gates(
                conn, query_fn=_stub({("o/r", 9): _closed()}),
            )
        assert conn.execute(
            "SELECT COUNT(*) c FROM task_comments WHERE task_id = ?", (tid,)
        ).fetchone()["c"] == 1


def test_closed_unmerged_advisory_is_keyed_to_the_current_pr_set(
    kanban_home: Path,
) -> None:
    """Re-blocking on a different dead PR gets its own human advisory."""
    with kb.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#9 then unblock me")
        prg.reevaluate_pr_gates(conn, query_fn=_stub({("o/r", 9): _closed()}))

        # Simulate the operator re-pointing the existing blocked gate. The
        # re-evaluator keys off the latest blocked event, not stale task prose.
        with kb.write_txn(conn, allow_nested=True):
            kb._append_event(
                conn,
                tid,
                "blocked",
                {"reason": "replacement gate is o/r#10", "kind": "needs_input"},
            )
        prg.reevaluate_pr_gates(conn, query_fn=_stub({("o/r", 10): _closed()}))

        comments = [
            row["body"]
            for row in conn.execute(
                "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id", (tid,)
            )
        ]
    assert len(comments) == 2
    assert "o/r#9" in comments[0]
    assert "o/r#10" in comments[1]


def test_lookup_failure_is_a_noop(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        outcomes = prg.reevaluate_pr_gates(conn, query_fn=_stub({}))
        assert [o.action for o in outcomes] == ["lookup_failed"]
        assert kb.get_task(conn, tid).status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) c FROM task_comments WHERE task_id = ?", (tid,)
        ).fetchone()["c"] == 0


def test_lookup_failure_is_deduplicated_within_one_tick(
    kanban_home: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A shared failed lookup costs one call and one warning opportunity per tick."""
    calls: list = []
    with kb.connect() as conn:
        for _ in range(3):
            _blocked_card(conn, reason="merge o/r#7 then unblock me")
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({}, calls=calls),
        )
    assert calls == [("o/r", 7)]
    assert [o.action for o in outcomes] == ["lookup_failed"] * 3
    warnings = [
        record for record in caplog.records
        if "could not resolve PR state" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_lookup_failure_is_not_cached(kanban_home: Path) -> None:
    """A transient ``gh`` failure must not poison the card for 5 minutes."""
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        prg.reevaluate_pr_gates(conn, query_fn=_stub({}))
        prg.reevaluate_pr_gates(conn, query_fn=_stub({("o/r", 7): _merged()}))
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# Scope fencing
# ---------------------------------------------------------------------------


def test_block_with_no_pr_reference_is_never_touched(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="need Ace to pick the retention window")
        calls: list = []
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({}, calls=calls),
        )
        assert outcomes == []
        assert calls == []          # zero GitHub lookups burned
        assert kb.get_task(conn, tid).status == "blocked"


def test_transient_block_kind_is_out_of_scope(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(
            conn, reason="merge o/r#7 then unblock me", kind="transient",
        )
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({("o/r", 7): _merged()}),
        )
        assert outcomes == []
        assert kb.get_task(conn, tid).status == "blocked"


def test_capability_block_kind_is_in_scope(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(
            conn, reason="merge o/r#7 then unblock me", kind="capability",
        )
        prg.reevaluate_pr_gates(conn, query_fn=_stub({("o/r", 7): _merged()}))
        assert kb.get_task(conn, tid).status == "ready"


def test_already_unblocked_card_is_not_reconsidered(kanban_home: Path) -> None:
    """The gate reads the LAST blocked event, and only while still blocked."""
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        assert kb.unblock_task(conn, tid)
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({("o/r", 7): _merged()}),
        )
        assert outcomes == []
        events = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            )
        ]
        assert "gate_auto_resolved" not in events


def test_bare_number_without_repo_context_is_left_alone(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge PR #787 then unblock me")
        calls: list = []
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({}, calls=calls),
        )
        assert outcomes == []
        assert calls == []
        assert kb.get_task(conn, tid).status == "blocked"


def test_bare_number_resolves_against_the_card_body(kanban_home: Path) -> None:
    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(
            conn,
            reason="merge PR #787 then unblock me",
            body="work lands on ANG-Ventures/hermes-agent",
        )
        prg.reevaluate_pr_gates(
            conn,
            query_fn=_stub({("ANG-Ventures/hermes-agent", 787): _merged()}),
        )
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# Cap + cache
# ---------------------------------------------------------------------------


def test_lookup_cap_is_enforced_per_tick(kanban_home: Path) -> None:
    calls: list = []
    with kanban_db_connect.connect() as conn:
        for n in range(5):
            _blocked_card(conn, reason=f"merge o/r#{100 + n} then unblock me")
        outcomes = prg.reevaluate_pr_gates(
            conn,
            query_fn=_stub({}, calls=calls),
            max_lookups=2,
        )
    assert len(calls) == 2
    # Cards beyond the budget are silently deferred to the next tick, not
    # reported as a failure.
    assert sum(1 for o in outcomes if o.action == "budget_exhausted") == 3


def test_unique_pr_is_queried_once_per_tick(kanban_home: Path) -> None:
    calls: list = []
    with kanban_db_connect.connect() as conn:
        for _ in range(3):
            _blocked_card(conn, reason="merge o/r#7 then unblock me")
        prg.reevaluate_pr_gates(
            conn,
            query_fn=_stub({("o/r", 7): _merged()}, calls=calls),
        )
        assert calls == [("o/r", 7)]
        assert all(
            r["status"] == "ready"
            for r in conn.execute("SELECT status FROM tasks")
        )


def test_open_state_is_cached_for_the_ttl_then_requeried(kanban_home: Path) -> None:
    calls: list = []
    clock = {"now": 1_000_000.0}
    with kanban_db_connect.connect() as conn:
        _blocked_card(conn, reason="merge o/r#7 then unblock me")
        qf = _stub({("o/r", 7): _open_pr()}, calls=calls)
        prg.reevaluate_pr_gates(conn, query_fn=qf, now=clock["now"])
        prg.reevaluate_pr_gates(conn, query_fn=qf, now=clock["now"] + 60)
        assert len(calls) == 1, "second tick inside the TTL must reuse the cache"
        prg.reevaluate_pr_gates(
            conn, query_fn=qf, now=clock["now"] + prg.CACHE_TTL_SECONDS + 1,
        )
        assert len(calls) == 2, "TTL expiry must re-query"


def test_closed_state_is_cached_for_ttl_then_requeried_after_reopen(
    kanban_home: Path,
) -> None:
    """CLOSED is reversible on GitHub; only MERGED may be cached forever."""
    calls: list = []
    responses = iter([_closed(), _merged()])

    def query(repo: str, number: int):
        calls.append((repo, number))
        return next(responses)

    with kb.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        prg.reevaluate_pr_gates(conn, query_fn=query, now=1_000_000.0)
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        prg.reevaluate_pr_gates(conn, query_fn=query, now=1_000_060.0)
        assert len(calls) == 1
        prg.reevaluate_pr_gates(
            conn,
            query_fn=query,
            now=1_000_000.0 + prg.CACHE_TTL_SECONDS + 1,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
    assert calls == [("o/r", 7), ("o/r", 7)]


def test_merged_state_is_cached_permanently(kanban_home: Path) -> None:
    calls: list = []
    with kanban_db_connect.connect() as conn:
        _blocked_card(conn, reason="merge o/r#7 then unblock me")
        qf = _stub({("o/r", 7): _merged()}, calls=calls)
        prg.reevaluate_pr_gates(conn, query_fn=qf, now=1_000_000.0)
        _blocked_card(conn, reason="also merge o/r#7 then unblock me")
        prg.reevaluate_pr_gates(conn, query_fn=qf, now=1_000_000.0 + 86_400)
    assert len(calls) == 1, "MERGED is irreversible — never re-query it"


# ---------------------------------------------------------------------------
# Integration: the real dispatcher tick, with gh stubbed at the subprocess seam
# ---------------------------------------------------------------------------


def test_dispatch_tick_resolves_a_satisfied_gate(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: one ``dispatch_once`` tick unblocks the card and reports it."""
    calls: list = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] == ["gh", "api"]
        assert argv[2] == "repos/o/r/pulls/7"
        return subprocess.CompletedProcess(
            argv, 0,
            stdout=json.dumps({
                "state": "closed",
                "merged_at": "2026-09-20T13:05:00Z",
                "merge_commit_sha": "deadbeefcafebabe",
            }),
            stderr="",
        )

    monkeypatch.setattr(prg.subprocess, "run", fake_run)

    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(
            conn, reason="merge https://github.com/o/r/pull/7 then unblock me",
        )

    with kanban_db_connect.connect() as conn:
        result = kanban_db_dispatch.dispatch_once(conn, spawn_fn=lambda *a, **k: None)

    assert tid in result.gate_auto_resolved
    assert len(calls) == 1
    with kanban_db_connect.connect() as conn:
        task = kb.get_task(conn, tid)
        # Unblocked, and now spawnable again (it was ready-phase when blocked).
        assert task.status in {"ready", "running"}
        assert conn.execute(
            "SELECT COUNT(*) c FROM task_events "
            "WHERE task_id = ? AND kind = 'gate_auto_resolved'",
            (tid,),
        ).fetchone()["c"] == 1


def test_dispatch_tick_queries_github_before_taking_dispatch_lock(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow GitHub I/O must never extend the board's single-writer lock hold."""
    lock_held = False
    calls: list = []

    @contextlib.contextmanager
    def tracked_lock(_db_path):
        nonlocal lock_held
        assert not lock_held
        lock_held = True
        try:
            yield True
        finally:
            lock_held = False

    def query(repo: str, number: int):
        assert not lock_held, "GitHub lookup ran under _dispatch_tick_lock"
        calls.append((repo, number))
        return _merged()

    monkeypatch.setattr(kanban_db_connect, "_dispatch_tick_lock", tracked_lock)
    monkeypatch.setattr(prg, "query_pr", query)

    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        result = kanban_db_dispatch.dispatch_once(
            conn, spawn_fn=lambda *a, **k: None,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status in {"ready", "running"}

    assert calls == [("o/r", 7)]
    assert result.gate_auto_resolved == [tid]


def test_dispatch_tick_leaves_a_non_pr_block_alone(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("no gh lookup should happen for a non-PR block")

    monkeypatch.setattr(prg.subprocess, "run", explode)

    with kanban_db_connect.connect() as conn:
        tid = _blocked_card(conn, reason="need Ace to choose the cap")

    with kanban_db_connect.connect() as conn:
        result = kanban_db_dispatch.dispatch_once(conn, spawn_fn=lambda *a, **k: None)

    assert result.gate_auto_resolved == []
    with kanban_db_connect.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_raising_lookup_warns_once_through_prefetch_and_reevaluate(
    kanban_home: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising ``gh`` seam is one failed lookup, so it is exactly one WARN.

    The prefetch pass and the locked re-evaluation pass are two halves of one
    tick. Before this test each half logged its own warning for the same failed
    unique PR, so a single unreachable PR paged the log twice per tick — the
    card's contract is "lookup failure = no action + one WARN".
    """
    calls: list = []

    def boom(repo: str, number: int):
        calls.append((repo, number))
        raise RuntimeError("gh exploded")

    with kb.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        caplog.clear()
        with caplog.at_level("WARNING"):
            prefetched = prg.prefetch_pr_gate_states(conn, query_fn=boom)
            outcomes = prg.reevaluate_pr_gates(
                conn, query_fn=boom, prefetched=prefetched,
            )

    assert calls == [("o/r", 7)]
    assert [o.action for o in outcomes] == ["lookup_failed"]
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"
    warnings = [
        record for record in caplog.records
        if record.levelname == "WARNING"
        and "kanban PR-gate" in record.getMessage()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "o/r#7" in warnings[0].getMessage()


def test_raising_lookup_without_prefetch_is_a_noop_with_one_warning(
    kanban_home: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Direct (no-prefetch) callers get the same single-warning no-op."""

    def boom(repo: str, number: int):
        raise RuntimeError("gh exploded")

    with kb.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        caplog.clear()
        with caplog.at_level("WARNING"):
            outcomes = prg.reevaluate_pr_gates(conn, query_fn=boom)
        assert [o.action for o in outcomes] == ["lookup_failed"]
        assert kb.get_task(conn, tid).status == "blocked"

    warnings = [
        record for record in caplog.records
        if record.levelname == "WARNING"
        and "kanban PR-gate" in record.getMessage()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "RuntimeError" in warnings[0].getMessage()




def test_dispatch_tick_runs_no_subprocess_under_dispatch_lock(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZERO subprocesses under the writer lock — the whole class, not just gh.

    Round-1 finding 4 was "network latency must not extend the board's
    single-writer lock". Moving ``query_pr`` into the unlocked prefetch closed
    the ``gh`` seam but left its sibling: ``repo_context()`` ->
    ``_remotes_for()`` shells out to ``git remote -v`` with the same 5 s
    timeout, and the locked revalidation pass re-parses every blocked card. A
    degraded workspace (stale NFS mount, reclaimed scratch dir, held
    ``index.lock``) therefore still held the lock for seconds per card.

    The guard is deliberately on the SUBPROCESS seam rather than on either
    function, so any future shell-out added to the locked pass fails here.
    """
    repo = tmp_path / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:o/r.git"],
        cwd=repo, check=True,
    )

    lock_held = False
    under_lock: list[list[str]] = []
    real_run = subprocess.run

    @contextlib.contextmanager
    def tracked_lock(_db_path):
        nonlocal lock_held
        lock_held = True
        try:
            yield True
        finally:
            lock_held = False

    def watched_run(argv, *a, **k):
        if lock_held:
            under_lock.append(list(argv)[:3])
        return real_run(argv, *a, **k)

    monkeypatch.setattr(kanban_db_connect, "_dispatch_tick_lock", tracked_lock)
    monkeypatch.setattr(prg.subprocess, "run", watched_run)
    monkeypatch.setattr(prg, "query_pr", lambda repo_, number: _merged())

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="gated card", workspace_path=str(repo),
        )
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="merge PR #7 then unblock me", kind="needs_input",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )

    with kb.connect() as conn:
        result = kanban_db_dispatch.dispatch_once(conn, spawn_fn=lambda *a, **k: None)

    assert under_lock == [], f"subprocess ran under the dispatch lock: {under_lock}"
    # The bare #7 still resolved against the workspace remote, so the gate
    # actually fired — this is not a vacuous green from a skipped card.
    assert result.gate_auto_resolved == [tid]


def test_prefetch_repo_context_is_not_reused_when_the_card_changes(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card re-pointed between prefetch and the locked pass fails safe.

    The prefetch's repo-context snapshot is only valid for the workspace/body
    it was taken from. If either moved, the locked pass must take NO action
    rather than resolve a bare ``#N`` against a stale repository.
    """
    repo = tmp_path / "wt"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:o/r.git"],
        cwd=repo, check=True,
    )

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="gated card", workspace_path=str(repo),
        )
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="merge PR #7 then unblock me", kind="needs_input",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        prefetched = prg.prefetch_pr_gate_states(
            conn, query_fn=_stub({("o/r", 7): _merged()}),
        )
        # The card moves to a different repository after the snapshot.
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(tmp_path / "elsewhere"), tid),
        )
        conn.commit()
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({("o/r", 7): _merged()}), prefetched=prefetched,
        )

    assert [o.action for o in outcomes] == []
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Harness safety: a stubbed oracle may never write to a live board
# ---------------------------------------------------------------------------


def test_explicit_hermes_home_with_ambient_pin_is_not_treated_as_sandboxed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``HERMES_HOME`` alone does NOT sandbox kanban — prove it, don't assume.

    ``HERMES_KANBAN_DB`` outranks ``HERMES_HOME`` in
    :func:`kanban_db.kanban_db_path`, which is precisely why redirecting only
    ``HERMES_HOME`` let a 2026-09-21 probe write to the production board.

    Three arms, because BOTH conditions are load-bearing and neither is
    sufficient alone:

    * pin set, no opt-in  -> False (the original incident shape)
    * pin cleared, still no opt-in -> False (containment alone is not proof —
      the fleet's own env satisfies containment against the live board)
    * opt-in declared, pin neutralised by it -> True
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    live = tmp_path / "live" / "kanban.db"
    live.parent.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # The pin wins, so the DB is NOT inside the sandbox home.
    assert prg._db_is_sandboxed() is False

    # Dropping the pin makes the DB land inside HERMES_HOME — containment now
    # holds, and it is STILL not sandboxed, because nothing declared isolation.
    monkeypatch.delenv("HERMES_KANBAN_DB")
    assert prg._db_is_sandboxed() is False

    # The positive opt-in is what actually sandboxes it. The pin is cleared
    # here so the assertion is about the FLAG and nothing else.
    monkeypatch.setenv("HERMES_KANBAN_SANDBOX", "1")
    assert prg._db_is_sandboxed() is True


def test_stubbed_oracle_against_a_non_sandbox_db_raises_instead_of_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact 2026-09-21 shape: stubbed ``query_pr`` + live board pin.

    The probe redirected ``HERMES_HOME``, stubbed the oracle to return MERGED
    unconditionally, and wrote 20 ``gate_auto_resolved`` events to production.
    That combination must now fail closed and loudly, before any card is read.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    live = tmp_path / "live" / "kanban.db"
    live.parent.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live))

    with pytest.raises(prg.SandboxEscape) as excinfo:
        prg.reevaluate_pr_gates(None, query_fn=_stub({("o/r", 7): _merged()}))

    message = str(excinfo.value)
    assert "STUBBED PR oracle" in message
    assert "HERMES_KANBAN_SANDBOX=1" in message
    # Nothing was created: the refusal precedes every read and every write.
    assert not live.exists()


def test_monkeypatching_the_module_attribute_does_not_vouch_for_the_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebinding ``prg.query_pr`` must not let a stub pass as the real oracle.

    The guard compares against the oracle captured at import, so a harness that
    monkeypatches the module attribute — which is what a probe calling
    ``dispatch_once`` with no explicit ``query_fn`` does — is still caught.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    live = tmp_path / "live" / "kanban.db"
    live.parent.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live))
    monkeypatch.setattr(prg, "query_pr", lambda repo, number: _merged())

    with pytest.raises(prg.SandboxEscape):
        prg.reevaluate_pr_gates(None)

    assert not live.exists()


def test_the_real_oracle_is_allowed_even_in_a_test_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard gates FABRICATION, not test execution.

    A genuine ``gh``-backed run carries real evidence, so it must pass even
    under pytest against a non-sandbox board — otherwise the guard would break
    the production dispatcher the moment anything set ``PYTEST_CURRENT_TEST``.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    live = tmp_path / "live" / "kanban.db"
    live.parent.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live))

    prg.assert_write_allowed(None)
    prg.assert_write_allowed(prg._REAL_QUERY_PR)


def test_a_sandboxed_board_allows_a_stubbed_oracle(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correctly-sandboxed harness is unaffected — the guard is not a tax.

    This is the negative control for the three tests above: the whole existing
    suite runs stubbed oracles, and must keep running them.
    """
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    prg.assert_write_allowed(_stub({}))

    with kb.connect() as conn:
        tid = _blocked_card(conn, reason="merge o/r#7 then unblock me")
        outcomes = prg.reevaluate_pr_gates(
            conn, query_fn=_stub({("o/r", 7): _merged()}),
        )
    assert [o.action for o in outcomes] == ["unblocked"]
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status in {"ready", "running"}


_BARE_PROBE = '''
import json, os, sys
sys.path.insert(0, {repo!r})
# The incident shape: a bare script. NOTHING marks this as a test context.
for marker in ("PYTEST_CURRENT_TEST", "HERMES_IN_PYTEST"):
    os.environ.pop(marker, None)
os.environ["HERMES_HOME"] = {home!r}
os.environ["HERMES_KANBAN_DB"] = {live!r}

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_gate as prg

# Fabricated oracle: every PR is MERGED, and gh is never consulted.
prg.query_pr = lambda repo, number: {{
    "state": "MERGED", "mergedAt": "2026-09-21T07:32:50Z",
    "mergeCommitSha": "deadbeefcafe1234",
}}

kb.init_db()
refused = False
with kb.connect() as conn:
    tid = kb.create_task(conn, title="gated", assignee="daedalus-opus")
    kb.claim_task(conn, tid)
    kb.block_task(
        conn, tid, reason="merge o/r#7 then unblock me", kind="needs_input",
        expected_run_id=kb.get_task(conn, tid).current_run_id,
    )
    try:
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
    except prg.SandboxEscape:
        refused = True
    status = kb.get_task(conn, tid).status
    events = conn.execute(
        "SELECT COUNT(*) c FROM task_events WHERE kind = 'gate_auto_resolved'"
    ).fetchone()["c"]
print(json.dumps({{
    "in_test_context": prg._in_test_context(),
    "refused": refused, "status": status, "gate_auto_resolved": events,
}}))
'''


def test_a_bare_probe_script_with_a_stubbed_oracle_is_refused(
    tmp_path: Path,
) -> None:
    """The 2026-09-21 incident EXACTLY: a bare ``python probe.py``, no markers.

    Every other sandbox test in this file runs under pytest, so all of them
    satisfy ``_in_test_context()`` for free — none of them can observe the arm
    that matters. The probe that actually wrote 20 ``gate_auto_resolved`` events
    to production was a plain script: it set neither ``PYTEST_CURRENT_TEST`` nor
    ``HERMES_IN_PYTEST``, so a guard preconditioned on a test marker returns
    before it ever looks at the oracle.

    Fabrication must therefore be provable from the ORACLE IDENTITY alone, which
    needs no opt-in. This test runs in a child interpreter with both markers
    scrubbed so the precondition cannot be satisfied accidentally.
    """
    repo_root = Path(__file__).resolve().parents[2]
    home = tmp_path / ".hermes"
    home.mkdir()
    live = tmp_path / "pretend_production" / "kanban.db"
    live.parent.mkdir()
    script = tmp_path / "probe.py"
    script.write_text(
        _BARE_PROBE.format(repo=str(repo_root), home=str(home), live=str(live))
    )

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTEST_", "HERMES_"))}
    env["PATH"] = os.environ.get("PATH", "")
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    observed = json.loads(proc.stdout.strip().splitlines()[-1])

    # The precondition the old guard depended on is genuinely absent...
    assert observed["in_test_context"] is False
    # ...and the write is refused anyway, on oracle identity alone.
    assert observed["refused"] is True
    assert observed["gate_auto_resolved"] == 0
    assert observed["status"] == "blocked"


_UNSANDBOXED_PROBE = '''
import json, os, sys
sys.path.insert(0, {repo!r})
# The careless shape: a bare script that sandboxes NOTHING. It simply inherits
# the dispatcher's worker env, which pins HERMES_KANBAN_DB at the live board.
for marker in ("PYTEST_CURRENT_TEST", "HERMES_IN_PYTEST"):
    os.environ.pop(marker, None)
os.environ["HERMES_HOME"] = {home!r}
os.environ["HERMES_KANBAN_DB"] = {live!r}
os.environ["HERMES_KANBAN_HOME"] = {live_root!r}

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_gate as prg

refused = False
try:
    prg.assert_write_allowed(lambda repo, number: {{"state": "MERGED"}})
except prg.SandboxEscape:
    refused = True
print(json.dumps({{
    "db": str(kb.kanban_db_path()),
    "shared_root": str(kb.kanban_home()),
    "declared_home": os.environ["HERMES_HOME"],
    "sandboxed": prg._db_is_sandboxed(),
    "refused": refused,
}}))
'''


def test_a_board_outside_the_declared_hermes_home_is_not_sandboxed(
    tmp_path: Path,
) -> None:
    """A live board must never be able to prove ITSELF sandboxed.

    ``_db_is_sandboxed`` is the only remaining escape valve once the guard keys
    on oracle identity, so its notion of "inside the sandbox" has to mean
    "inside the home this process DECLARED" — not "internally consistent".

    The hole: ``kanban_home()`` deliberately resolves the SHARED kanban root
    (the board is shared across profiles by design) and the pins outrank
    ``HERMES_HOME``. So for a probe that sandboxes nothing, the production DB
    sits under the production kanban root, ``target.is_relative_to(root)`` is
    True, and a fabricated oracle is waved through onto the live board — the
    strictly-more-careless sibling of the 2026-09-21 incident.

    Runs in a child interpreter: the pins must be real process env, and no
    pytest marker may be present.
    """
    repo_root = Path(__file__).resolve().parents[2]
    home = tmp_path / ".hermes"
    home.mkdir()
    live_root = tmp_path / "pretend_production"
    live = live_root / "kanban.db"
    live_root.mkdir()
    script = tmp_path / "unsandboxed_probe.py"
    script.write_text(
        _UNSANDBOXED_PROBE.format(
            repo=str(repo_root), home=str(home),
            live=str(live), live_root=str(live_root),
        )
    )

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTEST_", "HERMES_"))}
    env["PATH"] = os.environ.get("PATH", "")
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    observed = json.loads(proc.stdout.strip().splitlines()[-1])

    # Precondition: the DB is internally consistent with the SHARED root, which
    # is exactly what made the old predicate answer True.
    assert observed["db"].startswith(observed["shared_root"])
    # ...but it is NOT under the home this process declared.
    assert not observed["db"].startswith(observed["declared_home"])
    # So isolation is not proven, and the fabricated oracle is refused.
    assert observed["sandboxed"] is False
    assert observed["refused"] is True


_FLEET_ROOT_PROBE = '''
import json, os, sys
sys.path.insert(0, {repo!r})
# The FLEET shape: the declared HERMES_HOME IS the board's own root, which is
# exactly what ~20 installed launchd jobs export. No pins, no pytest marker.
for marker in ("PYTEST_CURRENT_TEST", "HERMES_IN_PYTEST"):
    os.environ.pop(marker, None)
for pin in ("HERMES_KANBAN_DB", "HERMES_KANBAN_HOME",
            "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_ATTACHMENTS_ROOT",
            "HERMES_KANBAN_SANDBOX"):
    os.environ.pop(pin, None)
os.environ["HERMES_HOME"] = {home!r}

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_gate as prg

prg.query_pr = lambda repo, number: {{
    "state": "MERGED", "mergedAt": "2026-09-21T07:32:50Z",
    "mergeCommitSha": "deadbeefcafe1234",
}}

kb.init_db()
refused = False
with kb.connect() as conn:
    tid = kb.create_task(conn, title="gated", assignee="daedalus-opus")
    kb.claim_task(conn, tid)
    kb.block_task(
        conn, tid, reason="merge o/r#7 then unblock me", kind="needs_input",
        expected_run_id=kb.get_task(conn, tid).current_run_id,
    )
    try:
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: None)
    except prg.SandboxEscape:
        refused = True
    status = kb.get_task(conn, tid).status
    events = conn.execute(
        "SELECT COUNT(*) c FROM task_events WHERE kind = 'gate_auto_resolved'"
    ).fetchone()["c"]

# The production control: the in-gateway dispatcher runs under exactly this
# env, so the real oracle and the no-argument call must still be allowed.
real_allowed = stub_allowed = None
try:
    prg.assert_write_allowed(None)
    prg.assert_write_allowed(prg._REAL_QUERY_PR)
    real_allowed = True
except prg.SandboxEscape:
    real_allowed = False
try:
    prg.assert_write_allowed(lambda repo, number: {{"state": "MERGED"}})
    stub_allowed = True
except prg.SandboxEscape:
    stub_allowed = False

print(json.dumps({{
    "db": str(kb.kanban_db_path()),
    "declared_home": os.environ["HERMES_HOME"],
    "sandboxed": prg._db_is_sandboxed(),
    "in_test_context": prg._in_test_context(),
    "refused": refused, "status": status, "gate_auto_resolved": events,
    "real_allowed": real_allowed, "stub_allowed": stub_allowed,
}}))
'''


def test_a_board_inside_the_declared_hermes_home_is_still_not_sandboxed(
    tmp_path: Path,
) -> None:
    """The FLEET-DEFAULT env must not prove itself sandboxed.

    Round-6 anchored :func:`_db_is_sandboxed` on containment under the DECLARED
    ``HERMES_HOME``. That closes the OUTSIDE case (the sibling test above) but
    the fleet's own environment satisfies the INSIDE case: ~20 installed launchd
    jobs export ``HERMES_HOME=~/.hermes`` and the live ``kanban.db`` sits
    directly inside it. Measured at head ``108e50da6c`` with no pins and no
    pytest marker: ``sandboxed=True``, stub ``ALLOWED`` — i.e. the 2026-09-21
    incident still writes, from the most ordinary env on the box.

    Containment is therefore not evidence. Isolation must require a POSITIVE
    opt-in that no production process sets (``HERMES_KANBAN_SANDBOX=1``, the
    very flag the refusal message already prescribes).

    Runs in a child interpreter because the env has to be real process state
    and no pytest marker may be present.
    """
    repo_root = Path(__file__).resolve().parents[2]
    home = tmp_path / "hermes-live"
    home.mkdir()
    script = tmp_path / "fleet_root_probe.py"
    script.write_text(_FLEET_ROOT_PROBE.format(repo=str(repo_root), home=str(home)))

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTEST_", "HERMES_"))}
    env["PATH"] = os.environ.get("PATH", "")
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    observed = json.loads(proc.stdout.strip().splitlines()[-1])

    # Precondition: this IS the fleet shape — the board resolves inside the
    # home this process declared, which is what the old predicate rewarded.
    assert observed["db"].startswith(observed["declared_home"])
    assert observed["in_test_context"] is False

    # Containment alone must no longer grant isolation.
    assert observed["sandboxed"] is False
    assert observed["stub_allowed"] is False
    assert observed["refused"] is True
    assert observed["gate_auto_resolved"] == 0
    assert observed["status"] == "blocked"

    # Production control: the real oracle is untouched in this same env.
    assert observed["real_allowed"] is True


def test_dispatch_once_propagates_a_sandbox_escape_instead_of_absorbing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end replay of the 2026-09-21 incident, through ``dispatch_once``.

    The dispatcher's PR-gate hooks are deliberately fail-open so a diagnostic
    can never brick a tick. A sandbox escape is the one exception: absorbing it
    would turn a loud, actionable refusal into a WARN the harness author never
    reads — which is precisely how the original probe got to write 20 events.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    live = tmp_path / "pretend_production" / "kanban.db"
    live.parent.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live))
    # The incident shape exactly: the module attribute is rebound, so the
    # dispatcher's own no-argument call picks up the fabricated oracle.
