"""Dispatcher reconciliation through real SQLite, gh subprocesses and HTTP."""
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_db_connect import connect
from tests.hermes_cli.test_kanban_pr_acceptance import github  # noqa: F401


URL = "https://github.com/acme/repo/pull/7"


@pytest.fixture(autouse=True)
def reset_cache():
    from hermes_cli import kanban_pr_reconcile as reconcile
    cache = getattr(reconcile, "_CACHE", {})
    cache.clear()
    yield
    cache.clear()


def review(conn, contract="acme/repo", metadata=None):
    tid = kb.create_task(conn, title="Publish", completion_contract=contract)
    assert kb.request_review(conn, tid, summary="Ready for review", metadata=(
        {"published_pr": URL} if metadata is None else metadata))
    return tid


def tick(conn, **kwargs):
    # No worker processes are needed to reconcile parked reviews.
    return dispatch.dispatch_once(conn, max_spawn=0, **kwargs)


def events(conn, tid):
    return [tuple(row) for row in conn.execute(
        "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (tid,))]


@pytest.mark.platforms("linux")
def test_dispatch_completes_verified_merged_review_once(github):
    github["pr_state"] = "MERGED"
    with connect() as conn:
        tid = review(conn)
        child = kb.create_task(conn, title="Dependent", parents=[tid])
        assert kb.get_task(conn, child).status == "todo"
        result = tick(conn)
        assert kb.get_task(conn, tid).status == "done"
        assert kb.get_task(conn, child).status == "ready"
        assert result.completed_merged_prs == [tid]
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])
        assert receipt["ok"] and receipt["merged"] and receipt["head_sha"] == github["head"]
        before = events(conn, tid)
        requests = list(github["requests"])
        assert tick(conn).completed_merged_prs == []
        assert events(conn, tid) == before
        assert github["requests"] == requests


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("state,head,conclusion,classification", [
    ("OPEN", "a" * 40, "success", "open"),
    ("CLOSED", None, "success", "closed"),
    ("MERGED", "a" * 40, "failure", "failure"),
    ("MERGED", "a" * 40, "pending", "pending"),
])
def test_nonaccepting_polls_are_read_only_and_cached(github, state, head, conclusion, classification):
    github.update(pr_state=state, head=head, conclusion=conclusion)
    with connect() as conn:
        tid = review(conn)
        before = events(conn, tid)
        task = kb.get_task(conn, tid)
        first = tick(conn)
        requests = list(github["requests"])
        second = tick(conn)
        assert kb.get_task(conn, tid) == task
        assert events(conn, tid) == before
        assert first.pr_reconciliation == second.pr_reconciliation == [(tid, URL, classification)]
        assert github["requests"] == requests


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("value", ["omitted", None, 42, "broken",
                                  "https://github.com/other/repo/pull/8",
                                  "https://github.com/acme/repo/pull/8"])
def test_review_receipt_inheritance_has_explicit_barriers(github, value):
    github["pr_state"] = "MERGED"
    with connect() as conn:
        tid = review(conn, contract=URL)
        kb._synthesize_ended_run(conn, tid, outcome="review_requested",
                                metadata={} if value == "omitted" else {"published_pr": value})
        tick(conn)
        assert (kb.get_task(conn, tid).status == "done") is (value == "omitted")
        assert bool(github["requests"]) is (value == "omitted")


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("mode", ["dry_run", "no_lock_key", "null_lock_key", "wrong_lock_key"])
def test_no_poll_without_write_permission_and_stable_lock(github, monkeypatch, mode):
    from hermes_cli import kanban_pr_reconcile as reconcile
    github["pr_state"] = "MERGED"
    with connect() as conn:
        tid = review(conn)
        before = events(conn, tid)
        if mode == "no_lock_key":
            def unavailable(**kwargs):
                raise OSError("unavailable")
            monkeypatch.setattr(kb, "kanban_db_path", unavailable)
        elif mode == "null_lock_key":
            monkeypatch.setattr(kb, "kanban_db_path", lambda **kwargs: None)
        elif mode == "wrong_lock_key":
            wrong = kb.kanban_db_path().with_name("other.db")
            monkeypatch.setattr(kb, "kanban_db_path", lambda **kwargs: wrong)
        result = tick(conn, dry_run=mode == "dry_run")
        assert not result.skipped_locked
        assert kb.get_task(conn, tid).status == "review"
        assert events(conn, tid) == before
        assert github["requests"] == []
        assert not reconcile._CACHE


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("phase", ["http", "transaction"])
def test_replaced_handoff_cannot_complete_with_stale_evidence(github, monkeypatch, phase):
    from hermes_cli import kanban_pr_acceptance_store as store
    github["pr_state"] = "MERGED"
    with connect() as conn:
        tid = review(conn)
        def replace(target):
            kb._synthesize_ended_run(target, tid, outcome="review_requested",
                                    metadata={"published_pr": URL.replace("/7", "/8")})
        if phase == "http":
            def race():
                with connect() as rival:
                    replace(rival)
            github["race"] = race
        else:
            original = store.record_acceptance
            def record(target, task_id, acceptance):
                assert target.in_transaction
                replace(target)
                return original(target, task_id, acceptance)
            monkeypatch.setattr(store, "record_acceptance", record)
        assert tick(conn).completed_merged_prs == []
        task = kb.get_task(conn, tid)
        assert task.status == "review" and task.completion_contract == "acme/repo"
        assert not conn.execute("SELECT id FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
                                (tid,)).fetchall()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("error,classification", [(401, "auth"), (500, "infra")])
def test_remote_failures_are_read_only_sanitized_and_back_off(github, monkeypatch, caplog, error, classification):
    from hermes_cli import kanban_pr_reconcile as reconcile
    clock = [1000.0]
    monkeypatch.setattr(reconcile.time, "monotonic", lambda: clock[0])
    github["http_error"] = error
    with connect() as conn:
        tid = review(conn)
        before, task = events(conn, tid), kb.get_task(conn, tid)
        for attempt in range(7):
            result = tick(conn)
            assert result.pr_reconciliation == [(tid, URL, classification)]
            count = len(github["requests"])
            cached = next(iter(reconcile._CACHE.values()))
            delay = cached.expires - clock[0]
            assert delay == min(60 * 2 ** min(attempt, 4), 900)
            assert tick(conn).pr_reconciliation == result.pr_reconciliation
            assert len(github["requests"]) == count
            clock[0] = cached.expires
        assert events(conn, tid) == before and kb.get_task(conn, tid) == task
        assert "fixture-secret" not in repr(result) + caplog.text
        github.pop("http_error")
        github["pr_state"] = "MERGED"
        assert tick(conn).completed_merged_prs == [tid]


@pytest.mark.platforms("linux")
def test_blocked_github_does_not_hold_board_lock_or_stop_sibling_tick(github, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from hermes_cli import kanban_db_connect as kbc
    entered, release = Event(), Event()
    github["pr_state"] = "MERGED"
    def block_http():
        entered.set()
        assert release.wait(10)
    github["race"] = block_http
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    with connect() as conn:
        tid = review(conn)
        def background_tick():
            with connect() as worker:
                return tick(worker)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(background_tick)
            try:
                assert entered.wait(10)
                with kbc._dispatch_tick_lock(kb.kanban_db_path()) as held:
                    assert held
                queued = kb.create_task(conn, title="ordinary dispatch")
                result = dispatch.dispatch_once(conn, max_spawn=1)
                assert not result.skipped_locked
                assert queued in result.skipped_unassigned
                assert kb.get_task(conn, tid).status == "review"
            finally:
                release.set()
            assert future.result(timeout=10).completed_merged_prs == [tid]
        assert github["requests"].count("/graphql") == 1


@pytest.mark.platforms("linux")
def test_cache_expiry_and_eviction_are_thread_safe(github, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from hermes_cli import kanban_pr_reconcile as reconcile
    monkeypatch.setattr(reconcile, "_CACHE_LIMIT", 4)
    with connect() as conn:
        tid = review(conn)
        snapshot = reconcile.snapshot_reviews(conn, kb.kanban_db_path().resolve())[0]
        path = kb.kanban_db_path()
        tick(conn)
        first = len(github["requests"])
        # Expiry forces a fresh read, but the intervening tick is cached.
        tick(conn)
        assert len(github["requests"]) == first
        with reconcile._CACHE_LOCK:
            next(iter(reconcile._CACHE.values())).expires = 0
        tick(conn)
        assert len(github["requests"]) > first
        def fetch(number):
            bound = replace(snapshot, published_pr=URL.replace("/7", f"/{number}"))
            board_path = path.with_name(f"board-{number % 3}.db")
            return reconcile._cached_acceptance(bound, board_path, allow_query=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(fetch, list(range(1, 13)) * 2))
        assert any(queried for _, queried in results)
        assert len(reconcile._CACHE) <= 4
        assert all(entry.receipt is not None for entry in reconcile._CACHE.values())
        assert kb.get_task(conn, tid).status == "review"


@pytest.mark.platforms("linux")
def test_poll_budget_does_not_starve_later_reviews(github, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import kanban_pr_reconcile as reconcile
    clock = [1000.0]
    monkeypatch.setattr(reconcile, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(reconcile, "_MAX_POLLS_PER_TICK", 2)
    with connect() as conn:
        for number in range(1, 4):
            review(conn, metadata={"published_pr": URL.replace("/7", f"/{number}")})
        tick(conn)
        assert github["requests"].count("/graphql") == 2
        clock[0] += 61
        tick(conn)
        assert {path for path in github["requests"] if "/pulls/" in path} == {
            f"/repos/acme/repo/pulls/{number}" for number in range(1, 4)}
