from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import threading

import pytest

from hermes_cli import kanban_db as kb


BROAD_BODY = """
Investigate and diagnose the current implementation. Implement the remediation.
Run focused tests, dual-runtime tests, Ruff, compilation, and the full suite.
Freeze hashes and receipts. Run one independent adversarial review. After PASS,
run a separate controller and prepare the gate without activating it.
"""


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _assessment_event(conn, task_id: str):
    events = [
        e for e in kb.list_events(conn, task_id) if e.kind == "granularity_assessed"
    ]
    assert len(events) == 1
    return events[0]


def _active_run_id(conn, task_id: str) -> int:
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None
    return int(task.current_run_id)


def test_warn_policy_records_assessment_without_changing_status(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Close implementation, review, and controller",
            body=BROAD_BODY,
            assignee="worker",
            granularity_policy="warn",
        )
        task = kb.get_task(conn, task_id)
        event = _assessment_event(conn, task_id)

    assert task is not None
    assert task.status == "ready"
    assert event.payload["verdict"] == "split"
    assert event.payload["policy"] == "warn"
    assert event.payload["auto_triaged"] is False


def test_triage_policy_stops_oversized_card_before_dispatch(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Close implementation, review, and controller",
            body=BROAD_BODY,
            assignee="worker",
            granularity_policy="triage",
        )
        task = kb.get_task(conn, task_id)
        event = _assessment_event(conn, task_id)

    assert task is not None
    assert task.status == "triage"
    assert event.payload["auto_triaged"] is True


def test_allow_policy_preserves_intentional_broad_card(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Intentional broad migration",
            body=BROAD_BODY,
            assignee="worker",
            granularity_policy="allow",
        )
        task = kb.get_task(conn, task_id)
        event = _assessment_event(conn, task_id)

    assert task is not None
    assert task.status == "ready"
    assert event.payload["verdict"] == "split"
    assert event.payload["policy"] == "allow"
    assert event.payload["auto_triaged"] is False


def test_explicit_policy_still_uses_configured_agent_max_turns(
    kanban_home,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "agent": {"max_turns": 8},
            "kanban": {"granularity_guard": "triage"},
        },
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Implement and verify parser correction",
            body="Patch parser.py and run its focused pytest regression.",
            granularity_policy="warn",
        )
        event = _assessment_event(conn, task_id)

    assert event.payload["policy"] == "warn"
    assert event.payload["max_turns"] == 8
    assert "estimated_turns_exceed_budget" in event.payload["reasons"]


def test_invalid_granularity_policy_fails_closed_to_warn(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Broad card",
            body=BROAD_BODY,
            assignee="worker",
            granularity_policy="surprise",
        )
        task = kb.get_task(conn, task_id)
        event = _assessment_event(conn, task_id)

    assert task is not None
    assert task.status == "ready"
    assert event.payload["policy"] == "warn"
    assert event.payload["policy_invalid"] is True


def test_review_key_deduplicates_same_postimage(kanban_home) -> None:
    digest = "a" * 64
    claims = "b" * 64
    review_key = f"{digest}:{claims}"
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Independent review",
            assignee="reviewer-a",
            review_key=review_key,
        )
        second = kb.create_task(
            conn,
            title="Duplicate independent review",
            assignee="reviewer-b",
            review_key=review_key,
        )

    assert second == first


def test_review_key_deduplication_is_atomic_across_concurrent_creators(
    kanban_home,
    monkeypatch,
) -> None:
    digest = "f" * 64
    claims = "e" * 64
    review_key = f"{digest}:{claims}"
    enter_write = threading.Barrier(2)
    original_write_txn = kb.write_txn

    @contextlib.contextmanager
    def synchronized_write_txn(conn):
        enter_write.wait(timeout=5)
        with original_write_txn(conn) as transaction:
            yield transaction

    monkeypatch.setattr(kb, "write_txn", synchronized_write_txn)
    results: list[str] = []
    errors: list[BaseException] = []

    def create_review() -> None:
        try:
            with kb.connect() as conn:
                results.append(
                    kb.create_task(
                        conn,
                        title="Concurrent exact review",
                        review_key=review_key,
                    )
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    workers = [threading.Thread(target=create_review) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert len(results) == 2
    assert len(set(results)) == 1

    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?",
            (f"review:v1:{review_key}",),
        ).fetchall()
    assert len(rows) == 1


def test_review_key_allows_new_hash_or_explicit_round(kanban_home) -> None:
    digest = "b" * 64
    claims = "c" * 64
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Review",
            review_key=f"{digest}:{claims}",
        )
        second = kb.create_task(
            conn,
            title="Review new hash",
            review_key=f"{'d' * 64}:{claims}",
        )
        round_two = kb.create_task(
            conn,
            title="Review same hash round two",
            review_key=f"{digest}:{claims}:round-2",
        )

    assert len({first, second, round_two}) == 3


def test_review_key_rejects_conflicting_idempotency_key(kanban_home) -> None:
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="review_key.*idempotency_key"):
            kb.create_task(
                conn,
                title="Ambiguous dedup contract",
                review_key=f"{'d' * 64}:{'e' * 64}",
                idempotency_key="other",
            )


@pytest.mark.parametrize(
    "body",
    [
        f"REVIEW_KEY: {'c' * 64}:{'d' * 64}\n\nReview the wrong postimage.",
        (
            f"REVIEW_KEY: {'a' * 64}:{'b' * 64}\n"
            f"REVIEW_KEY: {'a' * 64}:{'b' * 64}\n\nReview the postimage."
        ),
    ],
)
def test_exact_review_rejects_conflicting_or_multiple_body_identity(
    kanban_home,
    body,
) -> None:
    review_key = f"{'a' * 64}:{'b' * 64}"

    with kb.connect() as conn:
        with pytest.raises(ValueError, match="REVIEW_KEY"):
            kb.create_task(
                conn,
                title="Contradictory exact review",
                body=body,
                review_key=review_key,
            )
        rows = conn.execute(
            "SELECT id FROM tasks WHERE title = 'Contradictory exact review'"
        ).fetchall()

    assert rows == []


def test_generic_idempotency_key_cannot_reserve_review_namespace(kanban_home) -> None:
    postimage = "e" * 64
    claims = "f" * 64
    reserved = f"review:v1:{postimage}:{claims}"

    with kb.connect() as conn:
        with pytest.raises(ValueError, match="reserved review namespace"):
            kb.create_task(
                conn,
                title="Not actually a review",
                idempotency_key=reserved,
            )
        review_id = kb.create_task(
            conn,
            title="Independent review",
            review_key=f"{postimage}:{claims}",
        )
        task = kb.get_task(conn, review_id)

    assert task is not None
    assert task.idempotency_key == reserved


def test_exact_review_rejects_ambiguous_legacy_reserved_incumbent(kanban_home) -> None:
    postimage = "a" * 64
    claims = "b" * 64
    review_key = f"{postimage}:{claims}"
    reserved = f"review:v1:{review_key}"

    with kb.connect() as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, title, status, created_at, idempotency_key)
               VALUES ('poisoned', 'generic incumbent', 'ready', 1, ?)""",
            (reserved,),
        )
        conn.commit()

        with pytest.raises(ValueError, match="ambiguous legacy exact-review"):
            kb.create_task(
                conn,
                title="Real exact review",
                review_key=review_key,
                granularity_policy="allow",
            )
        rows = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? ORDER BY id",
            (reserved,),
        ).fetchall()

    assert [row["id"] for row in rows] == ["poisoned"]


def test_exact_review_persists_durable_contract_and_canonical_hashes(
    kanban_home,
) -> None:
    postimage = "C" * 64
    claims = "D" * 64
    review_key = f" {postimage}:{claims}:ROUND-2 "

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Exact review",
            review_key=review_key,
        )
        duplicate = kb.create_task(
            conn,
            title="Duplicate exact review",
            review_key=review_key.lower(),
        )
        row = conn.execute(
            """SELECT idempotency_key, review_contract,
                      review_postimage_sha256, review_claims_sha256
                 FROM tasks WHERE id = ?""",
            (task_id,),
        ).fetchone()

    assert duplicate == task_id
    assert (
        row["idempotency_key"]
        == f"review:v1:{postimage.lower()}:{claims.lower()}:round-2"
    )
    assert row["review_contract"] == "exact_review_v1"
    assert row["review_postimage_sha256"] == postimage.lower()
    assert row["review_claims_sha256"] == claims.lower()


def test_exact_review_rejects_multiple_non_archived_incumbents(kanban_home) -> None:
    postimage = "e" * 64
    claims = "f" * 64
    review_key = f"{postimage}:{claims}"
    reserved = f"review:v1:{review_key}"

    with kb.connect() as conn:
        conn.executemany(
            """INSERT INTO tasks
               (id, title, status, created_at, idempotency_key,
                review_contract, review_postimage_sha256, review_claims_sha256)
               VALUES (?, 'duplicate incumbent', 'ready', 1, ?,
                       'exact_review_v1', ?, ?)""",
            (
                ("duplicate-a", reserved, postimage, claims),
                ("duplicate-b", reserved, postimage, claims),
            ),
        )
        conn.commit()

        with pytest.raises(ValueError, match="ambiguous legacy exact-review"):
            kb.create_task(
                conn,
                title="Exact review",
                review_key=review_key,
            )


def test_compact_context_is_bounded_and_full_context_is_recoverable(
    kanban_home,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Child with large inherited context",
            body="Mandatory body marker. " + ("B" * 6000),
            assignee="worker",
        )
        parent_ids: list[str] = []
        for index in range(12):
            parent_id = kb.create_task(conn, title=f"parent-{index}")
            parent_ids.append(parent_id)
            conn.execute(
                "UPDATE tasks SET status = 'done', result = ? WHERE id = ?",
                (f"parent-result-{index}-" + ("R" * 4000), parent_id),
            )
            conn.execute(
                """INSERT INTO task_runs
                   (task_id, profile, status, started_at, ended_at, outcome,
                    summary, metadata, error)
                   VALUES (?, 'worker', 'done', 1, 2, 'completed', ?, ?, NULL)""",
                (
                    parent_id,
                    f"parent-summary-{index}-" + ("S" * 4000),
                    '{"evidence":"' + ("M" * 4000) + '"}',
                ),
            )
            conn.execute(
                "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
                (parent_id, task_id),
            )

        for attempt in range(6):
            conn.execute(
                """INSERT INTO task_runs
                   (task_id, profile, status, started_at, ended_at, outcome,
                    summary, metadata, error)
                   VALUES (?, 'worker', 'failed', ?, ?, 'crashed', ?, ?, ?)""",
                (
                    task_id,
                    10 + attempt,
                    20 + attempt,
                    f"attempt-summary-{attempt}-" + ("A" * 4000),
                    '{"attempt":' + str(attempt) + ',"blob":"' + ("X" * 4000) + '"}',
                    f"attempt-error-{attempt}-" + ("E" * 3000),
                ),
            )

        for index in range(20):
            kb.add_comment(
                conn,
                task_id,
                author="worker",
                body=f"comment-{index}-" + ("C" * 1500),
            )

        full = kb.build_worker_context(conn, task_id, compact=False)
        compact = kb.build_worker_context(conn, task_id, compact=True)

    assert "Mandatory body marker" in compact
    assert "attempt-summary-5" in compact
    assert "attempt-summary-0" not in compact
    assert parent_ids[0] in compact
    assert parent_ids[-1] in compact
    assert "full_context=true" in compact
    assert len(compact) <= 60_000
    assert len(compact) < len(full) // 2
    assert "attempt-summary-0" in full
    assert "comment-0" in full


def test_compact_context_has_aggregate_utf8_budget_with_many_parents(
    kanban_home,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="High fan-in child",
            body="bounded body",
            assignee="worker",
        )
        parent_rows = [
            (
                f"p{index:07d}",
                f"parent-{index}",
                "done" if index >= 7992 else "todo",
                1,
            )
            for index in range(8000)
        ]
        with kb.write_txn(conn):
            conn.executemany(
                "INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, ?, ?)",
                parent_rows,
            )
            conn.executemany(
                "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
                ((row[0], task_id) for row in parent_rows),
            )

        compact = kb.build_worker_context(conn, task_id, compact=True)

    assert len(compact.encode("utf-8")) <= kb._CTX_COMPACT_MAX_TOTAL_BYTES
    assert "full_context=true" in compact
    assert "7992 earlier parents" in compact


def test_budget_failure_persists_one_bounded_unverified_handoff(kanban_home) -> None:
    source = "completed: schema work\nremaining: independent review\n" + ("X" * 9000)
    partial = "[partial_unverified]\n" + source
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Budget handoff target",
            assignee="worker",
            max_retries=2,
        )
        assert kb.claim_task(conn, task_id) is not None

        blocked = kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (60/60)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            partial_summary=partial,
            event_payload_extra={"budget_used": 60, "budget_max": 60},
        )

        task = kb.get_task(conn, task_id)
        runs = kb.list_runs(conn, task_id)
        comments = kb.list_comments(conn, task_id)
        context = kb.build_worker_context(conn, task_id, compact=True)

    assert blocked is False
    assert task is not None and task.status == "ready"
    assert len(runs) == 1
    assert runs[0].summary is not None
    assert runs[0].metadata is not None
    assert runs[0].summary.startswith("[partial_unverified]")
    assert "COMPLETED: schema work" in runs[0].summary
    assert len(runs[0].summary) <= 4200
    assert runs[0].metadata["partial_summary_full"] == source
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert runs[0].metadata["partial_summary_sha256"] == digest
    assert f"FULL_HANDOFF_SHA256: {digest}" in runs[0].summary
    assert comments == []
    assert context.count("[partial_unverified]") == 1
    assert '_metadata_: `{"partial_summary_full"' not in context
    assert "CONTEXT_REF:" in context
    assert f'"sha256":"{digest}"' in context


def test_full_context_preserves_historical_character_caps(kanban_home) -> None:
    body = "é" * (kb._CTX_MAX_BODY_BYTES + 5)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Full compatibility", body=body)
        full = kb.build_worker_context(conn, task_id, compact=False)

    expected = "é" * kb._CTX_MAX_BODY_BYTES + "… [truncated, 5 chars omitted]"
    assert expected in full


def test_compact_context_prioritizes_latest_handoff_over_attachment_fanout(
    kanban_home,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Attachment-heavy retry")
        conn.executemany(
            """INSERT INTO task_attachments
               (task_id, filename, stored_path, content_type, size, uploaded_by, created_at)
               VALUES (?, ?, ?, NULL, 1, 'fixture', 1)""",
            (
                (
                    task_id,
                    f"attachment-{index}.txt",
                    "/tmp/" + (f"path-{index}-" * 80),
                )
                for index in range(150)
            ),
        )
        conn.execute(
            """INSERT INTO task_runs
               (task_id, profile, status, started_at, ended_at, outcome,
                summary, metadata, error)
               VALUES (?, 'worker', 'gave_up', 2, 3, 'gave_up', ?, ?, NULL)""",
            (
                task_id,
                "[partial_unverified]\nCOMPLETED: LATEST_HANDOFF_SENTINEL",
                '{"partial_summary_full":"private full source",'
                '"partial_summary_sha256":"' + ("a" * 64) + '"}',
            ),
        )
        conn.commit()
        compact = kb.build_worker_context(conn, task_id, compact=True)

    assert len(compact.encode("utf-8")) <= kb._CTX_COMPACT_MAX_TOTAL_BYTES
    assert "LATEST_HANDOFF_SENTINEL" in compact
    assert "CONTEXT_REF:" in compact
    assert '_metadata_: `{"partial_summary_full"' not in compact


def test_compact_context_reserves_latest_handoff_under_early_field_starvation(
    kanban_home,
) -> None:
    source = "FULL_SOURCE:" + ("🧭" * 20_000) + ":SOURCE_TAIL_SENTINEL"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="T" * 60_000,
            body="B" * 60_000,
            assignee="worker",
            granularity_policy="allow",
        )
        conn.executemany(
            """INSERT INTO task_attachments
               (task_id, filename, stored_path, content_type, size, uploaded_by, created_at)
               VALUES (?, ?, ?, ?, 1, 'fixture', 1)""",
            (
                (
                    task_id,
                    "F" * 20_000,
                    "/tmp/" + ("P" * 20_000),
                    "application/" + ("C" * 20_000),
                )
                for _ in range(20)
            ),
        )
        conn.execute(
            """INSERT INTO task_comments
               (task_id, author, body, created_at) VALUES (?, ?, ?, 1)""",
            (task_id, "A" * 20_000, "COMMENT:" + ("M" * 20_000)),
        )
        cursor = conn.execute(
            """INSERT INTO task_runs
               (task_id, profile, status, started_at, ended_at, outcome,
                summary, metadata, error)
               VALUES (?, 'worker', 'gave_up', 2, 3, 'gave_up', ?, ?, NULL)""",
            (
                task_id,
                "[partial_unverified]\nCOMPLETED: LATEST_HANDOFF_SENTINEL",
                json.dumps({
                    "partial_summary_full": source,
                    "partial_summary_sha256": digest,
                }),
            ),
        )
        assert cursor.lastrowid is not None
        run_id = int(cursor.lastrowid)
        conn.commit()
        compact = kb.build_worker_context(conn, task_id, compact=True)

    assert len(compact.encode("utf-8")) <= kb._CTX_COMPACT_MAX_TOTAL_BYTES
    assert "LATEST_HANDOFF_SENTINEL" in compact
    assert "CONTEXT_REF:" in compact
    assert f'"run_id":{run_id}' in compact
    assert f'"sha256":"{digest}"' in compact
    assert "--field partial_summary_full" in compact
    assert "[field truncated]" in compact


def test_read_run_metadata_field_recovers_exact_hash_bound_source(kanban_home) -> None:
    source = "FULL_SOURCE:" + ("🧭" * 20_000) + ":SOURCE_TAIL_SENTINEL"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Recovery target")
        other_task_id = kb.create_task(conn, title="Other task")
        cursor = conn.execute(
            """INSERT INTO task_runs
               (task_id, status, started_at, ended_at, outcome, metadata)
               VALUES (?, 'done', 1, 2, 'completed', ?)""",
            (
                task_id,
                json.dumps({
                    "partial_summary_full": source,
                    "partial_summary_sha256": digest,
                }),
            ),
        )
        assert cursor.lastrowid is not None
        run_id = int(cursor.lastrowid)
        conn.commit()

        recovered = kb.read_run_metadata_field(
            conn,
            task_id=task_id,
            run_id=run_id,
            field="partial_summary_full",
        )
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = 2 WHERE id = ?",
            (task_id,),
        )
        child_task_id = kb.create_task(
            conn,
            title="Child consuming parent handoff",
            parents=[task_id],
        )
        parent_context = kb.build_worker_context(
            conn,
            child_task_id,
            compact=True,
        )
        with pytest.raises(ValueError, match="does not belong"):
            kb.read_run_metadata_field(
                conn,
                task_id=other_task_id,
                run_id=run_id,
                field="partial_summary_full",
            )
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (
                json.dumps({
                    "partial_summary_full": source,
                    "partial_summary_sha256": "0" * 64,
                }),
                run_id,
            ),
        )
        conn.commit()
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            kb.read_run_metadata_field(
                conn,
                task_id=task_id,
                run_id=run_id,
                field="partial_summary_full",
            )

    assert recovered == source
    assert recovered.endswith("SOURCE_TAIL_SENTINEL")
    assert (
        f"hermes kanban context {task_id} --run-id {run_id} "
        "--field partial_summary_full"
    ) in parent_context


def test_budget_handoff_limit_is_utf8_bytes_not_codepoints() -> None:
    for payload in ("x" * 9000, "é" * 9000, "🧭" * 9000):
        handoff = kb._bounded_partial_handoff(payload)

        assert handoff is not None
        assert handoff.startswith("[partial_unverified]\n")
        assert "[section truncated]" in handoff
        assert len(handoff.encode("utf-8")) <= kb._KANBAN_PARTIAL_SUMMARY_MAX_BYTES


def test_budget_handoff_preserves_every_required_section_under_truncation() -> None:
    raw = "\n".join([
        "COMPLETED: " + ("discovery " * 2000),
        "CHANGED_FILES: parser.py",
        "TESTS: focused regression passed",
        "REMAINING: independent review",
        "RESUME: run the reviewer on the frozen postimage",
        "INVARIANTS: no production activation",
    ])

    handoff = kb._bounded_partial_handoff(raw, run_id=42)

    assert handoff is not None
    assert len(handoff.encode("utf-8")) <= kb._KANBAN_PARTIAL_SUMMARY_MAX_BYTES
    for section in (
        "COMPLETED:",
        "CHANGED_FILES:",
        "TESTS:",
        "REMAINING:",
        "RESUME:",
        "INVARIANTS:",
    ):
        assert section in handoff
    assert "FULL_HANDOFF_REF: task_runs:42.metadata.partial_summary_full" in handoff
    assert "FULL_HANDOFF_SHA256:" in handoff


def test_triage_once_replaces_first_blind_retry_then_blocks_repeat(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Oversized lifecycle card",
            assignee="worker",
            max_retries=5,
        )
        assert kb.claim_task(conn, task_id) is not None
        first_status = kb.record_iteration_budget_exhaustion(
            conn,
            task_id,
            expected_run_id=_active_run_id(conn, task_id),
            error="Iteration budget exhausted (60/60)",
            partial_summary="COMPLETED: implementation\nREMAINING: review",
            budget_used=60,
            budget_max=60,
            policy="triage_once",
        )

        first_task = kb.get_task(conn, task_id)
        first_events = kb.list_events(conn, task_id)
        assert first_status == "triage"
        assert first_task is not None and first_task.status == "triage"
        assert len([e for e in first_events if e.kind == "budget_triaged"]) == 1

        # A second exhausted attempt cannot recurse through decomposition
        # forever. Simulate a later re-promoted/reclaimed execution.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()
        assert kb.claim_task(conn, task_id) is not None
        second_status = kb.record_iteration_budget_exhaustion(
            conn,
            task_id,
            expected_run_id=_active_run_id(conn, task_id),
            error="Iteration budget exhausted (60/60)",
            partial_summary="REMAINING: still too broad",
            budget_used=60,
            budget_max=60,
            policy="triage_once",
        )

        second_task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
        assert kb.recompute_ready(conn) == 0
        after_recompute = kb.get_task(conn, task_id)

    assert second_status == "blocked"
    assert second_task is not None and second_task.status == "blocked"
    assert after_recompute is not None and after_recompute.status == "blocked"
    assert len([e for e in events if e.kind == "budget_triaged"]) == 1
    assert any(e.kind == "blocked" for e in events)


def test_budget_exhaustion_retry_policy_preserves_legacy_retry(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Retry-compatible card",
            assignee="worker",
            max_retries=2,
        )
        assert kb.claim_task(conn, task_id) is not None
        status = kb.record_iteration_budget_exhaustion(
            conn,
            task_id,
            expected_run_id=_active_run_id(conn, task_id),
            error="Iteration budget exhausted (60/60)",
            partial_summary="REMAINING: one atomic check",
            budget_used=60,
            budget_max=60,
            policy="retry",
        )
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert status == "ready"
    assert task is not None and task.status == "ready"
    timed_out = [event for event in events if event.kind == "timed_out"]
    assert len(timed_out) == 1
    assert timed_out[0].payload["budget_used"] == 60
    assert timed_out[0].payload["budget_max"] == 60
    assert timed_out[0].payload["budget_exhaustion_policy"] == "retry"


def test_budget_exhaustion_block_policy_stops_first_retry(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Fail-closed card",
            assignee="worker",
            max_retries=10,
        )
        assert kb.claim_task(conn, task_id) is not None
        status = kb.record_iteration_budget_exhaustion(
            conn,
            task_id,
            expected_run_id=_active_run_id(conn, task_id),
            error="Iteration budget exhausted (60/60)",
            partial_summary="REMAINING: operator decision",
            budget_used=60,
            budget_max=60,
            policy="block",
        )
        task = kb.get_task(conn, task_id)
        assert kb.recompute_ready(conn) == 0
        after_recompute = kb.get_task(conn, task_id)

    assert status == "blocked"
    assert task is not None and task.status == "blocked"
    assert after_recompute is not None and after_recompute.status == "blocked"


def test_budget_finalizer_is_idempotent_after_worker_already_blocked(
    kanban_home,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Worker-blocked before finalizer",
            assignee="worker",
            max_retries=2,
        )
        claim = kb.claim_task(conn, task_id)
        assert claim is not None
        worker_run_id = _active_run_id(conn, task_id)
        assert kb.block_task(
            conn,
            task_id,
            reason="review required",
            kind="needs_input",
        )
        before = kb.get_task(conn, task_id)
        before_events = kb.list_events(conn, task_id)
        before_runs = kb.list_runs(conn, task_id)

        status = kb.record_iteration_budget_exhaustion(
            conn,
            task_id,
            expected_run_id=worker_run_id,
            error="Iteration budget exhausted (60/60)",
            partial_summary="late duplicate finalizer summary",
            budget_used=60,
            budget_max=60,
            policy="retry",
        )
        after = kb.get_task(conn, task_id)
        after_events = kb.list_events(conn, task_id)
        after_runs = kb.list_runs(conn, task_id)

    assert status == "blocked"
    assert before is not None and after is not None
    assert after.consecutive_failures == before.consecutive_failures
    assert len(after_events) == len(before_events)
    assert len(after_runs) == len(before_runs) == 1
    assert after_runs[0].summary == before_runs[0].summary


def test_stale_worker_cannot_checkpoint_or_finalize_successor_run(
    kanban_home,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Generation-scoped finalizer",
            assignee="worker",
            max_retries=3,
        )
        assert kb.claim_task(conn, task_id) is not None
        first = kb.get_task(conn, task_id)
        assert first is not None and first.current_run_id is not None
        stale_run_id = first.current_run_id

        with kb.write_txn(conn):
            assert (
                kb._end_run(
                    conn,
                    task_id,
                    outcome="reclaimed",
                    status="reclaimed",
                )
                == stale_run_id
            )
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL WHERE id = ?",
                (task_id,),
            )

        assert kb.claim_task(conn, task_id) is not None
        successor = kb.get_task(conn, task_id)
        assert successor is not None and successor.current_run_id is not None
        successor_run_id = successor.current_run_id
        assert successor_run_id != stale_run_id
        events_before = kb.list_events(conn, task_id)
        runs_before = kb.list_runs(conn, task_id)

        assert (
            kb.record_iteration_budget_checkpoint(
                conn,
                task_id,
                expected_run_id=stale_run_id,
                budget_used=45,
                budget_max=60,
            )
            is False
        )
        assert (
            kb.record_iteration_budget_exhaustion(
                conn,
                task_id,
                expected_run_id=stale_run_id,
                error="stale run exhausted",
                partial_summary="REMAINING: belongs to stale run",
                budget_used=60,
                budget_max=60,
                policy="block",
            )
            == "stale_run"
        )

        after = kb.get_task(conn, task_id)
        events_after = kb.list_events(conn, task_id)
        runs_after = kb.list_runs(conn, task_id)

    assert after is not None
    assert after.status == "running"
    assert after.current_run_id == successor_run_id
    assert after.consecutive_failures == 0
    assert len(events_after) == len(events_before)
    assert len(runs_after) == len(runs_before)
    successor_run = next(run for run in runs_after if run.id == successor_run_id)
    assert successor_run.ended_at is None
    assert successor_run.summary is None


def test_budget_failure_guard_is_atomic_against_a_stale_second_finalizer(
    kanban_home,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Concurrent finalizer target",
            assignee="worker",
            max_retries=3,
        )
        assert kb.claim_task(conn, task_id) is not None
        assert (
            kb._record_task_failure(
                conn,
                task_id,
                error="first budget finalizer",
                outcome="timed_out",
                release_claim=True,
                end_run=True,
                require_running=True,
            )
            is False
        )

        # Model a second finalizer that passed an earlier out-of-transaction
        # status check while the first one still owned the running claim.
        assert (
            kb._record_task_failure(
                conn,
                task_id,
                error="stale duplicate finalizer",
                outcome="timed_out",
                release_claim=True,
                end_run=True,
                require_running=True,
            )
            is False
        )
        task = kb.get_task(conn, task_id)
        runs = kb.list_runs(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert task is not None
    assert task.status == "ready"
    assert task.consecutive_failures == 1
    assert len(runs) == 1
    assert len([event for event in events if event.kind == "timed_out"]) == 1


def test_pre_exhaustion_checkpoint_event_is_idempotent_per_run(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Checkpoint marker target",
            assignee="worker",
        )
        assert kb.claim_task(conn, task_id) is not None
        run_id = _active_run_id(conn, task_id)
        assert (
            kb.record_iteration_budget_checkpoint(
                conn,
                task_id,
                expected_run_id=run_id,
                budget_used=45,
                budget_max=60,
            )
            is True
        )
        assert (
            kb.record_iteration_budget_checkpoint(
                conn,
                task_id,
                expected_run_id=run_id,
                budget_used=46,
                budget_max=60,
            )
            is False
        )
        events = kb.list_events(conn, task_id)

    checkpoints = [event for event in events if event.kind == "budget_checkpoint"]
    assert len(checkpoints) == 1
    assert checkpoints[0].payload == {
        "budget_used": 45,
        "budget_max": 60,
        "remaining": 15,
    }
