"""Read-only, source-scoped recall of Kanban project progress."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


BOARD = "project-burndown"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with kb.connect(board=BOARD):
        pass
    return home


def _sub(conn, task_id, *, chat_id="project-updates", thread_id="", delivery_metadata=None):
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="discord",
        chat_id=chat_id,
        thread_id=thread_id,
        chat_type="thread" if thread_id else "group",
        user_id="operator-1",
        delivery_metadata=delivery_metadata,
    )


def _task(
    conn,
    title,
    *,
    parents=(),
    status="ready",
    branch_name=None,
    created_by=None,
    idempotency_key=None,
):
    task_id = kb.create_task(
        conn,
        title=title,
        body="Untrusted body must never become a filesystem instruction: /Users/not-trusted.",
        initial_status="running",
        parents=parents,
        branch_name=branch_name,
        created_by=created_by,
        idempotency_key=idempotency_key,
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    return task_id


def _run(
    conn,
    task_id,
    *,
    status,
    outcome=None,
    summary=None,
    metadata=None,
    error=None,
    started_at=100,
    ended_at=None,
):
    if ended_at is None and status != "running":
        ended_at = started_at + 1
    with kb.write_txn(conn):
        conn.execute(
            """
            INSERT INTO task_runs
                (task_id, status, started_at, ended_at, outcome, summary, metadata, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                status,
                started_at,
                ended_at,
                outcome,
                summary,
                json.dumps(metadata) if metadata is not None else None,
                error,
            ),
        )


def _source(
    *,
    chat_id="project-updates",
    thread_id=None,
    reply_to_message_id=None,
):
    from gateway.progress_queries import ProgressSource

    return ProgressSource(
        platform="discord",
        chat_id=chat_id,
        thread_id=thread_id,
        reply_to_message_id=reply_to_message_id,
    )


def test_burndown_question_summarizes_graph_receipts_and_remaining_work(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown")
        done = _task(conn, "Narrow credential exception", parents=[root], status="done")
        failed = _task(conn, "Repair provider exception", parents=[root], status="blocked")
        running = _task(conn, "Validate exception evidence", parents=[root], status="running")
        next_task = _task(conn, "Publish acceptance receipt", parents=[root], status="ready")
        _sub(conn, root)
        _run(
            conn,
            done,
            status="done",
            outcome="completed",
            summary="Closed with local evidence; prose commit deadbeef must not be trusted.",
            metadata={"commit": "a1b2c3d4", "branch": "codex/burndown-repair"},
        )
        _run(conn, failed, status="failed", outcome="gave_up", error="worker gave up")
        _run(conn, running, status="running", summary="Worker is validating receipts")
        kb.add_comment(conn, next_task, "worker", "Need the final acceptance receipt.")

    result = resolve_progress_query(
        "How did the burndown go and what else do we need to do?",
        source=_source(),
        board=BOARD,
    )

    assert result.handled is True
    assert result.reason == "resolved"
    assert "Exception Burndown" in result.response
    assert "1 completed" in result.response
    assert "1 failed attempt" in result.response
    assert "1 blocked" in result.response
    assert "1 running" in result.response
    assert "Publish acceptance receipt" in result.response
    assert "a1b2c3d4" in result.response
    assert "codex/burndown-repair" in result.response
    assert "deadbeef" not in result.response


def test_progress_query_isolated_to_trusted_source_subscription(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root, chat_id="private-audit")

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(chat_id="project-updates"), board=BOARD
    )

    assert result.handled is False
    assert result.reason == "no_match"
    assert result.response == ""
    assert "Exception Burndown" not in result.response


def test_progress_query_uses_explicit_board_not_current_or_other_board(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board="other-board") as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is False
    assert result.reason == "no_match"
    assert result.response == ""
    assert "Exception Burndown" not in result.response


def test_progress_query_summarizes_multiple_matching_roots(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        first = _task(conn, "July Exception Burndown")
        second = _task(conn, "August Exception Burndown")
        _sub(conn, first)
        _sub(conn, second)

    result = resolve_progress_query(
        "How did the exception burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    assert result.reason == "resolved_multiple"
    assert first in result.response
    assert second in result.response
    assert "Please name one" not in result.response


def test_single_subscribed_root_does_not_override_zero_topic_score(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)

    result = resolve_progress_query(
        "How did the release go?", source=_source(), board=BOARD
    )

    assert result.handled is False
    assert result.reason == "no_match"
    assert result.response == ""
    assert "Exception Burndown" not in result.response


def test_generic_how_did_it_go_requires_trusted_linkage(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)

    result = resolve_progress_query("How did it go?", source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "no_match"
    assert result.response == ""


@pytest.mark.parametrize(
    "request_text",
    [
        "Give me an update on the burndown fix remaining failures",
        "Give me an update on the burndown frobnicate remaining failures",
        "Status of ExampleProject deploy production?",
        "Progress on ExampleProject delete credentials?",
    ],
)
def test_status_topic_requires_every_term_to_be_bound_to_the_subscribed_graph(
    kanban_home, request_text
):
    from gateway.progress_queries import is_progress_query, resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "ExampleProject Burndown", status="done")
        _task(conn, "Repair remaining failures", parents=[root], status="done")
        _sub(conn, root)

    assert is_progress_query(request_text) is True
    result = resolve_progress_query(request_text, source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "no_match"
    assert result.response == ""


def test_fully_graph_bound_burndown_topic_is_handled(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "ExampleProject Burndown", status="done")
        _sub(conn, root)

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    assert result.reason == "resolved"
    assert root in result.response


def test_generic_how_did_it_go_resolves_one_trusted_reply_linked_root(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(
            conn,
            "Exception Burndown",
            status="done",
            created_by="specialist-routing",
            idempotency_key="specialist-routing:discord::project-updates::request-42",
        )
        _sub(conn, root, delivery_metadata={"origin_message_id": "request-42"})

    result = resolve_progress_query(
        "How did it go?", source=_source(reply_to_message_id="request-42"), board=BOARD
    )

    assert result.handled is True
    assert result.reason == "resolved"
    assert root in result.response


def test_explicit_task_id_selects_one_root_when_topic_is_ambiguous(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        first = _task(conn, "July Exception Burndown")
        second = _task(conn, "August Exception Burndown", status="done")
        _sub(conn, first)
        _sub(conn, second)

    result = resolve_progress_query(
        f"How did {second} go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    assert result.reason == "resolved"
    assert second in result.response
    assert first not in result.response


def test_non_progress_message_falls_through_without_database_response(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query("What should I cook for dinner?", source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "irrelevant"
    assert result.response == ""


def test_ordinary_what_else_chat_falls_through(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(
        "What else can you help with?", source=_source(), board=BOARD
    )

    assert result.handled is False
    assert result.reason == "irrelevant"


def test_question_shaped_action_request_falls_through_to_specialist_routing(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(
        "Can you start an exception burndown and patch the confirmed failures?",
        source=_source(),
        board=BOARD,
    )

    assert result.handled is False
    assert result.reason == "irrelevant"


def test_mixed_progress_and_action_clause_falls_through_to_specialist_routing(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(
        "How did the burndown go, and patch any remaining failures?",
        source=_source(),
        board=BOARD,
    )

    assert result.handled is False
    assert result.reason == "irrelevant"


def test_mixed_progress_and_direct_delegate_falls_through_to_specialist_routing(
    kanban_home,
):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(
        "How did the burndown go? Delegate the remaining failures.",
        source=_source(),
        board=BOARD,
    )

    assert result.handled is False
    assert result.reason == "irrelevant"


def test_mixed_progress_and_please_resolve_falls_through_to_specialist_routing(
    kanban_home,
):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(
        "How did the burndown go? Please resolve the remaining failures.",
        source=_source(),
        board=BOARD,
    )

    assert result.handled is False
    assert result.reason == "irrelevant"


@pytest.mark.parametrize(
    "request_text",
    [
        "How did the burndown go? I'd like you to remediate the remaining failures.",
        "How did the burndown go? Could you address the remaining failures?",
        "How did the burndown go, and please delegate the remaining failures?",
        "How did the burndown go; resolve the remaining failures.",
    ],
)
def test_mixed_progress_and_engineering_imperative_falls_through_to_specialist_routing(
    kanban_home, request_text
):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(request_text, source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "irrelevant"


def test_indirect_mixed_action_falls_through_to_specialist_routing(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(
        "How did the burndown go, and have the bot patch any remaining failures?",
        source=_source(),
        board=BOARD,
    )

    assert result.handled is False
    assert result.reason == "irrelevant"


@pytest.mark.parametrize(
    "request_text",
    [
        "How did the burndown go? I’d like you to fix the remaining failures.",
        "How did the burndown go? I'd like you to patch the remaining failures.",
        "How did the burndown go? I would like the bot to fix the remaining failures.",
    ],
)
def test_mixed_progress_and_intent_phrase_falls_through_to_specialist_routing(
    kanban_home, request_text
):
    from gateway.progress_queries import resolve_progress_query

    result = resolve_progress_query(request_text, source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "irrelevant"


def test_patch_completion_question_remains_a_progress_query(kanban_home):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query("Is the patch complete?") is True


@pytest.mark.parametrize(
    "request_text",
    [
        "How did the burndown go?",
        "How did the burndown go and what else do we need to do?",
        "Could you update me on the burndown progress?",
        "Can you update me on the burndown status?",
        "Would you update me on the exception burndown progress?",
        "Give me an update on the burndown",
        "What remains on the exception burndown?",
        "What's left on the exception burndown?",
        "Where are we on the exception burndown?",
        "Status of the exception burndown?",
        "Progress on the exception burndown?",
        "Is the patch complete?",
        "How did it go?",
    ],
)
def test_strict_progress_grammar_accepts_complete_read_only_questions(request_text):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query(request_text) is True


@pytest.mark.parametrize(
    "action_clause",
    [
        "fix the remaining failures",
        "delegate the remaining failures",
        "resolve the remaining failures",
        "continue with the remaining failures",
        "resume the remaining work",
        "handle the remaining failures",
        "proceed with the patches",
        "keep working on the failures",
        "frobnicate the remaining failures",
    ],
)
def test_strict_progress_grammar_rejects_any_appended_action_clause(action_clause):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query(f"How did the burndown go? Please {action_clause}.") is False


def test_raw_topic_action_tail_cannot_collapse_to_a_graph_match(kanban_home):
    from gateway.progress_queries import is_progress_query, resolve_progress_query

    request = "Give me an update on the nebula burndown and do the work"
    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Nebula Burndown", status="done")
        _sub(conn, root)

    assert is_progress_query(request) is False
    result = resolve_progress_query(request, source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "irrelevant"
    assert result.response == ""


@pytest.mark.parametrize(
    "request_text",
    [
        "Give me an update on the burndown while you fix the remaining failures",
        "Give me an update on the burndown while we continue the remaining work",
        "Give me an update on the burndown to resolve the remaining failures",
        "Give me an update on the burndown then proceed with the patches",
        "Give me an update on the burndown so we can finish",
        "Give me an update on the burndown I'd like you to fix",
        "What remains to address?",
    ],
)
def test_status_shaped_structural_topic_tails_are_not_progress_queries(
    kanban_home, request_text
):
    from gateway.progress_queries import is_progress_query, resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)

    assert is_progress_query(request_text) is False
    result = resolve_progress_query(request_text, source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "irrelevant"
    assert result.response == ""


@pytest.mark.parametrize(
    "request_text",
    [
        "Give me an update on the burndown please continue",
        "Give me an update on the burndown after you review the failures",
        "Give me an update on the burndown before we continue",
        "Give me an update on the burndown until I fix the failures",
        "Give me an update on the burndown also continue",
        "Give me an update on the burndown you should fix next",
    ],
)
def test_other_unbound_topic_tails_still_defer_to_graph_authority(
    kanban_home, request_text
):
    from gateway.progress_queries import is_progress_query, resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)

    assert is_progress_query(request_text) is True
    result = resolve_progress_query(request_text, source=_source(), board=BOARD)

    assert result.handled is False
    assert result.reason == "no_match"
    assert result.response == ""


@pytest.mark.parametrize(
    "request_text",
    [
        "Give me an update on the ExampleProject Phase 12 burndown",
        "How did ExampleProject Phase 12 go?",
        "Status of ExampleProject PATCH-67-7?",
        "Progress on ExampleProject card t_ab12cd?",
        "Could you update me on the ExampleProject adaptive-admission card t_ab12cd status?",
    ],
)
def test_strict_progress_grammar_accepts_noun_like_project_topics(request_text):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query(request_text) is True


@pytest.mark.parametrize(
    "request_text",
    [
        "How did the burndown go",
        "How did the burndown go and what else do I need to do?",
        "Could you update me about the burndown progress?",
        "Can you update me on the burndown?",
        "Give me an update about the burndown",
        "What is left on the exception burndown?",
        "How is the delegate migration progressing?",
        "Is the patch complete",
    ],
)
def test_strict_progress_grammar_rejects_unlisted_near_misses(request_text):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query(request_text) is False


@pytest.mark.parametrize(
    "request_text",
    [
        "Could you update me on the burndown progress?",
        "Give me an update on the burndown",
    ],
)
def test_status_update_phrasing_remains_a_read_only_progress_query(request_text):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query(request_text) is True


@pytest.mark.parametrize(
    "request_text",
    [
        "Please update the burndown code.",
        "Please update the burndown implementation.",
        "How did it go? Please update the code.",
        "How did it go? Please fix the code.",
    ],
)
def test_update_or_fix_code_requests_remain_actionable(request_text):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query(request_text) is False


@pytest.mark.parametrize(
    "request_text",
    [
        "Is the remediation complete?",
    ],
)
def test_question_shaped_engineering_status_remains_a_progress_query(request_text):
    from gateway.progress_queries import is_progress_query

    assert is_progress_query(request_text) is True


def test_progress_output_is_bounded_and_redacts_secret_and_path_content(kanban_home):
    from gateway.progress_queries import MAX_PROGRESS_RESPONSE_CHARS, resolve_progress_query

    secret = "ultra-secret-value"
    path = "/workspace/example-project/.env"
    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        _run(
            conn,
            root,
            status="done",
            outcome="completed",
            summary=(f"OPENAI_API_KEY={secret}; inspect {path}; " + "detail " * 2_000),
            metadata={"commit": "f00ba412", "branch": "codex/receipt"},
        )
        kb.add_comment(conn, root, "worker", f"TOKEN: {secret} and {path}")

    result = resolve_progress_query("How did the burndown go?", source=_source(), board=BOARD)

    assert result.handled is True
    assert len(result.response) <= MAX_PROGRESS_RESPONSE_CHARS
    assert secret not in result.response
    assert path not in result.response
    assert "f00ba412" in result.response


def test_progress_forces_authoritative_redaction_then_applies_local_egress_defense(
    kanban_home, monkeypatch
):
    import agent.redact as authoritative_redaction
    from gateway.progress_queries import resolve_progress_query

    monkeypatch.setattr(authoritative_redaction, "_REDACT_ENABLED", False)
    probes = {
        "vendor": "ghp_abcdefghijk123456789",
        "bearer": "bare-bearer-secret",
        "userinfo": "uri-password-secret",
        "root_path": "/root/.ssh/id_rsa",
        "opt_path": "/opt/hermes/credentials.json",
    }
    summary = (
        f"GitHub receipt {probes['vendor']}\n"
        f"Bearer {probes['bearer']}\n"
        f"Fetched https://operator:{probes['userinfo']}@example.test/report\n"
        f"Inspected {probes['root_path']} and {probes['opt_path']}\n"
        "Acceptance evidence remains visible."
    )
    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        _run(conn, root, status="done", outcome="completed", summary=summary)

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    for probe in probes.values():
        assert probe not in result.response
    assert "Acceptance evidence remains visible." in result.response


def test_local_egress_defense_redacts_standalone_github_token(monkeypatch):
    import agent.redact as authoritative_redaction
    from gateway.progress_queries import _safe_text

    token = "ghp_abcdefghijk123456789"
    monkeypatch.setattr(authoritative_redaction, "redact_sensitive_text", lambda text, **_: text)

    result = _safe_text(f"GitHub receipt {token} completed.")

    assert token not in result
    assert "completed." in result


def test_local_egress_defense_redacts_private_key_material(monkeypatch):
    import agent.redact as authoritative_redaction
    from gateway.progress_queries import _safe_text

    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "cHJpdmF0ZS1rZXktbWF0ZXJpYWw=\n"
        "-----END PRIVATE KEY-----"
    )
    monkeypatch.setattr(authoritative_redaction, "redact_sensitive_text", lambda text, **_: text)

    result = _safe_text(f"Receipt:\n{private_key}\nValidation complete.")

    assert "cHJpdmF0ZS1rZXktbWF0ZXJpYWw" not in result
    assert "BEGIN PRIVATE KEY" not in result
    assert "Validation complete." in result


def test_structured_branch_rejects_redacted_or_locally_secret_shaped_values(monkeypatch):
    import agent.redact as authoritative_redaction
    from gateway.progress_queries import _safe_branch

    token = "ghp_abcdefghijk123456789"
    assert _safe_branch(f"codex/{token}") is None

    monkeypatch.setattr(
        authoritative_redaction,
        "redact_sensitive_text",
        lambda text, **_: text.replace("safe-branch", "[redacted]"),
    )
    assert _safe_branch("codex/safe-branch") is None

    monkeypatch.setattr(authoritative_redaction, "redact_sensitive_text", lambda text, **_: text)
    assert _safe_branch(token) is None


def test_redactor_error_fails_closed_for_text_and_structured_identifiers(monkeypatch):
    import agent.redact as authoritative_redaction
    from gateway.progress_queries import _safe_branch, _safe_commit, _safe_text

    def raise_redactor_error(*args, **kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr(authoritative_redaction, "redact_sensitive_text", raise_redactor_error)

    assert _safe_text("ordinary receipt") == "[redacted]"
    assert _safe_branch("codex/safe-branch") is None
    assert _safe_commit("deadbeef") is None


def test_progress_redacts_complete_multiline_secret_values_but_keeps_other_prose(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        _run(
            conn,
            root,
            status="done",
            outcome="completed",
            summary=(
                "Authorization: Bearer header-sentinel with trailing secret words\n"
                "X-API-Key: x-header-sentinel plus another secret word\n"
                "Proxy-Authorization: proxy-sentinel with another tail\n"
                "DATABASE_PASSWORD=database-sentinel has an unquoted multiword tail\n"
                "Unrelated acceptance evidence remains visible."
            ),
        )

    result = resolve_progress_query("How did the burndown go?", source=_source(), board=BOARD)

    assert result.handled is True
    assert "header-sentinel" not in result.response
    assert "trailing secret words" not in result.response
    assert "x-header-sentinel" not in result.response
    assert "another secret word" not in result.response
    assert "proxy-sentinel" not in result.response
    assert "another tail" not in result.response
    assert "database-sentinel" not in result.response
    assert "unquoted multiword tail" not in result.response
    assert "Unrelated acceptance evidence remains visible." in result.response


def test_progress_redacts_inline_multiword_secret_values_and_preserves_prefix_prose(
    kanban_home,
):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        _run(
            conn,
            root,
            status="done",
            outcome="completed",
            summary=(
                "Worker note: Authorization: Bearer header-sentinel trailing-secret\n"
                "Deployment note: DATABASE_PASSWORD=database sentinel trailing secret"
            ),
        )

    result = resolve_progress_query("How did the burndown go?", source=_source(), board=BOARD)

    assert result.handled is True
    assert "Worker note:" in result.response
    assert "Deployment note:" in result.response
    assert "header-sentinel" not in result.response
    assert "trailing-secret" not in result.response
    assert "database sentinel trailing secret" not in result.response


def test_progress_redacts_aws_private_key_and_database_url_credentials(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    secret_values = (
        "aws-secret-sentinel trailing words",
        "aws-access-sentinel",
        "private-key-sentinel trailing words",
        "postgres://operator:db-password-sentinel@localhost/trading",
        "connection-sentinel trailing words",
    )
    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        _run(
            conn,
            root,
            status="done",
            outcome="completed",
            summary=(
                "AWS_SECRET_ACCESS_KEY=aws-secret-sentinel trailing words\n"
                "Worker note: AWS_ACCESS_KEY_ID: aws-access-sentinel\n"
                "Deploy note: PRIVATE_KEY=private-key-sentinel trailing words\n"
                "DATABASE_URL=postgres://operator:db-password-sentinel@localhost/trading\n"
                "Worker note: CONNECTION_STRING=connection-sentinel trailing words\n"
                "Unrelated acceptance evidence remains visible."
            ),
        )

    result = resolve_progress_query("How did the burndown go?", source=_source(), board=BOARD)

    assert result.handled is True
    for secret in secret_values:
        assert secret not in result.response
    assert "Unrelated acceptance evidence remains visible." in result.response


def test_progress_reads_existing_board_without_using_mutating_connection(kanban_home, monkeypatch):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)

    def reject_mutating_connect(*args, **kwargs):
        raise AssertionError("progress lookup must not call kanban_db.connect")

    monkeypatch.setattr(kb, "connect", reject_mutating_connect)
    result = resolve_progress_query("How did the burndown go?", source=_source(), board=BOARD)

    assert result.handled is True
    assert result.reason == "resolved"
    assert root in result.response


def test_progress_snapshot_includes_committed_wal_without_touching_source_sidecars(
    kanban_home,
):
    from gateway.progress_queries import resolve_progress_query

    source_path = kb.kanban_db_path(board=BOARD)
    conn = kb.connect(board=BOARD)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        wal_path = Path(str(source_path) + "-wal")
        assert wal_path.is_file()
        assert wal_path.stat().st_size > 0

        tracked_paths = [source_path, wal_path, Path(str(source_path) + "-shm")]

        def signatures():
            return {
                path: (
                    path.exists(),
                    path.stat().st_ino if path.exists() else None,
                    path.stat().st_size if path.exists() else None,
                    path.stat().st_mtime_ns if path.exists() else None,
                )
                for path in tracked_paths
            }

        before = signatures()
        result = resolve_progress_query(
            "How did the burndown go?", source=_source(), board=BOARD
        )

        assert result.handled is True
        assert result.reason == "resolved"
        assert root in result.response
        assert signatures() == before
    finally:
        conn.close()


def test_progress_snapshot_race_fails_unavailable(kanban_home, monkeypatch):
    import gateway.progress_queries as progress_queries

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)

    real_copy = progress_queries._copy_snapshot_file

    def racing_copy(source, destination, expected):
        real_copy(source, destination, expected)
        source_stat = source.stat()
        os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1))

    monkeypatch.setattr(progress_queries, "_copy_snapshot_file", racing_copy)
    result = progress_queries.resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is False
    assert result.reason == "unavailable"


def test_snapshot_copy_reads_only_captured_bytes_and_rejects_append(tmp_path, monkeypatch):
    import gateway.progress_queries as progress_queries

    source = tmp_path / "source.db"
    destination = tmp_path / "snapshot.db"
    captured = b"captured-database-bytes"
    appended = b"-appended-after-read"
    source.write_bytes(captured)
    expected = progress_queries._regular_file_signature(source)
    real_read = os.read
    requested_sizes = []
    did_append = False

    def append_after_first_read(descriptor, byte_count):
        nonlocal did_append
        requested_sizes.append(byte_count)
        data = real_read(descriptor, byte_count)
        if data and not did_append:
            did_append = True
            with source.open("ab") as writer:
                writer.write(appended)
        return data

    monkeypatch.setattr(os, "read", append_after_first_read)

    with pytest.raises(OSError, match="changed"):
        progress_queries._copy_snapshot_file(source, destination, expected)

    assert destination.read_bytes() == captured
    assert sum(requested_sizes) == len(captured) + 1


def test_progress_explicit_board_ignores_database_environment_override(
    kanban_home, monkeypatch, tmp_path
):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
    override_path = tmp_path / "override" / "kanban.db"
    with kb.connect(db_path=override_path):
        pass
    monkeypatch.setenv("HERMES_KANBAN_DB", str(override_path))

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    assert result.reason == "resolved"
    assert root in result.response


def test_progress_counts_each_canonical_failed_run_once_and_labels_attempts(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        retrying = _task(conn, "Retry failed patch", parents=[root], status="ready")
        _task(conn, "Running audit", parents=[root], status="running")
        _sub(conn, root)
        for outcome in ("spawn_failed", "failed", "timed_out", "crashed"):
            assert kb.claim_task(conn, retrying, claimer=f"worker:{outcome}") is not None
            assert kb._record_task_failure(
                conn,
                retrying,
                f"{outcome} receipt",
                outcome=outcome,
                failure_limit=99,
                release_claim=True,
                end_run=True,
            ) is False
        assert kb.claim_task(conn, retrying, claimer="worker:gave-up") is not None
        assert kb._record_task_failure(
            conn,
            retrying,
            "gave up receipt",
            outcome="crashed",
            failure_limit=99,
            force_trip=True,
            release_claim=True,
            end_run=True,
        ) is True
        assert [run.outcome for run in kb.list_runs(conn, retrying)] == [
            "spawn_failed",
            "failed",
            "timed_out",
            "crashed",
            "gave_up",
        ]

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert "1 completed, 5 failed attempts, 1 blocked, 1 running" in result.response


@pytest.mark.parametrize("status", ["triage", "scheduled"])
def test_progress_includes_root_only_remaining_work_in_next(kanban_home, status):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status=status)
        _sub(conn, root)

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    assert f"`{root}` Exception Burndown ({status})" in result.response
    assert "Next:" in result.response


def test_progress_uses_newest_receipt_across_entire_graph(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        child = _task(conn, "Acceptance follow-up", parents=[root], status="done")
        _sub(conn, root)
        _run(conn, root, status="done", summary="Older root receipt.", started_at=100)
        _run(conn, child, status="done", summary="Newest graph receipt.", started_at=200)

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert "Newest graph receipt." in result.response
    assert "Older root receipt." not in result.response


def test_progress_discloses_scope_when_graph_exceeds_task_limit(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        _run(conn, root, status="done", summary="Root receipt.", started_at=100)
        kb.add_comment(conn, root, "worker", "Root note.")
        for index in range(24):
            _task(conn, f"Completed child {index}", parents=[root], status="done")

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    assert "first 24 tasks" in result.response
    assert "counts and receipt" in result.response.casefold()
    assert "24 completed" in result.response
    assert "Newest receipt within first 24 tasks" in result.response
    assert "Latest receipt" not in result.response
    assert "Newest note within first 24 tasks" in result.response
    assert "Latest note" not in result.response


def test_truncated_progress_labels_next_list_as_partial(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
        _task(conn, "Ready child", parents=[root], status="ready")
        for index in range(23):
            _task(conn, f"Completed child {index}", parents=[root], status="done")

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board=BOARD
    )

    assert result.handled is True
    assert "Next (partial; first 24 tasks only):" in result.response
    assert "Ready child" in result.response
    assert " Next:" not in result.response


def test_missing_board_is_unavailable_without_creating_files(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    missing_path = kb.kanban_db_path(board="missing-progress-board")
    assert not missing_path.exists()
    assert not missing_path.parent.exists()

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board="missing-progress-board"
    )

    assert result.handled is False
    assert result.reason == "unavailable"
    assert not missing_path.exists()
    assert not missing_path.parent.exists()


def test_symlink_board_database_is_unavailable(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="done")
        _sub(conn, root)
    source_path = kb.kanban_db_path(board=BOARD)
    alias_path = kb.kanban_db_path(board="linked-progress-board")
    alias_path.parent.mkdir(parents=True)
    alias_path.symlink_to(source_path)

    result = resolve_progress_query(
        "How did the burndown go?", source=_source(), board="linked-progress-board"
    )

    assert result.handled is False
    assert result.reason == "unavailable"


def test_progress_reply_includes_a_safe_latest_comment_alongside_run_receipt(kanban_home):
    from gateway.progress_queries import resolve_progress_query

    with kb.connect(board=BOARD) as conn:
        root = _task(conn, "Exception Burndown", status="running")
        _sub(conn, root)
        _run(conn, root, status="running", summary="Worker is checking the failure receipt.")
        kb.add_comment(conn, root, "operator", "Next, compare the remaining failed gate.")

    result = resolve_progress_query("How did the burndown go?", source=_source(), board=BOARD)

    assert result.handled is True
    assert "Worker is checking the failure receipt." in result.response
    assert "Next, compare the remaining failed gate." in result.response
