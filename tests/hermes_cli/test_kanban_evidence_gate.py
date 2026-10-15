"""``kanban.evidence_gate``: config-scoped completion judge, metadata-as-evidence, and
decomposition-only completions turned into dependency waits (never a deadlock).

Every behavioural test runs on BOTH lifecycle surfaces — the CLI (``hermes kanban
complete`` / ``request-review``) and the worker tools (``kanban_complete`` /
``kanban_request_review``) — so the two cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

SURFACES = ("cli", "tool")
VERIFY_BODY = "Fix the thing.\n\nverify: `pytest -q tests/test_thing.py` green\n"
PLAIN_BODY = "Fix the thing. No machine-checkable acceptance line here.\n"


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _set_gate(home: Path, value: str | None) -> None:
    from hermes_cli import config as hconfig

    text = "{}\n" if value is None else f"kanban:\n  evidence_gate: {value}\n"
    (home / "config.yaml").write_text(text)
    # load_config caches by path+mtime; a same-second rewrite must not serve the old dict.
    for name in ("_CONFIG_CACHE", "_config_cache"):
        cache = getattr(hconfig, name, None)
        if isinstance(cache, dict):
            cache.clear()
    invalidate = getattr(hconfig, "invalidate_config_cache", None)
    if callable(invalidate):
        invalidate()


class Judge:
    """Stub for every judge entry point both surfaces use; records what it was shown."""

    def __init__(self, verdict: str = "done", reason: str = "", transport_failed: bool = False):
        self.verdict, self.reason, self.transport_failed = verdict, reason, transport_failed
        self.calls: list[str] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs.get("last_response", args[1] if len(args) > 1 else ""))
        return (self.verdict, self.reason, False, None, self.transport_failed)


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> Judge:
    from agent import auxiliary_client
    from hermes_cli import goals
    from tools import kanban_tools as tools

    stub = Judge()
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client",
                        lambda purpose: (object(), "judge-model"))
    monkeypatch.setattr(goals, "judge_goal", stub)
    monkeypatch.setattr(tools, "judge_goal", stub)
    return stub


def _claimed(monkeypatch: pytest.MonkeyPatch, *, body: str, goal_mode: bool = False) -> str:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Evidence-gated card", body=body, assignee="builder",
                             goal_mode=goal_mode)
        claimed = kb.claim_task(conn, tid, claimer="builder:1")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    return tid


def _complete(surface: str, tid: str, summary: str, metadata: dict | None = None,
              created_cards: list[str] | None = None) -> str:
    if surface == "cli":
        cmd = f"complete {tid} --summary {json.dumps(summary)}"
        if metadata is not None:
            cmd += f" --metadata '{json.dumps(metadata)}'"
        return kc.run_slash(cmd)
    from tools import kanban_tools as tools

    args: dict = {"summary": summary}
    if metadata is not None:
        args["metadata"] = metadata
    if created_cards is not None:
        args["created_cards"] = created_cards
    return tools._handle_complete(args)


def _request_review(surface: str, tid: str, summary: str, metadata: dict | None = None) -> str:
    if surface == "cli":
        cmd = f"request-review {tid} --summary {json.dumps(summary)}"
        if metadata is not None:
            cmd += f" --metadata '{json.dumps(metadata)}'"
        return kc.run_slash(cmd)
    from tools import kanban_tools as tools

    args: dict = {"summary": summary}
    if metadata is not None:
        args["metadata"] = metadata
    return tools._handle_request_review(args)


def _status(tid: str) -> str:
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        return task.status


# --- Part 1: scope is config-driven ------------------------------------------------------------


@pytest.mark.parametrize("surface", SURFACES)
def test_default_scope_does_not_judge_non_goal_card_with_verify_line(home, judge, monkeypatch, surface):
    """Unset ``kanban.evidence_gate`` is exactly upstream: only goal_mode cards are judged."""
    _set_gate(home, None)
    judge.verdict, judge.reason = "continue", "should never be asked"
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    _complete(surface, tid, "did it")
    assert judge.calls == []
    assert _status(tid) == "done"


@pytest.mark.parametrize("surface", SURFACES)
def test_verify_line_scope_judges_and_refuses_with_judge_reason(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "continue", "no pytest output was shown"
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    out = _complete(surface, tid, "did it")
    assert len(judge.calls) == 1
    assert "no pytest output was shown" in out
    assert _status(tid) == "running"


@pytest.mark.parametrize("surface", SURFACES)
def test_verify_line_scope_accepting_judge_completes(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    _complete(surface, tid, "pytest -q: 4 passed")
    assert len(judge.calls) == 1
    assert _status(tid) == "done"


@pytest.mark.parametrize("surface", SURFACES)
def test_verify_line_scope_skips_card_without_verify_line(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "continue", "should never be asked"
    tid = _claimed(monkeypatch, body=PLAIN_BODY)
    _complete(surface, tid, "did it")
    assert judge.calls == []
    assert _status(tid) == "done"
    # Positive control in the same scope: the verify-line card IS judged (and refused).
    gated = _claimed(monkeypatch, body=VERIFY_BODY)
    _complete(surface, gated, "did it")
    assert len(judge.calls) == 1
    assert _status(gated) == "running"


@pytest.mark.parametrize("surface", SURFACES)
def test_verify_line_scope_still_judges_goal_mode_card_without_verify_line(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "continue", "goal not met"
    tid = _claimed(monkeypatch, body=PLAIN_BODY, goal_mode=True)
    out = _complete(surface, tid, "did it")
    assert "goal not met" in out
    assert _status(tid) == "running"


@pytest.mark.parametrize("surface", SURFACES)
def test_all_scope_judges_card_without_verify_line(home, judge, monkeypatch, surface):
    _set_gate(home, "all")
    judge.verdict, judge.reason = "continue", "nothing verifiable"
    tid = _claimed(monkeypatch, body=PLAIN_BODY)
    out = _complete(surface, tid, "did it")
    assert "nothing verifiable" in out
    assert _status(tid) == "running"


@pytest.mark.parametrize("surface", SURFACES)
def test_verify_line_scope_gates_request_review_too(home, judge, monkeypatch, surface):
    """request-review is a terminal worker handoff: it must not bypass the scoped gate."""
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "continue", "no evidence for review"
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    out = _request_review(surface, tid, "ready for review")
    assert "no evidence for review" in out
    assert _status(tid) == "running"


@pytest.mark.parametrize("surface", SURFACES)
def test_judge_transport_failure_allows_completion(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason, judge.transport_failed = "continue", "relay 400", True
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    _complete(surface, tid, "did it")
    assert len(judge.calls) == 1
    assert _status(tid) == "done"


@pytest.mark.parametrize("surface", SURFACES)
def test_missing_judge_client_allows_completion(home, monkeypatch, surface):
    from agent import auxiliary_client

    _set_gate(home, "verify_line")
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda purpose: (None, None))
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    _complete(surface, tid, "did it")
    assert _status(tid) == "done"


@pytest.mark.parametrize(("body", "expected"), [
    ("verify: make test", True),
    ("  VERIFY: make test", True),
    ("intro\n\tVerify:   pytest\n", True),
    ("autonomy: draft\nverify: `pytest`\n", True),
    ("how to verify: read the code", False),       # not at line start
    ("verification: pytest", False),
    ("```\nverify: quoted template\n```\n", False),  # quoted material inside a fence
    ("~~~md\nverify: x\n~~~\nverify: real\n", True),
    ("", False),
    (None, False),
])
def test_verify_line_detection(body, expected):
    from hermes_cli.kanban_evidence_gate import has_verify_line

    assert has_verify_line(body) is expected


def test_unknown_scope_value_falls_back_to_default(home):
    from hermes_cli.kanban_evidence_gate import DEFAULT_SCOPE, configured_scope

    _set_gate(home, "everything-please")
    assert configured_scope() == DEFAULT_SCOPE == "goal_mode"


# --- Part 2: metadata is evidence --------------------------------------------------------------


@pytest.mark.parametrize("surface", SURFACES)
def test_metadata_only_evidence_reaches_judge(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    _complete(surface, tid, "closed, see metadata",
              metadata={"commit": "c8c4b16dfc", "records": ["run-0415.md"]})
    assert len(judge.calls) == 1
    assert "c8c4b16dfc" in judge.calls[0]
    assert "run-0415.md" in judge.calls[0]
    assert _status(tid) == "done"


@pytest.mark.parametrize("surface", SURFACES)
def test_metadata_evidence_reaches_review_judge(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    _request_review(surface, tid, "ready", metadata={"tests_run": "17 passed in 3.2s"})
    assert len(judge.calls) == 1
    assert "17 passed in 3.2s" in judge.calls[0]


def test_metadata_evidence_is_bounded():
    from hermes_cli.kanban_evidence_gate import METADATA_EVIDENCE_LIMIT, evidence_text

    text = evidence_text("summary", {"log": "x" * 50_000})
    assert text.startswith("summary")
    assert len(text) < METADATA_EVIDENCE_LIMIT + 200
    assert evidence_text("  only text  ", None) == "only text"
    assert evidence_text("t", {}) == "t"


# --- Part 3: decomposition-only completion -> dependency wait ----------------------------------


def _decomposed(monkeypatch) -> tuple[str, str, str, str]:
    """T (verify-line card, claimed) created c1 + c2 as its children (parent=T, the
    usual orchestrator shape) and c3, which is already done."""
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    with kbc.connect() as conn:
        c1 = kb.create_task(conn, title="child 1", assignee="builder", parents=[tid],
                            creator_task_id=tid)
        c2 = kb.create_task(conn, title="child 2", assignee="builder", parents=[tid],
                            creator_task_id=tid)
        c3 = kb.create_task(conn, title="child 3 (done)", assignee="builder", creator_task_id=tid)
        assert kb.complete_task(conn, c3, summary="done already")
        assert kb.get_task(conn, c1).status == "todo"  # waits on T: the deadlock shape
    return tid, c1, c2, c3


@pytest.mark.parametrize("surface", SURFACES)
def test_decomposition_only_completion_becomes_dependency_wait(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "continue", "only split into new cards; nothing fixed"
    tid, c1, c2, _c3 = _decomposed(monkeypatch)

    out = _complete(surface, tid, "split into c1 and c2")

    assert "dependency wait" in out and c1 in out and c2 in out
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "todo"
        assert task.block_kind == "dependency"
        assert sorted(kb.parent_ids(conn, tid)) == sorted([c1, c2])  # c3 (done) ignored
        assert c1 not in kb.child_ids(conn, tid) and c2 not in kb.child_ids(conn, tid)
        for child in (c1, c2):
            assert tid not in kb.parent_ids(conn, child)
            assert kb.get_task(conn, child).status == "ready"  # nobody is blocked on T
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "dependency_wait" in kinds and "completed" not in kinds

        # Finishing both children promotes T so the dispatcher re-runs it.
        assert kb.complete_task(conn, c1, summary="fixed 1")
        assert kb.get_task(conn, tid).status == "todo"
        assert kb.complete_task(conn, c2, summary="fixed 2")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "ready"


@pytest.mark.parametrize("surface", SURFACES)
def test_decomposition_with_judged_evidence_completes_normally(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    tid, c1, _c2, _ = _decomposed(monkeypatch)
    _complete(surface, tid, "fixed in abc123; tests 9 passed; follow-ups filed")
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"
        assert kb.parent_ids(conn, tid) == []
        assert tid in kb.parent_ids(conn, c1)


@pytest.mark.parametrize("surface", SURFACES)
def test_decomposition_blocked_verdict_is_a_block_hint_not_a_wait(home, judge, monkeypatch, surface):
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "blocked", "needs a credential nobody has"
    tid, c1, _, _ = _decomposed(monkeypatch)
    out = _complete(surface, tid, "split")
    assert "unachievable" in out and "needs a credential nobody has" in out
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert tid in kb.parent_ids(conn, c1)


@pytest.mark.parametrize("surface", SURFACES)
def test_decomposition_default_scope_is_untouched(home, judge, monkeypatch, surface):
    """goal_mode scope: a non-goal decomposition completes exactly as upstream does."""
    _set_gate(home, None)
    judge.verdict = "continue"
    tid, c1, _, _ = _decomposed(monkeypatch)
    _complete(surface, tid, "split")
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"
        assert tid in kb.parent_ids(conn, c1)
    assert judge.calls == []


@pytest.mark.parametrize("surface", SURFACES)
def test_decomposition_inversion_that_would_cycle_is_refused_untouched(home, judge, monkeypatch, surface):
    """c2 is still a descendant of T through c1 (T -> c1 -> c2); c2 -> T would be a
    cycle, so nothing is inverted and the judge's rejection stands."""
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "continue", "only split"
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    with kbc.connect() as conn:
        c1 = kb.create_task(conn, title="c1", assignee="builder", parents=[tid])  # not created by T
        c2 = kb.create_task(conn, title="c2", assignee="builder", parents=[c1], creator_task_id=tid)
    out = _complete(surface, tid, "split")
    assert "only split" in out
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.parent_ids(conn, tid) == []
        assert kb.parent_ids(conn, c1) == [tid]
        assert kb.parent_ids(conn, c2) == [c1]


def test_tool_created_cards_claim_counts_as_created(home, judge, monkeypatch):
    """A child the worker claims in ``created_cards`` (and actually created) is waited on."""
    _set_gate(home, "verify_line")
    judge.verdict, judge.reason = "continue", "only split"
    tid = _claimed(monkeypatch, body=VERIFY_BODY)
    with kbc.connect() as conn:
        child = kb.create_task(conn, title="claimed child", assignee="builder",
                               created_by="builder", creator_task_id=tid)
    out = _complete("tool", tid, "split", created_cards=[child])
    assert child in out
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "todo"
        assert kb.parent_ids(conn, tid) == [child]


@pytest.mark.parametrize("surface", SURFACES)
def test_default_scope_goal_mode_decomposition_is_refused_as_upstream(home, judge, monkeypatch, surface):
    """Default scope changes nothing upstream: a goal_mode card whose judge says ``continue`` is
    refused even with open created children — no inversion unless the operator opted in."""
    _set_gate(home, None)
    judge.verdict, judge.reason = "continue", "goal not met"
    tid = _claimed(monkeypatch, body=PLAIN_BODY, goal_mode=True)
    with kbc.connect() as conn:
        child = kb.create_task(conn, title="c", assignee="builder", parents=[tid], creator_task_id=tid)
    out = _complete(surface, tid, "split")
    assert "goal not met" in out
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.parent_ids(conn, child) == [tid]
