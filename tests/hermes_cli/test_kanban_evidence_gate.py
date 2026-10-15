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
