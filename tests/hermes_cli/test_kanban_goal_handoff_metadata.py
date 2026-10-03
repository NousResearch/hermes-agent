"""The real CLI/tool handoffs must expose metadata to the goal judge."""
import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as cli, kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import goals
from tools import kanban_tools as tools


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_SESSION_ID"):
        monkeypatch.delenv(key, raising=False)
    kb.init_db()
    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda name: (object(), "fixture"))

    def create(goal_mode=True):
        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="Verify required acceptance", goal_mode=goal_mode,
                                 assignee="test-worker")
            assert kb.claim_task(conn, tid)
            run_id = kb.get_task(conn, tid).current_run_id
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        return tid
    return create


def handoff(surface, action, tid, metadata, summary="Report artifact verified"):
    if surface == "tool":
        name = "kanban_complete" if action == "complete" else "kanban_request_review"
        return json.loads(tools.registry.dispatch(name, {"task_id": tid, "summary": summary, "metadata": metadata}))
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))
    argv = ["kanban", action, tid, "--summary", summary]
    if metadata is not None:
        argv += ["--metadata", json.dumps(metadata)]
    return cli.kanban_command(parser.parse_args(argv))


@pytest.mark.parametrize("surface", ["cli", "tool"])
@pytest.mark.parametrize("action", ["complete", "request-review"])
def test_metadata_reaches_real_judge_and_rejection_preserves_run(board, monkeypatch, surface, action):
    tid = board()
    seen = []
    # Exercise judge_goal's real prompt construction; only the external LLM is replaced.
    def call_judge(*args, **kwargs):
        prompt = args[2]
        seen.append(prompt)
        verdict = "continue" if "UNVERIFIED" in prompt else "done"
        return json.dumps({"verdict": verdict, "reason": "required verification missing"})
    monkeypatch.setattr(goals, "_call_goal_judge_llm", call_judge)
    metadata = {"evidence_boundary": {"required_image_semantics": "UNVERIFIED"}}
    handoff(surface, action, tid, metadata)
    assert seen and "UNVERIFIED" in seen[0], "handoff metadata disappeared before the judge"
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.latest_run(conn, tid).ended_at is None


@pytest.mark.parametrize("surface", ["cli", "tool"])
@pytest.mark.parametrize("action", ["complete", "request-review"])
@pytest.mark.parametrize("mode", ["success", "empty", "exception", "non-goal"])
def test_handoff_preserves_existing_success_and_fallback(board, monkeypatch, surface, action, mode):
    tid = board(goal_mode=mode != "non-goal")
    seen = []
    def call_judge(*args, **kwargs):
        seen.append(args[2])
        if mode == "exception":
            raise RuntimeError("fixture transport error")
        return json.dumps({"verdict": "done", "reason": "verified"})
    monkeypatch.setattr(goals, "_call_goal_judge_llm", call_judge)
    metadata = {} if mode == "empty" else {"acceptance": {"check": "PASS"}}
    handoff(surface, action, tid, metadata)
    assert bool(seen) is (mode != "non-goal")
    if mode == "success":
        assert "PASS" in seen[0]
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == ("done" if action == "complete" else "review")
        assert kb.latest_run(conn, tid).metadata == (metadata or None)


def test_long_summary_cannot_displace_metadata(monkeypatch):
    seen = []
    def call_judge(*args):
        assert args[3] == 7
        seen.append(args[2])
        return '{"verdict":"continue","reason":"missing verification"}'
    monkeypatch.setattr(goals, "_call_goal_judge_llm", call_judge)
    metadata = {"evidence_boundary": {"测试": "UNVERIFIED"}}
    goals.judge_goal("verify", "summary " * 1000, handoff_metadata=metadata, timeout=7)
    assert json.dumps(metadata, ensure_ascii=False) in seen[0]
    assert metadata == {"evidence_boundary": {"测试": "UNVERIFIED"}}


@pytest.mark.parametrize("surface", ["cli", "tool"])
def test_oversized_metadata_cannot_bypass_gate_on_context_error(board, monkeypatch, surface):
    tid = board()
    calls = []
    def context_error(*args):
        calls.append(args[2])
        raise RuntimeError("model context exhausted")
    monkeypatch.setattr(goals, "_call_goal_judge_llm", context_error)
    handoff(surface, "complete", tid, {"log": "x" * 20000, "required": "UNVERIFIED"})
    assert not calls, "oversized evidence must not cause a fail-open transport error"
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_cli_metadata_only_review_reaches_judge(board, monkeypatch):
    tid = board()
    seen = []
    def done(*args):
        seen.append(args[2])
        return '{"verdict":"done","reason":"checks verified"}'
    monkeypatch.setattr(goals, "_call_goal_judge_llm", done)
    handoff("cli", "request-review", tid, {"check": "PASS"}, summary="")
    assert seen and '"check": "PASS"' in seen[0]
