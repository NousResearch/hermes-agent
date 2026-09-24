"""Custom / Ollama-style chat-completions endpoints through REAL ``hermes -z`` processes.

* Ollama's chat renderer refuses a payload with no ``user`` message (HTTP 500 ``no user
  query found in messages``). The fake enforces that rule on every main request while
  the session goes through a tool continuation, a stream that drops mid-tool-call and
  is retried, and a ``--resume`` of the result: no request may ever lack the user turn
  (#120828 reports one escaping on a continuation/retry path). Ollama answers that
  refusal with a 500, which Hermes (correctly, it cannot tell) treats as a transient
  outage and keeps retrying under ``agent.auto_recovery_cycles`` for minutes; the test
  turns that recovery off and bounds each turn, and checks the recorded requests FIRST so
  a regression fails in seconds naming the request that lost the user turn.
* A tool call whose ``arguments`` are unrepairable JSON must be repaired or surfaced —
  the model has to learn its call did not run (a tool result naming the failure) or the
  turn has to fail visibly. Silently dropping the call and re-asking the identical
  question while the user is told the turn succeeded is the #119389 shape.
"""

from __future__ import annotations

import sys

import pytest

from tests.e2e.core.providers._openai_helpers import (
    READ_TOOL,
    HarnessError,
    Home,
    bug_assertions,
    chat_messages,
    custom_chat_config,
    db_tool_calls,
    db_messages,
    bounded_turn,
    known_marks,
    oneshot,
)
from tests.fakes.providers.chat_variants import CDropToolCall, CText, CTools, FakeChatVariantServer

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

STRICT_TURN_BUDGET = 60.0

KNOWN: dict[str, str] = {
    "unrepairable_args_surfaced": "#119389 unrepairable tool_call arguments silently dropped, turn reports success",
}


def known(name: str) -> list:
    return known_marks(KNOWN, name)


def test_ollama_strict_endpoint_always_receives_the_user_turn(tmp_path) -> None:
    h = Home(tmp_path)
    script = [
        CTools([(READ_TOOL, {"path": "a.txt"})]),       # tool continuation
        CDropToolCall(READ_TOOL, '{"path": "b.t'),       # stream dies mid tool call -> retry
        CTools([(READ_TOOL, {"path": "b.txt"})]),
        CText("FIRST-DONE"),
        CTools([(READ_TOOL, {"path": "a.txt"})]),       # resumed session continues with a tool
        CText("SECOND-DONE"),
    ]
    overrun: HarnessError | None = None
    with FakeChatVariantServer(script, strict_user_turn=True) as srv:
        cfg = custom_chat_config(srv.base_url, model="qwen3:27b")
        # The strict 500 is retryable to Hermes; no minutes-long auto-recovery on a regression.
        cfg["agent"] = {"auto_recovery_cycles": 0}
        h.write(cfg, dotenv={"OPENAI_API_KEY": "ollama"})
        (h.project / "a.txt").write_text("CANARY-A\n", encoding="utf-8")
        (h.project / "b.txt").write_text("CANARY-B\n", encoding="utf-8")
        try:
            first = bounded_turn(h, "read a.txt then b.txt", STRICT_TURN_BUDGET)
            second = bounded_turn(h, "read a.txt again", STRICT_TURN_BUDGET, resume=first.session_id)
        except HarnessError as exc:
            overrun = exc
        records = srv.main_records()

    no_user = [(i, [m.get("role") for m in chat_messages(r["body"])]) for i, r in enumerate(records)
               if not chat_messages(r["body"], "user")]
    assert no_user == [], f"requests (index, roles) {no_user} carried no user message (Ollama answers 500)"
    if overrun is not None:
        raise overrun
    assert first.proc.returncode == 0 and first.stdout.strip() == "FIRST-DONE", first.describe()
    assert second.proc.returncode == 0 and second.stdout.strip() == "SECOND-DONE", second.describe()
    assert [r["response"] for r in records] == [type(s).__name__ for s in script], "precondition: script consumed"
    # The retried stream must not leave a half-assembled tool call behind.
    persisted = [c["function"]["name"] for c in db_tool_calls(db_messages(h, first.session_id))]
    assert persisted == [READ_TOOL] * 3, persisted


@pytest.mark.parametrize("finish", [pytest.param("stop", marks=known("unrepairable_args_surfaced")),
                                    pytest.param("tool_calls", marks=known("unrepairable_args_surfaced"))])
def test_unrepairable_tool_arguments_are_surfaced_not_dropped(tmp_path, finish) -> None:
    h = Home(tmp_path)
    broken = '{"path": "out.md", "content": "# Title\\n\\nsays "quoted" and then, '  # unterminated, bad quotes
    with FakeChatVariantServer([CTools([("write_file", broken)], finish_reason=finish), CText("DONE")]) as srv:
        h.write(custom_chat_config(srv.base_url), dotenv={"OPENAI_API_KEY": "sk-fake"})
        (h.project / "out.md").write_text("OLD\n", encoding="utf-8")
        run = oneshot(h, "write the checkpoint to out.md")
        mains = srv.main_requests()

    assert len(mains) >= 2 or run.proc.returncode != 0, f"precondition: the turn continued or failed: {run.describe()}"
    told = len(mains) >= 2 and any(
        m.get("role") == "tool" or "write_file" in str(m.get("content") or "")
        for m in mains[1]["messages"][len(mains[0]["messages"]):])
    failed_visibly = run.proc.returncode != 0 or run.usage.get("failed")
    with bug_assertions():
        assert told or failed_visibly, (
            "the unparseable write_file call vanished: the next request never told the model it did not run "
            f"and the turn reported success ({run.stdout.strip()!r}); out.md={(h.project / 'out.md').read_text()!r}")
