"""PiRPCClient — native pi --mode rpc delegate contract.

Hermetic: exercises the JSONL protocol against a fake pi subprocess, never
the real binary or network. Pins the behaviors the parent relies on:
(1) tool markers + result footer surface on the shim message exactly like
    the pi-acp bridge's, so existing parsing in conversation_loop /
    delegate_tool is shared unchanged;
(2) extension_ui_request questions register as pending and free-text
    answers map per method (input/editor -> text, select -> option match
    or index, confirm -> yes/no), with safe auto-answer fallback;
(3) the pi-rpc provider resolves without any ACP env vars set.
"""

import stat
import sys
import textwrap
import threading
import time

import pytest

from agent.pi_rpc_client import (
    PiRPCClient,
    PendingQuestion,
    answer_oldest_pending_question,
    pending_questions,
)


# ---------------------------------------------------------------- fake pi

FAKE_PI = textwrap.dedent(
    """
    import json, sys
    def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
    send({"type": "ready"})
    prompt_id = None
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("type") == "prompt":
            prompt_id = msg["id"]
            # 1) tool-call markers (reasoning), 2) a question, 3) block
            # here (NOT in the for-loop) until answered, 4) final answer
            # echoing the response payload + footer.
            send({"type": "assistant", "thought":
                  '[pi-tool] bash {"cmd": "ls"}\n'
                  "[pi-tool:ok] bash -> result 12 bytes"})
            send({"type": "extension_ui_request", "id": 9001,
                  "method": "input", "title": "Which DB port?"})
            while True:
                r = json.loads(sys.stdin.readline())
                if r.get("type") == "extension_ui_response":
                    ans = r.get("text")
                    break
            footer = "{\"status\": \"end_turn\", \"duration_s\": 1.5, \"touched_files\": []}"
            send({"type": "assistant",
                  "text": "answered with: %s\n```pi-delegation-result\n%s\n```" % (ans, footer)})
            send({"type": "prompt_done", "id": prompt_id})
    """
)


@pytest.fixture
def fake_pi(tmp_path):
    p = tmp_path / "fake-pi"
    p.write_text("#!%s\n%s" % (sys.executable, FAKE_PI))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.fixture
def clean_registry():
    pending_questions.clear()
    yield
    pending_questions.clear()


def make_client(fake_pi):
    return PiRPCClient(acp_command=fake_pi, base_url="pi://rpc-test")


def create(client, content):
    return client.chat.completions.create(
        model="test-model", messages=[{"role": "user", "content": content}]
    )


# ------------------------------------------------- answer text mapping

@pytest.mark.parametrize(
    "method,steer,expected",
    [
        ("input", "PostgreSQL on 5432", {"text": "PostgreSQL on 5432"}),
        ("editor", "line1\nline2", {"text": "line1\nline2"}),
        ("input", "  trimmed  ", {"text": "trimmed"}),
        ("select", "2", {"value": "Use REST"}),
        ("select", "use grpc", {"value": "Use gRPC"}),
        ("confirm", "yes", {"confirmed": True}),
        ("confirm", "no", {"confirmed": False}),
        ("confirm", "cancel", {"confirmed": False}),
    ],
)
def test_answer_with_maps_per_method(method, steer, expected):
    q = PendingQuestion(method, "q", ["Use gRPC", "Use REST"])
    assert q.answer_with(steer) == expected


def test_select_fallbacks_never_crash():
    # non-matching, non-numeric text -> first option; empty options -> cancel
    q = PendingQuestion("select", "t", ["a", "b"])
    assert q.answer_with("whatever") == {"value": "a"}
    q2 = PendingQuestion("select", "t", [])
    assert q2.answer_with("anything") == {"cancelled": True}


def test_empty_input_is_cancelled_not_empty_text():
    q = PendingQuestion("input", "t", None)
    assert q.answer_with("   ") == {"cancelled": True}


def test_auto_answer_policy():
    assert PendingQuestion("confirm", "t", None).auto_answer() == {"confirmed": True}
    assert PendingQuestion("select", "t", ["a"]).auto_answer() == {"value": "a"}
    assert PendingQuestion("input", "t", None).auto_answer() == {"cancelled": True}


# ---------------------------------------------------------------- registry

def test_answer_oldest_routes_to_oldest(clean_registry):
    old = PendingQuestion("input", "old", None)
    old.created_at -= 10
    new = PendingQuestion("input", "new", None)
    pending_questions[old.id] = old
    pending_questions[new.id] = new
    assert answer_oldest_pending_question("the answer") is True
    assert old.id not in pending_questions
    assert new.id in pending_questions
    assert old.answer == {"text": "the answer"}
    assert old.answered.is_set()


def test_answer_oldest_empty_registry(clean_registry):
    assert answer_oldest_pending_question("x") is False


# ------------------------------------------------------- e2e over fake pi

def test_round_trip_markers_question_footer(fake_pi, clean_registry):
    client = make_client(fake_pi)
    # Background answerer stands in for steer_subagent's routing: as soon
    # as the child's question registers, answer it with free text.
    def answerer():
        for _ in range(200):
            if pending_questions:
                answer_oldest_pending_question("5432")
                return
            time.sleep(0.05)
    t = threading.Thread(target=answerer, daemon=True)
    t.start()
    completion = create(client, "delegate this")
    t.join(timeout=5)
    msg = completion.choices[0].message
    # free-text answer reached the child and is echoed in its final text
    assert "answered with: 5432" in msg.content
    # footer contract intact on the final message
    assert "pi-delegation-result" in msg.content
    # tool markers ride the reasoning field for shared parsing
    assert "[pi-tool:ok] bash" in (msg.reasoning or "")
    client.close()


def test_question_timeout_auto_answers(fake_pi, clean_registry, monkeypatch):
    monkeypatch.setattr("agent.pi_rpc_client._DEFAULT_QUESTION_TIMEOUT", 1.0)
    client = make_client(fake_pi)
    start = time.time()
    completion = create(client, "go")
    elapsed = time.time() - start
    # No steer ever arrives: after ~1s the input question is auto-answered
    # (cancelled policy), the run still completes with a footer.
    assert elapsed < 30
    assert "pi-delegation-result" in completion.choices[0].message.content
    assert pending_questions == {}
    client.close()


# ---------------------------------------------------------------- provider

def test_pi_rpc_provider_resolves_without_acp_env(monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    monkeypatch.delenv("HERMES_COPILOT_ACP_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_COPILOT_ACP_ARGS", raising=False)
    r = resolve_runtime_provider(requested="pi-rpc")
    assert r is not None
