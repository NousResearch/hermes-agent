"""C3 transcript ledger: what is persisted == what is replayed == what a resume shows, per step.

A real ``AIAgent`` + ``SessionDB`` runs scripted multi-turn conversations against the recording
fake provider. After EVERY step (turn, tool batch, /steer, interrupt, stream drop, /compress,
auto-compaction) the ledger asserts:

(a) exactly-once: no logical message appears twice in the model view that resume replays;
(b) the first request of turn N+1 replays exactly the persisted model view after turn N plus the
    new user message (a compaction step is the only sanctioned rewrite), and the durable view
    starts with what the turn sent (nothing sent to the model is left unpersisted);
(c) durable row counts never shrink, active rows only shrink at compaction, and every message
    the model ever saw is still in the display history exactly once (compaction archives, it
    never drops or duplicates — the kept window survives);
(f) sessions + session_model_usage totals equal what the provider billed;
and at the end of every scenario a FRESH ``hermes chat --resume`` process replays the persisted
history as a byte-identical request prefix (d), and the whole request stream only breaks its
prefix at the declared compaction boundaries (C17).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.history._helpers import (
    NO_BACKGROUND_REVIEW,
    Script,
    big,
    OFFLINE_CONFIG,
    InProcessSession,
    Ledger,
    assert_inputs_shown_once,
    is_summary,
    lineage,
    Spawned,
    assert_replay_equals_persisted,
    assert_usage_matches,
    canon,
    child_env,
    first_divergence,
    integrity_ok,
    prefix_breaks,
    run_oneshot,
    views,
)
from tests.fakes.fake_llm_provider import (
    DropMidStream,
    Error,
    FakeLLMServer,
    Hang,
    Text,
    ToolCall,
    write_hermes_home,
)

# ---------------------------------------------------------------------------------------------
# scripted model


Step = tuple[str, Any]
STEER_TEXT = "also mention the steer marker"
INTERRUPTED_TURN = "this request gets interrupted"
# The interrupted request hangs HANG_S and then drops; a working interrupt ends the turn long before.
INTERRUPT_HANG_S = 60.0
INTERRUPT_DEADLINE_S = 30.0


def bulky(*labels: str) -> list[Step]:
    """Turns whose ANSWERS are large: summarizing them genuinely shrinks the context (the lean tail
    keeps user messages verbatim, so bulk in user text would not compress)."""
    steps: list[Step] = []
    for label in labels:
        steps += [("script", [Text(big(f"answer-{label}"))]), ("turn", f"tell me about {label}")]
    return steps


def _steer_then(tool: ToolCall, text: str, script: Script) -> Callable[[dict], Any]:
    def act(_record: dict) -> Any:
        assert script.session is not None
        assert script.session.agent.steer(text)
        script.steered.append(text)
        return tool
    return act


def _interrupt_during_request(script: Script) -> Callable[[dict], Any]:
    def act(_record: dict) -> Any:
        assert script.session is not None
        import threading

        threading.Thread(target=script.session.agent.interrupt, daemon=True).start()
        return Hang(seconds=INTERRUPT_HANG_S)
    return act


def scenario(name: str, script: Script) -> list[Step]:
    """Steps: ("turn", text) / ("compress", args) / ("script", [actions for the next turn])."""
    s = script
    if name == "plain":
        return [("turn", "hello there"),
                ("script", [Text("thinking out loud answer", reasoning="private chain of thought")]),
                ("turn", "second question"), ("turn", "third question"), ("turn", "fourth question")]
    if name == "tool_batches":
        return [("script", [ToolCall("terminal", {"command": "echo single-1"}), Text("single done")]),
                ("turn", "run one tool"),
                ("script", [ToolCall("terminal", {"command": "echo par-a"},
                                     parallel=[("terminal", {"command": "echo par-b"}),
                                               ("terminal", {"command": "echo par-c"})]),
                            ToolCall("terminal", {"command": "echo chained"}), Text("batch done")]),
                ("turn", "run a parallel batch then one more"),
                ("turn", "and a plain follow-up")]
    if name == "steer":
        return [("turn", "warm up"),
                ("script", [_steer_then(ToolCall("terminal", {"command": "echo steered-tool"}),
                                        STEER_TEXT, s), Text("steer seen")]),
                ("turn", "do a tool while I steer"),
                ("turn", "after the steer")]
    if name == "interrupt":
        return [("turn", "warm up"),
                ("script", [_interrupt_during_request(s)]),
                ("turn", INTERRUPTED_TURN),
                ("turn", "the follow-up after the interrupt"),
                ("turn", "one more")]
    if name == "stream_faults":
        return [("turn", "warm up"),
                ("script", [DropMidStream("a partial reply that the network cut", after_chars=18),
                            Text("recovered after the drop")]),
                ("turn", "stream this and drop"),
                ("script", [Error(500, "scripted outage", retry_after=0.2), Text("recovered after a 500")]),
                ("turn", "this one hits a 500 first"),
                ("turn", "and a clean one")]
    if name == "manual_compress":
        return [*bulky("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"), ("compress", ""),
                ("turn", "first turn past full compress"), ("turn", "next turn past full compress")]
    if name == "compress_here":
        return [*bulky("golf", "hotel", "india", "juliet", "kilo", "lima"), ("compress", "here 1"),
                ("turn", "first turn past compress here"), ("turn", "next turn past compress here")]
    if name == "auto_compaction":
        # The provider reports a prompt over the compaction threshold; the NEXT turn must compact
        # before it is sent, and nothing the user saw may be lost from the durable transcript.
        return [*bulky("mike", "november", "oscar", "papa", "quebec"),
                ("script", [Text("reply under pressure", prompt_tokens=120_000, completion_tokens=9)]),
                ("turn", "this reply reports context pressure"),
                ("turn", "the turn that must compact first"), ("turn", "a turn past auto compaction")]
    if name == "micro_compaction":
        # Opt-in rolling micro-compaction folds the oldest exchange (tool rows included) into a
        # summary after every turn; summarized rows must stay in the durable archive.
        steps: list[Step] = []
        for label in ("romeo", "sierra", "tango", "uniform", "victor", "whiskey"):
            steps += [("script", [ToolCall("terminal", {"command": f"echo tool-{label}"}),
                                  Text(big(f"answer-{label}", 6000))]),
                      ("turn", f"run a tool about {label}")]
        return steps
    raise AssertionError(name)


SCENARIO_CONFIG = {
    "micro_compaction": "compression:\n  micro_compact: true\n  micro_compact_every_n_turns: 1\n",
}

SCENARIOS = ["plain", "tool_batches", "steer", "interrupt", "stream_faults",
             "manual_compress", "compress_here", "auto_compaction", "micro_compaction"]
COMPACTING = {"manual_compress", "compress_here", "auto_compaction", "micro_compaction"}


# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    # The hermetic conftest's per-test HERMES_HOME (a tmp dir): the in-process live-system guard
    # refuses a <HOME>/.hermes layout, and the children must share this very state.db.
    hermes_home = Path(os.environ["HERMES_HOME"])
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workdir)
    script = Script()
    spawned = Spawned()
    with FakeLLMServer(script) as srv:
        write_hermes_home(hermes_home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-e2e")
        yield {"srv": srv, "script": script, "home": home, "hermes_home": hermes_home,
               "workdir": workdir, "spawned": spawned}
        leaked = spawned.reap()
        assert not leaked, f"subprocesses outlived the test: {leaked}"


@pytest.mark.parametrize("name", SCENARIOS)
def test_transcript_ledger(world, name):
    srv, script, hermes_home = world["srv"], world["script"], world["hermes_home"]
    if name in SCENARIO_CONFIG:
        write_hermes_home(hermes_home, srv.base_url,
                          extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + SCENARIO_CONFIG[name])
    typed = [arg for kind, arg in scenario(name, Script()) if kind == "turn"]
    assert not [(a, b) for a in typed for b in typed if a != b and a in b], "scenario inputs must be unique"
    session = InProcessSession(srv.base_url, hermes_home, f"ledger-{name}")
    script.session = session
    ledger = Ledger(srv, hermes_home)
    try:
        for i, (kind, arg) in enumerate(scenario(name, script)):
            label = f"{i}:{kind}:{str(arg)[:24]}"
            if kind == "script":
                script.actions.extend(arg)
            elif kind == "turn":
                ledger.inputs.append(arg)
                t0 = time.monotonic()
                result = session.turn(arg)
                took = time.monotonic() - t0
                script.raise_errors(label)
                ledger.inputs += [x for x in script.steered if x not in ledger.inputs]
                if arg == INTERRUPTED_TURN:
                    assert result.get("interrupted") is True and took < INTERRUPT_DEADLINE_S, (
                        f"{label}: interrupt did not end the hung request (interrupted="
                        f"{result.get('interrupted')!r} after {took:.1f}s)")
                else:
                    assert result.get("final_response") is not None, f"{label}: {result}"
                compaction = name == "micro_compaction" or (
                    name == "auto_compaction" and arg == "the turn that must compact first")
                ledger.step(session.sid, label, compaction=compaction)
            elif kind == "compress":
                res = session.compress(arg)
                flags = {k: res.summary.get(k) for k in ("noop", "aborted", "fallback_used", "refused_would_grow")}
                assert res.status == "compressed" and not any(flags.values()), (
                    f"{label}: /compress {arg!r} did not compact: {res.status} {flags} "
                    f"{res.before_tokens}->{res.after_tokens} tokens")
                ledger.step(session.sid, label, compaction=True, turn=False)
        sid = session.sid
        if name == "steer":
            # The ledger's inputs check then requires the steer row shown exactly once.
            assert script.steered == [STEER_TEXT], f"the steer never landed: {script.steered}"
    finally:
        session.close()
    assert not script.actions, f"scripted responses never consumed: {script.actions}"

    # C17: the request stream only breaks its prefix at declared compaction boundaries, and a
    # one-shot compaction breaks it exactly once.
    main = srv.main_requests()
    breaks = prefix_breaks(main)
    allowed = set(ledger.compaction_request_idx)
    unexpected = [(i, why) for i, why in breaks if i not in allowed]
    assert not unexpected, "prompt-cache prefix broke outside compaction:\n" + "\n".join(
        f"  request {i}: {why}" for i, why in unexpected)
    if name in COMPACTING:
        assert len(breaks) == (len(breaks) if name == "micro_compaction" else 1) and breaks, (
            f"expected the compaction boundary to break the prefix (exactly once for one-shot compaction), "
            f"got {breaks} (declared at {sorted(allowed)})")
        model, _ = views(hermes_home, sid)
        assert any(is_summary(m) for m in model), "compaction left no summary in the model view"

    # (f) usage: what the provider billed == what state.db accounts for the whole lineage.
    assert_usage_matches(hermes_home, lineage(hermes_home, sid), srv.requests, f"scenario {name}")
    integrity_ok(hermes_home)

    # (d) resume in a FRESH `hermes chat --resume` process: its first request replays the persisted
    # model view and is a byte-identical extension of the last in-process request. Messages only:
    # the tools array of an in-process AIAgent is a harness construction choice; the cross-surface
    # tools comparison lives in test_prefix_stability.py (real surfaces only).
    persisted, _ = views(hermes_home, sid)
    n_before = len(main)
    env = child_env(world["home"], hermes_home, world["workdir"])
    _out, resumed_sid = run_oneshot(env, world["workdir"], f"resume check for {name}", world["spawned"], resume=sid)
    assert resumed_sid == sid, f"--resume {sid} continued a different session {resumed_sid}"
    after = srv.main_requests()
    assert len(after) > n_before, "the resumed process sent no request"
    first, last = after[n_before], main[-1]
    assert_replay_equals_persisted(first, persisted, f"fresh-process resume of {name}")
    if name != "micro_compaction":  # a post-turn micro pass legitimately rewrote history after `last`
        for j, (a, b) in enumerate(zip(last["messages"], first["messages"])):
            assert canon(a) == canon(b), (
                f"fresh-process resume changed messages[{j}] ({a.get('role')}): "
                f"{first_divergence(canon(a), canon(b))}")
        assert len(first["messages"]) > len(last["messages"])
    ledger.step(sid, "fresh-process resume", compaction=name == "micro_compaction")
    assert_usage_matches(hermes_home, lineage(hermes_home, sid), srv.requests, f"scenario {name} after resume")


def test_micro_compaction_resumed_display_shows_each_input_once(world):
    srv, script, hermes_home = world["srv"], world["script"], world["hermes_home"]
    write_hermes_home(hermes_home, srv.base_url,
                      extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + SCENARIO_CONFIG["micro_compaction"])
    session = InProcessSession(srv.base_url, hermes_home, "ledger-micro-display")
    inputs: list[str] = []
    try:
        for kind, arg in scenario("micro_compaction", script):
            if kind == "script":
                script.actions.extend(arg)
            else:
                inputs.append(arg)
                session.turn(arg)
        sid = session.sid
    finally:
        session.close()
    model, display = views(hermes_home, sid)
    assert_inputs_shown_once(inputs, display, model, "after six micro-compacted turns")
