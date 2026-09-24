"""Copilot ACP wire conformance, part 1: the multi-call tool flow, resume, and the process lifecycle.

``provider: copilot-acp`` makes Hermes spawn an external ACP agent (``copilot --acp --stdio``) per model
call and speak the Agent Client Protocol to it over stdio. Here the agent is
``tests/fakes/providers/copilot_acp.py``, a fake that validates every request against the published
ACP schema and replays scripted turns. Everything on the Hermes side is real: the ``hermes chat -q``
process, runtime resolution from ``config.yaml`` + the profile ``.env`` (``HERMES_COPILOT_ACP_COMMAND``
/ ``HERMES_COPILOT_ACP_ARGS``), the ACP client, the agent loop, the ``read_file`` tool and ``state.db``.

Contract under test (documented in ``agent/copilot_acp_client.py`` and the ACP spec):

* each model call is a fresh agent process: ``initialize`` -> ``session/new`` (absolute cwd) ->
  model selection via the advertised ``model`` config option -> ``session/prompt``; every request is
  schema-valid;
* ACP has no tools channel: Hermes' tools travel in the prompt text, a ``<tool_call>`` block in the
  agent's message runs a REAL Hermes tool, and the result is in the next call's prompt;
* ``--resume`` in a new process runs in a new agent process whose prompt carries the persisted history
  (turn 1 in order, then the new question), with nothing duplicated;
* agent-side ``session/request_permission`` is never granted (Hermes has no human channel there) and
  ``fs/read_text_file`` is confined to the session cwd;
* no agent process outlives the CLI, including one that ignores SIGTERM and stdin EOF.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.e2e.core.providers._native_helpers import (
    ChatResult,
    KnownSymptom,
    NativeHome,
    assert_no_duplicate_assistant_text,
    latest_session,
    make_home,
    messages,
    run_chat,
    session_ids,
    tool_calls_of,
    wait_until,
)
from tests.fakes.providers import copilot_acp as acp

pytest.importorskip("acp.schema", reason="the fake validates against the agent-client-protocol package (acp extra)")

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX launcher script + /proc pid checks"),
    pytest.mark.live_system_guard_bypass,
]

MODELS = ["fake-model-a", "fake-model-b"]
CONFIGURED_MODEL = "fake-model-b"  # NOT the agent's default: selection must happen on the wire
CANARY = "CANARY-7731-acp"
SECRET = "OUTSIDE-CWD-SECRET-5512"
Q1 = "Read canary.txt and tell me what it says (Q1-marker)."
Q2 = "Now summarise what you found (Q2-marker)."
FINAL_ONE = f"The file says {CANARY} (FINAL-ONE)"
FINAL_TWO = "Summary: the canary was read (FINAL-TWO)"
LATE_TEXT = "LATE-ANSWER-65788"
# Compaction: the system prompt + tool bridge alone is ~18K estimated tokens; eight ~1.2K-token file reads
# cross this absolute threshold mid-turn (ACP reports no usage, so Hermes estimates).
COMPACT_THRESHOLD = 21_000
COMPACT_FILES = 8
COMPACT_ASK = "Read f1.txt through f8.txt one by one, then say done (COMPACT-ASK)."
SUMMARY = "SUMMARY-ACP-7f3: files f1..fN were read; each is lorem ipsum filler."
FINAL_COMPACT = "All eight files read (FINAL-COMPACT)."


KNOWN: dict[str, str] = {
    "late_chunk": "#65788 agent_message_chunk emitted after the session/prompt result is dropped",
}


@dataclass
class Scenario:
    nh: NativeHome
    fake: acp.AcpFake
    runs: list[ChatResult] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _home(root: Path, turns: list[list[dict[str, Any]]], **fake_kw: Any) -> Scenario:
    fake = acp.AcpFake(root / "acp", turns, models=MODELS, **fake_kw)
    nh = make_home(root, {"provider": "copilot-acp", "default": CONFIGURED_MODEL}, env_file=fake.env())
    return Scenario(nh, fake)


def _alive(pid: int) -> bool:
    """True while ``pid`` is a live (non-zombie) process."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def _reap(fake: acp.AcpFake) -> None:
    """Teardown: SIGKILL any fake agent this scenario spawned that is still alive (by recorded pid)."""
    for pid in fake.pids():
        if _alive(pid):
            with contextlib.suppress(ProcessLookupError):  # exited between the check and the kill
                os.kill(pid, signal.SIGKILL)


# ── scenarios (independent; run concurrently in the module fixture) ──────────────────────────


def _flow(root: Path) -> Scenario:
    """Turn 1 (two model calls): agent-side permission + a Hermes read_file call, then fs reads + the
    answer. Turn 2: ``--resume`` in a new process. The agent ignores SIGTERM and stdin EOF."""
    project = NativeHome(root).project
    turns = [
        [acp.tool_call("agent-native-1", "run shell", kind="execute"), acp.permission("agent-native-1"),
         acp.message(acp.hermes_tool_call("call_rf_1", "read_file", {"path": str(project / "canary.txt")}))],
        [acp.fs_read(str(project / "canary.txt")), acp.fs_read(str(root / "outside" / "secret.txt")),
         acp.thought("REASONING-ONE: the tool result names the canary"), acp.message(FINAL_ONE)],
        [acp.message(FINAL_TWO)],
    ]
    sc = _home(root, turns, ignore_sigterm=True)
    (sc.nh.project / "canary.txt").write_text(CANARY + "\n", encoding="utf-8")
    (root / "outside").mkdir()
    (root / "outside" / "secret.txt").write_text(SECRET + "\n", encoding="utf-8")
    sc.runs.append(run_chat(sc.nh, Q1))
    if sc.runs[0].returncode == 0 and session_ids(sc.nh):
        sc.extra["turn1_pids"] = sc.fake.pids()
        sc.runs.append(run_chat(sc.nh, Q2, resume=latest_session(sc.nh)))
    return sc


def _late(root: Path) -> Scenario:
    """The agent answers ``session/prompt`` first and streams the message chunk 250 ms later."""
    sc = _home(root, [[acp.result(), acp.message(LATE_TEXT, delay=0.25)], [acp.message("SECOND-TRY-65788")]])
    sc.runs.append(run_chat(sc.nh, "Say the late answer."))
    return sc


def _compaction(root: Path) -> Scenario:
    """Eight read_file calls in one turn cross ``compression.threshold_tokens``; the summarizer runs
    through the same ACP provider (no tool bridge -> the fake answers ``SUMMARY``)."""
    project = NativeHome(root).project
    turns = [[acp.thought(f"step {i}"), acp.message(acp.hermes_tool_call(
        f"call_f{i}", "read_file", {"path": str(project / f"f{i}.txt")}))] for i in range(1, COMPACT_FILES + 1)]
    fake = acp.AcpFake(root / "acp", [*turns, [acp.message(FINAL_COMPACT)]], models=MODELS, aux_text=SUMMARY)
    nh = make_home(root, {"provider": "copilot-acp", "default": CONFIGURED_MODEL}, env_file=fake.env(),
                   extra_config={"compression": {"threshold_tokens": COMPACT_THRESHOLD, "protect_last_n": 4}})
    for i in range(1, COMPACT_FILES + 1):
        (nh.project / f"f{i}.txt").write_text(f"file {i} " + "lorem ipsum dolor " * 250 + "\n", encoding="utf-8")
    sc = Scenario(nh, fake)
    sc.runs.append(run_chat(nh, COMPACT_ASK))
    return sc


SCENARIOS: dict[str, Callable[[Path], Scenario]] = {"flow": _flow, "late": _late, "compaction": _compaction}


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("copilot_acp")
    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as pool:
        futures = {name: pool.submit(fn, base / name) for name, fn in SCENARIOS.items()}
        done = {name: fut.result() for name, fut in futures.items()}
    yield done
    for sc in done.values():
        _reap(sc.fake)


def _flow_ok(outcomes: dict[str, Scenario]) -> Scenario:
    sc = outcomes["flow"]
    assert len(sc.runs) == 2 and all(r.returncode == 0 for r in sc.runs), "\n\n".join(r.describe() for r in sc.runs)
    return sc


def _calls(fake: acp.AcpFake) -> dict[int, list[dict[str, Any]]]:
    """Inbound client requests grouped per agent process (pid), in arrival order."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for rec in fake.inbound():
        if rec["msg"] and rec["msg"].get("method"):
            grouped.setdefault(rec["pid"], []).append(rec)
    return grouped


# ── tests ──────────────────────────────────────────────────────────────────────────────────────


def test_hermes_tool_call_round_trips_through_acp_and_persists(outcomes):
    """A ``<tool_call>`` in the agent's message runs Hermes' real read_file; the result reaches the
    NEXT call's prompt; the CLI prints the answer; state.db pairs the call and result by id."""
    sc = _flow_ok(outcomes)
    assert FINAL_ONE in sc.runs[0].stdout, sc.runs[0].describe()
    prompts = sc.fake.main_prompts()
    assert len(prompts) == 3, f"expected 3 model calls (tool, answer, resumed answer), got {len(prompts)}"
    first, second = acp.prompt_text(prompts[0]), acp.prompt_text(prompts[1])
    assert '"name": "read_file"' in first, "read_file schema was not offered through the prompt tool bridge"
    assert CANARY not in first, "the canary leaked into the prompt before the tool ran"
    assert CANARY in second and second.index(Q1) < second.index(CANARY), (
        "the read_file result must follow the user turn in the next call's prompt")

    rows = messages(sc.nh, latest_session(sc.nh))
    calls = [(r, tc) for r in rows if r["role"] == "assistant" for tc in tool_calls_of(r)]
    assert [(tc["id"], tc["function"]["name"]) for _, tc in calls] == [("call_rf_1", "read_file")], calls
    results = [r for r in rows if r["role"] == "tool"]
    assert [r["tool_call_id"] for r in results] == ["call_rf_1"] and CANARY in results[0]["content"], results
    finals = [r for r in rows if r["role"] == "assistant" and FINAL_ONE in (r["content"] or "")]
    assert len(finals) == 1 and rows.index(finals[0]) > rows.index(results[0]), rows
    assert "<tool_call>" not in "".join(r["content"] or "" for r in rows if r["role"] == "assistant"), (
        "raw tool-call bridge markup was persisted as assistant text")
    assert "REASONING-ONE" in (finals[0].get("reasoning") or ""), "agent_thought_chunk text was not kept as reasoning"


def test_every_request_is_schema_valid_and_selects_the_configured_model(outcomes):
    """Per process: initialize -> session/new (absolute project cwd) -> set_config_option(model) ->
    session/prompt on the issued sessionId; zero requests the fake had to reject."""
    sc = _flow_ok(outcomes)
    assert sc.fake.invalid() == [], f"requests rejected by the ACP schema: {sc.fake.invalid()}"
    grouped = _calls(sc.fake)
    assert len(grouped) == 3, f"one agent process per model call expected, got {len(grouped)}"
    for pid, recs in grouped.items():
        methods = [r["msg"]["method"] for r in recs if r["msg"]["method"] not in ("session/cancel",)]
        assert methods == ["initialize", "session/new", "session/set_config_option", "session/prompt"], (pid, methods)
        init, new, select, prompt = (r["msg"]["params"] for r in recs[:4])
        assert init["protocolVersion"] == acp.PROTOCOL_VERSION
        assert Path(new["cwd"]) == sc.nh.project.resolve(), new
        assert select["value"] == CONFIGURED_MODEL and select["configId"] == "model", select
        assert select["sessionId"] == prompt["sessionId"], (select, prompt)


def test_agent_permission_is_never_granted_and_fs_reads_stay_in_cwd(outcomes):
    """The agent's own permission request is refused; fs/read_text_file inside the session cwd returns
    the file, outside it returns a JSON-RPC error and no content."""
    sc = _flow_ok(outcomes)
    outcomes_seen = [r["msg"]["result"]["outcome"] for r in sc.fake.records() if r.get("kind") == "permission_outcome"]
    assert len(outcomes_seen) == 1, sc.fake.records()
    assert outcomes_seen[0].get("outcome") != "selected", f"Hermes granted an agent-side permission: {outcomes_seen}"
    reads = [r for r in sc.fake.records() if r.get("kind") == "fs_read_result"]
    assert len(reads) == 2 and not any(r["errors"] for r in reads), reads
    inside, outside = reads[0]["msg"], reads[1]["msg"]
    assert CANARY in inside["result"]["content"], inside
    assert "error" in outside and SECRET not in str(outside), f"fs read escaped the session cwd: {outside}"


def test_resume_reaches_the_agent_with_persisted_history_in_order(outcomes):
    """``--resume`` in a new CLI process: a new agent process whose prompt carries turn 1 (question, tool
    result, answer) in order and then the new question; one session, nothing persisted twice. How the
    ACP session is opened (seeded ``session/new`` or ``session/load``) is not part of the contract."""
    sc = _flow_ok(outcomes)
    assert FINAL_TWO in sc.runs[1].stdout, sc.runs[1].describe()
    resumed_pids = [pid for pid in _calls(sc.fake) if pid not in sc.extra["turn1_pids"]]
    assert len(resumed_pids) == 1, "the resumed turn must run in its own agent process"
    text = acp.prompt_text(sc.fake.main_prompts()[-1])
    order = [text.find(s) for s in (Q1, CANARY, FINAL_ONE, Q2)]
    assert -1 not in order and order == sorted(order), f"resumed prompt lost or reordered history: {order}"
    sessions = session_ids(sc.nh)
    rows = messages(sc.nh, sessions[-1])
    assert len(sessions) == 1 and [r["content"] for r in rows if r["role"] == "user"] == [Q1, Q2], rows
    assert_no_duplicate_assistant_text(rows, FINAL_ONE)
    assert_no_duplicate_assistant_text(rows, FINAL_TWO)


def test_no_agent_process_outlives_the_cli(outcomes):
    """Every spawned agent is gone once the CLI exits, even one ignoring SIGTERM and stdin EOF."""
    sc = _flow_ok(outcomes)
    pids = sc.fake.pids()
    wedged = {r["pid"] for r in sc.fake.events("signal")}
    assert len(pids) == 3 and wedged, "vacuity: the scenario must spawn 3 agents that ignored SIGTERM"
    wait_until(lambda: not [p for p in pids if _alive(p)], 10.0, f"agent processes {pids} to exit")


def _transcript(record: dict[str, Any]) -> str:
    text = acp.prompt_text(record)
    return text[text.find("Conversation transcript:"):]


def test_compaction_in_an_acp_session_keeps_the_next_prompt_valid_and_grounded(outcomes):
    """Auto compaction mid-turn: the summary is produced through the ACP agent itself, the next
    main-turn prompt is schema-valid and carries the summary + the user's ask + the protected tail
    (latest tool result) while the summarized tool output is gone; the turn completes once."""
    sc = outcomes["compaction"]
    run = sc.runs[0]
    assert run.returncode == 0 and FINAL_COMPACT in run.stdout, run.describe()
    assert sc.fake.invalid() == [], f"requests rejected by the ACP schema: {sc.fake.invalid()}"
    aux = sc.fake.aux_prompts()
    assert aux, "compaction never called the summarizer through the ACP provider"
    assert "file 2 lorem" in acp.prompt_text(aux[0]), "the summarizer did not receive the history to compact"
    after = [r for r in sc.fake.main_prompts() if r["t"] > aux[0]["t"]]
    assert after, "no main-turn call followed the compaction"
    final = _transcript(after[-1])
    assert SUMMARY in final and COMPACT_ASK in final, "post-compaction prompt lost the summary or the user's ask"
    assert f"file {COMPACT_FILES} lorem" in final, "post-compaction prompt lost the latest tool result"
    assert "file 2 lorem" not in final, "summarized tool output is still resent after compaction"
    rows = messages(sc.nh, latest_session(sc.nh))
    assert_no_duplicate_assistant_text(rows, FINAL_COMPACT)
    assert any(FINAL_COMPACT in (r["content"] or "") for r in rows if r["role"] == "assistant"), rows


@pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN["late_chunk"])
def test_message_chunk_after_prompt_result_reaches_the_user(outcomes):
    sc = outcomes["late"]
    run = sc.runs[0]
    assert run.returncode == 0, run.describe()
    assert sc.fake.invalid() == [], sc.fake.invalid()
    rows = messages(sc.nh, latest_session(sc.nh))
    if LATE_TEXT not in run.stdout:
        raise KnownSymptom(f"late chunk dropped: {len(sc.fake.main_prompts())} model calls, stdout={run.stdout!r}")
    assert_no_duplicate_assistant_text(rows, LATE_TEXT)
    assert len(sc.fake.main_prompts()) == 1, "a delivered late chunk must not trigger an empty-response retry"
