"""Bedrock Converse / ConverseStream wire conformance: tools, signed reasoning, resume, compaction.

Real ``hermes chat -q`` subprocesses with ``model.provider: bedrock`` talk to the real boto3
``bedrock-runtime`` client, redirected by botocore's documented ``AWS_ENDPOINT_URL_BEDROCK_RUNTIME``
override to the loopback fake in ``tests/fakes/providers/bedrock_converse.py``. The fake verifies the
SigV4 signature, validates each body against the botocore service model plus Converse's conversation
rules (tool pairing, signed reasoning replay, final-assistant thinking), and streams real
``application/vnd.amazon.eventstream`` frames. Independent scenarios run concurrently in one module
fixture; each test asserts one property of the wire requests, ``state.db`` rows or CLI output.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("botocore")

from tests.e2e.core._pending_fixes import known_gate  # noqa: E402
from tests.e2e.core.providers._native_helpers import (  # noqa: E402
    ChatResult, KnownSymptom, NativeHome, latest_session, make_home, messages, run_chat, session_ids, tool_calls_of,
)
from tests.fakes.providers.bedrock_converse import (  # noqa: E402
    ACCESS_KEY, REGION, SECRET_KEY, FakeBedrock, Reasoning, Text, ToolUse, Turn, seq,
)

MODEL = "deepseek.v3-v1:0"

# Red on current main for a tracked, open bug: key -> (the bug's own failure-message pattern, reason).
KNOWN: dict[str, tuple[str, str]] = {
    "resume_drops_reasoning": (r"^resumed assistant tool-use turn replayed without signed reasoningContent: ",
                               "#121293 --resume replays Bedrock assistant turns without their signed reasoningContent"),
}

SEED_TEXT = "codeword PELICAN-5501"
SEED2_TEXT = "second codeword OSPREY-7712"
R_A1 = "The user wants both the seed file and an echo; run them in parallel first."
R_A2 = "Both results are back; now read the second seed before answering."
R_A3 = "I have both codewords and the echo, so I can answer now."
FINAL_A = "Answer: PELICAN-5501 / OSPREY-7712 / MARK-42."
R_B1 = "Resume scenario: read the seed file to learn the codeword before replying."
R_B2 = "The seed file holds the codeword, report it back verbatim to the user."
FINAL_B1 = "The codeword is PELICAN-5501."
FINAL_B2 = "Still PELICAN-5501 after the resume."
SUMMARY_TOKEN = "SUMMARY-OK-BEDROCK"
FINAL_C = "Compaction scenario finished: DONE-C."
COMPACTION_FILES = 8


def _home(root: Path, fake: FakeBedrock, extra_config: dict[str, Any] | None = None) -> NativeHome:
    """Bedrock home: fake static creds in the profile .env (the chain Hermes loads), endpoint via env."""
    nh = make_home(root, {"provider": "bedrock", "default": MODEL, "context_length": 64000},
                   env_file={"AWS_ACCESS_KEY_ID": ACCESS_KEY, "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
                             "AWS_REGION": REGION},
                   extra_config=extra_config)
    (nh.project / "seed.txt").write_text(SEED_TEXT + "\n", encoding="utf-8")
    (nh.project / "seed2.txt").write_text(SEED2_TEXT + "\n", encoding="utf-8")
    return nh


def _endpoint_env(fake: FakeBedrock) -> dict[str, str]:
    return {"AWS_ENDPOINT_URL_BEDROCK_RUNTIME": fake.endpoint}


# --------------------------------------------------------------------------------------------------
# Scenarios (each owns its fake + home; run concurrently)
# --------------------------------------------------------------------------------------------------


def _scenario_tools(root: Path) -> dict[str, Any]:
    """Parallel toolUse (read_file + terminal) -> toolResults, a second tool round, then the answer."""
    def first(_rec: dict[str, Any]) -> Turn:
        return Turn([Reasoning(R_A1), ToolUse("read_file", {"path": str(root / "project" / "seed.txt")}),
                     ToolUse("terminal", {"command": "seq 42 42 | sed s/^/MARK-/"})])

    fake = FakeBedrock(seq(first, lambda _r: Turn([Reasoning(R_A2), ToolUse(
        "read_file", {"path": str(root / "project" / "seed2.txt")})]), Turn([Reasoning(R_A3), Text(FINAL_A)])))
    with fake:
        nh = _home(root, fake)
        result = run_chat(nh, "Read seed.txt, echo a marker, then read seed2.txt and report.",
                          env=_endpoint_env(fake))
        return {"fake": fake, "nh": nh, "result": result, "requests": fake.snapshot()}


def _scenario_resume(root: Path) -> dict[str, Any]:
    """Turn 1 (reasoning + toolUse, then reasoning + text) in one process; turn 2 via --resume in another."""
    fake = FakeBedrock(seq(
        lambda _r: Turn([Reasoning(R_B1), ToolUse("read_file", {"path": str(root / "project" / "seed.txt")})]),
        Turn([Reasoning(R_B2), Text(FINAL_B1)]), Turn([Text(FINAL_B2)])))
    with fake:
        nh = _home(root, fake)
        first = run_chat(nh, "What codeword is in seed.txt?", env=_endpoint_env(fake))
        sid = latest_session(nh) if first.returncode == 0 else ""
        second = run_chat(nh, "Say it again.", resume=sid, env=_endpoint_env(fake)) if sid else first
        return {"fake": fake, "nh": nh, "first": first, "second": second, "sid": sid, "requests": fake.snapshot()}


def _compaction_responder(root: Path) -> Callable[[dict[str, Any]], Turn]:
    """Main turns (with toolConfig) read files and report huge input usage; aux calls are summaries."""
    main_calls: list[int] = []

    def respond(rec: dict[str, Any]) -> Turn:
        if "toolConfig" not in rec["body"]:
            return Turn([Text(f"## Goal\nRead the files ({SUMMARY_TOKEN}).\n## Progress\n- files read so far\n")],
                        input_tokens=600)
        main_calls.append(1)
        n = len(main_calls)
        # Small usage first so the transcript is long enough to compact, then far past the threshold.
        usage = 40_000 if n >= 6 else 3_000
        if n <= COMPACTION_FILES:
            return Turn([Reasoning(f"Step {n}: read f{n}.txt next and keep going through the list."),
                         ToolUse("read_file", {"path": str(root / "project" / f"f{n}.txt")})], input_tokens=usage)
        return Turn([Reasoning("Every file has been read; write the final answer."), Text(FINAL_C)],
                    input_tokens=usage)

    return respond


def _scenario_compaction(root: Path) -> dict[str, Any]:
    fake = FakeBedrock(_compaction_responder(root))
    with fake:
        nh = _home(root, fake, extra_config={"compression": {"threshold_tokens": 12_000, "protect_last_n": 4}})
        for i in range(1, COMPACTION_FILES + 1):
            (nh.project / f"f{i}.txt").write_text(f"file {i} " + "lorem ipsum dolor " * 250 + "\n", encoding="utf-8")
        result = run_chat(nh, f"Read f1.txt through f{COMPACTION_FILES}.txt one by one, then say done.",
                          env=_endpoint_env(fake))
        return {"fake": fake, "nh": nh, "result": result, "requests": fake.snapshot()}


SCENARIOS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "tools": _scenario_tools, "resume": _scenario_resume, "compaction": _scenario_compaction,
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    base = tmp_path_factory.mktemp("bedrock_turns")
    with ThreadPoolExecutor(len(SCENARIOS)) as pool:
        futures = {name: pool.submit(fn, base / name) for name, fn in SCENARIOS.items()}
        return {name: fut.result() for name, fut in futures.items()}


# --------------------------------------------------------------------------------------------------
# Assertion helpers
# --------------------------------------------------------------------------------------------------


def _ok(result: ChatResult) -> None:
    assert result.returncode == 0, result.describe()


def _accepted(requests: list[dict[str, Any]]) -> None:
    rejected = [r["rejected"] for r in requests if r.get("rejected")]
    assert not rejected, f"Bedrock (fake) rejected {len(rejected)} request(s): {rejected}"


def _main(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in requests if "toolConfig" in (r.get("body") or {})]


def _results(message: dict[str, Any]) -> dict[str, str]:
    return {b["toolResult"]["toolUseId"]: json.dumps(b["toolResult"]["content"])
            for b in message["content"] if "toolResult" in b}


def _tool_uses(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b["toolUse"] for b in blocks if "toolUse" in b]


def _blob(request: dict[str, Any]) -> str:
    return json.dumps(request["body"].get("messages", []))


# --------------------------------------------------------------------------------------------------
# a. multi-turn with tools
# --------------------------------------------------------------------------------------------------


def test_every_call_is_sigv4_signed_for_bedrock_with_the_profile_credentials(runs: dict[str, Any]) -> None:
    requests = [r for run in runs.values() for r in run["requests"]]
    assert requests, "no request reached the Bedrock fake"
    _accepted(requests)
    for rec in requests:
        assert rec["auth"]["key"] == ACCESS_KEY and rec["auth"]["service"] == "bedrock", rec["headers"]
        assert rec["auth"]["region"] == REGION, rec["auth"]
        assert rec["model"] == MODEL and rec["op"] in ("Converse", "ConverseStream"), rec["path"]


def test_parallel_tool_uses_round_trip_as_tool_results_paired_by_id(runs: dict[str, Any]) -> None:
    run = runs["tools"]
    _ok(run["result"])
    requests = run["requests"]
    _accepted(requests)
    assert [r["op"] for r in requests] == ["ConverseStream"] * 3, [r["op"] for r in requests]
    first_ids = [tu["toolUseId"] for tu in _tool_uses(requests[0]["emitted"])]
    second = requests[1]["body"]["messages"]
    # The assistant turn goes back byte-for-byte: signed reasoning first, then both toolUse blocks.
    assert second[-2] == {"role": "assistant", "content": requests[0]["emitted"]}, second[-2]
    results = _results(second[-1])
    assert list(results) == first_ids, f"toolResult ids {list(results)} != toolUse ids {first_ids}"
    assert SEED_TEXT in results[first_ids[0]] and "MARK-42" in results[first_ids[1]], results
    third = requests[2]["body"]["messages"]
    assert third[:len(second)] == second, "history prefix changed between tool rounds"
    assert third[-2] == {"role": "assistant", "content": requests[1]["emitted"]}, third[-2]
    (second_id,) = [tu["toolUseId"] for tu in _tool_uses(requests[1]["emitted"])]
    assert SEED2_TEXT in _results(third[-1])[second_id], third[-1]


def test_tool_turn_prints_final_answer_and_persists_paired_rows(runs: dict[str, Any]) -> None:
    run = runs["tools"]
    _ok(run["result"])
    assert FINAL_A in run["result"].stdout, run["result"].describe()
    rows = messages(run["nh"])
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "tool", "assistant", "tool", "assistant"], rows
    issued = [[tu["toolUseId"] for tu in _tool_uses(r["emitted"])] for r in run["requests"][:2]]
    assert [tc["id"] for tc in tool_calls_of(rows[1])] == issued[0]
    assert [r["tool_call_id"] for r in rows[2:4]] == issued[0]
    assert [tc["id"] for tc in tool_calls_of(rows[4])] == issued[1] and rows[5]["tool_call_id"] == issued[1][0]
    assert rows[-1]["content"] == FINAL_A


def _reasoning_rows(nh: NativeHome) -> list[str]:
    return [r["reasoning_content"] for r in messages(nh) if r["role"] == "assistant" and r["reasoning_content"]]


def test_persisted_reasoning_equals_the_streamed_reasoning_text(runs: dict[str, Any]) -> None:
    _ok(runs["tools"]["result"])
    persisted = _reasoning_rows(runs["tools"]["nh"])
    assert [p.replace("\n\n", "") for p in persisted] == [R_A1, R_A2, R_A3], persisted
    with known_gate(KNOWN, "reasoning_shredded", raises=KnownSymptom):
        if persisted != [R_A1, R_A2, R_A3]:  # same text once the blank lines are removed: exactly the bug
            raise KnownSymptom(f"persisted reasoning has blank lines between streamed deltas: {persisted}")


# --------------------------------------------------------------------------------------------------
# b. signed reasoning replay across --resume
# --------------------------------------------------------------------------------------------------


def test_resume_in_new_process_replays_tool_history_valid_for_converse(runs: dict[str, Any]) -> None:
    run = runs["resume"]
    _ok(run["first"])
    _ok(run["second"])
    assert FINAL_B2 in run["second"].stdout, run["second"].describe()
    requests = run["requests"]
    _accepted(requests)
    assert len(requests) == 3 and session_ids(run["nh"]) == [run["sid"]], (len(requests), session_ids(run["nh"]))
    resumed = requests[2]["body"]["messages"]
    (tool_use,) = _tool_uses(requests[0]["emitted"])
    assert [m["role"] for m in resumed] == ["user", "assistant", "user", "assistant", "user"], resumed
    assert _tool_uses(resumed[1]["content"]) == [tool_use], resumed[1]
    assert SEED_TEXT in _results(resumed[2])[tool_use["toolUseId"]], resumed[2]
    assert {"text": FINAL_B1} in resumed[3]["content"] and resumed[4]["content"] == [{"text": "Say it again."}]
    rows = messages(run["nh"], run["sid"])
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "assistant", "user", "assistant"], rows
    assert rows[-1]["content"] == FINAL_B2


def test_resume_replays_signed_reasoning_verbatim(runs: dict[str, Any]) -> None:
    run = runs["resume"]
    _ok(run["first"])
    _ok(run["second"])
    requests = run["requests"]
    assert len(requests) == 3, f"expected 2 turn-1 calls + 1 resumed call, fake saw {len(requests)}"
    signed = [b for b in requests[0]["emitted"] if "reasoningContent" in b]
    assert signed, requests[0]["emitted"]
    # In-process the tool-use turn goes back with its signed reasoning ...
    assert requests[1]["body"]["messages"][1]["content"][0] == signed[0]
    # ... and after --resume (new process) it must be identical.
    resumed = requests[2]["body"]["messages"]
    with known_gate(KNOWN, "resume_drops_reasoning", raises=KnownSymptom):
        if not [b for b in resumed[1]["content"] if "reasoningContent" in b]:
            raise KnownSymptom(f"resumed assistant tool-use turn replayed without signed reasoningContent: "
                               f"{resumed[1]['content']}")
    assert resumed[1]["content"][0] == signed[0], resumed[1]["content"]
    assert [b for b in resumed[3]["content"] if "reasoningContent" in b] == [
        b for b in requests[1]["emitted"] if "reasoningContent" in b]


# --------------------------------------------------------------------------------------------------
# c. compaction in a reasoning session
# --------------------------------------------------------------------------------------------------


def test_compaction_in_reasoning_session_keeps_every_request_converse_valid(runs: dict[str, Any]) -> None:
    run = runs["compaction"]
    _ok(run["result"])
    assert FINAL_C in run["result"].stdout, run["result"].describe()
    requests = run["requests"]
    _accepted(requests)  # tool pairs, signatures and final-assistant thinking checked on every request
    summaries = [i for i, r in enumerate(requests) if r["op"] == "Converse" and "toolConfig" not in r["body"]]
    assert summaries, "compaction never called the summarizer over Converse"
    after = [r for r in _main(requests[summaries[0] + 1:])]
    assert after, "no main request after the compaction summary"
    before_len = max(len(r["body"]["messages"]) for r in _main(requests[:summaries[0]]))
    assert all(SUMMARY_TOKEN in _blob(r) for r in after), "a post-compaction request lost the summary"
    assert len(after[0]["body"]["messages"]) < before_len, "compaction did not shrink the wire history"
    last = after[-1]["body"]["messages"]
    # The open tool loop's final assistant turn still leads with the signed reasoning Bedrock issued.
    assert "reasoningContent" in last[-2]["content"][0] and _results(last[-1]), last[-2:]


def test_compaction_persists_summary_and_archives_compacted_rows(runs: dict[str, Any]) -> None:
    run = runs["compaction"]
    _ok(run["result"])
    every = messages(run["nh"], active_only=False)
    live = messages(run["nh"])
    assert any(r["compacted"] == 1 for r in every), "no row archived as compacted"
    assert any(SUMMARY_TOKEN in (r["content"] or "") for r in live), "summary not in the live transcript"
    assert live[-1]["role"] == "assistant" and live[-1]["content"] == FINAL_C, live[-1]
    live_ids = {r["tool_call_id"] for r in live if r["role"] == "tool"}
    issued = {tc["id"] for r in live if r["role"] == "assistant" for tc in tool_calls_of(r)}
    assert live_ids <= issued, f"orphaned live tool rows: {live_ids - issued}"


# --------------------------------------------------------------------------------------------------
# The fake itself rejects what Bedrock rejects (so a green scenario above is not a pass-through)
# --------------------------------------------------------------------------------------------------


def _mutate_request(case: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    msgs = kwargs["messages"]
    table: dict[str, Callable[[], None]] = {
        "orphan_tool_result": lambda: msgs[2]["content"][0]["toolResult"].update(toolUseId="tooluse_nobody"),
        "tampered_signature": lambda: msgs[1]["content"][0]["reasoningContent"]["reasoningText"].update(signature="Zm9yZ2Vk"),
        "unsigned_final_thinking": lambda: msgs[1]["content"].pop(0),
        "two_members_in_union": lambda: msgs[0]["content"][0].update(image={"format": "png", "source": {"bytes": b"x"}}),
        "blank_text": lambda: msgs[0]["content"][0].update(text="   "),
    }
    table[case]()
    return kwargs


@pytest.mark.parametrize("case", ["valid", "orphan_tool_result", "tampered_signature", "unsigned_final_thinking",
                                  "two_members_in_union", "blank_text", "bad_secret"])
def test_fake_rejects_requests_bedrock_rejects(case: str) -> None:
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config
    from botocore.exceptions import ClientError, ParamValidationError

    fake = FakeBedrock(seq(lambda _r: Turn([Reasoning("sig check reasoning"), ToolUse("noop", {})]),
                           Turn([Text("fine")])))
    with fake:
        client = boto3.client(
            "bedrock-runtime", region_name=REGION, endpoint_url=fake.endpoint, aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key="wrong-secret" if case == "bad_secret" else SECRET_KEY,
            config=Config(retries={"max_attempts": 1}, parameter_validation=False))
        base = [{"role": "user", "content": [{"text": "go"}]}]
        first = client.converse(modelId=MODEL, messages=base) if case != "bad_secret" else None
        blocks = first["output"]["message"]["content"] if first else []
        tool_id = next((b["toolUse"]["toolUseId"] for b in blocks if "toolUse" in b), "none")
        kwargs = {"modelId": MODEL, "messages": base + [
            {"role": "assistant", "content": blocks or [{"text": "x"}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": tool_id, "content": [{"text": "ok"}]}}]}]}
        if case not in ("valid", "bad_secret"):
            kwargs = _mutate_request(case, kwargs)
        try:
            client.converse(**kwargs)
        except (ClientError, ParamValidationError) as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
        else:
            code = "OK"
    expected = {"valid": "OK", "bad_secret": "InvalidSignatureException"}.get(case, "ValidationException")
    assert code == expected, (case, code, fake.snapshot()[-1].get("rejected"))
