"""Bedrock Converse / ConverseStream fault semantics through the real ``hermes chat -q`` CLI.

Each scenario owns a loopback Bedrock fake (``tests/fakes/providers/bedrock_converse.py``) that the real
boto3 client reaches via ``AWS_ENDPOINT_URL_BEDROCK_RUNTIME``; faults are the service's documented ones
in the AWS JSON error shape (``x-amzn-ErrorType`` + ``{"message"}``) or ``:message-type exception``
event-stream frames, plus connection drops. Assertions: requests counted at the fake, the retried
request's body, ``state.db`` rows (no duplicated assistant content) and what the CLI prints.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("botocore")

from tests.e2e.core.providers._native_helpers import (  # noqa: E402
    ChatResult, KnownSymptom, NativeHome, assert_no_duplicate_assistant_text, make_home, messages, run_chat,
)
from tests.fakes.providers.bedrock_converse import (  # noqa: E402
    ACCESS_KEY, REGION, SECRET_KEY, Drop, FakeBedrock, HttpError, Reasoning, Reply, StreamException, Text,
    Turn, seq,
)

MODEL = "deepseek.v3-v1:0"
FINAL = "Recovered answer ZEBRA-7731 is complete and whole."
REASONING = "Reasoning for the recovered answer, streamed in several deltas."
VALIDATION_MARK = "FAKE-VALIDATION-9921"
IAM_DENIAL = ("User: arn:aws:iam::123456789012:user/e2e is not authorized to perform: "
              "bedrock:InvokeModelWithResponseStream on resource: arn:aws:bedrock:us-east-1::foundation-model/"
              + MODEL)

KNOWN = {
    "validation_retried": "#121294 a Bedrock 400 ValidationException is retried and reported as 'temporarily unavailable'",
    "eof_before_message_stop": "#109988 a ConverseStream that ends before messageStop is accepted as the answer",
}


CHUNK = 7
# Event index 3 text deltas into the answer: messageStart, reasoning deltas, signature, contentBlockStop.
CUT_MID_TEXT = 1 + -(-len(REASONING) // CHUNK) + 2 + 3


def _answer() -> Turn:
    return Turn([Reasoning(REASONING), Text(FINAL, chunk=CHUNK)])


@dataclass
class Scenario:
    replies: tuple[Reply, ...]
    env: dict[str, str] = field(default_factory=dict)
    by_op: dict[str, tuple[Reply, ...]] = field(default_factory=dict)


# Each fault precedes a good answer: a retryable fault must be retried into it, a terminal one must not.
SCENARIOS: dict[str, Scenario] = {
    # botocore's own retries are off (documented AWS_MAX_ATTEMPTS) so the 429 reaches Hermes' loop.
    "throttle_http": Scenario((HttpError("ThrottlingException", "Too many requests, please wait before trying again."),
                               _answer()), env={"AWS_MAX_ATTEMPTS": "1"}),
    "unavailable_http": Scenario((HttpError("ServiceUnavailableException", "Bedrock is unable to process your request."),
                                  _answer()), env={"AWS_MAX_ATTEMPTS": "1"}),
    "throttle_in_stream": Scenario((StreamException(_answer(), "throttlingException",
                                                    "Too many tokens, please wait before trying again.", after=CUT_MID_TEXT),
                                    _answer())),
    "drop_mid_stream": Scenario((Drop(_answer(), after=CUT_MID_TEXT), _answer())),
    "eof_before_message_stop": Scenario((Drop(_answer(), after=CUT_MID_TEXT, clean=True), _answer())),
    "validation": Scenario((HttpError("ValidationException", f"The model returned the following errors: "
                                      f"malformed input request: {VALIDATION_MARK}"), _answer())),
    "stream_denied_falls_back": Scenario((), by_op={
        "ConverseStream": (HttpError("AccessDeniedException", IAM_DENIAL),), "Converse": (_answer(),)}),
}


def _responder(sc: Scenario):
    if not sc.by_op:
        return seq(*sc.replies)
    per_op = {op: seq(*replies) for op, replies in sc.by_op.items()}
    return lambda rec: per_op[rec["op"]](rec)


def _run(name: str, root: Path) -> dict[str, Any]:
    sc = SCENARIOS[name]
    fake = FakeBedrock(_responder(sc))
    with fake:
        nh = make_home(root, {"provider": "bedrock", "default": MODEL, "context_length": 64000},
                       env_file={"AWS_ACCESS_KEY_ID": ACCESS_KEY, "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
                                 "AWS_REGION": REGION})
        result = run_chat(nh, "Give me the recovered answer.",
                          env={"AWS_ENDPOINT_URL_BEDROCK_RUNTIME": fake.endpoint, **sc.env})
        return {"nh": nh, "result": result, "requests": fake.snapshot()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    base = tmp_path_factory.mktemp("bedrock_faults")
    with ThreadPoolExecutor(len(SCENARIOS)) as pool:
        futures = {name: pool.submit(_run, name, base / name) for name in SCENARIOS}
        return {name: fut.result() for name, fut in futures.items()}


def _assistant_rows(nh: NativeHome) -> list[dict[str, Any]]:
    return [r for r in messages(nh) if r["role"] == "assistant"]


def _recovered(run: dict[str, Any], expected_requests: int) -> None:
    """Exit 0, the full answer printed once, one assistant row, the retry re-sent the same history."""
    result: ChatResult = run["result"]
    assert result.returncode == 0, result.describe()
    requests = run["requests"]
    assert not [r["rejected"] for r in requests if r.get("rejected")], requests
    assert len(requests) == expected_requests, f"fake saw {len(requests)} requests\n{result.describe()}"
    assert all(r["body"]["messages"] == requests[0]["body"]["messages"] for r in requests), \
        "the retry did not re-send the original history (partial output leaked into the request?)"
    assert result.stdout.count(FINAL) == 1, result.describe()
    rows = _assistant_rows(run["nh"])
    assert_no_duplicate_assistant_text(rows, FINAL[:12])
    assert [r["content"] for r in rows] == [FINAL], rows


@pytest.mark.parametrize("name", ["throttle_http", "unavailable_http", "throttle_in_stream"])
def test_retryable_fault_is_retried_once_into_the_answer(name: str, runs: dict[str, Any]) -> None:
    _recovered(runs[name], expected_requests=2)


def test_mid_stream_drop_retries_without_duplicated_persisted_content(runs: dict[str, Any]) -> None:
    run = runs["drop_mid_stream"]
    _recovered(run, expected_requests=2)
    assert run["requests"][0]["reply"] == "Drop", run["requests"][0]


@pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN["eof_before_message_stop"])
def test_stream_ending_before_message_stop_is_not_accepted(runs: dict[str, Any]) -> None:
    run = runs["eof_before_message_stop"]
    assert run["result"].returncode == 0, run["result"].describe()
    assert run["requests"] and run["requests"][0]["reply"] == "Drop", run["requests"]
    rows = [r["content"] for r in _assistant_rows(run["nh"])]
    sent = len(run["requests"])
    # Symptom: the truncated first stream is persisted as the final answer and never retried.
    if sent == 1 and rows and FINAL.startswith(rows[-1]) and rows[-1] != FINAL:
        raise KnownSymptom(f"{KNOWN['eof_before_message_stop']}: requests={sent} rows={rows}")
    assert (sent, rows) == (2, [FINAL]), f"requests={sent} rows={rows}"


@pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN["validation_retried"])
def test_validation_exception_is_surfaced_once_without_retry(runs: dict[str, Any]) -> None:
    run = runs["validation"]
    result: ChatResult = run["result"]
    sent = len(run["requests"])
    assert sent >= 1 and run["requests"][0]["reply"] == "HttpError", run["requests"]
    # Symptom: the 400 is retried (the scripted success behind it is reached).
    if sent > 1:
        raise KnownSymptom(f"{KNOWN['validation_retried']}: fake saw {sent} requests")
    shown = result.stdout + result.stderr
    assert result.returncode != 0 and FINAL not in shown, result.describe()
    assert VALIDATION_MARK in shown and "ValidationException" in shown, result.describe()
    assert "temporarily unavailable" not in shown, result.describe()
    assert [r["content"] for r in _assistant_rows(run["nh"])] != [FINAL]


def test_streaming_iam_denial_falls_back_to_converse(runs: dict[str, Any]) -> None:
    run = runs["stream_denied_falls_back"]
    result: ChatResult = run["result"]
    assert result.returncode == 0, result.describe()
    requests = run["requests"]
    assert [r["op"] for r in requests] == ["ConverseStream", "Converse"], [r["op"] for r in requests]
    assert requests[1]["body"]["messages"] == requests[0]["body"]["messages"]
    assert FINAL in result.stdout, result.describe()
    rows = _assistant_rows(run["nh"])
    assert [(r["content"], r["reasoning_content"]) for r in rows] == [(FINAL, REASONING)], rows
