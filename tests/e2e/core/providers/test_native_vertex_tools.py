"""Vertex AI wire conformance: OAuth minting, multi-turn tools, thought-signature replay after resume.

The real ``hermes chat -q`` CLI talks to Vertex through the standard ``HTTPS_PROXY`` +
``SSL_CERT_FILE`` channel; the fake (``tests/fakes/providers/vertex.py``) terminates TLS for
``us-central1-aiplatform.googleapis.com``, validates every request against the Vertex
OpenAI-compatibility contract, and mints tokens for the REAL ``google-auth`` JWT exchange.

Scenarios run concurrently (one fake + one hermetic home each) in a module fixture:

* ``session``  — turn 1 calls ``read_file`` (signed call), turn 2 runs in a NEW process via
  ``--resume`` and calls it again; ``--reasoning high`` throughout.
* ``refresh``  — tokens are minted with a short ``expires_in``; the vendor expires them right
  after the first response, so the next request 401s and must be retried with a re-minted token.
* ``default_toolset`` — a turn with Hermes' default toolsets (every tool schema goes to Vertex).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("google.auth", reason="Vertex minting needs google-auth (CI installs it)")

from tests.e2e.core.providers._native_helpers import (  # noqa: E402
    ChatResult,
    KnownSymptom,
    NativeHome,
    latest_session,
    make_home,
    messages,
    run_chat,
    tool_calls_of,
)
from tests.fakes.providers.vertex import (  # noqa: E402
    PROJECT,
    REGION,
    SA_EMBEDDED_PROJECT,
    Call,
    FakeVertex,
    Say,
    hermes_setup,
    signatures_on_wire,
)

# key -> "#issue reason"; a strict xfail turns red the moment the bug is fixed.
KNOWN: dict[str, str] = {
    "default_toolset": "#109115 terminal.notify anyOf[boolean, array] is rejected by Vertex's FunctionDeclaration "
                       "translator, so every default-toolset turn 400s",
}

SECRET_1 = "PINEAPPLE-42"
SECRET_2 = "MANGO-77"
FINAL_1 = "Turn one answer: the note says PINEAPPLE-42."
FINAL_2 = "Turn two answer: the second note says MANGO-77."
FINAL_REFRESH = "Answer after the token was re-minted."
FILE_ONLY = ("-t", "file")


class Precondition(RuntimeError):
    """A scenario broke before reaching the property under test (never masquerades as a KNOWN xfail)."""


def require(ok: Any, what: str) -> None:
    if not ok:
        raise Precondition(what)


def _home(tmp: Path, name: str, fake: FakeVertex) -> NativeHome:
    nh = make_home(tmp / name / "h", **hermes_setup(fake))
    (nh.project / "note1.txt").write_text(f"note: {SECRET_1}\n", encoding="utf-8")
    (nh.project / "note2.txt").write_text(f"note: {SECRET_2}\n", encoding="utf-8")
    return nh


def _fake(tmp: Path, name: str, script: list[Any]) -> FakeVertex:
    fake = FakeVertex(tmp / name / "fake", project=PROJECT, region=REGION, sa_project=SA_EMBEDDED_PROJECT, script=script)
    fake.start()
    return fake


def run_session(tmp: Path) -> dict[str, Any]:
    fake = _fake(tmp, "session", [
        Call([("read_file", {"path": "note1.txt"})]), Say(FINAL_1),
        Call([("read_file", {"path": "note2.txt"})]), Say(FINAL_2),
    ])
    nh = _home(tmp, "session", fake)
    args = (*FILE_ONLY, "--reasoning", "high")
    turn1 = run_chat(nh, "Read note1.txt and tell me what it says.", env=fake.child_env(), args=args)
    boundary = len(fake.requests)
    tokens_after_turn1 = len(fake.token_requests)
    sid = latest_session(nh) if turn1.returncode == 0 else None
    turn2 = run_chat(nh, "Now read note2.txt.", env=fake.child_env(), args=args, resume=sid) if sid else None
    return {"fake": fake, "nh": nh, "turn1": turn1, "turn2": turn2, "boundary": boundary,
            "tokens_after_turn1": tokens_after_turn1, "sid": sid}


def run_refresh(tmp: Path) -> dict[str, Any]:
    fake = _fake(tmp, "refresh", [Call([("read_file", {"path": "note1.txt"})], expire_tokens_after=True), Say(FINAL_REFRESH)])
    # Below google-auth's refresh window, so a re-mint yields a NEW token rather than the cached one.
    fake.token_policy.expires_in = 120
    nh = _home(tmp, "refresh", fake)
    turn = run_chat(nh, "Read note1.txt.", env=fake.child_env(), args=FILE_ONLY)
    return {"fake": fake, "nh": nh, "turn": turn}


def run_default_toolset(tmp: Path) -> dict[str, Any]:
    fake = _fake(tmp, "default_toolset", [Say("Default toolset answer.")])
    nh = _home(tmp, "default_toolset", fake)
    return {"fake": fake, "nh": nh, "turn": run_chat(nh, "Say hello.", env=fake.child_env())}


SCENARIOS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "session": run_session, "refresh": run_refresh, "default_toolset": run_default_toolset,
}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Any:
    tmp = tmp_path_factory.mktemp("vertex_tools")
    with ThreadPoolExecutor(max_workers=len(SCENARIOS)) as pool:
        futures = {name: pool.submit(fn, tmp) for name, fn in SCENARIOS.items()}
        out = {name: f.result() for name, f in futures.items()}
    yield out
    for res in out.values():
        res["fake"].stop()


def _ok(turn: ChatResult | None, what: str) -> ChatResult:
    require(turn is not None and turn.returncode == 0, f"{what} failed:\n{turn.describe() if turn else 'not run'}")
    assert turn is not None
    return turn


def _rejections(fake: FakeVertex) -> str:
    return json.dumps([(r["status"], r["rejected"]) for r in fake.rejected()], indent=1)


def test_tool_result_goes_back_paired_to_the_signed_call(results: dict[str, Any]) -> None:
    """Turn 1: the model's ``read_file`` call runs for real and its result goes back as a ``tool``
    message paired to the call id, with the call's thought signature replayed byte-for-byte."""
    res = results["session"]
    fake, nh = res["fake"], res["nh"]
    turn1 = _ok(res["turn1"], "turn 1")
    assert FINAL_1 in turn1.stdout, turn1.describe()
    assert not fake.rejected(), f"Vertex rejected requests:\n{_rejections(fake)}"
    first, second = fake.main_requests()[:2]
    (issued,) = first["tool_calls"]
    msgs = second["body"]["messages"]
    asst = next(m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
    (sent,) = asst["tool_calls"]
    assert sent["id"] == issued["id"] and sent["function"]["name"] == "read_file"
    assert sent["extra_content"] == issued["extra_content"], "thought signature not replayed verbatim"
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == [issued["id"]]
    assert SECRET_1 in tool_msgs[0]["content"], "the real read_file result did not reach the model"
    rows = messages(nh, res["sid"])[:4]  # turn 2 (--resume) appends to the same session afterwards
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "assistant"], rows
    persisted = tool_calls_of(rows[1])
    assert [tc["id"] for tc in persisted] == [issued["id"]]
    assert rows[2]["tool_call_id"] == issued["id"] and rows[3]["content"] == FINAL_1


def test_wire_scheme_bearer_and_single_token_mint(results: dict[str, Any]) -> None:
    """Every request hits the configured project/location path on the regional host with a bearer the
    OAuth endpoint minted from a verified SA JWT; one mint per process, reused across API calls."""
    res = results["session"]
    fake = res["fake"]
    _ok(res["turn1"], "turn 1")
    _ok(res["turn2"], "turn 2")
    assert all(r["claims"] for r in fake.token_requests), fake.token_requests
    assert [c["target"] for c in fake.connects if c["allowed"]], "no request reached Vertex through the proxy"
    per_process = [fake.requests[: res["boundary"]], fake.requests[res["boundary"]:]]
    minted = fake.minted_tokens()
    assert res["tokens_after_turn1"] == 1 and len(minted) == 2, (
        f"expected one token exchange per process, saw {len(fake.token_requests)}: {fake.token_requests}")
    for token, reqs in zip(minted, per_process):
        assert len(reqs) >= 2
        assert {r["auth"] for r in reqs} == {f"Bearer {token}"}, "bearer not the minted token / not reused"
    expected_path = f"/v1beta1/projects/{PROJECT}/locations/{REGION}/endpoints/openapi/chat/completions"
    assert {(r["host"], r["path"]) for r in fake.requests} == {(f"{REGION}-aiplatform.googleapis.com", expected_path)}
    assert {r["body"]["model"] for r in fake.requests} == {"google/gemini-3-flash-preview"}


def test_reasoning_effort_reaches_vertex_as_thinking_config(results: dict[str, Any]) -> None:
    """``--reasoning high`` rides in the documented ``extra_body.google.thinking_config`` wrapper
    (a bare top-level ``google`` key is silently ignored by Vertex) and never alongside
    ``reasoning_effort`` (Vertex allows only one of the two)."""
    res = results["session"]
    _ok(res["turn1"], "turn 1")
    for rec in res["fake"].main_requests():
        body = rec["body"]
        thinking = body.get("extra_body", {}).get("google", {}).get("thinking_config")
        assert thinking and thinking.get("thinking_level") == "high", {k: v for k, v in body.items() if k != "messages"}
        assert "reasoning_effort" not in body


def test_thought_signature_replayed_after_resume_in_new_process(results: dict[str, Any]) -> None:
    """Turn 2 runs in a fresh process: its first request replays turn 1's signed call (same id, same
    signature bytes) from state.db, and Vertex accepts every request of the resumed turn."""
    res = results["session"]
    fake = res["fake"]
    turn2 = _ok(res["turn2"], "turn 2 (--resume)")
    assert FINAL_2 in turn2.stdout, turn2.describe()
    assert not fake.rejected(), f"Vertex rejected requests:\n{_rejections(fake)}"
    turn1_call = fake.main_requests()[0]["tool_calls"][0]
    resumed = fake.requests[res["boundary"]]
    replayed = {tc["id"]: tc for m in resumed["body"]["messages"] if m.get("role") == "assistant"
                for tc in m.get("tool_calls") or []}
    assert turn1_call["id"] in replayed, "turn 1's tool call missing from the resumed request"
    assert replayed[turn1_call["id"]].get("extra_content") == turn1_call["extra_content"]
    final = fake.main_requests()[-1]["body"]
    assert signatures_on_wire(final) == fake.issued_signatures, "resumed turn lost or reordered signatures"
    assert SECRET_2 in json.dumps(final["messages"])


def test_expired_token_is_reminted_and_request_retried(results: dict[str, Any]) -> None:
    """The vendor expires the bearer mid-turn: the 401 UNAUTHENTICATED triggers a fresh JWT exchange
    and the SAME request is retried once with the new token; the user only sees the answer."""
    res = results["refresh"]
    fake, nh = res["fake"], res["nh"]
    turn = _ok(res["turn"], "refresh turn")
    assert FINAL_REFRESH in turn.stdout, turn.describe()
    statuses = [r.get("status") for r in fake.requests]
    assert statuses == [200, 401, 200], statuses
    stale, retried = fake.requests[1], fake.requests[2]
    assert stale["auth"] == fake.requests[0]["auth"] and retried["auth"] != stale["auth"]
    assert retried["auth"] == f"Bearer {fake.minted_tokens()[-1]}", "retry did not use the newest minted token"
    assert retried["body"]["messages"] == stale["body"]["messages"]
    assert "401" not in turn.stdout and "UNAUTHENTICATED" not in turn.stdout
    rows = messages(nh, latest_session(nh))
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "assistant"], rows


@pytest.mark.xfail(strict=True, raises=KnownSymptom, reason=KNOWN["default_toolset"])
def test_default_toolset_schemas_accepted_by_vertex(results: dict[str, Any]) -> None:
    """With Hermes' default toolsets every tool declaration must survive Vertex's translation."""
    res = results["default_toolset"]
    fake = res["fake"]
    require(fake.requests and fake.requests[0]["auth"].startswith("Bearer ya29."), "turn never reached Vertex")
    schema_rejects = [r["rejected"] for r in fake.rejected() if "schema type should be ARRAY" in (r["rejected"] or "")]
    if schema_rejects:
        raise KnownSymptom(f"Vertex rejected the tool declarations: {schema_rejects[0]}")
    assert "Default toolset answer." in res["turn"].stdout, res["turn"].describe()
