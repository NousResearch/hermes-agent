"""Chat-completions reasoning dialects through REAL Hermes processes.

* OpenRouter-style ``reasoning_details`` (signed reasoning blocks) must be replayed
  verbatim on the next call of the turn and on the next turn after a restart — to the
  routes that read them. The route is reached by vendor-host impersonation
  (``base_url: http://openrouter.ai/...`` + ``HTTP_PROXY`` at the fake, see
  ``tests/fakes/providers/chat_variants.py``), because replay is gated on the host.
* The same field must NOT reach an endpoint that does not read it (strict schemas 400
  the whole request), while state.db keeps it for a later switch back.
* DeepSeek-style ``reasoning_content`` is echoed on the assistant tool-call message,
  within the turn and after ``--resume``.
* A long session on a route that accepts replayed reasoning only up to a cumulative
  budget must keep answering instead of wedging on a non-retryable 400 (#118182),
  driven through one long-lived ``tui_gateway`` backend.
"""

from __future__ import annotations

import json
import sys

import pytest

from tests.e2e.core.providers._openai_helpers import (
    READ_TOOL,
    Home,
    chat_messages,
    custom_chat_config,
    db_messages,
    oneshot,
)
from tests.e2e.core.providers._openai_tui import TuiGateway
from tests.fakes.providers.chat_variants import CText, CTools, FakeChatVariantServer

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

KNOWN: dict[str, str] = {
    "reasoning_budget_session_keeps_answering":
        "#118182 replayed reasoning_details grow past a route budget and wedge the session on a 400",
}


def known(name: str) -> list:
    return [pytest.mark.xfail(strict=True, raises=AssertionError, reason=KNOWN[name])] if name in KNOWN else []


def _rd(tag: str, text: str | None = None) -> list[dict]:
    return [{"type": "reasoning.text", "text": text or f"thinking {tag}", "signature": f"sig-{tag}",
             "format": "anthropic-claude-v1", "index": 0}]


def _proxy_env(srv: FakeChatVariantServer) -> dict[str, str]:
    url = srv.proxy_url
    return {"HTTP_PROXY": url, "http_proxy": url, "HTTPS_PROXY": url, "https_proxy": url, "NO_PROXY": "", "no_proxy": ""}


def _impersonated_config(host: str, model: str = "vendor/reasoner") -> dict:
    return {"model": {"provider": "custom", "base_url": f"http://{host}/api/v1", "default": model,
                      "context_length": 128000}}


def _assistant_field(body: dict, field: str) -> list:
    return [m.get(field) for m in chat_messages(body, "assistant")]


def _two_turn_reasoning_session(h: Home, env: dict | None) -> tuple:
    (h.project / "notes.txt").write_text("CANARY-RD\n", encoding="utf-8")
    first = oneshot(h, "read notes.txt", env=env)
    assert first.proc.returncode == 0 and first.stdout.strip() == "ONE", first.describe()
    second = oneshot(h, "again", resume=first.session_id, env=env)
    assert second.proc.returncode == 0 and second.stdout.strip() == "TWO", second.describe()
    return first, second


def _reasoning_script(field: str) -> list:
    return [
        CTools([(READ_TOOL, {"path": "notes.txt"})], **{field: _rd("1") if field == "reasoning_details" else "ds-1"}),
        CText("ONE", **{field: _rd("2") if field == "reasoning_details" else "ds-2"}),
        CText("TWO"),
    ]


def test_openrouter_reasoning_details_replayed_verbatim(tmp_path) -> None:
    h = Home(tmp_path)
    with FakeChatVariantServer(_reasoning_script("reasoning_details")) as srv:
        h.write(_impersonated_config("openrouter.ai"), dotenv={"OPENAI_API_KEY": "sk-or-fake"})
        first, _ = _two_turn_reasoning_session(h, _proxy_env(srv))
        records = srv.main_records()
        invalid = srv.invalid_requests()

    assert invalid == [], invalid
    assert {r["host"] for r in records} == {"openrouter.ai"}, "precondition: the impersonated route was used"
    mains = [r["body"] for r in records]
    assert len(mains) == 3, [m.get("messages") for m in mains]
    # Next call of the same turn: the tool-call message carries its signed block unchanged.
    assert _assistant_field(mains[1], "reasoning_details") == [_rd("1")], chat_messages(mains[1], "assistant")
    # After a process restart, both blocks come back from state.db, in order, unmodified.
    assert _assistant_field(mains[2], "reasoning_details") == [_rd("1"), _rd("2")], chat_messages(mains[2], "assistant")


def test_reasoning_details_withheld_from_routes_that_do_not_read_them(tmp_path) -> None:
    h = Home(tmp_path)
    with FakeChatVariantServer(_reasoning_script("reasoning_details")) as srv:
        h.write(custom_chat_config(srv.base_url), dotenv={"OPENAI_API_KEY": "sk-fake"})
        first, _ = _two_turn_reasoning_session(h, None)
        mains = srv.main_requests()

    leaked = [(i, m) for i, b in enumerate(mains) for m in chat_messages(b, "assistant") if "reasoning_details" in m]
    assert leaked == [], f"reasoning_details sent to a route with a strict schema: {leaked}"
    stored = [json.loads(r["reasoning_details"]) for r in db_messages(h, first.session_id)
              if r["role"] == "assistant" and r["reasoning_details"]]
    assert stored == [_rd("1"), _rd("2")], "state.db must keep the blocks for a later switch back"


def test_reasoning_content_echoed_on_tool_call_messages(tmp_path) -> None:
    h = Home(tmp_path)
    with FakeChatVariantServer(_reasoning_script("reasoning_content")) as srv:
        h.write(custom_chat_config(srv.base_url, model="deepseek-reasoner"), dotenv={"OPENAI_API_KEY": "sk-fake"})
        _two_turn_reasoning_session(h, None)
        mains = srv.main_requests()
        invalid = srv.invalid_requests()

    assert invalid == [], invalid
    tool_msg = [m for m in chat_messages(mains[1], "assistant") if m.get("tool_calls")]
    assert [m.get("reasoning_content") for m in tool_msg] == ["ds-1"], chat_messages(mains[1], "assistant")
    assert _assistant_field(mains[2], "reasoning_content") == ["ds-1", "ds-2"], chat_messages(mains[2], "assistant")


@pytest.mark.parametrize("scenario", [pytest.param("budget", marks=known("reasoning_budget_session_keeps_answering"))])
def test_long_session_does_not_wedge_on_replayed_reasoning_budget(tmp_path, scenario) -> None:
    """Each turn mints ~1 KB of reasoning; the route 400s ("Provider returned error", not
    retryable) once the replayed total passes 4 KB. Every turn must still be answered."""
    turns = 8
    script = [CText(f"ANSWER-{i}", reasoning_details=_rd(str(i), "r" * 1000)) for i in range(turns)]
    h = Home(tmp_path)
    answers: list[str] = []
    with FakeChatVariantServer(script, reasoning_budget_chars=4000) as srv:
        h.write(_impersonated_config("inference-api.nousresearch.com"), dotenv={"OPENAI_API_KEY": "sk-fake"})
        gw = TuiGateway(h, _proxy_env(srv))
        try:
            sid = gw.call("session.create", {"cols": 120})["session_id"]
            for i in range(turns):
                answers.append(gw.turn(sid, f"question {i}"))
        finally:
            gw.close()
        records = srv.main_records()
    assert records and {r["host"] for r in records} == {"inference-api.nousresearch.com"}, "precondition: impersonated"
    rejected = [i for i, r in enumerate(records) if r.get("response") == "route_rejection"]
    assert rejected, "precondition: the replayed total crossed the route budget at least once"
    missing = [i for i in range(turns) if f"ANSWER-{i}" not in answers[i]]
    assert not missing, f"turns {missing} were not answered once replayed reasoning passed the budget: {answers}"
