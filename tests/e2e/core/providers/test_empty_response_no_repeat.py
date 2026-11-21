"""Empty-response recovery never re-sends an earlier request of the same turn.

A model that keeps answering with reasoning only walks Hermes through the whole recovery
ladder (post-tool nudge, two completion reminders, the capped plain retry). Every request
of that turn must differ from EVERY earlier one, not just from the one before it: an
identical request makes a reasoning model re-derive the same reasoning in new words, the
"paraphrased reasoning" loop. Checked on each route family, because what makes a retry
different depends on the route:

* ``custom``: a generic chat-completions endpoint gets no reasoning controls, so only the
  request-local hints can change the request;
* a reasoning-capable chat-completions route (Nous Portal, reached by vendor-host
  impersonation): reasoning-off goes out as ``extra_body.reasoning``;
* native Anthropic Messages: reasoning-off goes out as ``thinking``.

Real ``hermes -z`` processes; only the vendor HTTP boundary is faked.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from tests.e2e.core.providers._anthropic_helpers import start_rig
from tests.e2e.core.providers._openai_helpers import READ_TOOL, Home, bounded_turn, custom_chat_config
from tests.fakes.providers.anthropic_messages import Reply, Thinking, ToolUse
from tests.fakes.providers.chat_variants import CText, CTools, FakeChatVariantServer

pytestmark = [pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated"),
              pytest.mark.live_system_guard_bypass]

TURN_BUDGET = 150.0
REASONING = "Let me restate the task and plan the steps before answering."
SIG = "EqSig+/AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/AAA=="


def _prompt(body: dict[str, Any]) -> str:
    """What the model is asked: the conversation (and Anthropic's top-level system prompt).

    Reasoning controls are deliberately left out: many endpoints ignore ``reasoning_effort``,
    so a retry that only flips it is the same prompt again for them."""
    return json.dumps({"system": body.get("system"), "messages": body["messages"]}, sort_keys=True, ensure_ascii=False)


def _repeats(bodies: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(later, earlier) index pairs of requests whose prompt is byte-identical to an earlier one."""
    seen: dict[str, int] = {}
    hits = []
    for i, body in enumerate(bodies):
        key = _prompt(body)
        if key in seen:
            hits.append((i, seen[key]))
        else:
            seen[key] = i
    return hits


def _assert_ladder_never_repeats(bodies: list[dict[str, Any]], after_tool: bool, describe: str) -> None:
    # Precondition: the turn really went past both reminders into the plain retry
    # (original + 2 reminders + 1 retry, plus the post-tool nudge and the tool round when after_tool).
    assert len(bodies) >= (6 if after_tool else 4), f"ladder did not run to the plain retry: {len(bodies)}\n{describe}"
    assert len(bodies) <= 10, f"ladder did not stop: {len(bodies)} requests\n{describe}"
    assert _repeats(bodies) == [], (
        f"request(s) {_repeats(bodies)} (later, earlier) re-send an earlier prompt of the same turn\n{describe}")


# Chat-completions routes ---------------------------------------------------------------


def _chat_responder(after_tool: bool):
    state = {"main": 0}

    def respond(record: dict[str, Any]):
        state["main"] += 1
        if after_tool and state["main"] == 1:
            return CTools([(READ_TOOL, {"path": "notes.txt"})])
        return CText("", reasoning_content=REASONING)  # reasoning only, clean stop

    return respond


def _proxy_env(srv: FakeChatVariantServer) -> dict[str, str]:
    url = srv.proxy_url
    return {"HTTP_PROXY": url, "http_proxy": url, "HTTPS_PROXY": url, "https_proxy": url, "NO_PROXY": "", "no_proxy": ""}


@pytest.mark.parametrize("after_tool", [False, True], ids=["user", "after-tool"])
@pytest.mark.parametrize("route", ["custom", "nous-portal"])
def test_chat_completions_recovery_never_repeats_a_request(tmp_path, route: str, after_tool: bool) -> None:
    h = Home(tmp_path)
    with FakeChatVariantServer(_chat_responder(after_tool)) as srv:
        if route == "custom":
            config, env = custom_chat_config(srv.base_url), None
        else:
            config = {"model": {"provider": "custom", "base_url": "http://inference-api.nousresearch.com/api/v1",
                                "default": "vendor/reasoner", "context_length": 128000}}
            env = _proxy_env(srv)
        h.write(config, dotenv={"OPENAI_API_KEY": "sk-fake"})
        h.project.mkdir(parents=True, exist_ok=True)
        (h.project / "notes.txt").write_text("CANARY-NOTES\n", encoding="utf-8")
        run = bounded_turn(h, "Answer briefly: what is in notes.txt?", TURN_BUDGET, env=env)
        records = srv.main_records()
        invalid = srv.invalid_requests()

    describe = run.describe() if hasattr(run, "describe") else run.proc.stderr[-2000:]
    assert invalid == [], invalid
    bodies = [r["body"] for r in records]
    if route == "nous-portal":
        assert {r["host"] for r in records} == {"inference-api.nousresearch.com"}, "precondition: impersonated route"
        # Precondition: this route really carries reasoning controls, so reasoning-off is on the wire.
        assert any(b.get("reasoning_effort") == "none" or (b.get("reasoning") or {}).get("enabled") is False
                   for b in bodies), [{k: v for k, v in b.items() if k not in ("messages", "tools")} for b in bodies]
    _assert_ladder_never_repeats(bodies, after_tool, describe)


# Native Anthropic Messages ---------------------------------------------------------------


def _anthropic_responder(after_tool: bool):
    state = {"main": 0}

    def respond(record: dict[str, Any]):
        state["main"] += 1
        if after_tool and state["main"] == 1:
            return Reply([Thinking(REASONING, SIG), ToolUse(READ_TOOL, {"path": "notes.txt"})])
        return Reply([Thinking(REASONING, SIG)], stop_reason="end_turn")  # thinking only

    return respond


@pytest.mark.parametrize("after_tool", [False, True], ids=["user", "after-tool"])
def test_anthropic_recovery_never_repeats_a_request(tmp_path, after_tool: bool) -> None:
    rig = start_rig(tmp_path / "rig", _anthropic_responder(after_tool))
    try:
        (rig.project / "notes.txt").write_text("CANARY-NOTES\n", encoding="utf-8")
        proc = rig.run("-z", "Answer briefly: what is in notes.txt?", timeout=TURN_BUDGET)
        bodies = [r["body"] for r in rig.srv.main_requests()]
        schema_errors = rig.srv.schema_errors()
        describe = f"exit={proc.returncode}\n{proc.stderr[-1500:]}\n{rig.log_tail(1500)}"
    finally:
        rig.stop()
    assert not schema_errors, schema_errors
    _assert_ladder_never_repeats(bodies, after_tool, describe)
