"""Secrets never reach disk, the platform, or the next provider request through the messaging gateway.

Same contract and scenario table as ``test_secret_redaction.py`` (see its docstring), driven through the
long-lived surface: a real ``GatewayRunner`` child with the gateway's own file logging installed
(agent.log, errors.log, gateway.log) and a fake platform adapter whose journal is what the user sees.
Extra sink here: the platform wire (replies, tool-progress lines and error notices are egress, scrubbed
with ``redact_for_egress``). One gateway boot serves every scenario, each in its own chat.
"""

from __future__ import annotations

import sys

import pytest

from tests.e2e.core.security._helpers import write_home
from tests.e2e.core.security._redact import (
    CONFIG, SCENARIOS, Ctx, Director, LoggingGateway, Secrets, World, assert_harness_sane, cells, chat_for, check,
    collect, echo_preconditions, prompt_for, seed_workspace,
)
from tests.e2e.core.delivery._fake_platform import wait_until
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell commands and process groups")

# Keyed by cell id ``<scenario>-<sink>``: only the platform wire of the streamed answer is the known gap;
# the same answer reaching gateway.log, state.db, an export or the next request stays a plain red.
KNOWN: dict[str, str] = {
    "assistant_text-platform": "#56039 streamed gateway replies reach the platform without secret redaction",
}

PLATFORM = "fk_tg"


def _turn(gw: LoggingGateway, text: str, message_id: str, chat: str, replies_before: int) -> None:
    gw.inject(PLATFORM, text, message_id, chat)
    wait_until(lambda: len(gw.platform_view().visible(PLATFORM, chat)) > replies_before,
               f"a reply in {chat}", timeout=90, proc=gw.proc, log=gw.log)
    gw.wait_idle([chat], f"{chat} idle")


@pytest.fixture(scope="module")
def gw_world(tmp_path_factory) -> World:
    root = tmp_path_factory.mktemp("redact-gw")
    ws, keys = root / "ws", Secrets()
    seed_workspace(ws, keys)
    ctx = Ctx(keys, ws, root / "home" / ".hermes" / ".env")
    with FakeLLMServer(Director(ctx), api_key=keys.provider, record_get=True) as llm:
        ctx.port = llm.port
        gw = LoggingGateway(root, platforms={PLATFORM: "telegram"}, llm_base_url=llm.base_url)
        write_home(gw.hermes_home, llm.base_url, api_key=keys.provider, env=keys.env(), config=CONFIG)
        gw.start()
        try:
            for name, scenario in SCENARIOS.items():
                chat = chat_for(name)
                _turn(gw, prompt_for(name, keys), f"m-{name}", chat, 0)
                if scenario.followup:
                    seen = len(gw.platform_view().visible(PLATFORM, chat))
                    _turn(gw, prompt_for(name, keys, followup=True), f"m-{name}-2", chat, seen)
            wait_until(lambda: (gw.hermes_home / "logs" / "gateway.log").exists(), "gateway.log written",
                       timeout=30, proc=gw.proc, log=gw.log)
        finally:
            gw.stop()
        sinks = collect(gw.home, list(llm.requests), gw.spool / "platform.jsonl")
        gets = [r for r in llm.requests if r["kind"] == "get"]
    assert_harness_sane(sinks, gateway=True)
    logs = "\n".join(sinks.texts["logs"].values())
    return World(keys, sinks, echo_preconditions(ws, keys, gets, logs), {n: "(gateway)" for n in SCENARIOS})


@pytest.mark.parametrize("scenario, sink", cells(KNOWN, platform=True))
def test_gateway_turn_never_persists_delivers_or_replays_a_secret(gw_world: World, scenario: str, sink: str) -> None:
    check(gw_world, scenario, sink)
