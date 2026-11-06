"""EP-2 rig (#145171): cron delivery -> ``/compress`` -> human input through the REAL gateway.

Scripted rebuild of the manual live check NF5 in #134889 ("cron -> /compress -> human"). The
gateway is the shipped entry point (``python -m gateway.run``) with the chaos lane's in-memory
platform loaded as a real user plugin, its own in-process cron ticker and its real session
store, state.db and slash dispatch. Only the LLM (``tests/fakes/fake_llm_provider``) and the
chat platform are fake.

Steps, all in one DM chat:

1. six human turns;
2. a cron job whose origin is that chat (``attach_to_session: true``) is created and triggered; the
   gateway's ticker runs it, delivers its report to the chat and mirrors it into the chat's
   session transcript;
3. the human sends ``/compress``;
4. the human sends one more turn.

Invariants: every human prompt reaches the model exactly once and the cron prompt exactly once
(and never in the chat's own context as a prompt of its own); the follow-up turn is answered once;
the request it makes carries the compaction summary in place of at least one whole earlier turn
(every turn it drops reached the summarizer), keeps the remaining turns in order, and still carries
the cron report verbatim; state.db keeps the summary, the follow-up prompt and its answer as live
rows, no summarized turn stays live, the DB passes ``integrity_check``, and SIGTERM drains the
gateway.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.chaos import _gateway_fake_platform as fake_platform
from tests.e2e.core.chaos._gateway_harness import GatewayProc
from tests.e2e.core.chaos._helpers import (
    REPO_ROOT,
    TURN_DEADLINE_S,
    hermetic_env,
    integrity_ok,
    kill_tagged,
    new_tag,
    persisted_messages,
    python_exe,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.waiting import wait_until

CHAT = "chaos-chat"
HUMAN_TURNS = 6
# The in-process ticker runs every 60 s; a triggered job fires on the next tick, then runs a turn.
CRON_DEADLINE_S = 240.0
HOME_ENV = "CHAOSFAKE_HOME_CHANNEL"

# The chaos plugin registers no cron home env var, so cron would refuse the platform as a delivery
# target. This plugin is the same adapter with that one registration field added.
PLUGIN_INIT = (
    "from tests.e2e.core.chaos._gateway_fake_platform import (\n"
    "    ALLOW_ALL_ENV, PLATFORM_NAME, ChaosFakeAdapter)\n\n\n"
    "def register(ctx):\n"
    "    ctx.register_platform(\n"
    "        name=PLATFORM_NAME, label='Chaos Fake', adapter_factory=ChaosFakeAdapter,\n"
    "        check_fn=lambda: True, validate_config=lambda _cfg: True,\n"
    "        is_connected=lambda _cfg: True, allow_all_env=ALLOW_ALL_ENV,\n"
    f"        cron_deliver_env_var={HOME_ENV!r})\n"
)

COMPRESSION_CONFIG = (
    "compression:\n"
    "  enabled: true\n"
    "  protect_last_n: 2\n"
    "auxiliary:\n"
    "  title_generation:\n"
    "    enabled: false\n"
    # The gateway refreshes remote model catalogs 30 s after boot, and this run outlives that.
    "model_catalog:\n"
    "  enabled: false\n"
)

CREATE_JOB = """
import json, sys
from cron.jobs import create_job, trigger_job
spec = json.loads(sys.argv[1])
job = create_job(prompt=spec["prompt"], schedule="every 1h", name="ep2-digest", deliver="origin",
                 origin=spec["origin"], attach_to_session=True)
trigger_job(job["id"])
print(job["id"])
"""


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _last_user(messages: list[dict[str, Any]]) -> str:
    return next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")


def _whole(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_text(m.get("content")) for m in messages)


class Script:
    """Answers keyed by the marker in the newest user message; summaries carry ``summary``."""

    def __init__(self, nonce: str) -> None:
        self.nonce = nonce
        self.cron_marker = f"[[cron:{nonce}]]"
        self.report = f"CRON-REPORT-{nonce}"
        self.summary = f"SUMMARY-{nonce}"

    def marker(self, n: int) -> str:
        return f"[[human:{self.nonce}:{n}]]"

    def end(self, n: int) -> str:
        return f"[[end:{self.nonce}:{n}]]"

    def human(self, n: int) -> str:
        # Long enough that a short summary shrinks the conversation (a growing summary is refused). The end
        # marker sits past the summarizer's auto focus quote (the first 260 chars of the newest user turns).
        return (f"{self.marker(n)} please remember item {n}: " + " ".join(f"detail-{n}-{i}" for i in range(600))
                + f" {self.end(n)}")

    def answer(self, n: int) -> str:
        return f"noted item {n} for {self.nonce}"

    def main(self, rec: dict[str, Any]) -> Text:
        last = _last_user(rec["body"].get("messages") or [])
        for n in range(HUMAN_TURNS + 1, 0, -1):
            if f"[[human:{self.nonce}:{n}]]" in last:
                return Text(self.answer(n))
        if self.cron_marker in last:
            return Text(self.report)
        return Text("ok")

    def aux(self, _rec: dict[str, Any]) -> Text:
        return Text(f"## Goal\nRemember the numbered items ({self.summary}).\n"
                    "## Progress\n### Done\n- Noted the early items.\n## Next Steps\n- Keep going.\n")


def _turn(gw: GatewayProc, text: str, what: str, needle: str | None = None) -> list[str]:
    """Send one message and wait until the gateway finished it and released the chat; return what was
    sent to the chat. With ``needle``, the answer must arrive first: a message that lands while startup
    restore is still running is parked (an early processing-complete) and replayed later."""
    start = gw.mark()
    msg_id = gw.send_user(text, chat=CHAT)
    since = start
    if needle is not None:
        answer = gw.wait_for(lambda e: e.ev == "send" and needle in (e.data.get("text") or ""),
                             TURN_DEADLINE_S, what, since=start)
        assert answer is not None, f"{what}: unanswered: {gw.sends_since(start)}\n{gw.log_tail()}"
        since = answer.idx
    done = gw.wait_for(lambda e: e.ev == "complete" and e.data.get("id") == msg_id, TURN_DEADLINE_S, what,
                       since=since)
    assert done is not None, f"{what}: not completed in {TURN_DEADLINE_S}s\n{gw.log_tail()}"
    idle = gw.wait_for(lambda e: e.ev == "hb" and not e.data.get("busy"), TURN_DEADLINE_S, what, since=done.idx)
    assert idle is not None, f"{what}: chat never released\n{gw.log_tail()}"
    return gw.sends_since(start)


def _create_and_trigger_job(gw: GatewayProc, tag: str, prompt: str) -> str:
    spec = {"prompt": prompt, "origin": {"platform": fake_platform.PLATFORM_NAME, "chat_id": CHAT,
                                         "chat_type": "dm", "user_id": "chaos-user"}}
    out = subprocess.run(
        [python_exe(), "-c", CREATE_JOB, json.dumps(spec)], cwd=str(REPO_ROOT),
        env=hermetic_env(gw.home, gw.hermes_home, tag), capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, f"cron job create failed: {out.stderr[-2000:]}"
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.platforms("linux")
def test_cron_delivery_then_compress_then_human_input(tmp_path: Path) -> None:
    script = Script(uuid.uuid4().hex[:10])
    srv = FakeLLMServer(script.main, aux=script.aux)
    srv.start()
    tag = new_tag()
    gw = GatewayProc(tmp_path, srv.base_url, tag)
    (gw.hermes_home / "plugins" / fake_platform.PLUGIN_DIR_NAME / "__init__.py").write_text(
        PLUGIN_INIT, encoding="utf-8")
    cfg_path = gw.hermes_home / "config.yaml"
    cfg = cfg_path.read_text(encoding="utf-8")
    assert "compression:\n  enabled: false\n" in cfg
    cfg_path.write_text(cfg.replace("compression:\n  enabled: false\n", COMPRESSION_CONFIG), encoding="utf-8")
    try:
        gw.start()
        for n in range(1, HUMAN_TURNS + 1):
            _turn(gw, script.human(n), f"human turn {n}", needle=script.answer(n))

        start = gw.mark()
        _create_and_trigger_job(gw, tag, f"{script.cron_marker} write the status report")
        delivered = gw.wait_for(lambda e: e.ev == "send" and e.data.get("chat") == CHAT
                                and script.report in (e.data.get("text") or ""),
                                CRON_DEADLINE_S, "cron delivery", since=start)
        assert delivered is not None, f"cron report never reached the chat\n{gw.log_tail()}"
        # The mirror lands right after the send; the chat's session must hold it before /compress.
        wait_until(lambda: any(script.report in _text(m.get("content"))
                               for m in persisted_messages(gw.state_db)
                               if "[Cron delivery:" in _text(m.get("content"))),
                   timeout=60, what="cron delivery mirrored into the chat session")

        compress_reply = "\n".join(_turn(gw, "/compress", "/compress"))
        # Title generation is off, so every auxiliary call is the /compress summarizer.
        summary_calls = srv.aux_requests()
        main_before = len(srv.main_requests())

        sent = _turn(gw, script.human(HUMAN_TURNS + 1), "follow-up after /compress",
                     needle=script.answer(HUMAN_TURNS + 1))
        final_rows = persisted_messages(gw.state_db)
    finally:
        gw.stop()
        kill_tagged(tag)
        srv.stop()

    mains = srv.main_requests()
    prompts = [_last_user(b.get("messages") or []) for b in mains]
    turns = range(1, HUMAN_TURNS + 2)
    # Each human prompt reached the model once, and the cron prompt once, in its own session.
    for n in turns:
        hits = sum(script.marker(n) in p for p in prompts)
        assert hits == 1, f"human turn {n} reached the model {hits} times"
    cron_runs = [b for b in mains if script.cron_marker in _last_user(b.get("messages") or [])]
    assert len(cron_runs) == 1, f"cron prompt reached the model {len(cron_runs)} times"
    assert not any(script.marker(n) in _whole(cron_runs[0].get("messages") or []) for n in turns), (
        "the cron job ran inside the chat's context")

    assert summary_calls, f"/compress never asked the summarizer; reply was: {compress_reply!r}"
    summarizer_input = "\n".join(_whole(b.get("messages") or []) for b in summary_calls)
    follow_ups = [b for b in mains[main_before:] if script.marker(HUMAN_TURNS + 1) in _last_user(b.get("messages") or [])]
    assert len(follow_ups) == 1 and sum(script.answer(HUMAN_TURNS + 1) in s for s in sent) == 1, (
        f"follow-up not answered exactly once: {sent}")
    whole = _whole(follow_ups[0].get("messages") or [])
    assert script.summary in whole, f"follow-up request does not carry the compaction summary ({compress_reply!r})"
    # The summary quotes user messages (clipped), so a turn counts as kept only when it is there whole.
    kept = [n for n in turns if script.human(n) in whole]
    summarized = [n for n in turns if n not in kept]
    assert summarized, "the follow-up still carries every earlier turn verbatim"
    # The end marker, not the start one: the auto focus block quotes the start of the newest user turns.
    assert all(script.end(n) in summarizer_input for n in summarized), "a turn vanished without being summarized"
    positions = [whole.index(script.human(n)) for n in kept]
    assert positions == sorted(positions) and kept[-1] == HUMAN_TURNS + 1, f"turn order broken: {kept}"
    later = [whole.index(script.human(n)) for n in kept if n > max(summarized)]
    assert whole.index(script.summary) < min(later), "the summary is not ahead of the turns it precedes"
    # The tail keeps at least the last 3 messages, so the mirror (the newest row before /compress) stays
    # verbatim. The summarizer input can't vouch for it: its auto focus block quotes the mirror either way.
    assert script.report in whole, "compression lost the cron report"

    live = _whole(final_rows)
    assert script.summary in live and script.answer(HUMAN_TURNS + 1) in live, "state.db lost the follow-up"
    assert any(m.get("role") == "user" and script.human(HUMAN_TURNS + 1) in _text(m.get("content"))
               for m in final_rows), "state.db lost the follow-up user message"
    live_rows = [_text(m.get("content")) for m in final_rows]
    assert not any(script.human(n) in live_rows for n in summarized), "a summarized turn is still a live row"
    assert integrity_ok(gw.state_db) == "ok"
    # Same shutdown bar as the chaos lane: SIGTERM drains the gateway by itself (no signal death).
    assert gw.shutdown_s is not None and gw.exit_code is not None and gw.exit_code >= 0, (
        f"gateway did not drain on SIGTERM (exit {gw.exit_code})\n{gw.log_tail()}")
