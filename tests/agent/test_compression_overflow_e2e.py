"""End-to-end replay of an observed overflow (local model at llama.cpp, 131,072 window, no max_tokens):
no request over the model window may ever be SENT.

Observed path: a profile driven over the gateway
API (``POST /v1/runs``) -> ``api_server_runs`` -> ``APIServerAdapter._create_agent()`` ->
``AIAgent.run_conversation()`` -> ``agent.conversation_loop._run_conversation_turn`` ->
``_run_api_retry_loop`` -> ``build_api_request`` -> ``perform_api_call``. Two compressions aborted
(the summary reasoning stream looped), the cooldown blocked the threshold trigger, and the wire
fit check (char/4 estimate) priced the final request at 113,292 tokens while the provider counted
131,808 (dense source files tokenize ~1.17x the estimate): it "fit" the 116,327 budget, was sent,
and got HTTP 400 "prompt (131808 tokens) leaves no room to answer".

This test builds the agent with the SAME factory the gateway API uses (``_create_agent``, platform
``api_server``) and drives ``run_conversation`` the way ``api_server_runs`` does, against a real
loopback OpenAI-compatible provider that

* counts prompt tokens denser than Hermes's estimate (json chars / 3.56, the ratio measured on
  the failed request dump) and reports them as usage,
* answers HTTP 400 "leaves no room to answer" for any prompt that does not fit the window,
* streams every compression summary as a reasoning repetition loop (aborted by the runaway check,
  ``summary_generation_aborted``),

while the scripted model replays a growing code review: read_file results of 36K-51K chars (the
sizes from the run), then smaller reads that step through the gap between the provider's count
and the estimate.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

# Import the lazy chain at collection time so the home-I/O guard never sees bootstrap reads.
import agent.conversation_loop
import agent.process_bootstrap

from tests.fakes.fake_llm_provider import MODEL_ID, Error, FakeLLMServer, Text, ToolCall

WINDOW = 131_072
# Provider-side density measured on the failed request: 469,949 JSON chars = 131,808 tokens.
JSON_CHARS_PER_TOKEN = 3.56
# read_file result sizes (chars) from the observed session, in order, then small reads (6K/5K/1K/16K files).
LARGE_READS = [44_397, 29_347, 16_620, 45_642, 41_991, 51_160, 21_610, 47_860, 10_668, 36_210]
SMALL_READS = [6_068, 5_191, 1_188, 6_098, 4_996] * 4
LOOP_LINE = "Wait, let me re-check the parser.py findings before writing the summary.\n"


def provider_tokens(body: dict) -> int:
    chars = len(json.dumps(body.get("messages", []))) + len(json.dumps(body.get("tools") or []))
    return math.ceil(chars / JSON_CHARS_PER_TOKEN)


def _swift_file(path: Path, chars: int, rng: random.Random) -> Path:
    words = ("let", "var", "func", "guard", "return", "self", "card", "price", "scan", "store",
             "await", "try", "\"id\"", "[String: Any]", "?.", "->", "{", "}", "(", ")")
    lines, total, n = [], 0, 0
    while total < chars:
        line = "    " + " ".join(rng.choice(words) for _ in range(rng.randint(4, 12)))
        lines.append(line)
        total += len(line) + 8  # read_file adds a "N|" line-number gutter
        n += 1
    path.write_text("//  " + path.name + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path


class _Provider:
    def __init__(self) -> None:
        self.main_sizes: list[int] = []
        self.rejected: list[int] = []
        self.summary_calls = 0
        self.script: list = []

    def main(self, rec: dict):
        tokens = provider_tokens(rec["body"])
        self.main_sizes.append(tokens)
        if tokens >= WINDOW:
            self.rejected.append(tokens)
            return Error(400, f"prompt ({tokens} tokens) leaves no room to answer in the context "
                              f"({WINDOW}); requests are never truncated")
        if self.script:
            return self.script.pop(0)
        return Text("Review complete.", chunk_chars=64)

    def aux(self, rec: dict):
        self.summary_calls += 1
        # A thinking model stuck in a loop: reasoning only, far past the 160K-char check point.
        return Text("", reasoning=LOOP_LINE * (200_000 // len(LOOP_LINE)), chunk_chars=4000)


@pytest.fixture
def gateway_agent(tmp_path, monkeypatch):
    """An AIAgent built by the gateway API's own factory, pointed at the loopback provider."""
    prov = _Provider()
    server = FakeLLMServer(prov.main, aux=prov.aux, prompt_tokens_fn=provider_tokens)
    server.start()
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        f"  base_url: {server.base_url}\n"
        f"  default: {MODEL_ID}\n"
        f"  context_length: {WINDOW}\n"
        "agent:\n"
        "  api_max_retries: 1\n"
        "updates:\n"
        "  check: false\n"
        "telemetry:\n"
        "  shared_metrics:\n"
        "    enabled: false\n"
        "auxiliary:\n"
        "  title_generation:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("OPENAI_API_KEY=sk-fake-e2e\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_a, **_k: {"file"})

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    db = hermes_state.SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: db)
    agent = adapter._create_agent(session_id="run_e2e_overflow")
    try:
        yield agent, prov, tmp_path / "workspace"
    finally:
        try:
            agent.close()
        finally:
            db.close()
            server.stop()


def test_gateway_run_never_sends_a_request_over_the_window(gateway_agent):
    agent, prov, workdir = gateway_agent
    assert agent.platform == "api_server"
    assert agent.context_compressor.context_length == WINDOW
    assert agent.max_tokens is None

    workdir.mkdir()
    rng = random.Random(9_549)
    steps: list = []
    for i, size in enumerate(LARGE_READS + SMALL_READS):
        path = _swift_file(workdir / f"File{i:02d}.swift", size, rng)
        steps.append(ToolCall("read_file", {"path": str(path)}, text=f"Reading file {i}."))
    steps.append(Text("Review complete: findings recorded.", chunk_chars=64))
    prov.script = steps

    result = agent.run_conversation(
        user_message="TASK: Do a full code review of the app in the workspace.",
        conversation_history=[], task_id="run_e2e_overflow")

    sizes = prov.main_sizes
    assert prov.summary_calls >= 1, "the scenario must exercise aborted compressions"
    assert max(sizes) > 98_304, f"the conversation never crossed the compression threshold: {sizes}"
    assert not prov.rejected, (
        f"requests over the {WINDOW:,}-token window were sent: {prov.rejected} (all sizes: {sizes})")
    assert all(s < WINDOW for s in sizes)
    # Every scripted step was consumed and the turn finished normally.
    assert not prov.script, f"turn stopped early: {result.get('error')!r} sizes={sizes}"
    assert result.get("final_response") == "Review complete: findings recorded."
    assert not result.get("failed")


def test_gateway_run_fails_clearly_instead_of_sending_an_unfittable_request(gateway_agent):
    """Nothing to clear (the user message alone is over the window): the turn ends with a clear
    error and the oversized request is never sent."""
    agent, prov, _workdir = gateway_agent
    prov.script = [Text("should never be reached")]
    huge = "Please review this paste:\n" + ("let card = store.price(for: id) ?? 0\n" * 14_000)

    result = agent.run_conversation(user_message=huge, conversation_history=[], task_id="run_e2e_unfit")

    assert not prov.rejected, f"an over-window request was sent: {prov.rejected}"
    assert all(s < WINDOW for s in prov.main_sizes), prov.main_sizes
    assert result.get("failed") is True
    assert "does not fit the model context window" in (result.get("error") or "")
    assert result.get("turn_exit_reason") == "request_exceeds_context_window"
