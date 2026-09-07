"""Real compressor -> auxiliary router -> Responses SDK runtime handoff.

Only the HTTP transport is synthetic. Route comparison, runtime construction,
summary serialization, client creation and Responses adaptation execute normally.
"""
import asyncio
import json
import socket
import threading

import httpx
import pytest
import yaml

import agent.auxiliary_client as aux
from agent.context_compressor import ContextCompressor, pin_summary_route
from agent.conversation_compression import CompressionCommitFence
from run_agent import AIAgent


MODEL = "gpt-5.6-session-fixture"
URL = "http://127.0.0.1:1/v1"
KEY = "synthetic-only-no-provider-access"
SUMMARY = "## Goal\nSynthetic checkpoint.\n## Completed Actions\n" + (
    "Verified the synthetic data and preserved the pending local task.\n" * 4
)


def reasoning(effort):
    return None if effort is None else {"enabled": True, "effort": effort}


def history():
    return [
        {"role": "user" if n % 2 == 0 else "assistant",
         "content": f"Synthetic turn {n}: " + "local fixture context " * 180}
        for n in range(28)
    ]


def response_bytes(body, text=SUMMARY):
    output = [{"type": "message", "id": "msg_fixture", "role": "assistant",
               "status": "completed", "content": [
                   {"type": "output_text", "text": text, "annotations": []}]}]
    response = {"id": "resp_fixture", "object": "response", "created_at": 1,
                "model": body["model"], "status": "completed", "output": output,
                "usage": {"input_tokens": 1000, "output_tokens": 7,
                          "total_tokens": 1007,
                          "input_tokens_details": {"cached_tokens": 600},
                          "output_tokens_details": {"reasoning_tokens": 3}},
                "error": None, "incomplete_details": None}
    events = [
        {"type": "response.created", "response": dict(response, status="in_progress", output=[])},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
         "item_id": "msg_fixture", "delta": text},
        {"type": "response.completed", "response": response},
    ]
    return "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n"
                   for e in events).encode()


@pytest.fixture
def wire(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "model": {"provider": "custom", "default": MODEL, "base_url": URL,
                  "context_length": 65536},
        "agent": {"reasoning_effort": "low"},
        "context": {"engine": "compressor"},
        "compression": {"enabled": True},
        "auxiliary": {"compression": {"provider": "auto", "model": "", "timeout": 5},
                      "title_generation": {"enabled": False},
                      "background_review": {"enabled": False}},
        "session": {"save_json": False},
    }

    def configure(task=None):
        if task is not None:
            cfg["auxiliary"]["compression"] = {"provider": "auto", "model": "", "timeout": 5, **task}
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))

    configure()
    requests = []
    hook = [None]
    lock = threading.Lock()

    def handle(_transport, request):
        assert request.url.host == "127.0.0.1" and request.url.port == 1
        assert request.url.path == "/v1/responses" and request.method == "POST"
        body = json.loads(request.content)
        with lock:
            requests.append(body)
        if hook[0]:
            replacement = hook[0](request, body)
            if replacement is not None:
                return replacement
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=response_bytes(body), request=request)

    async def async_handle(transport, request):
        return handle(transport, request)

    def no_socket(*args, **kwargs):
        raise AssertionError("Unexpected real network in synthetic compression test")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_handle)
    monkeypatch.setattr(socket.socket, "connect", no_socket)
    aux.clear_runtime_main()
    aux.shutdown_cached_clients()
    agents = []

    def make(effort="high", sid="session-fixture", **kwargs):
        agent = AIAgent(
            api_key=KEY, base_url=URL, provider="custom", model=MODEL,
            api_mode="codex_responses", reasoning_config=reasoning(effort),
            enabled_toolsets=[], quiet_mode=True, session_id=sid,
            skip_context_files=True, skip_memory=True, skip_background_review=True,
            save_trajectories=False, max_iterations=2,
            **kwargs,
        )
        agent._cached_system_prompt = "Synthetic system."
        agents.append(agent)
        return agent

    yield make, requests, configure, hook
    for agent in agents:
        agent.close()
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()


def compress(agent, pooled=False):
    kw = {} if pooled else {"commit_fence": CompressionCommitFence()}
    return agent._compress_context(history(), "Synthetic system.", force=True,
                                   approx_tokens=30000, **kw)


@pytest.mark.parametrize("pooled", [False, True])
def test_current_session_effort_reaches_responses(wire, pooled):
    make, requests, _, _ = wire
    agent = make()
    compressed, _ = compress(agent, pooled)
    assert len(compressed) < 28
    assert len(requests) == 1
    assert requests[0]["reasoning"]["effort"] == "high"
    assert requests[0]["stream"] is True
    assert len(requests[0]["input"]) == 1
    assert requests[0]["prompt_cache_key"]


@pytest.mark.parametrize("effort,expected", [("low", "low"), (None, None), ("none", "none")])
def test_reused_compressor_uses_current_effort(wire, effort, expected):
    make, requests, _, _ = wire
    agent = make()
    compress(agent)
    agent.reasoning_config = reasoning(effort)
    compress(agent, pooled=True)
    assert requests[0].get("reasoning", {}).get("effort") == "high"
    assert requests[1].get("reasoning", {}).get("effort") == expected


def test_disabled_session_reasoning_respected(wire):
    make, requests, _, _ = wire
    agent = make()
    agent.reasoning_config = {"enabled": False}
    compress(agent)
    # The custom profile represents disabled reasoning as effort=none.
    assert requests[0]["reasoning"]["effort"] == "none"


@pytest.mark.parametrize("task", [{"reasoning_effort": "medium"},
                                  {"extra_body": {"reasoning": {"effort": "medium"}}}])
def test_auxiliary_override_keeps_ownership(wire, task):
    make, requests, configure, _ = wire
    configure(task)
    compress(make())
    assert requests[0]["reasoning"]["effort"] == "medium"


def test_pinned_fallback_owns_effort(wire):
    make, requests, _, _ = wire
    agent = make()
    with pin_summary_route({"provider": "custom", "model": "gpt-5.6-fallback-fixture",
                            "base_url": URL, "api_key": KEY, "api_mode": "codex_responses",
                            "reasoning_config": reasoning("medium")}):
        compress(agent)
    assert requests[0]["model"] == "gpt-5.6-fallback-fixture"
    assert requests[0]["reasoning"]["effort"] == "medium"


def test_runtime_snapshot_is_deep(wire):
    make, _, _, _ = wire
    agent = make()
    agent.reasoning_config["nested"] = {"values": [1]}
    frozen = agent._current_main_runtime()
    agent.reasoning_config["nested"]["values"].append(2)
    agent.reasoning_config["effort"] = "low"
    assert frozen["reasoning_config"] == {"enabled": True, "effort": "high", "nested": {"values": [1]}}


@pytest.mark.asyncio
async def test_parallel_async_sessions_do_not_leak(wire):
    make, requests, _, hook = wire
    agents = [make(effort, f"session-{effort}") for effort in ("high", "low")]
    barrier = threading.Barrier(2)
    def rendezvous(*_):
        barrier.wait(timeout=15)
    hook[0] = rendezvous
    await asyncio.gather(*(asyncio.to_thread(compress, agent) for agent in agents))
    assert sorted(r["reasoning"]["effort"] for r in requests) == ["high", "low"]
    assert len({r["prompt_cache_key"] for r in requests}) == 2


def test_standalone_construct_update_and_clear(wire):
    _, requests, _, _ = wire
    config = reasoning("high")
    c = ContextCompressor(MODEL, base_url=URL, provider="custom", api_key=KEY,
                          api_mode="codex_responses", config_context_length=65536,
                          quiet_mode=True, reasoning_config=config)
    config["effort"] = "low"
    assert c._generate_summary(history())
    c.update_model(MODEL, 65536, URL, KEY, "custom", "codex_responses")
    assert c._micro_summarize_one("Synthetic exchange")
    changed = reasoning("medium")
    c.update_model(MODEL, 65536, URL, KEY, "custom", "codex_responses", reasoning_config=changed)
    changed["effort"] = "low"
    assert c._generate_summary(history())
    c.update_model("gpt-5.6-other-fixture", 65536, URL, KEY, "custom", "codex_responses")
    assert c._generate_summary(history())
    assert [r.get("reasoning", {}).get("effort") for r in requests] == ["high", "high", "medium", None]


def test_foreign_session_runtime_not_borrowed(wire):
    make, requests, _, _ = wire
    own, foreign = make("high", "own"), make("low", "foreign")
    with aux.scoped_runtime_main(foreign._current_main_runtime()):
        assert own.context_compressor._generate_summary(history())
    assert requests[0]["reasoning"]["effort"] == "high"


def test_model_switch_current_runtime_overrides_compressor_snapshot(wire):
    make, requests, _, _ = wire
    agent = make()
    agent.switch_model("gpt-5.6-other-fixture", "custom", api_key=KEY,
                       base_url=URL, api_mode="codex_responses")
    assert agent.reasoning_config == reasoning("low")
    compress(agent)
    assert (requests[0]["model"], requests[0]["reasoning"]["effort"]) == (agent.model, "low")


def test_active_fallback_and_primary_restore_keep_their_own_effort(wire):
    make, requests, _, _ = wire
    agent = make(fallback_model={"provider": "custom", "model": "gpt-5.6-fallback-fixture",
                                "base_url": URL, "api_key": KEY, "api_mode": "codex_responses"})
    assert agent._try_activate_fallback()
    # Upstream resolves the fallback effort from per-model/global config (low here), not
    # from the fallback entry; the invariant is that compression follows the CURRENT
    # agent effort instead of the compressor's stale construction snapshot (high).
    assert agent.reasoning_config == reasoning("low")
    compress(agent)
    assert (requests[-1]["model"], requests[-1]["reasoning"]["effort"]) == ("gpt-5.6-fallback-fixture", "low")
    agent._restore_primary_runtime()
    # The init-time primary snapshot predates reasoning_config, so the restore keeps the
    # agent's live value; the model itself must return to the primary route.
    compress(agent)
    assert (requests[-1]["model"], requests[-1]["reasoning"]["effort"]) == (MODEL, "low")


def test_summary_retry_keeps_frozen_runtime(wire):
    make, requests, _, hook = wire
    agent = make()
    compressor = agent.context_compressor
    compressor.summary_model = "gpt-5.6-summary-fixture"

    def first_empty(_request, body):
        if len(requests) == 1:
            agent.reasoning_config["effort"] = "low"
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=response_bytes(body, text=""))

    hook[0] = first_empty
    compress(agent)
    assert [r["model"] for r in requests] == ["gpt-5.6-summary-fixture", MODEL]
    assert requests[-1]["reasoning"]["effort"] == "high"


@pytest.mark.asyncio
async def test_async_auxiliary_uses_real_responses_bridge(wire):
    make, requests, _, _ = wire
    agent = make()
    response = await aux.async_call_llm(task="compression", messages=history()[:2],
                                      main_runtime=agent._current_main_runtime())
    assert response.choices[0].message.content
    assert requests[0]["reasoning"]["effort"] == "high"


def test_snapshot_update_explicit_none_and_nested_copy(wire):
    make, requests, _, _ = wire
    c = make().context_compressor
    settings = {"effort": "low", "nested": {"values": [1]}}
    c.update_model(MODEL, 65536, URL, KEY, "custom", "codex_responses", reasoning_config=settings)
    settings["nested"]["values"].append(2)
    assert c.reasoning_config["nested"]["values"] == [1]
    c.update_model(MODEL, 65536, URL, KEY, "custom", "codex_responses", reasoning_config=None)
    assert c._generate_summary(history())
    assert "reasoning" not in requests[-1]


def test_nested_disabled_override_wins_over_profile_effort(wire):
    make, requests, configure, _ = wire
    configure({"extra_body": {"reasoning": {"enabled": False}}})
    compress(make())
    assert "reasoning" not in requests[0]


def test_turn_finalizer_micro_summary_uses_current_session(wire):
    make, requests, _, hook = wire
    agent = make()
    c = agent.context_compressor
    c._micro_compact_enabled = True
    agent.compression_enabled = False

    def after_main(_request, _body):
        if len(requests) == 1:
            # In-turn fallback/effort change occurs after turn_context's entry
            # snapshot; finalization must publish the current agent runtime.
            agent.reasoning_config = reasoning("low")

    hook[0] = after_main
    result = agent.run_conversation("Synthetic next turn", conversation_history=history())
    assert result["completed"]
    assert len(requests) == 2
    assert [r["reasoning"]["effort"] for r in requests] == ["high", "low"]
