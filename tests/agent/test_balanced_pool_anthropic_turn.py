"""A ``balanced`` Anthropic pool, end to end through the real agent.

Between turns the session moves to the credential with the most headroom, the streamed request
goes out on that credential (the pre-request refresh keeps it), and the response's plan windows
and organization are recorded against it.
"""

import json
import os
import time
from pathlib import Path

import httpx
import pytest

from agent import credential_pool_balanced as balanced
from agent.credential_pool import PooledCredential, load_pool

_BASE = "https://api.anthropic.com"
_SSE_EVENTS = (
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 1}}),
    ("message_stop", {"type": "message_stop"}),
)


@pytest.fixture(autouse=True)
def _fresh_process_view(monkeypatch):
    monkeypatch.setattr(balanced, "_observed", {})
    monkeypatch.setattr(balanced, "_warm", {})


@pytest.fixture
def pool(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # no Claude Code login to borrow
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "model:\n  provider: anthropic\ncredential_pool_strategies:\n  anthropic: balanced\n")
    rows = [
        PooledCredential(
            provider="anthropic", id=cid, label=cid, auth_type="oauth", priority=index, source="manual:hermes_pkce",
            access_token=f"cc-{cid}-access-token", refresh_token=f"cc-{cid}-refresh-token",
            expires_at_ms=int((time.time() + 3600) * 1000), base_url=_BASE,
        ).to_dict()
        for index, cid in enumerate(("full", "free", "spare"))
    ]
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": rows}}))
    return load_pool("anthropic")


@pytest.fixture
def wire(monkeypatch):
    """Answers every request with a streamed reply whose headers report the plan windows."""
    sent = []

    def send(client, request, **kwargs):
        sent.append(request)
        reset = int(time.time()) + 3600
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in _SSE_EVENTS)
        return httpx.Response(200, request=request, content=body.encode(), headers={
            "content-type": "text/event-stream",
            "anthropic-organization-id": "org-team",
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-5h-status": "allowed",
            "anthropic-ratelimit-unified-5h-utilization": "0.15",
            "anthropic-ratelimit-unified-5h-reset": str(reset),
        })

    monkeypatch.setattr(httpx.Client, "send", send)
    return sent


def _agent(pool, entry_id: str, session_id: str = "session-1"):
    from run_agent import AIAgent

    entry = next(e for e in pool.entries() if e.id == entry_id)
    agent = AIAgent(
        api_key=entry.runtime_api_key, base_url=_BASE, provider="anthropic", api_mode="anthropic_messages",
        model="claude-test", credential_pool=pool, session_id=session_id,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
    )
    agent._credential_pool_entry_id = entry.id
    agent._interrupt_requested = False
    return agent


def test_a_cold_session_moves_to_headroom_and_its_turn_runs_and_reports_there(pool, wire):
    balanced.observe("anthropic", "full", {"anthropic-ratelimit-unified-5h-utilization": "0.9"})
    balanced.observe("anthropic", "free", {"anthropic-ratelimit-unified-5h-utilization": "0.1"})
    agent = _agent(pool, "full")

    agent._restore_primary_runtime()
    agent._interruptible_streaming_api_call(
        {"model": "claude-test", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})

    assert agent._credential_pool_entry_id == "free"
    assert wire[-1].headers["Authorization"] == "Bearer cc-free-access-token"
    assert balanced.warm_entry("anthropic", "session-1") == "free"
    report = balanced._observed[("anthropic", "free")]
    assert report.group == "org-team" and report.windows["5h"].used == 0.15


def test_sessions_rotating_off_a_spent_seat_together_spread(pool, wire):
    """Every session on a seat whose window fills hits the 429 within moments of the others."""
    for entry_id, used in (("full", "1.0"), ("free", "0.4"), ("spare", "0.45")):
        balanced.observe("anthropic", entry_id, {"anthropic-ratelimit-unified-5h-utilization": used})
    spent = {"reason": "quota_exhausted", "reset_at": time.time() + 3600}
    landed = []

    for session_id in ("s1", "s2"):  # each agent loads its own pool, as gateway sessions do
        agent = _agent(load_pool("anthropic"), "full", session_id)
        recovered, _retried = agent._recover_with_credential_pool(
            status_code=429, has_retried_429=False, error_context=spent)
        assert recovered
        landed.append(agent._credential_pool_entry_id)

    assert landed == ["free", "spare"]
