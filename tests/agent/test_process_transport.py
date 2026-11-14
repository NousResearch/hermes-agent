"""External-process providers keep a local server's stall patience, keyed on auth_type (#149555).

The ``process://`` URL used to earn that patience by accident, because ``is_local_endpoint`` reads a
dot-less scheme host as a LAN name (#129153). The fixture's URL has a dot on purpose: the profile's
``auth_type`` decides now, and the budget no longer changes with the model name.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from agent.acp_client import ACPClient

PROVIDER = "patience-fixture"


@pytest.fixture
def process_agent(monkeypatch):
    from hermes_constants import get_hermes_home
    from providers import get_provider_profile
    from run_agent import AIAgent

    for name in ("HERMES_API_CALL_STALE_TIMEOUT", "HERMES_STREAM_STALE_TIMEOUT", "HERMES_LOCAL_STREAM_STALE_TIMEOUT",
                 "HERMES_API_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    plugin = get_hermes_home() / "plugins" / PROVIDER
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        f"name: {PROVIDER}\nkind: model-provider\nversion: 0.1.0\ndescription: test fixture\n")
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        f"register_provider(ProviderProfile(name='{PROVIDER}', auth_type='external_process', "
        "base_url='process://patience.fixture', supports_streaming=True))\n")
    profile = get_provider_profile(PROVIDER)
    monkeypatch.setattr(profile, "create_client", lambda **kwargs: object())
    return AIAgent(provider=PROVIDER, model="claude-opus-5-5[1m]", api_key="fixture", base_url=profile.base_url,
                   enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True)


def _budgets(agent, model: str) -> tuple[float, float]:
    from agent.chat_completion_helpers import _StreamingCall

    agent.model = model
    payload = {"model": model, "messages": [{"role": "user", "content": "x" * 1_300_000}]}
    call = _StreamingCall(agent, payload, None)
    call._resolve_stale_timeout()
    return agent._compute_non_stream_stale_timeout(payload), call._stream_stale_timeout


def _write_config(text: str) -> None:
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(text)


def test_external_process_stall_budget_is_one_explicit_deadline(process_agent):
    """A ~325K-token request to a child process: unset, the request timeout bounds a buffered call and the 900s
    local ceiling bounds a stream, the same for every model (opus was cut at 240s while haiku waited without
    limit). An explicit value is the operator's deadline on both paths; 180 is not mistaken for the unset default."""
    for model in ("claude-opus-5-5[1m]", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert _budgets(process_agent, model) == (1800.0, 900.0), model
    for seconds in (180, 1200):
        _write_config(f"providers:\n  {PROVIDER}:\n    stale_timeout_seconds: {seconds}\n")
        assert _budgets(process_agent, "claude-haiku-5-5") == (float(seconds), float(seconds)), seconds


_DEAF_AFTER_SESSION = """import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    result = {"protocolVersion": 1} if request["method"] == "initialize" else {"sessionId": "s1"}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
    if request["method"] == "session/new":
        break
time.sleep(120)
"""


def test_buffered_call_to_a_child_that_stops_reading_times_out(process_agent, monkeypatch, tmp_path):
    """An ACP child that answers session/new and then stops reading blocks the prompt write before
    the client's own deadline is checked; the request timeout must still end the call."""
    from providers import get_provider_profile

    server = tmp_path / "deaf_acp.py"
    server.write_text(_DEAF_AFTER_SESSION)
    spawned = []

    class _DeafClient(ACPClient):
        def _spawn(self):
            spawned.append(super()._spawn())
            return spawned[-1]

    monkeypatch.setattr(get_provider_profile(PROVIDER), "create_client", lambda **kwargs: _DeafClient(
        command=sys.executable, args=[str(server)], acp_cwd=str(tmp_path)))
    _write_config(f"providers:\n  {PROVIDER}:\n    request_timeout_seconds: 2\n")
    process_agent.model = "claude-haiku-5-5"
    outcome = {}

    def call():
        try:
            process_agent._interruptible_api_call(
                {"model": process_agent.model, "messages": [{"role": "user", "content": "x" * 4_000_000}]})
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=call, daemon=True)
    started = time.monotonic()
    worker.start()
    try:
        worker.join(30)
        assert not worker.is_alive(), "the buffered call to a deaf child never ended"
        assert isinstance(outcome.get("error"), TimeoutError)
        assert time.monotonic() - started < 30
    finally:
        for proc in spawned:
            proc.kill()
        worker.join(5)
