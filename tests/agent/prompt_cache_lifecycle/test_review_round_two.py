"""Second-round review invariants (alt-glitch on 893b54f9), one each:

- Hermes' own managed llama-server unloads an idle model (and its prompt cache): its cache is certainly
  cold once the unload has run, while a user's own local server never expires on the clock.
- A malformed entry in ``cache/prompt_cache_routes.json`` is dropped, never raised into agent build or
  usage accounting.
- Two local servers on one host (different ports) learn separately.
- A late keep-alive miss that lands after its run ended with no successor still decides step 2/3.
- The compaction hand-off carries a real keep-alive miss to the child session.
- A finite admission (``chat -q`` / ``-z`` through the gateway) arms no idle keep-alive.
"""

import json
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl
import agent.prompt_cache_routes as routes
from tests.agent.prompt_cache_lifecycle.test_backtest_rulings import _Client, _chat, _fire, _respond
from tests.agent.prompt_cache_lifecycle.test_backtest_rulings import _setup as _backtest_setup

_setup = _backtest_setup  # the autouse fixture: pricing, inert timers, a scratch home

_LOCAL = "http://127.0.0.1:8080/v1"


def _record_write(agent, at):
    from agent.prompt_cache_clock import record_cache_response
    from agent.usage_pricing import CanonicalUsage

    agent._prompt_cache_attempt = {"sent_at": at, "marker_ttl": None}
    return record_cache_response(agent, CanonicalUsage(input_tokens=3, cache_read_tokens=50_000), model=agent.model,
                                 provider=agent.provider, base_url=agent.base_url, api_mode="chat_completions")


@pytest.mark.parametrize("managed", [True, False])
def test_a_managed_local_model_goes_cold_after_its_unload_and_a_users_own_server_never(monkeypatch, managed):
    from agent.prompt_cache_clock import cache_certainly_cold, cache_certainly_warm, cache_window
    from hermes_cli.local_runtime import growth

    monkeypatch.setattr(growth, "is_managed_endpoint", lambda base_url: managed and base_url == _LOCAL)
    agent = SimpleNamespace(provider="custom", base_url=_LOCAL, model="qwen3.8-27b", api_mode="chat_completions",
                            session_id="s", _session_db=None, _prompt_cache_clock_restored=True,
                            context_compressor=SimpleNamespace(compression_count=0))
    t0 = 1_000_000.0
    _record_write(agent, t0)
    window = cache_window(agent)
    assert cache_certainly_warm(window, now=t0 + 5 * 60)
    if managed:  # unloaded after 15 min idle (+ two 120 s sweeps): /context and the offload see it
        assert not cache_certainly_warm(window, now=t0 + 20 * 60)
        assert cache_certainly_cold(window, now=t0 + 20 * 60)
    else:
        assert not cache_certainly_cold(window, now=t0 + 24 * 3600)


def test_a_malformed_route_store_entry_is_dropped_never_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "cache" / "prompt_cache_routes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        routes.route_key("nous", "https://x/v1", "m", True): {"updated": "x"},
        routes.route_key("nous", "https://x/v1", "n", True): {"updated": 1e12, "miss_gaps": ["a"]},
        routes.route_key("nous", "https://x/v1", "o", True): {"updated": 1e12, "reads": "lots"},
    }), encoding="utf-8")
    for model in ("m", "n", "o"):
        assert routes.rule_for("nous", "https://x/v1", model, True, tier_s=3600, now=1e12) is None
        routes.observe(provider="nous", base_url="https://x/v1", model=model, explicit=True, read=0, write=0,
                       gap_s=10, tier_s=3600, now=1e12, session_id="s")


def test_two_local_servers_on_one_host_learn_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for session in ("a", "b"):  # llama.cpp on :8080 never reads back, twice
        routes.observe(provider="custom", base_url="http://127.0.0.1:8080/v1", model="qwen", explicit=False,
                       read=0, write=0, gap_s=10, tier_s=1800, now=1e6, session_id=session)
    assert routes.rule_for("custom", "http://127.0.0.1:8080/v1", "qwen", False, now=1e6) == "no-cache"
    assert routes.rule_for("custom", "http://127.0.0.1:11434/v1", "qwen", False, now=1e6) is None


# ── warmer: late miss, compaction hand-off ──────────────────────────────────────────────────────────



def test_a_late_miss_after_its_run_ended_with_no_successor_still_decides_step_two():
    """The keep-alive is on the wire when the turn settles (no new run), and its reply is the miss that
    ends warming: the user turn's budget is over, so step 2 decides on the prompt the turn holds."""
    client = _Client([("chatcmpl-x", 0, 119_950)])
    agent = _chat(client=client)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    warmer._miss_streak = (1, 1)  # an earlier real miss was retried
    run = warmer._run
    real_send = warmer._send

    def _send_then_settle(r):
        reply = real_send(r)
        warmer._clear_run()  # the run ended (turn settled / cancelled) while the reply was in flight
        return reply

    warmer._send = _send_then_settle
    warmer._refresh(run)
    assert warmer._run is None and warmer._spent is not None
    assert warmer.edge_decision(now=warmer._spent.due_at).compact == "due"


def test_the_compaction_hand_off_carries_a_real_miss_to_the_child_session():
    from agent.compression_automation import carry_compression_automation

    client = _Client([("chatcmpl-x", 0, 119_950)] * 2)
    agent = _chat(client=client, session_id="parent", _session_db=None)
    warmer = pcl.get_prompt_cache_warmer(agent)
    assert _respond(agent) is None
    agent._executing_tools = True
    _fire(warmer)
    _fire(warmer)  # two real misses in a row
    assert pcl.prompt_cache_warming_disabled(agent)
    agent.session_id = "child"  # the compaction moved the chat
    carry_compression_automation(agent, "parent")
    assert pcl.get_prompt_cache_warmer(_chat(session_id="child")).warming_disabled


def test_a_finite_admission_arms_no_idle_keep_alive():
    from gateway.session_finite import finite_turn_scope

    agent = _chat(client=_Client([]))
    with finite_turn_scope(True, unattended=False):
        assert _respond(agent) is None  # a tool wait would still be warmed
        assert pcl.get_prompt_cache_warmer(agent)._run.idle_ok is False
    pcl.end_turn_prompt_cache_warming(agent)
    assert pcl.get_prompt_cache_warmer(agent)._run is None
