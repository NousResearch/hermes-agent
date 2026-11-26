"""Second-round review invariants (alt-glitch on 893b54f9), one each:

- Hermes' own managed llama-server unloads an idle model (and its prompt cache): its cache is certainly
  cold once the unload has run, while a user's own local server never expires on the clock.
- A malformed entry in ``cache/prompt_cache_routes.json`` is dropped, never raised into agent build or
  usage accounting.
- Two local servers on one host (different ports) learn separately.
- A late keep-alive miss that lands after its run ended with no successor still decides step 2/3.
- The compaction hand-off carries a real keep-alive miss to the child session.
- A finite admission (``chat -q`` / ``-z`` through the gateway) arms no idle keep-alive.
- A retried miss that reported no cache write (the clock did not move) waits a full window, never resends at once.
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
    monkeypatch.setattr("agent.prompt_cache_clock.time.time", lambda: t0)  # the reply finished as it was sent
    _record_write(agent, t0)
    window = cache_window(agent)
    assert cache_certainly_warm(window, now=t0 + 5 * 60)
    if managed:  # unloaded after 15 min idle (+ two 120 s sweeps): /context and the offload see it
        assert not cache_certainly_warm(window, now=t0 + 20 * 60)
        assert cache_certainly_cold(window, now=t0 + 20 * 60)
    else:
        assert not cache_certainly_cold(window, now=t0 + 24 * 3600)


def test_a_managed_local_models_unload_is_counted_from_when_its_reply_finished(monkeypatch):
    """The managed server's 15 min idle wait starts when the model goes idle: a reply that took ten
    minutes is still warm nine minutes after it finished, and cold once the unload (+ two sweeps) ran."""
    import agent.prompt_cache_clock as clock
    from hermes_cli.local_runtime import growth

    monkeypatch.setattr(growth, "is_managed_endpoint", lambda base_url: base_url == _LOCAL)
    agent = SimpleNamespace(provider="custom", base_url=_LOCAL, model="qwen3.8-27b", api_mode="chat_completions",
                            session_id="s", _session_db=None, _prompt_cache_clock_restored=True,
                            context_compressor=SimpleNamespace(compression_count=0))
    t0 = 1_000_000.0
    monkeypatch.setattr(clock.time, "time", lambda: t0 + 10 * 60)  # the reply finished ten minutes after the send
    _record_write(agent, t0)
    window = clock.cache_window(agent)
    assert not clock.cache_certainly_cold(window, now=t0 + 19 * 60)
    assert clock.cache_certainly_cold(window, now=t0 + 30 * 60)


def test_a_malformed_route_store_entry_is_dropped_never_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "cache" / "prompt_cache_routes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        routes.route_key("nous", "https://x/v1", "m", True): {"updated": "x"},
        routes.route_key("nous", "https://x/v1", "n", True): {"updated": 1e12, "miss_gaps": ["a"]},
        routes.route_key("nous", "https://x/v1", "o", True): {"updated": 1e12, "reads": "lots"},
    }), encoding="utf-8")
    # Valid JSON that decodes to infinity / NaN (``1e999``, ``NaN``): int() of either raises in rule_for.
    raw = path.read_text(encoding="utf-8")[:-1] + ", " + ", ".join(
        f'"{routes.route_key("nous", "https://x/v1", model, True)}": {body}' for model, body in (
            ("p", '{"updated": 1e12, "attempts": 1e999, "reads": 0, "miss_gaps": [5, 5], "miss_gaps_sessions": ["a", "b"]}'),
            ("q", '{"updated": 1e12, "write_on_hit": NaN}'),
            ("r", '{"updated": 1e12, "hit_gaps": [1e999]}'),
            ("s", '{"updated": 1e999}'))) + "}"
    path.write_text(raw, encoding="utf-8")
    for model in ("m", "n", "o", "p", "q", "r", "s"):
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


def _retried_miss_run(monkeypatch):
    """Fire one keep-alive that misses on an automatic cache (no write counter) and return (run, sent_before)."""
    import time

    from agent.prompt_cache_clock import record_cache_response
    from agent.usage_pricing import CanonicalUsage

    class _CC:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        @staticmethod
        def effective_output_cap(*, max_tokens=None, max_completion_tokens=None, **_):
            return max_completion_tokens or max_tokens

        def with_options(self, **kw):
            return self

        def _create(self, **kw):
            self.calls.append(kw)
            usage = SimpleNamespace(prompt_tokens=120_000, completion_tokens=1, total_tokens=120_001,
                                    prompt_tokens_details=SimpleNamespace(cached_tokens=0))
            return SimpleNamespace(id="chatcmpl-x", usage=usage, choices=[])

    client = _CC()
    agent = _chat(api_mode="chat_completions", client=client, provider="custom", base_url="https://gw.example/v1",
                  model="deepseek-v4")
    agent.client = client
    t0 = time.time() - 1000  # the cached reply landed ~17 min ago: the 30m window is still guaranteed
    agent._prompt_cache_attempt = {"sent_at": t0, "marker_ttl": None}
    record_cache_response(agent, CanonicalUsage(input_tokens=50, cache_read_tokens=120_000), model=agent.model,
                          provider=agent.provider, base_url=agent.base_url, api_mode="chat_completions")
    request = {"model": "deepseek-v4", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
    response = SimpleNamespace(id="chatcmpl-0", usage=SimpleNamespace(
        prompt_tokens=120_050, completion_tokens=5, total_tokens=120_055,
        prompt_tokens_details=SimpleNamespace(cached_tokens=120_000)))
    assert pcl.get_prompt_cache_warmer(agent).start(request, response) is None, pcl.get_prompt_cache_warmer(agent).status
    warmer = pcl.get_prompt_cache_warmer(agent)
    agent._executing_tools = True
    before = time.time()
    _fire(warmer)  # the keep-alive misses (read 0): retried once
    run = warmer._run
    assert len(client.calls) == 1 and run is not None
    return run, before


def test_a_retried_miss_with_no_write_counter_waits_a_window_before_the_next_send(monkeypatch):
    """An automatic (OpenAI-spec) cache reports a miss as plain input: no read, no write, so the clock does not
    move. The retry must wait a full window from the miss, not fire at once off the old deadline."""
    run, before = _retried_miss_run(monkeypatch)
    assert run.next_warm_at >= before + 0.8 * pcl.cache_warming_delay_s(run.ttl_s), "the retry fires at once"


def test_a_retried_miss_keeps_a_shorter_lifetime_learned_at_that_miss(monkeypatch):
    """The route store learns a 10-minute lifetime from this very miss: the retry warms on 10 minutes, never on the
    tier it just replaced (a 25-minute wait would land after the entry is gone)."""
    from agent.prompt_cache_lifecycle import warmer as warmer_mod

    real = warmer_mod.learned_window
    learned = {"on": False}

    def _learned(agent, window, marked):
        if not learned["on"]:
            return real(agent, window, marked)
        deadline, tier_s = window
        return (deadline - (tier_s - 600.0), 600.0), "lifetime:600"

    monkeypatch.setattr(warmer_mod, "learned_window", _learned)
    real_miss = warmer_mod.PromptCacheWarmer._miss_ends_warming

    def _miss(self, run, message):
        learned["on"] = True  # this miss supplies the second session's sample
        return real_miss(self, run, message)

    monkeypatch.setattr(warmer_mod.PromptCacheWarmer, "_miss_ends_warming", _miss)
    run, before = _retried_miss_run(monkeypatch)
    assert run.ttl_s == 600.0
    assert run.next_warm_at <= before + pcl.cache_warming_delay_s(600.0) + 5, "the retry waits on the replaced tier"
