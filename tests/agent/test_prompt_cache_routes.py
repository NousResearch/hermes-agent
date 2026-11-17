"""Hermes learns how each route really caches, from the usage it already receives (no probes).

Invariants: (a) never read back after two attempts -> no-cache: no keep-alives, and an explicit route
stops sending markers; (b) two hits that re-wrote the prefix -> no keep-alives; (c) two early misses ->
keep-alives on the learned lifetime (never past the tier), or none when that is under a minute; (d) an
automatic cache that dies early while the explicit one reads -> prefer markers. One stray miss never
downgrades; facts expire after seven days; each profile learns on its own."""

from types import SimpleNamespace

import pytest

import agent.prompt_cache_routes as routes
from agent.prompt_cache_lifecycle import learned_window

_NOUS = "https://inference-api.nousresearch.com/v1"
_T0 = 1_000_000.0


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    return tmp_path


def _obs(model, *, explicit=True, read=0, write=0, gap=10.0, tier=3600.0, at=_T0):
    routes.observe(provider="nous", base_url=_NOUS, model=model, explicit=explicit, read=read, write=write,
                   gap_s=gap, tier_s=tier, now=at)


def _agent(model):
    return SimpleNamespace(provider="nous", base_url=_NOUS, model=model)


def test_a_a_route_that_never_reads_back_is_learned_no_cache_and_loses_its_markers(home, monkeypatch):
    monkeypatch.setattr(routes.time, "time", lambda: _T0)
    _obs("openai/gpt-5.6-luna", write=6013)  # write, read back 2 s later: wrote again
    assert routes.rule_for("nous", _NOUS, "openai/gpt-5.6-luna", True) is None  # one sample decides nothing
    _obs("openai/gpt-5.6-luna", write=6013)
    assert routes.rule_for("nous", _NOUS, "openai/gpt-5.6-luna", True) == "no-cache"
    assert learned_window(_agent("openai/gpt-5.6-luna"), (_T0 + 3600, 3600.0), True) == (None, "no-cache")
    assert routes.marker_preference("nous", _NOUS, "openai/gpt-5.6-luna") is False
    # its requests now go without markers; that automatic cache is not warmed until it reads back
    assert learned_window(_agent("openai/gpt-5.6-luna"), (_T0 + 1800, 1800.0), False) == (None, "no-cache")
    assert (home / "a" / "cache" / "prompt_cache_routes.json").is_file()


def test_b_a_route_that_rewrites_on_every_hit_gets_no_keep_alives(home, monkeypatch):
    monkeypatch.setattr(routes.time, "time", lambda: _T0)
    for _ in range(2):
        _obs("google/gemini-3.5-flash", read=8582, write=8582)
    assert learned_window(_agent("google/gemini-3.5-flash"), (_T0 + 3600, 3600.0), True) == (None, "write-on-hit")
    _obs("qwen/qwen3.7-max", read=30_597, write=14)  # a normal hit writes only the new tail
    _obs("qwen/qwen3.7-max", read=30_597, write=14)
    assert routes.rule_for("nous", _NOUS, "qwen/qwen3.7-max", True) is None


def test_c_two_early_misses_move_keep_alives_onto_the_learned_lifetime(home, monkeypatch):
    monkeypatch.setattr(routes.time, "time", lambda: _T0)
    _obs("qwen/qwen3.7-max", explicit=False, read=28_800, gap=2, tier=1800)
    _obs("qwen/qwen3.7-max", explicit=False, write=0, gap=240, tier=1800)
    _obs("qwen/qwen3.7-max", explicit=False, write=0, gap=300, tier=1800)
    deadline = _T0 + 1800
    window, rule = learned_window(_agent("qwen/qwen3.7-max"), (deadline, 1800.0), False)
    assert rule == "lifetime:240" and window == (deadline - 1560, 240.0)  # fires at 5/6 of 240 s, not 25 min

    _obs("qwen/qwen3.5-397b", explicit=False, read=20_000, gap=2, tier=1800)
    for gap in (40, 45):  # a lifetime whose keep-alive would be due in under a minute is not warmed
        _obs("qwen/qwen3.5-397b", explicit=False, write=0, gap=gap, tier=1800)
    assert learned_window(_agent("qwen/qwen3.5-397b"), (deadline, 1800.0), False) == (None, "short-lived")


def test_d_an_early_dying_automatic_cache_with_a_reading_explicit_one_prefers_markers(home, monkeypatch):
    monkeypatch.setattr(routes.time, "time", lambda: _T0)
    for gap in (240, 260):
        _obs("qwen/qwen3.7-plus", explicit=False, gap=gap, tier=1800)
    assert routes.marker_preference("nous", _NOUS, "qwen/qwen3.7-plus") is None  # explicit not proven yet
    _obs("qwen/qwen3.7-plus", explicit=True, read=8785, write=14, gap=780)
    assert routes.marker_preference("nous", _NOUS, "qwen/qwen3.7-plus") is True


def test_a_single_miss_never_downgrades_a_route(home, monkeypatch):
    monkeypatch.setattr(routes.time, "time", lambda: _T0)
    _obs("qwen/qwen3.7-plus", read=8585, gap=2)
    _obs("qwen/qwen3.7-plus", write=8585, gap=2)  # one stray miss at 2 s (routing)
    _obs("qwen/qwen3.7-plus", read=8585, gap=200)
    assert routes.rule_for("nous", _NOUS, "qwen/qwen3.7-plus", True, tier_s=3600) is None
    assert learned_window(_agent("qwen/qwen3.7-plus"), (_T0 + 3600, 3600.0), True) == ((_T0 + 3600, 3600.0), None)


def test_learned_facts_expire_after_seven_days(home, monkeypatch):
    for _ in range(2):
        _obs("openai/gpt-5.6-luna", write=6013)
    monkeypatch.setattr(routes.time, "time", lambda: _T0 + routes.EXPIRY_SECONDS - 1)
    assert routes.rule_for("nous", _NOUS, "openai/gpt-5.6-luna", True) == "no-cache"
    monkeypatch.setattr(routes.time, "time", lambda: _T0 + routes.EXPIRY_SECONDS + 1)
    assert routes.rule_for("nous", _NOUS, "openai/gpt-5.6-luna", True) is None


def test_each_profile_learns_on_its_own(home, monkeypatch):
    monkeypatch.setattr(routes.time, "time", lambda: _T0)
    for _ in range(2):
        _obs("openai/gpt-5.6-luna", write=6013)
    assert routes.rule_for("nous", _NOUS, "openai/gpt-5.6-luna", True) == "no-cache"
    monkeypatch.setenv("HERMES_HOME", str(home / "b"))  # another profile, same process
    assert routes.rule_for("nous", _NOUS, "openai/gpt-5.6-luna", True) is None
    assert not (home / "b" / "cache" / "prompt_cache_routes.json").exists()


def test_the_clock_feeds_the_store_from_real_responses(home, monkeypatch):
    """Two real loop responses on a never-reading explicit route: the clock folds each and the store
    learns from the second (the first is a write with nothing before it)."""
    from agent.prompt_cache_clock import record_cache_response
    from agent.usage_pricing import CanonicalUsage

    agent = SimpleNamespace(provider="nous", base_url=_NOUS, model="openai/gpt-5.6-luna", api_mode="chat_completions",
                            session_id="s", _session_db=None, _prompt_cache_clock_restored=True,
                            context_compressor=SimpleNamespace(compression_count=0))
    for at in (_T0, _T0 + 5, _T0 + 10):
        agent._prompt_cache_attempt = {"sent_at": at, "marker_ttl": "1h"}
        record_cache_response(agent, CanonicalUsage(input_tokens=3, cache_write_tokens=6013), model=agent.model,
                              provider="nous", base_url=_NOUS, api_mode="chat_completions")
    monkeypatch.setattr(routes.time, "time", lambda: _T0 + 20)
    assert routes.rule_for("nous", _NOUS, "openai/gpt-5.6-luna", True) == "no-cache"
