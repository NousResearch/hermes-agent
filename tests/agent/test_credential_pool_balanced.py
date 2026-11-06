"""The ``balanced`` pool strategy: sessions go where the quota is, and never lose a warm prompt cache.

Headers below are trimmed from live responses (Anthropic OAuth 200, ChatGPT Codex 200, 2026-10-17).
"""

import json
import time
from types import SimpleNamespace

import pytest

from agent import credential_pool_balanced as balanced
from agent.turn_credential_placement import place_credential_between_turns


@pytest.fixture(autouse=True)
def _fresh_process_view(monkeypatch):
    monkeypatch.setattr(balanced, "_observed", {})
    monkeypatch.setattr(balanced, "_warm", {})


@pytest.fixture
def pool_of(tmp_path, monkeypatch):
    def _seed(*ids):
        home = tmp_path / "hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "config.yaml").write_text("credential_pool_strategies:\n  anthropic: balanced\n")
        rows = [
            {"id": cid, "label": cid, "auth_type": "api_key", "priority": index, "source": "manual",
             "access_token": f"key-{cid}"}
            for index, cid in enumerate(ids)
        ]
        (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": rows}}))
        from agent.credential_pool import load_pool
        return load_pool("anthropic")
    return _seed


def _anthropic_headers(used_5h: float, *, org: str = "org-team") -> dict[str, str]:
    reset = int(time.time()) + 3600
    return {
        "anthropic-organization-id": org,
        "anthropic-ratelimit-unified-5h-utilization": str(used_5h),
        "anthropic-ratelimit-unified-5h-reset": str(reset),
        "anthropic-ratelimit-unified-5h-status": "allowed",
        "anthropic-ratelimit-unified-7d-utilization": "0.12",
        "anthropic-ratelimit-unified-7d-reset": str(reset + 5 * 86400),
        "anthropic-ratelimit-unified-status": "allowed",
    }


def _report(entry_id: str, used_5h: float, *, org: str = "org-team", session: str | None = None) -> None:
    balanced.observe("anthropic", entry_id, _anthropic_headers(used_5h, org=org),
                     session_id=session, warm_for=300 if session else 0)


def test_quota_windows_read_both_vendors_headers():
    codex = {
        "x-codex-plan-type": "pro",
        "x-codex-primary-used-percent": "37", "x-codex-primary-window-minutes": "10080",
        "x-codex-primary-reset-at": "1792885503",
        "x-codex-secondary-used-percent": "0", "x-codex-secondary-window-minutes": "0",
        "x-codex-secondary-reset-at": "",
    }

    anthropic_windows = balanced.quota_windows(_anthropic_headers(0.5))
    codex_windows = balanced.quota_windows(codex)

    assert {name: w.used for name, w in anthropic_windows.items()} == {"5h": 0.5, "7d": 0.12}
    assert codex_windows == {"10080m": balanced.QuotaWindow(used=0.37, resets_at=1792885503.0)}


def test_new_session_goes_to_the_most_headroom_spread_by_warm_sessions(pool_of):
    pool = pool_of("a", "b", "c")
    _report("a", 0.2, session="s1")
    _report("a", 0.2, session="s2")  # most headroom, but two sessions are already warm on it
    _report("b", 0.5)
    _report("c", 0.9)

    assert pool.select().id == "b"


def test_rotation_off_a_spent_seat_stays_in_its_cache_group(pool_of):
    pool = pool_of("a", "c", "b")  # priority order alone would rotate a -> c
    _report("a", 0.1, org="org-team")
    _report("b", 0.7, org="org-team")  # sibling seat: shares a's prompt cache
    _report("c", 0.4, org="org-personal")  # more headroom, but a full cache rewrite
    assert pool.select().id == "a"

    assert pool.mark_exhausted_and_rotate(status_code=429).id == "b"


def test_subagents_leased_together_spread_over_the_accounts_with_headroom(pool_of):
    pool = pool_of("a", "b", "c")
    _report("a", 0.95)  # first by priority, but nearly spent
    _report("b", 0.3)
    _report("c", 0.4)

    assert [pool.acquire_lease() for _child in range(3)] == ["b", "c", "a"]


def test_between_turns_a_cold_session_moves_and_a_warm_one_returns_to_its_cache(pool_of):
    pool = pool_of("a", "b")
    _report("a", 0.9)
    _report("b", 0.1)
    balanced.observe("anthropic", "a", {}, session_id="warm-here", warm_for=300)
    balanced.observe("anthropic", "a", {}, session_id="warm-there", warm_for=300)
    balanced.observe("anthropic", "a", {}, session_id="cold", warm_for=300, now=time.time() - 301)
    swaps: list[tuple[str, str]] = []

    # "warm-there" is a rebuilt agent that a fresh select() put on b while its cache is on a.
    for session_id, on in (("warm-here", "a"), ("warm-there", "b"), ("cold", "a")):
        agent = SimpleNamespace(
            _credential_pool=pool, _credential_pool_entry_id=on, session_id=session_id, model="claude-opus-4-7",
            base_url=None, _swap_credential=lambda entry, sid=session_id: swaps.append((sid, entry.id)),
        )
        place_credential_between_turns(agent)

    assert swaps == [("warm-there", "a"), ("cold", "b")]


def test_sessions_placed_together_spread_before_either_reports_back(pool_of):
    pool = pool_of("a", "b", "c")
    _report("a", 0.5)
    _report("b", 0.4)
    _report("c", 0.45)
    swaps: list[tuple[str, str]] = []

    for session_id in ("s1", "s2"):  # no response in between: the first placement alone must count
        agent = SimpleNamespace(
            _credential_pool=pool, _credential_pool_entry_id="a", session_id=session_id, provider="anthropic",
            model="claude-opus-4-7", base_url=None, _use_prompt_caching=True, _cache_ttl="5m",
            _swap_credential=lambda entry, sid=session_id: swaps.append((sid, entry.id)),
        )
        place_credential_between_turns(agent)

    assert swaps == [("s1", "b"), ("s2", "c")]


def test_only_a_balanced_pool_records_responses():
    def _agent(pool_balanced: bool, session_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            provider="openai-codex", session_id=session_id, _credential_pool_entry_id="codex-1",
            _credential_pool=SimpleNamespace(provider="openai-codex", balanced=pool_balanced),
        )
    response = SimpleNamespace(headers={
        "x-codex-primary-used-percent": "37", "x-codex-primary-window-minutes": "10080",
        "x-codex-primary-reset-at": "1792885503"})

    balanced.observe_agent_response(_agent(False, "fill-first"), response)
    assert balanced._observed == {} and balanced.warm_entry("openai-codex", "fill-first") is None

    balanced.observe_agent_response(_agent(True, "balanced"), response)
    assert balanced._observed[("openai-codex", "codex-1")].windows["10080m"].used == 0.37
    assert balanced.warm_entry("openai-codex", "balanced") == "codex-1"


def test_a_bigger_plan_takes_load_in_proportion_without_a_plan_table():
    """Headroom is a fraction, so a 20x account fills four times slower than a 5x and keeps
    winning new sessions until the percentages meet: no plan-size table to maintain."""
    capacity = {"max20x": 4.0, "max5x": 1.0}
    spent = dict.fromkeys(capacity, 0.0)
    entries = [SimpleNamespace(id=entry_id) for entry_id in capacity]
    for _session in range(50):
        for entry_id, cap in capacity.items():
            balanced.observe("anthropic", entry_id, {"anthropic-ratelimit-unified-5h-utilization": str(spent[entry_id] / cap / 100)})
        spent[balanced.choose("anthropic", entries).id] += 1

    assert spent["max20x"] / spent["max5x"] == pytest.approx(4, rel=0.15)
