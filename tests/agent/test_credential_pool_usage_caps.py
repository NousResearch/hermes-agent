"""Soft usage caps: a session leaves a subscription login at its cap, and the check never holds up a request."""
import threading
import time
from types import SimpleNamespace

import pytest

from agent import credential_pool_usage_caps as caps
from hermes_cli.auth import write_credential_pool

WEEK = time.time() + 3 * 86400


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # no login is auto-discovered from the real home
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for state in (caps._cache, caps._warned_at, caps._guard_stuck):
        state.clear()
    monkeypatch.setattr(caps, "_guard_open_until", 0.0)
    _config(monkeypatch, seven_day=80)


def _config(monkeypatch, **section):
    from agent import credential_pool

    monkeypatch.setattr(credential_pool, "_load_config_safe",
                        lambda: {"credential_pool_usage_caps": {"anthropic": section}})


def _usage(monkeypatch, **used):
    monkeypatch.setattr(caps, "read_usage", lambda entry: (
        {"seven_day": (used[entry.label], WEEK)} if used.get(entry.label) is not None else None))


def _session(*labels):
    from agent.credential_pool import load_pool

    write_credential_pool("anthropic", [
        dict(id=f"row-{label}", label=label, source="manual:hermes_pkce", auth_type="oauth", priority=index,
             access_token=f"fixture-access-{label}", refresh_token=f"fixture-refresh-{label}",
             expires_at_ms=int((time.time() + 8 * 3600) * 1000)) for index, label in enumerate(labels)])
    pool = load_pool("anthropic")
    agent = SimpleNamespace(_credential_pool=pool, model="claude-opus-5-5", _credential_pool_revert_id=None,
                            _credential_pool_entry_id=f"row-{labels[0]}", swaps=[])
    agent._swap_credential = lambda target: agent.swaps.append(target.label) or True
    return agent, pool


def _status(pool, label):
    return next(entry for entry in pool.entries() if entry.label == label)


def test_a_login_at_its_cap_is_benched_until_the_window_resets_and_the_session_moves_on(monkeypatch):
    agent, pool = _session("main", "backup")
    _usage(monkeypatch, main=81.0, backup=10.0)
    assert caps.enforce_usage_caps(agent) is True
    assert agent.swaps == ["backup"] and agent._credential_pool_revert_id == "row-main"
    benched = _status(pool, "main")
    assert benched.last_status == "exhausted" and benched.last_error_reason == "usage_cap"
    assert benched.last_error_reset_at == WEEK
    assert pool.select(model="claude-opus-5-5").label == "backup"


def test_under_the_cap_unknown_usage_and_an_unconfigured_provider_change_nothing(monkeypatch):
    agent, pool = _session("main", "backup")
    _usage(monkeypatch, main=79.0, backup=10.0)
    assert caps.enforce_usage_caps(agent) is False
    _usage(monkeypatch, main=None, backup=10.0)
    assert caps.enforce_usage_caps(agent) is False
    _usage(monkeypatch, main=99.0, backup=10.0)
    from agent import credential_pool

    monkeypatch.setattr(credential_pool, "_load_config_safe", dict)
    assert caps.enforce_usage_caps(agent) is False
    assert agent.swaps == [] and _status(pool, "main").last_status != "exhausted"


def test_a_per_account_cap_overrides_the_provider_cap_and_nothing_is_benched_when_all_are_capped(monkeypatch):
    _config(monkeypatch, seven_day=80, accounts={"backup": {"seven_day": 50}})
    agent, pool = _session("main", "backup")
    _usage(monkeypatch, main=90.0, backup=60.0)  # backup is under 80 but over its own 50
    assert caps.enforce_usage_caps(agent) is False
    assert agent.swaps == [] and _status(pool, "main").last_status != "exhausted"


def test_a_hanging_usage_read_costs_the_cap_not_the_request(monkeypatch):
    monkeypatch.setattr(caps, "_GUARD_BUDGET_SECONDS", 0.3)
    monkeypatch.setattr(caps, "_GUARD_SETTLE_SECONDS", 0.0)
    agent, _pool = _session("main", "backup")
    release, reads = threading.Event(), []

    def hang(entry):
        reads.append(entry.label)
        release.wait(30)
        return {"seven_day": (99.0 if entry.label == "main" else 10.0, WEEK)}

    monkeypatch.setattr(caps, "read_usage", hang)
    started = time.monotonic()
    assert caps.enforce_usage_caps(agent) is False
    assert time.monotonic() - started < 2.0  # the budget, not the hang
    assert caps.enforce_usage_caps(agent) is False and reads == ["main"]  # switched off: no second helper
    release.set()  # the read was only slow: the check switches itself back on
    deadline = time.monotonic() + 5
    while caps.guard_state() != {"off_for": 0.0, "stuck": 0} and time.monotonic() < deadline:
        time.sleep(0.02)
    assert agent.swaps == []  # the late decision was dropped
    assert caps.enforce_usage_caps(agent) is True and agent.swaps == ["backup"]


def test_concurrent_turns_through_the_real_usage_read_and_cache_never_block(monkeypatch):
    from agent import account_usage

    monkeypatch.setattr(caps, "_CACHE_TTL_SECONDS", 0.02)  # cache hits and misses both happen
    window = SimpleNamespace(label="Current week", used_percent=10.0, reset_at=None)
    monkeypatch.setattr(account_usage, "_fetch_anthropic_account_usage",
                        lambda api_key=None: SimpleNamespace(windows=[window]))
    agent, _pool = _session("main", "backup")
    finished = []

    def turn(index):
        for _ in range(40):
            caps.enforce_usage_caps(agent)
        finished.append(index)

    threads = [threading.Thread(target=turn, args=(index,), daemon=True) for index in range(12)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    assert len(finished) == 12 and caps.guard_state() == {"off_for": 0.0, "stuck": 0}
