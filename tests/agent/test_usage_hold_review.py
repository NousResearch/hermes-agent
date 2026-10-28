"""Hold bookkeeping contracts: whose deadline, which models, which endpoint, and that the
hold file can neither lose a sibling process's hold nor change a turn's outcome."""

from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from agent import usage_hold
from agent.quota_resume import plan_for_failed_turn, plan_quota_resume

NOW = 1_800_000_000.0
HOUR = 3600.0


@dataclass(frozen=True)
class _Entry:
    last_status: str
    last_error_reset_at: Optional[object] = None


class _Pool:
    def __init__(self, entries):
        self._entries = entries

    def has_available(self):
        return False

    def next_available_at(self):
        return NOW + HOUR  # the pool's own retry schedule, including its local cooldown guess

    def entries(self):
        return list(self._entries)


@dataclass(frozen=True)
class _Window:
    label: str
    used_percent: float
    reset_at: float
    scope: str = "account"


@dataclass(frozen=True)
class _Snapshot:
    windows: tuple
    available: bool = True


def test_pool_cooldown_without_a_reported_reset_is_not_a_deadline():
    """An exhausted credential with no reported reset only has the pool's own retry guess."""
    pool = _Pool([_Entry("exhausted", None), _Entry("exhausted", NOW + 2 * HOUR)])
    plan = plan_quota_resume(failure_reason="rate_limit", error_context={}, provider="anthropic",
                             credential_pool=pool, now=NOW, allow_usage_api=False)
    assert plan.resume_at is None and plan.eligible is False


def test_model_family_window_never_holds_other_models(monkeypatch, tmp_path):
    from agent import account_usage

    seen = {}

    def _fake(provider, **kwargs):
        seen.update(kwargs)
        return _Snapshot(windows=(_Window("Opus week", 100.0, NOW + 5 * HOUR, scope="model"),
                                  _Window("Current week", 40.0, NOW + 50 * HOUR)))

    monkeypatch.setattr(account_usage, "fetch_account_usage", _fake)
    sonnet = plan_quota_resume(failure_reason="rate_limit", error_context={}, provider="anthropic",
                               model="sonnet-4-6", now=NOW, grace_seconds=0.0)
    assert sonnet.resume_at is None
    opus = plan_quota_resume(failure_reason="rate_limit", error_context={}, provider="anthropic",
                             model="opus-4-6", now=NOW, grace_seconds=0.0)
    assert (opus.resume_at, opus.scope) == (NOW + 5 * HOUR, "model")

    # The Opus-only deadline resumes that turn but parks no other anthropic work.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    usage_hold.record_from_plan(
        {**opus.to_dict(), "resume_at": time.time() + HOUR}, provider="anthropic")
    assert usage_hold.active_hold("anthropic") is None


def test_usage_lookup_uses_the_failing_routes_credential(monkeypatch, tmp_path):
    from agent import account_usage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen = {}

    def _fake(provider, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(account_usage, "fetch_account_usage", _fake)
    agent = SimpleNamespace(_credential_pool=None, model="gpt-5", api_key="sk-the-failing-one")
    plan_for_failed_turn(agent, SimpleNamespace(reason="rate_limit"), error_context={},
                         provider="openai-codex", base_url="https://codex.example/backend")
    assert seen.get("api_key") == "sk-the-failing-one"
    assert seen.get("base_url") == "https://codex.example/backend"


def test_endpoint_path_case_is_a_distinct_account():
    assert usage_hold.hold_key("custom", "HTTP://Host.Example/TeamA/v1/") == "custom:http://host.example/TeamA/v1"
    assert usage_hold.hold_key("custom", "http://host.example/TeamA/v1") != usage_hold.hold_key(
        "custom", "http://host.example/teama/v1")


def test_unwritable_hold_file_never_changes_the_turn_outcome(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    usage_hold.record_hold("anthropic", until=time.time() + HOUR)

    def _disk_full(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(usage_hold, "atomic_json_write", _disk_full)
    agent = SimpleNamespace(_turn_received_provider_response=True, provider="anthropic", base_url="")
    usage_hold.release_after_turn(agent, completed=True, failed=False)  # must not raise
    usage_hold.record_from_plan({"reason": "rate_limit", "resume_at": time.time() + HOUR},
                                provider="openai-codex")  # must not raise


def _record_many(home: str, prefix: str, count: int) -> None:
    import os

    os.environ["HERMES_HOME"] = home
    from agent import usage_hold as uh

    for i in range(count):
        uh.record_hold(f"{prefix}{i}", until=time.time() + HOUR)


def test_concurrent_processes_never_drop_each_others_holds(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_record_many, args=(str(tmp_path), p, 25)) for p in ("a", "b", "c")]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0
    stored = json.loads((tmp_path / "usage_holds.json").read_text(encoding="utf-8-sig"))
    assert len(stored) == 75
