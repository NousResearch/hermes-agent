"""``agent.reconnect_attention_after`` is read from the profile whose scope is bound at call time.

Regression for #115635: the threshold was a module constant frozen from ``os.environ`` at import,
so a multiplexed secondary escalated at the LAUNCH profile's value and a live config edit needed a
gateway restart. Two homes, A -> B -> A, per AGENTS.md § "One process may serve many profiles".
"""
import asyncio
import time

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.run import GatewayRunner


def _home_with_threshold(root, name, seconds):
    home = root / name
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"reconnect_attention_after": seconds}}), encoding="utf-8")
    return home


def test_threshold_follows_bound_profile_scope_a_b_a(tmp_path, monkeypatch):
    home_a = _home_with_threshold(tmp_path, "a", 10)
    home_b = _home_with_threshold(tmp_path, "b", 100000)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    now = time.monotonic()
    queued_20s_ago = {"queued_at": now - 20}

    assert gateway_run._reconnect_needs_attention(dict(queued_20s_ago), now) is True
    with gateway_run._profile_runtime_scope(home_b, hydrate_secrets=False):
        assert gateway_run._reconnect_needs_attention(dict(queued_20s_ago), now) is False
    assert gateway_run._reconnect_needs_attention(dict(queued_20s_ago), now) is True

    # Live edit of the bound profile's config takes effect on the next call, no restart.
    (home_a / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"reconnect_attention_after": 100000}}), encoding="utf-8")
    assert gateway_run._reconnect_needs_attention(dict(queued_20s_ago), now) is False


@pytest.mark.asyncio
async def test_secondary_reconnect_loop_escalates_under_own_profile(tmp_path, monkeypatch):
    """A secondary profile's reconnect loop flags ``<profile>:<platform>`` NEEDS_ATTENTION at ITS
    threshold — the launch profile's (100000 here) must not suppress it."""
    launch = _home_with_threshold(tmp_path, "launch", 100000)
    secondary = _home_with_threshold(tmp_path, "sec", 0.01)
    monkeypatch.setenv("HERMES_HOME", str(launch))

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw)))

    class _RetryableAdapter:
        has_fatal_error = True
        fatal_error_retryable = True

    async def failing_attempt(profile_name, platform):
        return _RetryableAdapter(), False

    async def noop_disconnect(adapter, platform):
        return None

    monkeypatch.setattr(runner, "_secondary_reconnect_attempt", failing_attempt)
    monkeypatch.setattr(runner, "_safe_adapter_disconnect", noop_disconnect)
    monkeypatch.setattr(runner, "_routed_profile_home", lambda name: secondary)
    monkeypatch.setattr(gateway_run, "_reconnect_backoff", lambda attempts: 0.02)

    task = asyncio.create_task(runner._run_secondary_profile_reconnect("sec", Platform.DISCORD))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(kw.get("needs_attention") for _k, kw in writes):
        await asyncio.sleep(0.02)
    runner._running = False
    await asyncio.wait_for(task, timeout=2)

    flagged = [(key, kw) for key, kw in writes if kw.get("needs_attention")]
    assert [key for key, _kw in flagged] == ["sec:discord"], writes
    assert flagged[0][1]["platform_state"] == "retrying" and flagged[0][1].get("retrying_since")
