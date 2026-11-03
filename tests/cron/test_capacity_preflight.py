"""Opt-in cron subscription preflight; no live provider calls."""
from datetime import UTC, datetime, timedelta

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from cron.capacity_preflight import select_capacity_route, _usage_snapshot
from cron.scheduler import _CronJobConfig, _resolve_job_runtime
from unittest.mock import patch


def test_cron_resolver_preflights_only_opted_in_unpinned_job():
    chain = [{"provider": "openai-codex", "model": "gpt-6-sol"}]
    def resolve(**kwargs):
        return {"provider": kwargs.get("requested") or "anthropic", "api_key": "token", "base_url": ""}
    usage = {"anthropic": snap("anthropic", window("Current week", 98)),
             "openai-codex": snap("openai-codex", window("Session", 20), window("Weekly", 20))}
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve), \
         patch("cron.capacity_preflight._usage_snapshot", create=True, side_effect=lambda runtime: usage[runtime["provider"]]) as fetch:
        for settings, job, expected in (({}, {}, "anthropic"),
                                        ({"enabled": True, "threshold_percent": 95}, {"model": "claude-opus-5"}, "anthropic"),
                                        ({"enabled": True, "threshold_percent": 95}, {}, "openai-codex")):
            jc = _CronJobConfig(cfg={"cron": {"capacity_preflight": settings}, "fallback_providers": chain},
                                model="claude-opus-5", model_cfg={}, cron_default_provider="")
            runtime, model = _resolve_job_runtime(job, "job", jc)
            assert (runtime["provider"], model) == (expected, "gpt-6-sol" if expected == "openai-codex" else "claude-opus-5")
        assert fetch.call_count == 2



def test_pooled_primary_credential_does_not_route_on_ambient_snapshot():
    jc = _CronJobConfig(cfg={"cron": {"capacity_preflight": {"enabled": True}},
                             "fallback_providers": [{"provider": "openai-codex", "model": "gpt-6-sol"}]},
                        model="claude-opus-5", model_cfg={}, cron_default_provider="")
    from unittest.mock import MagicMock
    pool = MagicMock()
    pool.entries.return_value = [object(), object()]
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"provider": "anthropic", "api_key": "ambient"}), \
         patch("agent.credential_pool.load_pool", return_value=pool), \
         patch("cron.capacity_preflight._usage_snapshot") as fetch:
        runtime, model = _resolve_job_runtime({}, "job", jc)
    assert (runtime["provider"], model) == ("anthropic", "claude-opus-5")
    fetch.assert_not_called()


def test_failed_usage_fetch_keeps_original_route():
    jc = _CronJobConfig(cfg={"cron": {"capacity_preflight": {"enabled": True, "threshold_percent": 95}},
                             "fallback_providers": [{"provider": "openai-codex", "model": "gpt-6-sol"}]},
                        model="claude-opus-5", model_cfg={}, cron_default_provider="")
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"provider": "anthropic", "api_key": "token"}), \
         patch("cron.capacity_preflight._usage_snapshot", create=True, side_effect=RuntimeError("network down")):
        runtime, model = _resolve_job_runtime({}, "job", jc)
    assert (runtime["provider"], model) == ("anthropic", "claude-opus-5")



def test_usage_snapshot_reuses_bounded_read_only_fetch_and_failure_is_throttled():
    from cron.capacity_preflight import _attempts
    _attempts.clear()
    fresh = snap("anthropic", window("Current session", 20))
    with patch("agent.account_usage.fetch_account_usage", return_value=fresh) as fetch:
        assert _usage_snapshot({"provider": "anthropic", "api_key": "a"}) == fresh
        assert _usage_snapshot({"provider": "anthropic", "api_key": "a"}) == fresh
        assert fetch.call_count == 1
        assert fetch.call_args.kwargs["read_only"] is True
    _attempts.clear()
    with patch("agent.account_usage.fetch_account_usage", side_effect=RuntimeError("offline")) as fetch:
        assert _usage_snapshot({"provider": "anthropic", "api_key": "b"}) is None
        assert _usage_snapshot({"provider": "anthropic", "api_key": "b"}) is None
        assert fetch.call_count == 1

def snap(provider, *windows, age=0):
    return AccountUsageSnapshot(provider=provider, source="usage_api",
                                fetched_at=datetime.now(UTC) - timedelta(seconds=age),
                                windows=tuple(windows))


def window(label, used, scope="account"):
    return AccountUsageWindow(label=label, used_percent=used, scope=scope)


def test_near_full_account_window_selects_complete_first_healthy_fallback():
    primary = {"provider": "anthropic", "api_key": "primary"}
    chain = [{"provider": "anthropic", "model": "claude-sonnet-4", "base_url": "x"},
             {"provider": "openai-codex", "model": "gpt-6-sol"}]
    snapshots = {"anthropic": snap("anthropic", window("Current session", 96), window("Current week", 70)),
                 "openai-codex": snap("openai-codex", window("Session", 20), window("Weekly", 8))}
    selected = select_capacity_route(primary, "claude-opus-5", chain, 95,
                                     lambda provider, model, entry: snapshots[provider])
    assert selected == chain[1]
    assert primary == {"provider": "anthropic", "api_key": "primary"}




def test_elapsed_reset_does_not_route_from_obsolete_window():
    expired = AccountUsageWindow(label="Current session", used_percent=99,
                                 reset_at=datetime.now(UTC) - timedelta(minutes=2))
    chain = [{"provider": "openai-codex", "model": "gpt-6-sol"}]
    primary = snap("anthropic", expired, window("Current week", 40))
    healthy = snap("openai-codex", window("Session", 10), window("Weekly", 10))
    assert select_capacity_route({"provider": "anthropic"}, "claude-opus-5", chain, 95,
                                 lambda provider, *_: primary if provider == "anthropic" else healthy) is None


def test_partial_fallback_windows_do_not_count_as_healthy():
    chain = [{"provider": "openai-codex", "model": "gpt-6-sol"}]
    snapshots = {"anthropic": snap("anthropic", window("Current session", 99), window("Current week", 40)),
                 "openai-codex": snap("openai-codex", window("Session", 10))}
    assert select_capacity_route({"provider": "anthropic"}, "claude-opus-5", chain, 95,
                                 lambda provider, *_: snapshots[provider]) is None


def test_unknown_fallback_is_not_treated_as_healthy():
    primary = {"provider": "anthropic"}
    chain = [{"provider": "openai-codex", "model": "gpt-6-sol"}]
    assert select_capacity_route(primary, "claude-opus-5", chain, 95,
                                 lambda provider, model, entry: snap("anthropic", window("Current week", 99))
                                 if provider == "anthropic" else None) is None


def test_model_scoped_opus_limit_does_not_displace_sonnet():
    chain = [{"provider": "openai-codex", "model": "gpt-6-sol"}]
    assert select_capacity_route({"provider": "anthropic"}, "claude-sonnet-4", chain, 90,
                                 lambda *_: snap("anthropic", window("Current week", 30),
                                                 window("Opus week", 100, "model"))) is None


def test_stale_primary_and_missing_numeric_usage_fail_open():
    chain = [{"provider": "openai-codex", "model": "gpt-6-sol"}]
    for snapshot in (snap("anthropic", window("Current week", 99), age=3600),
                     snap("anthropic", window("Current week", None))):
        assert select_capacity_route({"provider": "anthropic"}, "claude-opus-5", chain, 90,
                                     lambda *_: snapshot) is None
