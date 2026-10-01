"""Cron delivery selects the same owner with or without multiplex adapters."""

import asyncio
from pathlib import Path
import threading

import pytest

from agent import secret_scope
import cron.scheduler as scheduler
from cron.jobs import create_job, mark_job_run, use_cron_store
from cron.scheduler_preflight import SharedRouteAdapters
from cron.scheduler_provider import InProcessCronScheduler
from gateway import run as gateway_run
from gateway.config import Platform, load_gateway_config
from hermes_constants import get_hermes_home
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner
from tools.cronjob_tools import _run_claimed_job


@pytest.fixture
def delivery_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    homes = {name: root / "profiles" / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("display:\n  language: en\n")
    (root / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: false\n  profile_routes:\n"
        "    - {platform: telegram, profile: a, chat_id: '100'}\n"
        "    - {platform: telegram, profile: b, chat_id: '200'}\n"
        "    - {platform: telegram, profile: a, chat_id: '999', enabled: false}\n"
        "platforms:\n  telegram:\n    enabled: true\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    runner, primary = make_restart_runner()
    runner.config = load_gateway_config()
    runner._primary_profile_name = "default"
    runner._profile_adapters = {}
    runner._profile_configs = {}
    runner._served_profile_homes = {"default": root, **homes}
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    jobs = {}
    for name, home in homes.items():
        with gateway_run._profile_runtime_scope(home), use_cron_store(home):
            jobs[name] = create_job("report", "every 1h", name=name, deliver="telegram:100,telegram:200,telegram:999")
    secret_scope.set_multiplex_active(True)
    try:
        yield runner, primary, homes, jobs
    finally:
        secret_scope.set_multiplex_active(False)


def _choices(adapters, primary, own=None):
    result = []
    for chat in ("100", "200", "999", "unmatched"):
        target = {"platform": "telegram", "chat_id": chat, "thread_id": None}
        adapter = (adapters.get(Platform.TELEGRAM, target) if isinstance(adapters, SharedRouteAdapters)
                   else (adapters or {}).get(Platform.TELEGRAM))
        result.append("primary" if adapter is primary else "own" if own is not None and adapter is own else None)
    return result


@pytest.mark.parametrize("multiplex", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_ticker_manual_run_and_interrupt_notice_share_delivery_policy(delivery_profiles, monkeypatch, multiplex, owned):
    runner, primary, homes, jobs = delivery_profiles
    runner.config.multiplex_profiles = multiplex
    own = RestartTestAdapter() if owned else None
    runner._profile_adapters = {name: {Platform.TELEGRAM: own} if owned else {} for name in homes}
    stop = threading.Event()
    ticker_choices = []

    def tick(*, adapters, **kwargs):
        ticker_choices.append(_choices(adapters, primary, own))
        if len(ticker_choices) == 3:
            stop.set()

    monkeypatch.setattr(scheduler, "tick", tick)
    InProcessCronScheduler().start(
        stop, profile_homes=[("a", homes["a"]), ("b", homes["b"]), ("a", homes["a"])],
        adapters=runner.adapters, profile_adapters=runner._profile_adapters, default_profile="default")
    manual_choices = []

    def execute(job, *, adapters, **kwargs):
        manual_choices.append(_choices(adapters, primary, own))
        mark_job_run(job["id"], True)
        return True

    monkeypatch.setattr(scheduler, "run_one_job", execute)
    for name in ("a", "b", "a"):
        with gateway_run._profile_runtime_scope(homes[name]), use_cron_store(homes[name]):
            assert _run_claimed_job(jobs[name]) == {"claimed": True, "success": True, "error": None}
    expected = [["own"] * 4] * 3 if owned else [["primary", None, None, None], [None, "primary", None, None], ["primary", None, None, None]]
    assert (ticker_choices, manual_choices) == (expected, expected)

    async def notify():
        for name in ("a", "b", "a"):
            with gateway_run._profile_runtime_scope(homes[name]):
                await runner._notify_interrupted_cron_run(jobs[name]["id"], homes[name], set())

    asyncio.run(notify())
    selected = own if owned else primary
    assert selected is not None
    assert [str(chat) for chat, _text, _metadata in selected.sent_calls] == (
        ["100", "200", "999"] * 3 if owned else ["100", "200", "100"])
    assert get_hermes_home() not in homes.values()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["startup", "reconnect"])
async def test_owned_bot_failure_refuses_primary_until_reconnected(delivery_profiles, monkeypatch, entry):
    from cron import scheduler_thread
    from gateway.run_delivery_queue_watch import _drain_restart_safe_cron_deliveries
    from gateway.run_startup import _start_gateway_start_cron_and_housekeeping

    runner, primary, homes, jobs = delivery_profiles
    runner.config.multiplex_profiles = True
    runner._profile_failed_platforms = {}
    runner._plugin_rewire_unsubscribe = {}
    runner._platform_lock_takeover_on_start = False
    runner._busy_text_mode = "queue"
    runner._external_drain_active = False
    (homes["a"] / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n")
    (homes["a"] / ".env").write_text("TELEGRAM_BOT_TOKEN=owned-a-token\n")
    ready, release = asyncio.Event(), asyncio.Event()
    created = []

    class RecoveringAdapter(RestartTestAdapter):
        def __init__(self, config):
            super().__init__()
            self.config = config

        async def connect(self, *, is_reconnect=False):
            if is_reconnect:
                ready.set()
                await release.wait()
                return True
            return entry == "reconnect"

    def create_adapter(platform, config):
        adapter = RecoveringAdapter(config)
        created.append(adapter)
        return adapter

    async def no_obligations(*args, **kwargs):
        return 0

    monkeypatch.setattr(runner, "_create_adapter", create_adapter)
    monkeypatch.setattr(runner, "_redeliver_failed_obligations_for_platform", no_obligations)
    monkeypatch.setattr(runner, "_schedule_planned_restart_replay", lambda: None)
    monkeypatch.setattr(runner, "_schedule_resume_pending_sessions", lambda **kwargs: None)
    captured = {}

    class CapturedTicker:
        def __init__(self, target, *, args, kwargs, stop_event):
            captured.update(kwargs)

        def start(self):
            return None

    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", CapturedTicker)
    monkeypatch.setattr(gateway_run, "_start_gateway_housekeeping", lambda *args, **kwargs: None)
    stop, _, _, housekeeping = _start_gateway_start_cron_and_housekeeping(runner)
    stop.set()
    housekeeping.join(timeout=5)
    assert not housekeeping.is_alive()
    runner._running = entry == "reconnect"
    connected = await runner._start_one_profile_adapters("a", homes["a"], {})
    assert connected == (1 if entry == "reconnect" else 0)
    runner._running = True
    if entry == "reconnect":
        await runner._handle_profile_adapter_fatal_error("a", Platform.TELEGRAM, created[0])
    task = None
    try:
        await asyncio.wait_for(ready.wait(), 5)
        task = runner._profile_failed_platforms["a"][Platform.TELEGRAM]
        assert isinstance(task, asyncio.Task) and not task.done()
        assert runner._profile_adapters["a"] == {}
        observations = []

        async def observe():
            own = runner._profile_adapters["a"].get(Platform.TELEGRAM)
            ticker, manual, drained = [], [], []
            tick_stop = threading.Event()

            def tick(*, adapters, **kwargs):
                ticker.append(_choices(adapters, primary, own))
                if len(ticker) == 3:
                    tick_stop.set()

            monkeypatch.setattr(scheduler, "tick", tick)
            inputs = {**captured, "profile_homes": [(name, homes[name]) for name in ("a", "b", "a")], "profile_gate": None}
            InProcessCronScheduler().start(tick_stop, **inputs)

            def execute(job, *, adapters, **kwargs):
                manual.append(_choices(adapters, primary, own))
                mark_job_run(job["id"], True)
                return True

            monkeypatch.setattr(scheduler, "run_one_job", execute)
            for name in ("a", "b", "a"):
                with gateway_run._profile_runtime_scope(homes[name]), use_cron_store(homes[name]):
                    assert _run_claimed_job(jobs[name]) == {"claimed": True, "success": True, "error": None}
                    await runner._notify_interrupted_cron_run(jobs[name]["id"], homes[name], set())
            monkeypatch.setattr(scheduler, "drain_delivery_queue", lambda adapters, loop: drained.append(_choices(adapters, primary, own)))
            _drain_restart_safe_cron_deliveries(runner.adapters, asyncio.get_running_loop(), runner)
            notices = ([str(chat) for chat, _, _ in primary.sent_calls],
                       [str(chat) for chat, _, _ in own.sent_calls] if own else [])
            primary.sent_calls.clear()
            observations.append((ticker, manual, drained, notices))

        await observe()
        release.set()
        await asyncio.wait_for(task, 5)
        await observe()
        blocked = [[None] * 4, [None, "primary", None, None], [None] * 4]
        restored = [["own"] * 4, [None, "primary", None, None], ["own"] * 4]
        assert observations == [
            (blocked, blocked, [["primary"] * 4, [None] * 4], (["200"], [])),
            (restored, restored, [["primary"] * 4, ["own"] * 4], (["200"], ["100", "200", "999"] * 2)),
        ]
        assert (runner._profile_failed_platforms, [adapter.config.token for adapter in created]) == ({}, ["owned-a-token"] * 2)
    finally:
        release.set()
        if task is not None:
            await asyncio.wait_for(task, 5)
        for unsubscribe in runner._plugin_rewire_unsubscribe.values():
            unsubscribe()
