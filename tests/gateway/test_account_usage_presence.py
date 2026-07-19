from datetime import datetime, timezone
import asyncio
import json
import threading
import time

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from gateway.account_usage_presence import (
    AccountUsagePresenceApplyResult,
    AccountUsagePresenceCapabilities,
    AccountUsagePresenceController,
    AccountUsagePresenceRestoreResult,
    account_usage_presence_payload_from_snapshot,
)
from gateway.config import AccountUsagePresenceConfig


RESET_AT = datetime(2030, 1, 2, 3, 4, tzinfo=timezone.utc)


def _snapshot(
    *,
    used_percent: float = 25,
    label: str = "Session",
    source: str = "usage_api",
    reset_at: datetime | None = RESET_AT,
) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="openai-codex",
        source=source,
        fetched_at=datetime.now(timezone.utc),
        plan="Pro",
        windows=(
            AccountUsageWindow(
                label=label,
                used_percent=used_percent,
                reset_at=reset_at,
            ),
        ),
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _RateLimited(RuntimeError):
    def __init__(self, retry_after: float) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


class _Adapter:
    def __init__(
        self,
        platform: str,
        *,
        capabilities: AccountUsagePresenceCapabilities,
        baseline: dict | None = None,
    ) -> None:
        self.platform = platform
        self.account_usage_presence_capabilities = capabilities
        self._baseline = baseline
        self.applied = []
        self.restored = []
        self.apply_error: Exception | None = None
        self.restore_result = AccountUsagePresenceRestoreResult.RESTORED

    def account_usage_presence_state_key(self) -> str:
        return self.platform

    async def capture_account_usage_presence_baseline(self):
        return self._baseline

    def build_account_usage_presence_owned_state(self, payload, baseline):
        if baseline is None:
            return None
        return {"display_name": f"owned-{payload.remaining_percent}"}

    async def apply_account_usage_presence(self, payload, baseline):
        if self.apply_error is not None:
            raise self.apply_error
        self.applied.append((payload, baseline))
        return True

    async def apply_account_usage_presence_if_owned(
        self,
        payload,
        baseline,
        expected_owned,
    ) -> AccountUsagePresenceApplyResult:
        changed = await self.apply_account_usage_presence(payload, baseline)
        return (
            AccountUsagePresenceApplyResult.APPLIED
            if changed
            else AccountUsagePresenceApplyResult.RETRY
        )

    async def restore_account_usage_presence(self, baseline, owned):
        self.restored.append((baseline, owned))
        return self.restore_result


class _ApplyResultAdapter(_Adapter):
    def __init__(self, *args, apply_results, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_results = iter(apply_results)

    async def apply_account_usage_presence(self, payload, baseline):
        self.applied.append((payload, baseline))
        return next(self._apply_results)


def _config(**overrides) -> AccountUsagePresenceConfig:
    data = {
        "enabled": True,
        "provider": "openai-codex",
        "platforms": ["telegram", "discord"],
        "update_interval_seconds": 300,
        "stale_after_seconds": 900,
    }
    data.update(overrides)
    return AccountUsagePresenceConfig.from_dict(data)


def test_snapshot_uses_first_percentage_window():
    payload = account_usage_presence_payload_from_snapshot(_snapshot(used_percent=12.4))

    assert payload is not None
    assert payload.label == "Session"
    assert payload.remaining_percent == 88


def test_payload_equality_ignores_unrendered_snapshot_metadata():
    first = account_usage_presence_payload_from_snapshot(
        _snapshot(source="usage_api", reset_at=RESET_AT)
    )
    second = account_usage_presence_payload_from_snapshot(
        _snapshot(
            source="different_endpoint",
            reset_at=datetime(2031, 5, 6, tzinfo=timezone.utc),
        )
    )

    assert first is not None
    assert first == second


def test_snapshot_can_select_a_named_window_case_insensitively():
    snapshot = AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="Five hour", used_percent=10),
            AccountUsageWindow(label="Seven day", used_percent=82),
        ),
    )

    payload = account_usage_presence_payload_from_snapshot(snapshot, window_label="seven DAY")

    assert payload is not None
    assert payload.label == "Seven day"
    assert payload.remaining_percent == 18


def test_snapshot_without_matching_percentage_window_is_unsupported():
    snapshot = AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="Credits", detail="$5 remaining"),),
    )

    assert account_usage_presence_payload_from_snapshot(snapshot) is None
    assert account_usage_presence_payload_from_snapshot(snapshot, window_label="missing") is None


@pytest.mark.parametrize("used_percent", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_percentage_is_unsupported(used_percent):
    assert account_usage_presence_payload_from_snapshot(_snapshot(used_percent=used_percent)) is None


@pytest.mark.parametrize(
    ("used_percent", "remaining_percent"),
    [
        (0, 100),
        (50, 50),
        (51, 49),
        (80, 20),
        (81, 19),
        (99, 1),
        (100, 0),
        (150, 0),
    ],
)
def test_percent_clamping(used_percent, remaining_percent):
    payload = account_usage_presence_payload_from_snapshot(_snapshot(used_percent=used_percent))

    assert payload is not None
    assert payload.remaining_percent == remaining_percent


@pytest.mark.asyncio
async def test_one_provider_fetch_fans_out_to_two_platforms(tmp_path):
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
    )
    discord = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    calls = []

    def fetch(provider):
        calls.append(provider)
        return _snapshot()

    controller = AccountUsagePresenceController(
        _config(),
        lambda: {"telegram": telegram, "discord": discord},
        fetcher=fetch,
        state_path=tmp_path / "account-usage-presence.json",
    )

    await controller.refresh_once()

    assert calls == ["openai-codex"]
    assert len(telegram.applied) == 1
    assert len(discord.applied) == 1
    assert telegram.applied[0][1] == {"display_name": "Hermes"}


@pytest.mark.asyncio
async def test_unchanged_payload_skips_api_but_replacement_adapter_reapplies(tmp_path):
    current = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    adapters = {"discord": current}
    controller = AccountUsagePresenceController(
        _config(platforms=["discord"]),
        lambda: adapters,
        fetcher=lambda provider: _snapshot(),
        state_path=tmp_path / "state.json",
    )

    await controller.refresh_once()
    await controller.refresh_once()
    assert len(current.applied) == 1

    replacement = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    adapters["discord"] = replacement
    await controller.refresh_once()

    assert len(replacement.applied) == 1


@pytest.mark.asyncio
async def test_transient_fetch_failure_keeps_value_only_until_stale_ttl(tmp_path):
    adapter = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    snapshots = [_snapshot(), None, None]
    clock = _Clock()
    controller = AccountUsagePresenceController(
        _config(platforms=["discord"], stale_after_seconds=600),
        lambda: {"discord": adapter},
        fetcher=lambda provider: snapshots.pop(0),
        state_path=tmp_path / "state.json",
        monotonic=clock,
    )

    await controller.refresh_once()
    clock.now = 599
    await controller.refresh_once()
    assert len(adapter.applied) == 2
    assert adapter.applied[-1][0].cached is True

    clock.now = 601
    await controller.refresh_once()
    assert len(adapter.applied) == 3
    unknown = adapter.applied[-1][0]
    assert unknown.label == "Usage"
    assert unknown.remaining_percent is None
    assert unknown.cached is False


@pytest.mark.asyncio
async def test_first_fetch_failure_surfaces_unknown_instead_of_stale_value(tmp_path):
    adapter = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["discord"]),
        lambda: {"discord": adapter},
        fetcher=lambda provider: None,
        state_path=tmp_path / "state.json",
    )

    await controller.refresh_once()

    assert len(adapter.applied) == 1
    assert adapter.applied[0][0].label == "Usage"
    assert adapter.applied[0][0].remaining_percent is None


@pytest.mark.asyncio
async def test_adapter_retry_after_is_respected_without_blocking_other_platforms(tmp_path):
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
    )
    telegram.apply_error = _RateLimited(120)
    discord = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    clock = _Clock()
    controller = AccountUsagePresenceController(
        _config(),
        lambda: {"telegram": telegram, "discord": discord},
        fetcher=lambda provider: _snapshot(),
        state_path=tmp_path / "state.json",
        monotonic=clock,
    )

    await controller.refresh_once()
    telegram.apply_error = None
    clock.now = 119
    await controller.refresh_once()

    assert telegram.applied == []
    assert len(discord.applied) == 1

    clock.now = 121
    await controller.refresh_once()
    assert len(telegram.applied) == 1


@pytest.mark.asyncio
async def test_provider_retry_after_suppresses_fetch_until_deadline(tmp_path):
    clock = _Clock()
    calls = []

    def fetch(provider):
        calls.append(provider)
        if len(calls) == 1:
            raise _RateLimited(120)
        return _snapshot()

    adapter = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["discord"]),
        lambda: {"discord": adapter},
        fetcher=fetch,
        state_path=tmp_path / "state.json",
        monotonic=clock,
    )

    await controller.refresh_once()
    clock.now = 119
    await controller.refresh_once()
    assert calls == ["openai-codex"]

    clock.now = 121
    await controller.refresh_once()
    assert calls == ["openai-codex", "openai-codex"]
    assert adapter.applied[-1][0].remaining_percent == 75


@pytest.mark.asyncio
async def test_baseline_is_persisted_before_mutation_and_restored_on_stop(tmp_path):
    state_path = tmp_path / "account-usage-presence" / "journal.json"
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: _snapshot(),
        state_path=state_path,
    )

    await controller.refresh_once()

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    entry = saved["entries"]["telegram"]
    assert entry["baseline"]["display_name"] == "Hermes"
    assert entry["owned"]["display_name"] == "owned-75"
    assert entry["phase"] == "owned"

    await controller.stop()

    assert telegram.restored == [
        ({"display_name": "Hermes"}, {"display_name": "owned-75"})
    ]
    assert not state_path.exists() or json.loads(state_path.read_text())["entries"] == {}


@pytest.mark.asyncio
async def test_later_false_apply_keeps_prior_owned_journal_for_stop(tmp_path):
    state_path = tmp_path / "journal.json"
    telegram = _ApplyResultAdapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
        apply_results=(True, False),
    )
    snapshots = iter((_snapshot(used_percent=25), _snapshot(used_percent=26)))
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: next(snapshots),
        state_path=state_path,
    )

    await controller.refresh_once()
    await controller.refresh_once()

    entry = json.loads(state_path.read_text(encoding="utf-8"))["entries"]["telegram"]
    assert entry["phase"] == "owned"
    assert entry["baseline"] == {"display_name": "Hermes"}
    assert entry["owned"] == {"display_name": "owned-75"}

    await controller.stop()

    assert telegram.restored == [
        ({"display_name": "Hermes"}, {"display_name": "owned-75"})
    ]


@pytest.mark.asyncio
async def test_later_false_apply_recovers_prior_owned_after_rollback_write_failure(
    tmp_path,
    monkeypatch,
):
    import gateway.account_usage_presence as presence_module

    class StatefulAdapter(_ApplyResultAdapter):
        def __init__(self, *args, remote_state, **kwargs):
            super().__init__(*args, **kwargs)
            self.remote_state = dict(remote_state)

        async def apply_account_usage_presence(self, payload, baseline):
            changed = await super().apply_account_usage_presence(payload, baseline)
            if changed:
                self.remote_state = self.build_account_usage_presence_owned_state(
                    payload,
                    baseline,
                )
            return changed

        async def restore_account_usage_presence(self, baseline, owned):
            self.restored.append((baseline, owned))
            if self.remote_state == owned:
                self.remote_state = dict(baseline)
                return AccountUsagePresenceRestoreResult.RESTORED
            if self.remote_state == baseline:
                return AccountUsagePresenceRestoreResult.ALREADY_BASELINE
            return AccountUsagePresenceRestoreResult.EXTERNAL

    state_path = tmp_path / "journal.json"
    baseline = {"display_name": "Hermes"}
    telegram = StatefulAdapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline=baseline,
        apply_results=(True, False),
        remote_state=baseline,
    )
    snapshots = iter((_snapshot(used_percent=25), _snapshot(used_percent=26)))
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: next(snapshots),
        state_path=state_path,
    )

    real_write = presence_module._write_state_atomically
    write_count = 0

    def fail_rollback_write(path, value):
        nonlocal write_count
        write_count += 1
        if write_count == 4:
            raise OSError("simulated rollback journal failure")
        real_write(path, value)

    monkeypatch.setattr(
        presence_module,
        "_write_state_atomically",
        fail_rollback_write,
    )
    await controller.refresh_once()
    await controller.refresh_once()

    pending = json.loads(state_path.read_text(encoding="utf-8"))["entries"][
        "telegram"
    ]
    assert pending["phase"] == "pending"
    assert pending["owned"] == {"display_name": "owned-74"}
    assert pending["previous_owned"] == {"display_name": "owned-75"}
    assert telegram.remote_state == {"display_name": "owned-75"}

    monkeypatch.setattr(
        presence_module,
        "_write_state_atomically",
        real_write,
    )
    restarted = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: None,
        state_path=state_path,
    )
    await restarted.stop()

    assert telegram.restored == [
        (baseline, {"display_name": "owned-74"}),
        (baseline, {"display_name": "owned-75"}),
    ]
    assert telegram.remote_state == baseline
    assert not state_path.exists() or json.loads(state_path.read_text())["entries"] == {}


@pytest.mark.asyncio
async def test_pending_transition_blocks_reapply_until_restart_recovery(tmp_path):
    class AmbiguousAdapter(_Adapter):
        def __init__(self, *args, remote_state, **kwargs):
            super().__init__(*args, **kwargs)
            self.remote_state = dict(remote_state)
            self.guarded_calls = 0

        async def apply_account_usage_presence_if_owned(
            self,
            payload,
            baseline,
            expected_owned,
        ) -> AccountUsagePresenceApplyResult:
            self.guarded_calls += 1
            if self.remote_state != expected_owned:
                return AccountUsagePresenceApplyResult.EXTERNAL
            if self.guarded_calls == 2:
                raise TimeoutError("ambiguous provider timeout")
            desired = self.build_account_usage_presence_owned_state(payload, baseline)
            assert desired is not None
            self.remote_state = dict(desired)
            self.applied.append((payload, baseline))
            return AccountUsagePresenceApplyResult.APPLIED

        async def restore_account_usage_presence(self, baseline, owned):
            self.restored.append((baseline, owned))
            if self.remote_state == owned:
                self.remote_state = dict(baseline)
                return AccountUsagePresenceRestoreResult.RESTORED
            return AccountUsagePresenceRestoreResult.EXTERNAL

    state_path = tmp_path / "journal.json"
    baseline = {"display_name": "Hermes"}
    telegram = AmbiguousAdapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline=baseline,
        remote_state=baseline,
    )
    snapshots = iter(
        (
            _snapshot(used_percent=25),
            _snapshot(used_percent=26),
            _snapshot(used_percent=27),
        )
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: next(snapshots),
        state_path=state_path,
    )

    await controller.refresh_once()
    await controller.refresh_once()
    await controller.refresh_once()

    assert telegram.guarded_calls == 2
    pending = json.loads(state_path.read_text())["entries"]["telegram"]
    assert pending["phase"] == "pending"
    assert pending["owned"] == {"display_name": "owned-74"}
    assert pending["previous_owned"] == {"display_name": "owned-75"}

    restarted = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: None,
        state_path=state_path,
    )
    await restarted.stop()

    assert telegram.restored == [
        (baseline, {"display_name": "owned-74"}),
        (baseline, {"display_name": "owned-75"}),
    ]
    assert telegram.remote_state == baseline


@pytest.mark.asyncio
async def test_external_identity_change_is_not_overwritten_by_next_generation(tmp_path):
    class GuardedAdapter(_Adapter):
        def __init__(self, *args, remote_state, **kwargs):
            super().__init__(*args, **kwargs)
            self.remote_state = dict(remote_state)

        async def apply_account_usage_presence_if_owned(
            self,
            payload,
            baseline,
            expected_owned,
        ) -> AccountUsagePresenceApplyResult:
            if self.remote_state != expected_owned:
                return AccountUsagePresenceApplyResult.EXTERNAL
            desired = self.build_account_usage_presence_owned_state(payload, baseline)
            assert desired is not None
            self.remote_state = dict(desired)
            self.applied.append((payload, baseline))
            return AccountUsagePresenceApplyResult.APPLIED

    state_path = tmp_path / "journal.json"
    baseline = {"display_name": "Hermes"}
    telegram = GuardedAdapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline=baseline,
        remote_state=baseline,
    )
    snapshots = iter((_snapshot(used_percent=25), _snapshot(used_percent=26)))
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: next(snapshots),
        state_path=state_path,
    )

    await controller.refresh_once()
    telegram.remote_state = {"display_name": "Operator override"}
    await controller.refresh_once()

    assert telegram.remote_state == {"display_name": "Operator override"}
    assert len(telegram.applied) == 1
    assert json.loads(state_path.read_text())["entries"] == {}


@pytest.mark.asyncio
async def test_initial_false_apply_discards_never_owned_pending_journal(tmp_path):
    state_path = tmp_path / "journal.json"
    telegram = _ApplyResultAdapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
        apply_results=(False,),
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: _snapshot(),
        state_path=state_path,
    )

    await controller.refresh_once()

    assert json.loads(state_path.read_text(encoding="utf-8"))["entries"] == {}


@pytest.mark.asyncio
async def test_external_identity_change_is_preserved_on_restore(tmp_path):
    state_path = tmp_path / "journal.json"
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
    )
    telegram.restore_result = AccountUsagePresenceRestoreResult.EXTERNAL
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: _snapshot(),
        state_path=state_path,
    )

    await controller.refresh_once()
    await controller.stop()

    assert telegram.restored
    assert not json.loads(state_path.read_text(encoding="utf-8"))["entries"]


@pytest.mark.asyncio
async def test_disabled_controller_recovers_saved_identity_without_fetching(tmp_path):
    state_path = tmp_path / "account-usage-presence" / "journal.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": {
                    "telegram": {
                        "baseline": {"display_name": "Hermes"},
                        "owned": {"display_name": "Hermes · Session 75%"},
                        "phase": "owned",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
    )
    controller = AccountUsagePresenceController(
        AccountUsagePresenceConfig(),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: pytest.fail("disabled controller must not fetch"),
        state_path=state_path,
    )

    await controller.start()

    assert telegram.restored == [
        ({"display_name": "Hermes"}, {"display_name": "Hermes · Session 75%"})
    ]
    assert json.loads(state_path.read_text(encoding="utf-8"))["entries"] == {}
    assert controller.task is None


@pytest.mark.asyncio
async def test_disabled_recovery_retries_after_late_adapter_connect(tmp_path):
    state_path = tmp_path / "journal.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "entries": {
                    "telegram": {
                        "baseline": {"display_name": "Hermes"},
                        "owned": {"display_name": "Hermes · Session 75%"},
                        "phase": "owned",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    adapters: dict = {}
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
    )
    controller = AccountUsagePresenceController(
        AccountUsagePresenceConfig(),
        lambda: adapters,
        fetcher=lambda provider: pytest.fail("disabled controller must not fetch"),
        state_path=state_path,
        recovery_interval_seconds=0.01,
    )

    await controller.start()
    assert telegram.restored == []
    assert controller.task is not None

    adapters["telegram"] = telegram
    for _ in range(100):
        if telegram.restored:
            break
        await asyncio.sleep(0.01)

    assert telegram.restored
    assert json.loads(state_path.read_text(encoding="utf-8"))["entries"] == {}
    await controller.stop()


@pytest.mark.asyncio
async def test_malformed_journal_blocks_identity_mutation(tmp_path):
    state_path = tmp_path / "journal.json"
    state_path.write_text("{not-json", encoding="utf-8")
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: _snapshot(),
        state_path=state_path,
    )

    await controller.refresh_once()

    assert telegram.applied == []
    assert state_path.read_text(encoding="utf-8") == "{not-json"


@pytest.mark.asyncio
async def test_symlink_journal_blocks_identity_mutation(tmp_path):
    target = tmp_path / "victim.txt"
    target.write_text("do-not-clobber", encoding="utf-8")
    state_path = tmp_path / "journal.json"
    state_path.symlink_to(target)
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: _snapshot(),
        state_path=state_path,
    )

    await controller.refresh_once()

    assert telegram.applied == []
    assert target.read_text(encoding="utf-8") == "do-not-clobber"


@pytest.mark.asyncio
async def test_baseline_persistence_failure_blocks_identity_mutation(
    tmp_path, monkeypatch
):
    telegram = _Adapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
    )

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "gateway.account_usage_presence._write_state_atomically",
        fail_write,
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["telegram"]),
        lambda: {"telegram": telegram},
        fetcher=lambda provider: _snapshot(),
        state_path=tmp_path / "state.json",
    )

    await controller.refresh_once()

    assert telegram.applied == []


@pytest.mark.asyncio
async def test_stop_does_not_wait_for_blocked_sync_fetch(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def blocking_fetch(provider):
        started.set()
        release.wait(timeout=5)
        return _snapshot()

    controller = AccountUsagePresenceController(
        _config(platforms=["discord"]),
        lambda: {},
        fetcher=blocking_fetch,
        state_path=tmp_path / "state.json",
        fetch_timeout_seconds=30,
    )

    try:
        await controller.start()
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

        before = time.monotonic()
        await controller.stop()
        elapsed = time.monotonic() - before

        worker = controller._fetch_thread
        assert elapsed < 0.5
        assert worker is not None
        assert worker.daemon
        assert worker.is_alive()
    finally:
        release.set()
        worker = controller._fetch_thread
        if worker is not None:
            worker.join(timeout=1)


@pytest.mark.asyncio
async def test_hung_adapter_times_out_without_blocking_other_platform(tmp_path):
    class HangingAdapter(_Adapter):
        async def apply_account_usage_presence(self, payload, baseline):
            await asyncio.Event().wait()

    telegram = HangingAdapter(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
    )
    discord = _Adapter(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(activity=True),
    )
    controller = AccountUsagePresenceController(
        _config(),
        lambda: {"telegram": telegram, "discord": discord},
        fetcher=lambda provider: _snapshot(),
        state_path=tmp_path / "state.json",
        adapter_timeout_seconds=0.01,
    )

    await controller.refresh_once()

    assert len(discord.applied) == 1


@pytest.mark.asyncio
async def test_shutdown_restore_runs_in_parallel_within_one_timeout(tmp_path):
    class SlowRestore(_Adapter):
        async def restore_account_usage_presence(self, baseline, owned):
            await asyncio.sleep(0.15)
            return await super().restore_account_usage_presence(baseline, owned)

    state_path = tmp_path / "journal.json"
    telegram = SlowRestore(
        "telegram",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Hermes"},
    )
    discord = SlowRestore(
        "discord",
        capabilities=AccountUsagePresenceCapabilities(display_name=True),
        baseline={"display_name": "Bot"},
    )
    controller = AccountUsagePresenceController(
        _config(),
        lambda: {"telegram": telegram, "discord": discord},
        fetcher=lambda provider: _snapshot(),
        state_path=state_path,
        adapter_timeout_seconds=1.0,
    )

    await controller.refresh_once()
    before = time.monotonic()
    await controller.stop()
    elapsed = time.monotonic() - before

    assert len(telegram.restored) == 1
    assert len(discord.restored) == 1
    assert elapsed < 0.28


@pytest.mark.asyncio
async def test_unsupported_adapter_is_explicit_noop(tmp_path):
    adapter = _Adapter(
        "matrix",
        capabilities=AccountUsagePresenceCapabilities(),
    )
    controller = AccountUsagePresenceController(
        _config(platforms=["matrix"]),
        lambda: {"matrix": adapter},
        fetcher=lambda provider: _snapshot(),
        state_path=tmp_path / "state.json",
    )

    await controller.refresh_once()

    assert adapter.applied == []
