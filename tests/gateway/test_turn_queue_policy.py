"""Turn-queue policy: human-class classification, the autonomous cap gate, and the busy
explainer. A Discord message waited 34m52s behind two heavy local lanes with zero visibility
(est-38ov8); these tests pin both halves of the fix.
"""
import asyncio
import time

import pytest

from gateway.turn_queue_policy import (
    AUTONOMOUS,
    DEFAULT_HUMAN_PLATFORMS,
    HUMAN,
    TurnQueuePolicy,
    principal_platform,
    resolve_policy,
    run_busy_explainer,
    schedule_busy_explainer,
    turn_capacity,
)
from tests.waiting import async_wait_until


def _policy(**over):
    base = dict(enabled=True, autonomous_cap=None,
                human_platforms=DEFAULT_HUMAN_PLATFORMS,
                explainer_after_s=30.0, explainer_refresh_s=120.0, explainer_max=2)
    base.update(over)
    return TurnQueuePolicy(**base)


# ------------------------------------------------------------------ classification


class TestPrincipalClassification:
    def test_human_platform_principal_is_human(self):
        assert _policy().class_of_principal(
            'messaging:["clankville","discord","1","1","5"]') == HUMAN

    def test_non_human_platform_principal_is_autonomous(self):
        p = _policy()
        assert p.class_of_principal('messaging:["p","local","x"]') == AUTONOMOUS
        assert p.class_of_principal('messaging:["p","api_server","x"]') == AUTONOMOUS

    def test_unparseable_principals_default_autonomous(self):
        # An unknown shape may not claim the human lane.
        p = _policy()
        assert p.class_of_principal('cron:job-id') == AUTONOMOUS
        assert p.class_of_principal('messaging:not-json') == AUTONOMOUS
        assert p.class_of_principal(None) == AUTONOMOUS

    def test_platform_extraction(self):
        assert principal_platform('messaging:["p","TELEGRAM","c"]') == "telegram"
        assert principal_platform('messaging:["p"]') is None
        assert principal_platform('local:x') is None

    def test_override_set_moves_local_into_human(self):
        p = _policy(human_platforms=frozenset({"local"}))
        assert p.class_of_principal('messaging:["p","local","x"]') == HUMAN
        assert p.class_of_principal('messaging:["p","discord","x"]') == AUTONOMOUS


# ------------------------------------------------------------------ config plumbing


class TestConfigResolution:
    def test_defaults(self):
        p = resolve_policy(None)
        assert p.enabled is True and p.autonomous_cap is None
        assert p.explainer_after_s == 30.0 and p.explainer_refresh_s == 120.0
        assert p.explainer_max == 2

    def test_flat_gateway_dict(self):
        p = resolve_policy({"turn_priority": "off", "autonomous_turn_cap": 3,
                            "busy_explainer_after_seconds": 0})
        assert p.enabled is False and p.autonomous_cap == 3
        assert p.explainer_after_s == 0.0

    def test_nested_gateway_dict_and_dotted_key(self):
        assert resolve_policy({"gateway": {"autonomous_turn_cap": 2}}).autonomous_cap == 2
        assert resolve_policy({"gateway.autonomous_turn_cap": 5}).autonomous_cap == 5

    def test_invalid_cap_falls_back_to_no_cap_and_unrecognized_mode_defaults_on(self):
        p = resolve_policy({"autonomous_turn_cap": "lots", "turn_priority": "weird"})
        assert p.autonomous_cap is None
        assert p.enabled is True  # unrecognized choice is not "off"
        assert resolve_policy({"turn_priority": "off"}).enabled is False

    def test_gateway_config_loader_coerces_keys(self):
        # Invariant: the YAML the user writes is read by the loader the gateway uses.
        from gateway.config import GatewayConfig
        cfg = GatewayConfig.from_dict({
            "turn_priority": "off",
            "autonomous_turn_cap": 4,
            "turn_priority_human_platforms": ["Discord", " local ", ""],
            "busy_explainer_after_seconds": 12.5,
            "busy_explainer_refresh_seconds": 90,
            "busy_explainer_max_messages": 1,
        })
        assert cfg.turn_priority == "off"
        assert cfg.autonomous_turn_cap == 4
        assert cfg.turn_priority_human_platforms == ["discord", "local"]
        assert cfg.busy_explainer_after_seconds == 12.5
        assert cfg.busy_explainer_refresh_seconds == 90.0
        assert cfg.busy_explainer_max_messages == 1
        p = resolve_policy(cfg)  # attribute form of the reader
        assert p.enabled is False and p.autonomous_cap == 4 and p.explainer_after_s == 12.5
        assert "local" in p.human_platforms and "telegram" not in p.human_platforms

    def test_bogus_loader_values_keep_defaults(self):
        from gateway.config import GatewayConfig
        cfg = GatewayConfig.from_dict({"autonomous_turn_cap": -2,
                                       "busy_explainer_after_seconds": "soon",
                                       "busy_explainer_max_messages": 99})
        assert cfg.autonomous_turn_cap is None
        assert cfg.busy_explainer_after_seconds == 30.0
        assert cfg.busy_explainer_max_messages == 10  # clamped, not dropped


# ------------------------------------------------------------------ the gate


class TestAutonomousGate:
    @pytest.mark.asyncio
    async def test_two_autonomous_serialise_at_cap_one(self):
        order = []
        hold = asyncio.Event()

        async def worker(name):
            async with turn_capacity(_policy(autonomous_cap=1), AUTONOMOUS):
                order.append(name)
                await hold.wait()

        a = asyncio.create_task(worker("a"))
        await async_wait_until(lambda: order == ["a"], what="first worker in the slot")
        b = asyncio.create_task(worker("b"))
        await asyncio.sleep(0)  # one loop tick to let a naive impl start b inline
        assert order == ["a"]  # b is still waiting for the single slot
        hold.set()
        await asyncio.wait_for(asyncio.gather(a, b), 2)
        assert order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_human_turns_never_wait_on_a_saturated_gate(self):
        policy = _policy(autonomous_cap=1)
        hold = asyncio.Event()

        async def auto():
            async with turn_capacity(policy, AUTONOMOUS):
                await hold.wait()

        t = asyncio.create_task(auto())
        await async_wait_until(lambda: t in asyncio.all_tasks() and not hold.is_set(),
                               what="auto worker holding the slot", timeout=2)
        async with turn_capacity(policy, HUMAN):
            pass  # immediate: no timeout needed
        hold.set()
        await asyncio.wait_for(t, 2)

    @pytest.mark.asyncio
    async def test_cap_off_is_the_no_op_path(self):
        async with turn_capacity(_policy(autonomous_cap=None), AUTONOMOUS) as gate:
            assert gate is None
        async with turn_capacity(_policy(autonomous_cap=0), AUTONOMOUS) as gate:
            assert gate is None

    @pytest.mark.asyncio
    async def test_cancelled_wait_does_not_leak_a_slot(self):
        policy = _policy(autonomous_cap=1)

        async def hold_then_release():
            async with turn_capacity(policy, AUTONOMOUS):
                waiter = asyncio.create_task(_hold_then_set(policy))
                await async_wait_until(lambda: True, what="loop tick")
                waiter.cancel()
                try:
                    await waiter
                except asyncio.CancelledError:
                    pass

        await hold_then_release()
        from gateway import turn_queue_policy as m
        loop_id = id(asyncio.get_running_loop())
        gates = [g for (lid, cap), g in m._gates.items() if lid == loop_id and cap == 1]
        assert gates and all(g.in_flight == 0 for g in gates)


async def _hold_then_set(policy):
    async with turn_capacity(policy, AUTONOMOUS):
        pass


# ------------------------------------------------------------------ explainer


class _Platform:
    def __init__(self, value):
        self.value = value


class _Source:
    def __init__(self, platform_value):
        self.platform = _Platform(platform_value)
        self.chat_id = "c1"


class _Event:
    internal = False
    _busy_explainer_eligible = True

    def __init__(self, platform_value="discord"):
        self.source = _Source(platform_value)

    def get_command(self):
        return None


class _CommandedEvent(_Event):
    def get_command(self):
        return "/new"


class _InternalEvent(_Event):
    internal = True


class _UnstampedEvent(_Event):
    _busy_explainer_eligible = False


class _RecorderRunner:
    def __init__(self, rows, sent, fail=False):
        self._rows, self._sent, self._fail, self._i = rows, sent, fail, 0

    def _delivery_adapter_for(self, source):
        return object()

    async def _send_busy_reply(self, event, adapter, content):
        if self._fail:
            raise RuntimeError("chat exploded")
        self._sent.append(content)

    def row_reader(self):
        row = self._rows[min(self._i, len(self._rows) - 1)]
        self._i += 1
        return row


async def _no_sleep(_seconds):
    pass


def _explainer(rows, sent, *, after=1, refresh=1, max_msgs=2, fail=False,
               holders=(2, 600)):
    runner = _RecorderRunner(rows, sent, fail=fail)
    policy = _policy(explainer_after_s=after, explainer_refresh_s=refresh,
                     explainer_max=max_msgs)
    return runner, policy, holders


class TestBusyExplainer:
    @pytest.mark.asyncio
    async def test_queued_wait_gets_one_receipt_then_stops_when_answered(self):
        sent = []
        runner, policy, holders = _explainer(
            [{"status": "queued"}, {"status": "terminal"}], sent)
        n = await run_busy_explainer(
            policy, db=None, admission_id="a1", event=_Event(), runner=runner,
            adapter=object(), started_ts=time.time() - 45, sleep=_no_sleep,
            row_reader=runner.row_reader, holder_probe=lambda: holders)
        assert n == 1
        assert "Message saved" in sent[0]
        assert "2 turns running, oldest 10m" in sent[0]

    @pytest.mark.asyncio
    async def test_started_turn_is_left_to_the_heartbeat(self):
        # A claimed (started) turn narrates itself via the running-turn heartbeat; the
        # explainer goes silent the moment the row is no longer queued.
        sent = []
        runner, policy, _ = _explainer([{"status": "started"}], sent, max_msgs=1)
        n = await run_busy_explainer(
            policy, db=None, admission_id="a1", event=_Event(), runner=runner,
            adapter=object(), started_ts=time.time() - 125, sleep=_no_sleep,
            row_reader=runner.row_reader, holder_probe=lambda: (0, 0))
        assert n == 0 and sent == []

    @pytest.mark.asyncio
    async def test_refreshes_are_bounded_by_max_messages(self):
        sent = []
        runner, policy, _ = _explainer([{"status": "queued"}], sent, max_msgs=3)
        n = await run_busy_explainer(
            policy, db=None, admission_id="a1", event=_Event(), runner=runner,
            adapter=object(), started_ts=time.time(), sleep=_no_sleep,
            row_reader=runner.row_reader, holder_probe=lambda: (1, 0))
        assert n == 3 and len(sent) == 3

    @pytest.mark.asyncio
    async def test_send_failure_stops_quietly(self):
        runner, policy, _ = _explainer([{"status": "queued"}], [], max_msgs=5, fail=True)
        n = await run_busy_explainer(
            policy, db=None, admission_id="a", event=_Event(), runner=runner,
            adapter=object(), started_ts=time.time(), sleep=_no_sleep,
            row_reader=runner.row_reader, holder_probe=lambda: (0, 0))
        assert n == 0  # nothing counted as sent; no exception escaped

    @pytest.mark.asyncio
    async def test_vanished_row_sends_nothing(self):
        sent = []
        runner, policy, _ = _explainer([None], sent)
        n = await run_busy_explainer(
            policy, db=None, admission_id="a", event=_Event(), runner=runner,
            adapter=object(), started_ts=time.time(), sleep=_no_sleep,
            row_reader=runner.row_reader, holder_probe=lambda: (0, 0))
        assert n == 0 and sent == []


class TestExplainerSchedulingGates:
    class _Authority:
        def __init__(self, runner):
            self.db = None
            self.runner = runner

    @staticmethod
    def _runner():
        class R:
            def _delivery_adapter_for(self, source):
                return object()
        return R()

    @pytest.mark.asyncio
    async def test_eligible_human_message_schedules_a_task(self):
        task = schedule_busy_explainer(self._Authority(self._runner()), _Event(),
                                       "a1", _policy(explainer_after_s=30))
        assert isinstance(task, asyncio.Task)
        task.cancel()

    @pytest.mark.asyncio
    async def test_gate_off_for_internal_command_unstamped_autonomous_and_knobs(self):
        a = self._Authority(self._runner())
        cases = [
            (_InternalEvent(), _policy()),
            (_CommandedEvent(), _policy()),
            (_UnstampedEvent(), _policy()),
            (_Event("local"), _policy()),
            (_Event("discord"), _policy(explainer_after_s=0)),
            (_Event("discord"), _policy(explainer_max=0)),
            # turn_priority: off kills the whole feature, receipts included.
            (_Event("discord"), _policy(enabled=False)),
        ]
        for event, policy in cases:
            assert schedule_busy_explainer(a, event, "x", policy) is None, event


# ------------------------------------------------------------------ drain wrapper


class _ExecStub:
    """Duck-typed SessionAuthority for the wrapper: runner.config + a recording inner."""

    def __init__(self, cfg, release=None):
        self.runner = type("R", (), {"config": cfg})()
        self.inside = []
        self._release = release

    async def _execute_claimed_inner(self, ref, live, row):
        self.inside.append(row["principal_id"])
        if row["principal_id"].startswith("local:") and self._release is not None:
            await self._release.wait()
        return ("ok", "completed")


class TestExecuteClaimedWrapper:
    @pytest.mark.asyncio
    async def test_wrapper_delegates_to_inner(self):
        from gateway.session_authority import SessionAuthority
        stub = _ExecStub({"autonomous_turn_cap": None})
        result = await SessionAuthority._execute_claimed(
            stub, None, None,
            {"principal_id": 'messaging:["p","discord","c"]', "admission_id": "a"})
        assert result == ("ok", "completed")
        assert stub.inside == ['messaging:["p","discord","c"]']

    @pytest.mark.asyncio
    async def test_cap_blocks_a_second_autonomous_execution_but_not_a_human(self):
        from gateway.session_authority import SessionAuthority
        release = asyncio.Event()
        stub = _ExecStub({"turn_priority": "human_first", "autonomous_turn_cap": 1},
                         release=release)
        a1 = asyncio.create_task(SessionAuthority._execute_claimed(
            stub, None, None, {"principal_id": "local:one", "admission_id": "a"}))
        await async_wait_until(lambda: stub.inside == ["local:one"],
                               what="first autonomous turn inside")
        a2 = asyncio.create_task(SessionAuthority._execute_claimed(
            stub, None, None, {"principal_id": "local:two", "admission_id": "b"}))
        await asyncio.sleep(0)
        assert stub.inside == ["local:one"]  # a2 waits behind the cap
        human = await asyncio.wait_for(SessionAuthority._execute_claimed(
            stub, None, None,
            {"principal_id": 'messaging:["p","telegram","c"]', "admission_id": "c"}), 1)
        assert human == ("ok", "completed")  # the human slipped past the busy gate
        release.set()
        await asyncio.wait_for(asyncio.gather(a1, a2), 2)
        # The human ran between the two autonomics — a2 held its place behind the cap.
        assert stub.inside == ["local:one", 'messaging:["p","telegram","c"]', "local:two"]
