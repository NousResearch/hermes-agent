"""Unit tests for agent.local_network — outage verdict, route scoping, wait ledger.

The verdict gates provider failover, so it must be right in BOTH directions: a false
"offline" would suppress a legitimate failover, a missed outage would switch routes for a
fault that is the user's own network. Route scoping and the wait ledger are the other half
of that contract: a local (loopback/LAN) fallback must stay eligible while the machine is
offline, and the configured wait cap must bound real wall-clock sleeping — not just a
counter that a longer sleep can outrun.
"""

from __future__ import annotations

import asyncio
import socket
import time
from unittest.mock import patch

import httpx
import pytest

from agent import local_network
from agent.error_classifier import FailoverReason
from agent.turn_retry_state import TurnRetryState


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    """The probe verdict is process-cached; tests must not inherit a neighbour's verdict."""
    local_network.reset_connectivity_cache()
    yield
    local_network.reset_connectivity_cache()


def _dns_failure(message: str = "Temporary failure in name resolution") -> httpx.ConnectError:
    """A ConnectError whose CAUSE carries the OS resolver text (the real SDK shape)."""
    error = httpx.ConnectError("Connection error.")
    error.__cause__ = socket.gaierror(-3, message)
    return error


class TestOutageVerdict:
    def test_dns_failure_with_dead_probe_is_an_outage(self):
        with patch.object(local_network, "probe_connectivity", return_value=False):
            assert local_network.local_network_outage(_dns_failure()) is True

    def test_dns_marker_with_live_probe_is_provider_side(self):
        """The machine is online, so the provider's own DNS/route is the problem — failover may help."""
        with patch.object(local_network, "probe_connectivity", return_value=True):
            assert local_network.local_network_outage(
                _dns_failure("Name or service not known")
            ) is False

    def test_read_timeout_is_never_an_outage(self):
        """A read timeout proves a connection WAS established — the provider was reachable."""
        with patch.object(local_network, "probe_connectivity", return_value=False):
            assert local_network.local_network_outage(httpx.ReadTimeout("timed out")) is False

    def test_connection_refused_is_provider_side(self):
        """The host answered with a RST: the local network works."""
        error = httpx.ConnectError("[Errno 111] Connection refused")
        with patch.object(local_network, "probe_connectivity", return_value=False):
            assert local_network.local_network_outage(error) is False

    def test_connect_timeout_with_dead_probe_is_an_outage(self):
        with patch.object(local_network, "probe_connectivity", return_value=False):
            assert local_network.local_network_outage(httpx.ConnectTimeout("timed out")) is True

    def test_unroutable_network_is_an_outage(self):
        with patch.object(local_network, "probe_connectivity", return_value=False):
            assert local_network.local_network_outage(
                httpx.ConnectError("[Errno 101] Network is unreachable")
            ) is True

    def test_unknown_probe_trusts_hard_markers_only(self):
        with patch.object(local_network, "probe_connectivity", return_value=None):
            assert local_network.local_network_outage(_dns_failure()) is True
            assert local_network.local_network_outage(httpx.ConnectError("Connection error.")) is False

    def test_plain_status_error_never_probes(self):
        """A 429/400 carries no connectivity evidence; probing there would be pure latency."""
        error = httpx.HTTPStatusError(
            "rate limit", request=httpx.Request("POST", "https://api.deepseek.com/v1"),
            response=httpx.Response(429),
        )
        with patch.object(local_network, "probe_connectivity") as probe:
            assert local_network.local_network_outage(error) is False
            probe.assert_not_called()

    def test_cache_only_verdicts_never_touch_a_socket(self):
        """The auxiliary ladder may run on an event loop: classification must be I/O-free there."""
        poisoned = patch.object(local_network.socket, "create_connection", side_effect=AssertionError("blocking I/O"))
        with poisoned, patch.object(local_network.socket, "getaddrinfo", side_effect=AssertionError("blocking I/O")):
            # No cached verdict: hard markers still decide, without probing.
            assert local_network.local_network_outage(_dns_failure(), allow_probe=False) is True
            assert local_network.local_network_outage(httpx.ConnectError("Connection error."), allow_probe=False) is False

        # A cached verdict is consumed as-is.
        with patch.object(local_network, "_probe_uncached", return_value=False):
            assert local_network.probe_connectivity() is False
        with patch.object(local_network, "_probe_uncached", side_effect=AssertionError("blocking I/O")):
            assert local_network.local_network_outage(_dns_failure(), allow_probe=False) is True


class TestProbe:
    def test_probe_verdict_is_cached_between_calls(self):
        with patch.object(local_network, "_probe_uncached", return_value=False) as raw_probe:
            assert local_network.probe_connectivity() is False
            assert local_network.probe_connectivity() is False
            raw_probe.assert_called_once()

    def test_probe_cache_expires(self):
        with patch.object(local_network, "_probe_uncached", return_value=False) as raw_probe:
            local_network.probe_connectivity(cache_ttl=0.0)
            local_network.probe_connectivity(cache_ttl=0.0)
            assert raw_probe.call_count == 2

    def test_resolver_answers_but_no_egress_is_offline(self):
        """A router/cache resolver can answer while the default route is dead: DNS is not egress."""
        with (
            patch.object(
                local_network.socket, "getaddrinfo",
                return_value=[(2, 1, 6, "", ("1.1.1.1", 443))],
            ),
            patch.object(local_network.socket, "create_connection", side_effect=OSError("unreachable")),
        ):
            assert local_network.probe_connectivity() is False
        # …and the verdict then suppresses the very switch an upstream DNS failure triggers.
        with patch.object(local_network, "probe_connectivity", return_value=False):
            assert local_network.local_network_outage(_dns_failure()) is True

    def test_dead_socket_is_offline(self):
        with patch.object(local_network.socket, "create_connection", side_effect=socket.timeout("t")):
            assert local_network.probe_connectivity() is False

    def test_refusal_proves_egress(self):
        """An RST from the far side still proves the packet left the machine."""
        with patch.object(local_network.socket, "create_connection", side_effect=ConnectionRefusedError()):
            assert local_network.probe_connectivity() is True

    def test_handshake_proves_egress(self):
        with patch.object(local_network.socket, "create_connection", return_value=socket.socket()):
            assert local_network.probe_connectivity() is True

    def test_probe_stops_at_the_deadline(self):
        """The sweep is bounded: it runs inside an error path and must not stall the retry loop."""
        tick = {"n": 0}

        def _clock() -> float:
            tick["n"] += 1
            return tick["n"] * 4.0

        with (
            patch.object(local_network.time, "monotonic", side_effect=_clock),
            patch.object(
                local_network.socket, "create_connection", side_effect=socket.timeout("t"),
            ) as connect,
        ):
            assert local_network.probe_connectivity(deadline=5.0) is False
        assert connect.call_count < len(local_network._PROBE_ENDPOINTS), (
            "the deadline must stop the sweep before every witness is tried"
        )

    def test_probe_fault_is_unknown(self):
        with patch.object(local_network.socket, "create_connection", side_effect=RuntimeError("weird")):
            assert local_network.probe_connectivity() is None


class TestRouteScope:
    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:11434/v1",
            "http://127.0.0.1:8000/v1",
            "http://[::1]:8000/v1",
            "http://ollama.local:11434/v1",
            "http://10.0.0.5:8000/v1",
            "http://192.168.1.7:1234/v1",
            "http://172.16.9.10:8000/v1",
            "http://gpu-box:8000/v1",
        ],
    )
    def test_local_routes_stay_reachable_without_the_wan(self, base_url):
        assert local_network.route_needs_external_network("custom", base_url) is False

    @pytest.mark.parametrize(
        "base_url",
        ["https://api.deepseek.com/v1", "https://openrouter.ai/api/v1", "http://93.184.216.34/v1"],
    )
    def test_external_routes_need_the_wan(self, base_url):
        assert local_network.route_needs_external_network("custom", base_url) is True

    def test_local_provider_slug_without_base_url_is_local(self):
        assert local_network.route_needs_external_network("ollama") is False
        assert local_network.route_needs_external_network("lmstudio") is False

    def test_cloud_provider_without_base_url_is_external(self):
        assert local_network.route_needs_external_network("openrouter") is True
        assert local_network.route_needs_external_network("deepseek") is True


class TestWaitPolicy:
    def test_wait_schedule_widens_then_holds(self):
        assert local_network.network_outage_wait_seconds(1) == 5.0
        assert local_network.network_outage_wait_seconds(3) == 20.0
        assert local_network.network_outage_wait_seconds(99) == local_network._OUTAGE_WAIT_SCHEDULE[-1]

    def test_max_wait_defaults_to_two_minutes(self):
        with patch("hermes_cli.config.load_config_readonly", return_value={}):
            assert local_network.network_outage_max_wait_seconds() == 120.0

    def test_max_wait_reads_config(self):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"agent": {"network_outage_max_wait_seconds": 30}},
        ):
            assert local_network.network_outage_max_wait_seconds() == 30.0

    def test_max_wait_zero_disables_the_hold(self):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"agent": {"network_outage_max_wait_seconds": 0}},
        ):
            assert local_network.network_outage_max_wait_seconds() == 0.0

    def test_plan_is_clamped_to_the_remaining_budget(self):
        """The plan is what the caller both charges and sleeps — never more than the budget."""
        assert local_network.outage_wait_plan(0.0, 1, cap=7.0) == 5.0
        assert local_network.outage_wait_plan(5.0, 2, cap=7.0) == 2.0   # non-aligned remainder
        assert local_network.outage_wait_plan(7.0, 3, cap=7.0) == 0.0
        assert local_network.outage_wait_plan(0.0, 1, cap=0.0) == 0.0


class _StubAgent:
    """Just enough AIAgent surface for the outage hold and the retry backoff."""

    def __init__(self) -> None:
        self.model, self.provider = "deepseek-flash", "deepseek"
        self.buffered: list[str] = []
        self.notices: list[str] = []

    def _buffer_status(self, message: str) -> None:
        self.buffered.append(message)

    def _buffer_diagnostic_status(self, message: str) -> None:
        self.buffered.append(message)

    def _emit_status(self, message: str) -> None:
        self.buffered.append(message)

    def _emit_diagnostic_status(self, message: str) -> None:
        self.notices.append(message)

    def _emit_wait_notice(self, message: str) -> None:
        self.notices.append(message)

    def _emit_diagnostic_wait(self, message: str) -> None:
        self.notices.append(message)

    def _client_log_context(self) -> str:
        return ""


class TestHoldWhileOffline:
    def test_hold_reserves_exactly_what_it_charges(self):
        from agent import turn_recovery

        agent, retry = _StubAgent(), TurnRetryState()
        with (
            patch.object(turn_recovery, "network_outage_max_wait_seconds", return_value=120.0),
            patch.object(turn_recovery, "outage_wait_plan", return_value=30.0),
        ):
            ceiling = turn_recovery._hold_primary_while_offline(
                agent, retry, retry_count=4, max_retries=3,
            )

        # The outage, not the provider, is spending the budget: one more attempt is allowed, and
        # the seconds reserved for it are recorded once (they are the seconds the loop will sleep).
        assert ceiling == 5
        assert retry.network_outage_waited == 30.0
        assert retry.network_outage_sleep == 30.0
        assert any("No internet access" in line for line in agent.buffered)

    def test_hold_stops_once_the_wait_budget_is_spent(self):
        from agent import turn_recovery

        agent, retry = _StubAgent(), TurnRetryState()
        retry.network_outage_waited = 130.0
        with patch.object(turn_recovery, "network_outage_max_wait_seconds", return_value=120.0):
            ceiling = turn_recovery._hold_primary_while_offline(
                agent, retry, retry_count=4, max_retries=3,
            )

        assert ceiling == 3
        assert retry.network_outage_waited == 130.0
        assert retry.network_outage_sleep == 0.0
        assert agent.buffered == []

    def test_zero_budget_fails_fast(self):
        from agent import turn_recovery

        agent, retry = _StubAgent(), TurnRetryState()
        with patch.object(turn_recovery, "network_outage_max_wait_seconds", return_value=0.0):
            assert turn_recovery._hold_primary_while_offline(
                agent, retry, retry_count=4, max_retries=3,
            ) == 3

    def test_cumulative_sleep_never_exceeds_the_cap(self):
        """The cap bounds the wall clock: what is charged is what is slept, and never more."""
        from agent import turn_recovery

        agent, retry = _StubAgent(), TurnRetryState()
        error = _dns_failure()
        slept: list[float] = []

        with (
            patch.object(turn_recovery, "network_outage_max_wait_seconds", return_value=7.0),
            patch.object(turn_recovery, "local_network_outage", return_value=True),
            # Keep the *provider-fault* backoff out of the measurement: this test is about the
            # outage ledger, and the normal schedule is a different (unrelated) wait.
            patch("agent.retry_utils.jittered_backoff", return_value=0.0),
        ):
            for attempt in range(1, 6):
                ceiling = turn_recovery._hold_primary_while_offline(
                    agent, retry, retry_count=attempt, max_retries=3,
                )
                slept.append(turn_recovery.compute_error_backoff(
                    agent, error, retry_count=attempt, max_retries=ceiling,
                    is_rate_limited=False, is_zai_coding_overload=False,
                    base_url="https://api.deepseek.com/v1", model="deepseek-flash",
                    outage_state=retry,
                ))

        assert slept[:2] == [5.0, 2.0], "5s then the 2s remainder — the cap is a wall-clock bound"
        assert retry.network_outage_waited == 7.0, "the outage ledger never exceeds the cap"
        assert retry.network_outage_sleep == 0.0, "a spent budget reserves no further outage wait"
        assert max(slept[2:]) == 0.0


class _RouteAgent(_StubAgent):
    """AIAgent surface used by route_classified_error's eager-fallback gate."""

    def __init__(self) -> None:
        super().__init__()
        self._fallback_index = 0
        self._fallback_chain = [{"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"}]
        self._credential_pool = None
        self.attempts: list[bool] = []
        self.activated = 0

    def _has_pending_fallback(self) -> bool:
        return self._fallback_index < len(self._fallback_chain)

    def _try_activate_fallback(self, reason=None, *, suppress_external_network: bool = False) -> bool:
        self.attempts.append(suppress_external_network)
        # An all-external chain finds no eligible candidate while the machine is offline.
        if suppress_external_network:
            return False
        self.activated += 1
        return True


def _route(agent, error, *, retry_count: int, max_retries: int = 3, base_url: str = "https://api.deepseek.com/v1"):
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import route_classified_error

    classified = classify_api_error(
        error, provider=agent.provider, model=agent.model, approx_tokens=1000, context_length=1048576,
    )
    assert classified.reason is FailoverReason.timeout, classified.reason
    return route_classified_error(
        agent, error, classified, TurnRetryState(), error_msg=str(error), error_context={},
        recovered_with_pool=False, base_url=base_url, model=agent.model,
        messages=[], api_messages=[], system_message="", active_system_prompt="",
        conversation_history=[], retry_count=retry_count, max_retries=max_retries,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1, effective_task_id="t",
    )


class TestEagerFallbackGate:
    def test_offline_transport_failure_suppresses_only_external_routes(self):
        agent = _RouteAgent()
        with patch.object(local_network, "probe_connectivity", return_value=False):
            verdict = _route(agent, _dns_failure(), retry_count=4)

        assert verdict.action == "fallthrough", "the turn must keep retrying the primary route"
        assert agent.attempts == [True], "the walk runs with external routes suppressed"
        assert agent.activated == 0
        assert verdict.max_retries == 5, "the retry ceiling must make room for the outage wait"

    def test_provider_side_transport_failure_falls_back_unsuppressed(self):
        agent = _RouteAgent()
        with patch.object(local_network, "probe_connectivity", return_value=True):
            verdict = _route(agent, httpx.ConnectError("Connection error."), retry_count=2)

        assert agent.attempts == [False]
        assert agent.activated == 1
        assert verdict.action == "break"

    def test_local_current_route_is_exempt_from_the_outage_hold(self):
        """A loopback route does not need the missing network: no wait, just its own failure."""
        agent = _RouteAgent()
        agent.provider = "custom"
        with patch.object(local_network, "probe_connectivity", return_value=False):
            verdict = _route(
                agent, httpx.ConnectError("Connection error."), retry_count=4,
                base_url="http://127.0.0.1:8000/v1",
            )

        assert verdict.max_retries == 3, "a local route must not extend the retry budget for the WAN"
        assert not any("No internet access" in line for line in agent.buffered)
        # Suppression still applies to the *candidates*: an external fallback stays out of play.
        assert agent.attempts == [True]


class TestAuxRouteScope:
    def test_offline_skips_external_aux_entries_and_keeps_loopback(self):
        from agent import auxiliary_client

        chain = [
            {"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"},
            {"provider": "custom", "model": "local-summarizer", "base_url": "http://127.0.0.1:8000/v1"},
        ]
        sentinel = object()
        with (
            patch.object(auxiliary_client, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
            patch.object(auxiliary_client, "_resolve_fallback_entry", return_value=(sentinel, "local-summarizer")),
            patch.object(auxiliary_client, "_candidate_context_window", return_value=None),
        ):
            client, model, label = auxiliary_client._try_configured_fallback_chain(
                "compression", "deepseek", suppress_external=True,
            )

        assert client is sentinel, "the loopback entry must stay eligible while the WAN is down"
        assert model == "local-summarizer"
        assert label == "fallback_chain[1](custom)"

    def test_online_keeps_the_configured_order(self):
        from agent import auxiliary_client

        chain = [
            {"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"},
            {"provider": "custom", "model": "local-summarizer", "base_url": "http://127.0.0.1:8000/v1"},
        ]
        sentinel = object()
        with (
            patch.object(auxiliary_client, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
            patch.object(auxiliary_client, "_resolve_fallback_entry", return_value=(sentinel, "cloud")),
            patch.object(auxiliary_client, "_candidate_context_window", return_value=None),
        ):
            _, model, label = auxiliary_client._try_configured_fallback_chain(
                "compression", "deepseek", suppress_external=False,
            )

        assert label == "fallback_chain[0](openrouter)"
        assert model == "cloud"

    def test_offline_with_only_external_entries_yields_nothing(self):
        from agent import auxiliary_client

        chain = [{"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"}]
        with (
            patch.object(auxiliary_client, "_get_auxiliary_task_config", return_value={"fallback_chain": chain}),
            patch.object(auxiliary_client, "_resolve_fallback_entry", return_value=(object(), "cloud")),
            patch.object(auxiliary_client, "_candidate_context_window", return_value=None),
        ):
            client, model, label = auxiliary_client._try_configured_fallback_chain(
                "compression", "deepseek", suppress_external=True,
            )

        assert (client, model, label) == (None, None, "")


class TestAsyncLaneLiveness:
    @pytest.mark.asyncio
    async def test_classification_on_the_loop_does_not_touch_a_socket(self):
        """The ladder is advanced by _drive_ladder_async: classification must never block the loop."""
        ticks = {"n": 0}

        async def _heartbeat() -> None:
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0.005)

        beat = asyncio.create_task(_heartbeat())
        await asyncio.sleep(0.02)
        before = ticks["n"]
        try:
            with patch.object(
                local_network, "_probe_uncached", side_effect=AssertionError("socket work on the loop"),
            ) as raw_probe:
                verdict = local_network.local_network_outage(_dns_failure(), allow_probe=False)
                await asyncio.sleep(0.02)
        finally:
            beat.cancel()

        assert verdict is True
        raw_probe.assert_not_called()
        assert ticks["n"] > before, "the event loop kept running through classification"

    @pytest.mark.asyncio
    async def test_off_loop_probe_keeps_the_loop_responsive(self):
        """A refresh that blocks in a worker thread must not stall unrelated coroutines."""
        ticks = {"n": 0}

        async def _heartbeat() -> None:
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0.005)

        def _slow_probe(**_kwargs):
            time.sleep(0.2)
            return False

        beat = asyncio.create_task(_heartbeat())
        await asyncio.sleep(0.02)
        before = ticks["n"]
        try:
            with patch.object(local_network, "_probe_uncached", side_effect=_slow_probe):
                assert await local_network.probe_connectivity_async() is False
        finally:
            beat.cancel()

        assert ticks["n"] > before + 5, "the heartbeat must keep ticking while the probe blocks off-loop"
