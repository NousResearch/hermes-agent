"""Raw standalone runners honor routing denials without retry or degradation."""
import asyncio
import logging
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from hermes_cli.routing_policy import RoutingPolicyError


@pytest.fixture
def policy(monkeypatch):
    from hermes_cli import routing_policy
    monkeypatch.setattr(routing_policy, 'current_routing_policy', lambda: {
        'enabled': True, 'deny': {'base_url_hosts': ['forbidden.invalid']}})


@pytest.mark.parametrize('terminal', [False, True])
def test_mini_raw_send_and_terminal_error(policy, terminal):
    from mini_swe_runner import MiniSWERunner
    runner = MiniSWERunner.__new__(MiniSWERunner)
    send = Mock(side_effect=RoutingPolicyError('denied'))
    runner.client = NS(base_url='https://allowed.invalid' if terminal else 'https://forbidden.invalid',
                       chat=NS(completions=NS(create=send)))
    runner.model, runner.tools, runner.logger = 'allowed', [], logging.getLogger('test')
    with pytest.raises(RoutingPolicyError):
        runner._call_model([])
    assert send.call_count == int(terminal)


@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
def test_trajectory_raw_send_and_terminal_error(policy, monkeypatch, async_mode, terminal):
    from trajectory_compressor import TrajectoryCompressor, CompressionConfig, TrajectoryMetrics
    comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
    comp.config = CompressionConfig(max_retries=2, retry_delay=0)
    comp.logger = logging.getLogger('test')
    monkeypatch.setattr('trajectory_compressor.jittered_backoff', lambda *a, **kw: 0)
    calls = []
    def send(**kw):
        calls.append(kw)
        raise RoutingPolicyError('denied')
    async def asend(**kw):
        return send(**kw)
    wire = NS(base_url='https://allowed.invalid' if terminal else 'https://forbidden.invalid',
              chat=NS(completions=NS(create=asend if async_mode else send)))
    comp.client = wire
    monkeypatch.setattr(comp, '_get_async_client', lambda: wire)
    with pytest.raises(RoutingPolicyError):
        if async_mode:
            asyncio.run(comp._generate_summary_async('content', TrajectoryMetrics()))
        else:
            comp._generate_summary('content', TrajectoryMetrics())
    assert len(calls) == int(terminal)


def test_mini_batch_denial_stops_before_next_task(policy, tmp_path):
    from mini_swe_runner import MiniSWERunner
    runner = MiniSWERunner.__new__(MiniSWERunner)
    runner.logger = logging.getLogger('test')
    runner.run_task = Mock(side_effect=RoutingPolicyError('denied'))
    with pytest.raises(RoutingPolicyError):
        runner.run_batch(['first', 'second'], str(tmp_path / 'results.jsonl'))
    assert runner.run_task.call_count == 1


def test_trajectory_directory_worker_denial_propagates(policy):
    from trajectory_compressor import TrajectoryCompressor, CompressionConfig, AggregateMetrics, _RunProgress
    comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
    comp.config, comp.aggregate_metrics = CompressionConfig(), AggregateMetrics()
    comp.logger = logging.getLogger('test')
    async def denied(entry):
        raise RoutingPolicyError('denied')
    comp.process_entry_async = denied
    async def run():
        progress = _RunProgress(Mock(), None, None, asyncio.Lock(), asyncio.Semaphore(1))
        with pytest.raises(RoutingPolicyError):
            await comp._process_one(progress, 'source.jsonl', 0, {})
    asyncio.run(run())
