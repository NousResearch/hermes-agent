"""A resume / ``session.info`` snapshot carries the session's own status-bar usage, so a viewer that switches
to another session shows that session's figures instead of keeping the previous one's. An agent that has called
the model answers live; otherwise the stored row answers, and since nothing persists latency it invents no t/s."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


async def _authority_with_stored_usage(tmp_path, monkeypatch, agent):
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    authority.runner._cached_agent_for = lambda route: agent
    authority.db.update_token_counts('s', input_tokens=150, output_tokens=40, cache_read_tokens=800,
                                     cache_write_tokens=50, api_call_count=3, model='m')
    return authority


async def _snapshot_usage(tmp_path, monkeypatch, agent):
    from gateway.session_contract import SessionRef
    from gateway.session_local_inventory import snapshot_info
    authority = await _authority_with_stored_usage(tmp_path, monkeypatch, agent)
    return (await snapshot_info(authority, SessionRef('p', 's')))['usage']


def _assert_stored_figures(usage):
    assert usage['cache_hit_pct'] == 80  # 800 read / (150 + 800 + 50) prompt
    assert usage['calls'] == 3 and usage['input'] == 150 and usage['output'] == 40
    assert 'avg_tps' not in usage and 'avg_latency_s' not in usage


@pytest.mark.asyncio
async def test_snapshot_with_an_unused_agent_carries_the_stored_cache_hit_rate(tmp_path, monkeypatch):
    _assert_stored_figures(await _snapshot_usage(tmp_path, monkeypatch, SimpleNamespace(model='m', session_api_calls=0)))


@pytest.mark.asyncio
async def test_an_old_session_with_no_agent_answers_from_its_stored_row(tmp_path, monkeypatch):
    from gateway.session_usage import snapshot_usage
    authority = await _authority_with_stored_usage(tmp_path, monkeypatch, None)
    _assert_stored_figures(snapshot_usage(authority.db, 's', None))


@pytest.mark.asyncio
async def test_snapshot_with_live_calls_carries_the_agent_figures(tmp_path, monkeypatch):
    agent = SimpleNamespace(model='m', session_api_calls=2, session_prompt_tokens=1000, session_cache_read_tokens=500,
                            _api_latency_history=[1.0, 1.0], _api_output_history=[40, 60])

    usage = await _snapshot_usage(tmp_path, monkeypatch, agent)

    assert usage['cache_hit_pct'] == 50 and usage['calls'] == 2
    assert usage['avg_tps'] == 50.0
