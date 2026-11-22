"""Summary-model recovery deadlines survive worker rebuilds (regression for #158813)."""

from contextlib import closing
from types import SimpleNamespace

import pytest

from agent import context_compressor_prellm
from agent.context_compressor import ContextCompressor
from persistence.sqlite.conversation._store.sessions import get_session_model_config_value
from tests.persistence_sqlite.workers._runtime_worker import worker as worker  # noqa: PLC0414 -- pytest fixture import
from tests.persistence_sqlite.workers._runtime_worker import worker_store


def _compressor(store):
    compressor = ContextCompressor(model="fixture", config_context_length=100000, quiet_mode=True)
    compressor.bind_session_state(store, "owned")
    return compressor


@pytest.mark.parametrize("probe_falls_back", [False, True], ids=["recovered", "still-failing"])
def test_summary_probe_recovers_across_worker_rebuilds(worker, monkeypatch, probe_falls_back):
    persistence, store = worker
    holder = persistence.holder()
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(context_compressor_prellm, "time", SimpleNamespace(time=lambda: clock.now))
    compressor = _compressor(store)
    compressor.record_completed_compaction(used_fallback=True)
    compressor.record_completed_compaction(used_fallback=True)

    assert compressor._fallback_streak_skip({})
    deadline = clock.now + compressor._ANTI_THRASH_RECOVERY_SECONDS
    assert get_session_model_config_value(holder, "owned", "_fallback_probe_at") == deadline
    assert store.failure is None

    # Reopen the worker journal as well as rebuilding the compressor: neither may carry
    # the deadline only in memory, and another compression must not restart its window.
    store.close()
    with closing(worker_store(holder, store.scope, store.path.parent)) as resumed:
        clock.now = deadline - 1
        waiting = _compressor(resumed)
        assert waiting._fallback_streak_skip({})
        waiting.record_completed_compaction(feasibility_skip=True)
        assert get_session_model_config_value(holder, "owned", "_fallback_probe_at") == deadline

        clock.now = deadline + 1
        probing = _compressor(resumed)
        assert not probing._fallback_streak_skip({})
        assert get_session_model_config_value(holder, "owned", "_fallback_probe_at") is None
        assert resumed.failure is None

        # A healthy probe lifts the bench; another fallback starts one new recovery
        # window, which must itself survive the next rebuilt agent.
        probing.record_completed_compaction(used_fallback=probe_falls_back)
        after_probe = _compressor(resumed)
        assert after_probe._fallback_streak_skip({}) is probe_falls_back
        next_deadline = clock.now + compressor._ANTI_THRASH_RECOVERY_SECONDS if probe_falls_back else None
        assert get_session_model_config_value(holder, "owned", "_fallback_probe_at") == next_deadline
        clock.now += 1
        assert _compressor(resumed)._fallback_streak_skip({}) is probe_falls_back
        assert get_session_model_config_value(holder, "owned", "_fallback_probe_at") == next_deadline
        assert resumed.failure is None
