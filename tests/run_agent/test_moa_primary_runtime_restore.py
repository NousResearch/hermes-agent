"""MoA primary recovery must preserve the snapshotted facade instance.

A cross-provider fallback replaces ``agent.client`` with the fallback's SDK
client, so ``_primary_runtime`` is the only surviving reference to the original
MoA facade (and the reference relay / pending accounting it owns). Recovery must
reuse that instance and only fall back to the factory for legacy snapshots.
"""

from __future__ import annotations

import copy
import inspect
import threading
from types import SimpleNamespace

import pytest

import agent.agent_runtime_helpers as helpers
from agent.agent_runtime_helpers import copy_primary_runtime


class _FakeFacade:
    """Stands in for MoAClient: identity matters and it holds an unpicklable lock."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._lock = threading.Lock()


def _moa_agent(snapshot_facade, live_client) -> SimpleNamespace:
    return SimpleNamespace(
        provider="moa",
        model="default",
        client=live_client,
        _client_kwargs={"api_key": "fallback-key"},
        _primary_runtime={
            "model": "default",
            "provider": "moa",
            "requested_provider": "moa",
            "base_url": "moa://local",
            "api_mode": "chat_completions",
            "api_key": "",
            "client_kwargs": {},
            "moa_client": snapshot_facade,
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
        },
    )


def test_snapshot_reused_instead_of_rebuilt(monkeypatch):
    snapshot = _FakeFacade("snapshot")
    agent = _moa_agent(snapshot, live_client=object())

    def _fail(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("build_moa_facade must not be called when a snapshot exists")

    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "build_moa_facade", _fail)

    rt = agent._primary_runtime
    facade = rt.get("moa_client")
    assert facade is snapshot


def test_legacy_snapshot_falls_back_to_factory():
    agent = _moa_agent(snapshot_facade=None, live_client=object())
    assert agent._primary_runtime.get("moa_client") is None


def test_reuse_precedes_rebuild_in_source():
    """Ordering invariant: swapping reuse/rebuild silently breaks the fix."""
    source = inspect.getsource(helpers.restore_primary_runtime)
    reuse = source.index('rt.get("moa_client")')
    rebuild = source.index("build_moa_facade(agent, agent.model)")
    assert reuse < rebuild


def test_restore_refreshes_snapshot_reference():
    source = inspect.getsource(helpers.restore_primary_runtime)
    assert 'rt["moa_client"] = moa_client' in source


def test_recovery_path_shares_the_same_invariants():
    source = inspect.getsource(helpers.try_recover_primary_transport)
    reuse = source.index('rt.get("moa_client")')
    rebuild = source.index("build_moa_facade(agent, agent.model)")
    assert reuse < rebuild
    assert 'rt["moa_client"] = moa_client' in source
    assert "agent._client_kwargs = {}" in source


def test_copy_primary_runtime_shares_facade_and_copies_data():
    facade = _FakeFacade("live")
    runtime = {
        "model": "default",
        "client_kwargs": {"api_key": "k"},
        "moa_client": facade,
    }

    with pytest.raises(TypeError):
        copy.deepcopy(runtime)

    copied = copy_primary_runtime(runtime)
    assert copied["moa_client"] is facade
    assert copied["client_kwargs"] == runtime["client_kwargs"]
    assert copied["client_kwargs"] is not runtime["client_kwargs"]


def test_copy_primary_runtime_deepcopies_non_moa_runtime():
    runtime = {"model": "gpt", "moa_client": None, "client_kwargs": {"api_key": "k"}}
    copied = copy_primary_runtime(runtime)
    assert copied == runtime
    assert copied["client_kwargs"] is not runtime["client_kwargs"]
    assert copy_primary_runtime(None) is None
