"""Tests for Honcho session-end auto-dream.

Every positive case is paired with a negative control: a hook that always fires
is indistinguishable from one that fires for the wrong reason.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest


class _InlineThread:
    """Run the thread body inline so assertions are deterministic."""

    def __init__(self, target):
        self._target = target

    def start(self):
        self._target()

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager


class _FakeClient:
    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def schedule_dream(self, **kwargs):
        if self.fail:
            raise RuntimeError("honcho unreachable")
        self.calls.append(kwargs)


def _config(
    *,
    auto_dream: bool = True,
    auto_dream_min_interval_seconds: int = 28800,
    peer_name: str = "kai",
    ai_peer: str = "hermes",
) -> HonchoClientConfig:
    """Real config object, so a field rename breaks these tests loudly."""
    return HonchoClientConfig(
        auto_dream=auto_dream,
        auto_dream_min_interval_seconds=auto_dream_min_interval_seconds,
        peer_name=peer_name,
        ai_peer=ai_peer,
    )


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    """Manager wired to a fake client, with the stamp isolated to tmp_path."""
    client = _FakeClient()
    m = HonchoSessionManager(honcho=client, config=_config())
    monkeypatch.setattr(
        m, "_auto_dream_stamp_path",
        lambda: tmp_path / "honcho-auto-dream.stamp",
    )
    # Run the dispatch thread body inline so assertions are deterministic.
    monkeypatch.setattr(
        "plugins.memory.honcho.session.threading.Thread",
        lambda target, name=None, daemon=None: _InlineThread(target),
    )
    return m, client


# --- it fires when it should ------------------------------------------------

def test_dispatches_a_dream_for_both_directions(mgr):
    m, client = mgr
    assert m.maybe_schedule_dream() is True
    assert client.calls == [
        {"observer": "kai"},
        {"observer": "kai", "observed": "hermes"},
    ]


def test_fires_again_once_the_interval_has_elapsed(mgr, tmp_path):
    m, client = mgr
    assert m.maybe_schedule_dream() is True
    (tmp_path / "honcho-auto-dream.stamp").write_text(str(time.time() - 30000))
    assert m.maybe_schedule_dream() is True
    assert len(client.calls) == 4


# --- negative controls: it must DECLINE -------------------------------------

def test_declines_when_auto_dream_is_off(tmp_path, monkeypatch):
    client = _FakeClient()
    m = HonchoSessionManager(honcho=client, config=_config(auto_dream=False))
    monkeypatch.setattr(m, "_auto_dream_stamp_path",
                        lambda: tmp_path / "s.stamp")
    assert m.maybe_schedule_dream() is False
    assert client.calls == []


def test_declines_while_throttled(mgr):
    m, client = mgr
    assert m.maybe_schedule_dream() is True
    assert m.maybe_schedule_dream() is False, "second call inside the interval"
    assert len(client.calls) == 2, "no extra dream was dispatched"


def test_declines_without_a_config(tmp_path, monkeypatch):
    m = HonchoSessionManager(honcho=_FakeClient(), config=None)
    monkeypatch.setattr(m, "_auto_dream_stamp_path", lambda: tmp_path / "s")
    assert m.maybe_schedule_dream() is False


# --- robustness -------------------------------------------------------------

def test_a_failing_dream_never_raises(tmp_path, monkeypatch):
    """A dead Honcho must not surface into session shutdown."""
    client = _FakeClient(fail=True)
    m = HonchoSessionManager(honcho=client, config=_config())
    monkeypatch.setattr(m, "_auto_dream_stamp_path", lambda: tmp_path / "s")
    monkeypatch.setattr(
        "plugins.memory.honcho.session.threading.Thread",
        lambda target, name=None, daemon=None: _InlineThread(target),
    )
    assert m.maybe_schedule_dream() is True   # dispatched; the failure is swallowed


def test_stamp_is_written_before_dispatch(tmp_path, monkeypatch):
    """A burst of session ends must not queue a burst of dreams.

    The stamp has to land even if the request itself is slow or fails, otherwise
    concurrent shutdowns each see "no stamp" and all dispatch.
    """
    client = _FakeClient(fail=True)
    m = HonchoSessionManager(honcho=client, config=_config())
    stamp = tmp_path / "s.stamp"
    monkeypatch.setattr(m, "_auto_dream_stamp_path", lambda: stamp)
    monkeypatch.setattr(
        "plugins.memory.honcho.session.threading.Thread",
        lambda target, name=None, daemon=None: _InlineThread(target),
    )
    m.maybe_schedule_dream()
    assert stamp.exists(), "stamp must be written even when the dream fails"


def test_single_peer_setup_does_not_dream_against_itself(tmp_path, monkeypatch):
    client = _FakeClient()
    m = HonchoSessionManager(honcho=client, config=_config(ai_peer="kai"))
    monkeypatch.setattr(m, "_auto_dream_stamp_path", lambda: tmp_path / "s")
    monkeypatch.setattr(
        "plugins.memory.honcho.session.threading.Thread",
        lambda target, name=None, daemon=None: _InlineThread(target),
    )
    m.maybe_schedule_dream()
    assert client.calls == [{"observer": "kai"}], "no self-targeted second dream"



def test_provider_session_end_triggers_the_dream(tmp_path, monkeypatch):
    """The unit tests call maybe_schedule_dream() directly; this proves the
    PROVIDER actually reaches it from on_session_end, which is the only path
    that runs in production.
    """
    import plugins.memory.honcho as honcho_pkg

    provider_cls = honcho_pkg.HonchoMemoryProvider

    client = _FakeClient()
    mgr = HonchoSessionManager(honcho=client, config=_config())
    monkeypatch.setattr(mgr, "_auto_dream_stamp_path", lambda: tmp_path / "s")
    monkeypatch.setattr(
        "plugins.memory.honcho.session.threading.Thread",
        lambda target, name=None, daemon=None: _InlineThread(target),
    )

    p = provider_cls.__new__(provider_cls)
    p._cron_skipped = False
    p._manager = mgr
    p._session_initialized = True
    p._init_thread = None
    p._sync_thread = None

    p.on_session_end([{"role": "user", "content": "x"}] * 20)

    assert client.calls == [
        {"observer": "kai"},
        {"observer": "kai", "observed": "hermes"},
    ], "on_session_end must reach schedule_dream"


def test_provider_session_end_respects_the_off_switch(tmp_path, monkeypatch):
    """Negative control for the test above: same path, autoDream off, silence."""
    import plugins.memory.honcho as honcho_pkg

    provider_cls = honcho_pkg.HonchoMemoryProvider

    client = _FakeClient()
    mgr = HonchoSessionManager(honcho=client, config=_config(auto_dream=False))
    monkeypatch.setattr(mgr, "_auto_dream_stamp_path", lambda: tmp_path / "s")

    p = provider_cls.__new__(provider_cls)
    p._cron_skipped = False
    p._manager = mgr
    p._session_initialized = True
    p._init_thread = None
    p._sync_thread = None

    p.on_session_end([{"role": "user", "content": "x"}] * 20)

    assert client.calls == [], "autoDream=False must keep the session-end path silent"
