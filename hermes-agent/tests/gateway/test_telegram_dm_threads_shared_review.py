"""Review follow-ups for ``dm_threads: shared``: per-thread batches, the routed profile's topic state,
hot-reloaded topics, provenance-keeping copies and rebuilt sources resolving their own bot's config."""

import weakref
from pathlib import Path
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_dm_threads import dm_thread_key_source, telegram_extra_for
from gateway.session_identity import RoutingIdentity
from tests.gateway.test_base_topic_sessions import DummyTelegramAdapter
from tests.gateway.test_telegram_dm_threads_shared import _dm, _store

SHARED = {"dm_threads": "shared"}


def _adapter(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, dict(SHARED))
    adapter = DummyTelegramAdapter()
    adapter.config.extra["dm_threads"] = "shared"
    adapter.set_session_store(store)
    return adapter, store


def test_shared_dm_text_batches_stay_per_thread(tmp_path, monkeypatch):
    """One session, but a batch keeps its first event's source: two threads' texts in one batch
    would get one answer in the first thread only."""
    adapter, _ = _adapter(tmp_path, monkeypatch)
    events = [MessageEvent(text="q", source=_dm(tid)) for tid in ("501", "777")]
    assert len({adapter._event_session_key(e) for e in events}) == 1
    assert len({adapter._text_batch_key(e) for e in events}) == 2


def test_topic_mode_read_from_the_routed_profiles_store(tmp_path, monkeypatch):
    """A primary bot routed to profile ``work`` keys before the runner enters that scope: /topic on
    in ``work`` (not in the launch profile) must still keep each topic its own session."""
    from hermes_state_registry import acquire
    adapter, _ = _adapter(tmp_path, monkeypatch)
    work = tmp_path / "profiles" / "work"
    work.mkdir(parents=True)
    acquire(work / "state.db").enable_telegram_topic_mode(chat_id="111", user_id="111", profile_name="work")
    identity = RoutingIdentity(transport_profile="default", runtime_profile="work",
                               authorization_home=tmp_path, runtime_home=work)

    def routed(tid):
        source = _dm(tid)
        source.profile = "work"
        object.__setattr__(source, "_identity", identity)
        return MessageEvent(text="q", source=source)

    assert len({adapter._event_session_key(routed(tid)) for tid in ("501", "777")}) == 2


def test_hot_reloaded_dm_topic_stays_isolated(monkeypatch):
    from tests.gateway.conftest import _ensure_telegram_mock
    _ensure_telegram_mock()
    from plugins.platforms.telegram.adapter import TelegramAdapter
    import hermes_cli.config as cfg
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=dict(SHARED)))
    topics = [{"chat_id": 111, "topics": [{"name": "W", "thread_id": 501}]}]
    monkeypatch.setattr(cfg, "load_config", lambda: {"platforms": {"telegram": {"extra": {"dm_topics": topics}}}})
    adapter._reload_dm_topics_from_config()
    assert dm_thread_key_source(_dm("501"), adapter.config.extra).thread_id == "501"


def test_shared_copy_keeps_provenance():
    source = _dm("501")
    marker = DummyTelegramAdapter()
    object.__setattr__(source, "_transport_adapter_ref", weakref.ref(marker))
    keyed = dm_thread_key_source(source, SHARED, lambda: None)
    assert keyed.thread_id is None and keyed._transport_adapter_ref() is marker


def test_rebuilt_source_uses_the_answering_bots_settings(monkeypatch):
    """A /loop tick rebuilds its source with no live adapter ref: the secondary bot that answers it
    sets ``shared`` while the launch bot keeps ``per_thread``; its key must match live traffic."""
    from gateway import run
    secondary = SimpleNamespace(platform=Platform.TELEGRAM, config=SimpleNamespace(extra=dict(SHARED)))
    runner = SimpleNamespace(_delivery_adapter_for=lambda source: secondary)
    monkeypatch.setattr(run, "_gateway_runner_ref", lambda: runner)
    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t", extra={})})
    assert telegram_extra_for(_dm("501"), config) == SHARED
