"""Automatic Hindsight work follows input origin; explicit tools remain available."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.memory_manager import MemoryManager
from agent.turn_origin import is_user_input_turn, turn_input_scope
from plugins.memory import load_memory_provider
from run_agent import AIAgent


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_dir = tmp_path / "hindsight"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({
        "mode": "local_external", "api_url": "http://localhost:9999",
        "bank_id": "test-bank", "memory_mode": "hybrid", "recall_sync": True,
    }), encoding="utf-8")
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("plugins.memory.hindsight._maybe_upgrade_client", lambda: None)
    # Real discovery, construction and config loading; only the external backend is replaced.
    p = load_memory_provider("hindsight", register_skills=False)
    assert p is not None
    p.initialize(session_id="test-session", platform="cli")
    p._prefetch_waits_for_retain = False
    monkeypatch.setattr(p, "_do_recall", Mock(return_value=("- relevant memory", 1)))
    monkeypatch.setattr(p, "_resolve_retain_target", lambda _doc: ("test-document", None))
    monkeypatch.setattr(p, "_enqueue_retain", Mock())
    yield p
    p.shutdown()


@pytest.mark.parametrize("kind", ["process_complete", "async_delegation_complete", "internal_notification"])
@pytest.mark.parametrize("sync", [False, True])
def test_runtime_turn_skips_all_automatic_paths_and_preserves_cached_recall(provider, kind, sync):
    p = provider
    p._recall_sync = sync
    p._prefetch_result, p._prefetch_count = "- earlier human memory", 1
    p._last_recall_returned, p._last_recall_count = True, 1
    with turn_input_scope(display_kind=kind):
        assert p.prefetch("runtime notification") == ""
        p.queue_prefetch("runtime notification")
        p.sync_turn("runtime notification", "notification response")
        assert p.recall_status() is None
    p._do_recall.assert_not_called()
    p._enqueue_retain.assert_not_called()
    assert p._prefetch_thread is None
    assert p._prefetch_result == "- earlier human memory"
    assert p._prefetch_count == 1
    assert p._turn_counter == 0
    assert p._session_turns == []
    # The next real input uses normal settings; async mode consumes the original cache.
    with turn_input_scope():
        result = p.prefetch("human question")
        p.sync_turn("human question", "answer")
    assert ("relevant memory" if sync else "earlier human memory") in result
    assert p._enqueue_retain.call_count == 1


def test_runtime_turn_does_not_advance_retain_cadence(provider):
    p = provider
    p._retain_every_n_turns = 2
    with turn_input_scope():
        p.sync_turn("first human question", "answer")
    with turn_input_scope(display_kind="internal_notification"):
        p.sync_turn("handoff notification", "response")
    assert p._turn_counter == 1
    p._enqueue_retain.assert_not_called()
    with turn_input_scope():
        p.sync_turn("second human question", "answer")
    assert p._turn_counter == 2
    p._enqueue_retain.assert_called_once()
    assert "handoff notification" not in "".join(p._session_turns)


def test_real_background_prefetch_captures_the_human_turn(provider):
    p = provider
    p._recall_sync = False
    entered, release = threading.Event(), threading.Event()
    origins = []

    def recall(query):
        entered.set()
        assert release.wait(3)
        origins.append((query, is_user_input_turn()))
        return "- queued human memory", 1

    p._do_recall.side_effect = recall
    try:
        with turn_input_scope():
            p.queue_prefetch("human question")
        assert entered.wait(3)
        with turn_input_scope(display_kind="process_complete"):
            release.set()
            p._prefetch_thread.join(timeout=3)
            assert not p._prefetch_thread.is_alive()
            assert p.prefetch("process completed") == ""
        assert origins == [("human question", True)]
        with turn_input_scope():
            assert "queued human memory" in p.prefetch("next question")
    finally:
        release.set()


def test_serialized_manager_queue_keeps_each_turn_origin(provider):
    p, manager = provider, MemoryManager()
    manager.add_provider(p)
    entered, release = threading.Event(), threading.Event()

    def hold_worker():
        entered.set()
        assert release.wait(3)

    try:
        manager._submit_background(hold_worker)
        assert entered.wait(3)
        with turn_input_scope():
            manager.sync_all("queued human question", "answer")
        with turn_input_scope(display_kind="async_delegation_complete"):
            manager.sync_all("delegation notification", "response")
            release.set()
            assert manager.flush_pending(timeout=3)
        p._enqueue_retain.assert_called_once()
        saved = "".join(p._session_turns)
        assert "queued human question" in saved
        assert "delegation notification" not in saved
    finally:
        release.set()
        manager.shutdown_all()


def test_explicit_tools_still_work_on_a_non_user_turn(provider, monkeypatch):
    p = provider
    recall = Mock(return_value=[SimpleNamespace(text="explicit recall")])
    reflect = Mock(return_value="explicit reflection")
    retain = Mock()
    monkeypatch.setattr(p, "_recall", recall)
    monkeypatch.setattr(p, "_reflect", reflect)
    monkeypatch.setattr(p, "_retain_batch", retain)
    manager = MemoryManager()
    manager.add_provider(p)
    with turn_input_scope(display_kind="internal_notification"):
        assert "explicit recall" in json.loads(manager.handle_tool_call("hindsight_recall", {"query": "q"}))["result"]
        assert "explicit reflection" in json.loads(manager.handle_tool_call("hindsight_reflect", {"query": "q"}))["result"]
        result = json.loads(manager.handle_tool_call("hindsight_retain", {"content": "chosen fact"}))
    assert result["result"] == "Memory stored successfully."
    recall.assert_called_once_with("q")
    reflect.assert_called_once_with("q")
    retain.assert_called_once()


def _bare_agent(manager):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "turn-origin-test"
    agent.platform = "desktop"
    agent.model = "test-model"
    agent._session_db = None
    agent._persist_disabled = True
    agent._parent_session_id = None
    agent._memory_manager = manager
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: agent.session_id
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    return agent


@pytest.mark.parametrize("cli_staging", [False, True])
def test_public_turn_entry_binds_origin_for_real_manager_and_provider(provider, monkeypatch, cli_staging):
    manager = MemoryManager()
    manager.add_provider(provider)
    agent = _bare_agent(manager)
    agent.platform = "cli" if cli_staging else "desktop"

    def loop(_agent, message, *_args, **_kwargs):
        recalled = manager.prefetch_all(message, session_id=agent.session_id)
        manager.sync_all(message, "answer", session_id=agent.session_id)
        manager.queue_prefetch_all(message, session_id=agent.session_id)
        assert manager.flush_pending(timeout=3)
        return {"final_response": recalled, "messages": [], "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        assert "relevant memory" in agent.run_conversation("human question")["final_response"]
        if cli_staging:
            agent._pending_cli_user_message = {"role": "user", "content": "notification", "display_kind": "process_complete"}
            kwargs = {}
        else:
            kwargs = {"persist_user_display_kind": "process_complete"}
        assert agent.run_conversation("notification", **kwargs)["final_response"] == ""
        # A stale staged row must not poison the next real turn on the reused agent.
        assert "relevant memory" in agent.run_conversation("next human question")["final_response"]
        assert provider._do_recall.call_count == 2
        assert provider._enqueue_retain.call_count == 2
        assert is_user_input_turn()
    finally:
        manager.shutdown_all()


def test_other_memory_provider_keeps_its_existing_non_human_policy():
    from agent.memory_provider import MemoryProvider

    other = Mock(spec=MemoryProvider)
    other.name = "other-provider"
    other.get_tool_schemas.return_value = []
    other.prefetch.return_value = "other provider memory"
    manager = MemoryManager()
    manager.add_provider(other)
    try:
        with turn_input_scope(display_kind="internal_notification"):
            assert "other provider memory" in manager.prefetch_all("notification")
            manager.sync_all("notification", "response")
            manager.queue_prefetch_all("notification")
            assert manager.flush_pending(timeout=3)
        other.prefetch.assert_called_once()
        other.sync_turn.assert_called_once()
        other.queue_prefetch.assert_called_once()
    finally:
        manager.shutdown_all()
