"""Behavior tests for the built-in memory → external provider bridge.

The bridge lives behind the MemoryManager interface
(``MemoryManager.notify_memory_tool_write``): the agent loop hands over the raw
built-in memory tool result + args, and the manager decides whether/what to
mirror to external providers. These tests drive that method with a fake
external provider and assert which ``on_memory_write`` calls land.
"""

import json

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Minimal external provider that records on_memory_write calls."""

    def __init__(self) -> None:
        self.calls = []

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_memory_write(self, action, target, content, metadata=None):
        self.calls.append({
            "action": action,
            "target": target,
            "content": content,
            "metadata": dict(metadata or {}),
        })


def _manager_with_provider():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


def test_notifies_remove_with_old_text_after_success():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "remove", "target": "memory", "old_text": "stale preference entry"},
    )
    assert provider.calls == [
        {
            "action": "remove",
            "target": "memory",
            "content": "",
            "metadata": {"old_text": "stale preference entry"},
        }
    ]






@pytest.mark.parametrize("tool_result", [None, [], object(), "not-json"])
def test_skips_unrecognized_tool_result_shape(tool_result):
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        tool_result,
        {"action": "add", "target": "memory", "content": "new fact"},
    )
    assert provider.calls == []






def test_build_metadata_callback_is_merged_per_op():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "memory", "content": "fact"},
        build_metadata=lambda: {"session_id": "s1", "tool_name": "memory"},
    )
    assert provider.calls == [
        {
            "action": "add",
            "target": "memory",
            "content": "fact",
            "metadata": {"session_id": "s1", "tool_name": "memory"},
        }
    ]


@pytest.mark.parametrize('target', ['memory', 'user'])
@pytest.mark.parametrize('batch,operations,previous', [
    (False, [{'action': 'remove', 'old_text': 'Prefers tea'}], ['Prefers tea']),
    (False, [{'action': 'replace', 'old_text': 'Prefers tea', 'content': 'Prefers coffee'}], ['Prefers tea']),
    (True, [{'action': 'remove', 'old_text': 'Prefers tea'}], ['Prefers tea']),
    (True, [{'action': 'replace', 'old_text': 'Prefers tea', 'new_text': 'Prefers coffee'}], ['Prefers tea']),
    (True, [
        {'action': 'add', 'content': 'Uses the blue notebook'},
        {'action': 'replace', 'old_text': 'blue notebook', 'new_text': 'Uses the green notebook'},
        {'action': 'remove', 'old_text': 'green notebook'},
    ], [None, 'Uses the blue notebook', 'Uses the green notebook']),
])
def test_committed_entry_identity_comes_from_locked_store(
    tmp_path, monkeypatch, target, batch, operations, previous
):
    from contextlib import contextmanager
    from tools import memory_tool_store
    from tools.memory_tool import MemoryStore, memory_tool

    monkeypatch.setattr('tools.memory_tool.get_memory_dir', lambda: tmp_path)
    store = MemoryStore()
    store.load_from_disk()
    store.add(target, 'Prefers tea')
    store.add(target, 'Prefers tea with milk')
    manager, provider = _manager_with_provider()
    locked = False
    lock = store._file_lock
    match = memory_tool_store._find_unique_match

    @contextmanager
    def checked_lock(path):
        nonlocal locked
        with lock(path):
            locked = True
            try:
                yield
            finally:
                locked = False

    def checked_match(entries, old_text):
        assert locked
        assert provider.calls == []
        return match(entries, old_text)

    monkeypatch.setattr(store, '_file_lock', checked_lock)
    monkeypatch.setattr(memory_tool_store, '_find_unique_match', checked_match)
    args = {'target': target, **({'operations': operations} if batch else operations[0])}
    result = memory_tool(store=store, **args)
    assert json.loads(result)['success'] is True
    assert not locked
    assert provider.calls == []
    reloaded = MemoryStore()
    reloaded.load_from_disk()
    assert reloaded._entries_for(target) == store._entries_for(target)

    manager.notify_memory_tool_write(result, args, build_metadata=lambda: {'session_id': 'test'})
    assert [call['action'] for call in provider.calls] == [op['action'] for op in operations]
    assert [call['metadata'].get('previous_content') for call in provider.calls] == previous
    assert all(call['metadata']['session_id'] == 'test' for call in provider.calls)
    assert 'Prefers tea with milk' in store._entries_for(target)


@pytest.mark.parametrize('failure', ['invalid_operation', 'over_budget', 'empty_store', 'disk_write'])
def test_uncommitted_batch_emits_no_entry_metadata_or_notifications(tmp_path, monkeypatch, failure):
    from tools.memory_tool import MemoryStore, memory_tool

    monkeypatch.setattr('tools.memory_tool.get_memory_dir', lambda: tmp_path)
    store = MemoryStore(memory_char_limit=100)
    store.load_from_disk()
    store.add('memory', 'Keep this entry')
    path = store._path_for('memory')
    before = path.read_bytes()
    manager, provider = _manager_with_provider()
    operations = [{'action': 'replace', 'old_text': 'Keep this', 'content': 'Changed entry'}]
    if failure == 'invalid_operation':
        operations.append({'action': 'remove', 'old_text': 'Absent entry'})
    elif failure == 'over_budget':
        operations.append({'action': 'add', 'content': 'x' * 101})
    elif failure == 'empty_store':
        operations.append({'action': 'remove', 'old_text': 'Changed entry'})
    else:
        def fail_write(*_args):
            raise OSError('disk unavailable')
        monkeypatch.setattr(store, '_write_file', fail_write)
    args = {'target': 'memory', 'operations': operations}
    if failure == 'disk_write':
        with pytest.raises(OSError, match='disk unavailable'):
            result = memory_tool(store=store, **args)
            manager.notify_memory_tool_write(result, args)
    else:
        result = memory_tool(store=store, **args)
        payload = json.loads(result)
        assert payload['success'] is False
        assert not any(key in payload for key in ('replaced_entries', 'removed_entries'))
        manager.notify_memory_tool_write(result, args)
    assert provider.calls == []
    assert path.read_bytes() == before


def test_previous_content_cannot_come_from_uncommitted_arguments():
    manager, provider = _manager_with_provider()
    args = {'action': 'remove', 'old_text': 'partial', 'previous_content': 'Untrusted argument'}
    manager.notify_memory_tool_write(
        {'success': True}, args, build_metadata=lambda: {'previous_content': 'Uncommitted metadata'}
    )
    assert provider.calls[0]['metadata'] == {'old_text': 'partial'}
