"""Edit approval reads the file through the terminal backend: an ssh path is not on this host."""
from types import SimpleNamespace

import pytest

from acp_adapter.edit_approval import (
    maybe_require_edit_approval, reset_edit_approval_requester, set_edit_approval_requester,
)

REMOTE_FILE = '/srv/remote/project/app.py'  # exists only on the backend


class _RemoteOps:
    def __init__(self, files):
        self.files = files

    def read_file_raw(self, path):
        if path not in self.files:
            return SimpleNamespace(not_found=True, error='File not found', content='')
        return SimpleNamespace(not_found=False, error=None, content=self.files[path])


def _propose(monkeypatch, backend, files, tool, args):
    from tools import file_tools
    from tools import file_tools_paths as paths
    monkeypatch.setattr(paths, '_terminal_env_type_for_task', lambda task_id='default': backend)
    monkeypatch.setattr(paths, '_resolve_path_for_task', lambda path, task_id='default': path)
    monkeypatch.setattr(file_tools, '_get_file_ops', lambda task_id='default': _RemoteOps(files))
    proposals = []
    token = set_edit_approval_requester(lambda p: proposals.append(p) or False)
    try:
        denied = maybe_require_edit_approval(tool, {'path': REMOTE_FILE, **args})
    finally:
        reset_edit_approval_requester(token)
    return proposals, denied


def test_patch_diff_is_built_from_the_remote_file(monkeypatch):
    proposals, denied = _propose(monkeypatch, 'ssh', {REMOTE_FILE: 'x = 1\n'}, 'patch',
                                 {'old_string': 'x = 1', 'new_string': 'x = 2'})
    assert proposals and proposals[0].old_text == 'x = 1\n' and proposals[0].new_text == 'x = 2\n'
    assert 'could not prepare diff' not in denied  # denied by the editor, not by a failed read


def test_write_file_over_an_existing_remote_file_is_an_edit_not_a_create(monkeypatch):
    proposals, _ = _propose(monkeypatch, 'ssh', {REMOTE_FILE: 'old\n'}, 'write_file', {'content': 'new\n'})
    assert proposals[0].old_text == 'old\n'


def test_write_file_to_a_missing_remote_path_is_a_create(monkeypatch):
    proposals, _ = _propose(monkeypatch, 'ssh', {}, 'write_file', {'content': 'new\n'})
    assert proposals[0].old_text is None


def test_patch_of_a_missing_remote_file_still_refuses(monkeypatch):
    proposals, denied = _propose(monkeypatch, 'ssh', {}, 'patch', {'old_string': 'a', 'new_string': 'b'})
    assert not proposals and 'could not prepare diff' in denied


def test_a_failed_remote_read_is_not_treated_as_a_missing_file(monkeypatch):
    from tools import file_tools
    from tools import file_tools_paths as paths
    monkeypatch.setattr(paths, '_terminal_env_type_for_task', lambda task_id='default': 'ssh')
    monkeypatch.setattr(paths, '_resolve_path_for_task', lambda path, task_id='default': path)
    broken = SimpleNamespace(read_file_raw=lambda path: SimpleNamespace(
        not_found=False, error='ssh transport down', content=''))
    monkeypatch.setattr(file_tools, '_get_file_ops', lambda task_id='default': broken)
    proposals = []
    token = set_edit_approval_requester(lambda p: proposals.append(p) or False)
    try:
        denied = maybe_require_edit_approval('write_file', {'path': REMOTE_FILE, 'content': 'x'})
    finally:
        reset_edit_approval_requester(token)
    assert not proposals and 'ssh transport down' in denied
