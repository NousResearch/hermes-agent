"""Editor diff preparation must use the same cwd as the actual file tool."""
import pytest

from acp_adapter.edit_approval import (
    maybe_require_edit_approval, set_edit_approval_requester, reset_edit_approval_requester,
)
from gateway.session_policy import build_policy, policy_scope


@pytest.mark.parametrize('tool,args', [
    ('write_file', {'content': 'replacement'}),
    ('patch', {'old_string': 'original', 'new_string': 'replacement'}),
])
def test_relative_editor_diff_reads_the_session_file_not_process_cwd(tmp_path, tool, args):
    path = tmp_path / 'unique-editor-diff.txt'
    path.write_text('original')
    policy = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture'}, {})
    proposals = []
    with policy_scope(policy):
        token = set_edit_approval_requester(lambda p: proposals.append(p) or False)
        try:
            denied = maybe_require_edit_approval(tool, {'path': path.name, **args})
        finally:
            reset_edit_approval_requester(token)
    assert proposals and proposals[0].old_text == 'original'
    assert proposals[0].path == str(path)
    assert proposals[0].new_text == 'replacement'
    assert 'denied' in denied
    assert path.read_text() == 'original'
