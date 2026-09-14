"""V4A edit cards resolve every referenced path in the session workspace."""
import pytest

from acp_adapter.edit_approval import build_edit_proposal
from gateway.session_policy import build_policy, policy_scope


@pytest.mark.parametrize('multiple', [False, True])
def test_v4a_permission_paths_use_execution_workspace(tmp_path, multiple):
    (tmp_path / 'one.txt').write_text('old one')
    (tmp_path / 'two.txt').write_text('old two')
    patch = '*** Begin Patch\n*** Update File: one.txt\n@@\n-old one\n+new one\n'
    if multiple:
        patch += '*** Update File: two.txt\n@@\n-old two\n+new two\n'
    patch += '*** End Patch'
    policy = build_policy({'source': 'acp', 'cwd': str(tmp_path), 'model': 'fixture'}, {})
    with policy_scope(policy):
        proposal = build_edit_proposal('patch', {'mode': 'patch', 'patch': patch})
    expected = [str(tmp_path / 'one.txt')]
    if multiple:
        expected.append(str(tmp_path / 'two.txt'))
    assert proposal.path == ', '.join(expected)
    assert proposal.old_text == (None if multiple else 'old one')
    assert proposal.new_text == patch  # V4A protocol card shows the exact patch.
    assert (tmp_path / 'one.txt').read_text() == 'old one'
