"""Approved lane contract: conversational drafts are not mini essays."""
from types import SimpleNamespace
import pytest
from x_argument_policy import requires_argument_pack
from x_voice_gate import voice_gate_issues

@pytest.mark.parametrize('lane', ['reply_draft', 'quote_tweet_scan'])
def test_conversation_does_not_require_argument_pack(lane):
    artifact = SimpleNamespace(lane=lane, body='What happens when the connection drops?',
                               pack=SimpleNamespace(context={}))
    assert requires_argument_pack(artifact) is False

@pytest.mark.parametrize('lane', ['original_thesis', 'morning_article', 'transform'])
def test_independent_opinions_still_require_pack(lane):
    assert requires_argument_pack(SimpleNamespace(lane=lane)) is True

@pytest.mark.parametrize('draft', [
    'I would test what happens when the connection drops.',
    'Can I run this without an account?',
    'I think the setup needs fewer steps.',
])
def test_questions_and_proposals_are_not_autobiographical_claims(draft):
    assert voice_gate_issues(draft) == []

@pytest.mark.parametrize('draft', [
    'I shipped this last week.',
    'I think I shipped this last week.',
    'We saved ten hours with this.',
    'Built this over the weekend.',
])
def test_unsubstantiated_experience_still_rejected(draft):
    assert any('first-hand' in issue for issue in voice_gate_issues(draft))
