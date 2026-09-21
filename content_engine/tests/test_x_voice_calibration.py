"""User-calibrated voice behaviour, not a claim of semantic slop detection."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from x_voice_gate import voice_gate_issues

@pytest.mark.parametrize('text', [
    'Local models deserve MORE support!!',
    'Another subscription?? Fuck that.',
    'Smaller models, local inference, better routing... plenty of ways to do it.',
])
def test_natural_emphasis_and_rough_punctuation_are_allowed(text):
    assert voice_gate_issues(text) == []

@pytest.mark.parametrize('text', [
    'VERY interested in how they handle that part.',
    'but this setup just saved the wrong lesson and confidently repeated it.',
])
def test_user_rejected_structural_phrases_are_rejected(text):
    assert any('structural slop' in issue for issue in voice_gate_issues(text))

@pytest.mark.parametrize('text', ['Great point! #AI', 'I shipped this yesterday.', 'RT if you agree'])
def test_style_permission_does_not_remove_other_checks(text):
    assert voice_gate_issues(text)
