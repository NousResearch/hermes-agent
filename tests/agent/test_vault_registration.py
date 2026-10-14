"""Registration recognition must not weaken login or password-change guards."""
from dataclasses import replace

import pytest

from agent.vault_login_classifier import LoginControl, classify_login_control
from agent.vault_registration import registration_pair, confirm_registration


NEW = LoginControl('new-password', 0, 1, 'Password', 'password', 'password')
CONFIRM = LoginControl('', 0, 2, 'Confirm Password', 'confirmPassword', 'password')


def test_recognizes_exact_registration_pair():
    assert registration_pair([NEW, CONFIRM]) == (NEW, CONFIRM)
    assert registration_pair([CONFIRM, NEW]) == (NEW, CONFIRM)


@pytest.mark.parametrize('controls', [
    [], [NEW], [NEW, CONFIRM, replace(NEW, index=3)],
    [NEW, replace(CONFIRM, index=1)],
    [replace(NEW, form_index=None), replace(CONFIRM, form_index=None)],
    [NEW, replace(CONFIRM, form_index=1)],
    [replace(NEW, autocomplete='current-password'), CONFIRM],
    [NEW, replace(CONFIRM, autocomplete='current-password')],
    [NEW, replace(CONFIRM, autocomplete='one-time-code')],
    [replace(NEW, max_length=6), CONFIRM],
    [NEW, replace(CONFIRM, max_length=6)],
    [replace(NEW, autocomplete=''), CONFIRM],
    [NEW, replace(CONFIRM, label='Password', name='passwordTwo')],
    [replace(NEW, label='Confirm Password', name='confirmPassword'), CONFIRM],
    [NEW, replace(CONFIRM, type='text')],
])
def test_refuses_nonregistration_or_ambiguity(controls):
    assert registration_pair(controls) is None


def test_login_classifier_is_unchanged():
    assert classify_login_control(NEW) is None
    assert classify_login_control(CONFIRM) is None
    current = replace(NEW, autocomplete='current-password')
    classified = classify_login_control(current)
    assert classified is not None
    assert classified.token == 'current-password'


@pytest.mark.parametrize('answer, expected', [('accept', True), ('decline', False), ('cancel', False), (None, False)])
def test_registration_requires_human_consent(monkeypatch, answer, expected):
    import tools.approval_prompt as prompt
    calls = []
    def consent(*args, **kwargs):
        calls.append((args, kwargs))
        return answer
    monkeypatch.setattr(prompt, 'request_elicitation_consent', consent)
    assert confirm_registration('synthetic account', 'https://example.test') is expected
    assert len(calls) == 1
    assert calls[0][1]['surface'] == 'vault-registration'
