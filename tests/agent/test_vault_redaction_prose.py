"""Short vault canaries must not fragment assistant prose; raw tool data stays strict."""
from types import SimpleNamespace

import pytest

from agent import redact
from agent.chat_completion_helpers import _assistant_content_for_storage
from agent.history_commentary import visible_commentary
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize('value,prose', [
    ('ا', 'سيدي جابر، اطلعت على ملف التسليم وسجل القبول.'),
    ('a', 'A harmless ordinary paragraph remains readable.'),
    ('ab', 'The alphabet table remains readable.'),
    ('123', 'The identifier 912345 stays intact.'),
    ('ا', 'هَذا نص عربي مَكتوب.'),
])
def test_assistant_prose_preserves_words_but_masks_standalone_canaries(tmp_path, value, prose):
    """Exercise the actual storage and history projections, not a renderer workaround."""
    home = tmp_path / 'profile-a'
    home.mkdir()
    token = set_hermes_home_override(str(home))
    agent = SimpleNamespace(_strip_think_blocks=lambda text: text)
    try:
        redact.register_vault_redaction_value(value)
        redact.register_vault_redaction_value('fixture-long-opaque-98')
        marker = '«redacted-vault-secret»'
        raw = prose + '\n' + value + '\nfixture-long-opaque-98'
        expected = prose + '\n' + marker + '\n' + marker
        stored = _assistant_content_for_storage(agent, SimpleNamespace(content=raw))
        assert stored == expected
        assert visible_commentary(raw, strip_thinking=lambda text: text) == expected
        # Tool/browser ingress still hides embedded values; prose mode must not reach it.
        assert redact.redact_sensitive_text(prose, force=True) != prose
        assert redact.redact_sensitive_text(value, force=True) == marker
    finally:
        redact.clear_vault_redaction_values()
        reset_hermes_home_override(token)


@pytest.fixture
def isolated_registry(tmp_path):
    token = set_hermes_home_override(str(tmp_path / 'isolated-vault-profile'))
    try:
        yield
    finally:
        redact.clear_vault_redaction_values()
        reset_hermes_home_override(token)


@pytest.mark.parametrize('prose', [False, True])
def test_markers_are_idempotent_and_matching_is_literal(isolated_registry, prose):
    marker = '«redacted-vault-secret»'
    for value in ('a', 'vault', 'secret', 'fixture.*[98]', marker + '-fixture-suffix'):
        redact.register_vault_redaction_value(value)
    raw = 'a vault secret fixture.*[98] ' + marker + '-fixture-suffix'
    expected = ' '.join([marker] * 5)
    result = redact.redact_registered_vault_values(raw, prose=prose)
    assert result == expected
    assert redact.redact_registered_vault_values(result, prose=prose) == expected


@pytest.mark.parametrize('flag', ['file_read', 'code_file', 'secret_file'])
def test_prose_mode_cannot_relax_file_redaction(isolated_registry, flag):
    redact.register_vault_redaction_value('xyz')
    assert redact.redact_sensitive_text('prefixxyzsuffix', vault_prose=True, **{flag: True}) == (
        'prefix«redacted-vault-secret»suffix')


def test_disabled_preference_does_not_expose_registered_values(isolated_registry, monkeypatch):
    monkeypatch.setattr(redact, '_redact_enabled', lambda: False)
    redact.register_vault_redaction_value('xyz')
    assert redact.redact_sensitive_text('xyz', vault_prose=True) == '«redacted-vault-secret»'
    assert redact.redact_sensitive_text('prefixxyzsuffix') == 'prefix«redacted-vault-secret»suffix'


def test_combining_marks_stay_attached_to_words(isolated_registry):
    redact.register_vault_redaction_value('a')
    for text in ('a\u0301', 'e\u0301a', '_a', 'a_'):
        assert redact.redact_sensitive_text(text, vault_prose=True) == text
    assert redact.redact_sensitive_text('(a)', vault_prose=True) == '(«redacted-vault-secret»)'


def test_prose_does_not_reveal_other_profile_values(isolated_registry, tmp_path):
    redact.register_vault_redaction_value('fixture-own-profile-opaque')
    token = set_hermes_home_override(str(tmp_path / 'unrelated-profile'))
    try:
        text = 'fixture-own-profile-opaque'
        assert redact.redact_sensitive_text(text, vault_prose=True) == text
    finally:
        reset_hermes_home_override(token)
