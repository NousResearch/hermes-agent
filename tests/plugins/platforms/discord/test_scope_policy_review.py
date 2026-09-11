import pytest
from plugins.platforms.discord.policy import resolve_scope_policy, validate_scope_policies

@pytest.mark.parametrize('version', [True, False, 1.0, '1'])
def test_schema_version_is_integer_not_coerced(version):
    with pytest.raises(ValueError):
        validate_scope_policies({'version': version})

def test_defaults_do_not_assign_trust_or_admission_to_dm():
    policy = resolve_scope_policy({'platform': {'defaults': {'conversation_trust':'full_trusted', 'require_mention':True}}}, None, '123')
    assert policy.conversation_trust is None
    assert policy.require_mention is None
