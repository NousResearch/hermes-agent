import pytest
from hermes_cli import session_transcript


@pytest.mark.parametrize('bad', [None, {}, {'include_tools': True}, {'include_tools': 1}, 'yes'])
def test_options_require_exact_boolean_fields(bad):
    with pytest.raises(ValueError):
        session_transcript.validate_options(bad)
