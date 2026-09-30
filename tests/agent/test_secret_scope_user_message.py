"""UnscopedSecretError reaches end users verbatim (compression-aborted replies, cron notices), so
`str(exc)` must be the one actionable sentence; the developer diagnosis stays in `__notes__`/logs.
"""

import pytest

import agent.secret_scope as ss


@pytest.fixture
def multiplex_on():
    ss.set_multiplex_active(True)
    try:
        yield
    finally:
        ss.set_multiplex_active(False)


def test_str_is_a_user_sentence_and_developer_detail_rides_the_notes(multiplex_on):
    with pytest.raises(ss.UnscopedSecretError) as info:
        ss.get_secret("OPENROUTER_API_KEY")
    exc = info.value
    text = str(exc)
    assert "OPENROUTER_API_KEY" in text  # still names what could not be read
    for jargon in ("set_secret_scope", "Workstream", "developer-guide", "os.environ"):
        assert jargon not in text
    assert any("set_secret_scope" in note for note in exc.__notes__)
