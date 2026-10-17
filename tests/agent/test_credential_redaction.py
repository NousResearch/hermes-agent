"""Registered credential values are scrubbed on every redaction path."""

import pytest

from agent.redact import (
    REGISTERED_SECRET_MASK,
    clear_registered_secrets,
    redact_registered_secrets,
    redact_sensitive_text,
    register_secret,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registered_secrets()
    yield
    clear_registered_secrets()


SECRET = "Zz9Qm4Tt7pWq42xy"  # >= the min registration length, no known key prefix
# (a value like "sk-live-..." would be masked by the prefix patterns anyway, which
# would hide whether the exact-value registry itself is doing the scrubbing.)


def test_registered_value_is_scrubbed_from_any_text():
    register_secret(SECRET)
    for force in (False, True):
        out = redact_sensitive_text(f"connecting with {SECRET} now", force=force)
        assert SECRET not in out, out
        assert REGISTERED_SECRET_MASK in out


def test_scrub_survives_shapeless_context():
    """No prefix/assignment/URL shape to catch it — only the exact-value registry does."""
    from agent.redact import redact_terminal_output

    register_secret(SECRET)
    assert redact_sensitive_text(f"hello {SECRET}", force=False).count(REGISTERED_SECRET_MASK) == 1
    assert SECRET not in redact_terminal_output(f"echo {SECRET}", "echo x")


def test_short_values_are_not_registered():
    """A 4-char value would mask innocent prose wherever it appears."""
    register_secret("short12")  # below the 8-char floor
    assert redact_sensitive_text("short12 is fine", force=True) == "short12 is fine"
    assert redact_registered_secrets("short12 is fine") == "short12 is fine"


def test_clear_removes_registered_values():
    register_secret(SECRET)
    clear_registered_secrets()
    assert redact_sensitive_text(SECRET, force=True) == SECRET


def test_longest_match_wins_so_substrings_do_not_shadow():
    register_secret("abcdefgh")
    register_secret("abcdefghij")  # superstring registered second
    out = redact_registered_secrets("prefix abcdefghij suffix")
    assert "abcdefghij" not in out
    assert out == f"prefix {REGISTERED_SECRET_MASK} suffix"


def test_non_string_and_empty_are_ignored():
    register_secret(None)
    register_secret("")
    assert redact_registered_secrets("nothing to do") == "nothing to do"