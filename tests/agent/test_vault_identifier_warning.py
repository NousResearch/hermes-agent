"""identifier_warning: a password typed into a login's username field is flagged before it becomes
agent-visible metadata (the identifier is returned by browser_vault_list; only the password is secret)."""
from __future__ import annotations

import pytest

from agent.vault_store import IDENTIFIER_WARNING, identifier_warning


@pytest.mark.parametrize("value", ["jane@example.com", "first.last+tag@sub.example.co", "janedoe", "jane.doe_42", "", "   "])
def test_quiet_for_real_usernames_and_emails(value):
    assert identifier_warning(value) is None


@pytest.mark.parametrize("value", [
    "Xk9#mQ2!vLp7Rt",                                   # password-shaped
    "c0rrectHorse8atteryStaple",                       # long mixed-case + digits
    "use the research skill and add this to the list",  # pasted chat text
    "x" * 70,                                          # far too long for a username
])
def test_fires_for_password_shapes_and_pasted_text(value):
    assert identifier_warning(value) == IDENTIFIER_WARNING


def test_warning_text_never_echoes_the_value():
    value = "Xk9#mQ2!vLp7Rt"
    assert value not in (identifier_warning(value) or "")
