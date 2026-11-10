"""The conversation backend raises the contract's authority refusal itself: the store functions raise the contract's
``AuthorityRefusal``, and the verb boundary passes it on unchanged."""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, RetiredSessionError, UnknownExecution
from persistence.sqlite.conversation._errors import refusals
from persistence.sqlite.conversation._store import retirement


def test_the_store_refusal_is_the_contract_class():
    assert retirement.AuthorityRefusal is AuthorityRefusal


def test_a_store_refusal_reaches_the_caller_unchanged():
    raised = retirement.AuthorityRefusal("session_busy", field="session", detail="a turn is running")
    with pytest.raises(AuthorityRefusal) as caught:
        with refusals(session_id="s1"):
            raise raised
    assert caught.value is raised and caught.value.__cause__ is None


def test_a_contract_subclass_is_not_rewrapped():
    raised = RetiredSessionError("s1", "not_found")
    with pytest.raises(RetiredSessionError) as caught:
        with refusals(session_id="s1"):
            raise raised
    assert caught.value is raised


def test_an_unknown_execution_refusal_names_the_session():
    with pytest.raises(UnknownExecution) as caught:
        with refusals(session_id="s1"):
            raise retirement.AuthorityRefusal("unknown_execution")
    assert caught.value.reason == "unknown_execution" and type(caught.value.__cause__) is AuthorityRefusal
