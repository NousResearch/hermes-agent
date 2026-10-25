"""Mutation identities and boolean intent must survive HTTP parsing unchanged."""
import pytest
from pydantic import ValidationError

from hermes_cli.web_models import SessionImport, SessionRename


def test_mutation_counters_reject_coercion():
    for model, extra in ((SessionRename, {}), (SessionImport, {'sessions': []})):
        for value in (True, False, '0', 1.0, -1):
            with pytest.raises(ValidationError):
                model(request_id='edit', expected_revision=value, **extra)
    for value in (True, '1', 1.0, -1):
        with pytest.raises(ValidationError):
            SessionRename(request_id='edit', expected_revision=0, expected_generation=value)
    assert SessionRename(expected_revision=3, expected_generation=2).expected_revision == 3


def test_sidebar_flags_reject_coercion():
    for field in ('archived', 'hidden', 'pinned', 'unread'):
        for value in (0, 1, 'false', 'true'):
            with pytest.raises(ValidationError):
                SessionRename(**{field: value})
        assert getattr(SessionRename(**{field: False}), field) is False
