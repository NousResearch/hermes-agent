"""Tests for hermes_cli/memory_oauth.py — _resolve_flow provider validation."""

import pytest
from fastapi import HTTPException


def test_resolve_flow_valid_provider():
    from hermes_cli.memory_oauth import _resolve_flow
    # honcho is a known memory provider
    flow = _resolve_flow("honcho")
    assert flow is not None


def test_resolve_flow_invalid_provider_name():
    from hermes_cli.memory_oauth import _resolve_flow
    with pytest.raises(HTTPException) as exc:
        _resolve_flow("not-a-valid!provider")
    assert exc.value.status_code == 404


def test_resolve_flow_nonexistent_provider():
    from hermes_cli.memory_oauth import _resolve_flow
    with pytest.raises(HTTPException) as exc:
        _resolve_flow("nonexistentproviderxyz")
    assert exc.value.status_code == 404
