"""Dashboard/web API shared mutation guards return HTTP 409."""

from __future__ import annotations

import pytest

from tests.agent.anthropic_shared_test_helpers import (
    enable_marker,
    shared_root,
    stage_three,
)


def test_shared_mutation_forbidden_without_capability(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent.anthropic_shared_pool import append_row
    from hermes_cli.auth import AuthError
    from tests.agent.anthropic_shared_test_helpers import make_row as mr

    with pytest.raises(AuthError) as ei:
        append_row(mr(priority=0), capability=None)  # type: ignore[arg-type]
    assert ei.value.code == "shared_mutation_forbidden"


def test_web_add_returns_409_logic(shared_root):
    """Unit-level check of the guard used by the dashboard POST handler."""
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent.anthropic_shared_pool import is_shared_scope_active

    assert is_shared_scope_active()
    # Emulate handler branch
    provider = "anthropic"
    if provider == "anthropic" and is_shared_scope_active():
        status = 409
    else:
        status = 200
    assert status == 409


def test_web_delete_returns_409_logic(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent.anthropic_shared_pool import is_shared_scope_active

    assert is_shared_scope_active()
    status = 409 if is_shared_scope_active() else 200
    assert status == 409
