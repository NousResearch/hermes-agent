"""CLI scope/add/remove contracts for shared Anthropic pool."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.agent.anthropic_shared_test_helpers import (
    enable_marker,
    make_row,
    shared_root,
    stage_three,
)


def test_scope_report(shared_root, capsys):
    from hermes_cli.anthropic_shared_auth import auth_scope_command

    auth_scope_command(SimpleNamespace(provider="anthropic", scope_mode=None))
    out = capsys.readouterr().out
    assert "scope: profile" in out


def test_enable_requires_three_and_attest(shared_root):
    from hermes_cli.anthropic_shared_auth import auth_scope_command
    from agent.anthropic_shared_pool import (
        append_row,
        get_shared_mutation_capability,
    )

    cap = get_shared_mutation_capability()
    for i in range(2):
        append_row(make_row(priority=i, refresh=f"fixture-oauth-refresh-cli-{i}"), capability=cap)
    with pytest.raises(SystemExit) as ei:
        auth_scope_command(
            SimpleNamespace(
                provider="anthropic",
                scope_mode="shared",
                attest_distinct_accounts=True,
            )
        )
    assert ei.value.code in (1, 2)


def test_enable_shared_happy_path(shared_root, capsys, monkeypatch):
    from agent.anthropic_shared_pool import (
        append_row,
        get_shared_mutation_capability,
        is_shared_scope_active,
        require_no_live_gateways_for_scope_change,
    )
    from hermes_cli.anthropic_shared_auth import auth_scope_command

    monkeypatch.setattr(
        "agent.anthropic_shared_pool.require_no_live_gateways_for_scope_change",
        lambda: None,
    )
    cap = get_shared_mutation_capability()
    for i in range(3):
        append_row(make_row(priority=i, refresh=f"fixture-oauth-refresh-cli3-{i}"), capability=cap)
    auth_scope_command(
        SimpleNamespace(
            provider="anthropic",
            scope_mode="shared",
            attest_distinct_accounts=True,
        )
    )
    assert is_shared_scope_active()
    out = capsys.readouterr().out
    assert "scope: shared" in out


def test_unscoped_add_while_shared_exits_2(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from hermes_cli.auth_commands import auth_add_command

    with pytest.raises(SystemExit) as ei:
        auth_add_command(
            SimpleNamespace(
                provider="anthropic",
                auth_type="oauth",
                shared=False,
                label=None,
                api_key=None,
            )
        )
    assert ei.value.code == 2


def test_active_remove_rejected(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from hermes_cli.auth_commands import auth_remove_command

    with pytest.raises(SystemExit) as ei:
        auth_remove_command(
            SimpleNamespace(provider="anthropic", target="1", shared=True)
        )
    assert ei.value.code == 2


def test_already_shared_enable_idempotent(shared_root, monkeypatch, capsys):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root, epoch="cccccccc-cccc-cccc-cccc-cccccccccccc")
    monkeypatch.setattr(
        "agent.anthropic_shared_pool.require_no_live_gateways_for_scope_change",
        lambda: None,
    )
    from hermes_cli.anthropic_shared_auth import auth_scope_command
    from agent.anthropic_shared_pool import read_scope_state

    before = read_scope_state().epoch
    auth_scope_command(
        SimpleNamespace(
            provider="anthropic",
            scope_mode="shared",
            attest_distinct_accounts=True,
        )
    )
    after = read_scope_state().epoch
    assert before == after
