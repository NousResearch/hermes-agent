"""Managed claim policy must be validated and applied from one snapshot."""
from unittest.mock import Mock

import pytest

pytest_plugins = ["tests.hermes_cli.test_config_effective_policy"]


@pytest.mark.parametrize("warm", [False, True])
def test_policy_replacement_does_not_widen_claims(policy_homes, monkeypatch, warm):
    from hermes_cli import config_effective, kanban_db_dispatch, managed_scope
    from hermes_cli.profiles import normalize_profile_name

    home, managed = policy_homes
    (home / "config.yaml").write_text("kanban: {dispatch_profiles: [default]}\n")
    (managed / "config.yaml").write_text("kanban: {dispatch_profiles: [researcher]}\n")
    if warm:
        config_effective.load_user_config_effective(
            fail_closed=True, strict_policy=True, side_effect_free=True
        )
    reads = 0

    def replacing_policy(**kwargs):
        nonlocal reads
        reads += 1
        return {"kanban": {"dispatch_profiles": ["sage"]}} if reads == 1 else {"kanban": None}

    monkeypatch.setattr(managed_scope, "load_managed_config", replacing_policy)
    assert kanban_db_dispatch._dispatch_profile_allowlist(normalize_profile_name) == frozenset({"sage"})
    assert reads == 1


def test_diagnostics_keep_original_denial_reason(policy_homes, monkeypatch):
    from hermes_cli import kanban_db_dispatch, managed_scope

    read = Mock(side_effect=[{"kanban": None}, {}])
    monkeypatch.setattr(managed_scope, "load_managed_config", read)
    assert kanban_db_dispatch.dispatch_profile_allowlist_summary() == (
        "none (fail-closed: managed kanban section is null or not a mapping)"
    )
    assert read.call_count == 1
