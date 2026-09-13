"""The principal grant must not be self-modifying.

``principal`` lifts the ``config.yaml`` hard-deny so the operator's own config can be edited
without a per-turn round trip. These tests pin the boundary around that grant: editing ordinary
keys stays allowed (and audited), while rewriting ``authority_profile`` or ``approvals`` — the keys
that *define* the grant and protect the exec-approval gate — is refused under every profile.

Without this boundary principal is self-granting: a prompt-injected agent could flip
``production`` -> ``principal`` and then disable ``approvals``, which is precisely the attack the
hard-deny it replaces exists to stop.
"""

from __future__ import annotations

import pytest

from hermes_cli import authority
from tools.file_tools_write_guards import (
    _authority_permits_config_edit,
    _check_authority_policy_patch,
    _check_authority_policy_write,
    _config_policy_delta,
)

_TEMPLATE = """authority_profile: {profile}
approvals:
  mode: {mode}
model: MiniMax-M3
"""


def _seed(tmp_path, *, profile: str = "principal", mode: str = "ask") -> str:
    (tmp_path / "config.yaml").write_text(
        _TEMPLATE.format(profile=profile, mode=mode), encoding="utf-8"
    )
    return str(tmp_path / "config.yaml")


@pytest.fixture()
def principal(monkeypatch, tmp_path):
    """Point the guard at a temp config and put the process in principal mode."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(
        authority, "allows", lambda action: action == "edit_config_yaml"
    )
    authority.reset_cache()
    yield tmp_path
    authority.reset_cache()


# ─── the grant itself is intact ───────────────────────────────────────────────

def test_principal_may_edit_unrelated_config_keys(principal):
    path = _seed(principal)
    proposed = _TEMPLATE.format(profile="principal", mode="ask").replace(
        "MiniMax-M3", "MiniMax-M4"
    )
    assert _check_authority_policy_write(path, proposed) is None


def test_principal_may_rewrite_the_whole_file_including_unchanged_policy_keys(principal):
    """A whole-file rewrite that leaves the policy keys alone must still be allowed.

    This is the common case (the agent rewrites config.yaml with the same profile/approvals), and
    refusing it would make principal useless.
    """
    path = _seed(principal)
    assert _check_authority_policy_write(
        path, _TEMPLATE.format(profile="principal", mode="ask")
    ) is None


# ─── …but not the keys that define it ─────────────────────────────────────────

def test_principal_cannot_escalate_to_itself(principal):
    """production -> principal by rewriting the file is exactly the self-grant to block."""
    path = _seed(principal, profile="production")
    err = _check_authority_policy_write(
        path, _TEMPLATE.format(profile="principal", mode="ask")
    )
    assert err is not None
    assert "authority_profile" in err
    assert "cannot rewrite" in err


def test_principal_cannot_downgrade_the_profile_either(principal):
    """Symmetric: hiding from the operator is as much a policy rewrite as escalating."""
    path = _seed(principal, profile="principal")
    err = _check_authority_policy_write(
        path, _TEMPLATE.format(profile="production", mode="ask")
    )
    assert err is not None and "authority_profile" in err


def test_principal_cannot_disable_approvals(principal):
    path = _seed(principal, mode="ask")
    err = _check_authority_policy_write(path, _TEMPLATE.format(profile="principal", mode="off"))
    assert err is not None
    assert "approvals" in err


def test_patch_to_config_is_refused_under_principal(principal):
    """Fragments cannot be inspected, so the fail-closed answer is refusal."""
    path = _seed(principal)
    err = _check_authority_policy_patch([path])
    assert err is not None
    assert "Refusing to patch the Hermes config" in err


# ─── production is untouched, and unknown states deny ─────────────────────────

def test_production_leaves_the_whole_file_to_the_existing_hard_deny(monkeypatch, tmp_path):
    """Under production this guard must be inert — _check_sensitive_path already denies."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(authority, "allows", lambda action: False)
    path = _seed(tmp_path, profile="production")
    assert _check_authority_policy_write(
        path, _TEMPLATE.format(profile="principal", mode="ask")
    ) is None
    assert _check_authority_policy_patch([path]) is None


def test_unknown_authority_state_denies_rather_than_crashes(monkeypatch):
    def _boom(action):
        raise RuntimeError("authority backend unavailable")

    monkeypatch.setattr(authority, "allows", _boom)
    assert _authority_permits_config_edit() is False


# ─── fail-closed parsing ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "proposed",
    ["authority_profile: principal\n  bad: indent\n", ":\n:", "\t- not a mapping"],
)
def test_unparseable_proposals_are_treated_as_a_change(proposed):
    assert _config_policy_delta("authority_profile: production\n", proposed) is not None


def test_missing_existing_config_counts_as_a_change():
    """No readable config means the keys cannot be shown to be preserved."""
    assert _config_policy_delta(None, "authority_profile: principal\n") == "authority_profile"


def test_identical_policy_keys_report_no_change():
    same = _TEMPLATE.format(profile="principal", mode="ask")
    assert _config_policy_delta(same, same) is None
