"""Inactive / sibling profile hard-deny coverage (#37617).

``is_write_denied`` historically only walked the active HERMES_HOME and the
Hermes root. Credential and session stores under ``<root>/profiles/<other>/``
were left writable. Soft cross-profile guard only covers skills/plugins/cron/
memories, so sibling ``.env`` / ``mcp-tokens/`` / ``pairing/`` / ``state.db``
got neither a hard deny nor a soft warning.

Scope follows the 2026-07-13 maintainer review on #37624 / #37625: extend
hard deny to paths that are *still* hard-denied on active+root — do NOT
restore the relaxed control-file policy from ``81e42335``
(``auth.json`` / ``config.yaml`` / ``webhook_subscriptions.json`` stay writable).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def dual_profiles(tmp_path, monkeypatch):
    """Active=alice, sibling=bob under a shared Hermes root."""
    root = tmp_path / ".hermes"
    alice = root / "profiles" / "alice"
    bob = root / "profiles" / "bob"
    alice.mkdir(parents=True)
    bob.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(alice))
    return root, alice, bob


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("x", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        ".anthropic_oauth.json",
        "mcp-tokens/tok.json",
        "pairing/telegram-approved.json",
        "state.db",
        "sessions/session_abc.json",
    ],
)
def test_sibling_still_hard_denied_paths_are_write_denied(dual_profiles, relative):
    from agent.file_safety import is_write_denied

    _root, alice, bob = dual_profiles
    target = _touch(bob / relative)
    assert is_write_denied(str(target)) is True
    # Active profile path stays denied (regression).
    assert is_write_denied(str(_touch(alice / relative))) is True


@pytest.mark.parametrize(
    "name",
    ["auth.json", "config.yaml", "webhook_subscriptions.json"],
)
def test_sibling_control_files_remain_writable(dual_profiles, name):
    """81e42335 contract: control files are writable on active AND sibling."""
    from agent.file_safety import is_write_denied

    _root, alice, bob = dual_profiles
    assert is_write_denied(str(_touch(bob / name))) is False
    assert is_write_denied(str(_touch(alice / name))) is False


def test_sibling_skills_not_hard_denied(dual_profiles):
    """skills/ stays a soft-guard concern, not a hard write deny."""
    from agent.file_safety import classify_cross_profile_target, is_write_denied

    _root, _alice, bob = dual_profiles
    target = _touch(bob / "skills" / "x.md")
    assert is_write_denied(str(target)) is False
    info = classify_cross_profile_target(str(target))
    assert info is not None
    assert info["target_profile"] == "bob"
    assert info["area"] == "skills"


def test_root_env_still_denied_under_named_profile(dual_profiles):
    from agent.file_safety import is_write_denied

    root, _alice, _bob = dual_profiles
    assert is_write_denied(str(_touch(root / ".env"))) is True
