"""Per-home kanban dispatch claim allowlist.

Regression tests for #110995: on a shared kanban board (one kanban.db mounted
across several Hermes homes) every home's ``profile_exists("default")`` is
unconditionally True, so any home's dispatcher could claim cards assigned to
``default``. ``kanban.dispatch_profiles`` (or the
``HERMES_KANBAN_DISPATCH_PROFILES`` env bridge) declares which assignees this
home may claim; anything else lands in ``skipped_nonspawnable``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def every_home_has_default(monkeypatch):
    """Reproduce the incident premise: ``profile_exists("default")`` is True."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


# Env bridge for the claim allowlist (mirrors
# kanban_db_dispatch.KANBAN_DISPATCH_PROFILES_ENV; spelled out so the tests
# stay red-on-base against code that lacks the constant).
_DISPATCH_PROFILES_ENV = "HERMES_KANBAN_DISPATCH_PROFILES"


@pytest.fixture
def no_env_bridge(monkeypatch):
    monkeypatch.delenv(_DISPATCH_PROFILES_ENV, raising=False)


def test_allowlist_env_restricts_default_claim(monkeypatch, every_home_has_default):
    monkeypatch.setenv(_DISPATCH_PROFILES_ENV, "sage,researcher")
    claim = kbd._profile_exists_fn()
    assert claim is not None
    assert claim("sage") is True
    assert claim("researcher") is True
    # The incident: this home must NOT claim another home's "default".
    assert claim("default") is False


def test_allowlist_env_none_claims_nothing(monkeypatch, every_home_has_default):
    monkeypatch.setenv(_DISPATCH_PROFILES_ENV, "none")
    claim = kbd._profile_exists_fn()
    assert claim is not None
    assert claim("sage") is False
    assert claim("default") is False


def test_allowlist_config_key_end_to_end(every_home_has_default, no_env_bridge):
    """Real config.yaml -> real load_config() -> gated predicate."""
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("kanban:\n  dispatch_profiles:\n    - sage\n")
    claim = kbd._profile_exists_fn()
    assert claim is not None
    assert claim("sage") is True
    assert claim("default") is False


def test_unset_allowlist_preserves_upstream(every_home_has_default, no_env_bridge):
    claim = kbd._profile_exists_fn()
    assert claim is not None
    assert claim("default") is True
