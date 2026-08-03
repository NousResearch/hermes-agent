"""Tests for the 1Password bootstrap-token reliability patches.

Two behaviours are covered:

1. ``load_hermes_dotenv()`` auto-loads ``~/.hermes/.op.env`` so the
   ``OP_SERVICE_ACCOUNT_TOKEN`` bootstrap token is available to
   ``apply_onepassword_secrets()`` in cron / subprocess / macOS / Docker
   contexts that inherit no shell state (no systemd EnvironmentFile, no
   ``op run``).  ``.op.env`` must never override a token already present
   in the environment (e.g. injected by a systemd ``EnvironmentFile``).

2. ``credential_pool._seed_from_env`` (via the inner
   ``_get_env_prefer_dotenv``) must prefer an already-resolved value from
   ``os.environ`` over a raw ``op://`` reference still sitting in ``.env``,
   while leaving the normal ``.env``-takes-precedence behaviour untouched
   for every non-``op://`` value.

These stay fully hermetic — the real ``op`` binary is never invoked and no
1Password integration is enabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader  # noqa: E402
import agent.credential_pool as credential_pool  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_op_token(monkeypatch):
    """Each test starts with OP_SERVICE_ACCOUNT_TOKEN unset and a clean cache."""
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    env_loader.reset_secret_source_cache()
    yield
    env_loader.reset_secret_source_cache()


# ---------------------------------------------------------------------------
# Patch 1 — .op.env bootstrap-token auto-load
# ---------------------------------------------------------------------------


def test_op_env_autoloads_bootstrap_token_in_cron_context(tmp_path, monkeypatch):
    """A fresh interpreter (no inherited shell state) picks up the token."""
    home = tmp_path / ".hermes"
    home.mkdir()
    # .env carries user secrets / op:// references but NOT the bootstrap token.
    (home / ".env").write_text("FOO=bar\n", encoding="utf-8")
    # The gitignored .op.env holds only the service-account token.
    (home / ".op.env").write_text(
        "OP_SERVICE_ACCOUNT_TOKEN=test-token\n", encoding="utf-8"
    )

    assert os.environ.get("OP_SERVICE_ACCOUNT_TOKEN") is None

    env_loader.load_hermes_dotenv(hermes_home=home)

    assert os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "test-token"


def test_op_env_does_not_override_existing_token(tmp_path, monkeypatch):
    """A token already in the environment (e.g. systemd EnvironmentFile) wins."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("FOO=bar\n", encoding="utf-8")
    (home / ".op.env").write_text(
        "OP_SERVICE_ACCOUNT_TOKEN=test-token\n", encoding="utf-8"
    )

    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "live-token")

    env_loader.load_hermes_dotenv(hermes_home=home)

    # override=False AND the explicit guard both protect the live token.
    assert os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "live-token"


def test_missing_op_env_is_a_noop(tmp_path):
    """No .op.env present must not raise and must not invent a token."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("FOO=bar\n", encoding="utf-8")

    env_loader.load_hermes_dotenv(hermes_home=home)

    assert os.environ.get("OP_SERVICE_ACCOUNT_TOKEN") is None


# ---------------------------------------------------------------------------
# Patch 2 — credential_pool prefers resolved value over raw op:// ref
# ---------------------------------------------------------------------------


def _seed_openrouter_token(monkeypatch, dotenv_value, environ_value):
    """Drive _seed_from_env('openrouter') and return the seeded access_token.

    _get_env_prefer_dotenv is a closure inside _seed_from_env, so we exercise
    it through the openrouter seeding path, which calls
    _get_env_prefer_dotenv('OPENROUTER_API_KEY') and stores the result as the
    pooled credential's access_token.
    """
    monkeypatch.setattr(
        credential_pool,
        "load_env",
        lambda: {"OPENROUTER_API_KEY": dotenv_value},
    )
    if environ_value is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", environ_value)
    # Never treat the synthetic source as suppressed.
    monkeypatch.setattr(
        "hermes_cli.auth.is_source_suppressed", lambda _p, _s: False
    )

    entries: list = []
    changed, sources = credential_pool._seed_from_env("openrouter", entries)
    assert changed and entries, "expected a seeded openrouter credential"
    return entries[0].access_token


def test_credential_pool_prefers_resolved_env_over_raw_op_ref(monkeypatch):
    """A raw op:// reference in .env must lose to the resolved os.environ value."""
    token = _seed_openrouter_token(
        monkeypatch,
        dotenv_value="op://Vault/Item/field",
        environ_value="resolved-value",
    )
    assert token == "resolved-value"


def test_credential_pool_still_prefers_dotenv_for_non_op_values(monkeypatch):
    """Regression guard: .env still beats os.environ for ordinary values."""
    token = _seed_openrouter_token(
        monkeypatch,
        dotenv_value="dotenv-value",
        environ_value="shell-value",
    )
    assert token == "dotenv-value"


def test_credential_pool_falls_back_to_env_when_dotenv_is_only_op_ref(monkeypatch):
    """An unresolved op:// in .env with no resolved env value yields the raw ref.

    This is the pre-resolution / misconfigured edge: there is nothing better
    to return, so behaviour is unchanged (the raw reference is surfaced rather
    than silently dropping the credential).
    """
    token = _seed_openrouter_token(
        monkeypatch,
        dotenv_value="op://Vault/Item/field",
        environ_value=None,
    )
    assert token == "op://Vault/Item/field"
