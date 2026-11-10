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
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader
from agent import credential_pool


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
    changed, _sources = credential_pool._seed_from_env("openrouter", entries)
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


def test_credential_pool_never_seeds_an_unresolved_op_ref(monkeypatch):
    """An op:// line with nothing resolved behind it is NOT a credential.

    Pre-resolution / misconfigured edge (a routed profile whose source could not
    authenticate): the old behaviour surfaced the raw reference, which seeded a pool
    row whose ``secret_fingerprint`` is the reference's hash — the provider then got a
    URL as its API key and answered 401, and the bogus row stuck to auth.json. Drop it.
    """
    monkeypatch.setattr(
        credential_pool, "load_env", lambda: {"OPENROUTER_API_KEY": "op://Vault/Item/field"}
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("hermes_cli.auth.is_source_suppressed", lambda _p, _s: False)

    entries: list = []
    changed, _sources = credential_pool._seed_from_env("openrouter", entries)
    assert not changed and not entries


def test_prefer_dotenv_config_getter_drops_an_unresolved_op_ref(monkeypatch):
    """``hermes_cli.config.get_env_value_prefer_dotenv`` refuses the reference too.

    Same rule on the provider-auth path (``_resolve_api_key_provider_secret``): a raw
    ``op://`` line whose scope value is the reference itself must resolve to empty
    rather than to the URL.
    """
    from hermes_cli import config as config_module

    monkeypatch.setattr(config_module, "load_env", lambda: {"COMMANDCODE_API_KEY": "op://Vault/Item/field"})
    monkeypatch.setattr(config_module, "_scoped_environ_get", lambda _k: "op://Vault/Item/field")
    assert config_module.get_env_value_prefer_dotenv("COMMANDCODE_API_KEY") == ""

    monkeypatch.setattr(config_module, "_scoped_environ_get", lambda _k: "resolved-value")
    assert config_module.get_env_value_prefer_dotenv("COMMANDCODE_API_KEY") == "resolved-value"
