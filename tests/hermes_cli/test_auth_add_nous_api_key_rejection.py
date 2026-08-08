"""Tests for #50671 — auth_add must reject --type api-key for OAuth-only providers.

Sibling-fix of #5807: ``get_nous_auth_status`` was fixed to read the credential
pool when the legacy ``providers`` state is empty. The intake side
(``auth_add_command``) had the matching gap — it accepted ``--type api-key``
for any provider without checking whether the runtime resolver could ever read
it back. This created silent dead credentials for ``nous`` (the runtime
resolver requires an inference-scoped JWT minted via OAuth device-code).

These tests assert the new contract: api-key intake is rejected at the CLI
layer for providers in ``_API_KEY_INCOMPATIBLE_PROVIDERS``, with a clear
remediation message pointing at OAuth.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli.auth_commands import (
    _API_KEY_INCOMPATIBLE_PROVIDERS,
    auth_add_command,
)


def _make_args(provider: str, **overrides) -> SimpleNamespace:
    """Build the args namespace ``auth_add_command`` reads from."""
    defaults = dict(
        provider=provider,
        auth_type="api-key",
        api_key="sk-test-fake-key",
        label="should-not-be-saved",
        portal_url=None,
        inference_url=None,
        client_id=None,
        scope=None,
        no_browser=False,
        manual_paste=False,
        timeout=None,
        insecure=False,
        ca_bundle=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_nous_in_api_key_incompatible_providers():
    """The set must contain ``nous`` — that is the whole point of #50671."""
    assert "nous" in _API_KEY_INCOMPATIBLE_PROVIDERS


def test_auth_add_nous_api_key_explicit_type_rejected(tmp_path, monkeypatch, capsys):
    """``hermes auth add nous --type api-key --api-key ...`` must raise SystemExit
    with an OAuth remediation hint, NOT silently write a PooledCredential.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    args = _make_args("nous")
    with pytest.raises(SystemExit) as excinfo:
        auth_add_command(args)

    msg = str(excinfo.value)
    assert "nous" in msg
    assert "OAuth" in msg
    # Remediation hint must be actionable — point at the exact command.
    assert "hermes auth add nous --type oauth" in msg


def test_auth_add_nous_api_key_does_not_create_pool_entry(tmp_path, monkeypatch):
    """Behaviour contract: after a rejected api-key intake, ``credential_pool.nous``
    must NOT contain a new entry. The CLI rejection should fail-fast before any
    write to the auth store.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    args = _make_args("nous")
    with pytest.raises(SystemExit):
        auth_add_command(args)

    store = json.loads(auth_path.read_text())
    pool_entries = store.get("credential_pool", {}).get("nous", [])
    assert pool_entries == [], (
        f"Expected no nous credential pool entries after rejected intake, "
        f"got: {pool_entries}"
    )


def test_auth_add_api_key_still_works_for_compatible_provider(tmp_path, monkeypatch, capsys):
    """Regression guard: providers NOT in ``_API_KEY_INCOMPATIBLE_PROVIDERS``
    must still accept ``--type api-key`` (this is the existing happy path for
    openrouter, anthropic api-key entries, custom providers, etc.).
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # openrouter takes manual api keys by design.
    args = _make_args("openrouter", label="my-test-key")
    auth_add_command(args)

    captured = capsys.readouterr()
    assert "Added openrouter credential" in captured.out


def test_auth_add_nous_api_key_rejection_leaves_auth_store_untouched(tmp_path, monkeypatch):
    """Behaviour contract: the rejection must run BEFORE ``load_pool()`` and the
    suppression-clearing loop, both of which can persist auth state
    (``load_pool`` re-saves changed entries; ``unsuppress_credential_source``
    saves the auth store). A rejected intake must therefore leave the entire
    auth store byte-identical — in particular a pre-existing
    ``suppressed_sources.nous`` entry must survive unchanged.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_path = hermes_home / "auth.json"
    initial_store = {
        "version": 1,
        "providers": {},
        "suppressed_sources": {"nous": ["env:NOUS_API_KEY"]},
    }
    auth_path.write_text(json.dumps(initial_store))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    before = auth_path.read_text()

    args = _make_args("nous")
    with pytest.raises(SystemExit):
        auth_add_command(args)

    after = auth_path.read_text()
    assert after == before, "Rejected api-key intake must not mutate the auth store"
    store = json.loads(after)
    assert store["suppressed_sources"]["nous"] == ["env:NOUS_API_KEY"], (
        "Pre-existing suppressed_sources.nous must survive a rejected intake"
    )


def test_interactive_add_forces_oauth_for_incompatible_provider(tmp_path, monkeypatch, capsys):
    """The interactive menu must NOT offer the api-key option for providers in
    ``_API_KEY_INCOMPATIBLE_PROVIDERS`` — it should route straight to OAuth
    without prompting for a type choice.
    """
    from hermes_cli import auth_commands as ac

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Answer the provider prompt, then any follow-up prompts (label, etc.)
    # with empty strings. A type-choice prompt must never appear for nous —
    # asserted below via the absence of the "1. API key" menu text.
    prompts = iter(["nous"])

    def _fake_input(prompt=""):
        assert "Type [1/2]" not in str(prompt), (
            "Interactive add offered an api-key/oauth type choice for nous"
        )
        try:
            return next(prompts)
        except StopIteration:
            return ""

    monkeypatch.setattr("builtins.input", _fake_input)

    captured_args = {}

    def _fake_auth_add(args):
        captured_args["auth_type"] = getattr(args, "auth_type", None)

    monkeypatch.setattr(ac, "auth_add_command", _fake_auth_add)

    ac._interactive_add()

    out = capsys.readouterr().out
    assert captured_args["auth_type"] == "oauth"
    assert "OAuth" in out
    # The api-key menu option must not be shown for nous.
    assert "1. API key" not in out
