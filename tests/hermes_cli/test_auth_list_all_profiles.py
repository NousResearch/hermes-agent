"""``hermes auth list --all-profiles``: every profile's store, and refresh
tokens that live in more than one file.

Providers that rotate the refresh token on every refresh (xai-oauth,
openai-codex, nous) issue single-use grants, so two SEPARATE auth.json files
holding the same refresh token revoke each other the first time either one
refreshes (#43589 / #48415). The root write-through added for those issues
protects a profile that *inherits* the grant from the root store; a profile
holding its own copy of the token has no protection and, before this flag,
no way to see the overlap. These tests pin the report's contract:

* only tokens in DISTINCT files count — a symlink to another profile's store
  is one file, and a profile without auth.json reads the root store;
* the report never prints a token, only its fingerprint;
* an unreadable store is reported and skipped without a side-effecting
  ``.corrupt`` copy, and the other stores are still audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import auth_commands
from hermes_cli.subcommands.auth import build_auth_parser

TOKEN = "rt-" + "x" * 80
OTHER_TOKEN = "rt-" + "y" * 80
FINGERPRINT = hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()[:12]
NO_DUPLICATES = "No refresh token is stored in more than one profile's file."


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    """Root at tmp/.hermes, profiles at tmp/.hermes/profiles/<name>."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _pool_store(refresh=None, *, provider="xai-oauth", label="personal", reason=None, entries=()):
    entry = {
        "id": "e1",
        "label": label,
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": "access",
    }
    if refresh:
        entry["refresh_token"] = refresh
    if reason:
        entry["last_error_reason"] = reason
    return {"version": 1, "providers": {}, "credential_pool": {provider: [entry, *entries]}}


def _home(root, profile):
    home = root if profile == "default" else root / "profiles" / profile
    home.mkdir(parents=True, exist_ok=True)
    return home


def _write(root, profile, payload):
    home = _home(root, profile)
    (home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")
    return home / "auth.json"


def _run(capsys, provider=None):
    auth_commands.auth_list_command(SimpleNamespace(provider=provider, all_profiles=True))
    return capsys.readouterr().out


def test_same_refresh_token_in_two_files_is_reported(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN, label="personal"))
    _write(hermes_root, "chii", _pool_store(TOKEN, label="personal"))

    out = _run(capsys)

    assert "Refresh tokens stored in more than one file:" in out
    assert f"xai-oauth refresh token {FINGERPRINT} in 2 files" in out
    assert "default (personal)" in out and "chii (personal)" in out
    assert NO_DUPLICATES not in out


def test_report_never_prints_the_token(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN))
    _write(hermes_root, "chii", _pool_store(TOKEN))

    out = _run(capsys)

    assert TOKEN not in out
    assert "x" * 20 not in out
    assert FINGERPRINT in out


def test_symlinked_store_is_one_file_not_a_duplicate(hermes_root, capsys):
    real = _write(hermes_root, "default", _pool_store(TOKEN))
    link = _home(hermes_root, "researcher") / "auth.json"
    link.symlink_to(real)

    out = _run(capsys)

    assert "researcher  (same file as default:" in out
    assert NO_DUPLICATES in out


def test_profile_without_store_reads_root_and_is_not_a_duplicate(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN))
    _home(hermes_root, "worker")  # profile dir, no auth.json

    out = _run(capsys)

    assert "worker  (no auth.json — reads the default profile's store)" in out
    assert NO_DUPLICATES in out


def test_unreadable_store_is_skipped_without_side_effects(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN))
    _write(hermes_root, "chii", _pool_store(TOKEN))
    broken = _home(hermes_root, "broken") / "auth.json"
    broken.write_text("{not json", encoding="utf-8")

    out = _run(capsys)

    assert "broken  (" in out and "unreadable — skipped" in out
    # The sweep is read-only: no auth.json.corrupt copy anywhere under the root.
    assert list(hermes_root.rglob("*.corrupt")) == []
    # ...and one bad store does not stop the other stores from being compared.
    assert f"xai-oauth refresh token {FINGERPRINT} in 2 files" in out


def test_distinct_tokens_report_no_duplicates(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN))
    _write(hermes_root, "chii", _pool_store(OTHER_TOKEN))

    out = _run(capsys)

    assert NO_DUPLICATES in out
    assert "Refresh tokens stored in more than one file:" not in out


def test_legacy_singleton_layout_counts_as_a_site(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN))
    _write(
        hermes_root,
        "chii",
        {
            "version": 1,
            "providers": {"xai-oauth": {"tokens": {"refresh_token": TOKEN, "access_token": "a"}}},
            "credential_pool": {},
        },
    )

    out = _run(capsys)

    assert f"xai-oauth refresh token {FINGERPRINT} in 2 files" in out
    assert "chii (legacy singleton)" in out


def test_provider_filter_limits_the_sweep(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN))
    _write(hermes_root, "chii", _pool_store(TOKEN))

    assert NO_DUPLICATES in _run(capsys, provider="openai-codex")
    assert f"xai-oauth refresh token {FINGERPRINT} in 2 files" in _run(capsys, provider="xai-oauth")


def test_same_token_twice_inside_one_file_is_not_cross_profile(hermes_root, capsys):
    twin = {
        "id": "e2",
        "label": "company",
        "auth_type": "oauth",
        "priority": 1,
        "source": "manual:device_code",
        "access_token": "access",
        "refresh_token": TOKEN,
    }
    _write(hermes_root, "default", _pool_store(TOKEN, entries=(twin,)))
    _write(hermes_root, "chii", _pool_store(OTHER_TOKEN))

    out = _run(capsys)

    assert NO_DUPLICATES in out


def test_last_error_reason_is_shown_as_evidence(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN))
    _write(hermes_root, "chii", _pool_store(TOKEN, reason="refresh_token_reused"))

    out = _run(capsys)

    assert "chii (personal) — last error: refresh_token_reused" in out


def test_per_profile_blocks_list_pooled_credentials(hermes_root, capsys):
    _write(hermes_root, "default", _pool_store(TOKEN, label="personal"))
    _write(hermes_root, "chii", {"version": 1, "providers": {}, "credential_pool": {}})

    out = _run(capsys)

    assert "default  (" in out
    assert "  xai-oauth (1 credentials): personal" in out
    assert "  (no pooled credentials)" in out


def test_parser_accepts_all_profiles_flag():
    parser = argparse.ArgumentParser()
    build_auth_parser(parser.add_subparsers(dest="command"), cmd_auth=lambda _args: None)

    assert parser.parse_args(["auth", "list", "--all-profiles"]).all_profiles is True
    assert parser.parse_args(["auth", "list"]).all_profiles is False
    assert parser.parse_args(["auth", "list", "xai-oauth", "--all-profiles"]).provider == "xai-oauth"
