"""``hermes auth list --all-profiles``: every profile's store, and refresh
tokens that live in more than one file.

Providers that rotate the refresh token on every refresh (xai-oauth,
openai-codex, nous) issue single-use grants, so two SEPARATE auth.json files
holding the same refresh token revoke each other the first time either one
refreshes (#43589 / #48415). The root write-through added for those issues
protects a profile that *inherits* the grant from the root store; a profile
holding its own copy of the token has no protection and, before this flag,
no way to see the overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import auth_commands
from hermes_cli.subcommands.auth import build_auth_parser

TOKEN = "rt-" + "x" * 80
OTHER_TOKEN = "rt-" + "y" * 80
FINGERPRINT = hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()[:12]
NO_DUPLICATES = "No refresh token is stored in more than one profile's file."
REPORTED = f"xai-oauth refresh token {FINGERPRINT} in 2 files"


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


LEGACY_STORE = {
    "version": 1,
    "providers": {"xai-oauth": {"tokens": {"refresh_token": TOKEN, "access_token": "a"}}},
    "credential_pool": {},
}


def _home(root, profile):
    home = root if profile == "default" else root / "profiles" / profile
    home.mkdir(parents=True, exist_ok=True)
    return home


def _write(root, profile, payload):
    home = _home(root, profile)
    (home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")
    return home / "auth.json"


def _run(capsys, *argv):
    """Through the real parser, so the flag's plumbing is exercised by every row."""
    parser = argparse.ArgumentParser()
    build_auth_parser(parser.add_subparsers(dest="command"), cmd_auth=lambda _args: None)
    auth_commands.auth_list_command(parser.parse_args(["auth", "list", *argv, "--all-profiles"]))
    return capsys.readouterr().out


@pytest.mark.parametrize(
    ("second_store", "argv", "expected"),
    [
        (_pool_store(TOKEN, reason="refresh_token_reused"), (), "chii (personal) — last error: refresh_token_reused"),
        (LEGACY_STORE, (), "chii (legacy singleton)"),
        (_pool_store(TOKEN), ("xai-oauth",), REPORTED),
        (_pool_store(TOKEN), ("openai-codex",), NO_DUPLICATES),
    ],
    ids=["pool-copy-with-evidence", "legacy-singleton-copy", "provider-filter-hit", "provider-filter-miss"],
)
def test_a_refresh_token_held_in_two_files_is_reported_by_fingerprint_only(hermes_root, capsys, second_store, argv, expected):
    """Both store layouts count as a site, the last error rides along as evidence, the provider
    filter narrows the sweep, and the token bytes never reach the output."""
    _write(hermes_root, "default", _pool_store(TOKEN, label="personal"))
    _write(hermes_root, "chii", second_store)

    out = _run(capsys, *argv)

    assert expected in out
    assert (REPORTED in out) is (expected != NO_DUPLICATES)
    assert "default (personal)" in out or expected == NO_DUPLICATES
    assert TOKEN not in out and "x" * 20 not in out


@pytest.mark.parametrize(
    ("second_profile", "line"),
    [
        ("symlink", "researcher  (same file as default:"),
        ("no-store", "worker  (no auth.json — reads the default profile's store)"),
        ("distinct-token", "chii  ("),
        ("twin-in-one-file", "chii  ("),
    ],
    ids=["symlinked-store", "profile-without-store", "distinct-tokens", "same-token-twice-in-one-file"],
)
def test_only_tokens_in_distinct_files_count(hermes_root, capsys, second_profile, line):
    """A symlink is one file, a profile without auth.json reads the root store, and a token
    repeated inside one file is not a cross-profile copy."""
    twin = {"id": "e2", "label": "company", "auth_type": "oauth", "priority": 1,
            "source": "manual:device_code", "access_token": "access", "refresh_token": TOKEN}
    real = _write(hermes_root, "default", _pool_store(TOKEN, entries=(twin,) if second_profile == "twin-in-one-file" else ()))
    if second_profile == "symlink":
        (_home(hermes_root, "researcher") / "auth.json").symlink_to(real)
    elif second_profile == "no-store":
        _home(hermes_root, "worker")
    else:
        _write(hermes_root, "chii", _pool_store(OTHER_TOKEN))

    out = _run(capsys)

    assert line in out
    assert NO_DUPLICATES in out
    assert "Refresh tokens stored in more than one file:" not in out


def test_an_unreadable_store_is_skipped_without_side_effects_and_the_rest_is_still_audited(hermes_root, capsys):
    """The sweep is read-only: no auth.json.corrupt copy lands in another profile, one bad store
    does not stop the others from being compared, and each readable store lists its pool."""
    _write(hermes_root, "default", _pool_store(TOKEN, label="personal"))
    _write(hermes_root, "chii", _pool_store(TOKEN))
    _write(hermes_root, "empty", {"version": 1, "providers": {}, "credential_pool": {}})
    (_home(hermes_root, "broken") / "auth.json").write_text("{not json", encoding="utf-8")

    out = _run(capsys)

    assert "broken  (" in out and "unreadable — skipped" in out
    assert list(hermes_root.rglob("*.corrupt")) == []
    assert REPORTED in out
    assert "  xai-oauth (1 credentials): personal" in out
    assert "  (no pooled credentials)" in out
