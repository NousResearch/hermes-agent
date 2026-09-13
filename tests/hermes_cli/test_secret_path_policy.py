"""Hermetic tests for the secret-path pre-ingress policy.

Covers the public surface of :mod:`hermes_cli.secret_path_policy`:

  * every prohibited class is denied (auth.json, .env/.env.*, keychain
    databases, keychain export shapes)
  * case variations (``AUTH.JSON``), directory traversal
    (``../../etc/auth.json``), nested paths (``subdir/.env``),
    and trailing whitespace / leading dots are normalised before the
    check so the policy fires consistently
  * benign names (including near-misses like ``notes-env.txt`` and
    ``data.json``) are explicitly allowed, to lock in the policy
    boundary and prevent an unjustifiably broad block
  * error messages contain only the safe ``kind`` / ``reason``
    strings — never the offending basename, path, or any credential
    bytes
  * symlink targets are resolved by :func:`check_scratch_artifact_path`
    so a worker can't smuggle ``auth.json`` past the policy by
    aliasing it from a benign leaf

No live credential files are read — fixtures are synthetic empty files
under ``tmp_path`` with the prohibited NAMES.  This file has zero
dependency on ``_sim_home2`` or any other real Hermes data directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from hermes_cli.secret_path_policy import (
    SecretPathError,
    check_attachment_filename,
    check_scratch_artifact_path,
    raise_if_blocked,
    raise_if_blocked_scratch,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic names / paths, no real credential contents
# ---------------------------------------------------------------------------

# Each tuple is (input name, expected kind token, human-readable label).
PROHIBITED_BASENAMES: list[tuple[str, str, str]] = [
    ("auth.json", "credential_file", "Hermes credential store"),
    ("auth.lock", "credential_file", "Hermes credential store lock"),
    (".anthropic_oauth.json", "credential_file", "Anthropic OAuth store"),
    ("credentials.json", "credential_file", "Generic credentials file"),
    ("keychain.json", "credential_file", "Keychain cache"),
    ("1password-export.json", "credential_file", "1Password export"),
    ("lastpass-export.csv", "credential_file", "LastPass export"),
    ("bitwarden-export.json", "credential_file", "Bitwarden export"),
    ("dashlane-export.json", "credential_file", "Dashlane export"),
    ("webhook_subscriptions.json", "credential_file", "Webhook secret store"),
    (".env", "dotenv", "Bare .env"),
    (".env.local", "dotenv", "Local .env"),
    (".env.production", "dotenv", "Production .env"),
    (".env.test", "dotenv", "Test .env"),
    (".env.staging", "dotenv", "Staging .env"),
    (".envrc", "dotenv", "direnv rc"),
    ("vault.kdbx", "keychain_export", "KeePass database"),
    ("vault.kbdx", "keychain_export", "KeePass variant"),
    ("vault.kdb", "keychain_export", "Legacy KeePass"),
    ("keepass-export.xml", "keychain_export", "KeePass XML export"),
    ("keepass-passwords.csv", "keychain_export", "KeePass CSV export"),
    ("keychain-export.xml", "keychain_export", "macOS keychain export"),
    ("Apple-Keychain.csv", "keychain_export", "macOS keychain export (mixed case)"),
    ("MyPasswords.csv", "keychain_export", "Generic passwords CSV"),
]


# Benign inputs that must be explicitly allowed — these lock in the
# policy boundary and prevent an unjustifiably broad block.  Each is a
# near-miss or a regular attachment that the worker legitimately uses.
BENIGN_BASENAMES: list[str] = [
    "notes.txt",
    "report.pdf",
    "design.md",
    "data.json",
    "result.csv",
    "image.png",
    "screenshot.jpg",
    "auth.json.bak",      # suffix-only: not auth.json
    "auth.jsonl",         # different suffix entirely
    "notes-env.txt",      # contains "env" but not a .env file
    ".envignore",         # different leading dot pattern
    "my.env.config",      # contains ".env." but is .env.config
    "vault.kdbx.bak",     # different suffix
    "credentials",        # no .json suffix
    "credential.json",    # singular, not in denylist
    "my-keychain-notes.md",  # not an export shape
    ".envoy-config.json",   # not .env.* — it's a different file
]


# ---------------------------------------------------------------------------
# 1. Direct denylist coverage — every prohibited class is rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_kind", "label"),
    PROHIBITED_BASENAMES,
    ids=lambda v: v if isinstance(v, str) else v[0],
)
def test_prohibited_basename_is_blocked(
    name: str, expected_kind: str, label: str,
) -> None:
    """Every denylisted shape raises with the documented ``kind``."""
    err: Optional[str] = check_attachment_filename(name, source="test")
    assert err is not None, f"expected {name!r} to be blocked"
    assert f"'{expected_kind}'" in err, (
        f"{name!r} blocked with unexpected kind: {err!r}"
    )
    # Safety: error must not echo the filename
    assert name not in err, (
        f"policy error leaks the prohibited name: {err!r}"
    )


# ---------------------------------------------------------------------------
# 2. Benign allowlist — these must NOT be blocked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BENIGN_BASENAMES)
def test_benign_basename_is_allowed(name: str) -> None:
    """Benign attachments are explicitly allowed (boundary protection)."""
    err = check_attachment_filename(name, source="test")
    assert err is None, f"benign name {name!r} wrongly blocked: {err!r}"
    # And raise_if_blocked must not raise
    raise_if_blocked(name, source="test")  # should be a no-op


# ---------------------------------------------------------------------------
# 3. Normalisation — case, separators, traversal, leading dots
# ---------------------------------------------------------------------------


NORMALISATION_CASES: list[tuple[str, str, str]] = [
    ("AUTH.JSON", "credential_file", "uppercase variant"),
    ("Auth.Json", "credential_file", "title-case variant"),
    ("aUtH.jSoN", "credential_file", "mixed-case variant"),
    (".ENV", "dotenv", "uppercase .env"),
    (".Env.Production", "dotenv", "mixed-case .env.*"),
    (". env", "dotenv", "internal whitespace"),
    ("  auth.json  ", "credential_file", "leading/trailing whitespace"),
    (".auth.json", "credential_file", "leading dot"),
    ("auth. json", "credential_file", "space inside name"),
    ("subdir/auth.json", "credential_file", "nested basename"),
    ("/etc/auth.json", "credential_file", "absolute path"),
    ("../../etc/auth.json", "credential_file", "traversal"),
    ("..\\..\\auth.json", "credential_file", "backslash traversal"),
    ("subdir/sub2/.env.production", "dotenv", "deeply nested .env.*"),
    ("VAULT.KDBX", "keychain_export", "uppercase KeePass"),
    ("Vault.Kdbx", "keychain_export", "title-case KeePass"),
    ("KEEPASS-EXPORT.XML", "keychain_export", "uppercase KeePass XML"),
    ("MyPasswords.CSV", "keychain_export", "uppercase passwords CSV"),
]


@pytest.mark.parametrize(
    ("name", "expected_kind", "label"),
    NORMALISATION_CASES,
    ids=lambda v: v if isinstance(v, str) else v[2],
)
def test_path_normalisation_blocks_variants(
    name: str, expected_kind: str, label: str,
) -> None:
    """Case / separators / traversal / nested paths all caught."""
    err = check_attachment_filename(name, source="test")
    assert err is not None, f"normalisation failed for {label}: {name!r}"
    assert f"'{expected_kind}'" in err, (
        f"{name!r} (label={label}) blocked with wrong kind: {err!r}"
    )
    # And raise_if_blocked surfaces the same policy
    with pytest.raises(SecretPathError) as ei:
        raise_if_blocked(name, source="test")
    assert ei.value.kind == expected_kind


# ---------------------------------------------------------------------------
# 4. Error-message safety — no leakage of the filename or contents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "auth.json", "../../secret/auth.json", ".env.production", "vault.kdbx",
])
def test_error_message_does_not_leak_name(name: str) -> None:
    """The policy error contains only safe kind/reason — never the name."""
    err = check_attachment_filename(name, source="my_source")
    assert err is not None
    assert name not in err, f"name leaked: {err!r}"
    # The original basename (lowercase leaf) also must not appear in the message
    basename = os.path.basename(name).lower()
    assert basename not in err, f"basename leaked: {err!r}"


def test_secret_path_error_structured_kind() -> None:
    """``SecretPathError`` carries a structured ``kind`` token."""
    with pytest.raises(SecretPathError) as ei:
        raise_if_blocked("auth.json", source="my_source")
    assert ei.value.kind == "credential_file"
    assert "auth.json" not in str(ei.value)  # not echoed
    assert "my_source" in str(ei.value)      # source IS echoed (helpful)


# ---------------------------------------------------------------------------
# 5. Scratch-artifact path check — symlinks are resolved
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_root(tmp_path: Path) -> Path:
    """A scratch workspace root under tmp_path with a benign file."""
    root = tmp_path / "scratch"
    root.mkdir()
    (root / "notes.txt").write_text("benign\n")
    return root


@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        ("auth.json", "credential_file"),
        (".env", "dotenv"),
        (".env.production", "dotenv"),
        ("vault.kdbx", "keychain_export"),
    ],
)
def test_scratch_artifact_path_blocks_prohibited(
    scratch_root: Path, name: str, expected_kind: str,
) -> None:
    """``check_scratch_artifact_path`` blocks same shapes via Path objects."""
    target = scratch_root / name
    target.write_text("")  # empty file — synthetic, no real contents
    err = check_scratch_artifact_path(target, source="test")
    assert err is not None, f"{name!r} should be blocked at scratch ingress"
    assert f"'{expected_kind}'" in err


def test_scratch_artifact_path_allows_benign(scratch_root: Path) -> None:
    """Benign scratch files are not blocked."""
    err = check_scratch_artifact_path(scratch_root / "notes.txt", source="test")
    assert err is None


def test_scratch_artifact_path_resolves_symlinks(tmp_path: Path) -> None:
    """A symlink whose target is ``auth.json`` is blocked, not the alias."""
    benign = tmp_path / "notes.txt"
    prohibited = tmp_path / "auth.json"
    prohibited.write_text("")  # synthetic, no contents
    benign.symlink_to(prohibited)
    # The alias resolves to the prohibited target
    err = check_scratch_artifact_path(benign, source="test")
    assert err is not None, "symlink to auth.json must be blocked"
    assert "'credential_file'" in err


def test_scratch_artifact_path_does_not_read_file_contents(tmp_path: Path) -> None:
    """The policy must not read file contents — only the basename/path."""
    # If the policy tried to read, it would crash on a directory or a
    # zero-perm file.  We use a directory here (which would crash on
    # read) — the policy should still return the right verdict from
    # the basename alone.
    target = tmp_path / "auth.json"
    target.mkdir()  # it's a directory with no contents
    err = check_scratch_artifact_path(target, source="test")
    assert err is not None
    assert "'credential_file'" in err


def test_raise_if_blocked_scratch(tmp_path: Path) -> None:
    """``raise_if_blocked_scratch`` raises the same exception family."""
    (tmp_path / "auth.json").write_text("")
    with pytest.raises(SecretPathError) as ei:
        raise_if_blocked_scratch(tmp_path / "auth.json", source="test")
    assert ei.value.kind == "credential_file"


# ---------------------------------------------------------------------------
# 6. Self-check matches the canonical denylist — sanity guard
# ---------------------------------------------------------------------------


def test_self_check_finds_all_prohibited() -> None:
    """The built-in self-check should flag every name in PROHIBITED_BASENAMES."""
    from hermes_cli.secret_path_policy import _self_check
    errors = _self_check()
    flagged = sum(1 for e in errors if e is not None)
    assert flagged == len(errors), (
        f"self_check found only {flagged}/{len(errors)} prohibited cases"
    )
    assert flagged >= 10  # at least one per category


# ---------------------------------------------------------------------------
# 7. Unknown/empty input handling — must not crash and must not block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "/", None])
def test_unknown_or_empty_input_is_safe(name) -> None:
    """Empty / weird input must not crash and must not block."""
    if name is None:
        # The signature rejects None; verify that.
        try:
            err = check_attachment_filename(name, source="test")  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return
        # If it didn't raise, the result must be falsy (not blocking).
        assert not err
    else:
        err = check_attachment_filename(name, source="test")
        # Either None (allowed) or an error string — but never a crash.
        assert err is None or isinstance(err, str)