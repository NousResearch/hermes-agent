"""Tests for the conflict-free contributor mapping system.

New contributor email → GitHub login mappings live as one file per email
under contributors/emails/ (additions never merge-conflict). The legacy
AUTHOR_MAP dict in scripts/release.py is frozen; release.py merges both at
import time with the directory winning on duplicates.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import release  # noqa: E402
from add_contributor import add_contributor, read_mapping_file  # noqa: E402


# ── directory loader behavior ─────────────────────────────────────────


def test_loader_reads_login_from_first_noncomment_line(tmp_path):
    d = tmp_path / "emails"
    d.mkdir()
    (d / "jane@example.com").write_text("# salvage PR #1\njanedoe\n# trailing note\n")
    mapping = release._load_contributor_dir(d)
    assert mapping == {"jane@example.com": "janedoe"}


def test_loader_strips_at_prefix_and_skips_dotfiles(tmp_path):
    d = tmp_path / "emails"
    d.mkdir()
    (d / "a@b.com").write_text("@somelogin\n")
    (d / ".gitkeep").write_text("_placeholder\n")
    mapping = release._load_contributor_dir(d)
    assert mapping == {"a@b.com": "somelogin"}


def test_loader_missing_directory_returns_empty(tmp_path):
    assert release._load_contributor_dir(tmp_path / "nope") == {}


def test_effective_map_merges_legacy_and_directory():
    # Invariant: every legacy entry survives into the effective map unless
    # shadowed by a directory entry, and the directory contributes on top.
    assert set(release.LEGACY_AUTHOR_MAP) <= (
        set(release.AUTHOR_MAP) | set(release._load_contributor_dir())
    )
    for email, login in release._load_contributor_dir().items():
        assert release.AUTHOR_MAP[email] == login


def test_resolve_author_uses_directory_entry(tmp_path, monkeypatch):
    d = tmp_path / "emails"
    d.mkdir()
    (d / "dirwin@example.com").write_text("dirwinner\n")
    merged = {**release.LEGACY_AUTHOR_MAP, **release._load_contributor_dir(d)}
    monkeypatch.setattr(release, "AUTHOR_MAP", merged)
    assert release.resolve_author("Dir Winner", "dirwin@example.com") == "@dirwinner"


# ── add_contributor.py CLI behavior ───────────────────────────────────


@pytest.fixture()
def emails_dir(tmp_path, monkeypatch):
    import add_contributor

    d = tmp_path / "contributors" / "emails"
    monkeypatch.setattr(add_contributor, "EMAILS_DIR", d)
    return d


def test_add_creates_mapping_file(emails_dir):
    rc = add_contributor("new@example.com", "newperson", "PR #999 salvage")
    assert rc == 0
    path = emails_dir / "new@example.com"
    assert path.is_file()
    assert read_mapping_file(path) == "newperson"
    assert "# PR #999 salvage" in path.read_text()


def test_add_is_idempotent(emails_dir):
    assert add_contributor("x@y.com", "xperson") == 0
    assert add_contributor("x@y.com", "xperson") == 0
    assert read_mapping_file(emails_dir / "x@y.com") == "xperson"


def test_add_refuses_conflicting_login(emails_dir):
    assert add_contributor("x@y.com", "xperson") == 0
    assert add_contributor("x@y.com", "someoneelse") == 1
    # original mapping untouched
    assert read_mapping_file(emails_dir / "x@y.com") == "xperson"


def test_add_refuses_login_conflicting_with_legacy_map(emails_dir):
    email, login = next(iter(release.LEGACY_AUTHOR_MAP.items()))
    assert add_contributor(email, login + "x") == 1
    assert not (emails_dir / email).exists()


def test_add_rejects_invalid_email_and_login(emails_dir):
    assert add_contributor("not-an-email", "ok") == 2
    assert add_contributor("has space@x.com", "ok") == 2
    assert add_contributor("a/b@x.com", "ok") == 2  # path separator
    assert add_contributor("a@b.com", "-bad-") == 2
    assert not emails_dir.exists() or not any(
        p for p in emails_dir.iterdir() if not p.name.startswith(".")
    )


def test_add_strips_at_prefix(emails_dir):
    assert add_contributor("z@z.com", "@zeta") == 0
    assert read_mapping_file(emails_dir / "z@z.com") == "zeta"


def test_cli_entrypoint_end_to_end(tmp_path):
    # Run the real script in a subprocess against a temp repo layout.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("add_contributor.py",):
        (scripts / name).write_text((SCRIPTS_DIR / name).read_text())
    # Minimal stub release.py so the legacy lookup import works
    (scripts / "release.py").write_text("LEGACY_AUTHOR_MAP = {}\n")
    proc = subprocess.run(
        [sys.executable, str(scripts / "add_contributor.py"),
         "cli@example.com", "cliperson", "via subprocess"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = (tmp_path / "contributors" / "emails" / "cli@example.com").read_text()
    assert out.splitlines()[0] == "cliperson"
