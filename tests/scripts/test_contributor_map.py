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






def test_effective_map_merges_legacy_and_directory():
    # Invariant: every legacy entry survives into the effective map unless
    # shadowed by a directory entry, and the directory contributes on top.
    assert set(release.LEGACY_AUTHOR_MAP) <= (
        set(release.AUTHOR_MAP) | set(release._load_contributor_dir())
    )
    for email, login in release._load_contributor_dir().items():
        assert release.AUTHOR_MAP[email] == login




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






def test_add_refuses_login_conflicting_with_legacy_map(emails_dir):
    email, login = next(iter(release.LEGACY_AUTHOR_MAP.items()))
    assert add_contributor(email, login + "x") == 1
    assert not (emails_dir / email).exists()




def test_add_accepts_legacy_consecutive_hyphen_login(emails_dir):
    # Legacy GitHub accounts with consecutive hyphens are real (Roger--Han);
    # current signup rules forbid them but existing logins remain valid.
    assert add_contributor("roger.hanhong@gmail.com", "Roger--Han") == 0
    assert (emails_dir / "roger.hanhong@gmail.com").read_text(
        encoding="utf-8"
    ).strip().endswith("Roger--Han")


def test_add_strips_at_prefix(emails_dir):
    assert add_contributor("z@z.com", "@zeta") == 0
    assert read_mapping_file(emails_dir / "z@z.com") == "zeta"


def test_cli_entrypoint_end_to_end(tmp_path):
    # Run the real script in a subprocess against a temp repo layout.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("add_contributor.py",):
        # Explicit encoding: add_contributor.py contains UTF-8 multi-byte
        # characters (an em dash), so the locale-default read_text() raises
        # UnicodeDecodeError on non-UTF-8 Windows locales (e.g. cp950).
        (scripts / name).write_text(
            (SCRIPTS_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    # Minimal stub release.py so the legacy lookup import works
    (scripts / "release.py").write_text("LEGACY_AUTHOR_MAP = {}\n")
    proc = subprocess.run(
        [sys.executable, str(scripts / "add_contributor.py"),
         "cli@example.com", "cliperson", "via subprocess"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = (tmp_path / "contributors" / "emails" / "cli@example.com").read_text(encoding="utf-8")
    assert out.splitlines()[0] == "cliperson"


# ── case-insensitive filename collisions ──────────────────────────────
#
# The mapping key IS the filename, so two emails differing only in case are the
# same file on Windows and on default macOS. When both exist, git writes one and
# then reports the other as modified in a FRESH clone, permanently: the repo can
# never be checked out clean on those platforms.
#
# This pair is a real conflict in the tree today and needs a maintainer to say
# which login is correct -- they point at different logins, so deleting either
# silently reassigns commits. It is pinned here so the breakage is visible and,
# more importantly, so it cannot spread.
KNOWN_CASE_CONFLICTS = {
    frozenset(
        {
            "agent@Agents-Mac-mini.local",
            "agent@agents-Mac-mini.local",
        }
    )
}

EMAILS_DIR = REPO_ROOT / "contributors" / "emails"


def test_no_new_case_insensitive_mapping_collisions():
    groups: dict[str, set[str]] = {}
    for entry in EMAILS_DIR.iterdir():
        if entry.is_file():
            groups.setdefault(entry.name.lower(), set()).add(entry.name)

    collisions = {frozenset(names) for names in groups.values() if len(names) > 1}
    unexpected = collisions - KNOWN_CASE_CONFLICTS

    assert not unexpected, (
        "contributor mappings differing only in case cannot coexist on "
        "case-insensitive filesystems (Windows, default macOS) — a fresh clone "
        f"there is permanently dirty: {sorted(sorted(c) for c in unexpected)}"
    )


def test_add_contributor_refuses_a_case_collision(tmp_path, monkeypatch):
    d = tmp_path / "emails"
    d.mkdir()
    (d / "agent@Example-Host.local").write_text("someone\n")

    import add_contributor as mod

    monkeypatch.setattr(mod, "EMAILS_DIR", d)

    assert mod.add_contributor("agent@example-host.local", "otherperson") == 1
    assert not (d / "agent@example-host.local").exists()
