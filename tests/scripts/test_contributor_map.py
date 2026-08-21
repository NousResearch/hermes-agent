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


# ── case-collision sidecar behavior (#88257) ──────────────────────────


@pytest.fixture()
def sidecar(tmp_path, monkeypatch):
    import add_contributor

    path = tmp_path / "contributors" / "emails.caseless.json"
    monkeypatch.setattr(add_contributor, "CASELESS_SIDECAR", path)
    return path


def test_sidecar_loader_reads_json_and_skips_malformed(tmp_path):
    good = tmp_path / "caseless.json"
    good.write_text('{"A@x.com": "alice", "empty@x.com": ""}\n')
    assert release._load_caseless_sidecar(good) == {"A@x.com": "alice"}
    assert release._load_caseless_sidecar(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert release._load_caseless_sidecar(bad) == {}
    wrong = tmp_path / "wrong.json"
    wrong.write_text('["not", "a", "dict"]')
    assert release._load_caseless_sidecar(wrong) == {}


def test_effective_map_includes_sidecar():
    sidecar_map = release._load_caseless_sidecar()
    for email, login in sidecar_map.items():
        assert release.AUTHOR_MAP[email] == login


def test_case_colliding_email_goes_to_sidecar(emails_dir, sidecar):
    # A case-variant of an existing mapping cannot become a file (it would
    # clobber the other entry on case-insensitive filesystems).
    emails_dir.mkdir(parents=True, exist_ok=True)
    (emails_dir / "agent@agents-mac-mini.local").write_text("momomojo\n")
    rc = add_contributor("agent@Agents-Mac-mini.local", "skip-agent")
    assert rc == 0
    # is_file() on the capital path is unusable on case-insensitive
    # filesystems (it resolves to the lowercase file); the on-disk name
    # list is the FS-neutral check: no capital-case entry was created.
    on_disk_names = [p.name for p in emails_dir.iterdir()]
    assert "agent@Agents-Mac-mini.local" not in on_disk_names
    import json

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["agent@Agents-Mac-mini.local"] == "skip-agent"
    # The pre-existing mapping is untouched.
    assert read_mapping_file(emails_dir / "agent@agents-mac-mini.local") == "momomojo"


def test_sidecar_entry_is_idempotent(emails_dir, sidecar):
    emails_dir.mkdir(parents=True, exist_ok=True)
    (emails_dir / "agent@agents-mac-mini.local").write_text("momomojo\n")
    assert add_contributor("agent@Agents-Mac-mini.local", "skip-agent") == 0
    assert add_contributor("agent@Agents-Mac-mini.local", "skip-agent") == 0  # present
    # Same email asking for a DIFFERENT login still refuses.
    assert add_contributor("agent@Agents-Mac-mini.local", "someoneelse") == 1


def test_collision_detection_is_case_insensitive(emails_dir):
    from add_contributor import _caseless_collision

    emails_dir.mkdir(parents=True, exist_ok=True)
    (emails_dir / "Jane@Example.com").write_text("janedoe\n")
    # On case-sensitive filesystems iterdir() shows the distinct name; on
    # case-insensitive ones it may show the requested name instead — both
    # must detect the collision via the casefold comparison.
    assert _caseless_collision("jane@example.com", emails_dir) is True
    assert _caseless_collision("other@example.com", emails_dir) is False


def test_exact_name_file_still_refuses_when_variant_also_exists(
    emails_dir, sidecar, monkeypatch
):
    """Case-sensitive state where BOTH files exist (origin/main was in this
    state): the exact-name mapping must still be honored — a different
    login must be refused instead of silently reassigning via the sidecar.
    Simulated by injecting the on-disk name list (both files cannot coexist
    on case-insensitive dev machines)."""
    import add_contributor as _ac

    emails_dir.mkdir(parents=True, exist_ok=True)
    (emails_dir / "agent@Agents-Mac-mini.local").write_text("skip-agent\n")
    monkeypatch.setattr(
        _ac,
        "_dir_names",
        lambda d=None: {"agent@Agents-Mac-mini.local", "agent@agents-mac-mini.local"},
    )
    assert _ac.add_contributor("agent@Agents-Mac-mini.local", "someoneelse") == 1
    assert _ac.add_contributor("agent@Agents-Mac-mini.local", "skip-agent") == 0






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
