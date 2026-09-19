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


# ── case-collision guard ──────────────────────────────────────────────
#
# A mapping filename IS an email address, and the hostname part of an email
# is case-insensitive per DNS. Two files differing only by case are therefore
# two spellings of one address — and on case-insensitive filesystems
# (macOS/Windows) they cannot both exist on disk, so a checkout shows one of
# them as permanently modified and `git rebase` refuses to run.
#
# These tests must pass on BOTH case-sensitive and case-insensitive
# filesystems, so they never assert on `path.exists()` for the variant
# spelling (that answer differs per platform). They count real directory
# entries instead.


def _entry_count(d):
    return len([p for p in d.iterdir() if p.is_file()])


def test_find_mapping_file_matches_case_insensitively(tmp_path):
    """The lookup compares real directory entry names, so it is FS-independent.

    ``Path.is_file()`` cannot carry this check: it answers True for a variant
    spelling on macOS/Windows and False on Linux, which is exactly why the
    collision only ever gets created on a case-sensitive filesystem. This test
    therefore asserts on the returned *stored spelling*, which is a pure string
    result and identical on every host.
    """
    import add_contributor as ac

    d = tmp_path / "emails"
    d.mkdir()
    (d / "agent@agents-Mac-mini.local").write_text("momomojo\n", encoding="utf-8")

    # Variant spelling resolves to the stored spelling, not the queried one.
    assert ac.find_mapping_file("agent@Agents-Mac-mini.local", d) == (
        "agent@agents-Mac-mini.local"
    )
    assert ac.find_mapping_file("AGENT@AGENTS-MAC-MINI.LOCAL", d) == (
        "agent@agents-Mac-mini.local"
    )
    # Exact spelling resolves to itself.
    assert ac.find_mapping_file("agent@agents-Mac-mini.local", d) == (
        "agent@agents-Mac-mini.local"
    )
    # Unrelated emails never match.
    assert ac.find_mapping_file("someone@else.example", d) is None
    # A missing directory is not an error.
    assert ac.find_mapping_file("a@b.com", tmp_path / "nope") is None


def test_find_case_variant_excludes_the_exact_spelling(tmp_path):
    """``find_case_variant`` is the "someone else already owns this" signal."""
    import add_contributor as ac

    d = tmp_path / "emails"
    d.mkdir()
    (d / "agent@agents-Mac-mini.local").write_text("momomojo\n", encoding="utf-8")

    assert ac.find_case_variant("agent@Agents-Mac-mini.local", d) == (
        "agent@agents-Mac-mini.local"
    )
    # Exact spelling is not a variant — that is the normal update path.
    assert ac.find_case_variant("agent@agents-Mac-mini.local", d) is None
    assert ac.find_case_variant("someone@else.example", d) is None


def test_find_mapping_file_prefers_exact_when_both_spellings_exist(tmp_path):
    """A case-sensitive checkout can hold both — the exact spelling wins.

    This is the state on origin/main today (two entries differing only in the
    hostname case). Resolving to the exact spelling keeps an already-correct
    mapping authoritative instead of letting entry order decide.
    """
    import add_contributor as ac

    d = tmp_path / "emails"
    d.mkdir()
    (d / "dev@host.local").write_text("lowercase-owner\n", encoding="utf-8")
    variant = d / "dev@Host.local"
    if variant.exists():  # case-insensitive host: cannot stage both
        pytest.skip("case-insensitive filesystem cannot hold both spellings")
    variant.write_text("uppercase-owner\n", encoding="utf-8")

    assert ac.find_mapping_file("dev@host.local", d) == "dev@host.local"
    assert ac.find_mapping_file("dev@Host.local", d) == "dev@Host.local"


def test_add_refuses_case_variant_of_existing_email(emails_dir):
    """A case-variant spelling must error out, not create a second file."""
    emails_dir.mkdir(parents=True, exist_ok=True)
    (emails_dir / "agent@agents-Mac-mini.local").write_text(
        "momomojo\n", encoding="utf-8"
    )

    rc = add_contributor("agent@Agents-Mac-mini.local", "skip-agent")

    assert rc == 1
    assert _entry_count(emails_dir) == 1, (
        "a case-variant must not add a second file — on macOS/Windows the two "
        "cannot coexist on disk"
    )
    # The pre-existing mapping is left untouched for a human to resolve.
    assert read_mapping_file(emails_dir / "agent@agents-Mac-mini.local") == "momomojo"


def test_add_refuses_case_variant_even_when_login_matches(emails_dir):
    """Same login is still a refusal — silently reusing hides the collision.

    Returning 0 here would let CI's --fix path believe it created a mapping
    for the variant spelling, so the underlying "two people share one
    machine-default email" problem would never surface to a human.
    """
    emails_dir.mkdir(parents=True, exist_ok=True)
    (emails_dir / "dev@Host.local").write_text("samelogin\n", encoding="utf-8")

    rc = add_contributor("dev@host.local", "samelogin")

    assert rc == 1
    assert _entry_count(emails_dir) == 1


def test_add_case_variant_error_names_both_spellings(emails_dir, capsys):
    """The error must be actionable: print both spellings and both logins."""
    emails_dir.mkdir(parents=True, exist_ok=True)
    (emails_dir / "agent@agents-Mac-mini.local").write_text(
        "momomojo\n", encoding="utf-8"
    )

    add_contributor("agent@Agents-Mac-mini.local", "skip-agent")

    err = capsys.readouterr().err
    assert "agent@agents-Mac-mini.local" in err
    assert "agent@Agents-Mac-mini.local" in err
    assert "momomojo" in err
    assert "skip-agent" in err


def test_add_exact_match_still_idempotent(emails_dir):
    """The guard must not break the existing same-spelling/same-login path."""
    assert add_contributor("exact@example.com", "someone") == 0
    assert add_contributor("exact@example.com", "someone") == 0
    assert _entry_count(emails_dir) == 1


def test_add_unrelated_emails_are_unaffected(emails_dir):
    """Emails that differ by more than case must still both be addable."""
    assert add_contributor("one@example.com", "personone") == 0
    assert add_contributor("two@example.com", "persontwo") == 0
    assert _entry_count(emails_dir) == 2


def test_lookup_folds_pairs_that_lower_misses(tmp_path):
    """The fold must not be narrower than a filesystem's own folding.

    ``str.lower()`` leaves German ß alone while ``casefold()`` maps it to
    ``ss``; a filesystem that folds the pair would alias the two spellings on
    disk while a ``lower()``-based guard waved the second one through. Guarding
    with ``casefold()`` keeps the check at least as aggressive as any
    filesystem's.
    """
    import add_contributor as ac

    d = tmp_path / "emails"
    d.mkdir()
    (d / "dev@strasse.local").write_text("owner\n", encoding="utf-8")

    assert "AGENT@STRASSE.LOCAL".lower() != "agent@straße.local".lower()
    assert ac.find_mapping_file("dev@straße.local", d) == "dev@strasse.local"


# ── audit_pr_attribution.is_mapped ────────────────────────────────────
#
# The audit's --fix path shells out to add_contributor.py. Now that
# add_contributor refuses a case-variant spelling, is_mapped must recognise
# the variant as already mapped — otherwise --fix would loop on an address it
# can never create a file for.


@pytest.fixture()
def audit_repo(tmp_path, monkeypatch):
    import audit_pr_attribution as audit

    (tmp_path / "contributors" / "emails").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "release.py").write_text(
        "LEGACY_AUTHOR_MAP = {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    return tmp_path


def test_is_mapped_uses_the_shared_case_insensitive_lookup(audit_repo, monkeypatch):
    """is_mapped must route through find_mapping_file, not re-probe the path.

    Asserting the delegation (rather than only the boolean) is what makes this
    meaningful on a case-insensitive host: the pre-fix ``.is_file()`` probe
    returns True for a variant spelling there, so a boolean-only assertion
    passes with or without the fix.
    """
    import audit_pr_attribution as audit

    seen: list[str] = []
    real = audit.find_mapping_file

    def _spy(email, emails_dir=None):
        seen.append(email)
        return real(email, emails_dir)

    monkeypatch.setattr(audit, "find_mapping_file", _spy)
    (audit_repo / "contributors" / "emails" / "agent@agents-Mac-mini.local").write_text(
        "momomojo\n", encoding="utf-8"
    )

    assert audit.is_mapped("agent@Agents-Mac-mini.local") is True
    assert seen == ["agent@Agents-Mac-mini.local"]


def test_is_mapped_accepts_exact_and_variant_spellings(audit_repo):
    import audit_pr_attribution as audit

    (audit_repo / "contributors" / "emails" / "agent@agents-Mac-mini.local").write_text(
        "momomojo\n", encoding="utf-8"
    )

    assert audit.is_mapped("agent@agents-Mac-mini.local") is True
    assert audit.is_mapped("agent@Agents-Mac-mini.local") is True


def test_is_mapped_still_rejects_unmapped_email(audit_repo):
    import audit_pr_attribution as audit

    (audit_repo / "contributors" / "emails" / "known@example.com").write_text(
        "known\n", encoding="utf-8"
    )

    assert audit.is_mapped("stranger@example.com") is False


def test_is_mapped_tolerates_missing_emails_dir(tmp_path, monkeypatch):
    """A repo without contributors/emails/ must not raise."""
    import audit_pr_attribution as audit

    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    assert audit.is_mapped("someone@example.com") is False


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
# The historical agent@Agents-Mac-mini.local / agent@agents-Mac-mini.local pair
# was removed from the tree (fcdae2cf0b), so there is no allowlist: any pair
# is a regression. scripts/check-case-collisions.py enforces the same
# invariant repo-wide in CI; this test keeps it visible next to the writer.
EMAILS_DIR = REPO_ROOT / "contributors" / "emails"


def test_no_case_insensitive_mapping_collisions():
    groups: dict[str, set[str]] = {}
    for entry in EMAILS_DIR.iterdir():
        if entry.is_file():
            groups.setdefault(entry.name.casefold(), set()).add(entry.name)

    collisions = {frozenset(names) for names in groups.values() if len(names) > 1}

    assert not collisions, (
        "contributor mappings differing only in case cannot coexist on "
        "case-insensitive filesystems (Windows, default macOS) — a fresh clone "
        f"there is permanently dirty: {sorted(sorted(c) for c in collisions)}"
    )


def test_add_contributor_refuses_a_case_collision(tmp_path, monkeypatch):
    d = tmp_path / "emails"
    d.mkdir()
    (d / "agent@Example-Host.local").write_text("someone\n")

    import add_contributor as mod

    monkeypatch.setattr(mod, "EMAILS_DIR", d)

    assert mod.add_contributor("agent@example-host.local", "otherperson") == 1
    assert not (d / "agent@example-host.local").exists()


def test_add_contributor_refuses_case_collision_even_for_same_login(emails_dir, capsys):
    # Same login, different spelling: still refused — the problem is the
    # filename pair, not the login. The exact spelling is what's "present".
    emails_dir.mkdir(parents=True)
    (emails_dir / "Foo@Example.com").write_text("foouser\n")

    assert add_contributor("foo@example.com", "foouser") == 1
    assert "Foo@Example.com" in capsys.readouterr().err
    assert sorted(p.name for p in emails_dir.iterdir()) == ["Foo@Example.com"]
    # Exact-case re-add is the ordinary idempotent path.
    assert add_contributor("Foo@Example.com", "foouser") == 0


def test_case_collision_uses_casefold(emails_dir):
    # casefold, not lower: matches how macOS/Windows fold non-ASCII (ß ~ ss).
    emails_dir.mkdir(parents=True)
    (emails_dir / "strasse@example.com").write_text("someone\n")
    assert add_contributor("STRASSE@example.com", "someone") == 1
    assert add_contributor("straße@example.com", "someone") == 1
