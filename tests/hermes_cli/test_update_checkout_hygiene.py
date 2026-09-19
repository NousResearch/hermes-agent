"""Pre-update checkout-hygiene audit (production incident 2026-09-16).

On production, repeated ``hermes update`` runs against a diverged checkout
silently accumulated 164 untracked files and 4 parked ``hermes-update-
autostash-*`` stashes over weeks; the next activation then had to untangle a
797-behind/dirty tree with backup-and-remove surgery. The audit
(``_audit_checkout_hygiene``) turns that silent accumulation into a hard
pre-fetch refusal: untracked-file count and stale-autostash count each have a
config limit, and passing either refuses the update before the fetch/stash/pull
touches anything.

Behavioral tests use real git repos; git failures are never refusals (the
updater's own guards own those); the audit reads its limits from the
``updates:`` config section. No production mocking of the code under test.
"""

import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import update_cmd


GIT = ["git"]


def _git(cwd, *args, check=True):
    return subprocess.run(GIT + list(args), cwd=cwd, capture_output=True, text=True, check=check)


def _make_repo(tmp_path, *, ignored_entry=None):
    """A real repo with one commit and a clean tree."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    if ignored_entry is not None:
        (tmp_path / ".gitignore").write_text(ignored_entry + "\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch):
    """The audit reads limits from the machine's real config otherwise."""
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: {})


def _add_stale_autostash(cwd, age_days, name=None):
    """Park a hermes-update-autostash entry aged ``age_days`` days."""
    stamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime("%Y%m%d-%H%M%S")
    name = name or f"hermes-update-autostash-{stamp}"
    (cwd / "tracked.txt").write_text("local change\n")
    _git(cwd, "stash", "push", "--include-untracked", "-m", name)
    return name


# ---------------------------------------------------------------------------
# Untracked-file accumulation
# ---------------------------------------------------------------------------

def test_clean_checkout_passes(tmp_path):
    repo = _make_repo(tmp_path)
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is None
    assert facts["untracked_count"] == 0
    assert facts["stale_autostashes"] == 0


def test_few_untracked_files_pass(tmp_path):
    repo = _make_repo(tmp_path)
    for i in range(5):
        (repo / f"scratch_{i}.py").write_text("x\n")
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is None
    assert facts["untracked_count"] == 5


def test_untracked_pile_refuses(tmp_path):
    """The production shape: dozens of leaked files → refuse before fetching."""
    repo = _make_repo(tmp_path)
    for i in range(26):  # default limit 25
        (repo / f"leaked_{i}.py").write_text("x\n")
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is not None
    assert "26 untracked files" in reason
    assert facts["untracked_count"] == 26


def test_ignored_files_never_count(tmp_path):
    repo = _make_repo(tmp_path, ignored_entry="node_modules/")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "pkg.js").write_text("x\n")
    for i in range(30):
        (repo / f"leaked_{i}.py").write_text("x\n") if i < 1 else None
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert facts["untracked_count"] == 1  # node_modules ignored
    assert reason is None


def test_untracked_limit_is_configurable(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    for i in range(3):
        (repo / f"scratch_{i}.py").write_text("x\n")
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: {"max_untracked_files": 2})
    reason, _ = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is not None
    assert "3 untracked files" in reason
    # limit 0 refuses on any untracked file
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: {"max_untracked_files": 0})
    reason, _ = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is not None


def test_invalid_limit_falls_back_to_default(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    for i in range(26):
        (repo / f"leaked_{i}.py").write_text("x\n")
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: {"max_untracked_files": "not-a-number"})
    reason, _ = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is not None  # default 25 still applies


# ---------------------------------------------------------------------------
# Stale-autostash accumulation
# ---------------------------------------------------------------------------

def test_stale_autostash_pile_refuses(tmp_path):
    """3 parked orphans (production had 3+) → refuse; each is unclaimed work."""
    repo = _make_repo(tmp_path)
    for age in (9, 11, 20):
        _add_stale_autostash(repo, age)
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is not None
    assert "3 update autostash entries" in reason
    assert facts["stale_autostashes"] == 3


def test_fresh_autostash_never_refuses(tmp_path):
    """A recent park (desktop --keep-stash minutes ago) is legitimate."""
    repo = _make_repo(tmp_path)
    _add_stale_autostash(repo, age_days=1)
    reason, _ = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is None


def test_non_autostash_stashes_do_not_count(tmp_path):
    repo = _make_repo(tmp_path)
    for i in range(6):  # a user's own WIP stashes, months old
        (repo / "tracked.txt").write_text(f"change {i}\n")
        _git(repo, "stash", "push", "-m", "my own WIP")
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is None
    assert facts["stale_autostashes"] == 0


def test_at_the_limit_passes(tmp_path):
    """Exactly the limit (2) is fine — the audit refuses only PAST the limit."""
    repo = _make_repo(tmp_path)
    for age in (9, 15):
        _add_stale_autostash(repo, age)
    reason, _ = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is None


# ---------------------------------------------------------------------------
# Git failures are never refusals
# ---------------------------------------------------------------------------

def test_not_a_git_repo_is_not_a_refusal(tmp_path):
    (tmp_path / "plain").mkdir()
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, tmp_path / "plain")
    assert reason is None
    assert facts["untracked_count"] is None
    assert facts["stale_autostashes"] is None


# ---------------------------------------------------------------------------
# Refusal text is actionable
# ---------------------------------------------------------------------------

def test_refusal_text_names_the_problem_and_the_override(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    for i in range(26):
        (repo / f"leaked_{i}.py").write_text("x\n")
    reason, facts = update_cmd._audit_checkout_hygiene(GIT, repo)
    assert reason is not None
    update_cmd._print_checkout_hygiene_refusal(reason, facts)
    out = capsys.readouterr().out
    assert "Update refused" in out
    assert "26 untracked files" in out
    assert "--force" in out          # the reviewed override
    assert "max_untracked_files" in out  # the config knob