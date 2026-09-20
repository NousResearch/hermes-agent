"""Regression tests for ``hermes update --branch <tag>`` (tag refs, not branches).

Live incident (2026-09-14, #100243 shape): ``hermes update --branch v2026.9.14
--switch-branch`` refused because origin/v2026.9.14 is a TAG, not a branch ref
(``git fetch origin <name>`` never creates ``origin/<name>`` for tags); the
manual fallback then overshot to main tip for ~35 minutes before correction.

``_fetch_update_target`` (introduced by the fix) resolves the --branch target as
branch OR version-shaped tag before refusing. These tests run against REAL git
repositories (init, commit, tag, clone, fetch) — not mocked subprocess.run — so
they exercise the actual fetch/refspec/rev-parse semantics the helper depends
on, mirroring ``test_update_parked_branch_guard.py``.
"""

import subprocess
from pathlib import Path

import pytest

import hermes_cli.update_cmd as update_cmd


GIT = ["git"]


def _git(cwd, *args, check=True):
    return subprocess.run(
        GIT + list(args), cwd=cwd, capture_output=True, text=True, check=check
    )


@pytest.fixture()
def origin_repo(tmp_path):
    """A real origin repo with commits c1..c3 on main and a release tag ``v9.9.9``
    pointing at c2 (one commit BEHIND main tip) — the incident shape."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "user.name", "Test")
    for i, name in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        (origin / name).write_text(f"c{i}\n")
        _git(origin, "add", name)
        _git(origin, "commit", "-qm", f"c{i}")
    # Release tag at c2: v9.9.9 is one commit behind main tip.
    _git(origin, "tag", "v9.9.9", "HEAD~1")
    return origin


def _clone(tmp_path, origin):
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "remote", "set-url", "origin", str(origin))
    return clone


@pytest.fixture(autouse=True)
def _no_config(monkeypatch):
    """Isolate the update pipeline from the machine's real config.yaml."""
    import hermes_cli.config as hermes_config

    monkeypatch.setattr(hermes_config, "load_config", lambda: {})


def _retarget(monkeypatch, clone):
    """Point update_cmd's PROJECT_ROOT at *clone* and make the update-lock
    machinery a no-op so tests exercise only the git pipeline."""
    import hermes_cli.main as hermes_main

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", clone)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda args: None)


# ---------------------------------------------------------------------------
# _fetch_update_target against real repos
# ---------------------------------------------------------------------------

class TestFetchUpdateTarget:
    def test_version_tag_name_resolves_as_tag(self, origin_repo, tmp_path, monkeypatch):
        clone = _clone(tmp_path, origin_repo)
        _retarget(monkeypatch, clone)
        kind, ref = update_cmd._fetch_update_target(GIT, "v9.9.9")
        assert kind == "tag"
        assert ref == "refs/tags/v9.9.9"
        # The explicit refspec must have materialized the tag in the clone.
        assert _git(clone, "rev-parse", "--verify", "--quiet", "refs/tags/v9.9.9").returncode == 0
        # And the peeled commit must be c2 (the tag target), not main tip c3.
        tagged = _git(clone, "rev-parse", "refs/tags/v9.9.9^{commit}").stdout.strip()
        c2 = _git(origin_repo, "rev-parse", "HEAD~1").stdout.strip()
        assert tagged == c2

    def test_branch_name_still_resolves_as_branch(self, origin_repo, tmp_path, monkeypatch):
        clone = _clone(tmp_path, origin_repo)
        _retarget(monkeypatch, clone)
        kind, ref = update_cmd._fetch_update_target(GIT, "main")
        assert (kind, ref) == ("branch", "origin/main")

    def test_tag_shaped_name_with_no_tag_anywhere_refuses(self, origin_repo, tmp_path, monkeypatch, capsys):
        clone = _clone(tmp_path, origin_repo)
        _retarget(monkeypatch, clone)
        with pytest.raises(SystemExit) as exc:
            update_cmd._fetch_update_target(GIT, "v8.8.8")
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "does not exist locally or on origin" in out
        assert "v8.8.8" in out
        # The improved refusal names both ref kinds for tag-shaped names.
        assert "Release tag or branch" in out

    def test_non_tag_name_missing_everywhere_refuses_plainly(self, origin_repo, tmp_path, monkeypatch, capsys):
        clone = _clone(tmp_path, origin_repo)
        _retarget(monkeypatch, clone)
        with pytest.raises(SystemExit) as exc:
            update_cmd._fetch_update_target(GIT, "no-such-branch")
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Branch 'no-such-branch' does not exist" in out

    def test_non_tag_name_network_failure_keeps_classified_diagnosis(
            self, origin_repo, tmp_path, monkeypatch, capsys):
        """A failed fetch for a NON-tag name must print the classified network
        diagnosis, not 'branch does not exist' (regression: the refactor must
        not swallow fetch failures into the missing-ref message)."""
        clone = _clone(tmp_path, origin_repo)
        _retarget(monkeypatch, clone)
        calls = {"n": 0}

        real_git_run = update_cmd._git_run

        def flaky_network(cmd, args, cwd=None, **kw):
            if "fetch" in args and calls["n"] == 0:
                calls["n"] += 1
                return subprocess.CompletedProcess(
                    cmd + args, 128, stdout="",
                    stderr="fatal: unable to access 'https://github.com/': "
                           "Could not resolve host github.com")
            return real_git_run(cmd, args, cwd=cwd, **kw)

        monkeypatch.setattr(update_cmd, "_git_run", flaky_network)
        with pytest.raises(SystemExit) as exc:
            update_cmd._fetch_update_target(GIT, "main")
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Network error" in out
        assert "does not exist" not in out


# ---------------------------------------------------------------------------
# Tag-mode checkout planning (downgrade / already-at-tag) against real repos
# ---------------------------------------------------------------------------

class TestTagModeCheckoutPlan:
    def _plan(self, monkeypatch, clone, name):
        _retarget(monkeypatch, clone)
        current = update_cmd._current_branch_name(GIT)
        return update_cmd._prepare_checkout_for_update(
            GIT, name, current, is_fork=False, assume_yes=True, gateway_mode=True,
            gw_input_fn=None, switch_branch=True, target_kind="tag",
            fetch_ref=f"refs/tags/{name}")

    def test_downgrade_to_tag_counts_as_update_and_reports_rollback(
            self, origin_repo, tmp_path, monkeypatch):
        """The incident shape: clone at main TIP (overshoot), then move to the tag
        one commit behind. commit_count must be >= 1 (not 'up to date') and the
        plan must carry the rollback count."""
        clone = _clone(tmp_path, origin_repo)
        plan = self._plan(monkeypatch, clone, "v9.9.9")
        assert plan.commit_count >= 1
        assert plan.tag_downgrade_count == 1
        # The detach already landed us ON the tag commit.
        head = _git(clone, "rev-parse", "HEAD").stdout.strip()
        c2 = _git(origin_repo, "rev-parse", "HEAD~1").stdout.strip()
        assert head == c2
        # No local branch may have been created or moved.
        branches = _git(clone, "branch", "--list", "v9.9.9").stdout.strip()
        assert branches == ""

    def test_already_at_tag_is_up_to_date_and_stays_detached(
            self, origin_repo, tmp_path, monkeypatch):
        clone = _clone(tmp_path, origin_repo)
        _git(clone, "fetch", "-q", "origin", "refs/tags/v9.9.9:refs/tags/v9.9.9")
        _git(clone, "checkout", "-q", "--detach", "refs/tags/v9.9.9")
        plan = self._plan(monkeypatch, clone, "v9.9.9")
        assert plan.commit_count == 0
        assert plan.tag_downgrade_count == 0
        assert plan.target_kind == "tag"
        head = _git(clone, "rev-parse", "HEAD").stdout.strip()
        c2 = _git(origin_repo, "rev-parse", "HEAD~1").stdout.strip()
        assert head == c2  # still exactly at the tag

    def test_tag_update_from_parked_branch_leaves_branch_untouched(
            self, origin_repo, tmp_path, monkeypatch, capsys):
        """Tag mode must skip the parked-branch guard entirely: no switch, no
        in-place merge — the parked branch and its commits stay put."""
        clone = _clone(tmp_path, origin_repo)
        _git(clone, "checkout", "-qb", "old-feature")
        (clone / "feature.txt").write_text("wip\n")
        _git(clone, "add", "feature.txt")
        _git(clone, "commit", "-qm", "feature work")
        parked_sha = _git(clone, "rev-parse", "old-feature").stdout.strip()

        plan = self._plan(monkeypatch, clone, "v9.9.9")
        assert plan.commit_count >= 1
        # old-feature still exists, still at its own SHA, and we are NOT on it.
        assert _git(clone, "rev-parse", "old-feature").stdout.strip() == parked_sha
        assert update_cmd._current_branch_name(GIT) == "HEAD"  # detached at the tag
        out = capsys.readouterr().out
        assert "left as-is" in out


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestLooksLikeVersionTag:
    @pytest.mark.parametrize("name,expected", [
        ("v2026.9.14", True),
        ("v2026.9.14.1", True),
        ("v0.21.3", True),
        ("v2026.9", False),
        ("2026.9.14", False),
        ("main", False),
        ("v2026.9.14-rc1", False),
        ("", False),
    ])
    def test_shape(self, name, expected):
        assert update_cmd._looks_like_version_tag(name) is expected
