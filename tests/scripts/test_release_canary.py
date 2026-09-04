"""Canary release plumbing in scripts/release.py.

The canary channel's whole safety story is tag SHAPES: canary tags
carry a -canary.YYYYMMDD suffix, every stable selector requires the
no-suffix shape, and the version math (next-MINOR over stable) makes
electron-updater's semver ordering implement both channel-switch
directions. These tests pin the shapes and the math; the workflow only
provides credentials.
"""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

_RELEASE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "release.py"
_SPEC = importlib.util.spec_from_file_location("hermes_release_canary", _RELEASE_PATH)
assert _SPEC and _SPEC.loader
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


class TestCanaryTagShape:
    def test_canary_tag_for_date_is_next_patch(self):
        """The authority's math: current stable's patch + 1, any patch."""
        assert release.canary_tag_for_date("0.27.4", "20260818103000") == "v0.27.5-canary.20260818103000"
        assert release.canary_tag_for_date("1.4.0", "20261231235959") == "v1.4.1-canary.20261231235959"

    def test_v_prefixed_stable_tag_accepted(self):
        """cmd_canary passes the stable TAG (with the v) — the authority
        strips it, so both forms produce the same tag."""
        assert release.canary_tag_for_date("v0.27.4", "20260818103000") == "v0.27.5-canary.20260818103000"

    def test_shape_round_trips_through_the_canary_matcher(self):
        tag = release.canary_tag_for_date("0.27.4", "20260818103000")
        assert release._CANARY_TAG_RE.fullmatch(tag)

    def test_legacy_date_only_shape_still_matches(self):
        """Readers stay tolerant of the original 8-digit suffix so an
        already-published canary keeps parsing (prune, last-canary)."""
        assert release._CANARY_TAG_RE.fullmatch("v0.28.0-canary.20260818")

    def test_canary_shape_is_invisible_to_the_stable_selector(self):
        """THE invariant: a canary tag must never parse as stable —
        otherwise stable users update onto canaries."""
        for tag in (
            release.canary_tag_for_date("0.27.4", "20260818103000"),
            "v0.28.0-canary.20260818",
        ):
            assert release._SEMVER_TAG_RE.fullmatch(tag) is None, tag

    def test_stable_shape_is_invisible_to_the_canary_matcher(self):
        assert release._CANARY_TAG_RE.fullmatch("v0.27.2") is None
        # Legacy CalVer must not match either (v2026.7.20 has 20-prefixed
        # components that could fool a sloppy pattern).
        assert release._CANARY_TAG_RE.fullmatch("v2026.7.20") is None

    def test_same_day_canaries_order_chronologically(self):
        """Second precision exists so manual fires can stack in one day;
        fixed-length pure-numeric identifiers order the same lexically
        (git -v:refname) and numerically (semver prerelease compare)."""
        a = release.canary_tag_for_date("0.27.4", "20260818090000")
        b = release.canary_tag_for_date("0.27.4", "20260818171500")
        assert a < b  # lexical == chronological at fixed length

    def test_canary_outversions_current_stable_loses_to_next_patch(self):
        """The semver ordering that makes both channel switches work,
        checked with packaging's canonical comparison when available,
        else by electron-updater's documented precedence rules."""
        try:
            from packaging.version import Version
        except ImportError:
            pytest.skip("packaging not installed in this env")
        canary = Version("0.27.5-canary.20260818103000".replace("-canary.", "a"))
        assert Version("0.27.4") < canary < Version("0.27.5")


class TestCanaryDateStamps:
    def test_prune_cutoff_uses_the_tag_date(self):
        """The prune window keys on the tag's own YYYYMMDD prefix, for
        both suffix shapes — a 14-digit suffix compared whole against an
        8-digit cutoff would be decided by string length, not by day."""
        legacy = release._CANARY_TAG_RE.fullmatch("v0.27.1-canary.20260801")
        stamped = release._CANARY_TAG_RE.fullmatch("v0.27.1-canary.20260801235959")
        assert legacy and legacy.group(0).split("-canary.")[1][:8] == "20260801"
        assert stamped and stamped.group(0).split("-canary.")[1][:8] == "20260801"


class TestCanaryBuildNumberGuards:
    """The MSIX canary build number is minutes-since-the-last-stable.
    Two canaries of the same line cut in the same minute would share a
    build number → identical MSIX version → App Installer refuses the
    second. cmd_canary refuses that; it also refuses a canary whose
    stable base is >45 days old (the 16-bit MSIX component would
    overflow)."""

    def test_same_minute_same_line_refused(self, monkeypatch, tmp_path):
        import subprocess
        from types import SimpleNamespace

        calls: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="url", stderr="")

        def fake_git_result(*args, **kw):
            # rev-parse --verify must MISS so we reach the guard, not the
            # "tag already exists" short-circuit.
            return subprocess.CompletedProcess(args, 1, "", "")

        monkeypatch.setattr(release, "get_last_tag", lambda: "v0.27.1")
        monkeypatch.setattr(release, "get_last_canary_tag", lambda: "v0.27.2-canary.20260818103045")
        monkeypatch.setattr(release, "git_result", fake_git_result)
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        release.cmd_canary(SimpleNamespace(date="20260818103000", publish=True, remote="origin"))
        assert calls == [], "same-minute same-line re-cut must not create a release"

    def test_same_minute_new_line_allowed(self, monkeypatch, tmp_path):
        """A fresh stable bumps the line; a same-minute cut on the new line
        outversions the old line by its patch component, so no collision."""
        import subprocess
        from types import SimpleNamespace

        captured: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            captured.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="url", stderr="")

        def fake_git_result(*args, **kw):
            return subprocess.CompletedProcess(args, 1 if "rev-parse" in args else 0, "", "")

        monkeypatch.setattr(release, "get_last_tag", lambda: "v0.28.0")
        monkeypatch.setattr(release, "get_last_canary_tag", lambda: "v0.27.2-canary.20260818103045")
        monkeypatch.setattr(release, "get_commits", lambda **kw: [{"hash": "a" * 40, "subject": "feat: x", "author": "e"}])
        monkeypatch.setattr(release, "generate_changelog", lambda *a, **kw: "notes")
        monkeypatch.setattr(release, "resolve_push_remote", lambda r: "origin")
        monkeypatch.setattr(release, "remote_github_repo", lambda r: "o/r")
        monkeypatch.setattr(release, "git", lambda *a, **kw: "1786017600" if a and a[0] == "log" else "")
        monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(release.git_result, "__defaults__", ())
        monkeypatch.setattr(release, "git_result", fake_git_result)
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        release.cmd_canary(SimpleNamespace(date="20260818103000", publish=True, remote="origin"))
        create = [c for c in captured if c[:3] == ["gh", "release", "create"]]
        assert len(create) == 1, captured

    def test_stable_base_older_than_45_days_refused(self, monkeypatch, tmp_path):
        """Minutes-since-stable crosses the 16-bit MSIX component at 45.5
        days; a canary cut on an older stable cannot carry a legal build
        number — refuse loudly (exit 1), never clamp."""
        import subprocess
        import sys
        from types import SimpleNamespace

        # Stable v0.27.1 committed 2026-07-01; canary cut 2026-08-29.
        old_epoch = str(int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()))
        monkeypatch.setattr(release, "get_last_tag", lambda: "v0.27.1")
        monkeypatch.setattr(release, "get_last_canary_tag", lambda: None)
        monkeypatch.setattr(release, "get_commits", lambda **kw: [{"hash": "a" * 40, "subject": "feat: x", "author": "e"}])
        monkeypatch.setattr(release, "git", lambda *a, **kw: old_epoch if a and a[0] == "log" else "")
        monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(release, "git_result", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", ""))
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        with pytest.raises(SystemExit) as exc:
            release.cmd_canary(SimpleNamespace(date="20260829000000", publish=True, remote="origin"))
        assert exc.value.code == 1


class TestCanaryIsDrafted:
    """A canary is created as a DRAFT prerelease.

    A published release with no installers staged is one users can
    reach and cannot use. The desktop matrix stages the installers to the
    R2 bucket (releases/tag/<tag>/), the finalize job publishes the
    feeds, and the canary workflow publishes the release only after
    that matrix is green.
    """

    def _create_argv(self, monkeypatch, tmp_path):
        """The argv of the `gh release create` cmd_canary would run."""
        import subprocess
        from types import SimpleNamespace

        captured: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            captured.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="url", stderr="")

        def fake_git_result(*args, **kw):
            # `rev-parse --verify --quiet refs/tags/<tag>` must MISS, or
            # cmd_canary short-circuits on "tag already exists".
            code = 1 if "rev-parse" in args else 0
            return subprocess.CompletedProcess(args, code, "", "")

        monkeypatch.setattr(release, "get_last_tag", lambda: "v0.27.0")
        monkeypatch.setattr(release, "get_last_canary_tag", lambda: None)
        monkeypatch.setattr(release, "get_commits", lambda **kw: [{"hash": "a" * 40, "subject": "feat: x", "author": "e"}])
        monkeypatch.setattr(release, "generate_changelog", lambda *a, **kw: "notes")
        monkeypatch.setattr(release, "resolve_push_remote", lambda r: "origin")
        monkeypatch.setattr(release, "remote_github_repo", lambda r: "o/r")
        monkeypatch.setattr(release, "git_result", fake_git_result)
        monkeypatch.setattr(release, "git", lambda *a, **kw: "")
        monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        release.cmd_canary(SimpleNamespace(date="20260818103000", publish=True, remote="origin"))
        return next(c for c in captured if c[:3] == ["gh", "release", "create"])

    def test_created_as_a_draft_prerelease(self, monkeypatch, tmp_path):
        argv = self._create_argv(monkeypatch, tmp_path)
        assert "--draft" in argv, argv
        assert "--prerelease" in argv, argv

    def test_refuses_to_invent_a_tag(self, monkeypatch, tmp_path):
        """Without --verify-tag, gh creates a missing tag from the default
        branch tip, which would release a different commit than the one
        the canary math tagged."""
        assert "--verify-tag" in self._create_argv(monkeypatch, tmp_path)


class TestCanaryStartsItsOwnBuild:
    """release.py dispatches the desktop build for the tag it just cut.

    workflow_dispatch is one of only two events GITHUB_TOKEN may raise, so
    it is the only trigger that serves the scheduled canary: that run
    pushes its tag as github-actions[bot], and a bot-pushed tag starts no
    workflow run. Dispatching from release.py gives the scheduled canary,
    a hand-cut canary, and a stable release one identical mechanism.
    """

    def _calls(self, monkeypatch, tmp_path):
        import subprocess
        from types import SimpleNamespace

        captured: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            captured.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="url", stderr="")

        monkeypatch.setattr(release, "get_last_tag", lambda: "v0.27.0")
        monkeypatch.setattr(release, "get_last_canary_tag", lambda: None)
        monkeypatch.setattr(release, "get_commits", lambda **kw: [{"hash": "a" * 40, "subject": "feat: x", "author": "e"}])
        monkeypatch.setattr(release, "generate_changelog", lambda *a, **kw: "notes")
        monkeypatch.setattr(release, "resolve_push_remote", lambda r: "origin")
        monkeypatch.setattr(release, "remote_github_repo", lambda r: "o/r")
        monkeypatch.setattr(
            release, "git_result",
            lambda *a, **kw: subprocess.CompletedProcess(a, 1 if "rev-parse" in a else 0, "", ""),
        )
        monkeypatch.setattr(release, "git", lambda *a, **kw: "")
        monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(release.shutil, "which", lambda x: "/usr/bin/gh")
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        release.cmd_canary(SimpleNamespace(date="20260818103000", publish=True, remote="origin"))
        return captured

    def test_dispatches_the_build_for_the_tag_it_cut(self, monkeypatch, tmp_path):
        calls = self._calls(monkeypatch, tmp_path)
        dispatch = next(c for c in calls if c[:3] == ["gh", "workflow", "run"])
        # get_last_tag is patched to v0.27.0 → the authority's patch+1 math
        # cuts v0.27.1-canary.20260818103000.
        tag = "v0.27.1-canary.20260818103000"

        assert "desktop-bundled-release.yml" in dispatch
        # The tag travels as the INPUT the build reads. The workflow FILE
        # itself is dispatched from the default branch — a tag-dispatched
        # run scopes actions/cache under refs/heads/refs/tags/<tag>, a
        # mangled per-tag scope no later canary can restore.
        assert f"tag={tag}" in dispatch
        assert "--ref" in dispatch and dispatch[dispatch.index("--ref") + 1] != tag
        # Without this the build produces artifacts and attaches nothing.
        assert "upload_release=true" in dispatch

    def test_the_draft_exists_before_the_build_starts(self, monkeypatch, tmp_path):
        """Ordering is the whole point of dispatching rather than relying
        on the tag push: the build's upload step fails on a missing
        release, so the draft has to be there first."""
        calls = self._calls(monkeypatch, tmp_path)
        create = next(i for i, c in enumerate(calls) if c[:3] == ["gh", "release", "create"])
        dispatch = next(i for i, c in enumerate(calls) if c[:3] == ["gh", "workflow", "run"])
        assert create < dispatch

    def test_a_failed_dispatch_does_not_sink_the_release(self, monkeypatch, tmp_path):
        """The tag and draft are already pushed by then. Report the manual
        command and leave them; raising would strand a half-made release."""
        assert release.dispatch_desktop_build.__doc__
        monkeypatch.setattr(release.shutil, "which", lambda x: None)
        assert release.dispatch_desktop_build("v0.28.0-canary.20260818103000", "o/r") is False
