"""Tests for hermes_cli.authority — profile-based permission layer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure hermes-agent is on sys.path
HERMES_AGENT = Path(r"C:\Users\bbask\AppData\Local\hermes\hermes-agent")
if str(HERMES_AGENT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT))

from hermes_cli import authority


# ─── Profile resolution ─────────────────────────────────────────────────────


def test_get_profile_defaults_to_production_when_unset(monkeypatch, tmp_path):
    """When config.yaml has no authority_profile key, default is production."""
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("model: foo\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.get_profile() == "production"


def test_get_profile_returns_production_when_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: production\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.get_profile() == "production"


def test_get_profile_returns_principal_when_set(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.get_profile() == "principal"


def test_get_profile_falls_back_to_production_for_garbage_value(monkeypatch, tmp_path):
    """An unknown value (typo, etc.) is treated as production — fail closed."""
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: superadmin\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.get_profile() == "production"


def test_get_profile_returns_production_when_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "nope.yaml")
    authority.reset_cache()
    assert authority.get_profile() == "production"


def test_get_profile_caches_for_5_seconds(monkeypatch, tmp_path):
    """The TTL avoids a disk read on every allows() call but stays fresh within a turn."""
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: production\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.get_profile() == "production"
    # Flip the file
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    # Within TTL: still cached
    assert authority.get_profile() == "production"
    # After reset: picks up new value
    authority.reset_cache()
    assert authority.get_profile() == "principal"


# ─── Production profile permissions ──────────────────────────────────────────


def test_production_blocks_edit_config_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: production\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("edit_config_yaml") is False


def test_production_blocks_pip_install(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: production\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("pip_install_venv") is False


def test_production_blocks_skill_create(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: production\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("skill_create") is False


# ─── Principal profile permissions ──────────────────────────────────────────


def test_principal_allows_edit_config_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("edit_config_yaml") is True


def test_principal_allows_pip_install(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("pip_install_venv") is True


def test_principal_allows_skill_create(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("skill_create") is True


def test_principal_allows_git_clone(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("git_clone_tools") is True


# ─── Absolute denies (both profiles) ─────────────────────────────────────────


def test_principal_blocks_commit_cookie_json(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("commit_cookie_json") is False


def test_production_blocks_commit_cookie_json(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: production\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("commit_cookie_json") is False


def test_principal_blocks_modify_hermes_agent_upstream(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("modify_hermes_agent_upstream_checkout") is False


def test_principal_blocks_bypass_approvals_deny(monkeypatch, tmp_path):
    """Even under principal, the smart-approval gate and approvals.deny
    globs cannot be bypassed. This is the prompt-injection floor."""
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("bypass_approvals_deny") is False
    assert authority.allows("disable_smart_approval") is False


def test_principal_blocks_edit_other_repo_hermes_md(monkeypatch, tmp_path):
    """Cross-repo .hermes.md files are out of scope for Austin's personal
    workspace profile — those repos have their own governance."""
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    assert authority.allows("edit_other_repo_hermes_md") is False


# ─── deny_reason and format_constraints ───────────────────────────────────────


def test_deny_reason_for_absolute_deny(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    msg = authority.deny_reason("commit_cookie_json")
    assert "absolute deny" in msg.lower()


def test_deny_reason_for_production_action(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: production\n", encoding="utf-8")
    authority.reset_cache()
    msg = authority.deny_reason("edit_config_yaml")
    assert "production" in msg
    assert "principal" in msg  # mentions how to unlock


def test_format_constraints_describes_active_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()
    text = authority.format_constraints()
    assert "principal" in text.lower()


# ─── log_action is non-blocking and non-failing ──────────────────────────────


def test_log_action_writes_a_line(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("authority_profile: principal\n", encoding="utf-8")
    authority.reset_cache()

    authority.log_action("pip_install_venv", target="ruff")

    log_dir = tmp_path / "cache" / "authority-actions"
    assert log_dir.exists()
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) == 1
    content = log_files[0].read_text(encoding="utf-8")
    assert "pip_install_venv" in content
    assert "ruff" in content


def test_log_action_silently_swallows_filesystem_errors(monkeypatch, tmp_path):
    """Logging must never block the action. Point HERMES_HOME at a path we
    can't write to."""
    # Read-only parent — we'll write logs into a child that can't be created
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o444)
    monkeypatch.setattr(authority, "HERMES_HOME", readonly)
    # Should not raise
    authority.log_action("pip_install_venv", target="ruff")



class TestSensitivePathGuardIntegration:
    """Integration: tools.file_tools_write_guards._check_sensitive_path
    must consult hermes_cli.authority.allows() — production hard-denies
    config.yaml (preserves upstream test contract), principal allows it
    (the whole point of the profile system).
    """

    def test_production_check_sensitive_path_denies_config_yaml(
        self, monkeypatch, tmp_path: Path
    ):
        """Production profile: _check_sensitive_path still returns the
        deny message verbatim (existing test contract)."""
        monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
        (tmp_path / "config.yaml").write_text("authority_profile: production\n",
                                            encoding="utf-8")
        # Make the config-resolver resolve to this tmp config.yaml
        monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
        authority.reset_cache()

        # Re-import the guard under the test's HERMES_HOME scope
        from tools.file_tools_write_guards import _check_sensitive_path
        import tools.file_tools_write_guards as g
        monkeypatch.setattr(g, "_hermes_config_resolved",
                            str((tmp_path / "config.yaml").resolve()))
        monkeypatch.setattr(g, "_hermes_config_resolved_loaded", True)

        err = _check_sensitive_path(str(tmp_path / "config.yaml"))
        assert err is not None
        assert "Refusing to write to Hermes config file" in err

    def test_principal_check_sensitive_path_allows_config_yaml(
        self, monkeypatch, tmp_path: Path
    ):
        """Principal profile: _check_sensitive_path returns None
        (allow) and writes an audit log entry."""
        monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
        (tmp_path / "config.yaml").write_text("authority_profile: principal\n",
                                            encoding="utf-8")
        monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
        authority.reset_cache()

        from tools.file_tools_write_guards import _check_sensitive_path
        import tools.file_tools_write_guards as g
        monkeypatch.setattr(g, "_hermes_config_resolved",
                            str((tmp_path / "config.yaml").resolve()))
        monkeypatch.setattr(g, "_hermes_config_resolved_loaded", True)

        # Pre-condition: profile should resolve to principal
        assert authority.get_profile() == "principal"

        err = _check_sensitive_path(str(tmp_path / "config.yaml"))
        assert err is None, f"expected allow, got: {err!r}"

        # Audit log should have an entry
        log_files = list((tmp_path / "cache" / "authority-actions").glob("*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text(encoding="utf-8")
        assert "edit_config_yaml" in content
        assert "profile=principal" in content



class TestBootstrapSession:
    """bootstrap_session() — install-deps path under the profile system.

    The function is the bridge between authority and the installer: it
    consults ``allows()`` per backend, respects ``dry_run`` semantics,
    and never invokes a real installer in tests (we mock _run_installer).
    """

    def test_production_dry_run_reports_skipped_production(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
        (tmp_path / "config.yaml").write_text(
            "authority_profile: production\n", encoding="utf-8"
        )
        monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
        authority.reset_cache()

        called = []
        monkeypatch.setattr(
            authority, "_run_installer",
            lambda b, s: called.append((b, s))
        )

        result = authority.bootstrap_session(
            packages=[("pip", "ruff"), ("npm", "typescript")],
            dry_run=True,
        )
        assert result == {
            "pip:ruff": "skipped_production",
            "npm:typescript": "skipped_production",
        }
        assert called == []  # production + dry_run → no actual installer call

    def test_principal_dry_run_reports_would_install(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
        (tmp_path / "config.yaml").write_text(
            "authority_profile: principal\n", encoding="utf-8"
        )
        monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
        authority.reset_cache()

        called = []
        monkeypatch.setattr(
            authority, "_run_installer",
            lambda b, s: called.append((b, s))
        )

        result = authority.bootstrap_session(
            packages=[("pip", "ruff")],
            dry_run=True,
        )
        assert result == {"pip:ruff": "would_install"}
        assert called == []  # dry_run → no installer call

    def test_principal_real_install_invokes_runner_and_logs(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
        (tmp_path / "config.yaml").write_text(
            "authority_profile: principal\n", encoding="utf-8"
        )
        monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
        authority.reset_cache()

        called = []
        monkeypatch.setattr(
            authority, "_run_installer",
            lambda b, s: called.append((b, s))
        )

        result = authority.bootstrap_session(
            packages=[("pip", "ruff"), ("npm", "typescript")],
            dry_run=False,
        )
        assert result == {"pip:ruff": "installed", "npm:typescript": "installed"}
        assert called == [("pip", "ruff"), ("npm", "typescript")]

        # Audit log should have one entry per install
        log_files = list((tmp_path / "cache" / "authority-actions").glob("*.log"))
        assert len(log_files) == 1
        log_content = log_files[0].read_text(encoding="utf-8")
        assert "pip_install_venv" in log_content
        assert "npm_install_global" in log_content
        assert "pip:ruff" in log_content

    def test_principal_real_install_failure_marks_failed(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.setattr(authority, "HERMES_HOME", tmp_path)
        (tmp_path / "config.yaml").write_text(
            "authority_profile: principal\n", encoding="utf-8"
        )
        monkeypatch.setattr(authority, "CONFIG_PATH", tmp_path / "config.yaml")
        authority.reset_cache()

        def boom(b, s):
            raise RuntimeError("disk full")
        monkeypatch.setattr(authority, "_run_installer", boom)

        result = authority.bootstrap_session(
            packages=[("pip", "ruff")], dry_run=False,
        )
        assert result == {"pip:ruff": "failed: disk full"}

    def test_default_packages_are_pip_only_safe(self):
        """The DEFAULT_SESSION_PACKAGES list must only use backend keys that
        exist in ``_action_for_backend()`` — otherwise bootstrap silently
        falls through to download_binary which isn't an actual installer."""
        backends = {b for b, _ in authority.DEFAULT_SESSION_PACKAGES}
        for backend in backends:
            action = authority._action_for_backend(backend)
            assert action in authority.PRINCIPAL_ALLOWS, (
                f"DEFAULT_SESSION_PACKAGES uses {backend!r} which maps to "
                f"{action!r} but that action isn't in PRINCIPAL_ALLOWS"
            )

    def test_action_for_backend_maps_each_known_backend(self):
        """Smoke test the backends-to-actions mapping table."""
        assert authority._action_for_backend("pip")   == "pip_install_venv"
        assert authority._action_for_backend("pipx")  == "pipx_install"
        assert authority._action_for_backend("npm")   == "npm_install_global"
        assert authority._action_for_backend("winget")== "winget_install_user"
        assert authority._action_for_backend("git")   == "git_clone_tools"
        assert authority._action_for_backend("unknown") == "download_binary"

    def test_run_installer_uses_argv_not_shell(self, monkeypatch, tmp_path: Path):
        """The subprocess call must use a list of argv tokens — never
        interpolated into a shell string — to prevent shell injection
        via the spec parameter."""
        captured = []
        import subprocess as _sp

        def fake_check_call(argv):
            captured.append(argv)
            return 0

        monkeypatch.setattr(_sp, "check_call", fake_check_call)

        # pip backend, malicious spec
        authority._run_installer("pip", "/tmp/x; rm -rf /; #")
        assert len(captured) == 1
        argv = captured[0]
        assert isinstance(argv, list)
        assert "/tmp/x; rm -rf /; #" in argv  # present as a SINGLE token
        # No shell=True, no interpolated string
        assert all(isinstance(x, str) for x in argv)
        # git backend
        authority._run_installer("git", "https://github.com/x/y.git")
        assert captured[1][:1] == ["git"]
        assert "https://github.com/x/y.git" in captured[1]
        # Unknown backend raises
        import pytest
        with pytest.raises(ValueError, match="unknown backend"):
            authority._run_installer("curl", "https://example.com")
