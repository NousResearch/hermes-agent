"""Tests for credential exclusion during profile export.

Profile exports should NEVER include auth.json or .env — these contain
API keys, OAuth tokens, and credential pool data. Users share exported
profiles; leaking credentials in the archive is a security issue.
"""

import tarfile

from hermes_cli.profiles import audit_profile, export_profile, _DEFAULT_EXPORT_EXCLUDE_ROOT


class TestCredentialExclusion:

    def test_auth_json_in_default_exclude_set(self):
        """auth.json must be in the default export exclusion set."""
        assert "auth.json" in _DEFAULT_EXPORT_EXCLUDE_ROOT

    def test_dotenv_in_default_exclude_set(self):
        """.env must be in the default export exclusion set."""
        assert ".env" in _DEFAULT_EXPORT_EXCLUDE_ROOT

    def test_named_profile_export_excludes_auth(self, tmp_path, monkeypatch):
        """Named profile export must not contain auth.json or .env."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "testprofile"
        profile_dir.mkdir(parents=True)

        # Create a profile with credentials
        (profile_dir / "config.yaml").write_text("model: gpt-4\n")
        (profile_dir / "auth.json").write_text('{"tokens": {"access": "sk-secret"}}')
        (profile_dir / ".env").write_text("OPENROUTER_API_KEY=sk-secret-key\n")
        (profile_dir / "SOUL.md").write_text("I am helpful.\n")
        (profile_dir / "memories").mkdir()
        (profile_dir / "memories" / "MEMORY.md").write_text("# Memories\n")

        monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: profile_dir)
        monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)

        output = tmp_path / "export.tar.gz"
        result = export_profile("testprofile", str(output))

        # Check archive contents
        with tarfile.open(result, "r:gz") as tf:
            names = tf.getnames()

        assert any("config.yaml" in n for n in names), "config.yaml should be in export"
        assert any("SOUL.md" in n for n in names), "SOUL.md should be in export"
        assert not any("auth.json" in n for n in names), "auth.json must NOT be in export"
        assert not any(".env" in n for n in names), ".env must NOT be in export"


class TestProfileAudit:
    def test_default_audit_follows_current_clone_all_contract(self, tmp_path, monkeypatch):
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        (default_home / "config.yaml").write_text("model:\n  default: gpt-4\n")
        (default_home / "logs").mkdir()
        (default_home / "logs" / "gateway.log").write_text("log\n")
        (default_home / "sessions").mkdir()
        (default_home / "sessions" / "old.json").write_text("{}\n")
        (default_home / "node_modules").mkdir()
        (default_home / "node_modules" / "package.json").write_text("{}\n")

        monkeypatch.setattr(
            "hermes_cli.profiles._get_default_hermes_home", lambda: default_home
        )
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: default_home)
        monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda _path: False)

        audit = audit_profile("default", top=20)
        entries = {entry["name"]: entry for entry in audit["top_entries"]}

        assert entries["sessions"]["clone_all_excluded"] is True
        assert entries["node_modules"]["clone_all_excluded"] is True
        assert entries["logs"]["clone_all_excluded"] is False

    def test_named_audit_excludes_history_but_keeps_logs(self, tmp_path, monkeypatch):
        profile_dir = tmp_path / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        (profile_dir / ".env").write_text("SECRET=redacted\n")
        (profile_dir / "logs").mkdir()
        (profile_dir / "logs" / "gateway.log").write_text("log\n")
        (profile_dir / "sessions").mkdir()
        (profile_dir / "sessions" / "old.json").write_text("{}\n")

        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: profile_dir)
        monkeypatch.setattr("hermes_cli.profiles._check_gateway_running", lambda _path: False)

        audit = audit_profile("coder", top=20)
        entries = {entry["name"]: entry for entry in audit["top_entries"]}

        assert entries["sessions"]["clone_all_excluded"] is True
        assert entries["logs"]["clone_all_excluded"] is False
        assert entries[".env"]["export_excluded"] is True
