"""Tests for credential exclusion + secret scrubbing during profile export.

Profile exports should NEVER include auth.json or .env — these contain
API keys, OAuth tokens, and credential pool data. Users share exported
profiles; leaking credentials in the archive is a security issue.

Secret-shaped strings that sneak into skills / persona / memory text are
force-redacted in the staged archive (same pass as sessions --redact).
The live profile on disk must stay untouched.
"""

import sys
import tarfile

import pytest
import yaml

from hermes_cli.profiles import export_profile, _DEFAULT_EXPORT_EXCLUDE_ROOT

# Long enough to match agent.redact prefix patterns (sk- + 10+ chars).
_LEAKED_KEY = "sk-or-v1-reallyLongSecretKeyValue12345678"


def _patch_named_profile(monkeypatch, profiles_root, profile_dir):
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: profile_dir)
    monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)


class TestCredentialExclusion:

    def test_auth_json_in_default_exclude_set(self):
        """auth.json must be in the default export exclusion set."""
        assert "auth.json" in _DEFAULT_EXPORT_EXCLUDE_ROOT


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

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        output = tmp_path / "export.tar.gz"
        result = export_profile("testprofile", str(output))

        # Check archive contents
        with tarfile.open(result, "r:gz") as tf:
            names = tf.getnames()

        assert any("config.yaml" in n for n in names), "config.yaml should be in export"
        assert any("SOUL.md" in n for n in names), "SOUL.md should be in export"
        assert not any("auth.json" in n for n in names), "auth.json must NOT be in export"
        assert not any(".env" in n for n in names), ".env must NOT be in export"


class TestExportSecretScrub:

    def test_named_profile_export_clears_structural_config_credentials(
        self, tmp_path, monkeypatch
    ):
        """Opaque config credentials are cleared even without vendor prefixes."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "scrubconfig"
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "providers": {"ollama": {"api_key": "opaque-api-value"}},
                    "dashboard": {
                        "basic_auth": {
                            "username": "operator",
                            "password_hash": "opaque-password-hash",
                            "secret": "opaque-session-secret",
                            "session_ttl_seconds": 43200,
                        }
                    },
                },
                sort_keys=False,
            )
        )

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        result = export_profile(
            "scrubconfig", str(tmp_path / "scrubconfig.tar.gz")
        )

        with tarfile.open(result, "r:gz") as tf:
            config_name = next(
                name for name in tf.getnames() if name.endswith("/config.yaml")
            )
            exported = yaml.safe_load(
                tf.extractfile(config_name).read().decode("utf-8")
            )

        assert exported["providers"]["ollama"]["api_key"] == ""
        basic = exported["dashboard"]["basic_auth"]
        assert basic["password_hash"] == ""
        assert basic["secret"] == ""
        assert basic["username"] == "operator"
        assert basic["session_ttl_seconds"] == 43200

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_structural_config_scrub_materializes_symlink_without_touching_source(
        self, tmp_path, monkeypatch
    ):
        """A symlinked config is scrubbed in staging, never through its target."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "linkconfig"
        profile_dir.mkdir(parents=True)
        outside = tmp_path / "outside-config.yaml"
        outside.write_text(
            yaml.safe_dump({"dashboard": {"basic_auth": {"secret": "opaque"}}})
        )
        (profile_dir / "config.yaml").symlink_to(outside)

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        result = export_profile(
            "linkconfig",
            str(tmp_path / "linkconfig.tar.gz"),
            extra_files={
                "config.yaml": yaml.safe_dump(
                    {"dashboard": {"basic_auth": {"secret": "overlay-secret"}}}
                )
            },
        )

        with tarfile.open(result, "r:gz") as tf:
            config_name = next(
                name for name in tf.getnames() if name.endswith("/config.yaml")
            )
            member = tf.getmember(config_name)
            exported = yaml.safe_load(
                tf.extractfile(member).read().decode("utf-8")
            )

        assert not member.issym()
        assert exported["dashboard"]["basic_auth"]["secret"] == ""
        assert yaml.safe_load(outside.read_text())["dashboard"]["basic_auth"]["secret"] == "opaque"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_extra_files_reject_symlinked_parent_escape(self, tmp_path, monkeypatch):
        """An overlay cannot write through a staged directory symlink."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "linkparent"
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text("model: gpt-4\n")
        outside = tmp_path / "outside"
        outside.mkdir()
        (profile_dir / "overlay").symlink_to(outside, target_is_directory=True)

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        with pytest.raises(ValueError, match="outside the staged profile"):
            export_profile(
                "linkparent",
                str(tmp_path / "linkparent.tar.gz"),
                extra_files={"overlay/injected.txt": "must not escape"},
            )

        assert not (outside / "injected.txt").exists()

    def test_named_profile_export_redacts_secrets_in_text(self, tmp_path, monkeypatch):
        """Leaked keys in skills / SOUL / memories must not leave the archive."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "scrubme"
        profile_dir.mkdir(parents=True)

        soul = profile_dir / "SOUL.md"
        soul.write_text(f"My key is {_LEAKED_KEY}\n")

        skill_dir = profile_dir / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        skill = skill_dir / "SKILL.md"
        skill.write_text(
            "---\nname: demo\ndescription: Demo.\n---\n"
            f"Use OPENROUTER_API_KEY={_LEAKED_KEY}\n"
        )

        memories = profile_dir / "memories"
        memories.mkdir()
        memory = memories / "MEMORY.md"
        memory.write_text(f"token {_LEAKED_KEY}\n")

        (profile_dir / "config.yaml").write_text("model: gpt-4\n")

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        result = export_profile("scrubme", str(tmp_path / "scrubme.tar.gz"))

        with tarfile.open(result, "r:gz") as tf:
            members = {
                name: tf.extractfile(name).read().decode("utf-8")
                for name in tf.getnames()
                if name.endswith((".md", ".yaml"))
            }

        blob = "\n".join(members.values())
        assert _LEAKED_KEY not in blob
        assert any("SOUL.md" in n for n in members)
        assert any("SKILL.md" in n for n in members)
        assert any("MEMORY.md" in n for n in members)

        # Live profile must keep the original plaintext.
        assert _LEAKED_KEY in soul.read_text()
        assert _LEAKED_KEY in skill.read_text()
        assert _LEAKED_KEY in memory.read_text()

    def test_export_redacts_through_symlink_without_touching_source(
        self, tmp_path, monkeypatch
    ):
        """Symlinked skill text is redacted in the archive, source file stays put."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "linkme"
        profile_dir.mkdir(parents=True)

        outside = tmp_path / "outside-skill.md"
        outside.write_text(f"secret {_LEAKED_KEY}\n")

        skill_dir = profile_dir / "skills" / "linked"
        skill_dir.mkdir(parents=True)
        link = skill_dir / "SKILL.md"
        link.symlink_to(outside)

        (profile_dir / "config.yaml").write_text("model: gpt-4\n")
        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        result = export_profile("linkme", str(tmp_path / "linkme.tar.gz"))

        with tarfile.open(result, "r:gz") as tf:
            skill_members = [n for n in tf.getnames() if n.endswith("SKILL.md")]
            assert skill_members
            archived = tf.extractfile(skill_members[0]).read().decode("utf-8")

        assert _LEAKED_KEY not in archived
        assert _LEAKED_KEY in outside.read_text()
        assert link.is_symlink()
