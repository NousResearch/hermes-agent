"""Tests for credential exclusion + secret scrubbing during profile export.

Profile exports should NEVER include auth.json or .env — these contain
API keys, OAuth tokens, and credential pool data. Users share exported
profiles; leaking credentials in the archive is a security issue.

Secret-shaped strings that sneak into skills / persona / memory text are
force-redacted in the staged archive (same pass as sessions --redact).
The live profile on disk must stay untouched.
"""

import tarfile

import pytest

from hermes_cli.profiles import export_profile

# Long enough to match agent.redact prefix patterns (sk- + 10+ chars).
_LEAKED_KEY = "sk-or-v1-reallyLongSecretKeyValue12345678"


def _patch_named_profile(monkeypatch, profiles_root, profile_dir):
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: profile_dir)
    monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)


class TestCredentialExclusion:

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

    def test_export_ships_no_credential_store_a_loader_reads(self, tmp_path, monkeypatch):
        """Every credential store Hermes loads from a profile home stays out of a named-profile
        export and stays user-owned on a distribution install: a distribution can neither ship
        one (nested ones included) nor opt one back in through ``distribution_owned``. .op.env
        (env_loader) and npmrc (source_build) have no suffix the text scrub pass edits, so
        exclusion is their only guard; the legacy Google Chat token file ships its client_secret."""
        import shutil
        from pathlib import Path

        from hermes_cli.profile_distribution import DistributionManifest, install_distribution, write_manifest
        from hermes_cli.profiles import PROFILE_CREDENTIAL_PATHS

        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "testprofile"
        (profile_dir / "platforms").mkdir(parents=True)
        (profile_dir / "config.yaml").write_text("model: gpt-4\n")
        (profile_dir / "platforms" / "keep.json").write_text("{}")
        stores = {".op.env", "npmrc", "google_chat_user_token.json", "google_chat_user_oauth_pending",
                  "workspace/meetings/node_token.json", *PROFILE_CREDENTIAL_PATHS}
        for rel in stores:
            is_dir = "." not in rel.rsplit("/", 1)[-1] and rel != "npmrc"  # token dirs vs single files
            target = profile_dir / rel / "store" if is_dir else profile_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fake-credential")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / ".hermes").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        tops = sorted({r.split("/")[0] for r in stores})
        for owned in ([], sorted(stores) + tops):  # legacy whole-payload, then explicit opt-in
            staged = tmp_path / f"dist{len(owned)}"
            shutil.copytree(profile_dir, staged)
            write_manifest(staged, DistributionManifest(name=staged.name, version="0.1.0", distribution_owned=owned))
            installed = install_distribution(str(staged), name=staged.name).target_dir
            assert (installed / "platforms" / "keep.json").exists()
            planted = sorted(r for r in stores if any(
                f.is_file() and f.read_text() == "fake-credential"
                for f in ((installed / r), *((installed / r).rglob("*") if (installed / r).is_dir() else ()))))
            assert not planted, (staged.name, planted)

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        with tarfile.open(export_profile("testprofile", str(tmp_path / "export.tar.gz")), "r:gz") as tf:
            names = set(tf.getnames())

        assert {"testprofile/config.yaml", "testprofile/platforms/keep.json"} <= names
        leaked = sorted(r for r in stores if any(n == f"testprofile/{r}" or n.startswith(f"testprofile/{r}/") for n in names))
        assert not leaked, leaked

    def test_export_ships_no_recovery_copy_hermes_writes_of_a_store(self, tmp_path, monkeypatch):
        """Every copy Hermes' own writers leave in a profile home (pre-update zip, update snapshot,
        config-migration .bak, config backup, corrupt auth.json) stays out of a named-profile
        export: each carries the same credentials, and none ends in a suffix the scrub edits."""
        from hermes_cli.auth import _load_auth_store
        from hermes_cli.backup import create_pre_update_backup, create_quick_snapshot
        from hermes_cli.config_backups import backup_config
        from hermes_cli.post_update import _backup_existing

        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "testprofile"
        (profile_dir / "pairing").mkdir(parents=True)
        (profile_dir / "config.yaml").write_text(f"model:\n  api_key: {_LEAKED_KEY}\n")
        (profile_dir / ".env").write_text(f"OPENROUTER_API_KEY={_LEAKED_KEY}\n")
        (profile_dir / "pairing" / "telegram-approved.json").write_text('{"123": {}}')
        (profile_dir / "auth.json").write_text('{"providers": {"x": {"api_key": "' + _LEAKED_KEY + '"')
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))

        copies = [
            create_pre_update_backup(hermes_home=profile_dir),
            profile_dir / "state-snapshots" / create_quick_snapshot(hermes_home=profile_dir),
            backup_config(profile_dir / "config.yaml", "setup"),
            *_backup_existing((profile_dir / ".env", profile_dir / "config.yaml")).values(),
        ]
        _load_auth_store(profile_dir / "auth.json")
        copies.append(profile_dir / "auth.json.corrupt")
        assert all(c and c.exists() for c in copies), copies
        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        with tarfile.open(export_profile("testprofile", str(tmp_path / "export.tar.gz")), "r:gz") as tf:
            names = set(tf.getnames())

        assert "testprofile/config.yaml" in names
        rels = [c.relative_to(profile_dir).as_posix() for c in copies]
        shipped = sorted(r for r in rels if any(n == f"testprofile/{r}" or n.startswith(f"testprofile/{r}/") for n in names))
        assert not shipped, shipped


class TestExportSecretScrub:

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

    @pytest.mark.require_symlinks
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
