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

    def test_named_profile_export_recursively_excludes_credential_artifacts(self, tmp_path, monkeypatch):
        """Nested credential stores and their backup artifacts must never enter exports."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "nested"
        profile_dir.mkdir(parents=True)

        ordinary = {
            "config.yaml": "model: gpt-4\n",
            "skills/demo/SKILL.md": "# Useful skill\n",
            "memories/notes/backup-plan.md": "Keep this user-facing backup plan.\n",
        }
        excluded = {
            "skills/demo/private/credentials/provider.json": "secret",
            "workspace/project/.secrets/token.txt": "secret",
            "memories/archive/auth.json.bak": "secret",
            "knowledge/migrations/credentials.tar.gz": "secret",
            "plugins/example/oauth.sqlite3": "secret",
            "scripts/debug/tokens.log": "secret",
        }
        for relpath, content in ordinary.items() | excluded.items():
            path = profile_dir / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)
        result = export_profile("nested", str(tmp_path / "nested.tar.gz"))

        with tarfile.open(result, "r:gz") as tf:
            names = set(tf.getnames())

        archived = {name.removeprefix("nested/") for name in names}
        assert set(ordinary) <= archived
        assert not (set(excluded) & archived)

    def test_credential_names_match_case_insensitively_and_cover_rotations(self, tmp_path, monkeypatch):
        """Case variants, logrotate/backup copies, SQLite sidecars and Hermes' own credential
        stores are excluded; same-named user directories deeper in the tree are not."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "cased"
        profile_dir.mkdir(parents=True)

        ordinary = {
            "config.yaml": "model: gpt-4\n",
            "skills/obsidian/vault/note.md": "# A note in a skill-owned vault\n",
            "skills/demo/pairing/README.md": "# Pairing docs\n",
            "memories/tokenizer.md": "Not a credential.\n",
        }
        excluded = {
            "skills/a/Credentials/provider.json": "secret",
            "workspace/b/.Secrets/key.txt": "secret",
            "skills/c/BOT-DESKTOP/Cookies": "secret",
            "skills/d/Auth.JSON": "secret",
            "skills/e/.ENV": "secret",
            "skills/f/.Env.Production": "secret",
            "scripts/g/tokens.log.1": "secret",
            "scripts/g/tokens.log.gz": "secret",
            "skills/h/secrets.yaml.bak": "secret",
            "plugins/i/token.db-journal": "secret",
            "plugins/i/token.db-wal": "secret",
            "skills/j/credentials.json": "secret",
            "skills/j/oauth.json": "secret",
            "skills/j/token.json": "secret",
            "google_token.json": "secret",
            "google_client_secret.json": "secret",
            ".anthropic_oauth.json": "secret",
            "auth/google_oauth.json": "secret",
            "mcp-tokens/server.json": "secret",
            "vault/vault.key": "secret",
            "browser-profile/Default/Cookies": "secret",
        }
        for relpath, content in ordinary.items() | excluded.items():
            path = profile_dir / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)
        result = export_profile("cased", str(tmp_path / "cased.tar.gz"))

        with tarfile.open(result, "r:gz") as tf:
            archived = {name.removeprefix("cased/") for name in tf.getnames()}

        assert set(ordinary) <= archived
        assert not (set(excluded) & archived), set(excluded) & archived

    def test_env_templates_are_kept_and_redacted(self, tmp_path, monkeypatch):
        """``.env.example`` and friends are documentation: exported, but still scrubbed."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "templates"
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text("model: gpt-4\n")
        templates = [
            ".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults",
            ".env.schema", "prod.env.example",
        ]
        skill_dir = profile_dir / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        for name in templates:
            (skill_dir / name).write_text(f"OPENROUTER_API_KEY={_LEAKED_KEY}\n")
        (skill_dir / ".env.local").write_text("OPENROUTER_API_KEY=live\n")

        _patch_named_profile(monkeypatch, profiles_root, profile_dir)
        result = export_profile("templates", str(tmp_path / "templates.tar.gz"))

        with tarfile.open(result, "r:gz") as tf:
            names = set(tf.getnames())
            for name in templates:
                member = f"templates/skills/demo/{name}"
                assert member in names
                assert _LEAKED_KEY not in tf.extractfile(member).read().decode("utf-8")
        assert "templates/skills/demo/.env.local" not in names


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
