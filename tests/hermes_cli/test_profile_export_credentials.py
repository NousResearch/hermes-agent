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


def _runtime_credential_stores(profile_dir):
    """Credential files created where the runtime puts them for this profile — each owner's own
    resolver, run with the profile's home bound — returned as profile-relative POSIX paths."""
    from pathlib import Path

    from gateway.pairing import PairingStore
    from hermes_cli.browser_connect import chrome_debug_data_dir, real_profile_copy_dir
    from hermes_constants import get_hermes_dir, reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(profile_dir)
    try:
        pairing = PairingStore()
        pairing.approve_code("telegram", pairing.generate_code("telegram", "4242", "alice"))
        written = [Path(chrome_debug_data_dir()) / "Default" / "Cookies",  # `hermes browser connect` CDP profile
                   Path(real_profile_copy_dir("chrome")) / "Default" / "Login Data",
                   get_hermes_dir("platforms/whatsapp/session", "whatsapp/session") / "creds.json",
                   get_hermes_dir("platforms/matrix/store", "matrix/store") / "crypto.db",
                   profile_dir / "browser-profiles" / "live" / "Default" / "Cookies"]  # hermes_cli/backup.py
    finally:
        reset_hermes_home_override(token)
    for path in written:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"api_key: {_LEAKED_KEY}\n", encoding="utf-8")
    approvals = list(pairing._dir.glob("*.json"))
    assert approvals, pairing._dir
    return [path.relative_to(profile_dir).as_posix() for path in [*approvals, *written]]


def test_named_profile_export_drops_every_credential_store(tmp_path, monkeypatch):
    """An export is shareable, so no credential store may ride along: everything the file tools
    read-deny as credentials (agent.file_safety), config backups (byte-exact config.yaml copies
    whose timestamp suffix escapes the redact pass), the 1Password token, and the stores the
    runtime creates for the profile — pairing approvals, browser profiles with Cookies / Login
    Data, messaging sessions. A skill's own ``backups/`` folder is user data and stays."""
    from agent.file_safety import _CREDENTIAL_FILE_NAMES, _READ_DENIED_DIRS

    profiles_root = tmp_path / "profiles"
    profile_dir = profiles_root / "work"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("model: gpt-4\n", encoding="utf-8")
    stores = [*_CREDENTIAL_FILE_NAMES, *(f"{d}/secret.bin" for d, *_ in _READ_DENIED_DIRS),
              "backups/config/config.yaml.good.20260929-083132", ".op.env"]
    for rel in stores:
        path = profile_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"api_key: {_LEAKED_KEY}\n", encoding="utf-8")
    stores += _runtime_credential_stores(profile_dir)
    (profile_dir / "skills" / "demo" / "backups").mkdir(parents=True)
    (profile_dir / "skills" / "demo" / "backups" / "notes.md").write_text("keep\n", encoding="utf-8")
    _patch_named_profile(monkeypatch, profiles_root, profile_dir)

    result = export_profile("work", str(tmp_path / "work.tar.gz"))

    with tarfile.open(result, "r:gz") as tf:
        names = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
        payload = b"".join(tf.extractfile(m).read() for m in tf.getmembers() if m.isfile())
    assert not [rel for rel in stores if rel.replace("\\", "/") in names], names
    assert _LEAKED_KEY.encode() not in payload
    assert "config.yaml" in names and "skills/demo/backups/notes.md" in names
