"""Tests for credential exclusion + secret scrubbing during profile export.

Profile exports should NEVER include auth.json or .env — these contain
API keys, OAuth tokens, and credential pool data. Users share exported
profiles; leaking credentials in the archive is a security issue.

Secret-shaped strings that sneak into skills / persona / memory text are
force-redacted in the staged archive (same pass as sessions --redact).
The live profile on disk must stay untouched.
"""

import json
import tarfile
from pathlib import Path

import pytest

from hermes_cli.profiles import create_profile, export_profile, get_profile_dir, import_profile

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


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


def _archive_blob(archive: Path) -> bytes:
    with tarfile.open(archive, "r:gz") as tf:
        return b"".join(tf.extractfile(m).read() for m in tf.getmembers() if m.isfile())


class TestExportKeepsStructuredFilesParseable:
    """The export scrub must leave JSON/YAML parseable after an import, and must not ship a
    secret that only its key name reveals when a file needs the value-by-value pass."""

    def test_yaml_config_round_trips_with_secrets_masked(self, profile_env):
        import hermes_yaml

        create_profile("coder", no_alias=True)
        (get_profile_dir("coder") / "config.yaml").write_text(
            "# personal settings\n"
            f"model:\n  default: anthropic/claude-sonnet-4\n  api_key: {_LEAKED_KEY}\n"
            "agent:\n  max_turns: 77\n"
            "dashboard:\n  password: hunter2hunter2\n",
            encoding="utf-8",
        )
        archive = export_profile("coder", str(profile_env / "coder.tar.gz"))
        imported = import_profile(str(archive), name="coder2")

        text = (imported / "config.yaml").read_text(encoding="utf-8")
        config = hermes_yaml.safe_load(text)
        assert config["agent"]["max_turns"] == 77
        assert config["model"]["default"] == "anthropic/claude-sonnet-4"
        assert "# personal settings" in text
        blob = _archive_blob(archive)
        assert _LEAKED_KEY.encode() not in blob and b"hunter2hunter2" not in blob

    def test_json_and_jsonl_round_trip_with_secrets_masked(self, profile_env):
        create_profile("coder", no_alias=True)
        source = get_profile_dir("coder")
        (source / "cron").mkdir(exist_ok=True)
        jobs = {"jobs": [
            {"id": "j1", "prompt": f"Summarise open PRs with OPENROUTER_API_KEY={_LEAKED_KEY}",
             "token": "OpaqueValue123456xyz"},
            {"id": "j2", "prompt": "Post the standup notes"},
        ]}
        (source / "cron" / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        (source / "sessions" / "s1.jsonl").write_text(
            json.dumps({"role": "tool", "content": f"env\nMY_API_KEY={_LEAKED_KEY}"}) + "\n"
            + json.dumps({"role": "assistant", "content": "done"}) + "\n",
            encoding="utf-8",
        )
        archive = export_profile("coder", str(profile_env / "coder.tar.gz"))
        imported = import_profile(str(archive), name="coder2")

        restored = json.loads((imported / "cron" / "jobs.json").read_text(encoding="utf-8"))
        assert [job["id"] for job in restored["jobs"]] == ["j1", "j2"]
        assert restored["jobs"][0]["prompt"].startswith("Summarise open PRs with OPENROUTER_API_KEY=")
        lines = (imported / "sessions" / "s1.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["role"] for line in lines] == ["tool", "assistant"]
        blob = _archive_blob(archive)
        assert _LEAKED_KEY.encode() not in blob and b"OpaqueValue123456xyz" not in blob
