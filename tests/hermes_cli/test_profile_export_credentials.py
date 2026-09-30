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
from plugins.teams_pipeline.store import DEFAULT_TEAMS_PIPELINE_STORE_FILENAME

# Long enough to match agent.redact prefix patterns (sk- + 10+ chars).
_LEAKED_KEY = "sk-or-v1-reallyLongSecretKeyValue12345678"


def _patch_named_profile(monkeypatch, profiles_root, profile_dir):
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: profile_dir)
    monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)


# Stores named here as well as in PROFILE_CREDENTIAL_PATHS, so dropping one from the list fails a test:
# the loaders' and writers' own file names (.op.env and npmrc have no suffix the scrub edits).
_EXTRA_STORES = {
    ".op.env", "npmrc", "honcho.json", "google_chat_user_token.json", "google_chat_user_oauth_pending",
    "workspace/meetings/node_token.json", "weixin/accounts", ".copilot_jwt.json", "proxy", "chrome-debug",
    "home/.git-credentials", "home/.config/gh/hosts.yml", "backups", "state-snapshots",
    DEFAULT_TEAMS_PIPELINE_STORE_FILENAME, "mem0.json",
    "browser-profiles/live/Default/Cookies", "browser_profiles/default/Default/Login Data",
}


def _seed_stores(root):
    """Lay every credential store into a profile-shaped tree at *root*; returns their paths."""
    from hermes_cli.profiles import PROFILE_CREDENTIAL_PATHS

    stores = _EXTRA_STORES | PROFILE_CREDENTIAL_PATHS
    (root / "platforms").mkdir(parents=True)
    (root / "config.yaml").write_text("model: gpt-4\n")
    (root / "platforms" / "keep.json").write_text("{}")
    for rel in stores:
        is_dir = "." not in rel.rsplit("/", 1)[-1] and rel != "npmrc"  # token dirs vs single files
        target = root / rel / "store" if is_dir else root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fake-credential")
    return stores


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

    def test_named_export_ships_no_credential_store_or_copy_of_one(self, tmp_path, monkeypatch):
        """No credential store (any case spelling) and no copy Hermes' own writers leave of one
        (pre-update zip, update snapshot, config backup, migration .bak, corrupt auth.json) reaches a
        named-profile export. .op.env, npmrc, home/ CLI stores and the .bak copies have no suffix the
        scrub edits, so exclusion is their only guard; a hand-named config copy is the user's and
        ships scrubbed."""
        from hermes_cli.auth import _load_auth_store
        from hermes_cli.backup import create_pre_update_backup, create_quick_snapshot
        from hermes_cli.config_backups import backup_config
        from hermes_cli.post_update import _backup_existing
        from hermes_cli.profiles import PROFILE_CREDENTIAL_PATHS

        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "testprofile"
        _seed_stores(profile_dir)
        (profile_dir / "config.yaml").write_text(f"model:\n  api_key: {_LEAKED_KEY}\n")
        (profile_dir / "config.yaml.bak-my-note").write_text(f"model:\n  api_key: {_LEAKED_KEY}\n")
        (profile_dir / "GOOGLE_CHAT_USER_TOKENS").mkdir(exist_ok=True)
        (profile_dir / "GOOGLE_CHAT_USER_TOKENS" / "upper.json").write_text("fake-credential")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        copies = [
            create_pre_update_backup(hermes_home=profile_dir),
            profile_dir / "state-snapshots" / create_quick_snapshot(hermes_home=profile_dir),
            backup_config(profile_dir / "config.yaml", "setup"),
            *_backup_existing((profile_dir / ".env", profile_dir / "config.yaml")).values(),
        ]
        _load_auth_store(profile_dir / "auth.json")  # "fake-credential" is not JSON: quarantined
        copies.append(profile_dir / "auth.json.corrupt")
        assert all(c and c.exists() for c in copies), copies
        _patch_named_profile(monkeypatch, profiles_root, profile_dir)

        with tarfile.open(export_profile("testprofile", str(tmp_path / "export.tar.gz")), "r:gz") as tf:
            members = {m.name: m for m in tf.getmembers()}
            note = tf.extractfile("testprofile/config.yaml.bak-my-note").read().decode()

        assert {"testprofile/config.yaml", "testprofile/platforms/keep.json"} <= set(members)
        assert _LEAKED_KEY not in note
        rels = {*_EXTRA_STORES, *PROFILE_CREDENTIAL_PATHS, "google_chat_user_tokens/upper.json",
                *(c.relative_to(profile_dir).as_posix() for c in copies)}
        folded = {n.casefold() for n in members}
        leaked = sorted(r for r in rels if any(
            n == f"testprofile/{r}".casefold() or n.startswith(f"testprofile/{r}/".casefold()) for n in folded))
        assert not leaked, leaked

    @pytest.mark.parametrize("declare_owned", [False, True])
    def test_distribution_can_neither_plant_nor_replace_a_store(self, tmp_path, monkeypatch, declare_owned):
        """A distribution never installs a credential store or a recovery copy of one, nested ones
        included, whether it ships the whole payload or names each store in ``distribution_owned``,
        while a sibling such as ``platforms/keep.json`` still installs. An update that ships a FILE
        where the profile has a directory holding stores is refused before anything is written."""
        from pathlib import Path

        from hermes_cli.profile_distribution import (
            DistributionError, DistributionManifest, install_distribution, write_manifest,
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / ".hermes").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        stores = _seed_stores(tmp_path / "v1") | {"auth.json.corrupt", ".env.bak-20260928T084018Z",
                                                  "config.yaml.bak-20260928T084018Z"}
        for copy in ("auth.json.corrupt", ".env.bak-20260928T084018Z", "config.yaml.bak-20260928T084018Z"):
            (tmp_path / "v1" / copy).write_text("fake-credential")
        # PLATFORMS/PAIRING: on a case-insensitive filesystem that spelling IS the pairing store.
        owned = sorted(stores | {r.split("/")[0] for r in stores} | {"SOUL.md", "PLATFORMS/PAIRING"}) if declare_owned else []

        def stage(label, soul):
            staged = tmp_path / label
            staged.mkdir(exist_ok=True)
            (staged / "SOUL.md").write_text(soul)
            write_manifest(staged, DistributionManifest(name="dist", version="0.1.0", distribution_owned=owned))
            return staged

        installed = install_distribution(str(stage("v1", "v1")), name="dist").target_dir
        assert (installed / "platforms" / "keep.json").exists()
        planted = sorted(r for r in stores if any(
            f.is_file() and f.read_text() == "fake-credential"
            for f in ((installed / r), *((installed / r).rglob("*") if (installed / r).is_dir() else ()))))
        assert not planted, planted

        live = [installed / "platforms" / "pairing" / "approved.json",
                installed / "platforms" / "whatsapp" / "session" / "creds.json"]
        for store in live:
            store.parent.mkdir(parents=True)
            store.write_text("installer-credential")
        for shipped_file in ("platforms", "platforms/whatsapp"):
            update = stage(f"v2-{shipped_file.count('/')}", "v2")
            (update / shipped_file).parent.mkdir(parents=True, exist_ok=True)
            (update / shipped_file).write_text("not a directory")
            with pytest.raises(DistributionError):
                install_distribution(str(update), name="dist", force=True)
            assert [s.read_text() for s in live] == ["installer-credential"] * 2
            assert (installed / "SOUL.md").read_text() == "v1"


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
