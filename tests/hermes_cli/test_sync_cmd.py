"""Tests for `hermes sync` — secret exclusion, config allow-listing, init
defaults, status output shape, and the share install-command format.

These tests exercise the pure logic of ``hermes_cli.sync_cmd`` without hitting
a real git remote. Where git is needed (push abort path) we operate against a
local staging repo only.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixture — isolate HERMES_HOME so load_config / save_config / get_hermes_home
# all point at a throwaway dir.
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Minimal config so load_config() has something to read.
    (home / "config.yaml").write_text(
        yaml.safe_dump({"display": {"skin": "matrix"}}), encoding="utf-8"
    )
    return home


# ---------------------------------------------------------------------------
# build_synced_config_subset — only allow-listed keys ever leave the profile
# ---------------------------------------------------------------------------

class TestConfigAllowlist:
    def test_only_allowlisted_keys_survive(self):
        from hermes_cli.sync_cmd import build_synced_config_subset, CONFIG_ALLOWLIST

        full_config = {
            "display": {"skin": "matrix", "theme": "dark"},
            "default_toolsets": ["core", "web"],
            # everything below is a secret / must NOT sync
            "model": {"default": "gpt", "api_key": "sk-supersecret"},
            "providers": {"openai": {"api_key": "sk-leak-me"}},
            "gateway": {"token": "ghp_xxxx"},
            "anthropic_api_key": "sk-ant-leak",
        }

        subset = build_synced_config_subset(full_config)

        # Allow-listed keys present
        assert subset["display"]["skin"] == "matrix"
        assert subset["default_toolsets"] == ["core", "web"]

        # Non-allow-listed keys absent at every level
        assert "model" not in subset
        assert "providers" not in subset
        assert "gateway" not in subset
        assert "anthropic_api_key" not in subset
        # display.theme is NOT in the allowlist (only display.skin)
        assert "theme" not in subset.get("display", {})

        # Defensive: serialize the subset and confirm no secret string leaks.
        dumped = yaml.safe_dump(subset)
        assert "sk-supersecret" not in dumped
        assert "sk-leak-me" not in dumped
        assert "ghp_xxxx" not in dumped
        assert "sk-ant-leak" not in dumped

        # The allowlist is exactly the two documented keys.
        assert set(CONFIG_ALLOWLIST) == {"display.skin", "default_toolsets"}

    def test_missing_keys_are_omitted_not_nulled(self):
        from hermes_cli.sync_cmd import build_synced_config_subset

        subset = build_synced_config_subset({"display": {"theme": "dark"}})
        # display.skin missing → no display key emitted at all
        assert subset == {}


# ---------------------------------------------------------------------------
# Secret scan — a fake .env / hardcoded key must be detected
# ---------------------------------------------------------------------------

class TestSecretScan:
    def test_blocks_hardcoded_api_key(self, tmp_path):
        from hermes_cli.sync_cmd import scan_paths_for_secrets

        skill = tmp_path / "skills" / "leaky"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "# Leaky skill\n"
            'api_key = "AKIAIOSFODNN7EXAMPLE12345678"\n',
            encoding="utf-8",
        )

        findings = scan_paths_for_secrets([skill], tmp_path)
        assert findings, "expected the hardcoded credential to be detected"
        ids = {pid for _rel, pid, _snip in findings}
        # Either the generic hardcoded_secret or the AWS-specific pattern fires.
        assert ids & {"hardcoded_secret", "aws_access_key_leaked"}

    def test_blocks_private_key_block(self, tmp_path):
        from hermes_cli.sync_cmd import scan_paths_for_secrets

        f = tmp_path / "id_rsa.txt"
        # Construct the PEM header at runtime so this test file itself does not
        # contain a literal key marker (avoids tripping repo-level scanners).
        marker = "-----BEGIN " + "PRIVATE KEY-----"
        f.write_text(marker + "\nZmFrZWtleWRhdGE=\n", encoding="utf-8")
        findings = scan_paths_for_secrets([f], tmp_path)
        ids = {pid for _rel, pid, _snip in findings}
        assert "embedded_private_key" in ids

    def test_clean_skill_passes(self, tmp_path):
        from hermes_cli.sync_cmd import scan_paths_for_secrets

        skill = tmp_path / "skills" / "clean"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "# Clean skill\nReads config from the environment.\n"
            'key = os.environ.get("MY_SETTING")\n',
            encoding="utf-8",
        )
        findings = scan_paths_for_secrets([skill], tmp_path)
        # os.environ.get of a NON-secret name is fine; no credential leak.
        assert findings == []


# ---------------------------------------------------------------------------
# _mirror_into_staging — curated subset is copied WITHOUT dropping legitimately
# named user content. Regression guard for the `*cache*` over-exclusion bug
# that silently dropped skills like ``response-cache-helper`` from backups.
# ---------------------------------------------------------------------------

class TestMirror:
    def test_cache_named_skill_is_not_dropped(self, isolated_home):
        """A skill or file whose name contains 'cache'/'lock' must still sync.

        The original implementation passed ``*cache*`` and ``*.lock`` to
        ``shutil.ignore_patterns`` for the skills-tree copy, which silently
        excluded user content with those substrings — data loss on restore.
        """
        from hermes_cli.sync_cmd import _mirror_into_staging

        skills = isolated_home / "skills"
        # Legit user skills with "trap" substrings in their names.
        (skills / "response-cache-helper").mkdir(parents=True)
        (skills / "response-cache-helper" / "SKILL.md").write_text(
            "name: response-cache-helper\n", encoding="utf-8"
        )
        (skills / "normal-skill").mkdir(parents=True)
        (skills / "normal-skill" / "SKILL.md").write_text(
            "name: normal-skill\n", encoding="utf-8"
        )
        (skills / "normal-skill" / "cache_notes.md").write_text(
            "notes\n", encoding="utf-8"
        )
        # Build junk that SHOULD still be excluded.
        (skills / "normal-skill" / "__pycache__").mkdir()
        (skills / "normal-skill" / "__pycache__" / "x.pyc").write_text("junk")

        staging = isolated_home / "staging"
        staging.mkdir()
        _mirror_into_staging(isolated_home, staging)

        mirrored = sorted(p.name for p in (staging / "skills").iterdir())
        assert "response-cache-helper" in mirrored, (
            "cache-named skill was dropped from the backup — data loss"
        )
        assert (staging / "skills" / "normal-skill" / "cache_notes.md").exists(), (
            "cache-named file was dropped from the backup"
        )
        # Build junk is still excluded.
        assert not (staging / "skills" / "normal-skill" / "__pycache__").exists()

    def test_fresh_machine_pull_restores_skills_over_empty_dir(self, isolated_home):
        """First pull on a new device must restore skills even though Hermes
        pre-creates an empty ``skills/`` dir at startup.

        Regression guard: the original restore guarded on ``dst.exists()``,
        which is True for the empty scaffolded dir, so a non-interactive pull
        (stdin EOF -> answer 'no') SKIPPED the skill restore entirely. The fix
        only prompts when the local dir has REAL content.
        """
        from hermes_cli.sync_cmd import _restore_from_staging

        # staging (remote mirror) has real skills + memory
        staging = isolated_home / ".sync-git"
        (staging / "skills" / "git-helper").mkdir(parents=True)
        (staging / "skills" / "git-helper" / "SKILL.md").write_text("name: git-helper\n")
        (staging / "memories").mkdir(parents=True)
        (staging / "memories" / "MEMORY.md").write_text("real memory\n")

        # local profile is freshly scaffolded: EMPTY skills dir + BLANK memory stub
        (isolated_home / "skills").mkdir(exist_ok=True)
        (isolated_home / "memories").mkdir(exist_ok=True)
        (isolated_home / "memories" / "MEMORY.md").write_text("   \n")  # blank stub

        # Force=False, and DON'T provide stdin — _confirm_overwrite must never
        # be reached for empty/blank local content.
        restored = _restore_from_staging(isolated_home, staging, force=False)

        assert "skills" in restored, "fresh-machine pull skipped skills — data not restored"
        assert (isolated_home / "skills" / "git-helper" / "SKILL.md").is_file()
        assert (isolated_home / "memories" / "MEMORY.md").read_text().strip() == "real memory"

    def test_fresh_machine_pull_overwrites_default_soul(self, isolated_home):
        """First pull must restore the user's persona over the seeded default
        SOUL.md without prompting.

        Hermes seeds the canonical ``DEFAULT_SOUL_MD`` on first run. That stub
        is not user data — keeping it would mean a new device silently runs the
        stock persona instead of the synced one. The restore must recognise the
        default template as overwritable.
        """
        from hermes_cli.sync_cmd import _restore_from_staging
        from hermes_cli.default_soul import DEFAULT_SOUL_MD

        staging = isolated_home / ".sync-git"
        staging.mkdir(exist_ok=True)
        (staging / "SOUL.md").write_text("You are a pirate. Arr.\n", encoding="utf-8")

        # Local SOUL.md is the seeded default stub.
        (isolated_home / "SOUL.md").write_text(DEFAULT_SOUL_MD, encoding="utf-8")

        restored = _restore_from_staging(isolated_home, staging, force=False)

        assert "SOUL.md" in restored, "default SOUL.md was not overwritten by the synced persona"
        assert (isolated_home / "SOUL.md").read_text().strip() == "You are a pirate. Arr."

    def test_real_user_soul_is_still_protected(self, isolated_home, monkeypatch):
        """A NON-default local SOUL.md must still be protected: when stdin is
        unavailable (EOF -> 'no'), the synced persona is skipped, not silently
        clobbered.
        """
        from hermes_cli.sync_cmd import _restore_from_staging

        staging = isolated_home / ".sync-git"
        staging.mkdir(exist_ok=True)
        (staging / "SOUL.md").write_text("remote persona\n", encoding="utf-8")
        (isolated_home / "SOUL.md").write_text("MY HAND-WRITTEN PERSONA\n", encoding="utf-8")

        # No stdin -> _confirm_overwrite hits EOFError -> returns False (skip).
        def _raise(*a, **k):
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise)
        restored = _restore_from_staging(isolated_home, staging, force=False)

        assert "SOUL.md" not in restored
        assert (isolated_home / "SOUL.md").read_text().strip() == "MY HAND-WRITTEN PERSONA"


# ---------------------------------------------------------------------------
# .gitignore — secret/ephemera exclusion is present
# ---------------------------------------------------------------------------

class TestGitignore:
    def test_excludes_all_required_secret_paths(self):
        from hermes_cli.sync_cmd import GITIGNORE_CONTENT

        # Secret names that must be excluded (match anywhere).
        for line in [".env", ".env.*", "auth.json", "*.pem", "*.key"]:
            assert line in GITIGNORE_CONTENT, f"missing .gitignore exclusion: {line}"
        # Root-anchored profile artifacts.
        for line in ["/config.yaml", "/state.db", "/logs/", "/*cache*/", "/checkpoints/"]:
            assert line in GITIGNORE_CONTENT, f"missing .gitignore exclusion: {line}"

    def test_gitignore_behavior_real_git(self, tmp_path):
        """Run the actual .gitignore through ``git check-ignore``.

        Regression guard: an UNANCHORED ``*cache*/`` / ``*.lock`` pattern makes
        git ignore nested user skills like ``skills/response-cache-helper/`` —
        silent data loss that the mirror-layer unit test cannot catch because
        the file lands in staging but git refuses to track it. This test asserts
        both that profile-root artifacts ARE ignored and that nested cache/lock
        -named user skills are NOT.
        """
        import shutil
        import subprocess

        if shutil.which("git") is None:
            pytest.skip("git not available")

        from hermes_cli.sync_cmd import GITIGNORE_CONTENT

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / ".gitignore").write_text(GITIGNORE_CONTENT, encoding="utf-8")

        def ignored(rel: str) -> bool:
            return subprocess.run(
                ["git", "check-ignore", "-q", rel], cwd=repo
            ).returncode == 0

        # MUST be ignored — real profile-root secrets / ephemera.
        assert ignored("config.yaml")
        assert ignored(".env")
        assert ignored("auth.json")
        assert ignored("state.db")
        assert ignored("logs/agent.log")
        assert ignored("document_cache/x.json")

        # MUST NOT be ignored — legitimately-named nested user content.
        assert not ignored("skills/response-cache-helper/SKILL.md"), (
            "nested cache-named skill is being ignored by git — data loss"
        )
        assert not ignored("skills/poetry-lock-helper/SKILL.md"), (
            "nested lock-named skill is being ignored by git — data loss"
        )
        assert not ignored("skills/git-helper/SKILL.md")
        assert not ignored("memories/MEMORY.md")


# ---------------------------------------------------------------------------
# init — PRIVATE by default
# ---------------------------------------------------------------------------

class TestInitPrivateByDefault:
    def test_default_visibility_is_private(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc

        # No gh, paste a remote, accept the default (empty input → private).
        monkeypatch.setattr(sc, "_git_available", lambda: True)
        monkeypatch.setattr(sc, "_gh_available", lambda: False)
        monkeypatch.setattr(sc, "_has_tty", lambda: True)  # simulate an interactive terminal
        monkeypatch.setattr(sc, "_ensure_staging_repo", lambda staging: None)
        monkeypatch.setattr(sc, "_set_remote", lambda staging, url: None)

        inputs = iter([
            "",  # visibility choice → default = private
            "git@github.com:me/hermes-profile.git",  # pasted remote
        ])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))

        args = type("A", (), {"visibility": None, "remote": None})()
        sc.cmd_sync_init(args)

        saved = sc._read_sync_config()
        assert saved["visibility"] == "private"
        assert saved["remote"] == "git@github.com:me/hermes-profile.git"
        assert "initialized_at" in saved

    def test_explicit_public_flag_respected(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc

        monkeypatch.setattr(sc, "_git_available", lambda: True)
        monkeypatch.setattr(sc, "_gh_available", lambda: False)
        monkeypatch.setattr(sc, "_ensure_staging_repo", lambda staging: None)
        monkeypatch.setattr(sc, "_set_remote", lambda staging, url: None)

        args = type("A", (), {
            "visibility": "public",
            "remote": "https://github.com/me/pub.git",
        })()
        sc.cmd_sync_init(args)

        saved = sc._read_sync_config()
        assert saved["visibility"] == "public"


# ---------------------------------------------------------------------------
# status — output shape
# ---------------------------------------------------------------------------

class TestStatusShape:
    def test_uninitialized_status(self, isolated_home):
        import hermes_cli.sync_cmd as sc

        buf = io.StringIO()
        with redirect_stdout(buf):
            sc.cmd_sync_status(type("A", (), {})())
        out = buf.getvalue()
        assert "Not initialized" in out

    def test_initialized_status_shows_remote_and_visibility(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc

        sc._write_sync_config({
            "remote": "git@github.com:me/hermes-profile.git",
            "visibility": "private",
            "last_sync_at": "2026-01-01T00:00:00+00:00",
        })
        # Avoid real git work in this shape test.
        monkeypatch.setattr(sc, "_git_available", lambda: False)

        buf = io.StringIO()
        with redirect_stdout(buf):
            sc.cmd_sync_status(type("A", (), {})())
        out = buf.getvalue()
        assert "git@github.com:me/hermes-profile.git" in out
        assert "private" in out
        assert "2026-01-01" in out


# ---------------------------------------------------------------------------
# share — install command mirrors `npx skills add <url> --skill <name>`
# ---------------------------------------------------------------------------

class TestShare:
    def test_public_prints_install_command(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc

        # A skill must exist on disk.
        skill = isolated_home / "skills" / "my-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("# my-skill\n", encoding="utf-8")

        sc._write_sync_config({
            "remote": "git@github.com:me/hermes-profile.git",
            "visibility": "public",
        })

        buf = io.StringIO()
        with redirect_stdout(buf):
            sc.cmd_sync_share(type("A", (), {"skill_name": "my-skill", "push": False})())
        out = buf.getvalue()
        assert "npx skills add https://github.com/me/hermes-profile --skill my-skill" in out

    def test_private_warns_about_access(self, isolated_home):
        import hermes_cli.sync_cmd as sc

        skill = isolated_home / "skills" / "priv-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("# priv-skill\n", encoding="utf-8")

        sc._write_sync_config({
            "remote": "git@github.com:me/secret.git",
            "visibility": "private",
        })

        buf = io.StringIO()
        with redirect_stdout(buf):
            sc.cmd_sync_share(type("A", (), {"skill_name": "priv-skill", "push": False})())
        out = buf.getvalue()
        assert "PRIVATE" in out and "access" in out.lower()


# ---------------------------------------------------------------------------
# remote → github url normalization
# ---------------------------------------------------------------------------

class TestRemoteNormalization:
    @pytest.mark.parametrize("remote,expected", [
        ("git@github.com:me/repo.git", "https://github.com/me/repo"),
        ("https://github.com/me/repo.git", "https://github.com/me/repo"),
        ("https://github.com/me/repo", "https://github.com/me/repo"),
        ("ssh://git@github.com/me/repo.git", "https://github.com/me/repo"),
        ("git@gitlab.com:me/repo.git", None),
    ])
    def test_normalize(self, remote, expected):
        from hermes_cli.sync_cmd import _remote_to_github_url

        assert _remote_to_github_url(remote) == expected


# ---------------------------------------------------------------------------
# Command-level secret-abort — the most security-critical branches. Prove that
# `hermes sync push` and `hermes sync share --push` actually REFUSE to commit
# when the staged content trips the secret scanner (not just that the scanner
# detects it in isolation).
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _git(args, cwd):
    import subprocess
    return subprocess.run(["git", *args], cwd=str(cwd), text=True, capture_output=True)


class TestPushSecretAbort:
    def _init_repo(self, isolated_home, tmp_path):
        """Configure sync against a local bare remote so push can run for real."""
        import shutil
        if shutil.which("git") is None:
            pytest.skip("git not available")
        from hermes_cli.sync_cmd import _write_sync_config
        bare = tmp_path / "remote.git"
        _git(["init", "--bare", "-q", str(bare)], cwd=tmp_path)
        _write_sync_config({"remote": str(bare), "visibility": "private"})
        return bare

    def test_push_aborts_and_pushes_nothing_when_secret_present(
        self, isolated_home, tmp_path
    ):
        from hermes_cli import sync_cmd

        bare = self._init_repo(isolated_home, tmp_path)
        # A skill with an embedded AWS-shaped key.
        skill = isolated_home / "skills" / "leaky"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: leaky\n---\naws_secret_access_key = AKIAIOSFODNN7EXAMPLE0123456789abcdef\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc:
            sync_cmd.cmd_sync_push(_Args(message=None))
        assert exc.value.code == 1

        # The bare remote must have received NOTHING — no refs, no commits.
        out = _git(["log", "--oneline", "-1"], cwd=bare)
        assert out.returncode != 0 or not out.stdout.strip(), (
            "secret-abort still pushed a commit to the remote"
        )

    def test_push_succeeds_when_clean(self, isolated_home, tmp_path):
        from hermes_cli import sync_cmd

        bare = self._init_repo(isolated_home, tmp_path)
        skill = isolated_home / "skills" / "clean"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: clean\n---\nUse git.\n", encoding="utf-8")
        # local git identity for the staging commit (CI runners may lack one)
        staging = sync_cmd._sync_dir()
        sync_cmd._ensure_staging_repo(staging)
        _git(["config", "user.email", "test@example.com"], cwd=staging)
        _git(["config", "user.name", "test"], cwd=staging)

        sync_cmd.cmd_sync_push(_Args(message="clean push"))

        # The clean skill landed on the remote.
        ls = _git(["ls-tree", "-r", "--name-only", "HEAD"], cwd=bare)
        assert "skills/clean/SKILL.md" in ls.stdout


class TestMirrorSymlink:
    def test_symlink_in_skill_is_not_dereferenced(self, isolated_home, tmp_path):
        """A symlink inside a skill must be copied AS a link, not have its
        target's contents pulled into the backup (secret-leak guard)."""
        import os
        from hermes_cli.sync_cmd import _mirror_into_staging

        # A secret file OUTSIDE the profile that a malicious/accidental symlink
        # points at.
        secret = tmp_path / "id_rsa"
        secret.write_text("-----BEGIN PRIVATE KEY-----\nLEAK\n-----END PRIVATE KEY-----\n")

        skill = isolated_home / "skills" / "linky"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: linky\n---\nx\n", encoding="utf-8")
        try:
            os.symlink(secret, skill / "key")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")

        staging = isolated_home / ".sync-git"
        staging.mkdir()
        _mirror_into_staging(isolated_home, staging)

        copied = staging / "skills" / "linky" / "key"
        assert copied.is_symlink(), "symlink was dereferenced — target contents copied in"
        # The secret bytes must NOT be present as a real file in staging.
        assert not (copied.exists() and not copied.is_symlink())


# ---------------------------------------------------------------------------
# Zip export — share skills as a self-contained archive (individual or group),
# no sync init / remote required. Same pre-export secret scan as git-share.
# ---------------------------------------------------------------------------

class TestZipExport:
    def _make_skill(self, home, name, body="Use git carefully.\n"):
        sp = home / "skills" / name
        sp.mkdir(parents=True)
        (sp / "SKILL.md").write_text(f"---\nname: {name}\n---\n{body}", encoding="utf-8")
        return sp

    def test_export_single_skill(self, isolated_home, tmp_path):
        from hermes_cli.sync_cmd import _export_skills_zip
        import zipfile

        self._make_skill(isolated_home, "git-helper")
        out = tmp_path / "git-helper.zip"
        written = _export_skills_zip(isolated_home, ["git-helper"], out)

        assert written == ["git-helper"]
        assert out.is_file()
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        # Drop-in layout: skills/<name>/... so `unzip -d ~/.hermes/` just works.
        assert "skills/git-helper/SKILL.md" in names

    def test_export_group(self, isolated_home, tmp_path):
        from hermes_cli.sync_cmd import _export_skills_zip
        import zipfile

        self._make_skill(isolated_home, "alpha")
        self._make_skill(isolated_home, "beta")
        out = tmp_path / "group.zip"
        written = _export_skills_zip(isolated_home, ["alpha", "beta"], out)

        assert set(written) == {"alpha", "beta"}
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert {"skills/alpha/SKILL.md", "skills/beta/SKILL.md"} <= names

    def test_export_aborts_on_secret(self, isolated_home, tmp_path):
        from hermes_cli.sync_cmd import _export_skills_zip

        sp = self._make_skill(isolated_home, "leaky")
        (sp / "SKILL.md").write_text(
            "---\nname: leaky\n---\naws_secret_access_key = AKIAIOSFODNN7EXAMPLE0123456789abcdef\n",
            encoding="utf-8",
        )
        out = tmp_path / "leaky.zip"
        with pytest.raises(SystemExit) as exc:
            _export_skills_zip(isolated_home, ["leaky"], out)
        assert exc.value.code == 1
        assert not out.exists(), "zip was written despite a detected secret"

    def test_export_missing_skill_errors(self, isolated_home, tmp_path):
        from hermes_cli.sync_cmd import _export_skills_zip

        with pytest.raises(SystemExit) as exc:
            _export_skills_zip(isolated_home, ["does-not-exist"], tmp_path / "x.zip")
        assert exc.value.code == 1

    def test_share_command_zip_branch_no_init_required(self, isolated_home, tmp_path, monkeypatch):
        """`hermes sync share <skill> --zip PATH` works with NO sync config."""
        from hermes_cli import sync_cmd

        self._make_skill(isolated_home, "standalone")
        out = tmp_path / "out.zip"

        class _A:
            skill_name = ["standalone"]
            zip = str(out)
            push = False

        # No _write_sync_config call → sync is uninitialized; zip must still work.
        sync_cmd.cmd_sync_share(_A())
        assert out.is_file()


# ---------------------------------------------------------------------------
# `hermes skills export` — the second entry point, sharing the same core as
# `sync share --zip` (one implementation, two command trees).
# ---------------------------------------------------------------------------

class TestSkillsExportEntryPoint:
    def _make_skill(self, home, name):
        sp = home / "skills" / name
        sp.mkdir(parents=True)
        (sp / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n", encoding="utf-8")

    def test_skills_export_single(self, isolated_home, tmp_path):
        from hermes_cli.skills_hub import do_skills_export
        import zipfile

        self._make_skill(isolated_home, "git-helper")
        out = tmp_path / "x.zip"
        do_skills_export(["git-helper"], output=str(out))
        assert out.is_file()
        with zipfile.ZipFile(out) as zf:
            assert "skills/git-helper/SKILL.md" in zf.namelist()

    def test_skills_export_all_skips_dotdirs(self, isolated_home, tmp_path):
        from hermes_cli.skills_hub import do_skills_export
        import zipfile

        self._make_skill(isolated_home, "alpha")
        self._make_skill(isolated_home, "beta")
        (isolated_home / "skills" / ".hub").mkdir()  # provenance dir — must be skipped
        out = tmp_path / "all.zip"
        do_skills_export([], output=str(out), export_all=True)
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
        assert {"skills/alpha/SKILL.md", "skills/beta/SKILL.md"} <= names
        assert not any(".hub" in n for n in names)

    def test_skills_export_aborts_on_secret(self, isolated_home, tmp_path):
        from hermes_cli.skills_hub import do_skills_export

        sp = isolated_home / "skills" / "leaky"
        sp.mkdir(parents=True)
        (sp / "SKILL.md").write_text(
            "aws_secret_access_key = AKIAIOSFODNN7EXAMPLE0123456789abcdef\n", encoding="utf-8"
        )
        out = tmp_path / "leak.zip"
        with pytest.raises(SystemExit) as exc:
            do_skills_export(["leaky"], output=str(out))
        assert exc.value.code == 1
        assert not out.exists()


# ---------------------------------------------------------------------------
# Slash entry point + non-interactive (no-TTY) safety — the in-chat surface
# for CLI / TUI / desktop must never hang on input().
# ---------------------------------------------------------------------------

class TestSlashAndNonInteractive:
    def test_slash_status_routes_to_status(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc
        called = {}
        monkeypatch.setattr(sc, "cmd_sync_status", lambda args: called.setdefault("status", True))
        sc.handle_sync_slash("/sync status")
        assert called.get("status")

    def test_slash_share_zip_parsed(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc
        seen = {}
        monkeypatch.setattr(sc, "cmd_sync_share", lambda args: seen.update(args.__dict__))
        sc.handle_sync_slash("/sync share alpha beta --zip /tmp/t.zip")
        assert seen["skill_name"] == ["alpha", "beta"]
        assert seen["zip"] == "/tmp/t.zip"
        assert seen["push"] is False

    def test_slash_push_passes_message(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc
        seen = {}
        monkeypatch.setattr(sc, "cmd_sync_push", lambda args: seen.update(args.__dict__))
        sc.handle_sync_slash('/sync push -m "backup before refactor"')
        assert seen["message"] == "backup before refactor"

    def test_init_no_tty_defaults_private_and_errors_without_remote(self, isolated_home, monkeypatch):
        """No TTY + no gh + no --remote → clear SystemExit, never an input() hang."""
        import hermes_cli.sync_cmd as sc
        monkeypatch.setattr(sc, "_git_available", lambda: True)
        monkeypatch.setattr(sc, "_gh_available", lambda: False)
        monkeypatch.setattr(sc, "_has_tty", lambda: False)
        monkeypatch.setattr(sc, "_ensure_staging_repo", lambda staging: None)
        # input() must NOT be called — if it is, this raises instead of hanging.
        monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(AssertionError("prompted with no TTY")))

        args = type("A", (), {"visibility": None, "remote": None})()
        with pytest.raises(SystemExit) as exc:
            sc.cmd_sync_init(args)
        assert exc.value.code == 1

    def test_init_no_tty_with_remote_succeeds_private(self, isolated_home, monkeypatch):
        import hermes_cli.sync_cmd as sc
        monkeypatch.setattr(sc, "_git_available", lambda: True)
        monkeypatch.setattr(sc, "_gh_available", lambda: False)
        monkeypatch.setattr(sc, "_has_tty", lambda: False)
        monkeypatch.setattr(sc, "_ensure_staging_repo", lambda staging: None)
        monkeypatch.setattr(sc, "_set_remote", lambda staging, url: None)
        monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(AssertionError("prompted with no TTY")))

        args = type("A", (), {"visibility": None, "remote": "git@github.com:me/p.git"})()
        sc.cmd_sync_init(args)
        saved = sc._read_sync_config()
        assert saved["visibility"] == "private"  # safe default with no TTY
        assert saved["remote"] == "git@github.com:me/p.git"
