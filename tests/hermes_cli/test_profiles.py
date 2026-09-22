"""Comprehensive tests for hermes_cli.profiles module.

Tests cover: validation, directory resolution, CRUD operations, active profile
management, export/import, renaming, alias collision checks, profile isolation,
and shell completion generation.
"""

import json
import io
import os
import shutil
import sys
import tarfile
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from hermes_cli import profiles
from hermes_cli.profiles import (
    _clone_all_copytree_ignore,
    normalize_profile_name,
    validate_profile_name,
    get_profile_dir,
    create_profile,
    delete_profile,
    list_profiles,
    set_active_profile,
    get_active_profile,
    get_active_profile_name,
    resolve_profile_env,
    check_alias_collision,
    create_wrapper_script,
    remove_wrapper_script,
    validate_alias_name,
    rename_profile,
    export_profile,
    import_profile,
    _get_profiles_root,
    _get_default_hermes_home,
    seed_profile_skills,
    NO_BUNDLED_SKILLS_MARKER,
    backfill_profile_envs,
    profiles_to_serve,
)
from hermes_cli.config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Shared fixture: redirect Path.home() and HERMES_HOME for profile tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Set up an isolated environment for profile tests.

    * Path.home() -> tmp_path  (so _get_profiles_root() = tmp_path/.hermes/profiles)
    * HERMES_HOME  -> tmp_path/.hermes  (so get_hermes_home() agrees)
    * Creates the bare-minimum ~/.hermes directory.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


# ===================================================================
# TestValidateProfileName
# ===================================================================

class TestNormalizeProfileName:
    """Tests for normalize_profile_name()."""

    def test_title_case_normalized(self):
        assert normalize_profile_name("Jules") == "jules"
        assert normalize_profile_name("  Librarian ") == "librarian"


class TestValidateProfileName:
    """Tests for validate_profile_name()."""

    @pytest.mark.parametrize("name", ["coder", "work-bot", "a1", "my_agent"])
    def test_valid_names_accepted(self, name):
        # Should not raise
        validate_profile_name(name)


    @pytest.mark.parametrize("name", ["UPPER", "has space", ".hidden", "-leading"])
    def test_invalid_names_rejected(self, name):
        with pytest.raises(ValueError):
            validate_profile_name(name)


# ===================================================================
# TestGetProfileDir
# ===================================================================

class TestGetProfileDir:
    """Tests for get_profile_dir()."""

    def test_default_returns_hermes_home(self, profile_env):
        tmp_path = profile_env
        result = get_profile_dir("default")
        assert result == tmp_path / ".hermes"

    @pytest.mark.parametrize("name", ["..", "../outside", "../../tmp", "a/b", "a\\b", ".hidden", "has space"])
    def test_traversal_and_invalid_names_rejected(self, name, profile_env):
        # The name becomes a path component under profiles/; invalid ids must
        # raise instead of escaping the root.
        with pytest.raises(ValueError):
            get_profile_dir(name)

    @pytest.mark.parametrize("name", ["..", "../outside", "a/b"])
    def test_profile_exists_false_for_invalid_names(self, name, profile_env):
        assert profiles.profile_exists(name) is False


# ===================================================================
# TestCreateProfile
# ===================================================================

class TestCreateProfile:
    """Tests for create_profile()."""


    def test_seeds_placeholder_env_file(self, profile_env):
        """Fresh profiles get their own .env (owner-only) so channel/env
        writes are profile-scoped from day one instead of falling through
        to the shell environment / root install."""
        import stat
        profile_dir = create_profile("coder", no_alias=True)
        env_path = profile_dir / ".env"
        assert env_path.exists()
        content = env_path.read_text(encoding="utf-8")
        # Placeholder only — no credentials leak in from anywhere.
        assert all(
            line.startswith("#") or not line.strip()
            for line in content.splitlines()
        )
        mode = stat.S_IMODE(env_path.stat().st_mode)
        assert mode == 0o600


    def test_fresh_profile_inherits_a_usable_model(self, profile_env):
        """A profile created without a clone source still resolves a provider.

        Without this it gets no config.yaml at all, so its very first turn dies
        with "No LLM provider configured" — created, but unable to run. Fresh
        means fresh skills and SOUL, not unreachable.
        """
        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text(
            "model:\n  provider: nous\n  default: some/model\n"
        )

        profile_dir = create_profile("coder", no_alias=True)

        cfg = yaml.safe_load((profile_dir / "config.yaml").read_text())
        assert cfg["model"]["provider"] == "nous"
        assert cfg["model"]["default"] == "some/model"


    def test_new_profile_uses_global_route_without_copying_it(self, profile_env, monkeypatch):
        default_home = profile_env / ".hermes"
        (default_home / "ROUTING_POLICY.md").write_text(
            "---\nprofile_defaults:\n  web:\n    search_backend: tavily\n"
            "  mcp_servers:\n    parallel_search:\n      enabled: true\n---\n",
            encoding="utf-8",
        )

        profile_dir = create_profile("routed", no_alias=True)
        raw = yaml.safe_load((profile_dir / "config.yaml").read_text(encoding="utf-8"))
        assert "web" not in raw
        assert raw["mcp_servers"]["parallel_search"]["enabled"] is True

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        from hermes_cli.config import load_config
        assert load_config()["web"]["search_backend"] == "tavily"


    def test_fresh_profile_inherits_its_custom_provider_gateway(self, profile_env):
        """The inherited model may point at a custom `providers:` gateway (self-hosted / local
        endpoint). Copying `model` alone left the new bot with `model.provider: my-gateway` and
        "Unknown provider 'my-gateway'" on its first turn (#101885 / #94071 class); the provider
        definition must travel with the model it backs, and nothing else from `providers:` does.
        """
        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text(
            "model:\n  provider: my-gateway\n  default: my-finetune\n"
            "providers:\n  my-gateway:\n    api: https://llm.internal.example.com/v1\n    key_env: GW_KEY\n"
            "  unrelated:\n    api: https://other.example.com/v1\n"
        )

        profile_dir = create_profile("coder", no_alias=True)

        cfg = yaml.safe_load((profile_dir / "config.yaml").read_text())
        assert cfg["model"] == {"provider": "my-gateway", "default": "my-finetune"}
        assert cfg["providers"] == {"my-gateway": {"api": "https://llm.internal.example.com/v1", "key_env": "GW_KEY"}}

    def test_fresh_profile_model_is_copied_not_linked(self, profile_env):
        """Profiles stay independent islands.

        The model block is copied at creation, so later edits to the source
        profile never reach one already created from it.
        """
        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text(
            "model:\n  provider: nous\n  default: some/model\n"
        )
        profile_dir = create_profile("coder", no_alias=True)

        (default_home / "config.yaml").write_text(
            "model:\n  provider: other\n  default: changed/model\n"
        )

        cfg = yaml.safe_load((profile_dir / "config.yaml").read_text())
        assert cfg["model"]["provider"] == "nous"
        assert cfg["model"]["default"] == "some/model"




    def test_clone_config_copies_files(self, profile_env):
        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        # Create source config files in default profile
        (default_home / "config.yaml").write_text("model: test")
        (default_home / ".env").write_text("KEY=val")
        (default_home / "SOUL.md").write_text("Be helpful.")

        profile_dir = create_profile("coder", clone_config=True, no_alias=True)

        cloned_config = yaml.safe_load((profile_dir / "config.yaml").read_text())
        assert cloned_config["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert cloned_config["model"] == "test"
        assert (profile_dir / ".env").read_text().strip() == "KEY=val"
        assert (profile_dir / "SOUL.md").read_text() == "Be helpful."

    def test_clone_sync_imports_carries_manifest_but_never_links_profiles(self, profile_env):
        """--sync-imports copies import-sync.json (a pointer at EXTERNAL agent trees) and nothing
        else changes: the clone still gets its own config/skills copies, never a live link."""
        from hermes_cli.agent_import_sync import SYNC_MANIFEST_NAME, load_sync_manifest

        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text("model: test")
        manifest = {"version": 1, "agents": {"claude-code": {
            "source": str(profile_env / ".claude"), "digest": "d", "overwrite": False,
            "last_import": 1, "imported_skills": ["s1"]}}}
        (default_home / SYNC_MANIFEST_NAME).write_text(json.dumps(manifest))

        plain = create_profile("plain", clone_config=True, no_alias=True)
        assert not (plain / SYNC_MANIFEST_NAME).exists()

        synced = create_profile("synced", clone_config=True, sync_imports=True, no_alias=True)
        assert load_sync_manifest(synced)["agents"] == manifest["agents"]
        # Editing the source afterwards does not reach the clone: still an independent island.
        (default_home / "config.yaml").write_text("model: changed")
        assert yaml.safe_load((synced / "config.yaml").read_text())["model"] == "test"

    @staticmethod
    def _home_with_linked_skill(profile_env):
        """Source home: ``skills/foo`` links into an ``external_dirs`` root, ``skills/local`` is physical."""
        default_home = profile_env / ".hermes"
        external = profile_env / "agents-skills"
        (external / "foo").mkdir(parents=True)
        (external / "foo" / "SKILL.md").write_text("# external foo\n", encoding="utf-8")
        (default_home / "skills" / "local").mkdir(parents=True)
        (default_home / "skills" / "local" / "SKILL.md").write_text("# local\n", encoding="utf-8")
        (default_home / "config.yaml").write_text(f"model: test\nskills:\n  external_dirs:\n    - {external}\n")
        return default_home, external

    @pytest.mark.parametrize("clone_kwargs", [{"clone_config": True}, {"clone_all": True}])
    def test_clone_recreates_skill_junctions_and_skips_dangling_ones(self, profile_env, monkeypatch, clone_kwargs):
        """A junctioned skill stays a link (one candidate with its external original), a dangling
        junction is skipped without failing the clone. The reparse-point predicate and CreateJunction
        are Windows-only; simulate both so the copy/re-create contract runs on every host."""
        default_home, external = self._home_with_linked_skill(profile_env)
        # copytree sees plain directories (what a junction looks like to os.stat on Windows).
        (default_home / "skills" / "foo").mkdir()
        (default_home / "skills" / "foo" / "SKILL.md").write_text("# a physical copy would come from here\n")
        (default_home / "skills" / "gone").mkdir()
        targets = {str(default_home / "skills" / "foo"): str(external / "foo"),
                   str(default_home / "skills" / "gone"): str(profile_env / "nowhere")}
        monkeypatch.setattr(profiles, "_junction_target", lambda path: targets.get(path), raising=False)

        def _create_junction(target, dst):
            if not os.path.isdir(target):
                raise OSError("target missing")  # what _winapi.CreateJunction does for a dangling junction
            os.symlink(target, dst, target_is_directory=True)
        monkeypatch.setitem(sys.modules, "_winapi", types.SimpleNamespace(CreateJunction=_create_junction))

        clone = create_profile("clone", no_alias=True, **clone_kwargs)
        foo = clone / "skills" / "foo"
        assert foo.is_symlink() and foo.resolve() == (external / "foo").resolve()
        assert (foo / "SKILL.md").read_text(encoding="utf-8") == "# external foo\n"
        assert (clone / "skills" / "local" / "SKILL.md").is_file()
        assert not (clone / "skills" / "gone").exists()
        from tools.skills_tool import _collect_skill_candidates
        assert len(_collect_skill_candidates("foo", None, [clone / "skills", external])) == 1

    @pytest.mark.windows_only
    def test_clone_keeps_real_ntfs_junction(self, profile_env):
        import _winapi
        default_home, external = self._home_with_linked_skill(profile_env)
        _winapi.CreateJunction(str(external / "foo"), str(default_home / "skills" / "foo"))

        clone = create_profile("clone", clone_config=True, no_alias=True)
        foo = clone / "skills" / "foo"
        assert os.lstat(foo).st_reparse_tag == profiles.stat.IO_REPARSE_TAG_MOUNT_POINT
        assert foo.resolve() == (external / "foo").resolve()
        from tools.skills_tool import _collect_skill_candidates
        assert len(_collect_skill_candidates("foo", None, [clone / "skills", external])) == 1

    def test_sync_imports_requires_a_clone_source(self, profile_env):
        with pytest.raises(ValueError, match="--sync-imports requires"):
            create_profile("lonely", sync_imports=True, no_alias=True)

    def test_clone_all_does_not_copy_cron_jobs(self, profile_env):
        # Cron jobs are scheduled work bound to the source profile + origin channel; a clone
        # that inherits jobs.json fires every job twice (two gateways, same job ids).
        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text("model: test")
        (default_home / "cron").mkdir()
        (default_home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{"id": "abc123def456"}]}))
        (default_home / "cron" / "output").mkdir()

        profile_dir = create_profile("coder", clone_all=True, no_alias=True)

        assert (profile_dir / "cron").is_dir()
        assert not any((profile_dir / "cron").iterdir())
        assert yaml.safe_load((profile_dir / "config.yaml").read_text())["model"] == "test"

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="special files need a POSIX filesystem")
    def test_clone_all_skips_special_files(self, profile_env):
        # A live source profile holds special files copytree cannot copy (e.g. a suffixless
        # agent-browser control socket); one of them must not abort the whole clone.
        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text("model: test")
        browser_dir = default_home / "home" / ".agent-browser"
        browser_dir.mkdir(parents=True)
        (browser_dir / "state.json").write_text("{}")
        os.mkfifo(browser_dir / "control")

        profile_dir = create_profile("coder", clone_all=True, no_alias=True)

        assert (profile_dir / "home" / ".agent-browser" / "state.json").is_file()
        assert not (profile_dir / "home" / ".agent-browser" / "control").exists()





# ===================================================================
# TestNoSkillsOptOut
# ===================================================================

class TestNoSkillsOptOut:
    """Tests for `hermes profile create --no-skills` and the opt-out marker."""

    def test_no_skills_writes_marker_and_skips_seeding(self, profile_env):
        profile_dir = create_profile("orchestrator", no_alias=True, no_skills=True)

        # Marker file is present
        marker = profile_dir / NO_BUNDLED_SKILLS_MARKER
        assert marker.is_file(), "expected .no-bundled-skills marker in profile root"
        assert "--no-skills" in marker.read_text()

        # skills/ dir exists (profile bootstrapping still creates the dir) but
        # contains nothing yet because create_profile itself doesn't seed.
        assert (profile_dir / "skills").is_dir()
        assert list((profile_dir / "skills").iterdir()) == []




    def test_delete_marker_re_enables_seeding(self, profile_env, monkeypatch):
        """Deleting .no-bundled-skills opts the profile back into a full sync.

        The sync subprocess runs in BOTH states: with the marker present,
        sync_skills() itself seeds only the essential skills and reports
        ``skipped_opt_out``; without it, a normal full sync happens.
        """
        import subprocess as _sp

        profile_dir = create_profile("orchestrator", no_alias=True, no_skills=True)
        assert (profile_dir / NO_BUNDLED_SKILLS_MARKER).is_file()

        # Marker present: the subprocess still runs (essential-only seeding
        # happens inside sync_skills) and its skipped_opt_out flag surfaces.
        called = []
        stdout_by_call = [
            '{"copied": ["hermes-agent"], "skipped_opt_out": true}',
            '{"copied": []}',
        ]
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (called.append(a), _sp.CompletedProcess(
                args=a, returncode=0,
                stdout=stdout_by_call[min(len(called) - 1, 1)], stderr="",
            ))[1],
        )
        r1 = seed_profile_skills(profile_dir, quiet=True)
        assert r1.get("skipped_opt_out") is True
        assert r1.get("copied") == ["hermes-agent"]
        assert len(called) == 1

        # Delete marker → next call is a normal full sync.
        (profile_dir / NO_BUNDLED_SKILLS_MARKER).unlink()
        r2 = seed_profile_skills(profile_dir, quiet=True)
        assert r2 == {"copied": []}
        assert len(called) == 2


# ===================================================================
# TestBackfillProfileEnvs
# ===================================================================

class TestBackfillProfileEnvs:
    """Tests for backfill_profile_envs() — the `hermes update` pass that
    gives pre-#44792 profiles (created before .env seeding) their own
    .env, copied from the default install so credentials don't break."""

    def test_copies_default_env_into_envless_profiles(self, profile_env):
        import stat
        tmp_path = profile_env
        (tmp_path / ".hermes" / ".env").write_text("OPENROUTER_API_KEY=root-key\n")
        p1 = create_profile("old1", no_alias=True)
        p2 = create_profile("old2", no_alias=True)
        # Simulate pre-#44792 profiles: no .env
        (p1 / ".env").unlink()
        (p2 / ".env").unlink()

        backfilled = backfill_profile_envs(quiet=True)

        assert sorted(backfilled) == ["old1", "old2"]
        for p in (p1, p2):
            assert (p / ".env").read_text() == "OPENROUTER_API_KEY=root-key\n"
            assert stat.S_IMODE((p / ".env").stat().st_mode) == 0o600


    def test_placeholder_when_default_has_no_env(self, profile_env):
        p = create_profile("noroot", no_alias=True)
        (p / ".env").unlink()

        backfilled = backfill_profile_envs(quiet=True)

        assert backfilled == ["noroot"]
        content = (p / ".env").read_text(encoding="utf-8")
        assert all(
            line.startswith("#") or not line.strip()
            for line in content.splitlines()
        )

    def test_no_profiles_root_is_noop(self, profile_env):
        assert backfill_profile_envs(quiet=True) == []


# ===================================================================
# TestDeleteProfile
# ===================================================================

class TestDeleteProfile:
    """Tests for delete_profile()."""


    def test_rmtree_failure_raises(self, profile_env):
        profile_dir = create_profile("coder", no_alias=True)
        set_active_profile("coder")

        with patch("hermes_cli.profiles._cleanup_gateway_service"), \
             patch("hermes_cli.profiles.time.sleep"), \
             patch("hermes_cli.profiles.shutil.rmtree", side_effect=PermissionError("locked")):
            with pytest.raises(RuntimeError, match="Could not remove profile directory"):
                delete_profile("coder", yes=True)

        assert profile_dir.is_dir()
        assert get_active_profile() == "default"

    def test_delete_purges_profile_keyed_identity(self, profile_env):
        """A deleted profile must not keep routing/heartbeat/delivery identity (#111926, delete side).

        The name is baked into ``agent:<name>:*`` routing keys, ``gateway_heartbeats.profile`` and
        ``delivery_obligations``. Left behind, a later event on a chat keyed to the deleted name
        enters the routing index, resolves a profile whose directory is gone, and logs
        ``Profile '<name>' does not exist`` on every subsequent event.
        """
        from hermes_state import SessionDB
        import time

        tmp_path = profile_env
        create_profile("gone", no_alias=True)
        create_profile("keepme", no_alias=True)
        scope = str(tmp_path / ".hermes" / "sessions")
        db = SessionDB(tmp_path / ".hermes" / "state.db")
        db.save_gateway_routing_entry(
            "agent:gone:feishu:dm:chatA",
            json.dumps({"session_key": "agent:gone:feishu:dm:chatA",
                        "origin": {"platform": "feishu", "chat_id": "chatA",
                                   "profile": "gone"}}),
            scope=scope)
        db.save_gateway_routing_entry(
            "agent:keepme:feishu:dm:chatB",
            json.dumps({"session_key": "agent:keepme:feishu:dm:chatB",
                        "origin": {"platform": "feishu", "chat_id": "chatB",
                                   "profile": "keepme"}}),
            scope=scope)
        db.register_backend_heartbeat(
            backend_id="be-gone", pid=1, started_at=time.time(), profile="gone", host="h")
        db.register_backend_heartbeat(
            backend_id="be-keep", pid=2, started_at=time.time(), profile="keepme", host="h")
        db.close()

        # No live multiplexer: nothing else owns the store, so this process purges the durable rows.
        with patch("hermes_cli.profiles._cleanup_gateway_service"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
            delete_profile("gone", yes=True)

        check = SessionDB(tmp_path / ".hermes" / "state.db")
        try:
            assert set(check.load_gateway_routing_entries(scope=scope)) == {
                "agent:keepme:feishu:dm:chatB"}
            assert check._read_one(
                "SELECT COUNT(*) AS n FROM gateway_heartbeats WHERE profile = ?",
                ("gone",))["n"] == 0
            assert check._read_one(
                "SELECT COUNT(*) AS n FROM gateway_heartbeats WHERE profile = ?",
                ("keepme",))["n"] == 1
        finally:
            check.close()

    def test_delete_reports_pending_settlement_for_a_live_multiplexer(self, profile_env, capsys):
        """With a live multiplexer the owner process purges, so the CLI must not race it (#111926).

        That process holds the routing index in memory and writes it back, so a CLI-side DELETE
        would be undone by its next save. When it cannot be reached the delete is NOT a clean
        success: the identity settlement is reported as pending, with the retry named.
        """
        from hermes_state import SessionDB
        from hermes_cli.profiles import ProfileIdentitySettlementPending

        tmp_path = profile_env
        create_profile("gone", no_alias=True)
        scope = str(tmp_path / ".hermes" / "sessions")
        db = SessionDB(tmp_path / ".hermes" / "state.db")
        db.save_gateway_routing_entry(
            "agent:gone:feishu:dm:chatA",
            json.dumps({"session_key": "agent:gone:feishu:dm:chatA"}),
            scope=scope)
        db.close()

        with patch("hermes_cli.profiles._cleanup_gateway_service"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=True):
            with pytest.raises(ProfileIdentitySettlementPending,
                               match="identity settlement is still pending") as ei:
                delete_profile("gone", yes=True)

        # Typed partial success: the filesystem delete completed, the identity did not, and the
        # payload carries what a surfacing caller needs to report it and retry.
        assert ei.value.profile == "gone"
        assert ei.value.retry_command == "hermes profile purge-identity gone"
        assert not ei.value.path.exists()
        assert isinstance(ei.value, RuntimeError)  # the CLI handler catches RuntimeError

        assert "hermes profile purge-identity gone" in capsys.readouterr().err
        check = SessionDB(tmp_path / ".hermes" / "state.db")
        try:
            # The CLI left the identity alone rather than racing the live owner.
            assert set(check.load_gateway_routing_entries(scope=scope)) == {
                "agent:gone:feishu:dm:chatA"}
        finally:
            check.close()

    def test_backend_scan_only_matches_this_profile(self, profile_env, monkeypatch):
        """The backend PID scan binds by --profile selector and skips self."""
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            # Backend bound to coder → matched.
            FakeProc(101, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
            # Interactive chat for coder → NOT a backend subcommand, skipped.
            FakeProc(102, ["python", "-m", "hermes_cli.main", "--profile", "coder", "chat"]),
            # Backend for a different profile → skipped.
            FakeProc(103, ["python", "-m", "hermes_cli.main", "--profile", "other", "serve"]),
            # This very process → skipped even if it matched.
            FakeProc(self_pid, ["python", "-m", "hermes_cli.main", "--profile", "coder", "serve"]),
        ]

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        pids = profiles._profile_bound_backend_pids("coder", profile_dir)
        assert pids == [101]

    def test_backend_scan_matches_shebang_exec_of_hermes_shim(self, profile_env, monkeypatch):
        """A `hermes` console-script shim spawned directly (e.g. Electron's
        findOnPath('hermes') resolution) reports argv[0] as the interpreter
        (python3) and argv[1] as the shim's path -- not "hermes" -- because
        the OS execs the shebang. The scanner must still recognize it so
        profile delete doesn't leave a zombie Desktop-spawned backend behind
        (issue: deleting a Desktop profile kept reappearing after relaunch).
        """
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python3", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            # Shebang-exec'd shim bound to coder → matched despite argv[0]
            # being the python interpreter, not "hermes".
            FakeProc(201, ["/usr/bin/python3", "/Users/x/.local/bin/hermes", "--profile", "coder", "serve",
                            "--host", "127.0.0.1", "--port", "0"]),
            # Same shape but a different profile → skipped.
            FakeProc(202, ["/usr/bin/python3", "/Users/x/.local/bin/hermes", "--profile", "other", "serve"]),
            # Non-hermes script run by python3 → skipped.
            FakeProc(203, ["/usr/bin/python3", "/Users/x/some_script.py", "--profile", "coder", "serve"]),
        ]

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        pids = profiles._profile_bound_backend_pids("coder", profile_dir)
        assert pids == [201]

    def test_backend_scan_rejects_unrelated_hermes_prefixed_script(self, profile_env, monkeypatch):
        """A user's own script that happens to start with "hermes" (e.g.
        hermes-notes.py, hermes-unrelated-tool) must NOT be misidentified as
        the console-script shim just because argv[0] is a python interpreter
        and argv[1]'s basename starts with "hermes" -- only the actual known
        console-script entry points (hermes, hermes-agent, hermes-acp) count.
        """
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python3", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            # Looks like the shim by prefix alone, but is the user's own
            # unrelated tool -- must be rejected, not killed by profile delete.
            FakeProc(301, ["/usr/bin/python3", "/Users/x/scripts/hermes-notes.py",
                            "--profile", "coder", "serve"]),
            FakeProc(302, ["/usr/bin/python3", "/Users/x/scripts/hermes-unrelated-tool",
                            "--profile", "coder", "serve"]),
        ]

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        pids = profiles._profile_bound_backend_pids("coder", profile_dir)
        assert pids == []

    def test_backend_scan_matches_all_known_console_script_shims(self, profile_env, monkeypatch):
        """The other two real console-script entry points (hermes-agent,
        hermes-acp -- see pyproject.toml [project.scripts]) must also be
        recognized via the shebang-exec path, not just the primary "hermes"
        shim.
        """
        create_profile("coder", no_alias=True)
        profile_dir = get_profile_dir("coder")

        class FakeProc:
            def __init__(self, pid, cmdline, username="me"):
                self.pid = pid
                self.info = {"pid": pid, "name": "python3", "username": username, "cmdline": cmdline}

            def parent(self):
                return None

            def username(self):
                return "me"

            def environ(self):
                return {}

        self_pid = os.getpid()
        procs = [
            FakeProc(401, ["/usr/bin/python3", "/Users/x/.local/bin/hermes-agent",
                            "--profile", "coder", "serve"]),
            FakeProc(402, ["/usr/bin/python3", "/Users/x/.local/bin/hermes-acp",
                            "--profile", "coder", "serve"]),
        ]

        fake_psutil = types.SimpleNamespace(
            process_iter=lambda attrs=None: iter(procs),
            Process=lambda pid=None: FakeProc(self_pid, []),
            NoSuchProcess=Exception,
            AccessDenied=Exception,
            ZombieProcess=Exception,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        pids = profiles._profile_bound_backend_pids("coder", profile_dir)
        assert set(pids) == {401, 402}


# ===================================================================
# TestListProfiles
# ===================================================================

class TestListProfiles:
    """Tests for list_profiles()."""

    def test_returns_default_when_no_named_profiles(self, profile_env):
        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert "default" in names

    def test_includes_named_profiles(self, profile_env):
        create_profile("alpha", no_alias=True)
        create_profile("beta", no_alias=True)
        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert "alpha" in names
        assert "beta" in names

    def test_lazy_skill_count_never_walks_in_the_polled_request(self, profile_env, monkeypatch):
        """Polled surfaces (``profiles.list`` RPC, ``GET /api/profiles``) must render
        ``skill_count`` without any skill-tree walk on the request thread; the count arrives
        from one background refresh per profile per recheck window (#114041). Control: the
        synchronous ``list_profiles()`` still walks and reports the fresh number."""
        import threading
        import tui_gateway.server as srv

        skills = profile_env / ".hermes" / "skills" / "cat"
        for i in range(3):
            (skills / f"s{i}").mkdir(parents=True)
            (skills / f"s{i}" / "SKILL.md").write_text("# s\n", encoding="utf-8")
        profiles._SKILL_COUNT_CACHE.clear()
        profiles._SKILL_COUNT_NEXT_CHECK.clear()

        walks: list[str] = []
        real_walk = profiles._walk_skill_count

        def spy(skills_dir):
            walks.append(threading.current_thread().name)
            return real_walk(skills_dir)

        monkeypatch.setattr(profiles, "_walk_skill_count", spy)

        def _rpc():
            return srv._methods["profiles.list"](1, {"include_sessions": False})["result"]["profiles"]

        first = _rpc()
        assert walks == [] or set(walks) == {"hermes-skill-count"}
        assert first[0]["skill_count"] in (0, 3)  # 0 until the refresh lands, never a stall
        for t in threading.enumerate():
            if t.name == "hermes-skill-count":
                t.join(timeout=10)
        assert walks == ["hermes-skill-count"]
        assert _rpc()[0]["skill_count"] == 3
        assert list_profiles(lazy_skill_count=True)[0].skill_count == 3
        assert walks == ["hermes-skill-count"]  # a second poll inside the window schedules nothing

        # GET /api/profiles (the router's own ``lazy_skill_count=True`` call) and the per-keystroke
        # ``@<profile>`` completion must be just as walk-free: same spy, still one background walk.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from hermes_cli.web_routers import profiles as profiles_router
        from tui_gateway import methods_complete
        app = FastAPI()
        app.include_router(profiles_router.router)
        # Cold cache: a synchronous list_profiles() in either caller would walk on the request thread.
        profiles._SKILL_COUNT_CACHE.clear()
        resp = TestClient(app).get("/api/profiles")
        assert resp.status_code == 200
        assert resp.json()["profiles"][0]["name"] == "default"
        assert walks == ["hermes-skill-count"]
        assert any(i["text"] == "@default" for i in methods_complete._profile_mention_items("def"))
        assert walks == ["hermes-skill-count"]

        # Control: the detail/CLI path counts synchronously on the caller's thread.
        profiles._SKILL_COUNT_CACHE.clear()
        assert list_profiles()[0].skill_count == 3
        assert walks[-1] == threading.current_thread().name

    def test_skill_count_survives_subtree_vanishing_mid_walk(self, profile_env, monkeypatch):
        """A skill removed while the tree is being counted (concurrent install/update) must
        degrade the count, not abort profile enumeration with ``FileNotFoundError``."""
        skills = profile_env / ".hermes" / "skills" / "cat"
        for i in range(4):
            (skills / f"s{i}" / "references").mkdir(parents=True)
            (skills / f"s{i}" / "SKILL.md").write_text("# s\n", encoding="utf-8")
        profiles._SKILL_COUNT_CACHE.clear()
        real_scandir = os.scandir

        class _Listing:
            """A pre-read scandir result (context manager + iterator, like the real one)."""
            def __init__(self, entries):
                self._it = iter(entries)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._it)

            def close(self):
                pass

        def vanishing_scandir(path=".", *args, **kwargs):
            with real_scandir(path, *args, **kwargs) as listing:
                entries = list(listing)
            if not isinstance(path, int) and os.fspath(path) == str(skills):
                # Listed, then gone before the walk descends into it.
                shutil.rmtree(skills / "s3", ignore_errors=True)
            return _Listing(entries)

        monkeypatch.setattr(os, "scandir", vanishing_scandir)
        assert profiles._count_skills(profile_env / ".hermes") == 3
        assert [p.name for p in list_profiles()] == ["default"]


# ===================================================================
# TestActiveProfile
# ===================================================================

class TestActiveProfile:
    """Tests for set_active_profile() / get_active_profile()."""


    def test_set_to_default_removes_file(self, profile_env):
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        set_active_profile("coder")
        active_path = tmp_path / ".hermes" / "active_profile"
        assert active_path.exists()

        set_active_profile("default")
        assert not active_path.exists()


# ===================================================================
# TestGetActiveProfileName
# ===================================================================

class TestGetActiveProfileName:
    """Tests for get_active_profile_name()."""


    def test_profile_path_returns_profile_name(self, profile_env, monkeypatch):
        tmp_path = profile_env
        create_profile("coder", no_alias=True)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        assert get_active_profile_name() == "coder"

    def test_custom_path_returns_default(self, profile_env, monkeypatch):
        """A custom HERMES_HOME (Docker, etc.) IS the default root."""
        tmp_path = profile_env
        custom = tmp_path / "some" / "other" / "path"
        custom.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        # With Docker-aware roots, a custom HERMES_HOME is the default —
        # not "custom".  The user is on the default profile of their
        # custom deployment.
        assert get_active_profile_name() == "default"


# ===================================================================
# TestResolveProfileEnv
# ===================================================================



# ===================================================================
# TestAliasCollision
# ===================================================================

class TestAliasCollision:
    """Tests for check_alias_collision()."""





    @pytest.mark.windows_only
    def test_windows_checks_bat_extension(self, profile_env):
        wrapper_dir = profile_env / ".local" / "bin"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        bat_path = wrapper_dir / "mybot.bat"
        bat_path.write_text("@echo off\r\nhermes -p mybot %*\r\n")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=str(bat_path),
            )
            result = check_alias_collision("mybot")
        assert result is None  # our own wrapper, safe to overwrite

    def test_traversal_alias_rejected_before_path_lookup(self, profile_env):
        """A path-traversal alias is rejected without ever shelling out to which/where."""
        with patch("subprocess.run") as mock_run:
            result = check_alias_collision("../../.bashrc")
        assert result is not None
        assert "invalid alias name" in result.lower()
        mock_run.assert_not_called()


# ===================================================================
# TestWrapperScript
# ===================================================================

class TestWrapperScript:
    """Tests for create_wrapper_script() and remove_wrapper_script()."""

    def test_creates_sh_on_posix(self, profile_env, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.shutil.which", lambda name: "/opt/hermes/bin/hermes")
        from hermes_cli.profiles import create_wrapper_script
        wrapper = create_wrapper_script("mybot")
        assert wrapper is not None
        assert wrapper.name == "mybot"
        content = wrapper.read_text()
        assert content.startswith("#!/bin/sh")
        assert "exec /opt/hermes/bin/hermes -p mybot" in content


    @pytest.mark.windows_only
    def test_remove_finds_bat_on_windows(self, profile_env):
        from hermes_cli.profiles import create_wrapper_script, remove_wrapper_script
        wrapper = create_wrapper_script("mybot")
        assert wrapper is not None
        assert wrapper.exists()
        removed = remove_wrapper_script("mybot")
        assert removed is True
        assert not wrapper.exists()






# ===================================================================
# TestWrapperScriptSecurity — path-traversal hardening
# ===================================================================

class TestWrapperScriptSecurity:
    """A crafted alias name must not escape the wrapper directory."""




    def test_create_wrapper_rejects_traversal(self, profile_env):
        sentinel = profile_env / ".bashrc"
        sentinel.write_text("keep", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid alias name"):
            create_wrapper_script("../../.bashrc", target="coder")
        # The traversal target was not touched.
        assert sentinel.read_text(encoding="utf-8") == "keep"

    def test_create_wrapper_rejects_absolute_path(self, profile_env, tmp_path):
        target = tmp_path / "abs-wrapper"
        with pytest.raises(ValueError, match="Invalid alias name"):
            create_wrapper_script(str(target))
        assert not target.exists()



# ===================================================================
# TestFindAliasForProfile — display-side reverse lookup
# ===================================================================

class TestFindAliasForProfile:
    """Tests for find_alias_for_profile() and alias display in list/show."""

    def test_profile_named_alias(self, profile_env):
        from hermes_cli.profiles import create_wrapper_script, find_alias_for_profile
        create_wrapper_script("steve")
        assert find_alias_for_profile("steve") == "steve"


    def test_ignores_unrelated_files(self, profile_env):
        # ~/.local/bin commonly holds unrelated binaries; they must not match.
        from hermes_cli.profiles import _get_wrapper_dir, find_alias_for_profile
        wrapper_dir = _get_wrapper_dir()
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        (wrapper_dir / "pip").write_text("#!/bin/sh\nexec python -m pip \"$@\"\n")
        assert find_alias_for_profile("steve") is None


    def test_list_profiles_surfaces_custom_alias(self, profile_env):
        from hermes_cli.profiles import (
            create_profile,
            create_wrapper_script,
            list_profiles,
        )
        create_profile("steve", no_alias=True)
        create_wrapper_script("qiaobusi", target="steve")
        info = next(p for p in list_profiles() if p.name == "steve")
        assert info.alias_name == "qiaobusi"
        assert info.alias_path is not None
        assert info.alias_path.name == "qiaobusi"


# ===================================================================
# TestRenameProfile
# ===================================================================

class TestRenameProfile:
    """Tests for rename_profile()."""

    def test_renames_directory(self, profile_env):
        tmp_path = profile_env
        create_profile("oldname", no_alias=True)
        old_dir = tmp_path / ".hermes" / "profiles" / "oldname"
        assert old_dir.is_dir()

        # Mock alias collision to avoid subprocess calls
        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            new_dir = rename_profile("oldname", "newname")

        assert not old_dir.is_dir()
        assert new_dir.is_dir()
        assert new_dir == tmp_path / ".hermes" / "profiles" / "newname"

    def test_renames_root_honcho_host_without_changing_ai_peer(self, profile_env):
        tmp_path = profile_env
        create_profile("ssi_health", no_alias=True)
        honcho_path = tmp_path / ".hermes" / "honcho.json"
        honcho_path.write_text(json.dumps({
            "hosts": {
                "hermes.ssi_health": {
                    "recallMode": "hybrid",
                    "writeFrequency": "async",
                    "sessionStrategy": "per-session",
                    "saveMessages": True,
                    "peerName": "user-peer",
                    "aiPeer": "ssi_health",
                    "workspace": "hermes",
                    "enabled": True,
                }
            }
        }))

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            rename_profile("ssi_health", "heimdall")

        cfg = json.loads(honcho_path.read_text())
        assert "hermes.ssi_health" not in cfg["hosts"]
        assert cfg["hosts"]["hermes_heimdall"]["aiPeer"] == "ssi_health"
        assert cfg["hosts"]["hermes_heimdall"]["peerName"] == "user-peer"

    def test_multiplexed_rename_unroutes_old_then_hot_serves_new(self, profile_env):
        """Under a live multiplexer the old name is tombstoned + unrouted BEFORE the directory
        moves and the new name is hot-served after, so a stale runtime mkdir of the old home is
        refused instead of resurrecting a ghost served profile (#109267)."""
        from hermes_constants import mkdir_under_hermes_home
        tmp_path = profile_env
        create_profile("oldname", no_alias=True)
        old_dir = tmp_path / ".hermes" / "profiles" / "oldname"
        new_dir = tmp_path / ".hermes" / "profiles" / "newname"

        calls = []

        def _record_notify(name):
            # Snapshot the world at each multiplexer signal to pin ordering.
            calls.append((name, old_dir.exists(), new_dir.exists(), profiles.named_profile_is_deleted(old_dir)))
            if name == "oldname" and old_dir.exists():
                # A still-live component of the multiplexer writing into the old home mid-teardown.
                with pytest.raises(FileNotFoundError):
                    mkdir_under_hermes_home(old_dir / "logs")

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=True), \
             patch("hermes_cli.profiles._notify_multiplexer", side_effect=_record_notify):
            rename_profile("oldname", "newname")

        # (name, old_exists, new_exists, old_tombstoned): unroute first, hot-serve last.
        assert calls[0] == ("oldname", True, False, True)
        assert calls[-1] == ("newname", False, True, False)
        assert not old_dir.exists() and new_dir.is_dir()
        assert not profiles.named_profile_is_deleted(old_dir)  # a future 'oldname' is not born deleted

    def test_unmultiplexed_rename_does_not_signal_multiplexer(self, profile_env):
        """No live multiplexer → rename must neither tombstone nor ping (single-profile installs)."""
        tmp_path = profile_env
        create_profile("oldname", no_alias=True)
        old_dir = tmp_path / ".hermes" / "profiles" / "oldname"

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=False), \
             patch("hermes_cli.profiles._notify_multiplexer") as notify:
            new_dir = rename_profile("oldname", "newname")

        notify.assert_not_called()
        assert not (tmp_path / ".hermes" / "profiles" / ".deleted").exists()
        assert not old_dir.exists() and new_dir.is_dir()

    def test_rename_migrates_session_identity_without_live_gateway(self, profile_env):
        """No live gateway → the CLI performs the durable rekey itself so a renamed profile's session
        keys / profile_name / routing rows follow the new name (else inbound events on the old name's
        chats resolve to a nonexistent profile and flood errors.log)."""
        from hermes_state import SessionDB
        tmp_path = profile_env
        create_profile("oldname", no_alias=True)
        old_dir = tmp_path / ".hermes" / "profiles" / "oldname"
        # Seed a session owned by the old profile in the profile's own store + the root routing index.
        pdb = SessionDB(old_dir / "state.db")
        pdb.create_session(
            "sess1", "feishu", session_key="agent:oldname:feishu:dm:chatA",
            profile_name="oldname", chat_id="chatA", chat_type="dm")
        pdb.close()
        root_db = SessionDB(tmp_path / ".hermes" / "state.db")
        root_db.save_gateway_routing_entry(
            "agent:oldname:feishu:dm:chatA",
            json.dumps({"session_key": "agent:oldname:feishu:dm:chatA", "session_id": "sess1",
                        "origin": {"platform": "feishu", "chat_id": "chatA", "profile": "oldname"}}),
            scope=str(tmp_path / ".hermes" / "sessions"))
        root_db.close()

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
            rename_profile("oldname", "newname")

        new_dir = tmp_path / ".hermes" / "profiles" / "newname"
        moved_db = SessionDB(new_dir / "state.db")
        row = moved_db._read_one(
            "SELECT session_key, profile_name FROM sessions WHERE id = ?", ("sess1",))
        assert row["session_key"] == "agent:newname:feishu:dm:chatA"
        assert row["profile_name"] == "newname"
        moved_db.close()
        root_db2 = SessionDB(tmp_path / ".hermes" / "state.db")
        routing = root_db2.load_gateway_routing_entries(
            scope=str(tmp_path / ".hermes" / "sessions"))
        assert "agent:oldname:feishu:dm:chatA" not in routing
        assert "agent:newname:feishu:dm:chatA" in routing
        root_db2.close()

    def test_rename_delegates_identity_migration_to_live_gateway(self, profile_env):
        """Under a live multiplexer the CLI must NOT rewrite the routing DB directly (the gateway holds
        it in memory and would clobber the write); it delegates to the control verb instead."""
        tmp_path = profile_env
        create_profile("oldname", no_alias=True)

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=True), \
             patch("hermes_cli.profiles._notify_multiplexer"), \
             patch("gateway.control_socket.migrate_gateway_profile_identity",
                   return_value={"ok": True, "rekeyed": 1, "db": {}}) as verb, \
             patch("hermes_state_registry.acquire") as acquire:
            rename_profile("oldname", "newname")

        # Delegated to the gateway; the CLI's own durable-rewrite branch never ran.
        assert verb.call_count == 1
        assert verb.call_args.args[1:] == ("oldname", "newname")
        acquire.assert_not_called()


    def test_live_gateway_failure_does_not_rewrite_db_directly(self, profile_env, capsys):
        create_profile("oldname", no_alias=True)
        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=True), \
             patch("hermes_cli.profiles._notify_multiplexer"), \
             patch("gateway.control_socket.migrate_gateway_profile_identity", return_value=None), \
             patch("hermes_state_registry.acquire") as acquire:
            rename_profile("oldname", "newname")
        acquire.assert_not_called()
        assert "Restart the gateway" in capsys.readouterr().err

    def test_migrate_identity_command_repairs_a_failed_live_migration(self, profile_env, capsys):
        """The failed-live-migration end state must be recoverable: `hermes profile
        migrate-identity <old> <new>` rekeys the durable rows once no gateway holds the store, and
        is idempotent (a second run has nothing left to rekey but still succeeds)."""
        from hermes_cli.profile_cmd import cmd_profile
        from hermes_state import SessionDB
        from argparse import Namespace
        tmp_path = profile_env
        create_profile("oldname", no_alias=True)
        old_dir = tmp_path / ".hermes" / "profiles" / "oldname"
        pdb = SessionDB(old_dir / "state.db")
        pdb.create_session(
            "sess1", "feishu", session_key="agent:oldname:feishu:dm:chatA",
            profile_name="oldname", chat_id="chatA", chat_type="dm")
        pdb.close()
        root_db = SessionDB(tmp_path / ".hermes" / "state.db")
        root_db.save_gateway_routing_entry(
            "agent:oldname:feishu:dm:chatA",
            json.dumps({"session_key": "agent:oldname:feishu:dm:chatA", "session_id": "sess1",
                        "origin": {"platform": "feishu", "chat_id": "chatA", "profile": "oldname"}}),
            scope=str(tmp_path / ".hermes" / "sessions"))
        root_db.close()

        # Rename under a live multiplexer whose control verb answers nothing: the CLI warns and
        # leaves the (in-memory-owned) store alone, so the rows still name the old profile.
        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
             patch("hermes_cli.profiles._live_default_multiplexer", return_value=True), \
             patch("hermes_cli.profiles._notify_multiplexer"), \
             patch("gateway.control_socket.migrate_gateway_profile_identity", return_value=None):
            rename_profile("oldname", "newname")
        assert "hermes profile migrate-identity oldname newname" in capsys.readouterr().err

        # Gateway restarted/stopped → the retry command repairs both stores.
        with patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
            cmd_profile(Namespace(profile_action="migrate-identity",
                                  old_name="oldname", new_name="newname"))
            assert "✓ Session/routing identity migrated" in capsys.readouterr().out
            # Idempotent: nothing left to rekey, still a success.
            cmd_profile(Namespace(profile_action="migrate-identity",
                                  old_name="oldname", new_name="newname"))

        moved_db = SessionDB(tmp_path / ".hermes" / "profiles" / "newname" / "state.db")
        row = moved_db._read_one(
            "SELECT session_key, profile_name FROM sessions WHERE id = ?", ("sess1",))
        assert row is not None
        assert row["session_key"] == "agent:newname:feishu:dm:chatA"
        assert row["profile_name"] == "newname"
        moved_db.close()
        root_db2 = SessionDB(tmp_path / ".hermes" / "state.db")
        routing = root_db2.load_gateway_routing_entries(
            scope=str(tmp_path / ".hermes" / "sessions"))
        assert "agent:oldname:feishu:dm:chatA" not in routing
        assert "agent:newname:feishu:dm:chatA" in routing
        root_db2.close()

    def test_rename_records_previous_name(self, profile_env):
        create_profile("oldname", no_alias=True)

        # Mock alias collision to avoid subprocess calls
        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            new_dir = rename_profile("oldname", "newname")

        # The rename history is recorded in the new profile's metadata ...
        assert profiles.read_profile_meta(new_dir)["previous_names"] == ["oldname"]
        # ... and surfaces through list_profiles (the gateway's profiles.list
        # source), so Bot Mode group chats can re-link stale member handles.
        info = next(p for p in list_profiles() if p.name == "newname")
        assert info.previous_names == ["oldname"]

    def test_rename_accumulates_previous_names(self, profile_env):
        create_profile("firstname", no_alias=True)

        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            rename_profile("firstname", "secondname")
            rename_profile("secondname", "thirdname")

        info = next(p for p in list_profiles() if p.name == "thirdname")
        assert info.previous_names == ["firstname", "secondname"]

    def test_rename_succeeds_when_previous_name_write_fails(self, profile_env):
        create_profile("oldname", no_alias=True)

        # The history write is best-effort: it must never fail the rename.
        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
             patch("hermes_cli.profiles.write_profile_meta", side_effect=OSError("disk full")):
            new_dir = rename_profile("oldname", "newname")

        assert new_dir.is_dir()



class TestExportImport:
    """Tests for export_profile() / import_profile()."""







    # ---------------------------------------------------------------
    # Default profile export / import
    # ---------------------------------------------------------------


    def test_export_default_includes_profile_data(self, profile_env, tmp_path):
        """Profile data files end up in the archive (credentials excluded)."""
        # Write through HERMES_HOME, not get_profile_dir("default"): the latter resolves to the
        # OPERATOR's real install whenever basetest sits inside it, so this test used to
        # overwrite the live config.yaml / .env / MEMORY.md with its fixtures.
        default_dir = profile_env / ".hermes"
        (default_dir / "config.yaml").write_text("model: test")
        (default_dir / ".env").write_text("KEY=val")
        (default_dir / "SOUL.md").write_text("Be nice.")
        mem_dir = default_dir / "memories"
        mem_dir.mkdir(exist_ok=True)
        (mem_dir / "MEMORY.md").write_text("remember this")

        output = tmp_path / "export" / "default.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        export_profile("default", str(output))

        with tarfile.open(str(output), "r:gz") as tf:
            names = tf.getnames()

        assert "default/config.yaml" in names
        assert "default/.env" not in names  # credentials excluded
        assert "default/SOUL.md" in names
        assert "default/memories/MEMORY.md" in names


    def test_export_default_handles_broken_symlinks(self, profile_env, tmp_path):
        """Broken symlinks inside allowed artifacts are preserved, not crashed (#58394).

        ``shutil.copytree``'s default is ``symlinks=False``, which follows
        symlinks and crashes on broken ones. Use ``symlinks=True`` so stale
        symlinks inside *allowed* artifacts (e.g. ``skills/``) survive as
        symlinks; the link and its target are both retained.
        """
        # Same reason as above: never resolve the operator's real default home from a test.
        default_dir = profile_env / ".hermes"
        (default_dir / "config.yaml").write_text("ok")
        # Place broken symlink *inside* the allowed ``skills/`` tree so the
        # root-level allow-list passes the directory through; the
        # symlinks=True flag must then preserve the link instead of
        # following and crashing.
        broken_dir = default_dir / "skills" / "with-broken-links"
        broken_dir.mkdir(parents=True)
        (broken_dir / "broken_link").symlink_to("/nonexistent/path")
        # Valid symlink for comparison
        (broken_dir / "valid_target.txt").write_text("real data")
        (broken_dir / "valid_link").symlink_to(
            broken_dir / "valid_target.txt"
        )

        output = tmp_path / "export" / "default.tar.gz"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = export_profile("default", str(output))

        assert result.exists()
        with tarfile.open(str(result), "r:gz") as tf:
            names = set(tf.getnames())
        # Allowed artifact survived
        assert any(n.endswith("config.yaml") for n in names)
        # Broken symlink inside an allowed dir was preserved as a symlink
        # (without crashing) — tar entry name recorded as the link path.
        assert any(
            "with-broken-links/broken_link" in n for n in names
        ), (
            f"broken_link should survive; tarfile names: {sorted(names)[:30]}"
        )
        # Valid symlink + target also kept
        assert any("valid_link" in n for n in names)
        assert any("valid_target.txt" in n for n in names)




# ===================================================================
# TestProfileIsolation
# ===================================================================

class TestProfileIsolation:
    """Verify that two profiles have completely separate paths."""

    def test_separate_config_paths(self, profile_env):
        create_profile("alpha", no_alias=True)
        create_profile("beta", no_alias=True)
        alpha_dir = get_profile_dir("alpha")
        beta_dir = get_profile_dir("beta")
        assert alpha_dir / "config.yaml" != beta_dir / "config.yaml"
        assert str(alpha_dir) not in str(beta_dir)


# ===================================================================
# TestGetProfilesRoot / TestGetDefaultHermesHome (internal helpers)
# ===================================================================

class TestInternalHelpers:
    """Tests for _get_profiles_root() and _get_default_hermes_home()."""



    def test_default_hermes_home_docker(self, tmp_path, monkeypatch):
        """In Docker, _get_default_hermes_home() returns HERMES_HOME itself."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        home = _get_default_hermes_home()
        assert home == docker_home



    def test_create_profile_docker(self, tmp_path, monkeypatch):
        """Profile created in Docker lands under HERMES_HOME/profiles/."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        result = create_profile("orchestrator", no_alias=True)
        expected = docker_home / "profiles" / "orchestrator"
        assert result == expected
        assert expected.is_dir()




# ===================================================================
# TestWriteProfileMetaDurability
# ===================================================================

class TestWriteProfileMetaDurability:
    """``profile.yaml`` must survive an interrupted ``write_profile_meta``.

    ``write_profile_meta`` is a read-modify-write whose docstring promises
    "unspecified fields preserve existing values".  Its read half swallows
    any parse error and falls back to ``{}``, so a truncated profile.yaml is
    not transient corruption — the *next* call reads ``{}`` and silently and
    permanently drops every field the caller did not explicitly pass.
    """

    @staticmethod
    def _seed(tmp_path):
        profile_dir = tmp_path / "coder"
        profile_dir.mkdir()
        profiles.write_profile_meta(
            profile_dir, description="Curated by hand", description_auto=False
        )
        return profile_dir

    @staticmethod
    def _interrupted_write(profile_dir):
        """Run a ``write_profile_meta`` whose serialization fails mid-call.

        The pre-fix code called ``yaml.safe_dump``; ``utils.atomic_yaml_write``
        calls ``yaml.dump``.  Breaking both keeps this serializer-agnostic, so
        it measures durability rather than the choice of entry point.  A
        scoped ``MonkeyPatch.context`` is used instead of the fixture so the
        patch is reverted immediately, without touching the session-wide env
        isolation that shares the function-scoped ``monkeypatch`` instance.
        """
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated interruption mid-write")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(yaml, "safe_dump", _boom)
            mp.setattr(yaml, "dump", _boom)
            with pytest.raises(RuntimeError):
                profiles.write_profile_meta(profile_dir, description_auto=True)

    def test_failed_write_leaves_existing_file_intact(self, tmp_path):
        profile_dir = self._seed(tmp_path)
        path = profile_dir / "profile.yaml"
        before = path.read_text(encoding="utf-8")
        assert "Curated by hand" in before

        self._interrupted_write(profile_dir)

        # A truncating open() destroys the file before the dump ever runs.
        assert path.read_text(encoding="utf-8") == before

    def test_failed_write_does_not_silently_drop_unspecified_fields(self, tmp_path):
        profile_dir = self._seed(tmp_path)

        self._interrupted_write(profile_dir)

        # The user retries; this call does not pass a description, so the
        # documented merge contract requires the existing one to survive.
        profiles.write_profile_meta(profile_dir, description_auto=True)
        meta = profiles.read_profile_meta(profile_dir)
        assert meta["description"] == "Curated by hand"
        assert meta["description_auto"] is True

    def test_emoji_description_is_written_as_real_utf8(self, tmp_path):
        """Astral-plane chars must not become ``\\UXXXXXXXX`` escapes (#51356).

        Supersedes #51808, which fixed this symptom alone by adding
        ``allow_unicode=True``; ``atomic_yaml_write`` sets it internally.
        """
        profile_dir = tmp_path / "wizard"
        profile_dir.mkdir()
        profiles.write_profile_meta(profile_dir, description="Code wizard 🧙 ✨")

        raw = (profile_dir / "profile.yaml").read_text(encoding="utf-8")
        assert "\\U" not in raw
        assert "🧙" in raw
        assert profiles.read_profile_meta(profile_dir)["description"] == "Code wizard 🧙 ✨"

    def test_symlinked_profile_yaml_survives_the_write(self, tmp_path):
        """Guard on the conversion, not a behavior change.

        Dotfile managers (chezmoi/stow) symlink profile.yaml into a tracked
        repo.  ``open(path, "w")`` wrote through the link; a naive
        ``os.replace`` would detach it.  ``atomic_replace`` resolves the link
        first (GitHub #16743), so the link must still be intact afterwards.
        """
        profile_dir = tmp_path / "linked"
        profile_dir.mkdir()
        real_dir = tmp_path / "dotfiles"
        real_dir.mkdir()
        real = real_dir / "profile.yaml"
        real.write_text("description: from dotfiles\n", encoding="utf-8")
        (profile_dir / "profile.yaml").symlink_to(real)

        profiles.write_profile_meta(profile_dir, description="updated")

        assert (profile_dir / "profile.yaml").is_symlink()
        assert "updated" in real.read_text(encoding="utf-8")
        assert [p.name for p in profile_dir.iterdir() if p.name.endswith(".tmp")] == []


# ===================================================================
# Edge cases and additional coverage
# ===================================================================

class TestEdgeCases:
    """Additional edge-case tests."""





    def test_gateway_running_check_falls_back_to_runtime_state(self, profile_env):
        """A live gateway whose PID-file/lock check fails closed (separate-process
        reader, e.g. the dashboard s6 service in Docker) is still detected via the
        profile's gateway_state.json validated against the live process table.

        Regression: the Profiles view used to show "Gateway stopped" while the
        sidebar (which already has this fallback) showed "Gateway running" for the
        same live gateway. See get_running_pid() short-circuiting on an
        unheld runtime lock before it inspects the PID record.
        """
        import os
        import gateway.status as gw_status
        from hermes_cli.profiles import _check_gateway_running

        tmp_path = profile_env
        default_home = tmp_path / ".hermes"
        default_home.mkdir(parents=True, exist_ok=True)

        # Write a realistic gateway_state.json pointing at THIS live process with
        # a gateway-shaped argv, so get_runtime_status_running_pid validates it.
        live_pid = os.getpid()
        (default_home / "gateway_state.json").write_text(
            json.dumps(
                {
                    "pid": live_pid,
                    "kind": "hermes-gateway",
                    "argv": ["hermes", "gateway", "run"],
                    "start_time": gw_status._get_process_start_time(live_pid),
                    "gateway_state": "running",
                    "active_agents": 0,
                }
            ),
            encoding="utf-8",
        )

        # Primary pid-file/lock check returns None (no lock held by this reader),
        # exactly as it does for a separate-process dashboard. The fallback must
        # then read the state file and confirm the gateway is alive by checking
        # the recorded PID's live command line. In the real separate-process
        # scenario that PID belongs to the live gateway, so mock its command
        # line to a bare ``gateway run`` (this is the default/root home, which
        # runs the gateway with no profile flag).
        with patch("gateway.status.get_running_pid", return_value=None), patch(
            "gateway.status._read_process_cmdline",
            return_value="hermes gateway run --replace",
        ):
            assert _check_gateway_running(default_home) is True







    def test_clone_from_named_profile(self, profile_env):
        """Clone config from a named (non-default) profile."""
        tmp_path = profile_env
        # Create source profile with config
        source_dir = create_profile("source", no_alias=True)
        (source_dir / "config.yaml").write_text("model: cloned")
        (source_dir / ".env").write_text("SECRET=yes")

        target_dir = create_profile(
            "target", clone_from="source", clone_config=True, no_alias=True,
        )
        cloned_config = yaml.safe_load((target_dir / "config.yaml").read_text())
        assert cloned_config["_config_version"] == DEFAULT_CONFIG["_config_version"]
        assert cloned_config["model"] == "cloned"
        assert (target_dir / ".env").read_text().strip() == "SECRET=yes"



class TestProfilesToServe:
    """profiles_to_serve(multiplex) — the gateway's profile-enumeration chokepoint."""


    def test_off_returns_only_active_named(self, profile_env, monkeypatch):
        # A named profile's gateway runs with HERMES_HOME pointing at the
        # profile dir; get_active_profile_name() infers the name from there.
        create_profile("coder", no_alias=True)
        monkeypatch.setenv("HERMES_HOME", str(get_profile_dir("coder")))
        serve = profiles_to_serve(multiplex=False)
        assert len(serve) == 1
        assert serve[0][0] == "coder"
        assert serve[0][1] == get_profile_dir("coder")

    def test_on_returns_default_plus_all_named(self, profile_env):
        create_profile("coder", no_alias=True)
        create_profile("writer", no_alias=True)
        serve = dict(profiles_to_serve(multiplex=True))
        assert set(serve) == {"default", "coder", "writer"}
        assert serve["default"] == _get_default_hermes_home()
        assert serve["coder"] == get_profile_dir("coder")


# ---------------------------------------------------------------------------
# resolve_profile_env spelling preservation (#82581 junction follow-up)
# ---------------------------------------------------------------------------


class TestResolveProfileEnvSpelling:
    """resolve_profile_env() keeps the configured HERMES_HOME spelling as

    the launch root (junction installs) while preserving the pre-existing
    profile-path handling and existence/validation semantics.
    """

    def test_resolution_matrix_preserves_configured_spelling(self, monkeypatch, tmp_path):
        """Resolution matrix over the four pre-existing invariants: root env
        -> <root>/profiles/<name>; profile-shaped env -> <root>/profiles/<name>
        with no nesting; profile-shaped env + default -> <root>; custom roots
        never fall back to the platform default.
        """
        root = tmp_path / "configured-root"
        custom = tmp_path / "custom-hermes"
        for profile_dir in (root / "profiles" / "beta", root / "profiles" / "coder", custom / "profiles" / "beta"):
            profile_dir.mkdir(parents=True)
            (profile_dir / "config.yaml").write_text("{}\n")  # identity marker: a bare dir does not resolve
        cases = [
            (root, "coder", root / "profiles" / "coder"),
            (root / "profiles" / "alpha", "beta", root / "profiles" / "beta"),
            (root / "profiles" / "alpha", "default", root),
            (custom, "beta", custom / "profiles" / "beta"),
        ]
        for env_home, profile, expected in cases:
            monkeypatch.setenv("HERMES_HOME", str(env_home))
            assert Path(resolve_profile_env(profile)) == expected

    def test_missing_named_profile_still_raises(self, monkeypatch, tmp_path):
        root = tmp_path / "configured-root"
        monkeypatch.setenv("HERMES_HOME", str(root))
        with pytest.raises(FileNotFoundError):
            resolve_profile_env("nope")

    def test_unset_env_falls_back_to_default_root(self, monkeypatch):
        # No HERMES_HOME: the platform default root applies (existing contract).
        monkeypatch.delenv("HERMES_HOME", raising=False)
        assert Path(resolve_profile_env("default")) == _get_default_hermes_home()


# ===================================================================
# TestCloneAllExcludesRuntimeTrees
# ===================================================================

class TestCloneAllExcludesRuntimeTrees:
    """``--clone-all`` from the default profile must not copy the machine-scoped
    runtime trees the local-models flow puts under ``~/.hermes``: ``models/``
    (GGUF weights, tens of GB), ``runtimes/`` (llama.cpp binaries) and ``node/``
    (managed Node). ``backup.py`` already excludes exactly these; the clone-all
    ignore list had not followed.
    """

    RUNTIME_TREES = ("models", "runtimes", "node")

    def _seed(self, home):
        (home / "models").mkdir(); (home / "models" / "big.gguf").write_bytes(b"\0" * 64)
        (home / "runtimes" / "llamacpp" / "bin").mkdir(parents=True)
        (home / "runtimes" / "llamacpp" / "bin" / "llama-server").write_text("bin")
        (home / "node" / "bin").mkdir(parents=True)
        (home / "node" / "bin" / "node").write_text("bin")
        (home / "skills" / "greet").mkdir(parents=True)
        (home / "skills" / "greet" / "SKILL.md").write_text("# greet\n")
        (home / "config.yaml").write_text("model: test\n")

    def test_ignore_drops_runtime_trees_only_at_the_default_root(self, profile_env):
        default_home = profile_env / ".hermes"
        self._seed(default_home)
        # a skill that happens to carry a nested models/ dir is user data
        (default_home / "skills" / "greet" / "models").mkdir()
        ignore = _clone_all_copytree_ignore(default_home)

        at_root = ignore(str(default_home), ["models", "runtimes", "node", "skills", "config.yaml"])
        nested = ignore(str(default_home / "skills" / "greet"), ["models", "SKILL.md"])

        assert set(at_root) == set(self.RUNTIME_TREES)
        assert not nested

        # Gated on the default profile: a named profile that really has a
        # models/ dir of its own must not have it dropped when used as source.
        source = create_profile("source", no_alias=True)
        for tree in self.RUNTIME_TREES:
            (source / tree).mkdir()
        assert not _clone_all_copytree_ignore(source)(str(source), [*self.RUNTIME_TREES, "SOUL.md"])

    def test_runtime_trio_is_one_constant_shared_with_backup(self):
        """backup's exclusion list and the clone-all root gate must be built from the same
        constant; two literals drifting apart is how the models/ copy of #111718 crept in."""
        from hermes_cli import backup, profiles
        from hermes_constants import LOCAL_RUNTIME_ROOT_DIRS
        assert LOCAL_RUNTIME_ROOT_DIRS == frozenset(self.RUNTIME_TREES)
        assert backup._EXCLUDED_ROOT_DIRS is LOCAL_RUNTIME_ROOT_DIRS
        assert LOCAL_RUNTIME_ROOT_DIRS <= profiles._CLONE_ALL_DEFAULT_EXCLUDE_ROOT

    def test_clone_all_from_default_skips_runtime_trees_but_keeps_the_rest(self, profile_env):
        default_home = profile_env / ".hermes"
        self._seed(default_home)

        clone = create_profile("clone", clone_all=True, no_alias=True)

        for name in self.RUNTIME_TREES:
            assert not (clone / name).exists(), name
        assert (clone / "skills" / "greet" / "SKILL.md").is_file()
        assert (clone / "config.yaml").is_file()
