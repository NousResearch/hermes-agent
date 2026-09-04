"""Tests for cmd_update — branch fallback when remote branch doesn't exist."""

import hashlib
import subprocess
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest

from hermes_cli.main import cmd_update, PROJECT_ROOT
from hermes_cli import main_web_build
from hermes_cli import main_install_repair
from hermes_cli import update_cmd


def _make_run_side_effect(branch="main", verify_ok=True, commit_count="0"):
    """Build a side_effect function for subprocess.run that simulates git commands."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        # git rev-parse --abbrev-ref HEAD  (get current branch)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        # git rev-parse --verify origin/{branch}  (check remote branch exists)
        if "rev-parse" in joined and "--verify" in joined:
            rc = 0 if verify_ok else 128
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        # git rev-list HEAD..origin/{branch} --count
        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        # Fallback: return a successful CompletedProcess with empty stdout
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


@pytest.fixture
def mock_args():
    return SimpleNamespace()


# ---------------------------------------------------------------------------
# Managed-uv compatibility for tests that patch shutil.which
# ---------------------------------------------------------------------------
# The production code now uses ``ensure_uv()`` / ``update_managed_uv()``
# instead of ``shutil.which("uv")``.  Many tests in this file patch
# ``shutil.which`` to control whether uv is "available" — these autouse
# fixtures make the managed_uv functions delegate to the patched
# ``shutil.which`` so the existing test setup keeps working without
# per-test changes.
@pytest.fixture(autouse=True)
def _patch_managed_uv(request):
    """Make managed_uv helpers follow shutil.which mocking in tests."""
    import shutil

    # resolve_uv delegates to shutil.which("uv") so that test patches
    # on shutil.which flow through naturally.
    def _fake_resolve_uv():
        return shutil.which("uv")

    def _fake_ensure_uv(**_kwargs):
        return shutil.which("uv")

    def _fake_update_managed_uv(**_kwargs):
        return None  # never actually self-update in tests

    with patch("hermes_cli.managed_uv.resolve_uv", side_effect=_fake_resolve_uv), \
         patch("hermes_cli.managed_uv.ensure_uv", side_effect=_fake_ensure_uv), \
         patch("hermes_cli.managed_uv.update_managed_uv", side_effect=_fake_update_managed_uv), \
         patch(
             "hermes_cli.update_cmd._post_update_sqlite_runtime_status",
             return_value=(True, None),
         ):
        yield


@pytest.fixture(autouse=True)
def _patch_gateway_discovery():
    """Keep cmd_update's gateway auto-restart phase off this machine's gateways.

    The restart phase used to swallow every exception at debug level, so these
    end-to-end tests never noticed it touching real gateway discovery. Since
    the phase is surfaced (#78574: an aborted restart now fails the update),
    an unmocked ``find_gateway_pids`` on a box with a live gateway reaches the
    conftest live-system guard and turns into a spurious ``sys.exit(1)``.
    Discovery returning nothing makes the phase a clean no-op for every test
    in this module (none of them assert on gateway restarts).
    """
    with patch("hermes_cli.gateway.find_gateway_pids", return_value=[]), \
         patch("hermes_cli.gateway.supports_systemd_services", return_value=False), \
         patch("hermes_cli.gateway.find_profile_gateway_processes", return_value=[]):
        yield


class TestCmdUpdateNpmLockfileCache:
    @staticmethod
    def _cache_file(hermes_root, project_root):
        cache_key = hashlib.sha256(str(project_root).encode()).hexdigest()[:12]
        return hermes_root / f".npm_lock_hash_{cache_key}"



    def test_record_npm_lockfile_hash(self, tmp_path, monkeypatch):
        from hermes_cli import main as hm

        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}')

        update_cmd._record_npm_lockfile_hash(tmp_path)

        assert (
            self._cache_file(tmp_path, tmp_path).read_text()
            == update_cmd._npm_manifests_digest()
        )

    def test_package_json_only_edit_defeats_skip(self, tmp_path, monkeypatch):
        """Reviewer scenario (#61580): dev edits package.json WITHOUT running
        npm — lockfile unchanged. `hermes update` must still install (the
        npm-install fallback is what syncs node_modules in that state)."""
        from hermes_cli import main as hm

        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}')
        (tmp_path / "package.json").write_text('{"dependencies": {}}')
        (tmp_path / "node_modules").mkdir()
        update_cmd._record_npm_lockfile_hash(tmp_path)
        assert hm._npm_lockfile_changed(tmp_path) is False

        (tmp_path / "package.json").write_text(
            '{"dependencies": {"left-pad": "^1.0.0"}}'
        )
        assert hm._npm_lockfile_changed(tmp_path) is True







    def test_update_uses_one_shared_npm_cache_across_profiles(
        self, tmp_path, monkeypatch
    ):
        """The npm cache describes checkout-global node_modules, not a profile."""
        from hermes_cli import main as hm
        import hermes_constants

        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "package.json").write_text("{}")
        shared_root = tmp_path / ".hermes"
        named_profile = shared_root / "profiles" / "work"
        named_profile.mkdir(parents=True)

        monkeypatch.setattr(hm, "PROJECT_ROOT", checkout)
        monkeypatch.setattr(hermes_constants.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(
            hermes_constants, "find_node_executable", lambda _name: "/usr/bin/npm"
        )

        cache_roots = []
        with patch.object(
            hm,
            "_npm_lockfile_changed",
            side_effect=lambda root: cache_roots.append(root) or False,
        ):
            monkeypatch.setenv("HERMES_HOME", str(shared_root))
            update_cmd._update_node_dependencies()

            monkeypatch.setenv("HERMES_HOME", str(named_profile))
            update_cmd._update_node_dependencies()

        assert cache_roots == [shared_root, shared_root]


class TestCmdUpdateTermuxUvBootstrap:
    """Regression tests for Termux-specific uv bootstrap behavior."""

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_termux_uv_bootstrap_uses_binary_only_install(
        self, mock_run, _mock_which, monkeypatch
    ):
        from hermes_cli import main as hm

        mock_run.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        monkeypatch.setattr(hm, "_is_termux_env", lambda env=None: True)

        uv_bin = update_cmd._ensure_uv_for_termux(["/termux/python", "-m", "pip"])

        assert uv_bin is None
        assert mock_run.call_count == 1
        assert mock_run.call_args.args[0] == [
            "/termux/python",
            "-m",
            "pip",
            "install",
            "uv",
            "--only-binary",
            ":all:",
        ]
        assert mock_run.call_args.kwargs["cwd"] == PROJECT_ROOT
        assert mock_run.call_args.kwargs["check"] is False

    @patch("subprocess.run")
    def test_termux_reuses_existing_path_uv_without_pip(self, mock_run, monkeypatch):
        """A uv already on PATH (e.g. ``pkg install uv``) is reused before pip runs."""
        from hermes_cli import main as hm

        pkg_uv = "/data/data/com.termux/files/usr/bin/uv"
        monkeypatch.setattr(hm, "_is_termux_env", lambda env=None: True)
        # Production resolve_uv only checks $HERMES_HOME/bin/uv; model an empty
        # managed dir so the PATH probe is what surfaces the packaged uv.
        monkeypatch.setattr("hermes_cli.managed_uv.resolve_uv", lambda: None)
        monkeypatch.setattr("shutil.which", lambda name: pkg_uv if name == "uv" else None)

        uv_bin = update_cmd._ensure_uv_for_termux(["/termux/python", "-m", "pip"])

        assert uv_bin == pkg_uv
        mock_run.assert_not_called()


class TestUpdateManagedPythonEnvIsolation:
    """Regression for the uv-env isolation fix (third-party UV_PYTHON_INSTALL_DIR
    must not hijack the update's pip install).

    The update path builds uv_env via managed_python_env() (drops
    VIRTUAL_ENV/PYTHONPATH/UV_PYTHON, pins UV_MANAGED_PYTHON=1 + UV_NO_CONFIG=1,
    forces UV_PYTHON_INSTALL_DIR to .hermes-runtime/python), then re-points
    VIRTUAL_ENV at this install's venv. These tests lock that contract in.
    """

    def test_managed_env_drops_third_party_uv_install_dir(self):
        from hermes_cli.managed_uv import managed_python_env

        poisoned = {
            "UV_PYTHON_INSTALL_DIR": r"C:\WorkBuddy\python",
            "UV_PYTHON": r"C:\WorkBuddy\python\python.exe",
            "UV_SYSTEM_PYTHON": "1",
            "UV_NO_MANAGED_PYTHON": "1",
            "VIRTUAL_ENV": r"C:\Some\Other\venv",
            "PYTHONPATH": r"C:\Some\site-packages",
        }
        env = managed_python_env()

        # Third-party UV_PYTHON_INSTALL_DIR must not survive into the env.
        assert env.get("UV_PYTHON_INSTALL_DIR", "") != r"C:\WorkBuddy\python"
        assert "WorkBuddy" not in env.get("UV_PYTHON_INSTALL_DIR", "")
        # Managed pins are set; the hijack guards are explicitly cleared.
        assert env.get("UV_MANAGED_PYTHON") == "1"
        assert env.get("UV_NO_CONFIG") == "1"
        assert env.get("UV_PYTHON") is None
        assert env.get("UV_SYSTEM_PYTHON") is None
        assert env.get("UV_NO_MANAGED_PYTHON") is None
        assert env.get("VIRTUAL_ENV") is None
        assert env.get("PYTHONPATH") is None
        # Sanity: the poisoned values did exist on input (guards the test itself).
        assert poisoned["UV_PYTHON_INSTALL_DIR"].startswith("C:\\WorkBuddy")

    def test_update_uv_env_points_venv_and_runtime_store(self):
        """The update's final uv_env must carry VIRTUAL_ENV=this venv while the
        managed store path is still the UV_PYTHON_INSTALL_DIR."""
        from hermes_cli import main as hm
        from hermes_cli.managed_uv import managed_python_env

        uv_env = managed_python_env()
        uv_env["VIRTUAL_ENV"] = str(PROJECT_ROOT / "venv")

        assert uv_env["VIRTUAL_ENV"] == str(PROJECT_ROOT / "venv")
        # Managed store stays the install-scoped runtime dir, not a third-party one.
        assert ".hermes-runtime" in uv_env.get("UV_PYTHON_INSTALL_DIR", "")
        assert uv_env.get("UV_MANAGED_PYTHON") == "1"
        assert uv_env.get("UV_NO_CONFIG") == "1"


class TestCmdUpdateBranchFallback:
    """cmd_update falls back to main when current branch has no remote counterpart."""




    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_on_fork_checks_upstream_when_origin_up_to_date(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        """Regression for issue #26172: forks whose local HEAD already matches
        origin/main must still consult upstream/main before printing
        "Already up to date!" — otherwise a fork that's caught up to its own
        origin but behind NousResearch/hermes-agent silently misses updates.
        """
        from hermes_cli import main as hm

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(hm, "_sync_with_upstream_if_needed") as sync_mock:
            cmd_update(mock_args)

        expected_git_cmd = (
            ["git", "-c", "windows.appendAtomically=false"] if hm._is_windows() else ["git"]
        )
        sync_mock.assert_called_once_with(
            expected_git_cmd,
            PROJECT_ROOT,
            assume_yes=False,
            input_fn=None,
        )
        captured = capsys.readouterr()
        assert "Already up to date!" in captured.out

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_yes_on_fork_without_upstream_does_not_claim_up_to_date(
        self, mock_run, _mock_which, capsys
    ):
        """#97052 review: genuine fork, no upstream remote, HEAD == origin/main,
        --yes. The prompt is skipped without mutating remotes, and because the
        official repo was never consulted the completion line must not claim
        plain "Already up to date!"."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(
            update_cmd, "_has_upstream_remote", return_value=False
        ), patch.object(
            update_cmd, "_should_skip_upstream_prompt", return_value=False
        ), patch.object(
            update_cmd, "_add_upstream_remote"
        ) as add_remote, patch.object(
            update_cmd, "_mark_skip_upstream_prompt"
        ) as mark_skip, patch("builtins.input") as stdin_input:
            cmd_update(SimpleNamespace(yes=True))

        stdin_input.assert_not_called()
        add_remote.assert_not_called()
        mark_skip.assert_not_called()
        captured = capsys.readouterr()
        assert "Skipping upstream setup (non-interactive run)." in captured.out
        assert "official repo not checked" in captured.out
        assert "Already up to date!" not in captured.out

    @pytest.mark.parametrize(
        ("health_after_repair", "runtime_status", "expected_runtime_checks"),
        [
            (True, (False, SimpleNamespace(sqlite_version_string="3.46.1")), 1),
            (False, (True, None), 0),
        ],
    )
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_current_checkout_python_repair_failure_is_durable(
        self,
        mock_run,
        _mock_which,
        mock_args,
        health_after_repair,
        runtime_status,
        expected_runtime_checks,
    ):
        """Python repair must not bypass runtime and durable outcome checks."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_args.gateway = True
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(hm, "_sync_with_upstream_if_needed"), patch.object(
            update_cmd,
            "_venv_core_imports_healthy",
            side_effect=[
                (False, "broken before repair"),
                (health_after_repair, "broken after repair"),
            ],
        ), patch.object(
            hm, "_install_python_dependencies_with_optional_fallback"
        ), patch.object(
            hm, "_refresh_active_lazy_features"
        ), patch.object(
            hm, "_restore_active_tool_dependencies"
        ), patch.object(
            update_cmd, "_write_update_incomplete_marker"
        ), patch.object(
            hm, "_clear_update_incomplete_marker"
        ), patch.object(
            update_cmd,
            "_post_update_sqlite_runtime_status",
            return_value=runtime_status,
        ) as runtime_check, patch.object(
            update_cmd, "_write_gateway_update_exit_code"
        ) as write_gateway_exit, patch(
            "hermes_cli.update_receipt.finalize_update_receipt"
        ) as finalize_receipt, patch(
            "hermes_cli.update_receipt.finalize_pending_update_receipt"
        ):
            with pytest.raises(SystemExit) as exit_info:
                cmd_update(mock_args)

        assert exit_info.value.code == 1
        assert runtime_check.call_count == expected_runtime_checks
        write_gateway_exit.assert_called_once_with(False)
        finalize_receipt.assert_called_once_with("partial")

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_current_checkout_node_repair_verification_failure_is_durable(
        self, mock_run, _mock_which, mock_args
    ):
        """A failed Node-path runtime check must fail durable outcomes."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_args.gateway = True
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(hm, "_sync_with_upstream_if_needed"), patch.object(
            update_cmd,
            "_repair_node_deps_on_current_checkout",
            return_value=False,
        ), patch.object(
            update_cmd, "_write_gateway_update_exit_code"
        ) as write_gateway_exit, patch(
            "hermes_cli.update_receipt.finalize_update_receipt"
        ) as finalize_receipt, patch(
            "hermes_cli.update_receipt.finalize_pending_update_receipt"
        ):
            with pytest.raises(SystemExit) as exit_info:
                cmd_update(mock_args)

        assert exit_info.value.code == 1
        write_gateway_exit.assert_called_once_with(False)
        finalize_receipt.assert_called_once_with("partial")
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_fork_upstream_sync_that_moves_head_runs_post_update_steps(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        """A fork sync that pulls code must continue through post-update work."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        # The first two reads bracket the upstream sync (aaaaaaa -> bbbbbbb:
        # the sync moved HEAD). The NEXT two bracket the pull inside the
        # normal update path (bbbbbbb -> ccccccc) — the head-moved no-op
        # guard added after this PR exits 1 when that pair is equal, so the
        # mock must show the pull advancing HEAD too.
        shas = iter(["aaaaaaa", "bbbbbbb", "bbbbbbb", "ccccccc"])

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(
            update_cmd,
            "_capture_head_sha",
            side_effect=lambda *_args, **_kwargs: next(shas, "ccccccc"),
        ), patch(
            # The full post-update path runs the fleet version check, which
            # reads the REAL machine's profile gateway_state.json files —
            # live gateways on a dev box read as STALE vs this checkout and
            # exit 1. Pin an empty fleet: this test asserts the post-update
            # path RUNS, not the fleet's health.
            "hermes_cli.update_receipt.collect_fleet_versions",
            return_value=[],
        ), patch(
            # Same isolation for the restart phase: without these, the real
            # machine's live gateways enter the restart discovery, the
            # mocked-subprocess restart phase can't verify replacements, and
            # the fail-closed contract (#78574) exits 1 (locally the
            # live-system guard blocks the os.kill outright).
            "hermes_cli.gateway.find_gateway_pids",
            return_value=[],
        ), patch(
            "hermes_cli.gateway.find_profile_gateway_processes",
            return_value=[],
        ), patch(
            "hermes_cli.gateway._get_service_pids",
            return_value=set(),
        ), patch.object(
            hm, "_sync_with_upstream_if_needed"
        ), patch.object(
            hm,
            "_reload_updated_runtime_modules",
            # Reaching the reload step IS the proof the post-update path ran
            # (the bug returned from "Already up to date!" before it). Abort
            # the pipeline right here: everything past this point (skills
            # sync, desktop rebuild, gateway restart, fleet check) would run
            # for real against the host machine.
            side_effect=SystemExit(0),
        ) as post_update_step:
            with pytest.raises(SystemExit) as exit_info:
                cmd_update(mock_args)

        assert exit_info.value.code == 0
        post_update_step.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Already up to date!" not in captured.out

    def test_update_non_interactive_runs_safe_config_migrations(self, mock_args, capsys):
        """Dashboard/web updates apply non-interactive migrations before restart."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=["MISSING_KEY"]
        ), patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[{"key": "new.option", "default": True}],
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(1, 2)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": ["new.option"]},
        ) as migrate_config, patch("hermes_cli.main.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdout.isatty.return_value = False
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            migrate_config.assert_called_once_with(interactive=False, quiet=False)
            captured = capsys.readouterr()
            assert "applying safe config migrations" in captured.out
            assert "API keys require manual entry" in captured.out


class TestCmdUpdateMigrationPrompt:
    """The config-migration prompt names what changed and skips the prompt
    entirely when only the config format version moved.

    Regression guard for the contentless-prompt report (ScottFive / Tt2021):
    previously the prompt printed only counts ("1 new config option") and
    asked "configure them now?" even for pure version bumps, where saying
    yes looked like a no-op.
    """

    def test_version_bump_only_applies_silently_without_prompt(
        self, mock_args, capsys
    ):
        """Only the version moved → apply non-interactively, never prompt."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=[]
        ), patch(
            "hermes_cli.config.get_missing_config_fields", return_value=[]
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(5, 24)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": [], "warnings": []},
        ) as mock_migrate:
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            mock_migrate.assert_called_once_with(interactive=False, quiet=True)
            out = capsys.readouterr().out
            assert "Updating config format (v5 → v24)" in out
            assert "no new settings to configure" in out
            # The misleading question must NOT appear for a pure version bump.
            assert "configure them now" not in out.lower()

    def test_version_bump_only_surfaces_migration_resets(
        self, mock_args, capsys
    ):
        """A quiet version-bump migration that RESETS a user setting must say so.

        Regression for #86656: the v33→v34 personality reset ran with
        quiet=True and its results dict was discarded, so the update printed
        "no new settings to configure" while silently wiping
        display.personality. Migration-step mutations (config_added) and
        warnings must be re-surfaced even in the silent branch.
        """
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=[]
        ), patch(
            "hermes_cli.config.get_missing_config_fields", return_value=[]
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(33, 34)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={
                "env_added": [],
                "config_added": ["display.personality=none (one-time reset)"],
                "warnings": ["Disabled suspicious MCP server 'evil'"],
            },
        ):
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            out = capsys.readouterr().out
            assert "Updating config format (v33 → v34)" in out
            assert "no new settings to configure" in out
            # The migration's mutation note and warning must NOT be swallowed.
            assert "display.personality=none (one-time reset)" in out
            assert "Disabled suspicious MCP server 'evil'" in out

    def test_new_options_are_listed_by_name_before_prompt(
        self, mock_args, capsys
    ):
        """New env/config keys are printed by name so the user can decide."""
        env_items = [
            {"name": "FOO_API_KEY", "description": "Foo service API key"},
        ]
        cfg_items = [
            {"key": "display.new_widget", "description": "New config option: display.new_widget"},
        ]
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input", return_value="n"), patch(
            "hermes_cli.config.get_missing_env_vars", return_value=env_items
        ), patch(
            "hermes_cli.config.get_missing_config_fields", return_value=cfg_items
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(1, 24)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": [], "warnings": []},
        ), patch("hermes_cli.main.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            out = capsys.readouterr().out
            # Names, not just counts.
            assert "FOO_API_KEY" in out
            assert "Foo service API key" in out
            assert "display.new_widget" in out


class TestConfigVersionCheckUsesFreshModules:
    """Regression: config migration must use freshly-reloaded modules, not the
    sys.modules cache from before git pull.

    Before the fix, ``hermes update`` ran in the PRE-pull Python process.
    After ``git pull`` updated the source on disk, function-level imports
    returned the OLD cached ``hermes_cli.config`` module — so
    ``DEFAULT_CONFIG["_config_version"]`` was stale and
    ``check_config_version()`` reported ``(33, 33)`` "up to date" even though
    the freshly-pulled code had v34 with a migration to run. The personality
    reset migration (#81946) was silently skipped this way.
    """

    def test_run_config_check_fresh_reloads_modules(self):
        """_run_config_check_fresh must call _reload_config_modules which
        force-reloads the config modules from disk.

        Regression: config migration was silently skipped because
        sys.modules held the OLD hermes_cli.config with the OLD
        DEFAULT_CONFIG["_config_version"] after git pull.
        """
        from unittest.mock import patch

        import hermes_cli.update_cmd as update_cmd

        with patch.object(update_cmd, "_reload_config_modules") as mock_reload:
            update_cmd._run_config_check_fresh()

        mock_reload.assert_called_once()


class TestCmdUpdateProfileSkillSync:
    """cmd_update syncs bundled skills to all profiles, including the active one.

    Regression guard for #16176: previously the active profile was excluded
    from the seed_profile_skills loop, leaving it on stale skill content.
    """

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_active_profile_included_in_skill_sync(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        from pathlib import Path

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )

        default_p = SimpleNamespace(name="default", path=Path("/fake/.hermes"))
        active_p = SimpleNamespace(name="bit", path=Path("/fake/.hermes/profiles/bit"))
        other_p = SimpleNamespace(name="work", path=Path("/fake/.hermes/profiles/work"))
        all_profiles = [default_p, active_p, other_p]

        synced_paths = []

        def fake_seed(path, quiet=False):
            synced_paths.append(path)
            return {"copied": [], "updated": [], "user_modified": []}

        empty_sync = {"copied": [], "updated": [], "user_modified": [], "cleaned": []}

        with (
            patch("hermes_cli.profiles.list_profiles", return_value=all_profiles),
            patch("hermes_cli.profiles.seed_profile_skills", side_effect=fake_seed),
            patch("tools.skills_sync.sync_skills", return_value=empty_sync),
        ):
            cmd_update(mock_args)

        assert active_p.path in synced_paths, (
            f"Active profile 'bit' must be included in skill sync; got: {synced_paths}"
        )
        assert set(synced_paths) == {p.path for p in all_profiles}, (
            f"All profiles must be synced; got: {synced_paths}"
        )

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_single_profile_default_is_synced(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        from pathlib import Path

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )

        default_p = SimpleNamespace(name="default", path=Path("/fake/.hermes"))
        synced_paths = []

        def fake_seed(path, quiet=False):
            synced_paths.append(path)
            return {"copied": [], "updated": [], "user_modified": []}

        empty_sync = {"copied": [], "updated": [], "user_modified": [], "cleaned": []}

        with (
            patch("hermes_cli.profiles.list_profiles", return_value=[default_p]),
            patch("hermes_cli.profiles.seed_profile_skills", side_effect=fake_seed),
            patch("tools.skills_sync.sync_skills", return_value=empty_sync),
        ):
            cmd_update(mock_args)

        assert default_p.path in synced_paths


class TestCmdUpdateBranchFlag:
    """``hermes update --branch <name>`` targets the requested branch.

    The CLI default stays 'main'; --branch lets callers pick a different
    target without monkey-patching the implementation.
    """

    def _branch_side_effect(self, current_branch, target_branch, *, checkout_fails=False, track_fails=False, commit_count="0"):
        """Mock side-effect that knows about checkout/track behavior.

        - ``current_branch``  what ``git rev-parse --abbrev-ref HEAD`` returns
        - ``target_branch``   passed via --branch; what we expect the code to switch to
        - ``checkout_fails``  if True, ``git checkout <target>`` returns non-zero
                              (simulates branch absent locally; code should retry with -B)
        - ``track_fails``     if True, ``git checkout -B <target> origin/<target>`` ALSO fails
                              (simulates branch absent on origin too)
        - ``commit_count``    rev-list count returned (0 = up-to-date, >0 = behind)
        """

        def side_effect(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)

            if "rev-parse" in joined and "--abbrev-ref" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{current_branch}\n", stderr="")

            if "checkout" in joined and "-B" in joined:
                rc = 128 if track_fails else 0
                err = f"fatal: '{target_branch}' did not match any file(s) known to git\n" if track_fails else ""
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

            if "checkout" in joined and "-B" not in joined and "rev-parse" not in joined:
                rc = 128 if checkout_fails else 0
                err = f"error: pathspec '{target_branch}' did not match\n" if checkout_fails else ""
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

            if "rev-list" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        return side_effect

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_branch_flag_pulls_against_named_branch(self, mock_run, _mock_which, capsys):
        """--branch bb/gui makes rev-list and pull target origin/bb/gui."""
        mock_run.side_effect = self._branch_side_effect(
            current_branch="bb/gui", target_branch="bb/gui", commit_count="3"
        )
        args = SimpleNamespace(branch="bb/gui")

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]

        # rev-list must compare against origin/bb/gui, not origin/main
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert any("origin/bb/gui" in c for c in rev_list_cmds), rev_list_cmds
        assert not any("origin/main" in c for c in rev_list_cmds), rev_list_cmds

        # the ff-only merge must target origin/bb/gui
        merge_cmds = [c for c in commands if "merge --ff-only" in c]
        assert any("origin/bb/gui" in c and "origin/main" not in c for c in merge_cmds), merge_cmds


    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_branch_flag_fails_when_branch_missing_everywhere(self, mock_run, _mock_which, capsys):
        """If branch doesn't exist locally OR on origin, exit non-zero with clear error."""
        mock_run.side_effect = self._branch_side_effect(
            current_branch="main",
            target_branch="nonexistent",
            checkout_fails=True,
            track_fails=True,
            commit_count="0",
        )
        args = SimpleNamespace(branch="nonexistent")

        with pytest.raises(SystemExit) as exc_info:
            cmd_update(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "does not exist locally or on origin" in out
        assert "nonexistent" in out


class TestCmdUpdateCheckBranchFlag:
    """``hermes update --check --branch <name>`` honors the branch override.

    The check path used to call ``git rev-list HEAD..origin/<branch> --count``
    with ``check=True``. When the branch didn't exist on origin, the fetch
    silently succeeded (no refspec) but rev-list exited 128 and a raw
    ``CalledProcessError`` propagated to the user. These tests pin the
    friendlier behavior: detect-the-missing-ref before rev-list, exit 1
    with a clear message.
    """

    def _check_side_effect(
        self,
        target_branch: str,
        *,
        verify_ok: bool = True,
        commit_count: str = "0",
        upstream_fetch_ok: bool = True,
    ):
        """Mock side-effect for the _cmd_update_check git pipeline.

        - ``target_branch``      what we expect compare ref to point at
        - ``verify_ok``          if False, ``git rev-parse --verify --quiet
                                 origin/<branch>`` fails (branch missing
                                 on origin)
        - ``commit_count``       rev-list count (0 = up-to-date)
        - ``upstream_fetch_ok``  if False, ``git fetch upstream`` fails
                                 (forces fallback to origin on branch==main)
        """

        def side_effect(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)

            if "fetch" in joined and "upstream" in joined:
                rc = 0 if upstream_fetch_ok else 128
                err = "" if upstream_fetch_ok else "fatal: 'upstream' does not appear to be a git repository\n"
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

            if "fetch" in joined and "origin" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            if "rev-parse" in joined and "--verify" in joined:
                rc = 0 if verify_ok else 1
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

            if "rev-list" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        return side_effect

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_check_branch_compares_against_named_origin_branch(
        self, mock_run, _mock_method, capsys
    ):
        """--check --branch bb/gui compares against origin/bb/gui, never origin/main."""
        mock_run.side_effect = self._check_side_effect(
            target_branch="bb/gui", verify_ok=True, commit_count="2"
        )
        args = SimpleNamespace(check=True, branch="bb/gui")

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        # Non-main branch skips upstream probe entirely.
        assert not any("fetch" in c and "upstream" in c for c in commands), commands
        # Verify and rev-list both target origin/bb/gui.
        verify_cmds = [c for c in commands if "rev-parse" in c and "--verify" in c]
        assert any("origin/bb/gui" in c for c in verify_cmds), verify_cmds
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert any("origin/bb/gui" in c for c in rev_list_cmds), rev_list_cmds
        assert not any("origin/main" in c for c in rev_list_cmds), rev_list_cmds

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_check_branch_missing_on_origin_exits_cleanly(
        self, mock_run, _mock_method, capsys
    ):
        """If origin/<branch> doesn't exist, surface a friendly error and exit 1.

        Pre-fix this case raised CalledProcessError from rev-list's check=True
        and dumped a Python traceback to stdout.
        """
        mock_run.side_effect = self._check_side_effect(
            target_branch="ghost", verify_ok=False
        )
        args = SimpleNamespace(check=True, branch="ghost")

        with pytest.raises(SystemExit) as exc_info:
            cmd_update(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        # No raw Python traceback.
        assert "Traceback" not in out
        assert "CalledProcessError" not in out
        # Friendly message naming the branch.
        assert "ghost" in out
        assert "not found" in out

        # rev-list must never have been called once verify failed.
        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        assert not any("rev-list" in c for c in commands), commands

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_check_default_main_still_prefers_upstream(
        self, mock_run, _mock_method, capsys
    ):
        """No --branch (or --branch=None) preserves the upstream-then-origin probe."""
        mock_run.side_effect = self._check_side_effect(
            target_branch="main", verify_ok=True, commit_count="0"
        )
        args = SimpleNamespace(check=True, branch=None)

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        # Should have tried upstream first.
        assert any("fetch" in c and "upstream" in c for c in commands), commands
        # Compare ref is upstream/main (upstream fetch succeeded).
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert any("upstream/main" in c for c in rev_list_cmds), rev_list_cmds


class TestCmdUpdateZipBranchRefusal:
    """``hermes update --branch=<non-main>`` must refuse on the ZIP fallback path.

    The ZIP fallback hard-codes a GitHub archive URL for main.zip; honoring
    --branch arbitrarily would require remote-branch existence checks the
    fallback can't easily do. Refusing is the right move — silently lying
    about which branch got installed is the bug --branch was meant to prevent.
    """

    def test_zip_fallback_refuses_non_main_branch(self, capsys):
        from hermes_cli.update_cmd import _update_via_zip

        args = SimpleNamespace(branch="bb/gui")
        with pytest.raises(SystemExit) as exc_info:
            _update_via_zip(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "bb/gui" in out
        assert "not supported" in out
        # No actual download attempted.
        assert "Downloading latest version" not in out


def test_is_termux_env_true_for_termux_prefix():
    from hermes_cli import main as hm

    assert hm._is_termux_env({"PREFIX": "/data/data/com.termux/files/usr"}) is True


def test_load_installable_optional_extras_supports_termux_group(tmp_path, monkeypatch):
    from hermes_cli import main as hm

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "x"
version = "0.0.0"

[project.optional-dependencies]
all = ["x[mcp]"]
termux-all = ["x[termux]", "x[mcp]"]
mcp = ["mcp>=1"]
termux = ["rich>=14"]
""".strip()
    )
    monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)

    assert main_install_repair._load_installable_optional_extras(group="all") == ["mcp"]
    assert main_install_repair._load_installable_optional_extras(group="termux-all") == ["termux", "mcp"]


class TestNodeRuntimeNpmResolution:
    """Regression tests for #30271 — WSL must not run Windows npm against the
    Linux checkout, and a failed Node refresh must not report success."""






    def test_node_failure_returns_failed_labels_and_warns(
        self, tmp_path, monkeypatch, capsys
    ):
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_resolve_node_runtime_npm", lambda: "/usr/bin/npm")
        monkeypatch.setattr(
            hm,
            "_run_npm_install_deterministic",
            lambda *a, **k: subprocess.CompletedProcess([], 1, stdout="", stderr=""),
        )

        with patch(
            "tools.browser_tool_install.warm_agent_browser_npx_cache", return_value=True
        ):
            failed = update_cmd._update_node_dependencies()
        assert failed == ["ui-tui, web workspaces"]
        out = capsys.readouterr().out
        assert "mixed state" in out

    def test_wsl_update_skips_windows_npm_build_paths(self, mock_args, monkeypatch):
        """A Windows-only npm on WSL must not reach web or desktop builds."""
        from hermes_cli import main as hm
        import hermes_constants

        windows_npm = "/mnt/c/Program Files/nodejs/npm"
        monkeypatch.setattr(hm, "_is_windows", lambda: False)
        monkeypatch.setattr(hermes_constants, "is_wsl", lambda: True)
        monkeypatch.setattr(
            hermes_constants,
            "find_node_executable",
            lambda command: windows_npm if command == "npm" else None,
        )
        monkeypatch.setattr(
            hm.shutil,
            "which",
            lambda command, path=None: windows_npm if command == "npm" else "/usr/bin/uv",
        )
        monkeypatch.setenv("PATH", "/mnt/c/Program Files/nodejs")

        with patch("subprocess.run") as mock_run, \
             patch.object(main_web_build, "_web_ui_build_needed", return_value=True), \
             patch.object(hm, "_desktop_packaged_executable", return_value=None), \
             patch.object(hm, "_desktop_dist_exists", return_value=True), \
             patch.object(hm, "_run_npm_install_deterministic") as mock_npm_install, \
             patch.object(main_web_build, "_run_with_idle_timeout") as mock_idle_build, \
             patch.object(hm, "_run_logged_subprocess") as mock_desktop_build:
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )
            cmd_update(mock_args)

        mock_npm_install.assert_not_called()
        mock_idle_build.assert_not_called()
        mock_desktop_build.assert_not_called()
        assert all(
            not call.args or not call.args[0] or call.args[0][0] != windows_npm
            for call in mock_run.call_args_list
        )

    def test_update_rebuilds_desktop_that_disappears_mid_update(self):
        """A previously packaged Desktop must be rebuilt when its release tree vanishes."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        desktop_dir = PROJECT_ROOT / "apps" / "desktop"
        packaged_exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
        build_ok = subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch.object(
                hm, "_desktop_packaged_executable", side_effect=[packaged_exe, None]
            ) as packaged,
            patch.object(hm, "_desktop_dist_exists", return_value=False),
            patch.object(hm, "_resolve_node_runtime_npm", return_value="npm.cmd"),
            patch.object(hm, "_desktop_build_needed", return_value=True),
            patch.object(hm, "_run_logged_subprocess", return_value=build_ok) as desktop_build,
        ):
            had_desktop_app_before_update = update_cmd._desktop_app_present(desktop_dir)
            assert not update_cmd._desktop_app_present(desktop_dir)
            update_cmd._rebuild_desktop_after_update(
                desktop_dir,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )

        assert packaged.call_count == 2
        desktop_build.assert_called_once_with(
            [hm.sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"],
            cwd=PROJECT_ROOT,
            env=ANY,
        )

    def test_update_refreshes_linux_desktop_entry_when_build_is_current(
        self, tmp_path, monkeypatch
    ):
        """An up-to-date build must still repair a stale Linux launcher."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        desktop_dir = tmp_path / "apps" / "desktop"
        desktop_dir.mkdir(parents=True)
        (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
        packaged_exe = desktop_dir / "release" / "linux-unpacked" / "Hermes"

        monkeypatch.setattr(hm, "_desktop_packaged_executable", lambda _dir: packaged_exe)
        monkeypatch.setattr(hm, "_desktop_dist_exists", lambda _dir: False)
        monkeypatch.setattr(hm, "_resolve_node_runtime_npm", lambda: "/usr/bin/npm")
        monkeypatch.setattr(hm, "_desktop_build_needed", lambda *_args, **_kwargs: False)
        calls = []
        monkeypatch.setattr(hm, "_register_linux_desktop_entry", lambda: calls.append(True))

        assert update_cmd._rebuild_desktop_after_update(
            desktop_dir,
            had_desktop_app_before_update=False,
        )
        assert calls == [True]

    def test_git_failure_zip_fallback_rebuilds_missing_desktop(self, tmp_path, monkeypatch):
        """The Windows ZIP fallback keeps Desktop intact when replacing ``apps/``.

        Contract updated for the #70337/#87331 release-dir graft: the built
        desktop app (release/win-unpacked/Hermes.exe) is preserved THROUGH
        the swap — previously this test pinned the old repair shape (exe
        deleted by the swap, then rebuilt from scratch). The rebuild hook
        still runs (mocked _desktop_build_needed=True), but it now finds
        the packaged exe alive rather than missing.
        """
        import zipfile

        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        project_root = tmp_path / "hermes-agent"
        (project_root / ".git").mkdir(parents=True)
        desktop_dir = project_root / "apps" / "desktop"
        packaged_exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
        packaged_exe.parent.mkdir(parents=True)
        packaged_exe.write_bytes(b"desktop")

        def write_source_zip(_url, destination):
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("hermes-agent-main/apps/desktop/package.json", "{}")

        def fail_git_fetch(command, **_kwargs):
            if "fetch" in command:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        desktop_builds = []

        def rebuild_desktop(*_args, **_kwargs):
            desktop_builds.append(not packaged_exe.exists())
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        monkeypatch.setattr(hm, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(hm, "_is_windows", lambda: True)
        monkeypatch.setattr(hm, "_run_pre_update_backup", lambda _args: None)
        monkeypatch.setattr(hm, "_pause_windows_gateways_for_update", lambda: None)
        monkeypatch.setattr(hm, "_get_origin_url", lambda *_args: "")
        monkeypatch.setattr(
            hm,
            "_desktop_packaged_executable",
            lambda _desktop_dir: packaged_exe if packaged_exe.exists() else None,
        )
        monkeypatch.setattr(hm, "_desktop_dist_exists", lambda _desktop_dir: False)
        monkeypatch.setattr(hm, "_resolve_node_runtime_npm", lambda: "npm.cmd")
        monkeypatch.setattr(hm, "_desktop_build_needed", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(hm, "_run_logged_subprocess", rebuild_desktop)
        monkeypatch.setattr(hm, "_clear_bytecode_cache", lambda *_args: 0)
        monkeypatch.setattr(hm, "_record_bytecode_fingerprint", lambda: None)
        monkeypatch.setattr(hm, "_refresh_bootstrap_cache_scripts", lambda _branch: None)
        monkeypatch.setattr(
            hm, "_install_python_dependencies_with_optional_fallback", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr(hm, "_refresh_active_memory_provider_dependencies", lambda: None)
        monkeypatch.setattr(hm, "_build_web_ui", lambda *_args: None)
        monkeypatch.setattr(update_cmd, "_discard_lockfile_churn", lambda *_args: None)
        monkeypatch.setattr(update_cmd, "_normalize_managed_eol", lambda *_args: None)
        monkeypatch.setattr(
            update_cmd,
            "_validate_critical_modules_import",
            lambda *_args: (True, None, None),
        )
        monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda: [])
        monkeypatch.setattr(update_cmd, "_print_curator_first_run_notice", lambda: None)
        monkeypatch.setattr(update_cmd, "_print_curator_recent_run_notice", lambda: None)
        monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda _failures: None)
        monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path / "hermes-home")

        with (
            patch("hermes_cli.config.load_config", return_value={}),
            patch("subprocess.run", side_effect=fail_git_fetch),
            patch("urllib.request.urlretrieve", side_effect=write_source_zip),
            patch("hermes_cli.managed_uv.ensure_uv", return_value="uv"),
            patch("hermes_cli.managed_uv.update_managed_uv"),
            patch(
                "tools.skills_sync.sync_skills",
                return_value={
                    "copied": [],
                    "updated": [],
                    "user_modified": [],
                    "cleaned": [],
                    "relocated": [],
                },
            ),
            patch("hermes_cli.model_catalog.seed_cache_from_checkout", return_value=False),
        ):
            update_cmd._cmd_update_impl(
                SimpleNamespace(yes=True, force=True, force_venv=True, branch=None),
                gateway_mode=False,
            )

        # Release-dir graft (#70337): the packaged exe SURVIVES the swap, so
        # the rebuild hook observed it present (False), and the bytes are the
        # original build — never deleted, never rebuilt from nothing.
        assert desktop_builds == [False]
        assert packaged_exe.exists()
        assert packaged_exe.read_bytes() == b"desktop"


class TestUpdateNodeDependencies:
    """Unit tests for _update_node_dependencies — issue #43564.

    Root package.json has no dependencies of its own: agent-browser
    resolves at runtime via npx (tools/browser_tool.py), and @streamdown/math
    moved to apps/desktop/package.json since it's a desktop-only import.
    With nothing root-only left to protect, a single workspace-scoped
    install (ui-tui, web) is safe — apps/desktop is simply never named, so
    its ~200 MB Electron devDependency is never resolved. Skipping is
    governed by _npm_lockfile_changed (content hash over the lockfile +
    every workspace package.json), tested separately in
    TestNpmLockfileChanged.
    Uses a tmp_path root so tests never touch real node_modules.
    """

    @pytest.fixture(autouse=True)
    def _stub_npx_warmup(self):
        """The npx cache warm-up is covered by its own dedicated test below;
        stub it out everywhere else so it doesn't add a spurious npm/npx
        call to the workspace-install assertions in this class."""
        with patch("tools.browser_tool_install.warm_agent_browser_npx_cache", return_value=True):
            yield

    def _npm_calls(self, mock_run):
        return [
            call.args[0]
            for call in mock_run.call_args_list
            if call.args and "npm" in str(call.args[0][0])
        ]

    def _make_popen(self, calls, returncode=0, stderr_lines=()):
        """Fake subprocess.Popen recording each invocation's cmd/kwargs.

        _update_node_dependencies always runs npm with capture_output=False,
        which routes through the Popen-based stderr-teeing path in
        _run_npm_watching_for_engine_failure rather than subprocess.run.
        """

        class _FakeProc:
            def __init__(self, cmd, **kwargs):
                calls.append({"cmd": cmd, "kwargs": kwargs})
                self.stderr = iter(stderr_lines)

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def wait(self):
                return returncode

        return _FakeProc

    def _popen_npm_calls(self, calls):
        return [c["cmd"] for c in calls if c["cmd"] and "npm" in str(c["cmd"][0])]

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_install_names_ui_tui_and_web_workspaces(self, _which, mock_popen, tmp_path, monkeypatch):
        """Regression for #43564: install ui-tui + web directly. apps/desktop
        must never appear, so its Electron postinstall is never triggered.
        """
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda root: True)
        popen_calls = []
        mock_popen.side_effect = self._make_popen(popen_calls)

        update_cmd._update_node_dependencies()

        calls = self._popen_npm_calls(popen_calls)
        assert len(calls) == 1, f"expected exactly 1 npm call, got: {calls}"
        joined = " ".join(str(a) for a in calls[0])
        assert "--workspace ui-tui" in joined and "--workspace web" in joined, (
            f"expected ui-tui + web workspace selectors; actual: {calls[0]}"
        )
        assert "desktop" not in joined, (
            f"apps/desktop must not appear (avoids ~200 MB Electron download); actual: {calls[0]}"
        )
        assert "--workspaces=false" not in joined, (
            f"no root-only deps remain to protect; --workspaces=false is unnecessary now; actual: {calls[0]}"
        )

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_install_includes_workspace_root_to_protect_root_devdependencies(
        self, _which, mock_popen, tmp_path, monkeypatch
    ):
        """Root package.json still owns devDependencies (the shared ESLint
        flat config every workspace's own eslint.config.mjs imports) even
        though agent-browser and @streamdown/math were removed from root
        `dependencies` (#43564). --include-workspace-root keeps them from
        being pruned by this scoped install, while --workspace ui-tui
        --workspace web still excludes the unnamed apps/desktop workspace
        (confirmed empirically against npm 10.9.8 and 11.9.0 in PR #44772
        review)."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda root: True)
        popen_calls = []
        mock_popen.side_effect = self._make_popen(popen_calls)

        update_cmd._update_node_dependencies()

        calls = self._popen_npm_calls(popen_calls)
        assert len(calls) == 1
        joined = " ".join(str(a) for a in calls[0])
        assert "--include-workspace-root" in joined
        assert "desktop" not in joined

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_install_preserves_standard_flags(self, _which, mock_popen, tmp_path, monkeypatch):
        """--no-fund, --no-audit, --progress=false must survive."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda root: True)
        popen_calls = []
        mock_popen.side_effect = self._make_popen(popen_calls)

        update_cmd._update_node_dependencies()

        calls = self._popen_npm_calls(popen_calls)
        assert len(calls) == 1
        joined = " ".join(str(a) for a in calls[0])
        for flag in ("--no-fund", "--no-audit", "--progress=false"):
            assert flag in joined, f"{flag} missing from npm call; actual: {calls[0]}"

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_skips_install_when_deps_up_to_date(self, _which, mock_run, tmp_path, monkeypatch):
        """When _npm_lockfile_changed reports no change, npm must not be called."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda root: False)

        update_cmd._update_node_dependencies()

        assert not self._npm_calls(mock_run), (
            "npm must not run when _npm_lockfile_changed reports no change"
        )

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_runs_install_when_lockfile_changed(self, _which, mock_popen, tmp_path, monkeypatch):
        """When _npm_lockfile_changed reports a change, npm must run."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda root: True)
        popen_calls = []
        mock_popen.side_effect = self._make_popen(popen_calls)

        update_cmd._update_node_dependencies()

        calls = self._popen_npm_calls(popen_calls)
        assert len(calls) == 1, f"expected npm to run when lockfile changed; got: {calls}"

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_records_lockfile_hash_only_on_success(self, _which, mock_popen, tmp_path, monkeypatch):
        """A failed install must not record the lockfile hash (so the next
        run retries instead of wrongly believing deps are up to date)."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda root: True)
        recorded = []
        # _update_node_dependencies lives in update_cmd_deps and calls its own module-level
        # _record_npm_lockfile_hash, so that binding is the real seam (update_cmd's is dead).
        from hermes_cli import update_cmd_deps
        monkeypatch.setattr(update_cmd_deps, "_record_npm_lockfile_hash", lambda root: recorded.append(root))
        mock_popen.side_effect = self._make_popen([], returncode=1, stderr_lines=["npm ERR!\n"])

        update_cmd._update_node_dependencies()

        assert not recorded, "lockfile hash must not be recorded when npm install fails"

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_warms_npx_agent_browser_cache_regardless_of_install_result(
        self, _which, mock_popen, tmp_path, monkeypatch
    ):
        """The npx warm-up must fire even when the workspace install fails —
        it's independent of ui-tui/web dependency state (#43564)."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(hm, "_npm_lockfile_changed", lambda root: True)
        mock_popen.side_effect = self._make_popen([], returncode=1, stderr_lines=["npm ERR!\n"])

        with patch(
            "tools.browser_tool_install.warm_agent_browser_npx_cache", return_value=True
        ) as mock_warm:
            update_cmd._update_node_dependencies()

        mock_warm.assert_called_once()

    @patch("subprocess.run")
    @patch("shutil.which", return_value=None)
    def test_returns_silently_when_npm_not_found(self, _which, mock_run, tmp_path, monkeypatch):
        """No npm on PATH → return without calling subprocess."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)

        update_cmd._update_node_dependencies()

        mock_run.assert_not_called()

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_returns_silently_when_package_json_absent(self, _which, mock_run, tmp_path, monkeypatch):
        """No package.json → return without calling npm."""
        from hermes_cli import main as hm

        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)

        update_cmd._update_node_dependencies()

        mock_run.assert_not_called()

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="/usr/bin/npm")
    def test_install_runs_from_project_root(self, _which, mock_popen, tmp_path, monkeypatch):
        """npm install must execute from PROJECT_ROOT, not a workspace subdir."""
        from hermes_cli import main as hm

        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "package-lock.json").write_text("{}")
        monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)

        popen_calls = []
        mock_popen.side_effect = self._make_popen(popen_calls)

        update_cmd._update_node_dependencies()

        cwd_calls = [
            c["kwargs"].get("cwd")
            for c in popen_calls
            if c["cmd"] and "npm" in str(c["cmd"][0])
        ]
        assert cwd_calls, "expected at least one npm call"
        for cwd in cwd_calls:
            assert cwd == tmp_path, f"npm must run from PROJECT_ROOT; got cwd={cwd}"


class TestGitTrampolineSelfHeal:
    """Proactive Git-for-Windows trampoline self-heal (#87876).

    A broken bin\\git.exe / cmd\\git.exe shim (~46KB) refuses every git call
    with a "BUG (fork bomb)" guard instead of re-execing the real git-core
    binary. _ensure_non_trampoline_git detects this up front and swaps in a
    real git binary when one can be located, so the normal git update path
    survives instead of degrading to the ZIP fallback.
    """

    @staticmethod
    def _fake_run_healthy(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="git version 2.50.0.windows.1\n", stderr=""
        )

    @staticmethod
    def _fake_run_trampoline(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="BUG (fork bomb): tried to spawn itself, check your PATH\n",
        )

    def test_healthy_git_command_unchanged(self):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch("sys.platform", "win32"),
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_healthy,
            ),
            patch("hermes_cli.update_cmd._locate_real_git") as locate,
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        locate.assert_not_called()

    def test_trampoline_swaps_to_real_git(self, capsys):
        from pathlib import Path

        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        real = Path(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe")
        with (
            patch("sys.platform", "win32"),
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_trampoline,
            ),
            patch(
                "hermes_cli.update_cmd._locate_real_git", return_value=real
            ),
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == [str(real), "-c", "windows.appendAtomically=false"]
        out = capsys.readouterr().out
        assert "switching to real git" in out

    def test_trampoline_no_real_git_keeps_command(self, capsys):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch("sys.platform", "win32"),
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_trampoline,
            ),
            patch("hermes_cli.update_cmd._locate_real_git", return_value=None),
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        out = capsys.readouterr().out
        assert "ZIP path" in out

    def test_off_windows_noop(self):
        from hermes_cli import update_cmd

        git_cmd = ["git"]
        with (
            patch("sys.platform", "linux"),
            patch("hermes_cli.update_cmd.subprocess.run") as run,
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        run.assert_not_called()

    def test_portable_git_candidates_check_shared_root_first(self, tmp_path, monkeypatch):
        # Profile-scoped layout: HERMES_HOME = <root>/profiles/foo, but the
        # PortableGit tree lives under the SHARED root (monerostar review on
        # #88136). The candidate list must check get_default_hermes_root()
        # before the profile home.
        from hermes_cli import update_cmd

        root = tmp_path / "root"
        profile_home = root / "profiles" / "foo"

        monkeypatch.setattr(update_cmd, "get_default_hermes_root", lambda: root)
        monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: profile_home)

        candidates = update_cmd._portable_git_candidates()
        assert candidates[0] == (
            root / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
        )
        assert candidates[1] == (
            profile_home / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
        )
