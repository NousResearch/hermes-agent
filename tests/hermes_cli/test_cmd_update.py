"""Git trampoline recovery; branch updates use the real target-identity suite."""

import atexit
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main
from hermes_cli.main import cmd_update
from hermes_cli import update_cmd

_PRE_HEAD = "aaaa1111111111111111111111111111111111"
_POST_HEAD = "bbbb2222222222222222222222222222222222"


def _make_run_side_effect(
    branch="main",
    verify_ok=True,
    commit_count="0",
    *,
    pre_head=_PRE_HEAD,
    post_head=_POST_HEAD,
):
    """Build a side_effect function for subprocess.run that simulates git commands."""

    state = {"head": pre_head}

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "merge" in joined and "--ff-only" in joined:
            state["head"] = post_head
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        if "merge-base" in joined and "--is-ancestor" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        if "rev-parse" in joined and "^{commit}" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{post_head}\n", stderr="")

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        if "rev-parse" in joined and "--verify" in joined:
            rc = 0 if verify_ok else 128
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        if "rev-parse" in joined and "HEAD" in joined:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{state['head']}\n", stderr=""
            )

        if "fetch" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


@pytest.fixture
def mock_args():
    return SimpleNamespace()


@pytest.fixture(autouse=True)
def _patch_managed_uv(request):
    """Make managed_uv helpers follow shutil.which mocking in tests."""
    import shutil

    def _fake_resolve_uv():
        return shutil.which("uv")

    def _fake_ensure_uv(**_kwargs):
        return shutil.which("uv")

    def _fake_update_managed_uv(**_kwargs):
        return None

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
    """Keep cmd_update's gateway auto-restart phase off this machine's gateways."""
    with patch("hermes_cli.gateway.find_gateway_pids", return_value=[]), \
         patch("hermes_cli.gateway.supports_systemd_services", return_value=False), \
         patch("hermes_cli.gateway.find_profile_gateway_processes", return_value=[]), \
         patch.object(hermes_main, "_pause_windows_gateways_for_update", lambda: None), \
         patch.object(
             hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
         ), \
         patch.object(
             update_cmd, "_purge_stale_hermes_modules", lambda *a, **kw: None
         ), \
         patch.object(
             update_cmd, "_fleet_probe_expected_runtimes", lambda *a, **kw: False
         ), \
         patch.object(
             update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None
         ), \
         patch.object(hermes_main, "_detect_venv_python_processes", lambda *a, **k: []), \
         patch("hermes_cli.update_inventory.collect_runtime_inventory", return_value=None), \
         patch("hermes_cli.update_inventory.report_unaccounted_runtimes", return_value=False), \
         patch("hermes_cli.update_receipt.collect_fleet_versions", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _isolate_venv_holders(monkeypatch):
    """The update flow's venv-holder guard sees the live gateway processes on
    a dev machine and aborts with SystemExit 2 before reaching the branch
    logic under test.  Isolate it so the test exercises the intended path."""
    monkeypatch.setattr("hermes_cli.update_cmd_windows._detect_venv_python_processes", list)


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

    @pytest.mark.platforms("windows")
    def test_healthy_git_command_unchanged(self):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_healthy,
            ),
            patch("hermes_cli.update_cmd._locate_real_git") as locate,
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        locate.assert_not_called()

    @pytest.mark.platforms("windows")
    def test_trampoline_swaps_to_real_git(self, capsys):
        from pathlib import Path

        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        real = Path(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe")
        with (
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

    @pytest.mark.platforms("windows")
    def test_trampoline_no_real_git_keeps_command(self, capsys):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
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

    @pytest.mark.platforms("not windows")
    def test_off_windows_noop(self):
        from hermes_cli import update_cmd

        git_cmd = ["git"]
        with patch("hermes_cli.update_cmd.subprocess.run") as run:
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        run.assert_not_called()

    def test_portable_git_candidates_check_shared_root_first(self, tmp_path, monkeypatch):
        # Profile-scoped layout: HERMES_HOME = <root>/profiles/foo, but the
        # PortableGit tree lives under the SHARED root (monerostar review on
        # #88136). The candidate list must check get_default_hermes_root()
        # before the profile home.
        import hermes_constants
        from hermes_cli.update_cmd_git import _portable_git_candidates

        root = tmp_path / "root"
        profile_home = root / "profiles" / "foo"

        monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile_home)

        candidates = _portable_git_candidates()
        assert candidates[0] == (
            root / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
        )
        assert candidates[1] == (
            profile_home / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
        )


@pytest.mark.platforms("windows")
def test_full_flow_update_does_not_touch_live_windows_processes(tmp_path, monkeypatch):
    """Mocked git subprocess.run makes schtasks /Query look installed.

    Recorders sit on gateway spawn primitives so removing the autouse fixture
    stubs fails here with captured calls instead of silently spawning a gateway.
    """
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", lambda *a, **k: True)
    monkeypatch.setattr(hermes_config, "get_missing_env_vars", lambda required_only=True: [])
    monkeypatch.setattr(hermes_config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(hermes_config, "check_config_version", lambda **_kwargs: (5, 5))
    monkeypatch.setattr(
        hermes_config,
        "migrate_config",
        lambda **kw: {"env_added": [], "config_added": []},
    )
    monkeypatch.setattr(update_cmd, "_refresh_active_lazy_features", lambda *a, **kw: True)
    monkeypatch.setattr(
        update_cmd, "_prepare_git_command", lambda: (True, ["git"], False)
    )
    monkeypatch.setattr(
        update_cmd,
        "run_completion",
        lambda request: {"exit_code": 0, "receipt": None},
    )
    args = SimpleNamespace(branch="main", yes=True, gateway=False)

    captured = []
    _real_popen = subprocess.Popen

    def _record(name, result=None):
        def _fake(*args, **kwargs):
            captured.append((name, args, kwargs))
            return result

        return _fake

    def _fail_ready(*_args, **_kwargs):
        raise RuntimeError("sentinel: skip _wait_for_gateway_ready")

    def _gateway_guard_popen(cmd, *args, **kwargs):
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        joined = " ".join(str(x) for x in argv).lower()
        if "gateway" in joined and "hermes" in joined:
            captured.append(("Popen", tuple(argv), kwargs))
            proc = MagicMock(pid=4242, returncode=0)
            proc.wait.return_value = 0
            proc.poll.return_value = 0
            return proc
        return _real_popen(cmd, *args, **kwargs)

    _real_atexit_register = atexit.register

    def _fake_atexit_register(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "_resume_windows_gateways_after_update":
            captured.append(("atexit.register", (func,) + args, kwargs))
            return None
        return _real_atexit_register(func, *args, **kwargs)

    spawn_fake = _record("_spawn_detached", 4242)
    terminate_fake = _record("terminate_pid")
    stop_fake = _record("_stop_process_trees")
    kill_fake = _record(
        "_kill_stale_dashboard_processes",
        {"matched": [], "killed": [], "failed": [], "unrecovered": []},
    )

    run_side_effect = _make_run_side_effect(
        branch="main", verify_ok=True, commit_count="1"
    )

    with (
        patch("shutil.which", return_value=None),
        patch("subprocess.run", side_effect=run_side_effect),
        patch("subprocess.Popen", _gateway_guard_popen),
        patch("atexit.register", _fake_atexit_register),
        patch("hermes_cli.gateway_windows._spawn_detached", spawn_fake),
        patch("gateway.status.terminate_pid", terminate_fake),
        patch("hermes_cli.update_cmd_windows._stop_process_trees", stop_fake),
        patch.object(hermes_main, "_stop_process_trees", stop_fake, create=True),
        patch(
            "hermes_cli.dashboard_procs._kill_stale_dashboard_processes",
            kill_fake,
        ),
        patch.object(
            hermes_main, "_kill_stale_dashboard_processes", kill_fake, create=True
        ),
        patch("hermes_cli.gateway_windows._wait_for_gateway_ready", _fail_ready),
        patch.object(
            update_cmd, "_purge_stale_hermes_modules", lambda *a, **kw: None
        ),
    ):
        caught_exit = None
        try:
            cmd_update(args)
        except SystemExit as exc:
            caught_exit = exc.code

    assert not captured and caught_exit is None, (
        "full-flow update reached live Windows process primitives "
        "(fixture stubs missing?): calls={0!r}, SystemExit={1!r}".format(
            captured, caught_exit
        )
    )
