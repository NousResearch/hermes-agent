"""Tests for _web_ui_build_needed — staleness check for the web UI dist.

Critical invariant: the dashboard Vite build outputs to hermes_cli/web_dist/
(vite.config.ts: outDir: "../../hermes_cli/web_dist"), NOT web/dist/.
The sentinel must be checked in the correct output directory or the
freshness check is a no-op and the OOM rebuild always runs.
"""

import os
import time
from pathlib import Path
from unittest.mock import patch


from hermes_cli.main import _web_ui_build_needed, _build_web_ui, _run_npm_install_deterministic


def _touch(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if offset:
        t = time.time() + offset
        os.utime(path, (t, t))


def _make_web_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Return (web_dir, dist_dir) matching real repo layout."""
    web_dir = tmp_path / "web"
    web_dir.mkdir(parents=True)
    (web_dir / "package.json").touch()
    dist_dir = tmp_path / "hermes_cli" / "web_dist"
    return web_dir, dist_dir


class TestWebUIBuildNeeded:

    def test_returns_true_when_dist_missing(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_false_when_vite_manifest_fresh(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(web_dir / "src" / "App.tsx", offset=-10)
        _touch(dist_dir / ".vite" / "manifest.json")
        assert _web_ui_build_needed(web_dir) is False

    def test_returns_true_when_source_newer_than_manifest(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "src" / "App.tsx")
        assert _web_ui_build_needed(web_dir) is True

    def test_falls_back_to_index_html_when_manifest_missing(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(web_dir / "src" / "main.ts", offset=-10)
        _touch(dist_dir / "index.html")
        assert _web_ui_build_needed(web_dir) is False

    def test_web_dist_dir_not_web_dist_subdir(self, tmp_path):
        """Regression: sentinel must be in hermes_cli/web_dist/, NOT web/dist/."""
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(web_dir / "src" / "App.tsx", offset=-10)
        # Place manifest in wrong location (web/dist/) — should NOT count as fresh
        wrong_dist = web_dir / "dist" / ".vite" / "manifest.json"
        _touch(wrong_dist)
        # Correct location is empty → still needs build
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_true_when_package_lock_newer_than_dist(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        # With a single workspace root lockfile, the lockfile lives at the
        # project root (tmp_path), not inside web_dir.
        _touch(tmp_path / "package-lock.json")
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_true_when_vite_config_newer_than_dist(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "vite.config.ts")
        assert _web_ui_build_needed(web_dir) is True

    def test_ignores_node_modules(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        # package.json older than manifest; only node_modules file is newer
        _touch(web_dir / "package.json", offset=-20)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "node_modules" / "react" / "index.js")
        assert _web_ui_build_needed(web_dir) is False

    def test_ignores_dist_subdir_under_web(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        # package.json older than manifest; only web/dist file is newer
        _touch(web_dir / "package.json", offset=-20)
        _touch(dist_dir / ".vite" / "manifest.json", offset=-10)
        _touch(web_dir / "dist" / "assets" / "index.js")
        assert _web_ui_build_needed(web_dir) is False


class TestBuildWebUISkipsWhenFresh:

    def test_skips_npm_when_dist_is_fresh(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json")

        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run") as mock_run:
            result = _build_web_ui(web_dir)

        assert result is True
        mock_run.assert_not_called()

    def test_runs_npm_when_dist_missing(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)

        mock_cp = __import__("subprocess").CompletedProcess([], 0, stdout=b"", stderr=b"")
        build_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=mock_cp) as mock_run, \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_ok) as mock_idle:
            result = _build_web_ui(web_dir)

        assert result is True
        # npm install goes through subprocess.run; npm run build goes through
        # _run_with_idle_timeout (issue #33788).
        assert mock_run.call_count == 1   # install only
        assert mock_idle.call_count == 1  # build only

    def test_npm_install_uses_utf8_replace_output_decoding(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        mock_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.subprocess.run", return_value=mock_cp) as mock_run:
            result = _run_npm_install_deterministic("/usr/bin/npm", web_dir)

        assert result.returncode == 0
        _, kwargs = mock_run.call_args
        assert kwargs["text"] is True
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"

    def test_npm_install_sets_ci_to_suppress_postinstall_tty_output(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        mock_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.subprocess.run", return_value=mock_cp) as mock_run:
            _run_npm_install_deterministic(
                "/usr/bin/npm",
                web_dir,
                env={"PYTHON": "/nix/store/python"},
            )

        _, kwargs = mock_run.call_args
        assert kwargs["env"]["CI"] == "1"
        assert kwargs["env"]["PYTHON"] == "/nix/store/python"

    def test_npm_install_uses_workspace_web_scope(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        # Real workspace checkout: the single lockfile lives at the root, so
        # _workspace_root(web_dir) resolves to the parent and --workspace web
        # scopes the install. (Without a root lockfile, web_dir IS the root and
        # --workspace would be dropped — see test below and #42973.)
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        mock_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=mock_cp) as mock_run, \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_ok):
            result = _build_web_ui(web_dir)
        assert result is True
        install_cmd = mock_run.call_args[0][0]
        assert "--workspace" in install_cmd
        assert install_cmd[install_cmd.index("--workspace") + 1] == "web"

    def test_web_install_omits_workspace_when_web_has_own_lockfile(
        self, tmp_path, monkeypatch
    ):
        """web/ with its own lockfile => _workspace_root returns web_dir, so
        --workspace web would fail (npm can't find that workspace from inside
        web/). The flag must be dropped and the install run plainly from web_dir.
        Symmetric to the TUI fix in test_tui_npm_install.py. See #42973.

        With web's own lockfile present at cwd, _run_npm_install_deterministic
        uses ``npm ci`` (not ``npm install``).
        """
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/usr")

        install_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=install_cp) as mock_run, \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_cp):
            result = _build_web_ui(web_dir)

        assert result is True
        args, kwargs = mock_run.call_args
        assert "--workspace" not in args[0]
        assert args[0] == ["/usr/bin/npm", "ci", "--silent"]
        assert kwargs["cwd"] == web_dir

    def test_web_build_uses_idle_timeout_helper(self, tmp_path):
        """npm run build now goes through _run_with_idle_timeout (issue #33788).

        The install step keeps its capture_output behavior (the existing
        retry-on-EPERM contract depends on it); only the long-running build
        step is streamed + idle-killed.
        """
        web_dir, _ = _make_web_dir(tmp_path)

        install_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=install_cp), \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_cp) as mock_idle:
            result = _build_web_ui(web_dir)

        assert result is True
        # Build was invoked through the idle-timeout helper, not subprocess.run.
        mock_idle.assert_called_once()
        args, kwargs = mock_idle.call_args
        # Positional: [npm, "run", "build"]; cwd passed as kwarg.
        assert args[0] == ["/usr/bin/npm", "run", "build"]
        assert kwargs["cwd"] == web_dir

    def test_termux_web_install_is_workspace_scoped(self, tmp_path, monkeypatch):
        web_dir, _ = _make_web_dir(tmp_path)
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("TERMUX_VERSION", "1")

        install_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=install_cp) as mock_run, \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_cp):
            result = _build_web_ui(web_dir)

        assert result is True
        args, kwargs = mock_run.call_args
        assert args[0] == [
            "/usr/bin/npm",
            "ci",
            "--workspace",
            "web",
            "--include-workspace-root=false",
            "--silent",
        ]
        assert kwargs["cwd"] == tmp_path

    def test_desktop_web_install_uses_existing_workspace_root(
        self, tmp_path, monkeypatch
    ):
        web_dir, _ = _make_web_dir(tmp_path)
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/usr")

        install_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=install_cp) as mock_run, \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_cp):
            result = _build_web_ui(web_dir)

        assert result is True
        args, kwargs = mock_run.call_args
        assert args[0] == ["/usr/bin/npm", "ci", "--workspace", "web", "--silent"]
        assert kwargs["cwd"] == tmp_path


class TestBuildWebUIRetryAndStaleFallback:
    """Coverage for the retry + stale-dist fallback added in #23824 / issue #23817."""

    def test_retries_build_once_on_failure(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        Subprocess = __import__("subprocess")
        install_ok = Subprocess.CompletedProcess([], 0, stdout="", stderr="")
        # build attempt 1: fail; build attempt 2: success.
        build_fail = Subprocess.CompletedProcess([], 1, stdout="EPERM", stderr="")
        build_ok = Subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main._time.sleep") as mock_sleep, \
             patch("hermes_cli.main.subprocess.run", return_value=install_ok), \
             patch("hermes_cli.main._run_with_idle_timeout",
                   side_effect=[build_fail, build_ok]) as mock_idle:
            result = _build_web_ui(web_dir)

        assert result is True
        assert mock_idle.call_count == 2  # build + retry
        mock_sleep.assert_called_once_with(3)

    def test_falls_back_to_stale_dist_when_retry_also_fails(self, tmp_path, capsys):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        # Stale dist exists but is older than source
        _touch(dist_dir / "index.html", offset=-100)
        _touch(web_dir / "src" / "App.tsx")  # newer source -> build_needed=True

        Subprocess = __import__("subprocess")
        install_ok = Subprocess.CompletedProcess([], 0, stdout="", stderr="")
        build_fail = Subprocess.CompletedProcess([], 1, stdout="vite ENOMEM", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main._time.sleep"), \
             patch("hermes_cli.main.subprocess.run", return_value=install_ok), \
             patch("hermes_cli.main._run_with_idle_timeout",
                   side_effect=[build_fail, build_fail]):
            result = _build_web_ui(web_dir, fatal=True)

        # MUST return True (serve stale) — issue #23817 — even with fatal=True,
        # because cmd_dashboard passes fatal=True and is the primary caller.
        assert result is True
        out = capsys.readouterr().out
        assert "serving stale dist as fallback" in out
        assert "vite ENOMEM" in out  # combined output surfaced to user

    def test_hard_fails_when_no_dist_to_fall_back_to(self, tmp_path, capsys):
        web_dir, _ = _make_web_dir(tmp_path)

        Subprocess = __import__("subprocess")
        install_ok = Subprocess.CompletedProcess([], 0, stdout="", stderr="")
        build_fail = Subprocess.CompletedProcess([], 1, stdout="vite ENOMEM", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main._time.sleep"), \
             patch("hermes_cli.main.subprocess.run", return_value=install_ok), \
             patch("hermes_cli.main._run_with_idle_timeout",
                   side_effect=[build_fail, build_fail]):
            result = _build_web_ui(web_dir, fatal=True)

        assert result is False
        out = capsys.readouterr().out
        assert "Web UI build failed" in out
        assert "vite ENOMEM" in out
        assert "Run manually" in out


class TestLooksLikeInstallScriptFailure:
    """Tests for _looks_like_install_script_failure — detection heuristic."""

    def _result(self, stderr: str = "", returncode: int = 1):
        import subprocess
        return subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)

    def test_info_run_pattern(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result('info run electron-winstaller@5.4.0 install')) is True

    def test_eagain_spawn_pattern(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result('npm error spawn sh EAGAIN')) is True

    def test_err_script_pattern(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result('npm ERR! command sh -c node-gyp rebuild')) is True

    def test_network_error_returns_false(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result('npm ERR! code ECONNRESET')) is False

    def test_auth_failure_returns_false(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result('npm ERR! code E401\nnpm ERR! Unable to authenticate')) is False

    def test_lockfile_mismatch_returns_false(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result('npm ERR! code E405\nnpm ERR! Lockfile has conflict entries')) is False

    def test_empty_stderr_returns_false(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result()) is False

    def test_successful_result_returns_false(self):
        from hermes_cli.main import _looks_like_install_script_failure as check
        assert check(self._result(returncode=0)) is False
