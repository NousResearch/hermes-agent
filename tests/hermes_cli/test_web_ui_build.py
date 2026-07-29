"""Tests for _web_ui_build_needed — staleness check for the web UI dist.

The freshness check uses a SHA-256 content hash of the web source tree
(mirroring the desktop build), recorded in a stamp file under $HERMES_HOME,
NOT mtime comparison — so ``git pull`` / ``hermes update`` that rewrite
source mtimes without changing content no longer fool it.

Critical invariant: the dashboard Vite build outputs to hermes_cli/web_dist/
(vite.config.ts: outDir: "../../hermes_cli/web_dist"), NOT web/dist/.
The sentinel must be checked in the correct output directory or the
freshness check is a no-op and the OOM rebuild always runs.
"""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.main import (
    _web_ui_build_needed,
    _build_web_ui,
    _compute_web_ui_content_hash,
    _missing_web_build_tool,
    _run_npm_install_deterministic,
    _web_build_toolchain_ready,
    _web_toolchain_roots,
    _web_ui_stamp_path,
    _write_web_ui_build_stamp,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    """Keep web-build-stamp writes inside the test's tmp dir, never the real home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "_hermes_home"))


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
    """Content-hash staleness — replaces the old mtime comparison.

    The dashboard build hashes the web source tree (like the desktop build)
    instead of comparing mtimes, so git operations that rewrite mtimes
    without changing content no longer fool the freshness check.
    """

    @staticmethod
    def _root(web_dir: Path) -> Path:
        return web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent

    def _stamp_current(self, web_dir: Path) -> None:
        """Record a stamp matching web_dir's current source content."""
        _write_web_ui_build_stamp(self._root(web_dir), web_dir)





    def test_mtime_only_change_is_not_stale(self, tmp_path):
        """The whole point: bumping mtimes without changing bytes (what
        ``git pull`` / ``hermes update`` do) must NOT report stale."""
        web_dir, dist_dir = _make_web_dir(tmp_path)
        src = web_dir / "src" / "App.tsx"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("export const A = 1\n")
        (dist_dir / ".vite").mkdir(parents=True, exist_ok=True)
        (dist_dir / ".vite" / "manifest.json").write_text("{}")
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is False
        future = time.time() + 10_000
        os.utime(src, (future, future))
        os.utime(web_dir / "package.json", (future, future))
        assert _web_ui_build_needed(web_dir) is False



    def test_content_hash_is_deterministic(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("export const A = 1\n")
        root = self._root(web_dir)
        h1 = _compute_web_ui_content_hash(root, web_dir)
        h2 = _compute_web_ui_content_hash(root, web_dir)
        assert h1 == h2
        assert len(h1) == 64

    def test_write_stamp_creates_file_with_hash(self, tmp_path):
        import json as _json
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("export const A = 1\n")
        self._stamp_current(web_dir)
        stamp = _web_ui_stamp_path()
        assert stamp.is_file()
        data = _json.loads(stamp.read_text())
        assert data["contentHash"] == _compute_web_ui_content_hash(self._root(web_dir), web_dir)



class TestBuildWebUISkipsWhenFresh:







    def test_web_install_omits_workspace_and_scrubs_esbuild_override(
        self, tmp_path, monkeypatch
    ):
        """web/ with its own lockfile => _workspace_root returns web_dir, so
        --workspace web would fail (npm can't find that workspace from inside
        web/). The flag must be dropped and the install run plainly from web_dir.
        Symmetric to the TUI fix in test_tui_npm_install.py. See #42973.

        With web's own lockfile present at cwd, _run_npm_install_deterministic
        uses ``npm ci`` (not ``npm install``). The shared installer must also
        remove an inherited esbuild binary override so package/binary versions
        cannot diverge (#87405).
        """
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/usr")
        monkeypatch.setenv("ESBUILD_BINARY_PATH", "/opt/esbuild-0.28.2")

        install_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main.subprocess.run", return_value=install_cp) as mock_run, \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_cp) as mock_build:
            result = _build_web_ui(web_dir)

        assert result is True
        args, kwargs = mock_run.call_args
        assert "--workspace" not in args[0]
        assert Path(args[0][0]).name in {"npm", "npm.cmd"}
        assert args[0][1:] == ["ci", "--include=dev", "--silent", "--prefer-offline"]
        assert kwargs["cwd"] == web_dir
        assert "ESBUILD_BINARY_PATH" not in kwargs["env"]
        assert "ESBUILD_BINARY_PATH" not in mock_build.call_args.kwargs["env"]

    def test_workspace_root_install_names_update_closure(self, tmp_path, monkeypatch):
        """From the workspace root, _build_web_ui must install the SAME
        closure as `hermes update` (ui-tui + web + --include-workspace-root).

        The install helper prefers `npm ci`, which deletes node_modules before
        reifying the requested tree — a narrower `--workspace web`-only pass
        right after the update step silently pruned root devDependencies and
        the ui-tui workspace while exiting 0. See #43564/#64354.
        """
        web_dir, _ = _make_web_dir(tmp_path)
        # Root lockfile only => _workspace_root(web_dir) == tmp_path.
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        (tmp_path / "ui-tui").mkdir()
        (tmp_path / "ui-tui" / "package.json").write_text("{}", encoding="utf-8")
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
        cmd = args[0]
        assert "--include-workspace-root" in cmd
        assert cmd.count("--workspace") == 2
        assert "ui-tui" in cmd and "web" in cmd
        assert kwargs["cwd"] == tmp_path

    def test_workspace_root_install_skips_missing_ui_tui(self, tmp_path, monkeypatch):
        """A checkout without the ui-tui workspace must not name it — npm
        fails hard on a --workspace that doesn't exist."""
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
        cmd = mock_run.call_args[0][0]
        assert "ui-tui" not in cmd
        assert "--include-workspace-root" in cmd
        assert "web" in cmd

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


class TestBuildWebUIFlock:
    """Cross-process build serialization (salvaged from PR #63455).

    One process builds under an exclusive flock on <root>/.web_ui_build.lock;
    contenders either serve the existing (possibly stale) dist or, when no
    dist exists yet, block until the builder finishes. The staleness walk
    itself runs inside _do_build_web_ui, i.e. under the lock, so a process
    that queued behind a successful build skips the rebuild.
    """



    def test_contended_lock_without_dist_waits_then_skips_fresh_build(self, tmp_path):
        """First-ever build race: the waiter blocks, and once it acquires the
        lock the callee's own staleness check (running under the lock) sees
        the winner's output and skips a duplicate build."""
        import fcntl
        import threading
        from hermes_cli.main import _build_web_ui as build

        web_dir, dist_dir = _make_web_dir(tmp_path)
        # No dist yet — contender must take the blocking-wait path.
        lock_path = tmp_path / ".web_ui_build.lock"
        holder = open(lock_path, "a")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)

        def release_after_building():
            # Simulate the winning process finishing its build.
            _touch(dist_dir / ".vite" / "manifest.json")
            _write_web_ui_build_stamp(tmp_path, web_dir)
            holder.close()  # releases the flock

        t = threading.Timer(0.2, release_after_building)
        t.start()
        try:
            with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
                 patch("hermes_cli.main.subprocess.run") as mock_run:
                result = build(web_dir)
        finally:
            t.join()

        assert result is True
        mock_run.assert_not_called()  # fresh after the wait -> no rebuild

    def test_lock_file_is_gitignored(self):
        gitignore = Path(__file__).resolve().parents[2] / ".gitignore"
        assert ".web_ui_build.lock" in gitignore.read_text(encoding="utf-8")


def _link_shims(bin_dir: Path, *names: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (bin_dir / name).touch()


class TestWebBuildToolchainReady:
    """A tree is ready when the build can resolve tsc AND vite from any root.

    ``npm run build`` searches ``node_modules/.bin`` from the script's own
    package up through every ancestor, so a shim in either place counts.
    """

    def test_missing_toolchain_is_not_ready(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        assert _web_build_toolchain_ready(web_dir, tmp_path) is False


    def test_hoisted_shims_at_workspace_root_are_ready(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        _link_shims(tmp_path / "node_modules" / ".bin", "tsc", "vite")
        assert _web_build_toolchain_ready(web_dir, tmp_path) is True


    @pytest.mark.parametrize("shim", ["tsc.cmd", "tsc.ps1", "tsc.exe"])
    def test_windows_shim_extensions_count(self, tmp_path, shim):
        web_dir, _ = _make_web_dir(tmp_path)
        _link_shims(tmp_path / "node_modules" / ".bin", shim, "vite.cmd")
        assert _web_build_toolchain_ready(web_dir, tmp_path) is True


class TestWebToolchainRoots:
    def test_searches_the_package_and_its_workspace_root(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        assert _web_toolchain_roots(web_dir) == (web_dir, tmp_path)


class TestMissingWebBuildTool:
    """Every shell words an unresolvable binary differently."""

    @pytest.mark.parametrize(
        "output,expected",
        [
            ("sh: 1: tsc: not found\nnpm error code 127", "tsc"),
            ("bash: line 1: vite: command not found", "vite"),
            ("'tsc' is not recognized as an internal or external command", "tsc"),
            ("error TS2307: Cannot find module './x'", None),
            ("", None),
        ],
    )
    def test_detects_the_unresolvable_tool(self, output, expected):
        assert _missing_web_build_tool(output) == expected


class TestBuildRecoversFromMissingToolchain:
    def test_reinstalls_and_retries_when_the_build_cannot_resolve_tsc(self, tmp_path):
        """The generic retry reruns the same command, so it can't fix this alone."""
        web_dir, _ = _make_web_dir(tmp_path)
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        install_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_fail = __import__("subprocess").CompletedProcess(
            [], 127, stdout="sh: 1: tsc: not found\n", stderr=""
        )
        build_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")

        with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
             patch("hermes_cli.main._run_with_idle_timeout", side_effect=[build_fail, build_ok]) as mock_build, \
             patch("hermes_cli.main._web_ui_build_needed", return_value=True), \
             patch("hermes_cli.main._write_web_ui_build_stamp"), \
             patch("hermes_cli.main._time.sleep"):
            result = _build_web_ui(web_dir)

        assert result is True
        assert mock_install.call_count == 2
        assert mock_build.call_count == 2

    def test_healthy_tree_builds_without_an_extra_install(self, tmp_path):
        """No pre-build probing: a build that works is never second-guessed."""
        web_dir, _ = _make_web_dir(tmp_path)
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        _link_shims(web_dir / "node_modules" / ".bin", "tsc", "vite")
        install_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        build_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")

        with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
             patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
             patch("hermes_cli.main._run_with_idle_timeout", return_value=build_ok) as mock_build, \
             patch("hermes_cli.main._web_ui_build_needed", return_value=True), \
             patch("hermes_cli.main._write_web_ui_build_stamp"):
            result = _build_web_ui(web_dir)

        assert result is True
        assert mock_install.call_count == 1
        assert mock_build.call_count == 1



class TestNpmCiRetryOnInstallScriptFailure:
    """Teknium's review (2026-07-14) flagged that the retry path must
    preserve the deterministic contract:

    1. When ``npm ci`` fails with an install-script error, retry
       ``npm ci --ignore-scripts`` (NOT ``npm install --ignore-scripts`` —
       the previous version mutated committed lockfiles).
    2. Every subprocess call (initial + retry) must use ``env=run_env``
       (the ``CI=1``-forced env), not the caller's raw env. The previous
       version used ``env=env`` on the retry, breaking the CI=1 invariant.

    These tests exercise the retry path's command selection and env
    preservation — the two behaviors the previous version got wrong.
    """

    def _failed_ci(self):
        import subprocess
        return subprocess.CompletedProcess(
            [], 1, stdout="", stderr="info run electron-winstaller@5.4.0 install",
        )

    def test_npm_ci_install_script_failure_retries_npm_ci_with_ignore_scripts(self, tmp_path):
        """Lockfile-present path: install-script failure on `npm ci` →
        retry `npm ci --ignore-scripts`, NOT `npm install --ignore-scripts`.
        The retry preserves the lockfile-respecting contract."""
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        ci_fail = self._failed_ci()
        ci_retry_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.subprocess.run", side_effect=[ci_fail, ci_retry_ok]) as mock_run:
            result = _run_npm_install_deterministic("/usr/bin/npm", web_dir)

        assert result.returncode == 0
        # Two calls: initial npm ci (failed), retry npm ci --ignore-scripts (ok)
        assert mock_run.call_count == 2
        retry_args, _ = mock_run.call_args_list[1]
        retry_cmd = retry_args[0]
        # MUST be npm ci --ignore-scripts, NOT npm install --ignore-scripts.
        # The previous version did `install_cmd + ["--ignore-scripts"]` which
        # bubbled up to `npm install --ignore-scripts` on the lockfile-miss
        # fallback path — that was the bug Teknium flagged.
        assert retry_cmd[:2] == ["/usr/bin/npm", "ci"]
        assert "--ignore-scripts" in retry_cmd
        assert "install" not in retry_cmd[1]  # not `npm install --ignore-scripts`

    def test_npm_ci_retry_uses_run_env_with_ci_1(self, tmp_path):
        """The retry's env must equal `run_env` (CI=1 forced), not the
        caller's raw env. The previous version used `env=env` on the
        retry, dropping the CI=1 invariant for unicode-animations' postinstall
        tty animation bypass (nix/lib.nix npm ci hooks, TUI install path)."""
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        ci_fail = self._failed_ci()
        ci_retry_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.subprocess.run", side_effect=[ci_fail, ci_retry_ok]) as mock_run:
            _run_npm_install_deterministic(
                "/usr/bin/npm",
                web_dir,
                env={"PYTHON": "/nix/store/python"},
            )

        _, retry_kwargs = mock_run.call_args_list[1]
        # Both calls must use run_env: the CI=1-forced env built from the
        # caller's env + os.environ. Critically, the retry must have
        # CI=1 set, not be the raw caller's env.
        assert retry_kwargs["env"]["CI"] == "1"
        assert retry_kwargs["env"]["PYTHON"] == "/nix/store/python"
        # The two calls (initial + retry) must use the same env object —
        # if the retry used `env=env`, this comparison would show a
        # CI=1-forced dict vs the raw caller's dict.
        initial_env = mock_run.call_args_list[0][1]["env"]
        assert initial_env == retry_kwargs["env"]
        assert initial_env is retry_kwargs["env"] or initial_env == retry_kwargs["env"]

    def test_npm_ci_non_script_failure_skips_retry_and_falls_to_install(self, tmp_path):
        """If `npm ci` fails for a NON-install-script reason (lockfile
        conflict, network error, etc.), do NOT retry with --ignore-scripts.
        Fall through directly to `npm install`."""
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        ci_fail_non_script = __import__("subprocess").CompletedProcess(
            [], 1, stdout="", stderr="npm ERR! code E405 Lockfile has conflict entries",
        )
        install_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.subprocess.run", side_effect=[ci_fail_non_script, install_ok]) as mock_run:
            result = _run_npm_install_deterministic("/usr/bin/npm", web_dir)

        assert result.returncode == 0
        # Two calls: initial npm ci (failed), then `npm install` (no --ignore-scripts retry)
        assert mock_run.call_count == 2
        install_args, _ = mock_run.call_args_list[1]
        install_cmd = install_args[0]
        assert install_cmd[:2] == ["/usr/bin/npm", "install"]
        assert "--ignore-scripts" not in install_cmd

    def test_npm_install_fallback_also_retries_with_ignore_scripts(self, tmp_path):
        """The `npm install` fallback (lockfile-out-of-sync path) also
        retries with --ignore-scripts on install-script failure. The retry
        must use run_env, not the caller's raw env."""
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        ci_fail_non_script = __import__("subprocess").CompletedProcess(
            [], 1, stdout="", stderr="lockfile out of sync",
        )
        install_fail_script = __import__("subprocess").CompletedProcess(
            [], 1, stdout="", stderr="info run electron-winstaller@5.4.0 install",
        )
        install_retry_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch(
            "hermes_cli.main.subprocess.run",
            side_effect=[ci_fail_non_script, install_fail_script, install_retry_ok],
        ) as mock_run:
            result = _run_npm_install_deterministic(
                "/usr/bin/npm", web_dir, env={"PYTHON": "/nix/store/python"},
            )

        assert result.returncode == 0
        # Three calls: npm ci (failed), npm install (failed script), npm install --ignore-scripts (ok)
        assert mock_run.call_count == 3
        retry_call = mock_run.call_args_list[2]
        retry_args, retry_kwargs = retry_call.args, retry_call.kwargs
        retry_cmd = retry_args[0]
        # Retry command must be `npm install --ignore-scripts` (the fallback's retry path)
        assert retry_cmd[:2] == ["/usr/bin/npm", "install"]
        assert "--ignore-scripts" in retry_cmd
        # Retry env must have CI=1
        assert retry_kwargs["env"]["CI"] == "1"
        assert retry_kwargs["env"]["PYTHON"] == "/nix/store/python"


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
