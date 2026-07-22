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
    _run_npm_install_deterministic,
    _compute_web_ui_content_hash,
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

    def test_returns_true_when_dist_missing(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("export const A = 1\n")
        # Even with a matching stamp, a missing dist forces a build.
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_true_when_dist_present_but_no_stamp(self, tmp_path):
        """First run after upgrade to content-hash: no stamp -> one rebuild."""
        web_dir, dist_dir = _make_web_dir(tmp_path)
        (dist_dir / ".vite").mkdir(parents=True, exist_ok=True)
        (dist_dir / ".vite" / "manifest.json").write_text("{}")
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_false_when_stamp_matches_source(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("export const A = 1\n")
        (dist_dir / ".vite").mkdir(parents=True, exist_ok=True)
        (dist_dir / ".vite" / "manifest.json").write_text("{}")
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is False

    def test_falls_back_to_index_html_sentinel(self, tmp_path):
        """When the vite manifest is absent, index.html is the sentinel."""
        web_dir, dist_dir = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "main.ts").write_text("console.log(1)\n")
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "index.html").write_text("<html></html>")
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is False

    def test_web_dist_dir_not_web_dist_subdir(self, tmp_path):
        """Regression: sentinel must be in hermes_cli/web_dist/, NOT web/dist/."""
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("x\n")
        self._stamp_current(web_dir)
        # A manifest in the WRONG location (web/dist/) must not count as fresh.
        wrong = web_dir / "dist" / ".vite" / "manifest.json"
        wrong.parent.mkdir(parents=True, exist_ok=True)
        wrong.write_text("{}")
        # Correct location (hermes_cli/web_dist/) is empty -> still needs build.
        assert _web_ui_build_needed(web_dir) is True

    def test_returns_true_when_source_content_changes(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        src = web_dir / "src" / "App.tsx"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("export const A = 1\n")
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "index.html").write_text("<html></html>")
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is False
        src.write_text("export const A = 2\n")  # content edit
        assert _web_ui_build_needed(web_dir) is True

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

    def test_root_package_lock_content_change_is_stale(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "main.ts").write_text("console.log(1)\n")
        lock = tmp_path / "package-lock.json"
        lock.write_text('{"v": 1}')
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "index.html").write_text("<html></html>")
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is False
        lock.write_text('{"v": 2}')  # dependency change
        assert _web_ui_build_needed(web_dir) is True

    def test_gitignored_paths_excluded_from_hash(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        (tmp_path / ".gitignore").write_text("node_modules/\ndist/\n")
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("x\n")
        (dist_dir / ".vite").mkdir(parents=True, exist_ok=True)
        (dist_dir / ".vite" / "manifest.json").write_text("{}")
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is False
        # A new file under an ignored dir must not flip staleness.
        nm = web_dir / "node_modules" / "react" / "index.js"
        nm.parent.mkdir(parents=True, exist_ok=True)
        nm.write_text("module.exports = {}\n")
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

    def test_malformed_non_object_stamp_forces_rebuild(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("export const A = 1\n")
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "index.html").write_text("<html></html>")
        stamp = _web_ui_stamp_path()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("[]")
        assert _web_ui_build_needed(web_dir) is True


class TestBuildWebUISkipsWhenFresh:

    def test_skips_npm_when_dist_is_fresh(self, tmp_path):
        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / ".vite" / "manifest.json")
        # Record a stamp matching current source so the build is skipped.
        root = web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent
        _write_web_ui_build_stamp(root, web_dir)

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

    def test_npm_ci_forces_include_dev(self, tmp_path):
        """`npm ci` must pass --include=dev so an inherited NODE_ENV=production
        (e.g. from a container shell, or the bundled TUI launcher which sets
        NODE_ENV=production on its subprocess env) or an npm `omit=dev` config
        can't silently strip the build toolchain (tsc/vite/electron-builder),
        which otherwise fails the web/desktop build with `tsc: command not
        found` (exit 127) despite the install exiting 0."""
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        mock_cp = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.subprocess.run", return_value=mock_cp) as mock_run:
            _run_npm_install_deterministic("/usr/bin/npm", web_dir)

        args, _ = mock_run.call_args
        cmd = args[0]
        assert cmd[:2] == ["/usr/bin/npm", "ci"]
        assert "--include=dev" in cmd

    def test_npm_install_fallback_forces_include_dev_and_no_save(self, tmp_path):
        """When `npm ci` fails (lockfile out of sync) the `npm install`
        fallback must still force --include=dev (same NODE_ENV rationale as
        above) and must pass --no-save so the fallback never rewrites the
        committed lockfile — a drifted lockfile makes every future `npm ci`
        fail, a self-reinforcing cycle that keeps devDeps from installing."""
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "package-lock.json").write_text("{}", encoding="utf-8")

        ci_fail = __import__("subprocess").CompletedProcess([], 1, stdout="", stderr="lockfile out of sync")
        install_ok = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr="")
        with patch("hermes_cli.main.subprocess.run", side_effect=[ci_fail, install_ok]) as mock_run:
            result = _run_npm_install_deterministic("/usr/bin/npm", web_dir)

        assert result.returncode == 0
        assert mock_run.call_count == 2
        install_args, _ = mock_run.call_args_list[1]
        install_cmd = install_args[0]
        assert install_cmd[:2] == ["/usr/bin/npm", "install"]
        assert "--include=dev" in install_cmd
        assert "--no-save" in install_cmd

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
        assert args[0] == ["/usr/bin/npm", "ci", "--include=dev", "--silent"]
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
            "--include=dev",
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
        assert args[0] == ["/usr/bin/npm", "ci", "--include=dev", "--workspace", "web", "--silent"]
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


class TestBuildWebUIFlock:
    """Cross-process build serialization (salvaged from PR #63455).

    One process builds under an exclusive flock on <root>/.web_ui_build.lock;
    contenders either serve the existing (possibly stale) dist or, when no
    dist exists yet, block until the builder finishes. The staleness walk
    itself runs inside _do_build_web_ui, i.e. under the lock, so a process
    that queued behind a successful build skips the rebuild.
    """

    def test_contended_lock_with_dist_serves_stale_without_building(self, tmp_path):
        import fcntl
        from hermes_cli.main import _build_web_ui as build

        web_dir, dist_dir = _make_web_dir(tmp_path)
        _touch(dist_dir / "index.html")

        lock_path = tmp_path / ".web_ui_build.lock"
        holder = open(lock_path, "a")
        try:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
            with patch("hermes_cli.main._do_build_web_ui") as mock_do:
                result = build(web_dir)
        finally:
            holder.close()

        assert result is True
        mock_do.assert_not_called()  # served existing dist, no second build

    def test_uncontended_lock_builds_and_creates_lock_file(self, tmp_path):
        from hermes_cli.main import _build_web_ui as build

        web_dir, _ = _make_web_dir(tmp_path)
        with patch("hermes_cli.main._do_build_web_ui", return_value=True) as mock_do:
            result = build(web_dir)

        assert result is True
        mock_do.assert_called_once()
        assert (tmp_path / ".web_ui_build.lock").exists()

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
