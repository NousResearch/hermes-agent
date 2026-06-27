"""Tests for hermes_bootstrap — Windows UTF-8 stdio shim.

The bootstrap module is imported at the top of every Hermes entry point
(hermes, hermes-agent, hermes-acp, gateway, batch_runner, cli.py).  It
fixes Python's Windows UTF-8 defaults so print("café") doesn't crash and
subprocess children inherit UTF-8 mode.

Key invariants covered by these tests:

  1. Windows: env vars get set, stdio reconfigured, non-ASCII print works
  2. POSIX: complete no-op (we don't touch LANG/LC_* or anything else)
  3. Idempotent: safe to call multiple times
  4. Respects user opt-out: if the user explicitly sets PYTHONUTF8=0 or
     PYTHONIOENCODING=something-else, we leave those alone
  5. Load order: every Hermes entry point imports hermes_bootstrap as its
     first non-docstring import (before anything that might do file I/O
     or print to stdout)
"""

from __future__ import annotations

import io
import os
import platform
import subprocess
import sys
import textwrap

import pytest


# Import the module under test via an import-time side-effect check path.
# We need to be able to reset its state between tests, so we import it
# fresh in each test that manipulates _IS_WINDOWS.
def _fresh_import():
    """Return a freshly-imported hermes_bootstrap module.

    Drops any cached copy from sys.modules first so module-level code
    runs again and the platform check re-evaluates.
    """
    sys.modules.pop("hermes_bootstrap", None)
    import hermes_bootstrap  # noqa: WPS433
    return hermes_bootstrap


class TestWindowsBehavior:
    """Windows: the bootstrap does its job."""

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-specific behavior",
    )
    def test_env_vars_set_on_windows(self, monkeypatch):
        # Clear any pre-existing values and re-run bootstrap.
        monkeypatch.delenv("PYTHONUTF8", raising=False)
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        hb = _fresh_import()
        # Module-level apply_windows_utf8_bootstrap() ran during import.
        assert os.environ.get("PYTHONUTF8") == "1"
        assert os.environ.get("PYTHONIOENCODING") == "utf-8"
        assert hb._bootstrap_applied is True

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-specific behavior",
    )
    def test_stdout_reconfigured_to_utf8_on_windows(self):
        # The live process's stdout should now be UTF-8 (the Hermes CLI
        # runs on Windows with a pytest console that's cp1252 by default).
        # If reconfigure succeeded, sys.stdout.encoding is 'utf-8'.
        _fresh_import()
        # pytest may capture stdout, which makes encoding check flaky —
        # so instead verify the reconfigure call succeeded on the real
        # stream by attempting the failure case.
        out = sys.stdout
        reconfigure = getattr(out, "reconfigure", None)
        if reconfigure is None:
            pytest.skip("pytest replaced sys.stdout with a non-reconfigurable stream")
        # After bootstrap, encoding should be utf-8 (or the reconfigure
        # skipped because pytest's capture already set it to utf-8).
        assert out.encoding.lower() in {"utf-8", "utf8"}, (
            f"stdout encoding is {out.encoding!r} — bootstrap should have "
            "reconfigured it to UTF-8"
        )

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-specific behavior",
    )
    def test_child_process_inherits_utf8_mode(self):
        """A subprocess spawned from this process should inherit
        PYTHONUTF8=1 and be able to print non-ASCII to stdout."""
        _fresh_import()
        # Non-ASCII chars that would crash under cp1252: arrow, emoji.
        script = textwrap.dedent("""
            import sys
            print("em-dash \\u2014 arrow \\u2192 emoji \\U0001f680")
            sys.exit(0)
        """).strip()
        # Don't pass env= — let the child inherit os.environ, which
        # now contains PYTHONUTF8=1 courtesy of the bootstrap.
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"Child crashed printing non-ASCII despite UTF-8 bootstrap:\n"
            f"  stdout: {result.stdout!r}\n"
            f"  stderr: {result.stderr!r}"
        )
        decoded = result.stdout.decode("utf-8")
        assert "\u2014" in decoded
        assert "\u2192" in decoded
        assert "\U0001f680" in decoded


class TestUserOptOut:
    """If the user has explicitly set PYTHONUTF8 / PYTHONIOENCODING in
    their environment, we respect that (setdefault, not overwrite)."""

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Only meaningful on Windows where we'd otherwise set these",
    )
    def test_user_pythonutf8_zero_preserved(self, monkeypatch):
        monkeypatch.setenv("PYTHONUTF8", "0")
        _fresh_import()
        assert os.environ["PYTHONUTF8"] == "0", (
            "bootstrap must not overwrite an explicit user setting"
        )

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Only meaningful on Windows where we'd otherwise set these",
    )
    def test_user_pythonioencoding_preserved(self, monkeypatch):
        monkeypatch.setenv("PYTHONIOENCODING", "latin-1")
        _fresh_import()
        assert os.environ["PYTHONIOENCODING"] == "latin-1"


class TestPosixNoOp:
    """POSIX: zero behavior change.  We don't touch LANG, LC_*, or any
    stdio.  The goal is that Linux/macOS behave identically before and
    after this module is imported."""

    def test_noop_on_fake_posix(self, monkeypatch):
        """Even when imported, the bootstrap function must return False
        and leave env untouched when _IS_WINDOWS is False."""
        hb = _fresh_import()
        # Reset + fake POSIX
        hb._IS_WINDOWS = False
        hb._bootstrap_applied = False
        monkeypatch.delenv("PYTHONUTF8", raising=False)
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)

        result = hb.apply_windows_utf8_bootstrap()

        assert result is False
        assert "PYTHONUTF8" not in os.environ
        assert "PYTHONIOENCODING" not in os.environ
        assert hb._bootstrap_applied is False

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Real POSIX required for this check",
    )
    def test_real_posix_bootstrap_is_noop(self, monkeypatch):
        """On actual Linux/macOS, importing the module must not set
        PYTHONUTF8 or reconfigure stdio."""
        monkeypatch.delenv("PYTHONUTF8", raising=False)
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        hb = _fresh_import()
        assert hb._bootstrap_applied is False
        assert "PYTHONUTF8" not in os.environ
        assert "PYTHONIOENCODING" not in os.environ


class TestIdempotence:
    """Calling apply_windows_utf8_bootstrap() multiple times must be safe."""

    def test_second_call_returns_false(self):
        hb = _fresh_import()
        # First call already happened at import time.
        result = hb.apply_windows_utf8_bootstrap()
        assert result is False, (
            "Second call should return False (idempotent no-op)"
        )

    def test_no_exceptions_on_repeated_calls(self):
        hb = _fresh_import()
        for _ in range(5):
            hb.apply_windows_utf8_bootstrap()


class TestStdioReconfigureErrorHandling:
    """If sys.stdout/stderr/stdin have been replaced with streams that
    don't support reconfigure (e.g. by a test harness), the bootstrap
    must degrade gracefully rather than crash."""

    def test_non_reconfigurable_stream_does_not_crash(self, monkeypatch):
        """Replace sys.stdout with a BytesIO (no reconfigure method),
        then run the bootstrap and make sure it doesn't raise."""
        hb = _fresh_import()
        hb._IS_WINDOWS = True
        hb._bootstrap_applied = False

        fake = io.BytesIO()  # no .reconfigure attribute
        monkeypatch.setattr(sys, "stdout", fake)
        try:
            # Must not raise.
            hb.apply_windows_utf8_bootstrap()
        except Exception as exc:
            pytest.fail(f"bootstrap raised on non-reconfigurable stdout: {exc}")

    def test_reconfigure_oserror_is_caught(self, monkeypatch):
        """If reconfigure() itself raises (closed stream, etc.), swallow
        the error — the env-var half of the fix still applies."""
        hb = _fresh_import()
        hb._IS_WINDOWS = True
        hb._bootstrap_applied = False

        class _BrokenStream:
            encoding = "utf-8"
            def reconfigure(self, **kwargs):
                raise OSError("simulated: stream already closed")

        monkeypatch.setattr(sys, "stdout", _BrokenStream())
        monkeypatch.setattr(sys, "stderr", _BrokenStream())
        # Must not raise.
        hb.apply_windows_utf8_bootstrap()


class TestWindowsPlatformProbeGuard:
    def test_windows_bootstrap_disables_platform_syscmd_subprocess(self):
        hb = _fresh_import()
        hb._IS_WINDOWS = True
        hb._bootstrap_applied = False

        original = getattr(platform, "_syscmd_ver", None)
        try:
            hb.apply_windows_utf8_bootstrap()

            assert platform._syscmd_ver("Windows", "", "") == ("Windows", "", "")
        finally:
            if original is not None:
                platform._syscmd_ver = original


class TestDetachOrphanConsole:
    """detach_orphan_console() frees a solo-owned console (the uv pythonw→python
    phantom) but leaves a shared interactive console attached, and is a pure
    no-op on POSIX. It is intentionally NOT run at import time."""

    def test_noop_on_posix(self):
        hb = _fresh_import()
        hb._IS_WINDOWS = False
        assert hb.detach_orphan_console() is False

    def test_not_called_at_import_time(self):
        # The FreeConsole catch-all must be opt-in per background entry point,
        # never an import side effect (would detach the interactive CLI/TUI).
        import pathlib
        src = pathlib.Path(_fresh_import().__file__).read_text(encoding="utf-8")
        body = src.split("def detach_orphan_console")[0]
        assert "FreeConsole" not in body, (
            "FreeConsole must live only inside detach_orphan_console(), not in "
            "apply_windows_utf8_bootstrap() / module import path"
        )

    def _fake_ctypes(self, monkeypatch, window, nproc):
        import ctypes

        class _K:
            def __init__(self):
                self.freed = False
            def GetConsoleWindow(self):
                return window
            def GetConsoleProcessList(self, buf, n):
                return nproc
            def FreeConsole(self):
                self.freed = True

        k = _K()
        monkeypatch.setattr(ctypes, "windll", type("_W", (), {"kernel32": k})(), raising=False)
        return k

    def test_frees_when_solo_owner(self, monkeypatch):
        hb = _fresh_import()
        hb._IS_WINDOWS = True
        k = self._fake_ctypes(monkeypatch, window=1, nproc=1)
        assert hb.detach_orphan_console() is True
        assert k.freed is True

    def test_leaves_shared_console_attached(self, monkeypatch):
        hb = _fresh_import()
        hb._IS_WINDOWS = True
        k = self._fake_ctypes(monkeypatch, window=1, nproc=2)
        assert hb.detach_orphan_console() is False
        assert k.freed is False

    def test_noop_without_console(self, monkeypatch):
        hb = _fresh_import()
        hb._IS_WINDOWS = True
        k = self._fake_ctypes(monkeypatch, window=0, nproc=1)
        assert hb.detach_orphan_console() is False
        assert k.freed is False


class TestEntryPointsImportBootstrap:
    """Every Hermes entry point must import hermes_bootstrap as its
    first non-docstring import.  We check this by scanning source files
    rather than invoking the entry points (which would require a full
    agent context)."""

    # Entry points that invoke Hermes as a process.  Each one must
    # import hermes_bootstrap before doing any file I/O or stdout writes.
    ENTRY_POINTS = [
        "hermes_cli/main.py",   # hermes CLI (console_script)
        "run_agent.py",          # hermes-agent (console_script)
        "acp_adapter/entry.py",  # hermes-acp (console_script)
        "gateway/run.py",        # gateway
        "batch_runner.py",       # batch mode
        "cli.py",                # legacy direct-launch CLI
    ]

    @pytest.mark.parametrize("path", ENTRY_POINTS)
    def test_entry_point_imports_bootstrap(self, path):
        """The file must contain 'import hermes_bootstrap' and that
        line must appear before the first 'import' of anything else.

        We're lenient about the docstring (can be arbitrarily long) and
        about comment lines — just need to verify the first import
        statement is the bootstrap.

        Also lenient about a try/except wrapper around the import: entry
        points may guard the import against ``ModuleNotFoundError`` so a
        half-finished ``hermes update`` (git-reset landed new code but
        ``uv pip install -e .`` didn't finish re-registering
        ``hermes_bootstrap`` as a top-level module) leaves hermes
        recoverable instead of crashing on every invocation.  When the
        first top-level node is such a guarded-import block, we peek
        inside it to verify bootstrap is the imported module.
        """
        # Resolve relative to the hermes-agent repo root.  Tests live
        # at tests/test_hermes_bootstrap.py, so go up one dir.
        import pathlib
        here = pathlib.Path(__file__).resolve()
        repo_root = here.parent.parent  # tests/ -> repo root
        full_path = repo_root / path
        assert full_path.exists(), f"entry point missing: {full_path}"

        source = full_path.read_text(encoding="utf-8")

        # Find the first non-comment, non-blank line that starts with
        # 'import ' or 'from ', or a Try block whose body is the import.
        import ast
        tree = ast.parse(source)

        first_import_node = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                first_import_node = node
                break
            # Accept a guarded-import Try block where the body is a lone
            # Import node — this is the recovery-friendly form that lets
            # hermes start even when hermes_bootstrap hasn't been
            # re-registered in the venv yet.
            if isinstance(node, ast.Try) and len(node.body) == 1 and isinstance(
                node.body[0], (ast.Import, ast.ImportFrom)
            ):
                first_import_node = node.body[0]
                break

        assert first_import_node is not None, (
            f"{path}: no top-level imports found at all"
        )

        if isinstance(first_import_node, ast.Import):
            first_import_name = first_import_node.names[0].name
        else:  # ImportFrom
            first_import_name = first_import_node.module or ""

        assert first_import_name == "hermes_bootstrap", (
            f"{path}: first top-level import is {first_import_name!r}, "
            f"but it must be 'hermes_bootstrap' so UTF-8 stdio is "
            f"configured before anything else initializes.  Move the "
            f"'import hermes_bootstrap' line to be the first import."
        )


class TestHardenImportPath:
    """harden_import_path() must keep a same-named package in the launch
    directory from shadowing Hermes's own top-level modules — covering both
    the relative ('' / '.') and absolute-path forms the cwd can take on
    sys.path (issue #51286)."""

    def _run(self, hb, path_seed, env=None):
        original = sys.path[:]
        original_env = os.environ.get("HERMES_PYTHON_SRC_ROOT")
        try:
            sys.path[:] = path_seed
            if env is not None:
                os.environ["HERMES_PYTHON_SRC_ROOT"] = env
            elif "HERMES_PYTHON_SRC_ROOT" in os.environ:
                del os.environ["HERMES_PYTHON_SRC_ROOT"]
            hb.harden_import_path(src_root="/opt/hermes")
            return sys.path[:]
        finally:
            sys.path[:] = original
            if original_env is None:
                os.environ.pop("HERMES_PYTHON_SRC_ROOT", None)
            else:
                os.environ["HERMES_PYTHON_SRC_ROOT"] = original_env

    def test_relative_cwd_forms_removed(self):
        hb = _fresh_import()
        result = self._run(hb, ["", ".", "/opt/hermes", "/usr/lib/python"])
        assert "" not in result
        assert "." not in result

    def test_src_root_forced_to_front(self):
        hb = _fresh_import()
        result = self._run(hb, ["", "/opt/hermes", "/usr/lib/python"])
        assert result[0] == "/opt/hermes"

    def test_absolute_cwd_path_loses_to_src_root(self):
        # The real #51286 bug: the launch dir is present as its own absolute
        # path (venv activation / a project on PYTHONPATH), ahead of the
        # Hermes root.  The guard must relocate Hermes to the front.
        hb = _fresh_import()
        result = self._run(hb, ["/home/user/tg-ws-proxy", "/opt/hermes"])
        assert result[0] == "/opt/hermes"
        # The cwd absolute path may still appear (it can hold legit deps),
        # but only AFTER the Hermes root.
        assert result.index("/opt/hermes") < result.index("/home/user/tg-ws-proxy")

    def test_src_root_not_duplicated(self):
        hb = _fresh_import()
        result = self._run(hb, ["/opt/hermes", "/opt/hermes", ""])
        assert result.count("/opt/hermes") == 1

    def test_env_var_used_when_no_arg(self):
        hb = _fresh_import()
        original = sys.path[:]
        original_env = os.environ.get("HERMES_PYTHON_SRC_ROOT")
        try:
            sys.path[:] = ["", "/cwd/proj", "/usr/lib"]
            os.environ["HERMES_PYTHON_SRC_ROOT"] = "/env/hermes"
            hb.harden_import_path()
            assert sys.path[0] == "/env/hermes"
        finally:
            sys.path[:] = original
            if original_env is None:
                os.environ.pop("HERMES_PYTHON_SRC_ROOT", None)
            else:
                os.environ["HERMES_PYTHON_SRC_ROOT"] = original_env

    def test_defaults_to_module_dir(self):
        # With neither arg nor env var, the helper anchors on the bootstrap
        # module's own directory — the repo root for shipped entry points.
        hb = _fresh_import()
        original = sys.path[:]
        original_env = os.environ.get("HERMES_PYTHON_SRC_ROOT")
        try:
            sys.path[:] = ["", "/somewhere/else"]
            os.environ.pop("HERMES_PYTHON_SRC_ROOT", None)
            hb.harden_import_path()
            expected = os.path.dirname(os.path.abspath(hb.__file__))
            assert sys.path[0] == expected
        finally:
            sys.path[:] = original
            if original_env is not None:
                os.environ["HERMES_PYTHON_SRC_ROOT"] = original_env
