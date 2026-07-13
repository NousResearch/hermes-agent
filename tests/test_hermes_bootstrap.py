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
import subprocess
import sys
import textwrap
import unittest.mock as mock

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


class TestYamlCSafeShim:
    """Bootstrap rebinds yaml.safe_load to use the libyaml CSafeLoader
    on import.  PyYAML's pure-Python parser non-deterministically
    corrupts CPython object/refcount state under sustained load on
    some WSL2 / glibc combinations; the C-backed parser is immune.
    See module docstring for full rationale."""

    def test_safe_load_uses_csafe_loader_after_bootstrap(self):
        """After hermes_bootstrap import, yaml.safe_load must route
        through the libyaml C parser.  We check the rebound function's
        identifying attributes — it lives in hermes_bootstrap.py and
        invokes yaml.load with Loader=CSafeLoader."""
        _fresh_import()
        import yaml

        if not hasattr(yaml, "CSafeLoader"):
            pytest.skip("libyaml C extension not installed; shim is a no-op")

        # The rebound function is defined inside hermes_bootstrap.py.
        assert yaml.safe_load.__module__ == "hermes_bootstrap", (
            f"yaml.safe_load.__module__ is {yaml.safe_load.__module__!r}; "
            f"expected 'hermes_bootstrap' (shim should have rebound it)"
        )
        assert yaml.safe_load_all.__module__ == "hermes_bootstrap"

    def test_safe_load_round_trips_basic_yaml(self):
        """Sanity: the rebound safe_load actually parses YAML correctly
        (it's not just a dummy stub)."""
        _fresh_import()
        import yaml

        text = "a: 1\nb:\n  - x\n  - y\nc: hello\n"
        result = yaml.safe_load(text)
        assert result == {"a": 1, "b": ["x", "y"], "c": "hello"}

    def test_yaml_shim_is_idempotent(self):
        """Calling apply_yaml_csafe_shim twice must be a no-op the
        second time (returns False)."""
        hb = _fresh_import()
        # First call already happened at import time.
        assert hb.apply_yaml_csafe_shim() is False

    def test_shim_no_op_when_pyyaml_missing(self, monkeypatch):
        """If PyYAML isn't importable, the shim must silently no-op
        rather than raising — Hermes installs without YAML extras
        should still launch.

        Setting ``sys.modules['yaml'] = None`` makes ``import yaml``
        raise a real ImportError ("import of yaml halted"), so this
        exercises the shim's except branch, not just a missing
        attribute.
        """
        hb = _fresh_import()
        hb._yaml_shim_applied = False

        monkeypatch.setitem(sys.modules, "yaml", None)
        result = hb.apply_yaml_csafe_shim()

        assert result is False
        assert hb._yaml_shim_applied is False

    def test_shim_preserves_independently_customized_function(self, monkeypatch):
        """safe_load and safe_load_all are guarded independently: if
        someone customized only safe_load_all before the shim runs,
        that customization survives while safe_load still gets the
        CSafeLoader rebind."""
        hb = _fresh_import()

        import yaml

        if not hasattr(yaml, "CSafeLoader"):
            pytest.skip("libyaml C extension not installed; shim is a no-op")

        def pristine_safe_load(stream):
            raise NotImplementedError

        pristine_safe_load.__module__ = "yaml"

        def custom_safe_load_all(stream):
            raise NotImplementedError

        monkeypatch.setattr(yaml, "safe_load", pristine_safe_load)
        monkeypatch.setattr(yaml, "safe_load_all", custom_safe_load_all)
        hb._yaml_shim_applied = False

        assert hb.apply_yaml_csafe_shim() is True
        assert yaml.safe_load.__module__ == "hermes_bootstrap", (
            "the un-customized safe_load should still get rebound"
        )
        assert yaml.safe_load_all is custom_safe_load_all, (
            "a separately customized safe_load_all must not be clobbered"
        )

    def test_shim_no_op_when_libyaml_missing(self, monkeypatch):
        """If PyYAML is installed but the C extension isn't (no
        ``yaml.CSafeLoader``), the shim must silently no-op rather
        than rebinding to something that doesn't exist."""
        hb = _fresh_import()
        hb._yaml_shim_applied = False

        import yaml

        # Stand up a fake yaml module that pretends libyaml isn't there.
        class _FakeYaml:
            __name__ = "yaml"
            safe_load = yaml.safe_load
            # No CSafeLoader attribute.

        monkeypatch.setitem(sys.modules, "yaml", _FakeYaml())
        try:
            result = hb.apply_yaml_csafe_shim()
        finally:
            sys.modules["yaml"] = yaml

        assert result is False
        assert hb._yaml_shim_applied is False
