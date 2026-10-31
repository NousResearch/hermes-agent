from __future__ import annotations

from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from pm.toolchain_preflight import (
    require_native_cxx_for_sync,
    _resolve_cxx,
    _compiler_on_path,
    _first_executable,
    _sync_needs_matrix_native_build,
    _build_env_uses_clang,
    _build_env_uses_clang_from_resolved,
)


# ========== Unit tests for helper functions ==========


class TestFirstExecutable:
    """Test _first_executable parsing."""

    def test_simple_command(self):
        assert _first_executable("clang++") == "clang++"

    def test_command_with_flags(self):
        assert _first_executable("clang++ -pthread -std=c++17") == "clang++"

    def test_command_with_leading_whitespace(self):
        assert _first_executable("  g++ -O2") == "g++"

    def test_empty_string(self):
        assert _first_executable("") == ""

    def test_whitespace_only(self):
        assert _first_executable("   ") == ""


class TestCompilerOnPath:
    """Test _compiler_on_path detection."""

    def test_empty_command_returns_false(self):
        assert not _compiler_on_path("", {"PATH": "/usr/bin"})

    def test_absolute_path_that_exists(self, tmp_path, monkeypatch):
        compiler = tmp_path / "clang++"
        compiler.touch()
        assert _compiler_on_path(str(compiler), {"PATH": "/usr/bin"})

    def test_absolute_path_that_doesnt_exist(self, tmp_path):
        compiler = tmp_path / "nonexistent"
        assert not _compiler_on_path(str(compiler), {"PATH": "/usr/bin"})

    def test_command_found_via_which(self, monkeypatch):
        monkeypatch.setattr(
            "pm.toolchain_preflight.shutil.which",
            lambda cmd, path=None: "/usr/bin/clang++" if cmd == "clang++" else None,
        )
        assert _compiler_on_path("clang++", {"PATH": "/usr/bin"})

    def test_command_not_found(self, monkeypatch):
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)
        assert not _compiler_on_path("clang++", {"PATH": "/usr/bin"})


class TestResolveCxx:
    """Test _resolve_cxx compiler resolution logic."""

    def test_cxx_env_direct(self):
        result = _resolve_cxx({"CXX": "g++"}, None)
        assert result == "g++"

    def test_cxx_env_with_flags(self):
        result = _resolve_cxx({"CXX": "clang++ -pthread -std=c++17"}, None)
        assert result == "clang++"

    def test_cc_env_with_clang(self):
        result = _resolve_cxx({"CC": "clang"}, None)
        assert result == "clang++"

    def test_cc_env_with_clang_versioned(self):
        result = _resolve_cxx({"CC": "clang-17"}, None)
        assert result == "clang-17"

    def test_cc_env_with_gcc_returns_none(self):
        result = _resolve_cxx({"CC": "gcc"}, None)
        assert result is None

    def test_cc_env_with_clang_and_flags(self):
        result = _resolve_cxx({"CC": "clang -O2 -Wall"}, None)
        assert result == "clang++"

    def test_cxx_takes_precedence_over_cc(self):
        result = _resolve_cxx({"CXX": "g++", "CC": "clang"}, None)
        assert result == "g++"

    def test_sysconfig_fallback_success(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            proc = Mock()
            proc.returncode = 0
            proc.stdout = "clang++\n"
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        result = _resolve_cxx({}, python)
        assert result == "clang++"

    def test_sysconfig_fallback_with_flags(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            proc = Mock()
            proc.returncode = 0
            proc.stdout = "clang++ -pthread\n"
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        result = _resolve_cxx({}, python)
        assert result == "clang++"

    def test_sysconfig_fallback_returns_none_on_failure(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            proc = Mock()
            proc.returncode = 1
            proc.stdout = ""
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        result = _resolve_cxx({}, python)
        assert result is None

    def test_sysconfig_fallback_returns_none_on_timeout(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 30)

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        result = _resolve_cxx({}, python)
        assert result is None

    def test_sysconfig_fallback_returns_none_on_oserror(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        result = _resolve_cxx({}, python)
        assert result is None

    def test_sysconfig_fallback_returns_none_when_python_is_none(self):
        result = _resolve_cxx({}, None)
        assert result is None

    def test_sysconfig_fallback_empty_output(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            proc = Mock()
            proc.returncode = 0
            proc.stdout = ""
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        result = _resolve_cxx({}, python)
        assert result is None


class TestSyncNeedsMatrixNativeBuild:
    """Test _sync_needs_matrix_native_build logic."""

    def test_matrix_not_in_extras(self, monkeypatch):
        monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **kwargs: True)
        assert not _sync_needs_matrix_native_build(["web", "google"])

    def test_matrix_in_extras_but_not_supported(self, monkeypatch):
        monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **kwargs: False)
        assert not _sync_needs_matrix_native_build(["matrix"])

    def test_matrix_in_extras_and_supported(self, monkeypatch):
        monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **kwargs: extra == "matrix")
        assert _sync_needs_matrix_native_build(["matrix"])

    def test_matrix_among_other_extras(self, monkeypatch):
        monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **kwargs: extra == "matrix")
        assert _sync_needs_matrix_native_build(["web", "matrix", "google"])


class TestBuildEnvUsesClang:
    """Test _build_env_uses_clang detection."""

    def test_cxx_with_clang(self):
        assert _build_env_uses_clang({"CXX": "clang++"}, None)

    def test_cxx_with_clang_and_flags(self):
        assert _build_env_uses_clang({"CXX": "clang++ -pthread"}, None)

    def test_cc_with_clang(self):
        assert _build_env_uses_clang({"CC": "clang"}, None)

    def test_cxx_with_gcc_returns_false(self):
        assert not _build_env_uses_clang({"CXX": "g++"}, None)

    def test_cc_with_gcc_returns_false(self):
        assert not _build_env_uses_clang({"CC": "gcc"}, None)

    def test_sysconfig_clang(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            proc = Mock()
            proc.returncode = 0
            proc.stdout = "clang++\n"
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        assert _build_env_uses_clang({}, python)

    def test_sysconfig_gcc_returns_false(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            proc = Mock()
            proc.returncode = 0
            proc.stdout = "g++\n"
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        assert not _build_env_uses_clang({}, python)


class TestBuildEnvUsesClangFromResolved:
    """Test _build_env_uses_clang_from_resolved optimization."""

    def test_cxx_with_clang(self):
        assert _build_env_uses_clang_from_resolved({"CXX": "clang++"}, "clang++")

    def test_cxx_with_clang_ignores_resolved(self):
        # When CXX is set, it takes precedence even if resolved differs
        assert _build_env_uses_clang_from_resolved({"CXX": "clang++"}, "g++")

    def test_cc_with_clang(self):
        assert _build_env_uses_clang_from_resolved({"CC": "clang"}, "clang++")

    def test_uses_resolved_when_no_env_vars(self):
        assert _build_env_uses_clang_from_resolved({}, "clang++")

    def test_resolved_gcc_returns_false(self):
        assert not _build_env_uses_clang_from_resolved({}, "g++")

    def test_resolved_none_returns_false(self):
        assert not _build_env_uses_clang_from_resolved({}, None)

    def test_gcc_env_with_gcc_resolved_returns_false(self):
        # GCC in both env and resolved
        assert not _build_env_uses_clang_from_resolved({"CXX": "g++"}, "g++")

    def test_clang_env_with_none_resolved_returns_true(self):
        assert _build_env_uses_clang_from_resolved({"CC": "clang"}, None)


# ========== Integration tests for require_native_cxx_for_sync ==========


class TestRequireNativeCxxForSync:
    """Test the main preflight check function."""

    def test_matrix_extra_without_compiler_raises(self, monkeypatch):
        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")

        with pytest.raises(RuntimeError, match="clang\\+\\+ on PATH"):
            require_native_cxx_for_sync(["matrix"], build_env={"PATH": "/usr/bin"})

    def test_matrix_extra_with_compiler_ok(self, monkeypatch):
        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr(
            "pm.toolchain_preflight.shutil.which",
            lambda cmd, path=None: "/usr/bin/clang++" if cmd == "clang++" else None,
        )
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")

        require_native_cxx_for_sync(["matrix"], build_env={"PATH": "/usr/bin"})

    def test_matrix_extra_on_macos_skips_preflight(self, monkeypatch):
        """Matrix extra on non-Linux platforms doesn't require preflight."""
        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "darwin")

        # Should not raise even without compiler on macOS
        require_native_cxx_for_sync(["matrix"], build_env={"PATH": "/usr/bin"})

    def test_matrix_extra_on_windows_skips_preflight(self, monkeypatch):
        """Matrix extra on non-Linux platforms doesn't require preflight."""
        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "win32")

        # Should not raise even without compiler on Windows
        require_native_cxx_for_sync(["matrix"], build_env={"PATH": "/usr/bin"})

    def test_clang_cxx_env_without_matrix(self, monkeypatch):
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

        with pytest.raises(RuntimeError, match="Refusing to sync"):
            require_native_cxx_for_sync([], build_env={"CXX": "clang++ -pthread", "PATH": "/usr/bin"})

    def test_clang_cxx_env_with_compiler_present(self, monkeypatch):
        monkeypatch.setattr(
            "pm.toolchain_preflight.shutil.which",
            lambda cmd, path=None: "/usr/bin/clang++" if cmd == "clang++" else None,
        )

        # Should not raise when compiler is found
        require_native_cxx_for_sync([], build_env={"CXX": "clang++", "PATH": "/usr/bin"})

    def test_clang_cc_env_without_matrix(self, monkeypatch):
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

        with pytest.raises(RuntimeError, match="Refusing to sync"):
            require_native_cxx_for_sync([], build_env={"CC": "clang", "PATH": "/usr/bin"})

    def test_gcc_env_skips_preflight(self, monkeypatch):
        """GCC environment doesn't trigger preflight."""
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

        # Should not raise for gcc
        require_native_cxx_for_sync([], build_env={"CXX": "g++", "PATH": "/usr/bin"})

    def test_unrelated_extra_skips_preflight(self, monkeypatch):
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

        require_native_cxx_for_sync(["web"], build_env={"PATH": "/usr/bin"})

    def test_empty_extras_no_clang_skips_preflight(self, monkeypatch):
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

        # Should not raise with no extras and no clang env
        require_native_cxx_for_sync([], build_env={"PATH": "/usr/bin"})

    def test_absolute_compiler_path_works(self, tmp_path, monkeypatch):
        """Absolute path to compiler is recognized."""
        compiler = tmp_path / "clang++"
        compiler.touch()
        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")

        # Should not raise when CXX points to an existing file
        require_native_cxx_for_sync(
            ["matrix"],
            build_env={"CXX": str(compiler), "PATH": "/usr/bin"},
        )

    def test_custom_path_respected(self, tmp_path, monkeypatch):
        """Custom PATH in build_env is used for lookup."""
        custom_bin = tmp_path / "bin"
        custom_bin.mkdir()
        compiler = custom_bin / "clang++"
        compiler.touch()

        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")
        monkeypatch.setattr(
            "pm.toolchain_preflight.shutil.which",
            lambda cmd, path=None: str(compiler) if cmd == "clang++" and path == str(custom_bin) else None,
        )

        # Should not raise when compiler is in custom PATH
        require_native_cxx_for_sync(
            ["matrix"],
            build_env={"PATH": str(custom_bin)},
        )

    def test_uses_os_environ_when_build_env_is_none(self, monkeypatch):
        """Falls back to os.environ when build_env is None."""
        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")
        monkeypatch.setattr(
            "pm.toolchain_preflight.shutil.which",
            lambda cmd, path=None: "/usr/bin/clang++" if cmd == "clang++" else None,
        )

        # Should not raise when compiler is found via os.environ
        require_native_cxx_for_sync(["matrix"], build_env=None)

    def test_python_parameter_used_for_sysconfig(self, tmp_path, monkeypatch):
        """Python parameter is passed to sysconfig resolution."""
        python = tmp_path / "python"
        python.touch()

        def fake_run(cmd, **kwargs):
            assert str(python) in cmd
            proc = Mock()
            proc.returncode = 0
            proc.stdout = "clang++\n"
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

        # Should raise because compiler not found, but sysconfig was attempted
        with pytest.raises(RuntimeError, match="clang\\+\\+ on PATH"):
            require_native_cxx_for_sync([], build_env={"CXX": ""}, python=python)

    def test_matrix_not_supported_on_platform_skips_preflight(self, monkeypatch):
        """Matrix extra that's not supported on this platform skips preflight."""
        monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **kwargs: False)
        monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")

        # Should not raise when matrix is not supported
        require_native_cxx_for_sync(["matrix"], build_env={"PATH": "/usr/bin"})

    def test_subprocess_called_once_not_twice(self, tmp_path, monkeypatch):
        """Performance: verify subprocess is only spawned once, not twice."""
        python = tmp_path / "python"
        python.touch()

        subprocess_calls = []

        def fake_run(cmd, **kwargs):
            subprocess_calls.append(cmd)
            proc = Mock()
            proc.returncode = 0
            proc.stdout = "clang++\n"
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.subprocess.run", fake_run)
        monkeypatch.setattr(
            "pm.extras.extra_supported",
            lambda extra, **kwargs: extra == "matrix",
        )
        monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")
        monkeypatch.setattr(
            "pm.toolchain_preflight.shutil.which",
            lambda cmd, path=None: "/usr/bin/clang++" if cmd == "clang++" else None,
        )

        # This triggers both needs_matrix and uses_clang checks
        # Before optimization, this would spawn subprocess twice
        require_native_cxx_for_sync(["matrix"], build_env={}, python=python)

        # Verify subprocess was called exactly once
        assert len(subprocess_calls) == 1


# ========== Integration tests with PythonEnvironment.sync ==========


class TestPythonEnvironmentSyncIntegration:
    """Test that PythonEnvironment.sync integrates preflight correctly."""

    def test_sync_calls_preflight_before_uv_sync(self, tmp_path, monkeypatch):
        """Verify preflight is called before uv sync with correct parameters."""
        from pm.environment import PythonEnvironment

        preflight_called = []
        sync_called = []

        def fake_preflight(extras, *, build_env=None, python=None):
            preflight_called.append((extras, build_env, python))

        def fake_run(self, args, *, cwd, timeout):
            sync_called.append(args)
            proc = Mock()
            proc.returncode = 0
            proc.stderr = ""
            proc.stdout = ""
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.require_native_cxx_for_sync", fake_preflight)
        monkeypatch.setattr("pm.environment.PythonEnvironment._run", fake_run)

        project = tmp_path / "project"
        project.mkdir()
        (project / "uv.lock").write_text("")

        env = PythonEnvironment(
            uv=Path("/usr/bin/uv"),
            python=Path("/usr/bin/python3"),
            destination=tmp_path / "venv",
            cache=tmp_path / "cache",
            env={"PATH": "/usr/bin"},
        )

        env.sync(project, extras=["matrix", "web"])

        # Verify preflight was called
        assert len(preflight_called) == 1
        extras, build_env, python = preflight_called[0]
        assert list(extras) == ["matrix", "web"]
        assert build_env == {"PATH": "/usr/bin"}
        assert python == Path("/usr/bin/python3")

        # Verify uv sync was called after preflight
        assert len(sync_called) == 1
        assert "sync" in sync_called[0]

    def test_sync_preflight_failure_prevents_uv_sync(self, tmp_path, monkeypatch):
        """Preflight failure should prevent uv sync from running."""
        from pm.environment import PythonEnvironment

        sync_called = []

        def fake_preflight(extras, *, build_env=None, python=None):
            raise RuntimeError("Missing compiler")

        def fake_run(self, args, *, cwd, timeout):
            sync_called.append(args)
            proc = Mock()
            proc.returncode = 0
            proc.stderr = ""
            proc.stdout = ""
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.require_native_cxx_for_sync", fake_preflight)
        monkeypatch.setattr("pm.environment.PythonEnvironment._run", fake_run)

        project = tmp_path / "project"
        project.mkdir()
        (project / "uv.lock").write_text("")

        env = PythonEnvironment(
            uv=Path("/usr/bin/uv"),
            python=Path("/usr/bin/python3"),
            destination=tmp_path / "venv",
            cache=tmp_path / "cache",
            env={"PATH": "/usr/bin"},
        )

        with pytest.raises(RuntimeError, match="Missing compiler"):
            env.sync(project, extras=["matrix"])

        # Verify uv sync was NOT called
        assert len(sync_called) == 0

    def test_sync_without_matrix_extra_passes_preflight(self, tmp_path, monkeypatch):
        """Sync without matrix extra should pass preflight."""
        from pm.environment import PythonEnvironment

        preflight_called = []

        def fake_preflight(extras, *, build_env=None, python=None):
            preflight_called.append(extras)

        def fake_run(self, args, *, cwd, timeout):
            proc = Mock()
            proc.returncode = 0
            proc.stderr = ""
            proc.stdout = ""
            return proc

        monkeypatch.setattr("pm.toolchain_preflight.require_native_cxx_for_sync", fake_preflight)
        monkeypatch.setattr("pm.environment.PythonEnvironment._run", fake_run)

        project = tmp_path / "project"
        project.mkdir()
        (project / "uv.lock").write_text("")

        env = PythonEnvironment(
            uv=Path("/usr/bin/uv"),
            python=Path("/usr/bin/python3"),
            destination=tmp_path / "venv",
            cache=tmp_path / "cache",
            env={"PATH": "/usr/bin"},
        )

        # Should not raise
        env.sync(project, extras=["web", "google"])

        # Verify preflight was called with non-matrix extras
        assert len(preflight_called) == 1
        assert "matrix" not in preflight_called[0]
