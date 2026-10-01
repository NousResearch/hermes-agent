from __future__ import annotations

import pytest

from pm.toolchain_preflight import require_native_cxx_for_sync


def test_matrix_extra_without_compiler_raises(monkeypatch):
    monkeypatch.setattr(
        "pm.extras.extra_supported",
        lambda extra, **kwargs: extra == "matrix",
    )
    monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)
    monkeypatch.setattr("pm.toolchain_preflight.sys.platform", "linux")

    with pytest.raises(RuntimeError, match="clang\\+\\+ on PATH"):
        require_native_cxx_for_sync(["matrix"], build_env={"PATH": "/usr/bin"})


def test_matrix_extra_with_compiler_ok(monkeypatch):
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


def test_clang_cxx_env_without_matrix(monkeypatch):
    monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="Refusing to sync"):
        require_native_cxx_for_sync([], build_env={"CXX": "clang++ -pthread", "PATH": "/usr/bin"})


def test_unrelated_extra_skips_preflight(monkeypatch):
    monkeypatch.setattr("pm.toolchain_preflight.shutil.which", lambda *_a, **_k: None)

    require_native_cxx_for_sync(["web"], build_env={"PATH": "/usr/bin"})
