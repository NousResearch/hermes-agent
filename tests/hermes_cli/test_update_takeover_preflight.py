"""Takeover fails early when a carried native extra cannot compile (#122402).

The historical takeover carries installed extras (e.g. ``matrix``) from the
old checkout venv into the new graph. ``matrix`` pulls ``python-olm``, which
compiles vendored libolm with the managed interpreter's sysconfig ``CXX``.
When that executable is absent (Ubuntu hosts with only the GNU toolchain),
``uv sync`` dies late — after the checkout already moved. The preflight must
fail before ``sync_venv`` with an actionable install-or-override hint, and
must stay silent for selections that need no C++ compiler.
"""

from __future__ import annotations

import sys
import sysconfig
import types

import pytest

from hermes_cli import _update_takeover as takeover


def _isolate_toolchain(monkeypatch, *, cxx: str | None = "clang++ -pthread", found=False):
    """Force the current-process sysconfig path with a stubbed PATH lookup."""
    monkeypatch.setattr(takeover, "_build_python", lambda root: None)
    monkeypatch.setattr(
        sysconfig, "get_config_var", lambda name: cxx if name == "CXX" else None
    )
    monkeypatch.setattr(
        "shutil.which", lambda exe: "/usr/bin/" + exe if found else None
    )


def test_missing_cxx_fails_early_with_install_and_override_hint(tmp_path, monkeypatch):
    _isolate_toolchain(monkeypatch, cxx="clang++ -pthread", found=False)
    with pytest.raises(RuntimeError) as excinfo:
        takeover.preflight_native_toolchain(tmp_path, ["all", "matrix"])
    message = str(excinfo.value)
    assert "clang++" in message
    assert "matrix" in message
    assert "CC=gcc CXX=g++" in message


def test_present_cxx_needs_no_preflight(tmp_path, monkeypatch):
    _isolate_toolchain(monkeypatch, cxx="clang++ -pthread", found=True)
    takeover.preflight_native_toolchain(tmp_path, ["all", "matrix"])


def test_pure_python_selection_ignores_missing_compiler(tmp_path, monkeypatch):
    _isolate_toolchain(monkeypatch, cxx="clang++ -pthread", found=False)
    takeover.preflight_native_toolchain(tmp_path, ["all"])
    takeover.preflight_native_toolchain(tmp_path, None)
    takeover.preflight_native_toolchain(tmp_path, [])


def test_explicit_cxx_override_satisfies_preflight(tmp_path, monkeypatch):
    _isolate_toolchain(monkeypatch, cxx="clang++ -pthread", found=False)
    monkeypatch.setenv("CXX", "g++")
    monkeypatch.setattr("shutil.which", lambda exe: "/usr/bin/g++" if exe == "g++" else None)
    takeover.preflight_native_toolchain(tmp_path, ["all", "matrix"])


def test_unconfigured_cxx_defers_to_build(tmp_path, monkeypatch):
    """No CXX (e.g. MSVC targets): nothing to check, let the build report."""
    _isolate_toolchain(monkeypatch, cxx=None, found=False)
    takeover.preflight_native_toolchain(tmp_path, ["all", "matrix"])


def _stub_prepare_graph(monkeypatch, root, *, extras):
    """Replace prepare()'s whole PM/CLI graph with recording fakes."""
    calls = {}

    def _module(**attrs):
        module = types.ModuleType("fake")
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    class _Worker:
        def __init__(self, correlation):
            calls["worker_context"] = correlation

        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    def _sync(extras_arg, **kwargs):
        calls["sync"] = (extras_arg, kwargs)
        return None

    facts_path = root / "facts.json"
    state_dir = root / "state"
    state_dir.mkdir(exist_ok=True)
    monkeypatch.setitem(
        sys.modules, "hermes_cli.update_stage",
        _module(ensure_panel=lambda r: None, publish_stage=lambda s: None),
    )
    pm_stub = _module()
    pm_stub.receipt = _module(  # type: ignore[attr-defined]
        worker_context=_Worker, last_for_update=lambda correlation: {"update_id": correlation}
    )
    monkeypatch.setitem(sys.modules, "pm", pm_stub)
    monkeypatch.setitem(
        sys.modules, "pm.client",
        _module(ensure_tools_for_sync=lambda: calls.setdefault("tools", True),
                sync_venv=_sync, venv_is_current=lambda **kw: False),
    )
    monkeypatch.setitem(
        sys.modules, "pm.environments",
        _module(activation_environment=lambda r: {},
                install_state_dir=lambda r: state_dir,
                runtime_facts_path=lambda r: facts_path),
    )
    monkeypatch.setitem(
        sys.modules, "hermes_cli._launchers",
        _module(resolve_store_python=lambda r: root / "bin" / "python3"),
    )
    monkeypatch.setitem(
        sys.modules, "hermes_cli.venv_sync",
        _module(publish_launchers=lambda r: None),
    )
    monkeypatch.setitem(
        sys.modules, "pm.extras",
        _module(legacy_selection=lambda r: list(extras)),
    )
    return calls


def test_prepare_runs_preflight_before_sync(tmp_path, monkeypatch):
    calls = _stub_prepare_graph(monkeypatch, tmp_path, extras=["all", "matrix"])
    seen = {}

    def _preflight(root, extras):
        seen["args"] = (root, extras)
        raise RuntimeError("no clang++")

    monkeypatch.setattr(takeover, "preflight_native_toolchain", _preflight)
    with pytest.raises(RuntimeError, match="no clang\\+\\+"):
        takeover.prepare({"root": str(tmp_path), "update_id": "corr"})
    assert "sync" not in calls
    assert seen["args"][1] == ["all", "matrix"]


def test_prepare_syncs_when_preflight_passes(tmp_path, monkeypatch):
    calls = _stub_prepare_graph(monkeypatch, tmp_path, extras=["all"])
    seen = {}
    monkeypatch.setattr(
        takeover, "preflight_native_toolchain",
        lambda root, extras: seen.setdefault("args", (root, extras)),
    )
    python, env = takeover.prepare({"root": str(tmp_path), "update_id": "corr"})
    assert calls["sync"][0] == ["all"]
    assert seen["args"][1] == ["all"]
    assert python == tmp_path / "bin" / "python3"
