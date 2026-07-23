"""Entry-point lifecycle: recovery runs before ``hermes_cli.main`` imports.

Proves the #58004 / #57828 bootstrap contract — a wiped probed package
(``dotenv``) must not prevent marker recovery from starting, because recovery
lives in a dependency-light entry point that does not import
``hermes_cli.env_loader``.
"""

from __future__ import annotations

import builtins
import sys
import textwrap
import types
from pathlib import Path

import hermes_entry
import hermes_update_recovery as recovery


def test_pyproject_hermes_script_uses_bootstrap_entry():
    import tomllib

    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["hermes"] == "hermes_entry:main"
    py_modules = data["tool"]["setuptools"]["py-modules"]
    assert "hermes_entry" in py_modules
    assert "hermes_update_recovery" in py_modules


def test_entry_recovers_before_importing_main(monkeypatch):
    """``hermes_cli.main`` (dotenv via env_loader) must load only after recover."""
    order: list[str] = []

    def fake_recover(*, root=None, argv=None):
        order.append("recover")
        return True

    monkeypatch.setattr(recovery, "maybe_recover", fake_recover)

    stub = types.ModuleType("hermes_cli.main")

    def stub_main():
        order.append("main")
        return 0

    stub.main = stub_main  # type: ignore[attr-defined]

    # Drop any cached real module so the entry re-imports under our gate.
    monkeypatch.delitem(sys.modules, "hermes_cli.main", raising=False)

    real_import = builtins.__import__

    def gated_import(name, globals=None, locals=None, fromlist=(), level=0):
        loading_main = name == "hermes_cli.main" or (
            name == "hermes_cli" and fromlist and "main" in fromlist
        )
        if loading_main:
            assert "recover" in order, (
                "hermes_cli.main imported before bootstrap recovery — "
                f"order={order}"
            )
            order.append("import_main")
            sys.modules["hermes_cli.main"] = stub
            if name == "hermes_cli.main":
                return stub
            # ``from hermes_cli.main import main`` may import package first.
            pkg = real_import(name, globals, locals, fromlist, level)
            return pkg
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", gated_import)

    rc = hermes_entry.main([])
    assert rc == 0
    assert order[0] == "recover"
    assert "import_main" in order
    assert order.index("recover") < order.index("import_main")
    assert "main" in order


def test_bootstrap_repair_specs_preserve_pyproject_pins(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = [
              "python-dotenv>=1.2.1,<2",
              "PyYAML>=6.0.3,<7",
            ]
            """
        ),
        encoding="utf-8",
    )
    specs = recovery.repair_specs(["python-dotenv", "PyYAML"], tmp_path)
    assert specs == ["python-dotenv>=1.2.1,<2", "PyYAML>=6.0.3,<7"]


def test_bootstrap_force_reinstall_uses_pinned_specs(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = ["python-dotenv>=1.2.1,<2"]
            """
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    monkeypatch.setattr(recovery, "detect_broken_imports", lambda *a, **k: [])
    monkeypatch.setattr(
        recovery, "_install_target", lambda root: (["uv", "pip"], {"VIRTUAL_ENV": "v"})
    )

    assert recovery.force_reinstall_packages(["python-dotenv"], tmp_path) is True
    assert calls
    assert "python-dotenv>=1.2.1,<2" in calls[0]


def test_bootstrap_lazy_recovery_clears_marker_when_probes_healthy(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    recovery.lazy_refresh_marker_path(tmp_path).write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "detect_broken_imports", lambda *a, **k: [])
    status = recovery.recover_lazy_marker(tmp_path)
    assert status == "healthy"
    assert not recovery.lazy_refresh_marker_path(tmp_path).exists()


def test_bootstrap_skips_during_update_argv(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    recovery.update_marker_path(tmp_path).write_text("x\n", encoding="utf-8")
    called = {"n": 0}
    monkeypatch.setattr(
        recovery, "recover_core_marker", lambda *a, **k: called.__setitem__("n", 1)
    )
    assert recovery.maybe_recover(tmp_path, argv=["update"]) is False
    assert called["n"] == 0
    assert recovery.update_marker_path(tmp_path).exists()


def test_recovery_module_import_does_not_load_dotenv(monkeypatch):
    """Contract: loading recovery must work even when dotenv is absent."""
    import importlib

    monkeypatch.delitem(sys.modules, "dotenv", raising=False)
    monkeypatch.delitem(sys.modules, "hermes_cli.env_loader", raising=False)

    # Pretend dotenv is missing for any accidental import during reload.
    real_import = builtins.__import__

    def no_dotenv(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dotenv" or name.startswith("dotenv."):
            raise ModuleNotFoundError("No module named 'dotenv'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_dotenv)
    mod = importlib.reload(recovery)
    assert mod.markers_present is not None
    assert "dotenv" not in sys.modules
    assert "hermes_cli.env_loader" not in sys.modules
