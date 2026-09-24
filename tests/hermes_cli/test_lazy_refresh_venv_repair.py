"""Tests for lazy-backend refresh venv repair (#57828 / #58004)."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock

import hermes_cli.main as m
import hermes_cli.main_install_repair as hermes_cli_main_install_repair
from hermes_cli import main_install_repair
import pytest






def test_detect_returns_none_when_probe_subprocess_fails(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        m, "_resolve_install_target_python", lambda *a, **k: python
    )
    monkeypatch.setattr(
        hermes_cli_main_install_repair, "_resolve_install_target_python", lambda *a, **k: python
    )
    monkeypatch.setattr(
        m.subprocess,
        "run",
        MagicMock(side_effect=OSError("exec failed")),
    )
    assert main_install_repair._detect_broken_lazy_refresh_imports(["uv", "pip"]) is None




def test_repair_runs_force_reinstall_with_pyproject_pins(
    tmp_path, monkeypatch
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """\
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = [
              "pyyaml==6.0.3",
              "click==8.2.1",
            ]
        """
        )
    )
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)

    calls: list[list[str]] = []

    def fake_install(cmd, **kwargs):
        calls.append(cmd)

    detect_calls = {"count": 0}

    def fake_detect(prefix, *, env=None):
        detect_calls["count"] += 1
        return []

    monkeypatch.setattr(main_install_repair, "_run_package_only_install", fake_install)
    monkeypatch.setattr(main_install_repair, "_detect_broken_lazy_refresh_imports", fake_detect)

    ok = main_install_repair._repair_broken_lazy_refresh_imports(
        ["uv", "pip"],
        ["PyYAML", "click"],
        env={"VIRTUAL_ENV": str(tmp_path)},
    )
    assert ok is True
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--force-reinstall",
            "pyyaml==6.0.3",
            "click==8.2.1",
        ]
    ]
    assert detect_calls["count"] == 1


def test_refresh_repairs_venv_after_lazy_failure(tmp_path, monkeypatch):
    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(lazy_deps_mod, "active_features", lambda: ["platform.matrix"])
    monkeypatch.setattr(
        lazy_deps_mod,
        "refresh_active_features",
        lambda **kw: {"platform.matrix": "failed: pip install failed"},
    )

    repair_calls: list[list[str]] = []

    def fake_repair(prefix, packages, *, env=None):
        repair_calls.append(packages)
        return True

    monkeypatch.setattr(main_install_repair, "_detect_broken_lazy_refresh_imports", lambda *a, **k: ["PyYAML"])
    monkeypatch.setattr(main_install_repair, "_repair_broken_lazy_refresh_imports", fake_repair)

    ok = m._refresh_active_lazy_features(["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path)})

    assert ok is True
    assert repair_calls == [["PyYAML"]]


def test_refresh_uses_pre_rebuild_snapshot_when_provided(monkeypatch):
    """Replacement runtimes must not re-detect features after packages vanish."""
    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(
        lazy_deps_mod,
        "active_features",
        lambda: pytest.fail("post-rebuild detection must not run"),
    )
    restored = []
    monkeypatch.setattr(
        lazy_deps_mod,
        "restore_features",
        lambda features: restored.append(features) or {"platform.telegram": "restored"},
    )

    assert m._refresh_active_lazy_features(
        ["uv", "pip"], features=["platform.telegram"]
    ) is True
    assert restored == [["platform.telegram"]]




def test_restore_active_tool_dependencies_uses_static_allowlist(monkeypatch):
    calls = []
    monkeypatch.setattr(
        m,
        "_run_package_only_install",
        lambda cmd, *, env=None: calls.append((cmd, env)),
    )
    monkeypatch.setattr(
        hermes_cli_main_install_repair,
        "_run_package_only_install",
        lambda cmd, *, env=None: calls.append((cmd, env)),
    )

    env = {"VIRTUAL_ENV": "/tmp/venv"}
    m._restore_active_tool_dependencies(
        ["langfuse", "not-allowlisted"],
        ["uv", "pip"],
        env=env,
    )

    assert calls == [(["uv", "pip", "install", "langfuse", "--quiet"], env)]


def test_venv_repair_restores_snapshots_with_an_isolated_uv_env(tmp_path, monkeypatch):
    """The unhealthy-venv repair reinstalls the pre-rebuild lazy/tool snapshots through an env
    built by ``managed_python_env`` (#83914): a third-party ``UV_PYTHON_INSTALL_DIR`` /
    ``UV_PYTHON`` / ``VIRTUAL_ENV`` from unrelated software must not steer uv, and
    ``VIRTUAL_ENV`` points at this install's own venv."""
    from hermes_cli import update_cmd
    from hermes_constants import venv_python_path

    project = tmp_path / "hermes"
    venv_python = venv_python_path(project / "venv", windows=m._is_windows())
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    foreign = tmp_path / "workbuddy"
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(foreign / "python"))
    monkeypatch.setenv("UV_PYTHON", str(foreign / "python" / "python.exe"))
    monkeypatch.setenv("VIRTUAL_ENV", str(foreign / "venv"))

    calls = []
    monkeypatch.setattr(m, "PROJECT_ROOT", project)
    monkeypatch.setattr(m, "_abort_dependency_sync_if_self_locked", lambda *_a: None)
    monkeypatch.setattr(
        m, "_install_python_dependencies_with_optional_fallback",
        lambda prefix, *, env, group: calls.append(("core", prefix, env, group)))
    monkeypatch.setattr(
        m, "_refresh_active_lazy_features",
        lambda prefix, *, env, features: calls.append(("lazy", prefix, env, features)) or True)
    monkeypatch.setattr(
        m, "_restore_active_tool_dependencies",
        lambda deps, prefix, *, env: calls.append(("tools", prefix, env, deps)))
    monkeypatch.setattr(m, "_refresh_active_memory_provider_dependencies", lambda: None)
    monkeypatch.setattr(m, "_reapply_plugin_python_dependencies", lambda: None)
    monkeypatch.setattr(m, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(update_cmd, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(update_cmd, "_venv_core_imports_healthy", lambda: (True, "ok"))
    monkeypatch.setattr(
        update_cmd, "_repair_node_deps_on_current_checkout", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda **_k: "uv")

    assert update_cmd._repair_venv_on_current_checkout(
        assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
        had_desktop_app_before_update=False,
        active_lazy_features=["platform.telegram"],
        active_tool_dependencies=["langfuse"],
        _windows_gateway_resume=None,
    ) is True

    assert [(c[0], c[1], c[3]) for c in calls] == [
        ("core", ["uv", "pip"], "all"),
        ("lazy", ["uv", "pip"], ["platform.telegram"]),
        ("tools", ["uv", "pip"], ["langfuse"]),
    ]
    for _phase, _prefix, env, _extra in calls:
        assert env["VIRTUAL_ENV"] == str(project / "venv")
        assert str(foreign) not in env.get("UV_PYTHON_INSTALL_DIR", "")
        assert env.get("UV_PYTHON") is None
        assert env.get("UV_MANAGED_PYTHON") == "1"
