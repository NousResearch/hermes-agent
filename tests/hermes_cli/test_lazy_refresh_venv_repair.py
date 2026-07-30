"""Tests for lazy-backend refresh venv repair (#57828 / #58004)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import hermes_cli.main as m
import pytest






def test_detect_returns_none_when_probe_subprocess_fails(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        m, "_resolve_install_target_python", lambda *a, **k: python
    )
    monkeypatch.setattr(
        m.subprocess,
        "run",
        MagicMock(side_effect=OSError("exec failed")),
    )
    assert m._detect_broken_lazy_refresh_imports(["uv", "pip"]) is None




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

    monkeypatch.setattr(m, "_run_package_only_install", fake_install)
    monkeypatch.setattr(m, "_detect_broken_lazy_refresh_imports", fake_detect)

    ok = m._repair_broken_lazy_refresh_imports(
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


def test_refresh_repairs_venv_after_lazy_failure(tmp_path, monkeypatch, capsys):
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

    monkeypatch.setattr(m, "_detect_broken_lazy_refresh_imports", lambda *a, **k: ["PyYAML"])
    monkeypatch.setattr(m, "_repair_broken_lazy_refresh_imports", fake_repair)

    ok = m._refresh_active_lazy_features(["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path)})
    out = capsys.readouterr().out

    assert ok is True
    assert repair_calls == [["PyYAML"]]
    assert "Venv repair succeeded" in out
    assert "import probes" in out
    assert "Backends keep their previously-installed version" not in out


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










