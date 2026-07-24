"""Tests for lazy-backend refresh venv repair (#57828 / #58004)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import hermes_cli.main as m


def test_detect_broken_imports_returns_repair_package_names(
    tmp_path, monkeypatch
):
    venv_bin = tmp_path / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        m,
        "_resolve_install_target_python",
        lambda prefix, env: python,
    )

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = "yaml\nclick\n"
        result.returncode = 0
        return result

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    broken = m._detect_broken_lazy_refresh_imports(
        ["python", "-m", "pip"], env={"VIRTUAL_ENV": str(tmp_path)}
    )
    assert broken == ["PyYAML", "click"]


def test_detect_returns_none_when_venv_python_unresolved(monkeypatch):
    monkeypatch.setattr(m, "_resolve_install_target_python", lambda *a, **k: None)
    assert m._detect_broken_lazy_refresh_imports(["uv", "pip"]) is None


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


def test_detect_returns_none_when_probe_exits_nonzero(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        m, "_resolve_install_target_python", lambda *a, **k: python
    )

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.stdout = ""
        result.stderr = "boom"
        result.returncode = 1
        return result

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    assert m._detect_broken_lazy_refresh_imports(["uv", "pip"]) is None


def test_repair_via_probes_indeterminate_is_not_success(monkeypatch, capsys):
    monkeypatch.setattr(
        m, "_detect_broken_lazy_refresh_imports", lambda *a, **k: None
    )
    status = m._repair_venv_via_import_probes(["uv", "pip"])
    out = capsys.readouterr().out
    assert status == "indeterminate"
    assert "cannot confirm" in out


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


def test_refresh_returns_false_when_repair_fails(tmp_path, monkeypatch, capsys):
    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(lazy_deps_mod, "active_features", lambda: ["platform.matrix"])
    monkeypatch.setattr(
        lazy_deps_mod,
        "refresh_active_features",
        lambda **kw: {"platform.matrix": "failed: pip install failed"},
    )

    monkeypatch.setattr(m, "_detect_broken_lazy_refresh_imports", lambda *a, **k: ["PyYAML"])
    monkeypatch.setattr(
        m, "_repair_broken_lazy_refresh_imports", lambda *a, **k: False
    )

    ok = m._refresh_active_lazy_features(["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path)})
    out = capsys.readouterr().out

    assert ok is False
    assert "Venv repair incomplete" in out


def test_refresh_returns_false_when_probes_indeterminate(
    tmp_path, monkeypatch, capsys
):
    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(lazy_deps_mod, "active_features", lambda: ["platform.matrix"])
    monkeypatch.setattr(
        lazy_deps_mod,
        "refresh_active_features",
        lambda **kw: {"platform.matrix": "failed: pip install failed"},
    )
    monkeypatch.setattr(m, "_detect_broken_lazy_refresh_imports", lambda *a, **k: None)

    ok = m._refresh_active_lazy_features(["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path)})
    out = capsys.readouterr().out

    assert ok is False
    assert "lazy-refresh-incomplete" in out


def test_refresh_repairs_on_unexpected_lazy_exception(tmp_path, monkeypatch, capsys):
    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(lazy_deps_mod, "active_features", lambda: ["platform.matrix"])

    def boom(**kw):
        raise RuntimeError("refresh registry broke")

    monkeypatch.setattr(lazy_deps_mod, "refresh_active_features", boom)
    monkeypatch.setattr(m, "_detect_broken_lazy_refresh_imports", lambda *a, **k: ["click"])
    monkeypatch.setattr(
        m, "_repair_broken_lazy_refresh_imports", lambda *a, **k: True
    )

    ok = m._refresh_active_lazy_features(["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path)})
    out = capsys.readouterr().out

    assert ok is True
    assert "Lazy refresh failed unexpectedly" in out
    assert "Venv repair succeeded" in out


def test_lazy_marker_stays_until_repair_confirmed(tmp_path, monkeypatch):
    """Lazy marker is independent of the generic core ``.update-incomplete``."""
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)
    m._write_lazy_refresh_incomplete_marker()
    m._write_update_incomplete_marker()

    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(lazy_deps_mod, "active_features", lambda: ["platform.matrix"])
    monkeypatch.setattr(
        lazy_deps_mod,
        "refresh_active_features",
        lambda **kw: {"platform.matrix": "failed: pip install failed"},
    )
    monkeypatch.setattr(m, "_detect_broken_lazy_refresh_imports", lambda *a, **k: ["PyYAML"])
    monkeypatch.setattr(
        m, "_repair_broken_lazy_refresh_imports", lambda *a, **k: False
    )

    ok = m._refresh_active_lazy_features(["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path)})
    assert ok is False
    assert m._lazy_refresh_marker_path().exists()
    assert m._update_marker_path().exists(), "core marker must not be touched by lazy refresh"


def test_upgrade_pip_before_lazy_refresh_never_raises(monkeypatch):
    monkeypatch.setattr(
        m,
        "_run_package_only_install",
        MagicMock(side_effect=m.subprocess.CalledProcessError(1, "pip")),
    )
    m._upgrade_pip_before_lazy_refresh(["uv", "pip"])


def test_package_only_repair_does_not_quarantine_shims_on_windows(
    tmp_path, monkeypatch
):
    """Regression: package-only repairs must not rename hermes.exe on Windows."""
    fake_scripts = tmp_path / "venv" / "Scripts"
    fake_scripts.mkdir(parents=True)

    install_calls: list[list[str]] = []

    def fake_install(cmd, **kwargs):
        install_calls.append(cmd)

    monkeypatch.setattr(m, "_is_windows", lambda: True)
    monkeypatch.setattr(m, "_venv_scripts_dir", lambda: fake_scripts)
    monkeypatch.setattr(m, "_run_package_only_install", fake_install)
    monkeypatch.setattr(
        m, "_detect_broken_lazy_refresh_imports", lambda *a, **k: []
    )

    with patch("hermes_cli.main._quarantine_running_hermes_exe") as mock_quar:
        m._repair_broken_lazy_refresh_imports(
            ["uv", "pip"],
            ["PyYAML"],
            env={"VIRTUAL_ENV": str(tmp_path / "venv")},
        )

    mock_quar.assert_not_called()
    assert install_calls


def test_upgrade_pip_does_not_quarantine_shims_on_windows(tmp_path, monkeypatch):
    fake_scripts = tmp_path / "venv" / "Scripts"
    fake_scripts.mkdir(parents=True)

    install_calls: list[list[str]] = []

    def fake_install(cmd, **kwargs):
        install_calls.append(cmd)

    monkeypatch.setattr(m, "_is_windows", lambda: True)
    monkeypatch.setattr(m, "_venv_scripts_dir", lambda: fake_scripts)
    monkeypatch.setattr(m, "_run_package_only_install", fake_install)

    with patch("hermes_cli.main._quarantine_running_hermes_exe") as mock_quar:
        m._upgrade_pip_before_lazy_refresh(["uv", "pip"])

    mock_quar.assert_not_called()
    assert install_calls == [["uv", "pip", "install", "--upgrade", "pip"]]


def test_lazy_refresh_repair_specs_resolves_extras(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """\
            [project]
            name = "fake"
            version = "0.0.0"
            dependencies = [
              "PyJWT[crypto]==2.13.0",
              "cryptography==46.0.7",
            ]
        """
        )
    )
    monkeypatch.setattr(m, "PROJECT_ROOT", tmp_path)

    specs = m._lazy_refresh_repair_specs(["PyJWT", "cryptography"])
    assert specs == ["PyJWT[crypto]==2.13.0", "cryptography==46.0.7"]
