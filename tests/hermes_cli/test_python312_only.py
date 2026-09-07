"""Operational Python 3.12-only invariants.

Hermes runs on exactly Python 3.12. These behavior tests pin that contract
through parsed metadata and real resolver behavior, not source text:

- packaging metadata declares exactly >=3.12,<3.13 and the repo pins 3.12
  for the interpreter and typechecker;
- the managed-runtime request always targets 3.12, even from stale
  3.11/3.13/3.14 lives;
- a live venv that probes as stale 3.11/3.13/3.14 (even with safe SQLite)
  is NOT reported safe -- it must repair forward to 3.12;
- a candidate that does not probe as 3.12 is rejected, not promoted
  (so a downgrade guard can never reject every 3.12 candidate when
  migrating from 3.13/3.14).
- the [wake] extra resolves on 3.12: openwakeword is pinned, the
  uninstallable tflite-runtime transitive is dropped via an override, and
  ai-edge-litert (Darwin + Linux, never Windows) provides the TFLite
  Interpreter API through the ensure_tflite_runtime bridge; `uv sync
  --locked --python 3.12 --extra wake --dry-run` succeeds.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_requires_python_is_exactly_312_line() -> None:
    data = _pyproject()
    assert data["project"]["requires-python"] == ">=3.12,<3.13"


def test_python_version_file_pins_312() -> None:
    assert (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_ty_environment_targets_312() -> None:
    data = _pyproject()
    assert data["tool"]["ty"]["environment"]["python-version"] == "3.12"


def test_runtime_request_is_always_312() -> None:
    from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo
    from hermes_cli.managed_uv import _runtime_request

    for ver in [(3, 11, 15), (3, 12, 11), (3, 13, 7), (3, 14, 0)]:
        info = SQLiteRuntimeInfo(
            executable=Path("/venv/bin/python"),
            base_prefix=Path("/venv"),
            python_version=ver,
            sqlite_version=(3, 53, 1),
            sqlite_version_string="3.53.1",
            sqlite_source_id="fixed",
        )
        assert _runtime_request(info) == "3.12"


def test_rebuild_venv_defaults_to_312() -> None:
    import inspect

    import hermes_cli.managed_uv as managed_uv

    assert inspect.signature(managed_uv.rebuild_venv).parameters[
        "python_version"
    ].default == "3.12"


def test_real_312_interpreter_probes_as_312() -> None:
    assert sys.version_info[:2] == (3, 12), (
        f"supported test lane is exactly Python 3.12, got {sys.version!r} "
        f"from {sys.executable!r}"
    )
    result = subprocess.run(
        [sys.executable, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"interpreter probe must exit 0: {sys.executable!r}\n"
        f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
    )
    assert result.stdout.strip() == "3.12", (
        f"interpreter {sys.executable!r} probed as {result.stdout.strip()!r}, "
        f"expected '3.12'\nstderr: {result.stderr[-3000:]}"
    )


def _sqlite_info(python_version, sqlite_version=(3, 53, 1), vulnerable=False):
    from hermes_cli.sqlite_runtime import SQLiteRuntimeInfo

    vs = ".".join(str(p) for p in sqlite_version)
    return SQLiteRuntimeInfo(
        executable=Path("/venv/bin/python"),
        base_prefix=Path("/venv"),
        python_version=python_version,
        sqlite_version=sqlite_version,
        sqlite_version_string=vs,
        sqlite_source_id="vulnerable" if vulnerable else "fixed",
    )


@pytest.mark.parametrize("stale", [(3, 11, 15), (3, 13, 7), (3, 14, 0)])
def test_stale_python_with_safe_sqlite_is_not_safe(tmp_path, stale) -> None:
    """A stale minor with safe SQLite must still repair forward to 3.12."""
    from hermes_cli import managed_uv

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    live = root / "venv"
    bin_dir = live / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("live", encoding="utf-8")

    current = _sqlite_info(stale, sqlite_version=(3, 53, 1), vulnerable=False)
    provisioned_python = tmp_path / "gen" / "bin" / "python"
    provisioned_python.parent.mkdir(parents=True)
    provisioned_python.touch()
    provisioned = (
        tmp_path / "gen",
        provisioned_python,
        _sqlite_info((3, 12, 11), sqlite_version=(3, 53, 1), vulnerable=False),
    )
    with patch.object(managed_uv, "probe_sqlite_runtime", return_value=current), patch.object(
        managed_uv, "_install_safe_python_generation", return_value=provisioned
    ) as mock_install, patch.object(
        managed_uv, "_stage_candidate_venv", return_value=None
    ), patch.object(
        managed_uv, "_acquire_repair_lock",
        return_value=MagicMock(fd=1, path=root),
    ), patch.object(managed_uv, "_release_repair_lock"):
        result = managed_uv.repair_vulnerable_runtime("uv", project_root=root)
    # Staging returned None so the repair fails, but the key invariant is
    # that it ATTEMPTED a repair (did not report safe) for stale Python.
    assert result.status != "safe"
    mock_install.assert_called_once()


def test_candidate_off_312_line_is_rejected(tmp_path) -> None:
    """A probed 3.13 candidate for a 3.12 request is rejected, not promoted."""
    from hermes_cli import managed_uv

    current = _sqlite_info((3, 12, 10))
    off_line = _sqlite_info((3, 13, 7))

    def fake_run(cmd, **kwargs):
        if "install" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        gen = Path(kwargs["env"]["UV_PYTHON_INSTALL_DIR"])
        python = gen / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        return SimpleNamespace(returncode=0, stdout=str(python), stderr="")

    with patch.object(managed_uv.subprocess, "run", side_effect=fake_run), patch.object(
        managed_uv, "probe_sqlite_runtime", return_value=off_line
    ):
        result = managed_uv._attempt_install_generation(
            "uv",
            "3.12",
            project_root=tmp_path,
            python_root=tmp_path,
            current=current,
        )
    assert result is None


# ── [wake] extra: 3.12 resolver contract ──────────────────────────────────
# openwakeword 0.6.0 declares tflite-runtime on Linux, but tflite-runtime
# 2.14.0 ships only cp311 wheels (no cp312, no sdist), so `uv sync --extra
# wake` cannot resolve on exactly-3.12 without an override. The contract:
# drop the transitive on 3.12+ and ship ai-edge-litert (Darwin + Linux,
# never Windows) whose Interpreter API the ensure_tflite_runtime bridge
# aliases to tflite_runtime.interpreter (per the LiteRT migration guide:
# package rename only, no logic change).


def _wake_specs() -> list[str]:
    return _pyproject()["project"]["optional-dependencies"]["wake"]


def _override_specs() -> list[str]:
    return _pyproject()["tool"]["uv"]["override-dependencies"]


def _lock_data() -> dict:
    return tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))


def test_wake_extra_pins_openwakeword_exact() -> None:
    from packaging.requirements import Requirement

    reqs = [Requirement(s) for s in _wake_specs()]
    oww = [r for r in reqs if r.name.lower().replace("_", "-") == "openwakeword"]
    assert len(oww) == 1, f"wake extra must pin exactly one openwakeword: {_wake_specs()}"
    assert oww[0].specifier == "==0.6.0", f"openwakeword pin drifted: {oww[0]}"


def test_wake_extra_litert_covers_darwin_and_linux_not_windows() -> None:
    from packaging.requirements import Requirement

    reqs = [Requirement(s) for s in _wake_specs()]
    litert = [r for r in reqs if r.name.lower().replace("_", "-") == "ai-edge-litert"]
    assert litert, "wake extra must ship ai-edge-litert for the tflite bridge"
    for r in litert:
        assert r.specifier == "==2.1.6", f"ai-edge-litert pin drifted: {r}"
        assert r.marker is not None, f"ai-edge-litert must be platform-gated: {r}"
    combined = " or ".join(f"({r.marker})" for r in litert)
    from packaging.markers import Marker

    marker = Marker(combined)
    darwin = {
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "python_version": "3.12",
        "python_full_version": "3.12.3",
    }
    linux = {
        "sys_platform": "linux",
        "platform_system": "Linux",
        "python_version": "3.12",
        "python_full_version": "3.12.3",
    }
    windows = {
        "sys_platform": "win32",
        "platform_system": "Windows",
        "python_version": "3.12",
        "python_full_version": "3.12.3",
    }
    assert marker.evaluate(darwin), f"litert must install on macOS: {combined}"
    assert marker.evaluate(linux), f"litert must install on Linux: {combined}"
    assert not marker.evaluate(windows), f"litert must NOT install on Windows: {combined}"


def test_override_drops_tflite_runtime_on_312() -> None:
    from packaging.requirements import Requirement

    reqs = [Requirement(s) for s in _override_specs()]
    hits = [r for r in reqs if r.name.lower().replace("_", "-") == "tflite-runtime"]
    assert hits, "override-dependencies must gate tflite-runtime (no cp312 wheels)"
    assert len(hits) == 1, f"exactly one tflite-runtime override: {hits}"
    marker = hits[0].marker
    assert marker is not None, "tflite-runtime override must carry a python_version gate"
    old = {"python_version": "3.11", "python_full_version": "3.11.9"}
    new = {"python_version": "3.12", "python_full_version": "3.12.3"}
    assert marker.evaluate(old), f"override must keep tflite on <3.12: {hits[0]}"
    assert not marker.evaluate(new), f"override must drop tflite on 3.12+: {hits[0]}"


def test_uv_lock_matches_wake_contract() -> None:
    data = _lock_data()
    # Lock is generated for exactly 3.12 (uv normalizes >=3.12,<3.13).
    assert data["requires-python"] in ("==3.12.*", ">=3.12,<3.13"), data["requires-python"]
    overrides = data.get("manifest", {}).get("overrides", [])
    assert any(
        o.get("name") == "tflite-runtime" and "< '3.12'" in o.get("marker", "")
        for o in overrides
    ), f"uv.lock must record the tflite-runtime <3.12 override: {overrides}"
    pkgs = {p["name"]: p for p in data.get("package", [])}
    litert = pkgs.get("ai-edge-litert")
    assert litert is not None and litert.get("version") == "2.1.6"
    wheels = [w["url"] for w in litert.get("wheels", [])]
    assert wheels, "ai-edge-litert must ship wheels in the lock"
    assert all("cp312" in u for u in wheels), f"lock must be 3.12-only wheels: {wheels}"
    assert any("manylinux" in u and ("aarch64" in u or "x86_64" in u) for u in wheels), (
        f"ai-edge-litert must ship cp312 manylinux wheels for Linux: {wheels}"
    )
    assert any("macosx" in u and "arm64" in u for u in wheels), (
        f"ai-edge-litert must keep the macOS ARM64 wheel: {wheels}"
    )
    assert "tflite-runtime" not in pkgs, "tflite-runtime has no cp312 wheels and must be absent"
    oww = pkgs.get("openwakeword")
    assert oww is not None and oww.get("version") == "0.6.0"
    dep_names = [d.get("name") for d in oww.get("dependencies", [])]
    assert "tflite-runtime" not in dep_names, (
        f"openwakeword's tflite dep must be overridden out on 3.12: {dep_names}"
    )


def _resolve_uv_binary() -> str | None:
    for candidate in ("/usr/bin/uv", "/home/ubuntu/.hermes/bin/uv"):
        if Path(candidate).is_file():
            return candidate
    try:
        from hermes_cli.managed_uv import resolve_uv

        found = resolve_uv()
        if found:
            return found
    except Exception:
        pass
    return shutil.which("uv")


def test_wake_extra_syncs_locked_on_312() -> None:
    assert sys.version_info[:2] == (3, 12), (
        f"supported test lane is exactly Python 3.12, got {sys.version!r} "
        f"from {sys.executable!r}"
    )
    uv = _resolve_uv_binary()
    assert uv is not None, "a uv binary is required on PATH for the wake dry-run"
    result = subprocess.run(
        [uv, "sync", "--locked", "--python", "3.12", "--extra", "wake", "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"uv sync --locked --python 3.12 --extra wake --dry-run must succeed.\n"
        f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
    )


def test_ensure_tflite_runtime_bridges_litert_interpreter_api(monkeypatch) -> None:
    """The bridge aliases ai_edge_litert.interpreter to tflite_runtime.interpreter.

    openwakeword hardcodes `import tflite_runtime.interpreter` and uses only
    the Interpreter API (Interpreter/allocate_tensors/get_input_details/
    get_output_details/set_tensor/invoke/get_tensor), which LiteRT keeps
    compatible across the package rename. Assert the alias preserves that
    surface without installing real packages.
    """
    from tools import wake_word as ww

    for mod in ("tflite_runtime", "tflite_runtime.interpreter", "ai_edge_litert"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "ai_edge_litert.interpreter", raising=False)

    class _FakeInterpreter:
        def __init__(self, *a, **k):
            pass

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"shape": [1, 1], "index": 0}]

        def get_output_details(self):
            return [{"shape": [1, 1], "index": 0}]

        def set_tensor(self, *a):
            pass

        def invoke(self):
            pass

        def get_tensor(self, *a):
            return [[0]]

    fake_mod = types.ModuleType("ai_edge_litert.interpreter")
    fake_mod.Interpreter = _FakeInterpreter  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("ai_edge_litert")
    fake_pkg.interpreter = fake_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_edge_litert", fake_pkg)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", fake_mod)

    assert ww.ensure_tflite_runtime() is True
    import tflite_runtime.interpreter as bridged  # noqa: F401

    assert bridged is fake_mod
    for attr in (
        "Interpreter",
        "allocate_tensors",
        "get_input_details",
        "get_output_details",
        "set_tensor",
        "invoke",
        "get_tensor",
    ):
        if attr == "Interpreter":
            assert hasattr(bridged, attr), "bridge must expose Interpreter"
        else:
            assert hasattr(bridged.Interpreter, attr), f"Interpreter must keep {attr}"


def test_ensure_tflite_runtime_fails_closed_without_any_runtime(monkeypatch) -> None:
    from tools import wake_word as ww

    for mod in ("tflite_runtime", "tflite_runtime.interpreter", "ai_edge_litert"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "ai_edge_litert.interpreter", raising=False)
    assert ww.ensure_tflite_runtime() is False
