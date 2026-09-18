"""Behavioral regression coverage for the wheel/sdist distribution guard."""

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERMES_STATE_SUPPORT_MODULES = (
    "hermes_state_common",
    "hermes_state_portability",
    "hermes_state_schema",
    "hermes_state_search",
)


def _build_artifact(kind: str, tmp_path, *, nix_build: bool) -> subprocess.CompletedProcess[str]:
    """Invoke the real PEP 517 hook (build_sdist / build_wheel) as a subprocess.

    The wheel and sdist guards live in SEPARATE cmdclass entries in setup.py
    (the bdist_wheel one behind a try/except ImportError), so each hook needs
    its own regression coverage — a passing sdist test proves nothing about
    the wheel path.
    """
    env = os.environ.copy()
    # nix develop exports this too, so it must not grant permission to build
    # a distributable artifact.
    env["NIX_BUILD_TOP"] = "/build/devshell"
    if nix_build:
        env["HERMES_NIX_BUILD"] = "1"
    else:
        env.pop("HERMES_NIX_BUILD", None)
    # Redirect setuptools' scratch dirs (build/, *.egg-info) into tmp_path so
    # the allowed-marker build doesn't litter the real worktree.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    extra_cfg = tmp_path / "dist-extra.cfg"
    extra_cfg.write_text(
        f"[build]\nbuild_base = {scratch / 'build'}\n\n[egg_info]\negg_base = {scratch}\n",
        encoding="utf-8",
    )
    env["DIST_EXTRA_CONFIG"] = str(extra_cfg)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_{kind}; build_{kind}(r'{out}')".format(
                kind=kind, out=tmp_path
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("kind", ["sdist", "wheel"])
def test_artifact_build_rejects_nix_development_shell_environment(kind, tmp_path):
    result = _build_artifact(kind, tmp_path, nix_build=False)

    assert result.returncode != 0
    assert "Building wheels or sdists for hermes-agent is not supported" in result.stderr


@pytest.mark.parametrize(
    ("kind", "artifact_glob"),
    [("sdist", "hermes_agent-*.tar.gz"), ("wheel", "hermes_agent-*.whl")],
)
def test_artifact_build_allows_explicit_nix_package_build_marker(kind, artifact_glob, tmp_path):
    result = _build_artifact(kind, tmp_path, nix_build=True)

    assert result.returncode == 0, result.stderr
    artifacts = list(tmp_path.glob(artifact_glob))
    assert artifacts

    expected = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for pattern in ("plugin.yaml", "plugin.yml")
        for path in (PROJECT_ROOT / "plugins").rglob(pattern)
    }
    assert expected, "expected bundled plugin manifests under plugins/"

    if kind == "wheel":
        with zipfile.ZipFile(artifacts[0]) as wheel:
            shipped = set(wheel.namelist())
    else:
        with tarfile.open(artifacts[0]) as sdist:
            shipped = {
                name.split("/", 1)[1]
                for name in sdist.getnames()
                if "/" in name
            }

    missing = sorted(expected - shipped)
    assert not missing, f"{kind} omits bundled plugin manifests: {missing}"


def test_wheel_ships_importable_hermes_state_support_modules(tmp_path):
    """The built wheel, not the source tree, must provide SessionDB support modules."""
    result = _build_artifact("wheel", tmp_path, nix_build=True)
    assert result.returncode == 0, result.stderr
    wheel_path = next(tmp_path.glob("hermes_agent-*.whl"))

    with zipfile.ZipFile(wheel_path) as wheel:
        shipped = set(wheel.namelist())
    missing = sorted(
        f"{module}.py"
        for module in HERMES_STATE_SUPPORT_MODULES
        if f"{module}.py" not in shipped
    )
    assert not missing, f"wheel omits hermes_state support modules: {missing}"

    import_script = "\n".join(
        [
            "import importlib",
            "from pathlib import Path",
            "import sys",
            f"wheel_path = {str(wheel_path)!r}",
            "sys.path.insert(0, wheel_path)",
            f"module_names = {HERMES_STATE_SUPPORT_MODULES!r}",
            "for name in module_names:",
            "    module = importlib.import_module(name)",
            "    archive = getattr(module.__loader__, 'archive', None)",
            "    assert archive is not None, module.__loader__",
            "    assert Path(archive).resolve() == Path(wheel_path).resolve(), archive",
        ]
    )
    imported = subprocess.run(
        [sys.executable, "-I", "-c", import_script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
