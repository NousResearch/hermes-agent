"""Behavioral regression coverage for the wheel/sdist distribution guard."""

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUZZ_DASHBOARD_FILES = {
    "plugins/platforms/buzz/dashboard/assets/BuzzLogo24px.svg",
    "plugins/platforms/buzz/dashboard/build.py",
    "plugins/platforms/buzz/dashboard/dist/index.js",
    "plugins/platforms/buzz/dashboard/dist/style.css",
    "plugins/platforms/buzz/dashboard/manifest.json",
    "plugins/platforms/buzz/dashboard/plugin_api.py",
    "plugins/platforms/buzz/dashboard/src/index.js",
    "plugins/platforms/buzz/dashboard/src/style.css",
}


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
            shipped = {name for name in wheel.namelist() if not name.endswith("/")}
            artifact_bytes = {name: wheel.read(name) for name in BUZZ_DASHBOARD_FILES}
    else:
        with tarfile.open(artifacts[0]) as sdist:
            members = {
                member.name.split("/", 1)[1]: member
                for member in sdist.getmembers()
                if "/" in member.name and member.isfile()
            }
            shipped = set(members)
            artifact_bytes = {}
            for name in BUZZ_DASHBOARD_FILES:
                extracted = sdist.extractfile(members[name])
                assert extracted is not None
                artifact_bytes[name] = extracted.read()

    missing = sorted(expected - shipped)
    assert not missing, f"{kind} omits bundled plugin manifests: {missing}"

    shipped_buzz_dashboard = {
        name for name in shipped if name.startswith("plugins/platforms/buzz/dashboard/")
    }
    assert shipped_buzz_dashboard == BUZZ_DASHBOARD_FILES
    assert artifact_bytes["plugins/platforms/buzz/dashboard/src/index.js"] == artifact_bytes[
        "plugins/platforms/buzz/dashboard/dist/index.js"
    ]
    assert artifact_bytes["plugins/platforms/buzz/dashboard/src/style.css"] == artifact_bytes[
        "plugins/platforms/buzz/dashboard/dist/style.css"
    ]
