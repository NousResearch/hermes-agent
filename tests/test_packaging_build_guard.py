"""Behavioral regression coverage for the wheel/sdist distribution guard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    assert list(tmp_path.glob(artifact_glob))


def test_built_wheel_ships_importable_secret_source_conformance_kit(tmp_path):
    """The conformance kit has to import from the artifact, not just the checkout.

    website/docs/developer-guide/secret-source-plugin.md sends external
    secret-source authors to `from agent.secret_sources.testing import
    SecretSourceConformance` against an installed hermes-agent, and calls green
    conformance the review bar. A checkout-only import test cannot see that
    contract break: it stays green even if [tool.setuptools.packages.find] later
    stops shipping the package, because tests/ runs with the source tree on
    sys.path. So exercise the import against the wheel this module already
    builds, with the checkout kept off sys.path.
    """
    result = _build_artifact("wheel", tmp_path, nix_build=True)
    assert result.returncode == 0, result.stderr

    wheels = list(tmp_path.glob("hermes_agent-*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]

    env = os.environ.copy()
    # A wheel is a zip, so zipimport can import straight out of it: the assertion
    # then covers the artifact itself rather than an unpacked copy of it.
    env["PYTHONPATH"] = str(wheel)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agent.secret_sources.testing import SecretSourceConformance\n"
            "import agent.secret_sources.testing as kit\n"
            "print(SecretSourceConformance.__name__)\n"
            "print(kit.__file__)\n",
        ],
        # Not PROJECT_ROOT: the checkout must not be able to satisfy the import,
        # otherwise this asserts nothing about what setuptools shipped.
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    name, module_path = probe.stdout.splitlines()
    assert name == "SecretSourceConformance"
    assert Path(module_path).is_relative_to(wheel), (
        f"kit imported from {module_path}, not from the built wheel at {wheel}"
    )
