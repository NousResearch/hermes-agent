from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_faster_whisper_is_not_a_base_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert not any(dep.startswith("faster-whisper") for dep in deps)

    voice_extra = data["project"]["optional-dependencies"]["voice"]
    assert any(dep.startswith("faster-whisper") for dep in voice_extra)


def test_manifest_includes_bundled_skills():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft skills" in manifest
    assert "graft optional-skills" in manifest


def test_hermes_cli_subpackages_bundled_in_wheel():
    # Regression for #27664: `hermes proxy` subcommands crashed with
    # `ModuleNotFoundError: No module named 'hermes_cli.proxy'` on Homebrew
    # installs because packages.find listed only `hermes_cli`, not
    # `hermes_cli.*`, so subpackages with on-disk Python code were dropped
    # from the built wheel.
    from setuptools import find_packages

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]

    found = set(find_packages(where=str(REPO_ROOT), include=include))

    assert "hermes_cli" in found
    assert "hermes_cli.proxy" in found
    assert "hermes_cli.proxy.adapters" in found
