"""Guards for the multi-container Hermes WebUI install surface."""

from __future__ import annotations

from pathlib import Path
import runpy

from setuptools import Distribution
import setuptools


REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_under(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def test_docker_context_includes_license_file() -> None:
    """PEP 639 license-files metadata must resolve inside the Docker image."""
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "LICENSE" not in active_lines


def test_setup_uses_temporary_outputs_when_source_tree_is_read_only(
    monkeypatch,
) -> None:
    """WebUI installs from read-only /opt/hermes must not write build metadata."""
    captured: dict[str, object] = {}

    def capture_setup(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(setuptools, "setup", capture_setup)
    namespace = runpy.run_path(str(REPO_ROOT / "setup.py"))

    cmdclass = captured["cmdclass"]
    monkeypatch.setitem(
        cmdclass["build"].finalize_options.__globals__,
        "_source_tree_is_writable",
        lambda: False,
    )
    monkeypatch.setitem(
        cmdclass["egg_info"].finalize_options.__globals__,
        "_source_tree_is_writable",
        lambda: False,
    )

    build_cmd = cmdclass["build"](Distribution())
    build_cmd.initialize_options()
    build_cmd.finalize_options()
    assert not _is_under(build_cmd.build_base, REPO_ROOT)
    assert Path(build_cmd.build_base).name.startswith("hermes-agent-build")

    source_relative_build = cmdclass["build"](Distribution())
    source_relative_build.initialize_options()
    source_relative_build.build_base = "nested/build"
    source_relative_build.finalize_options()
    assert not _is_under(source_relative_build.build_base, REPO_ROOT)
    assert Path(source_relative_build.build_base).name.startswith("hermes-agent-build")

    egg_info_cmd = cmdclass["egg_info"](Distribution())
    egg_info_cmd.initialize_options()
    egg_info_cmd.finalize_options()
    assert egg_info_cmd.egg_base is not None
    assert not _is_under(egg_info_cmd.egg_base, REPO_ROOT)
    assert Path(egg_info_cmd.egg_base).name.startswith("hermes-agent-egg-info")

    source_relative_egg_info = cmdclass["egg_info"](Distribution())
    source_relative_egg_info.initialize_options()
    source_relative_egg_info.egg_base = "."
    source_relative_egg_info.finalize_options()
    assert source_relative_egg_info.egg_base is not None
    assert not _is_under(source_relative_egg_info.egg_base, REPO_ROOT)
    assert Path(source_relative_egg_info.egg_base).name.startswith(
        "hermes-agent-egg-info"
    )
