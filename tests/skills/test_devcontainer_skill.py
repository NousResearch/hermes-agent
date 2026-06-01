"""Smoke tests for the devcontainer optional skill.

The devcontainer skill writes config files (devcontainer.json, Dockerfile)
and validates them against common foot-guns. We do not invoke Docker or
the devcontainer CLI in these tests — the goal is to confirm the skill
conforms to the hardline format and the scaffolder produces correct output.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "devops" / "devcontainer"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "devcontainer"


def test_platforms_covers_all_three_os(frontmatter) -> None:
    # The devcontainer.json spec and Microsoft's official dev-container
    # images support all three OSes. The scaffolder itself is pure Python
    # with no platform-specific calls.
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}, (
        f"expected all three platforms, got {frontmatter['platforms']}"
    )


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Thomas Bale" in author, f"author should credit the contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_shipped_scripts_parse() -> None:
    src = (SKILL_DIR / "scripts" / "init.py").read_text()
    ast.parse(src)


def test_shipped_templates_exist() -> None:
    templates_dir = SKILL_DIR / "templates"
    assert templates_dir.is_dir(), f"missing templates dir: {templates_dir}"
    assert (templates_dir / "README.md").is_file(), "missing templates/README.md"
    assert (templates_dir / "devcontainer.example.json").is_file(), (
        "missing templates/devcontainer.example.json"
    )


def test_example_template_is_valid_json() -> None:
    example = (SKILL_DIR / "templates" / "devcontainer.example.json").read_text()
    config = json.loads(example)
    assert "image" in config, "example devcontainer.json must declare an image"
    assert isinstance(config.get("features"), dict), "features should be a map"


# ---------------------------------------------------------------------------
# End-to-end scaffolder tests
# ---------------------------------------------------------------------------


def _run_init(*args: str) -> subprocess.CompletedProcess:
    script = SKILL_DIR / "scripts" / "init.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_init_help_exits_zero() -> None:
    result = _run_init("--help")
    assert result.returncode == 0, result.stderr
    assert "scaffold" in result.stdout.lower() or "validate" in result.stdout.lower()


def test_init_dry_run_prints_valid_json(tmp_path) -> None:
    result = _run_init(
        str(tmp_path),
        "--python",
        "3.11",
        "--features",
        "docker-in-docker",
            "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["image"] == "mcr.microsoft.com/devcontainers/python:3.11-bookworm"
    assert any("docker-in-docker" in k for k in config["features"]), (
        f"expected a docker-in-docker feature, got {list(config['features'])}"
    )
    # The dry run must not have written anything.
    assert not (tmp_path / ".devcontainer").exists(), "dry run should not write files"


def test_init_scaffolds_python_config(tmp_path) -> None:
    result = _run_init(
        str(tmp_path),
        "--python",
        "3.12",
        "--features",
        "git",
        )
    assert result.returncode == 0, result.stderr
    config_path = tmp_path / ".devcontainer" / "devcontainer.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text())
    assert config["image"] == "mcr.microsoft.com/devcontainers/python:3.12-bookworm"
    assert any("features/git" in k for k in config["features"]), (
        f"expected a git feature, got {list(config['features'])}"
    )


def test_init_scaffolds_dockerfile_when_requested(tmp_path) -> None:
    result = _run_init(
        str(tmp_path),
        "--python",
        "3.11",
        "--dockerfile",
    )
    assert result.returncode == 0, result.stderr
    dev_dir = tmp_path / ".devcontainer"
    assert (dev_dir / "devcontainer.json").is_file()
    assert (dev_dir / "Dockerfile").is_file()
    config = json.loads((dev_dir / "devcontainer.json").read_text())
    assert "build" in config, "config with --dockerfile must have a 'build' block"
    assert "image" not in config, "config with --dockerfile must NOT have an 'image' field"
    assert config["build"]["dockerfile"] == "Dockerfile"
    dockerfile = (dev_dir / "Dockerfile").read_text()
    assert "FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm" in dockerfile
    assert "USER vscode" in dockerfile, "Dockerfile must drop back to the vscode user"


def test_init_scaffolds_node_config(tmp_path) -> None:
    result = _run_init(str(tmp_path), "--node", "20")
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / ".devcontainer" / "devcontainer.json").read_text())
    assert config["image"] == "mcr.microsoft.com/devcontainers/javascript-node:20-bookworm"


def test_init_includes_vscode_extensions(tmp_path) -> None:
    result = _run_init(
        str(tmp_path),
        "--python",
        "3.11",
        "--vscode-extensions",
        "ms-python.python,ms-python.debugpy",
        )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / ".devcontainer" / "devcontainer.json").read_text())
    extensions = config["customizations"]["vscode"]["extensions"]
    assert extensions == ["ms-python.python", "ms-python.debugpy"]


def test_init_includes_post_create_and_ports(tmp_path) -> None:
    result = _run_init(
        str(tmp_path),
        "--python",
        "3.11",
        "--post-create",
        "pip install -r requirements.txt",
        "--port",
        "3000,8000",
        )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / ".devcontainer" / "devcontainer.json").read_text())
    assert config["postCreateCommand"] == "pip install -r requirements.txt"
    assert config["forwardPorts"] == [3000, 8000]


def test_init_refuses_overwrite_without_force(tmp_path) -> None:
    (tmp_path / ".devcontainer").mkdir()
    (tmp_path / ".devcontainer" / "devcontainer.json").write_text("{}")
    result = _run_init(str(tmp_path), "--python", "3.11")
    assert result.returncode != 0, "expected non-zero exit on existing config"
    assert "refusing to overwrite" in result.stderr


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validate_passes_for_well_formed_config(tmp_path) -> None:
    dev_dir = tmp_path / ".devcontainer"
    dev_dir.mkdir()
    (dev_dir / "devcontainer.json").write_text(
        json.dumps(
            {
                "name": "Test",
                "image": "mcr.microsoft.com/devcontainers/python:3.12-bookworm",
                "features": {"ghcr.io/devcontainers/features/git:1": "latest"},
            }
        )
    )
    result = _run_init(str(tmp_path), "--validate")
    assert result.returncode == 0, result.stderr


def test_validate_fails_when_missing_image_and_build(tmp_path) -> None:
    dev_dir = tmp_path / ".devcontainer"
    dev_dir.mkdir()
    (dev_dir / "devcontainer.json").write_text(json.dumps({"name": "Test"}))
    result = _run_init(str(tmp_path), "--validate")
    assert result.returncode != 0
    assert "image" in result.stderr or "build" in result.stderr


def test_validate_fails_on_unpinned_feature(tmp_path) -> None:
    dev_dir = tmp_path / ".devcontainer"
    dev_dir.mkdir()
    (dev_dir / "devcontainer.json").write_text(
        json.dumps(
            {
                "image": "mcr.microsoft.com/devcontainers/python:3.12-bookworm",
                "features": {"ghcr.io/devcontainers/features/git": "latest"},
            }
        )
    )
    result = _run_init(str(tmp_path), "--validate")
    assert result.returncode != 0
    assert "version pin" in result.stderr or "pin" in result.stderr


def test_validate_fails_on_malformed_json(tmp_path) -> None:
    dev_dir = tmp_path / ".devcontainer"
    dev_dir.mkdir()
    (dev_dir / "devcontainer.json").write_text("{not valid json")
    result = _run_init(str(tmp_path), "--validate")
    assert result.returncode != 0
    assert "invalid JSON" in result.stderr or "JSON" in result.stderr
