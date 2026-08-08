"""Hermetic discovery and wrapper tests for the crabbox skill."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.skills_hub import OptionalSkillSource


REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_DIR = REPO_ROOT / "optional-skills"
SKILL_DIR = OPTIONAL_DIR / "autonomous-ai-agents" / "crabbox"
SKILL_PATH = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "crabbox.sh"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT_PATH), *args],
        cwd=SKILL_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture
def backend_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "backend-args"

    stub = """#!/bin/sh
printf '%s\\n' "$@" > "$CRABBOX_CAPTURE"
"""
    for name in ("islo", "crabbox"):
        executable = bin_dir / name
        executable.write_text(stub, encoding="utf-8")
        executable.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CRABBOX_CAPTURE"] = str(capture)
    return env, capture


def _captured_args(capture: Path) -> list[str]:
    return capture.read_text(encoding="utf-8").splitlines()


def test_official_source_discovers_and_packages_skill() -> None:
    source = OptionalSkillSource()
    source._optional_dir = OPTIONAL_DIR

    matches = source.search("crabbox")
    meta = next(item for item in matches if item.name == "crabbox")

    assert meta.identifier == "official/autonomous-ai-agents/crabbox"
    assert source.inspect(meta.identifier) == meta

    bundle = source.fetch(meta.identifier)
    assert bundle is not None
    assert {
        "SKILL.md",
        "references/orchestration-patterns.md",
        "scripts/crabbox.sh",
    } <= set(bundle.files)


def test_frontmatter_meets_contribution_rules() -> None:
    content = SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = OptionalSkillSource._parse_frontmatter(content)

    description = frontmatter["description"]
    assert len(description) <= 60
    assert description.endswith(".")
    assert frontmatter["author"] == "Yossi Eliaz (@zozo123), Hermes Agent"
    assert frontmatter["metadata"]["hermes"]["entrypoint"] == "./scripts/crabbox.sh"


def test_skill_command_emits_canonical_skill_file(tmp_path: Path) -> None:
    installed = tmp_path / "crabbox"
    shutil.copytree(SKILL_DIR, installed)

    result = subprocess.run(
        [str(installed / "scripts" / "crabbox.sh"), "skill"],
        cwd=installed,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == SKILL_PATH.read_text(encoding="utf-8")
    assert result.stderr == ""


def test_islo_workhorse_verbs_keep_positional_box_name(
    backend_env: tuple[dict[str, str], Path],
) -> None:
    env, capture = backend_env
    env["CRABBOX_BACKEND"] = "islo"

    result = _run("new", "build-42", "--", "npm", "test", env=env)

    assert result.returncode == 0, result.stderr
    assert _captured_args(capture) == ["use", "build-42", "--", "npm", "test"]


def test_crabbox_injects_id_for_single_box_verbs(
    backend_env: tuple[dict[str, str], Path],
) -> None:
    env, capture = backend_env
    env["CRABBOX_BACKEND"] = "crabbox"

    result = _run("status", "ci-box", "--json", env=env)

    assert result.returncode == 0, result.stderr
    assert _captured_args(capture) == ["status", "--id", "ci-box", "--json"]


def test_crabbox_new_preserves_remote_command_boundary(
    backend_env: tuple[dict[str, str], Path],
) -> None:
    env, capture = backend_env
    env["CRABBOX_BACKEND"] = "crabbox"

    result = _run("new", "ci-box", "--", "npm", "test", env=env)

    assert result.returncode == 0, result.stderr
    assert _captured_args(capture) == ["run", "--id", "ci-box", "--", "npm", "test"]


def test_crabbox_does_not_treat_flag_value_as_box_id(
    backend_env: tuple[dict[str, str], Path],
) -> None:
    env, capture = backend_env
    env["CRABBOX_BACKEND"] = "crabbox"

    result = _run("new", "--class", "beast", "--", "npm", "test", env=env)

    assert result.returncode == 0, result.stderr
    assert _captured_args(capture) == ["run", "--class", "beast", "--", "npm", "test"]


def test_crabbox_remove_strips_islo_force_flag(
    backend_env: tuple[dict[str, str], Path],
) -> None:
    env, capture = backend_env
    env["CRABBOX_BACKEND"] = "crabbox"

    result = _run("rm", "ci-box", "-f", env=env)

    assert result.returncode == 0, result.stderr
    assert _captured_args(capture) == ["stop", "--id", "ci-box"]


@pytest.mark.parametrize("name", ["*", "all", "--all", "-a"])
def test_remove_guard_rejects_bulk_targets_before_backend(
    backend_env: tuple[dict[str, str], Path],
    name: str,
) -> None:
    env, capture = backend_env
    env["CRABBOX_BACKEND"] = "islo"

    result = _run("rm", name, env=env)

    assert result.returncode == 2
    assert "refusing" in result.stderr
    assert not capture.exists()


def test_non_agent_backend_rejects_agent_flags_before_backend(
    backend_env: tuple[dict[str, str], Path],
) -> None:
    env, capture = backend_env
    env["CRABBOX_BACKEND"] = "crabbox"

    result = _run("new", "build-42", "--agent", "claude", env=env)

    assert result.returncode == 2
    assert "does not run autonomous agents" in result.stderr
    assert not capture.exists()
