"""Smoke tests for the bundled atomicmail email skill."""
from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "email" / "atomicmail"
)


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    match = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "atomicmail"


def test_blueprint_present(frontmatter) -> None:
    blueprint = frontmatter["metadata"]["hermes"]["blueprint"]
    assert blueprint["schedule"] == "0 * * * *"
    assert blueprint["no_agent"] is False
    assert "list_inbox.json" in blueprint["prompt"]


def test_launcher_exists_and_is_executable() -> None:
    launcher = SKILL_DIR / "scripts" / "atomicmail"
    assert launcher.is_file()
    assert launcher.stat().st_mode & stat.S_IXUSR


def test_launcher_scopes_credentials_to_hermes_home() -> None:
    # Credentials must resolve under the active HERMES_HOME so named profiles and
    # remote terminals read the same location the framework mounts credential files.
    bash = (SKILL_DIR / "scripts" / "atomicmail").read_text()
    assert "${HERMES_HOME:-$HOME/.hermes}/atomicmail" in bash
    win = (SKILL_DIR / "scripts" / "atomicmail.cmd").read_text()
    assert "HERMES_HOME" in win


def test_env_vars_limited_to_bearer_secret(frontmatter) -> None:
    # Only the bearer secret belongs in required_environment_variables; non-secret
    # settings must live under metadata.hermes.config (not allowlisted into sandboxes).
    env_names = {
        v["name"] for v in frontmatter.get("required_environment_variables", [])
    }
    assert env_names == {"ATOMIC_MAIL_API_KEY"}, env_names
    config_keys = {
        c["key"] for c in frontmatter["metadata"]["hermes"]["config"]
    }
    assert {
        "atomicmail.auth_url",
        "atomicmail.api_url",
        "atomicmail.scrypt_salt",
    } <= config_keys, config_keys


def test_required_bundle_paths_exist() -> None:
    for rel in (
        "lib/esm/skill/cli.js",
        "lib/presets/list_inbox.json",
        "lib/presets/send_mail.json",
    ):
        assert (SKILL_DIR / rel).is_file(), rel


def test_skill_lists_himalaya_as_related(frontmatter) -> None:
    related = frontmatter["metadata"]["hermes"].get("related_skills") or []
    assert "himalaya" in related


def test_himalaya_defers_to_atomicmail() -> None:
    himalaya_md = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "email"
        / "himalaya"
        / "SKILL.md"
    ).read_text()
    assert "atomicmail" in himalaya_md.lower()
    related_match = re.search(
        r"related_skills:\s*\[(.*?)\]",
        himalaya_md,
        re.DOTALL,
    )
    assert related_match
    assert "atomicmail" in related_match.group(1)
