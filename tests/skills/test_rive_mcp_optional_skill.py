"""Tests for the optional Rive MCP skill."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "creative" / "rive-mcp"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _description(skill_dir: Path) -> str:
    text = (skill_dir / "SKILL.md").read_text()
    match = re.search(r'^description:\s*"?([^"\n]+)"?\s*$', text, re.MULTILINE)
    assert match
    return match.group(1)


class TestRiveMcpSkillShape:
    def test_skill_lives_under_optional_skills(self):
        assert (SKILL_DIR / "SKILL.md").is_file()
        assert not (ROOT / "skills" / "creative" / "rive-mcp").exists()

    def test_description_is_catalog_sized(self):
        desc = _description(SKILL_DIR)
        assert len(desc) <= 60, desc
        assert desc.endswith(".")

    def test_optional_skill_source_fetches_skill(self):
        from tools.skills_hub import OptionalSkillSource

        source = OptionalSkillSource()
        assert source.fetch("official/creative/rive-mcp").name == "rive-mcp"


class TestRiveDoctor:
    def test_official_port_reachable(self):
        doctor = _load(SKILL_DIR / "scripts" / "rive_doctor.py", "rive_doctor")
        status = doctor.check(
            which={"node": "/bin/node", "npx": "/bin/npx"}.get,
            port_open=lambda host, port: (host, port) == ("127.0.0.1", 9791),
        )
        assert status["official_rive"]["reachable"] is True
        assert status["rivemcp"]["npx"] is True

    def test_no_paths_available(self):
        doctor = _load(SKILL_DIR / "scripts" / "rive_doctor.py", "rive_doctor_none")
        status = doctor.check(which=lambda _name: None, port_open=lambda *_a: False)
        assert status["official_rive"]["reachable"] is False
        assert status["rivemcp"]["npx"] is False
