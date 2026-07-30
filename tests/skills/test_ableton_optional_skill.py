"""Tests for the optional Ableton skill."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "creative" / "ableton"


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


class TestAbletonSkillShape:
    def test_skill_lives_under_optional_skills(self):
        assert (SKILL_DIR / "SKILL.md").is_file()
        assert not (ROOT / "skills" / "creative" / "ableton").exists()

    def test_description_is_catalog_sized(self):
        desc = _description(SKILL_DIR)
        assert len(desc) <= 60, desc
        assert desc.endswith(".")

    def test_optional_skill_source_fetches_skill(self):
        from tools.skills_hub import OptionalSkillSource

        source = OptionalSkillSource()
        assert source.fetch("official/creative/ableton").name == "ableton"


class TestAbletonDoctor:
    def test_uvx_socket_and_telemetry(self, monkeypatch, tmp_path):
        doctor = _load(SKILL_DIR / "scripts" / "ableton_doctor.py", "ableton_doctor")
        monkeypatch.setenv("ABLETON_MCP_DISABLE_TELEMETRY", "true")
        live = tmp_path / "Library" / "Preferences" / "Ableton" / "Live 12"
        live.mkdir(parents=True)

        status = doctor.check(
            which={"uvx": "/bin/uvx"}.get,
            port_open=lambda host, port: (host, port) == ("127.0.0.1", 9877),
            home=tmp_path,
        )

        assert status["uvx"] is True
        assert status["remote_script_socket"]["reachable"] is True
        assert status["telemetry_disabled_in_env"] is True
        assert status["mac_remote_script_candidates"] == [
            str(live / "User Remote Scripts" / "AbletonMCP")
        ]

    def test_uvx_missing(self):
        doctor = _load(
            SKILL_DIR / "scripts" / "ableton_doctor.py", "ableton_doctor_none"
        )
        status = doctor.check(which=lambda _name: None, port_open=lambda *_a: False)
        assert status["uvx"] is False
        assert status["remote_script_socket"]["reachable"] is False
