from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hermes_constants import get_optional_skills_dir

HEADER = "[/initiate-setup]"


def _skill_dir() -> Path:
    return get_optional_skills_dir(Path(__file__).resolve().parent.parent / "optional-skills") / "productivity" / "initiate-setup"


def _host_facts(skill_dir: Path) -> dict:
    spec = importlib.util.spec_from_file_location("initiate_setup_host_facts", skill_dir / "scripts" / "host_facts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect()


def build_initiate_setup_prompt(surface: str, tools, primary_profile: str) -> str:
    from hermes_cli.anon_auth import free_tier_route

    skill_dir = _skill_dir()
    block = {
        "surface": surface,
        "tools_present": sorted(set(tools)),
        "primary_profile": primary_profile,
        "guest_free_tier": free_tier_route(),
        **_host_facts(skill_dir),
    }
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8-sig").strip()
    facts = json.dumps(block, indent=2, ensure_ascii=False)
    return f"{HEADER}\n\n{skill}\n\n```json\n{facts}\n```"
