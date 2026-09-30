from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from hermes_constants import get_optional_skills_dir

HEADER = "[/initiate-setup]"

# The skill's inline-shell hook for host facts. The builder fills it in-process on every surface:
# skills.inline_shell is on only in the setup profile, and on Windows it needs Git Bash.
_HOST_FACTS_HOOK = re.compile(r"^!`[^`\n]*scripts/host_facts\.py`$", re.M)


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
    }
    # Same bytes the hook prints when the skill loads through inline shell.
    host = json.dumps(_host_facts(skill_dir), ensure_ascii=False, separators=(",", ":"))
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8-sig").strip()
    skill = _HOST_FACTS_HOOK.sub(lambda _: host, skill)
    facts = json.dumps(block, indent=2, ensure_ascii=False)
    return f"{HEADER}\n\n{skill}\n\n```json\n{facts}\n```"
