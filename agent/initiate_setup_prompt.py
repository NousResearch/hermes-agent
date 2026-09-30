from __future__ import annotations

import importlib.util
import json
import re
import threading
from functools import partial
from pathlib import Path
from typing import NamedTuple

from hermes_constants import get_optional_skills_dir, hermes_home_key

HEADER = "[/initiate-setup]"

# The skill's inline-shell hook for host facts. The builder fills it in-process on every surface:
# skills.inline_shell is on only in the setup profile, and on Windows it needs Git Bash.
_HOST_FACTS_HOOK = re.compile(r"^!`[^`\n]*scripts/host_facts\.py`$", re.M)


class _ScanJob(NamedTuple):
    thread: threading.Thread
    box: dict


# One user scan in flight per Hermes home. host_facts.py is loaded fresh on every call, so the jobs live here.
_SCANS: dict[str, _ScanJob] = {}
_SCANS_LOCK = threading.Lock()


def _skill_dir() -> Path:
    return get_optional_skills_dir(Path(__file__).resolve().parent.parent / "optional-skills") / "productivity" / "initiate-setup"


def _host_facts_module(skill_dir: Path):
    spec = importlib.util.spec_from_file_location("initiate_setup_host_facts", skill_dir / "scripts" / "host_facts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_user_scan() -> _ScanJob:
    """Start the user scan for the bound Hermes home unless one is already running there.

    The worker inherits the caller's profile scope, so the scan caches into that home's
    ``insights/profile.json``. A finished job is not reused: the next start reads that cache.
    Another process that scans the same home waits on this one through the scan's lease file.
    """
    from agent.memory_provider import spawn_context_thread

    key = hermes_home_key()
    with _SCANS_LOCK:
        job = _SCANS.get(key)
        if job is None or not job.thread.is_alive():
            box: dict = {}
            scan = _host_facts_module(_skill_dir()).scan_into
            job = _SCANS[key] = _ScanJob(spawn_context_thread(scan, name="initiate-setup-scan", args=(box,)), box)
            job.thread.start()
    return job


def build_initiate_setup_prompt(surface: str, tools, primary_profile: str) -> str:
    from hermes_cli.anon_auth import free_tier_route
    from hermes_cli.setup_profile import read_state

    skill_dir = _skill_dir()
    block = {
        "surface": surface,
        "tools_present": sorted(set(tools)),
        "primary_profile": primary_profile,
        "guest_free_tier": free_tier_route(),
        "setup_completed_at": read_state().get("completed_at"),
    }
    host_facts = _host_facts_module(skill_dir)
    # Waits on the scan the setup profile started at creation instead of scanning a second time.
    scanned = partial(host_facts.scan_outcome, *start_user_scan())
    # Same bytes the hook prints when the skill loads through inline shell.
    host = json.dumps(host_facts.collect(scanned), ensure_ascii=False, separators=(",", ":"))
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8-sig").strip()
    skill = _HOST_FACTS_HOOK.sub(lambda _: host, skill)
    facts = json.dumps(block, indent=2, ensure_ascii=False)
    return f"{HEADER}\n\n{skill}\n\n```json\n{facts}\n```"
