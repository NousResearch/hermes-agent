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
# The desktop opening the backend plays before the first model call (English only, app-owned copy).
INTRO = "Hi, I'm Hermes.\n\nLet's set things up for you. Then we'll get something cool done."

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


def _collect(host_facts) -> dict:
    # Waits on the scan the setup profile started at creation instead of scanning a second time.
    return host_facts.collect(partial(host_facts.scan_outcome, *start_user_scan()))


def collect_setup_cards() -> dict:
    """Card facts for a setup conversation whose ``/initiate-setup`` turn recorded none: its skill loaded
    through the inline-shell hook, which runs outside the bound profile, or compression gave it a new id."""
    host_facts = _host_facts_module(_skill_dir())
    return host_facts.setup_cards(_collect(host_facts))


def fork_card(cards: dict) -> dict:
    """The fork card as it is shown, its first rows built from the picks this setup conversation recorded."""
    return _host_facts_module(_skill_dir()).fork_card(cards)


def build_initiate_setup_prompt(surface: str, tools, primary_profile: str, session_id: str | None = None) -> str:
    """``session_id``: the desktop session the turn runs in; its ``setup_choose`` cards read these same facts."""
    from hermes_cli.anon_auth import free_tier_route
    from hermes_cli.setup_profile import read_state, record_cards

    skill_dir = _skill_dir()
    block = {
        "surface": surface,
        "tools_present": sorted(set(tools)),
        "primary_profile": primary_profile,
        "guest_free_tier": free_tier_route(),
        "setup_completed_at": read_state().get("completed_at"),
    }
    host_facts = _host_facts_module(skill_dir)
    collected = _collect(host_facts)
    if session_id:
        record_cards(session_id, host_facts.setup_cards(collected))
    # Same bytes the hook prints when the skill loads through inline shell.
    host = json.dumps(collected, ensure_ascii=False, separators=(",", ":"))
    skill = (skill_dir / "SKILL.md").read_text(encoding="utf-8-sig").strip()
    skill = _HOST_FACTS_HOOK.sub(lambda _: host, skill)
    facts = json.dumps(block, indent=2, ensure_ascii=False)
    return f"{HEADER}\n\n{skill}\n\n```json\n{facts}\n```"


NAME_QUESTION = "What should I call you?"
# A closed card counts as answered: the skill takes the default and never re-asks it.
_ANSWERED = ("submitted", "cancelled")


def _json_object(text) -> dict:
    try:
        value = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _opening_step(call) -> str | None:
    """``name`` or ``accent`` for the opening's cards, ``""`` for any other setup card, None otherwise."""
    function = (call.get("function") or {}) if isinstance(call, dict) else {}
    if function.get("name") != "setup_choose":
        return None
    args = _json_object(function.get("arguments"))
    if args.get("kind") == "accent":
        return "accent"
    return "name" if args.get("kind") == "question" and args.get("question") == NAME_QUESTION else ""


def _opening_so_far(history):
    """``(lines said, {card: reply})`` for the opening in ``history``; None once any other setup card was asked.

    Lines are matched by text: a resumed history keeps an unanswered card's row but drops its call."""
    said, replies, card_by_call = set(), {}, {}
    for message in history:
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            said.add(message["content"])
        for call in message.get("tool_calls") or ():
            step = _opening_step(call)
            if step == "":
                return None
            if step:
                card_by_call[call.get("id")] = step
        step = card_by_call.get(message.get("tool_call_id")) if message.get("role") == "tool" else None
        reply = _json_object(message.get("content")) if step else {}
        if reply.get("outcome") in _ANSWERED:
            replies[step] = reply
    return said, replies


def initiate_setup_prelude(message, surface: str, tools, history):
    """The desktop opening as a scripted prelude (``agent/turn_scripted_prelude.py``), or None.

    Only a desktop ``/initiate-setup`` turn gets it, and only for the opening cards the history has no
    answer for: the app sends the command again after a relaunch cut the opening off, and a line
    already in the chat is not said twice. Other surfaces keep the model-only opening. The
    skill starts after the accent answer.
    """
    if (surface != "desktop" or "setup_choose" not in tools or not isinstance(message, str)
            or not message.startswith(HEADER)):
        return None
    so_far = _opening_so_far(history)
    if so_far is None or {"name", "accent"} <= so_far[1].keys():
        return None
    return _opening(_host_facts_module(_skill_dir()).suggested_name(), *so_far)


def intro_resends(prompt: str) -> bool:
    """True while the desktop intro owns recovery of this ``/initiate-setup`` turn: after a relaunch the
    app sends the command again (the prelude replays only the unanswered opening cards), so a generic
    auto-continue of the cut-off turn would race it."""
    from hermes_cli.setup_profile import onboarding_eligible, read_state

    return prompt.startswith(HEADER) and onboarding_eligible() and read_state().get("intro") == "unseen"


def _opening(suggested: str | None, said: set, replies: dict):
    reply = replies.get("name")
    if reply is None:
        # ``options`` is required by the tool schema; the history must carry it even when empty.
        card = {"kind": "question", "question": NAME_QUESTION,
                "options": [{"id": "suggested", "label": suggested}] if suggested else [], "multi_select": False}
        reply = _json_object((yield "" if INTRO in said else INTRO, "setup_choose", card))
    if "accent" in replies:
        return
    picked = reply.get("picked")
    name = (suggested or "") if picked == "suggested" else picked.strip() if isinstance(picked, str) else ""
    line = f"Good to meet you, {name}." if name else "Good to meet you."
    accent = {"kind": "accent", "question": "Which colour?", "options": [], "multi_select": False}
    yield "" if line in said else line, "setup_choose", accent
