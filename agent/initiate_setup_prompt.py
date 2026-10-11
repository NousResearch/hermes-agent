from __future__ import annotations

import json

from agent import initiate_setup_facts

HEADER = "[/initiate-setup]"
# The desktop opening the backend plays before the first model call (English only, app-owned copy).
INTRO = "Hi, I'm Hermes.\n\nLet's set things up for you. Then we'll get something cool done."


def build_initiate_setup_prompt(surface: str, tools, primary_profile: str, session_id: str | None = None) -> str:
    """The skill, then one JSON block of facts the model reads as they are. ``session_id``: the desktop session
    the turn runs in; its ``setup_choose`` cards read the same facts."""
    from hermes_cli.anon_auth import free_tier_route
    from hermes_cli.setup_profile import read_cards, read_state, record_cards

    host = initiate_setup_facts.facts()
    if session_id:
        # Kickoff re-sends the command into a stalled chat: keep the picks already recorded there.
        record_cards(session_id, {**read_cards(session_id), **initiate_setup_facts.setup_cards(host)})
    block = {
        "surface": surface,
        "tools_present": sorted(set(tools)),
        "primary_profile": primary_profile,
        "guest_free_tier": free_tier_route(),
        "setup_completed_at": read_state().get("completed_at"),
        **host,
    }
    block = {key: value for key, value in block.items() if value is not None}
    skill = (initiate_setup_facts.skill_dir() / "SKILL.md").read_text(encoding="utf-8-sig").strip()
    return f"{HEADER}\n\n{skill}\n\n```json\n{json.dumps(block, indent=2, ensure_ascii=False)}\n```"


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
    return _opening(*so_far)


def intro_resends(prompt: str, surface: str) -> bool:
    """True while the desktop intro owns recovery of this ``/initiate-setup`` turn: after a relaunch the
    app sends the command again (the prelude replays only the unanswered opening cards), so a generic
    auto-continue of the cut-off turn would race it. A TUI chat has no intro to resend it."""
    from hermes_cli.setup_profile import onboarding_eligible, read_state

    return (surface == "desktop" and prompt.startswith(HEADER) and onboarding_eligible()
            and read_state().get("intro") == "unseen")


def _opening(said: set, replies: dict):
    reply = replies.get("name")
    if reply is None:
        # ``options`` is required by the tool schema. setup_choose adds the account's name as a row itself, so the
        # saved call never carries it.
        card = {"kind": "question", "question": NAME_QUESTION, "options": [], "multi_select": False}
        reply = _json_object((yield "" if INTRO in said else INTRO, "setup_choose", card))
    if "accent" in replies:
        return
    picked = reply.get("picked")
    label = reply.get("label") if isinstance(reply.get("label"), str) else ""
    name = label if picked == "suggested" else picked.strip() if isinstance(picked, str) else ""
    line = f"Good to meet you, {name}." if name else "Good to meet you."
    accent = {"kind": "accent", "question": "Which colour?", "options": [], "multi_select": False}
    yield "" if line in said else line, "setup_choose", accent
