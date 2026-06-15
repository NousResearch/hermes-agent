"""
Hermes pre_llm_call plugin — routes user query through neuro-skill before LLM sees it.

Architecture (user-verified, 2026-06-15):
  Hermes session start → build SkillRouter index (~2.3s)
  Each LLM call → router.query(user_message, top_k=3) → inject context
  LLM sees only top-3 skills, not 332

This solves the "AI doesn't know it needs routing" deadlock:
routing happens on the host side before the LLM prompt is assembled.
The LLM receives the routing result as context, not as a tool option.

Install: neuro-skill hermes install
"""

import os
import json
import threading
from pathlib import Path

_router = None
_router_lock = threading.Lock()
_skill_dirs = None


def _get_skill_dirs():
    home = Path.home()
    local = os.environ.get("LOCALAPPDATA", "") if os.name == "nt" else ""
    hermes_home = os.environ.get("HERMES_HOME", "")

    dirs = [
        # Claude Code paths
        str(home / ".claude" / ".skills-store" / "skills"),
        str(home / ".claude" / ".skills-store" / "agents"),
        str(home / ".claude" / "skills"),
        str(home / ".claude" / "agents"),
        str(home / ".claude" / ".agents" / "skills"),
        # Hermes paths
        str(home / ".hermes" / "skills"),
        str(home / ".hermes" / "agents"),
        str(home / ".hermes" / "skills-store"),
        str(home / ".config" / "hermes" / "skills"),
        str(Path(os.path.join(local, "hermes", "skills")) if local else ""),
        str(Path(os.path.join(local, "hermes-agent", "skills")) if local else ""),
    ]
    if hermes_home:
        dirs.extend([
            str(Path(hermes_home) / "skills"),
            str(Path(hermes_home) / "agents"),
        ])

    return sorted(set(d for d in dirs if Path(d).is_dir()))


def on_session_start(**kwargs):
    """Pre-build the router index. Called once per Hermes session.

    Cost: ~2.3s for 332 skills (one-time, amortized across session).
    All subsequent queries are <10ms (in-memory index).
    """
    global _router, _skill_dirs
    with _router_lock:
        if _router is not None:
            return  # already built
        try:
            from neuro_skill import SkillRouter
            _skill_dirs = _get_skill_dirs()
            r = SkillRouter()
            r.build(_skill_dirs)
            _router = r
        except Exception as e:
            import logging
            logging.getLogger("neuro_skill.hermes").warning(
                "Failed to build skill index: %s", e
            )


def pre_llm_call(**kwargs) -> dict:
    """Route user query and inject top-3 skills as context.

    Called before every LLM invocation. Reads user_message from kwargs,
    runs router.query(), returns {"context": str} which Hermes injects
    into the LLM prompt.

    Args (provided by Hermes):
        user_message: str         — the user's raw query (核心输入)
        conversation_history: list — prior turns
        session_id: str           — session identifier
        task_id: str              — task identifier
        turn_id: str              — dialogue turn
        is_first_turn: bool       — first message in session

    Returns:
        {"context": str} — Hermes injects this into the LLM prompt
    """
    global _router

    user_message = kwargs.get("user_message", "")
    if not user_message or not user_message.strip():
        return {}

    # Ensure router is built (first call after session start)
    if _router is None:
        try:
            from neuro_skill import SkillRouter
            _skill_dirs = _get_skill_dirs()
            r = SkillRouter()
            r.build(_skill_dirs)
            _router = r
        except Exception:
            return {}

    # Route
    try:
        results = _router.query(user_message, top_k=3, method="hybrid")
    except Exception:
        return {}

    if not results:
        return {}

    # Build structured context block
    lines = ["[Top 3 skills for this query]"]
    for i, (name, score) in enumerate(results):
        lines.append(f"  {i+1}. {name} ({score:.3f})")

    # Append rule hint if available
    from neuro_skill.router import _check_rules
    rule_match = _check_rules(user_message, _router._name_idx)
    if rule_match:
        lines.append(f"  Rule-matched: {rule_match} → use this one")

    lines.append("")
    lines.append("If none of these skills match, fall back to built-in tools.")

    return {"context": "\n".join(lines)}
