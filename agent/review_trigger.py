"""Post-turn background-review trigger: the turn-count clock, optionally widened by a plugin judgment.

The clock (``memory.nudge_interval`` user turns, ``skills.creation_nudge_interval`` tool iterations) is
the built-in trigger and stays the floor: with no ``request_background_review`` subscriber the behavior
is exactly the clock. A subscriber sees each completed turn and may ASK for a review the clock would not
fire yet — a correction on turn 3 of a session that ends at turn 4 is otherwise never reviewed
(#18369, #58669, #31597). It can only add a review, never suppress one the clock fired, and each kind
still honors its own kill switch: interval 0 or the tool absent means no review of that kind, whoever
asks (#82708).

The judgment runs on a daemon thread after the reply is final, so a plugin's network call never delays
delivery. A judged review resets that kind's clock, so the next clock review measures from it rather
than re-reviewing the same turns.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

HOOK_NAME = "request_background_review"
REVIEW_KINDS = ("memory", "skills")


def parse_review_request(results: Iterable[Any]) -> Set[str]:
    """Union of the review kinds hook results ask for. A result is ``{"review": "memory"}`` or
    ``{"review": ["memory", "skills"]}``; anything else (None, unknown kinds, other shapes) asks
    for nothing."""
    kinds: Set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        review = result.get("review")
        if isinstance(review, str):
            review = [review]
        if isinstance(review, (list, tuple, set, frozenset)):
            kinds.update(k for k in review if k in REVIEW_KINDS)
    return kinds


def allowed_review_kinds(agent: Any) -> Set[str]:
    """Kinds a review may run for on this agent — the same gates the clock uses."""
    tools = getattr(agent, "valid_tool_names", None) or ()
    allowed: Set[str] = set()
    if getattr(agent, "_memory_nudge_interval", 0) > 0 and "memory" in tools and getattr(agent, "_memory_store", None):
        allowed.add("memory")
    if getattr(agent, "_skill_nudge_interval", 0) > 0 and "skill_manage" in tools:
        allowed.add("skills")
    return allowed


def _has_subscriber() -> bool:
    try:
        from hermes_cli.lifecycle import has_hook
        return has_hook(HOOK_NAME)
    except Exception:
        logger.debug("%s subscriber check failed", HOOK_NAME, exc_info=True)
        return False


def _review_enabled() -> bool:
    try:
        from agent.background_review import load_background_review_settings
        return load_background_review_settings()[0]
    except Exception:
        return True


def _start_thread(target: Callable[[], None]) -> None:
    """Run ``target`` on a daemon thread carrying this thread's context (profile scope)."""
    from tools.thread_context import propagate_context_to_thread
    threading.Thread(target=propagate_context_to_thread(target), daemon=True, name="bg-review-trigger").start()


def _ask_plugins(payload: Dict[str, Any]) -> Set[str]:
    try:
        from hermes_cli.lifecycle import invoke_hook
        return parse_review_request(invoke_hook(HOOK_NAME, **payload))
    except Exception as exc:
        logger.warning("%s hook failed: %s", HOOK_NAME, exc)
        return set()


def trigger_background_review(
    agent: Any, *, messages_snapshot: List[Dict], clock_memory: bool, clock_skills: bool,
    user_message: Any, final_response: str, session_id: Optional[str], turn_id: Optional[str],
    platform: Optional[str],
) -> None:
    """Spawn the post-turn review the clock decided on, widened by any plugin request.

    Callers have already applied the turn-level gates (a final response, not interrupted, not
    ``skip_background_review``)."""
    if not _has_subscriber():
        if clock_memory or clock_skills:
            agent._spawn_background_review(
                messages_snapshot=messages_snapshot, review_memory=clock_memory, review_skills=clock_skills,
            )
        return

    clock = {k for k, fired in (("memory", clock_memory), ("skills", clock_skills)) if fired}
    # Nothing a plugin could add: skip its call (subagents never review; the clock already covers
    # every allowed kind).
    askable = set() if getattr(agent, "_delegate_depth", 0) > 0 else allowed_review_kinds(agent) - clock
    if not askable:
        if clock:
            agent._spawn_background_review(
                messages_snapshot=messages_snapshot, review_memory=clock_memory, review_skills=clock_skills,
            )
        return

    snapshot = list(messages_snapshot)
    payload = dict(
        session_id=session_id, turn_id=turn_id, platform=platform, model=getattr(agent, "model", None),
        user_message=user_message if isinstance(user_message, str) else str(user_message or ""),
        assistant_response=final_response or "", previous_assistant=_previous_assistant_text(snapshot),
        clock_memory=clock_memory, clock_skills=clock_skills,
        turns_since_memory=getattr(agent, "_turns_since_memory", 0),
        iters_since_skill=getattr(agent, "_iters_since_skill", 0),
    )

    def _judge_then_spawn() -> None:
        asked = _ask_plugins(payload) & askable if _review_enabled() else set()
        memory, skills = clock_memory or "memory" in asked, clock_skills or "skills" in asked
        if "memory" in asked:
            agent._turns_since_memory = 0
        if "skills" in asked:
            agent._iters_since_skill = 0
        if memory or skills:
            try:
                agent._spawn_background_review(
                    messages_snapshot=snapshot, review_memory=memory, review_skills=skills,
                )
            except Exception:
                logger.debug("background review spawn after judgment failed", exc_info=True)

    _start_thread(_judge_then_spawn)


def _previous_assistant_text(messages: List[Dict]) -> str:
    """Text of the last assistant message BEFORE the current user turn — what the user was replying to."""
    seen_user = False
    for message in reversed(messages):
        role = message.get("role")
        if role == "user":
            seen_user = True
            continue
        if seen_user and role == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text = "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
                if text.strip():
                    return text
    return ""
