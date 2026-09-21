"""Antigravity client runtime integrated with the normal AIAgent turn lifecycle."""
from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Dict, List

from agent.transports.antigravity_session import AntigravitySession

logger = logging.getLogger(__name__)


def _close_antigravity_session(agent) -> None:
    """Close and discard the external session; idempotent ownership boundary."""
    session = getattr(agent, "_antigravity_session", None)
    if session is not None:
        agent._antigravity_session = None
        agent._antigravity_usage_baseline = {}
        with suppress(Exception):
            session.close()


def _event_bridge(agent):
    from agent.transports.antigravity_event_projector import make_antigravity_event_bridge

    def protocol_error(message: str) -> None:
        logger.error("Antigravity protocol rejected event: %s", message)
        session = getattr(agent, "_antigravity_session", None)
        if session is not None:
            session.request_interrupt()

    return make_antigravity_event_bridge(agent, on_protocol_error=protocol_error)


def _ensure_antigravity_session(agent) -> None:
    if getattr(agent, "_antigravity_session", None) is not None:
        return
    from agent.runtime_cwd import resolve_agent_cwd
    session_factory = getattr(agent, "_antigravity_session_factory", None) or AntigravitySession
    configured_client_factory = getattr(agent, "_antigravity_client_factory", None)
    if configured_client_factory is None:
        from agent.transports.antigravity_cli import AntigravityClient
        from hermes_cli.runtime_provider import get_antigravity_runtime_config
        from hermes_constants import get_hermes_home
        config = get_antigravity_runtime_config()

        def default_client_factory(**kwargs):
            binary = config.get("binary")
            debug_log = get_hermes_home() / "logs" / "antigravity-protocol.log" if config["debug_protocol"] else None
            return AntigravityClient(
                None if binary in {None, "", "auto"} else binary,
                cwd=kwargs.get("cwd"),
                startup_timeout=float(config["startup_timeout_seconds"]),
                request_timeout=float(config["request_timeout_seconds"]),
                shutdown_timeout=float(config["shutdown_timeout_seconds"]),
                sandbox=bool(config["sandbox"]),
                debug_log=debug_log,
            )
        configured_client_factory = default_client_factory
    agent._antigravity_session = session_factory(
        cwd=getattr(agent, "session_cwd", None) or str(resolve_agent_cwd()),
        model=getattr(agent, "model", None),
        client_factory=configured_client_factory,
        projector_factory=getattr(agent, "_antigravity_projector_factory", None),
        event_callback=_event_bridge(agent),
    )


def _persist_projected_messages(agent, turn, messages: List[Dict[str, Any]]) -> bool:
    """Append projected rows once and report whether the DB flush succeeded."""
    from agent.message_metadata import append_message
    for message in turn.projected_messages:
        append_message(messages, message)
    if turn.projected_messages and getattr(agent, "_session_db", None) is not None:
        if agent._flush_messages_to_session_db(messages) is False:
            logger.warning("Antigravity projected-message flush failed (session=%s)", getattr(agent, "session_id", None))
            return False
    return True


def _record_usage(agent, usage: Any) -> dict[str, Any]:
    """Convert agy's conversation-cumulative usage into per-turn deltas."""
    agent.session_api_calls += 1
    if not isinstance(usage, dict):
        return {}
    baseline = getattr(agent, "_antigravity_usage_baseline", {})
    if not isinstance(baseline, dict):
        baseline = {}
    current: dict[str, int] = {}
    out: dict[str, int] = {}
    for source in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"):
        value = usage.get(source)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        value = max(int(value), 0)
        previous = baseline.get(source, 0)
        delta = value - previous if value >= previous else value
        current[source] = value
        out[source] = delta
    if "total_tokens" in out:
        agent.session_total_tokens += out["total_tokens"]
    # Hermes reporting uses prompt/completion names; agy exposes input/output names.
    prompt_delta = out.get("prompt_tokens", out.get("input_tokens"))
    if prompt_delta is not None:
        out.setdefault("prompt_tokens", prompt_delta)
        agent.session_prompt_tokens += prompt_delta
    completion_delta = out.get("completion_tokens", out.get("output_tokens"))
    if completion_delta is not None:
        out.setdefault("completion_tokens", completion_delta)
        agent.session_completion_tokens += completion_delta
    baseline.update(current)
    agent._antigravity_usage_baseline = baseline
    return out


def _restore_conversation_id(session: AntigravitySession, messages: List[Dict[str, Any]]) -> None:
    if session.conversation_id:
        return
    for message in reversed(messages):
        sidecar = message.get("_antigravity") if isinstance(message, dict) else None
        conversation_id = sidecar.get("conversation_id") if isinstance(sidecar, dict) else None
        if isinstance(conversation_id, str) and conversation_id:
            session.conversation_id = conversation_id
            return


def run_antigravity_turn(agent, *, user_message: Any, original_user_message: Any, messages: List[Dict[str, Any]],
                          effective_task_id: str, should_review_memory: bool = False) -> Dict[str, Any]:
    """Run one external turn and return the standard conversation-loop result shape."""
    _ensure_antigravity_session(agent)
    _restore_conversation_id(agent._antigravity_session, messages)
    turn = agent._antigravity_session.run_turn(user_message)
    if turn.should_retire:
        _close_antigravity_session(agent)
    persisted = _persist_projected_messages(agent, turn, messages)
    interrupted = bool(turn.interrupted or getattr(agent, "_interrupt_requested", False))
    interrupt_message = getattr(agent, "_interrupt_message", None) if interrupted else None
    if interrupted:
        agent.clear_interrupt()
    usage = _record_usage(agent, turn.usage)
    completed = not interrupted and turn.error is None
    result = {
        "final_response": turn.final_text, "messages": messages, "api_calls": 1,
        "completed": completed, "partial": not completed, "interrupted": interrupted,
        "error": turn.error, "agent_persisted": persisted,
        "antigravity_conversation_id": turn.conversation_id, **usage,
    }
    if interrupt_message:
        result["interrupt_message"] = interrupt_message
    return result


__all__ = ["run_antigravity_turn", "_ensure_antigravity_session", "_close_antigravity_session"]
