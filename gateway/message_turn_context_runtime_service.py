"""Shared context/history preparation for gateway foreground message turns."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable

from gateway.auto_background_runtime_service import (
    format_auto_background_ack,
    resolve_auto_background_dispatch,
)
from gateway.background_job_start_service import start_background_job
from gateway.config import Platform
from gateway.employee_routes import get_employee_routes
from gateway.message_preprocessing_runtime_service import (
    is_shared_thread_session,
    prepend_reply_context_if_missing,
    prepend_shared_thread_sender,
)
from gateway.onboarding_runtime_service import (
    FIRST_MESSAGE_ONBOARDING_NOTE,
    build_home_channel_prompt,
    home_channel_env_var_name,
    should_prompt_for_home_channel,
)
from gateway.session import build_session_context_prompt
from gateway.shared_group_history_runtime_service import prepare_history_for_agent
from gateway.session_hygiene_runtime_service import maybe_auto_compress_session_history


@dataclass(slots=True)
class GatewayPreparedMessageTurnContext:
    """Prepared context/history state before message-text enrichment.

    ``context_prompt`` is the **stable** session context only.
    Volatile must-deliver notes ride ``turn_sidecar_notes`` (api_content
    sidecar contract) — never append them into ``context_prompt``.
    """

    context_prompt: str
    history: list[dict[str, Any]]
    history_for_agent: list[dict[str, Any]]
    auto_background_response: str | None = None
    turn_sidecar_notes: list[str] | None = None


def load_gateway_privacy_redact_pii(config_path: Path) -> bool:
    """Read the per-message privacy.redact_pii switch from config.yaml."""

    try:
        import yaml

        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return bool((config.get("privacy") or {}).get("redact_pii", False))
    except Exception:
        return False


def build_gateway_auto_reset_context_note(*, session_entry: Any) -> str | None:
    """Return the auto-reset system note for sidecar delivery, or None."""

    if not getattr(session_entry, "was_auto_reset", False):
        return None

    reset_reason = getattr(session_entry, "auto_reset_reason", None) or "idle"
    if reset_reason == "daily":
        return (
            "[System note: The user's session was automatically reset by the daily schedule. "
            "This is a fresh conversation with no prior context.]"
        )
    return (
        "[System note: The user's previous session expired due to inactivity. "
        "This is a fresh conversation with no prior context.]"
    )


def prepend_gateway_auto_reset_context_note(
    context_prompt: str,
    *,
    session_entry: Any,
) -> str:
    """Deprecated prompt-prepend helper — prefer sidecar via ``build_gateway_auto_reset_context_note``.

    Kept for older unit tests; production and ``prepare_gateway_message_turn_context``
    use the sidecar list instead of mutating ``context_prompt``.
    """

    note = build_gateway_auto_reset_context_note(session_entry=session_entry)
    if not note:
        return context_prompt
    return f"{note}\n\n{context_prompt}"


async def maybe_send_gateway_auto_reset_notice(
    *,
    runner: Any,
    source: Any,
    event: Any,
    session_entry: Any,
    logger: Any,
) -> None:
    """Send the user-facing auto-reset notice when policy requires it."""

    if not getattr(session_entry, "was_auto_reset", False):
        return

    try:
        policy = runner.session_store.config.get_reset_policy(
            platform=source.platform,
            session_type=getattr(source, "chat_type", "dm"),
        )
        platform_name = source.platform.value if source.platform else ""
        had_activity = getattr(session_entry, "reset_had_activity", False)
        should_notify = (
            policy.notify
            and had_activity
            and platform_name not in policy.notify_exclude_platforms
        )
        if not should_notify:
            return

        adapter = runner.adapters.get(source.platform)
        if not adapter:
            return

        reset_reason = getattr(session_entry, "auto_reset_reason", None) or "idle"
        if reset_reason == "daily":
            reason_text = f"daily schedule at {policy.at_hour}:00"
        else:
            hours = policy.idle_minutes // 60
            mins = policy.idle_minutes % 60
            duration = f"{hours}h" if not mins else f"{hours}h {mins}m" if hours else f"{mins}m"
            reason_text = f"inactive for {duration}"
        notice = (
            f"◐ Session automatically reset ({reason_text}). "
            f"Conversation history cleared.\n"
            f"Use /resume to browse and restore a previous session.\n"
            f"Adjust reset timing in config.yaml under session_reset."
        )
        try:
            session_info = runner._format_session_info()
            if session_info:
                notice = f"{notice}\n\n{session_info}"
        except Exception:
            pass
        await adapter.send(
            source.chat_id,
            notice,
            metadata=getattr(event, "metadata", None),
        )
    except Exception as exc:
        logger.debug("Auto-reset notification failed (non-fatal): %s", exc)


def maybe_auto_load_gateway_topic_skill(
    *,
    event: Any,
    session_key: str,
    is_new_session: bool,
    logger: Any,
) -> None:
    """Auto-load the topic-bound skill for new DM-topic sessions."""

    if not is_new_session or not getattr(event, "auto_skill", None):
        return

    try:
        from agent.skill_commands import _load_skill_payload, _build_skill_message

        skill_name = event.auto_skill
        loaded = _load_skill_payload(skill_name, task_id=session_key)
        if not loaded:
            logger.warning(
                "[Gateway] DM topic skill '%s' not found in available skills",
                skill_name,
            )
            return

        loaded_skill, skill_dir, display_name = loaded
        activation_note = (
            f'[SYSTEM: This conversation is in a topic with the "{display_name}" skill '
            f"auto-loaded. Follow its instructions for the duration of this session.]"
        )
        skill_message = _build_skill_message(
            loaded_skill,
            skill_dir,
            activation_note,
            user_instruction=event.text,
        )
        if skill_message:
            event.text = skill_message
            logger.info(
                "[Gateway] Auto-loaded skill '%s' for DM topic session %s",
                skill_name,
                session_key,
            )
    except Exception as exc:
        logger.warning("[Gateway] Failed to auto-load topic skill '%s': %s", event.auto_skill, exc)


async def prepare_gateway_message_turn_context(
    *,
    runner: Any,
    event: Any,
    source: Any,
    context: Any,
    session_entry: Any,
    session_key: str,
    history: list[dict[str, Any]],
    is_new_session: bool,
    config_path: Path,
    hermes_home: Path,
    logger: Any,
    explicit_group_reply_note: str = "",
    visible_limit: int = 20,
    runtime_agent_kwargs_loader: Callable[[], dict[str, Any]] | None = None,
    build_session_context_prompt_fn: Callable[..., str] = build_session_context_prompt,
) -> GatewayPreparedMessageTurnContext:
    """Prepare context prompt, history, and auto-background routing for one turn.

    Volatile must-deliver notes are collected into ``turn_sidecar_notes`` and
    must not be folded into the stable ``context_prompt`` (sidecar-only contract).
    """

    redact_pii = load_gateway_privacy_redact_pii(config_path)
    context_prompt = build_session_context_prompt_fn(context, redact_pii=redact_pii)
    turn_sidecar_notes: list[str] = []
    if explicit_group_reply_note:
        turn_sidecar_notes.append(explicit_group_reply_note)

    auto_reset_note = build_gateway_auto_reset_context_note(session_entry=session_entry)
    if auto_reset_note:
        turn_sidecar_notes.append(auto_reset_note)
    await maybe_send_gateway_auto_reset_notice(
        runner=runner,
        source=source,
        event=event,
        session_entry=session_entry,
        logger=logger,
    )
    session_entry.was_auto_reset = False
    session_entry.auto_reset_reason = None

    maybe_auto_load_gateway_topic_skill(
        event=event,
        session_key=session_key,
        is_new_session=is_new_session,
        logger=logger,
    )

    history_for_agent = prepare_history_for_agent(
        history,
        shared_session_kind=getattr(context, "shared_session_kind", None),
        session_id=session_entry.session_id,
        logger=logger,
        visible_limit=visible_limit,
    )

    background_message_text = str(event.text or "")
    background_message_text = prepend_shared_thread_sender(
        message_text=background_message_text,
        user_name=source.user_name,
        shared_thread=is_shared_thread_session(
            source=source,
            thread_sessions_per_user=bool(
                getattr(runner.config, "thread_sessions_per_user", False)
            ),
        ),
    )
    background_message_text = prepend_reply_context_if_missing(
        message_text=background_message_text,
        reply_to_text=getattr(event, "reply_to_text", None),
        reply_to_message_id=getattr(event, "reply_to_message_id", None),
        history=history,
    )
    background_dispatch = resolve_auto_background_dispatch(
        event,
        background_message_text,
        auto_background_work_enabled=runner._get_auto_background_work(
            getattr(event.source, "platform", None)
        ),
        employee_routes=get_employee_routes(
            runner.config,
            platform=getattr(event.source, "platform", Platform.QQ_NAPCAT),
        ),
        conversation_history=list(history_for_agent or []),
    )
    if background_dispatch:
        task_id = start_background_job(
            store=runner._get_background_job_store(),
            launch_worker=runner._launch_background_worker,
            prompt=background_message_text,
            source=source,
            conversation_history=list(history_for_agent or []),
            context_prompt=context_prompt,
            session_key=session_key,
            job_kind="auto",
            worker_name=str(background_dispatch.get("worker_name") or ""),
            preloaded_skills=list(background_dispatch.get("preloaded_skills") or []),
            admin_user_ids=context.admin_user_ids,
            is_admin_user=context.is_admin_user,
            logger=logger,
        )
        return GatewayPreparedMessageTurnContext(
            context_prompt=context_prompt,
            history=history,
            history_for_agent=history_for_agent,
            auto_background_response=format_auto_background_ack(
                background_message_text,
                task_id,
                worker_name=str(background_dispatch.get("worker_name") or ""),
            ),
            turn_sidecar_notes=list(turn_sidecar_notes),
        )

    history = await maybe_auto_compress_session_history(
        history=history,
        session_entry=session_entry,
        session_store=runner.session_store,
        hermes_home=hermes_home,
        runtime_agent_kwargs_loader=runtime_agent_kwargs_loader or (lambda: {}),
        logger=logger,
    )

    # First-message onboarding: sidecar only (do not mutate context_prompt).
    if not history and not runner.session_store.has_any_sessions():
        note = (FIRST_MESSAGE_ONBOARDING_NOTE or "").strip()
        if note:
            turn_sidecar_notes.append(note)

    env_key = home_channel_env_var_name(source.platform)
    if should_prompt_for_home_channel(
        history=history,
        platform=source.platform,
        home_channel_configured=bool(env_key and os.getenv(env_key)),
    ):
        adapter = runner.adapters.get(source.platform)
        if adapter and source.platform is not None:
            await adapter.send(
                source.chat_id,
                build_home_channel_prompt(source.platform),
            )

    if source.platform == Platform.DISCORD:
        adapter = runner.adapters.get(Platform.DISCORD)
        guild_id = runner._get_guild_id(event)
        if guild_id and adapter and hasattr(adapter, "get_voice_channel_context"):
            voice_context = adapter.get_voice_channel_context(guild_id)
            if voice_context:
                turn_sidecar_notes.append(voice_context)

    return GatewayPreparedMessageTurnContext(
        context_prompt=context_prompt,
        history=history,
        history_for_agent=history_for_agent,
        turn_sidecar_notes=list(turn_sidecar_notes),
    )
