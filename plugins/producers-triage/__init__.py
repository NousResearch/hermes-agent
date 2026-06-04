from __future__ import annotations
import asyncio
import logging
import sys
from pathlib import Path
from gateway.platforms.base import MessageEvent, _reply_anchor_for_event, _thread_metadata_for_source
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home

# Dynamically import consent_manager from the producers profile scripts
scripts_dir = '/home/ameobius/projects/security-workstation/.hermes/profiles/producers/scripts'
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    import consent_manager
except ImportError:
    consent_manager = None

logger = logging.getLogger(__name__)

def register(ctx) -> None:
    """Register the producers triage hook."""
    ctx.register_hook("pre_gateway_dispatch", pre_dispatch_handler)
    logger.info("[producers-triage] plugin registered successfully")

def pre_dispatch_handler(event: MessageEvent, gateway: GatewayRunner, **kwargs) -> dict | None:
    # Only act if we are running in the 'producers' profile and the platform is Discord
    home_dir = get_hermes_home()
    if home_dir.name != "producers":
        return None
    
    source = event.source
    if not source or source.platform.value != "discord":
        return None

    # We only want to send ack to human users, not other bots
    if getattr(source, "is_bot", False):
        return None

    text = (event.text or "").lower().strip()
    
    # 1. Check for weekly digest consent triggers
    # Must look for: "кработ согласие" / "кработ отказ" etc.
    # Check if text starts with our bot triggers
    bot_triggers = ["кработ ", "крабик ", "краб ", "кработ,", "крабик,", "краб,"]
    matched_trigger = None
    for trigger in bot_triggers:
        if text.startswith(trigger):
            matched_trigger = trigger
            break
            
    if matched_trigger and consent_manager:
        command_text = text[len(matched_trigger):].strip()
        if command_text in ["согласие", "дайджест да", "дайджест ок", "согласен"]:
            consent_manager.grant_consent(source.user_id, source.user_name)
            ack_text = "принял — теперь твои наброски могут попадать в еженедельные дайджесты"
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(send_ack(gateway, source.chat_id, ack_text, event))
            except Exception as e:
                logger.warning(f"[producers-triage] failed to schedule consent ack: {e}")
            return {"action": "skip", "reason": "consent-granted"}
            
        elif command_text in ["отказ", "дайджест нет", "дайджест стоп", "не согласен"]:
            consent_manager.revoke_consent(source.user_id)
            ack_text = "убрал согласие — больше твои наброски в дайджесты не попадут"
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(send_ack(gateway, source.chat_id, ack_text, event))
            except Exception as e:
                logger.warning(f"[producers-triage] failed to schedule consent revoke ack: {e}")
            return {"action": "skip", "reason": "consent-revoked"}

    # 2. Custom ack messages based on channel or keywords (fast-path for other queries)
    ack_text = None
    
    # 2.1 Fast-path execute prompt doctor directly without model calling if structured
    if ("prompt doctor" in text or "разбери prompt" in text or "почини промпт" in text or "разбери промпт" in text) and source.chat_id in ["1509389604279554108", "1509389598923559053"]: # #💻〢воркфлоу or #🆘〢помощь
        # Try to parse fields using the router library directly
        try:
            import discord_interactive_router
            extracted = discord_interactive_router._extract_key_values(event.text or "")
            req_fields = ["prompt", "goal_or_genre", "failure_mode"]
            # Map aliases / alternative keys to extract values
            p_val = extracted.get("prompt")
            g_val = extracted.get("goal_or_genre") or extracted.get("goal")
            f_val = extracted.get("failure_mode") or extracted.get("rough_problem")
            
            if p_val and g_val and f_val:
                import prompt_doctor
                res_output = prompt_doctor.run_doctor(p_val, g_val, f_val)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(send_ack(gateway, source.chat_id, res_output, event))
                except Exception as e:
                    logger.warning(f"[producers-triage] failed to schedule prompt doctor execution: {e}")
                return {"action": "skip", "reason": "prompt-doctor-executed"}
        except Exception as ex:
            logger.warning(f"[producers-triage] offline prompt doctor execution failed: {ex}")

    if "prompt doctor" in text or "разбери prompt" in text or "почини промпт" in text or source.chat_id == "1509389604279554108": # #💻〢воркфлоу
        ack_text = "гляну твой промпт — сверяюсь со спеками суно — через минуту выкачу разбор"
    elif event.message_type.value == "audio" or "audio triage" in text or source.chat_id == "1509389602410365029": # #💡〢аудио-наработки
        ack_text = "принял аудио набросок — сейчас прогоню через анализатор спектра — подожди секунд сорок"
    elif source.chat_id == "1509389598923559053": # #🆘〢помощь
        ack_text = "принял твой кейс — изучаю логи и окружение — сейчас вернусь с ответом"
    elif source.chat_type == "dm" or source.chat_id == "1509389578015080609": # DM or home channel
        if text.startswith("/") or any(w in text for w in ["помоги", "help", "кработ", "крабик", "бот"]):
            ack_text = "принял запрос — сейчас соображу — секунду"

    if ack_text:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(send_ack(gateway, source.chat_id, ack_text, event))
            logger.info(f"[producers-triage] scheduled immediate ack: '{ack_text}' for chat {source.chat_id}")
        except Exception as e:
            logger.warning(f"[producers-triage] failed to schedule ack: {e}")

    # Allow normal execution to proceed to the main agent
    return None

async def send_ack(gateway: GatewayRunner, chat_id: str, text: str, event: MessageEvent):
    try:
        adapter = gateway.adapters.get(event.source.platform)
        if adapter:
            reply_to = _reply_anchor_for_event(event)
            metadata = _thread_metadata_for_source(event.source, reply_to)
            await adapter.send(chat_id, text, reply_to=reply_to, metadata=metadata)
            logger.info(f"[producers-triage] sent immediate ack successfully to {chat_id}")
    except Exception as e:
        logger.error(f"[producers-triage] error sending ack: {e}")
