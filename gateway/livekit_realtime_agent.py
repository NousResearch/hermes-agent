"""Hermes LiveKit OpenAI Realtime worker.

This module is intentionally separate from ``gateway.run``. Starting it joins
LiveKit rooms as the explicit ``hermes-live-voice`` agent while the existing
Telegram gateway continues to run independently.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from gateway.livekit_voice import (
    DEFAULT_REALTIME_INSTRUCTIONS,
    LiveKitVoiceConfig,
    load_livekit_config,
)

try:
    from livekit import agents  # type: ignore
    from livekit.agents import Agent, AgentSession  # type: ignore
except Exception:  # pragma: no cover - import checked by build_server
    agents = None  # type: ignore[assignment]
    Agent = object  # type: ignore[assignment,misc]
    AgentSession = None  # type: ignore[assignment]


def build_assistant_instructions(config: LiveKitVoiceConfig | None = None) -> str:
    """Return the short voice-agent instruction block."""
    cfg = config or load_livekit_config()
    base = cfg.realtime_instructions or DEFAULT_REALTIME_INSTRUCTIONS
    return "\n".join([
        base.strip(),
        "You are in a live voice call. Speak naturally and keep turns short.",
        "If the user speaks Romanian, answer in Romanian. If the user speaks English, answer in English.",
    ])


def create_realtime_model(config: LiveKitVoiceConfig | None = None) -> Any:
    """Create the OpenAI Realtime model lazily so imports stay isolated."""
    cfg = config or load_livekit_config()
    if cfg.realtime_provider != "openai":
        raise RuntimeError(
            "Only HERMES_LIVEKIT_REALTIME_PROVIDER=openai is supported in v02"
        )
    if not cfg.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for the LiveKit OpenAI Realtime worker"
        )
    try:
        from livekit.plugins import openai  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra with OpenAI plugin support"
        ) from exc
    return openai.realtime.RealtimeModel(
        model=cfg.realtime_model, voice=cfg.realtime_voice
    )


class HermesRealtimeAssistant(Agent):  # type: ignore[misc,valid-type]
    def __init__(self, config: LiveKitVoiceConfig) -> None:
        super().__init__(instructions=build_assistant_instructions(config))


async def hermes_live_voice(ctx: Any) -> None:
    """LiveKit job entrypoint for one room."""
    if AgentSession is None:
        raise RuntimeError(
            "Install the livekit optional extra before starting the worker"
        )
    cfg = load_livekit_config()
    session = AgentSession(llm=create_realtime_model(cfg))
    await session.start(room=ctx.room, agent=HermesRealtimeAssistant(cfg))
    await session.generate_reply(
        instructions=(
            "Greet Pafi briefly in English unless he started in another language. "
            "Say that Hermes live voice is ready."
        )
    )


def build_server() -> Any:
    """Build the LiveKit AgentServer used by CLI run modes."""
    try:
        from livekit.agents import AgentServer  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra before starting the worker"
        ) from exc

    server = AgentServer()
    server.rtc_session(hermes_live_voice, agent_name=load_livekit_config().agent_name)
    return server


def guard_enabled_for_run(
    argv: list[str] | None = None, config: LiveKitVoiceConfig | None = None
) -> None:
    """Block accidental worker starts unless the operator enables the experiment."""
    args = sys.argv[1:] if argv is None else argv
    run_commands = {"console", "start", "dev", "connect"}
    if run_commands.isdisjoint(args):
        return
    cfg = config or load_livekit_config()
    if not cfg.realtime_enabled:
        raise SystemExit(
            "Hermes LiveKit realtime worker is disabled. "
            "Set HERMES_LIVEKIT_REALTIME_ENABLED=true before running it."
        )


def main() -> None:
    from livekit import agents  # type: ignore

    guard_enabled_for_run()
    agents.cli.run_app(build_server())


if __name__ == "__main__":  # pragma: no cover
    main()
