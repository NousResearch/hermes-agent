"""teams_voice plugin — Microsoft Teams real-time voice/video (CVI) bridge driver.

Hosts an HMAC-authenticated WebSocket the companion Windows .NET media worker
(AzureBot / OpenClawBridge) dials into, and drives the call: dialogue (realtime
or streaming), perception (camera/screen vision), and the avatar rendering cues
(expression / visemes / show-to-caller). The worker renders the NV12 avatar tile;
this plugin sends the drivers.

Chat-plane integration (Teams messages, message actions, meeting-recap posting)
is handled by the existing ``plugins/platforms/teams`` adapter — this plugin is
the *media/voice* half and deliberately does not duplicate it.

Status: SCAFFOLD. The wire layer (bridge server, protocol, HMAC) and the
pure-logic ports (expression, visemes, group-call gate) are complete and the C#
worker can connect; the realtime speech-to-speech brain (``realtime/``) is a stub.
"""

from __future__ import annotations

import logging

from .cli import register_cli as _register_cli
from .cli import teams_voice_command as _teams_voice_command
from .tools import (
    TEAMS_VOICE_STATUS_SCHEMA,
    check_requirements,
    handle_teams_voice_status,
)

logger = logging.getLogger(__name__)


def _on_session_end(**_kwargs) -> None:
    """Best-effort hook placeholder.

    The bridge runs as its own server process, so there is nothing call-scoped to
    tear down on agent-session end today. Kept registered so the lifecycle wiring
    is stable as the realtime brain lands.
    """
    return None


def register(ctx) -> None:
    """Plugin entry point — register the status tool, CLI, and lifecycle hook.

    Called once by the plugin loader when ``teams_voice`` is enabled via
    ``plugins.enabled`` in config.yaml.
    """
    ctx.register_tool(
        name="teams_voice_status",
        toolset="teams_voice",
        schema=TEAMS_VOICE_STATUS_SCHEMA,
        handler=handle_teams_voice_status,
        check_fn=check_requirements,
        emoji="📞",
    )

    ctx.register_cli_command(
        name="teams-voice",
        help="Microsoft Teams voice/video (CVI) bridge (serve, status)",
        setup_fn=_register_cli,
        handler_fn=_teams_voice_command,
        description=(
            "Run the HMAC-authenticated bridge the Teams .NET media worker "
            "connects to. See: hermes teams-voice status"
        ),
    )

    ctx.register_hook("on_session_end", _on_session_end)
