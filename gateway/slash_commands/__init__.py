"""Gateway slash-command handlers for GatewayRunner.

The former monolithic module is composed from per-family leaf mixins. ``self``
remains the ``GatewayRunner`` in every moved method, and no leaf imports
``gateway.run`` at module scope.
"""

from __future__ import annotations

from gateway.session import AsyncSessionStore
from gateway.slash_commands._shared import (
    _RESET_CLEANUP_TIMEOUT_S,
    _clean_str,
    _int_value,
    _model_switch_skew_guard,
    logger,
)
from gateway.slash_commands.registry import GATEWAY_SLASH_HANDLERS
from gateway.slash_commands.session_lifecycle import SessionLifecycleCommandsMixin
from gateway.slash_commands.model import ModelCommandsMixin
from gateway.slash_commands.agents_ops import AgentsOpsCommandsMixin
from gateway.slash_commands.compress import CompressCommandsMixin
from gateway.slash_commands.runtime_flags import RuntimeFlagsCommandsMixin
from gateway.slash_commands.info import InfoCommandsMixin
from gateway.slash_commands.skills import SkillsCommandsMixin
from gateway.slash_commands.usage import UsageCommandsMixin
from gateway.slash_commands.goals import GoalsCommandsMixin
from gateway.slash_commands.reasoning import ReasoningCommandsMixin
from gateway.slash_commands.update import UpdateCommandsMixin
from gateway.slash_commands.approvals import ApprovalsCommandsMixin
from gateway.slash_commands.kanban import KanbanCommandsMixin
from gateway.slash_commands.voice import VoiceCommandsMixin
from gateway.slash_commands.home import HomeCommandsMixin
from gateway.slash_commands.memory import MemoryCommandsMixin


class GatewaySlashCommandsMixin(
    SessionLifecycleCommandsMixin,
    ModelCommandsMixin,
    AgentsOpsCommandsMixin,
    CompressCommandsMixin,
    RuntimeFlagsCommandsMixin,
    InfoCommandsMixin,
    SkillsCommandsMixin,
    UsageCommandsMixin,
    GoalsCommandsMixin,
    ReasoningCommandsMixin,
    UpdateCommandsMixin,
    ApprovalsCommandsMixin,
    KanbanCommandsMixin,
    VoiceCommandsMixin,
    HomeCommandsMixin,
    MemoryCommandsMixin,
):
    """In-session slash-command handlers for GatewayRunner."""

    async_session_store: AsyncSessionStore
