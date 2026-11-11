from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

import httpx

from agent.anthropic_credentials import _is_oauth_token, resolve_anthropic_token
from hermes_cli.auth import AuthError, _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.auth_codex import _codex_pool_route_base_url
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_time import safe_strftime

from agent.usage_gauges import BP_FULL

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger(__name__)

_DEPLETED_LINE = "Status: access depleted — top up to restore"
# Usage-window plans: allowances are shown as percent used.
_WINDOW_TOP_TIER_USED_UP = "Monthly large-model usage is used up."
_WINDOW_PAUSED_LINE = ("Status: included usage paused. Your subscription payment failed. Update your card to restore it. "
                       "Until then, Unitalk uses your credits.")
_WINDOW_PAUSED_OTHER_LINE = "Status: included usage paused. See Billing in the Portal. Until then, Unitalk uses your credits."