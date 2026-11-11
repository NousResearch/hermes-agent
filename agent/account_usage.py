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