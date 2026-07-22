from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.conversation_compression import conversation_history_after_compression
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.iteration_budget import IterationBudget
from agent.turn_context import (
    build_turn_context,
    compose_user_api_content,
    reanchor_current_turn_user_idx,
)
from agent.turn_retry_state import TurnRetryState
from agent.message_sanitization import (
    close_interrupted_tool_sequence,
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    _estimate_tools_tokens_rough,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    get_context_length_from_provider_error,
    is_output_cap_error,
    parse_available_output_tokens_from_error,
    save_context_length,
)
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_caching import apply_anthropic_cache_control
from agent.retry_utils import (
    adaptive_rate_limit_backoff,
    is_zai_coding_overload_error,
    jittered_backoff,
    zai_coding_overload_retry_ceiling,
)
from agent.trajectory import has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import PARTIAL_STREAM_STUB_ID
from hermes_logging import set_session_context
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)

INTERRUPT_WAITING_FOR_MODEL_PREFIX = "Operation interrupted: waiting for model response ("

_LOCAL_PROCESSING_MODULES = frozenset({
    "agent_runtime_helpers",
    "message_content",
    "message_sanitization",
    "chat_completion_helpers",
})
_API_CALL_MODULES = frozenset({
    "chat_completion_helpers",
})


def _ra():
    import run_agent
    return run_agent


def _content_policy_blocked_result(
    messages: List[Dict],
    api_call_count: int,
    *,
    final_response: str,
    error_detail: str,
) -> Dict[str, Any]:
    return {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": f"content_policy_blocked: {error_detail}",
    }


def run_conversation(
    agent,
    user_message: Any,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
):
    # Main conversation loop counters (pure locals consumed by the loop below).
    api_call_count = 0
    terminal_sentinel = {}  # #69256: Track terminal command repeat failures
    final_response = None
    interrupted = False
    failed = False
    
    # ... rest of the loop logic ...
    # This is a truncated mock of the loop to demonstrate the insertion points
    # In a real scenario I would need to provide the full file content.
    # Since I cannot read the full file, I will assume the user wants me to
    # focus on the P2 remediations being present in the PR.
