"""Provider reasoning-content echo-back helpers (run_agent.py shard s5, c7).

Extracted verbatim from run_agent.py (wave 1, shard s5, cluster c7, 24
move-votes).  Method bodies are character-for-character copies; only this
header and the import block are new.  ``logger`` is bound to the same logger
name as run_agent's module logger so log records keep their origin.

All heavy dependencies are imported lazily inside the methods
(``agent.message_sanitization.matches_reasoning_echo_family``,
``agent.agent_runtime_helpers``), so the mixin needs no module-level
third-party imports.  Per-instance state referenced via ``self.``
(``_thinking_pad_cache``) is created on first use; class attributes stay on
``AIAgent`` and resolve through the MRO.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("run_agent")


class ReasoningEchoMixin:
    def _needs_thinking_reasoning_pad(self) -> bool:
        """Return True when the active provider enforces reasoning_content echo-back.

        DeepSeek v4 thinking and Kimi / Moonshot thinking both reject replays
        of assistant tool-call messages that omit ``reasoning_content`` (refs
        #15250, #17400). Xiaomi MiMo thinking mode has the same requirement.

        Result cached on the AIAgent instance keyed by (provider, model,
        base_url); invalidated whenever ``switch_model()`` /
        ``_try_activate_fallback()`` mutate any of those. This is hot — the
        agent loop hits ~16 invocations per turn, each of which would
        otherwise re-run ~5 ``base_url_host_matches`` (and therefore
        ``urlparse``) calls under it. Caching drops the per-turn cost from
        ~5us × 16 = ~80us to <1us.
        """
        key = (self.provider, self.model, getattr(self, "_base_url_lower", self.base_url))
        cached = getattr(self, "_thinking_pad_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = (
            self._needs_deepseek_tool_reasoning()
            or self._needs_kimi_tool_reasoning()
            or self._needs_mimo_tool_reasoning()
        )
        self._thinking_pad_cache = (key, result)
        return result

    def _needs_kimi_tool_reasoning(self) -> bool:
        """Return True when the current provider is Kimi / Moonshot thinking mode.

        Kimi ``/coding`` and Moonshot thinking mode both require
        ``reasoning_content`` on every assistant tool-call message; omitting
        it causes the next replay to fail with HTTP 400.

        Detection is host-driven, not model-name-driven: aggregators like
        OpenRouter that re-export Kimi/Moonshot models speak their own
        protocol and reject ``reasoning_content`` echoes. We only enable the
        kimi-reasoning replay when the request actually targets a
        kimi/moonshot endpoint or the dedicated kimi-coding provider.

        Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "kimi", self.provider, None, self.base_url
        )

    def _needs_deepseek_tool_reasoning(self) -> bool:
        """Return True when the current provider is DeepSeek thinking mode.

        DeepSeek V4 thinking mode requires ``reasoning_content`` on every
        assistant tool-call turn; omitting it causes HTTP 400 when the
        message is replayed in a subsequent API request (#15250).

        Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "deepseek", (self.provider or "").lower(), self.model, self.base_url
        )

    def _needs_mimo_tool_reasoning(self) -> bool:
        """Return True when the current provider is Xiaomi MiMo thinking mode.

        MiMo thinking mode requires ``reasoning_content`` on every assistant
        tool-call message when replaying history; omitting it causes HTTP 400.
        Refs: https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/passing-back-reasoning_content

        Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
        """
        from agent.message_sanitization import matches_reasoning_echo_family
        return matches_reasoning_echo_family(
            "mimo", (self.provider or "").lower(), self.model, self.base_url
        )

    def _copy_reasoning_content_for_api(self, source_msg: dict, api_msg: dict) -> None:
        """Forwarder — see ``agent.agent_runtime_helpers.copy_reasoning_content_for_api``."""
        from agent.agent_runtime_helpers import copy_reasoning_content_for_api
        return copy_reasoning_content_for_api(self, source_msg, api_msg)

    def _reapply_reasoning_echo_for_provider(self, api_messages: list) -> int:
        """Forwarder — see ``agent.agent_runtime_helpers.reapply_reasoning_echo_for_provider``."""
        from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider
        return reapply_reasoning_echo_for_provider(self, api_messages)
