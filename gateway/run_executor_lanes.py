"""Reserved executor lane for interactive (human-chat) agent turns, bound onto ``GatewayRunner``.

The gateway's agent-turn pool is FIFO and shared by every platform. On installs with heavy
automated traffic (webhook/API-triggered turns) batch work occupies every worker and a human
chat message can wait hours before its turn even starts (observed: a telegram turn finishing
with ``time=8335s api_calls=0`` — pure queue wait). ``gateway.interactive_executor_workers`` > 0
gives human-driven platforms their own pool so batch backlog cannot delay a conversation.
0 (default) keeps the single shared pool, byte-identical to prior behavior.
"""

from __future__ import annotations

import concurrent.futures
import logging

logger = logging.getLogger("gateway.run")

# Human-driven chat surfaces that may use reserved interactive capacity. An explicit
# allowlist, not a denylist: API/webhook routes, relays, plugin adapters, and unknown or
# malformed values stay on the shared pool, so nothing new can starve a waiting person.
INTERACTIVE_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal", "matrix", "mattermost", "dingtalk",
    "feishu", "wecom", "bluebubbles", "weixin", "sms", "email", "line", "teams",
})


class GatewayExecutorLanesMixin:
    """Interactive-lane executor selection for GatewayRunner."""

    def _get_interactive_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """The reserved interactive pool, or the shared pool when no lane is configured.

        Reads ``self.config`` defensively: bare/legacy runners built without a config
        (tests, ``object.__new__``) behave as lane-off."""
        workers = getattr(getattr(self, "config", None), "interactive_executor_workers", None) or 0
        if workers <= 0:
            return self._get_executor()

        # Shares GatewayRunner._get_or_create_pool's lock/closing fence with the other pools.
        if getattr(self, "_interactive_executor", None) is None:
            logger.info("Gateway interactive executor: max_workers=%d", workers)
        return self._get_or_create_pool("_interactive_executor", workers, "hermes-gw-interactive")

    @staticmethod
    def _is_batch_platform(source) -> bool:
        """Whether a turn's source shares the main pool (everything not in INTERACTIVE_PLATFORMS).

        Fails safe: a missing/None/non-string platform routes to the SHARED pool — the pre-lane
        behavior — so it can never consume reserved interactive capacity."""
        raw = getattr(source, "platform", None)
        platform = getattr(raw, "value", raw)
        return not isinstance(platform, str) or platform not in INTERACTIVE_PLATFORMS
