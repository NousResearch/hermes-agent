"""Agent delegation — run the full Hermes agent for a one-shot voice consult.

The realtime model handles small talk itself and delegates real work (lookups,
files, web, actions) to the Hermes agent via the ``hermes_agent_consult`` tool.
This wraps ``run_agent.AIAgent`` (synchronous, tool-capable) and runs it off the
event loop with ``asyncio.to_thread`` so the call's audio keeps flowing.

The agent is built lazily and reused across consults within a call. A consult is
time-boxed; on timeout/error a short speakable message is returned so the model
can tell the caller gracefully rather than hanging.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentConsult:
    """Lazily-built, reusable one-shot agent runner for a call."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._agent = None  # run_agent.AIAgent, built on first use

    def _agent_kwargs(self) -> dict:
        """Build AIAgent kwargs from the configured ``model`` block.

        A bare ``AIAgent()`` leaves the model empty → 'Missed model deployment',
        so we pass the same provider/model/base_url/api_mode the CLI resolves from
        config.yaml ``model:`` (e.g. provider=azure-foundry, default=gpt-5.5)."""
        import os

        kwargs: dict = {"quiet_mode": True}
        try:
            from hermes_cli.config import cfg_get, load_config

            m = cfg_get(load_config(), "model") or {}
            if isinstance(m, dict):
                if m.get("default"):
                    kwargs["model"] = m["default"]
                if m.get("provider"):
                    kwargs["provider"] = m["provider"]
                if m.get("base_url"):
                    kwargs["base_url"] = m["base_url"]
                if m.get("api_mode"):
                    kwargs["api_mode"] = m["api_mode"]
        except Exception:  # noqa: BLE001
            pass
        if self._model:
            kwargs["model"] = self._model
        key = os.getenv("AZURE_FOUNDRY_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        if key and "api_key" not in kwargs:
            kwargs["api_key"] = key
        return kwargs

    def _run_sync(self, query: str) -> str:
        if self._agent is None:
            from run_agent import AIAgent  # heavy import — defer to first consult

            self._agent = AIAgent(**self._agent_kwargs())
        return self._agent.chat(query)

    async def ask(self, query: str, *, timeout_s: float = 45.0) -> str:
        """Run ``query`` through the agent and return a concise spoken result."""
        query = (query or "").strip()
        if not query:
            return "I didn't catch what you wanted me to look into."
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, query), timeout=timeout_s
            )
            return (result or "").strip() or "I didn't find anything to report."
        except asyncio.TimeoutError:
            return "That's taking a while — I'll keep working and follow up."
        except Exception:  # noqa: BLE001 — never let a consult crash the call
            logger.error("[teams_voice] agent consult failed", exc_info=True)
            return "Sorry, I ran into an error working on that."
