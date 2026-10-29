"""Browser page backend ABC: a browser that owns its pages outside Hermes's CDP session.

``browser_vault`` inspects and fills the page the agent is driving. For ``browser_exec`` that page sits
behind the task's CDP supervisor. A plugin browser with its own transport (Camofox's REST server) has
no CDP endpoint, so it registers one of these via :meth:`PluginContext.register_browser_page_backend`
and the vault evaluates its inspection, origin-check and fill scripts through it instead.

Core keeps the secret contract: it wraps every secret-bearing script so a page exception never reaches
the backend's error text, replaces any failure on that path with fixed text, and registers the filled
value with the browser-result redaction boundary. The backend only carries the script to its page.
"""

from __future__ import annotations

import abc
from typing import Any

from agent.provider_base import ProviderBase


class BrowserPageBackend(ProviderBase):
    """A plugin browser whose live page ``browser_vault`` can evaluate JavaScript on."""

    @abc.abstractmethod
    def has_page(self, task_id: str) -> bool:
        """True when this backend already holds a live page for ``task_id`` (the agent navigated it).

        Cheap and side-effect free: never open a tab or a session to answer. ``False`` lets the vault
        fall back to the CDP session."""

    @abc.abstractmethod
    def evaluate(self, task_id: str, expression: str, *, secret: bool) -> Any:
        """Evaluate ``expression`` on the task's current page and return its value; raise on failure.

        Never open a tab. With ``secret=True`` the expression embeds a credential: send it only over a
        transport that keeps it out of argv and off the network in clear text (loopback or HTTPS), never
        follow a redirect with it, and never log it. Exception text on that path is discarded by core."""
