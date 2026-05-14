"""
Browser Provider ABC
====================

Defines the pluggable-backend interface for cloud browser providers
(Browserbase, Browser Use, Firecrawl, …). Providers register instances via
:meth:`PluginContext.register_browser_provider`; the active one (selected via
``browser.cloud_provider`` in ``config.yaml``) services every cloud-mode
``browser_*`` tool call.

Providers live in ``<repo>/plugins/browser/<name>/`` (built-in, auto-loaded as
``kind: backend``) or ``~/.hermes/plugins/browser/<name>/`` (user, opt-in via
``plugins.enabled``).

This ABC mirrors :class:`agent.web_search_provider.WebSearchProvider` (PR
#25182) — same shape, same registration flow, same picker integration. The
legacy in-tree ``tools.browser_providers.base.CloudBrowserProvider`` ABC was
deleted in PR #25214 (this work) along with the per-vendor inline modules in
``tools/browser_providers/``; the lifecycle contract documented below is
preserved bit-for-bit so the tool wrapper (:mod:`tools.browser_tool`) does
not have to translate.

Session metadata contract (preserved from the legacy ``CloudBrowserProvider``)::

    {
        "session_name": str,        # unique name for agent-browser --session
        "bb_session_id": str,       # provider session ID (for close/cleanup)
        "cdp_url": str,             # CDP websocket URL
        "features": dict,           # feature flags that were enabled
        "external_call_id": str,    # optional, managed-gateway billing key
    }

``bb_session_id`` is a legacy key name kept verbatim for backward compat with
:mod:`tools.browser_tool` — it holds the provider's session ID regardless of
which provider is in use.
"""

from __future__ import annotations

import abc
from typing import Any, Dict


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class BrowserProvider(abc.ABC):
    """Abstract base class for a cloud browser backend.

    Subclasses must implement :meth:`name`, :meth:`is_available`, and the
    three lifecycle methods: :meth:`create_session`, :meth:`close_session`,
    :meth:`emergency_cleanup`.

    The lifecycle shape preserves the legacy ``CloudBrowserProvider`` contract
    bit-for-bit so the dispatcher in :mod:`tools.browser_tool` is a pure
    registry lookup — no per-provider conditionals, no shape translation.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in the ``browser.cloud_provider``
        config key.

        Lowercase, hyphens permitted to preserve existing user-visible names.
        Examples: ``browserbase``, ``browser-use``, ``firecrawl``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``. Defaults to ``name``."""
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Typically a cheap check (env var present, managed-gateway token
        readable, optional Python dep importable). Must NOT make network
        calls — this runs at tool-registration time and on every
        ``hermes tools`` paint.

        Mirrors the legacy ``CloudBrowserProvider.is_configured()`` method;
        renamed for parity with :class:`agent.web_search_provider.WebSearchProvider`.
        """

    @abc.abstractmethod
    def create_session(self, task_id: str) -> Dict[str, object]:
        """Create a cloud browser session and return session metadata.

        Must return a dict with at least::

            {
                "session_name": str,    # unique name for agent-browser --session
                "bb_session_id": str,   # provider session ID (for close/cleanup)
                "cdp_url": str,         # CDP websocket URL
                "features": dict,       # feature flags that were enabled
            }

        ``bb_session_id`` is a legacy key name kept for backward compat with
        the rest of :mod:`tools.browser_tool` — it holds the provider's
        session ID regardless of which provider is in use.

        May raise ``ValueError`` (missing credentials) or ``RuntimeError``
        (network / API failure); the dispatcher surfaces these to the user.
        """

    @abc.abstractmethod
    def close_session(self, session_id: str) -> bool:
        """Release / terminate a cloud session by its provider session ID.

        Returns True on success, False on failure. Should not raise — log and
        return False on any exception so the dispatcher's cleanup loop keeps
        moving across sessions.
        """

    @abc.abstractmethod
    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort session teardown during process exit.

        Called from atexit / signal handlers. Must tolerate missing
        credentials, network errors, etc. — log and move on. Must not raise.
        """

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``hermes tools`` picker.

        Used by :mod:`hermes_cli.tools_config` to inject this provider as a
        row in the Browser Automation picker. Shape mirrors the existing
        hardcoded entries in ``TOOL_CATEGORIES["browser"]``::

            {
                "name": "Browserbase",
                "badge": "paid",
                "tag": "Cloud browser with stealth and proxies",
                "env_vars": [
                    {"key": "BROWSERBASE_API_KEY",
                     "prompt": "Browserbase API key",
                     "url": "https://browserbase.com"},
                ],
                "post_setup": "agent_browser",
            }

        Default: minimal entry derived from :attr:`display_name`. Override to
        expose API key prompts, badges, managed-Nous gating, and the
        ``post_setup`` install hook.
        """
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }

    # ------------------------------------------------------------------
    # Backward-compat shims for the legacy CloudBrowserProvider API
    # ------------------------------------------------------------------
    #
    # The pre-PR-#25214 ABC exposed ``is_configured()`` and ``provider_name()``;
    # ``tools.browser_tool`` has ~6 callers that still use those names. Rather
    # than churn every callsite (and break out-of-tree downstream code that
    # subclassed CloudBrowserProvider), we expose the old names as thin
    # delegations to the new API. Subclasses MUST implement :meth:`is_available`
    # and :attr:`name`; they may override ``is_configured`` / ``provider_name``
    # for compatibility with the legacy ABC but it is not required.

    def is_configured(self) -> bool:  # pragma: no cover - trivial delegation
        """Backward-compat alias for :meth:`is_available`."""
        return self.is_available()

    def provider_name(self) -> str:  # pragma: no cover - trivial delegation
        """Backward-compat alias returning :attr:`display_name`."""
        return self.display_name
