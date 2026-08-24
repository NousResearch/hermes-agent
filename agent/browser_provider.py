"""Browser Provider ABC: pluggable cloud browser backends (Browserbase, Browser Use, Firecrawl, …).

Providers register via :meth:`PluginContext.register_browser_provider`; the active one (selected by
``browser.cloud_provider``) services every cloud-mode ``browser_*`` tool call. They live in
``<repo>/plugins/browser/<name>/`` (built-in) or ``~/.hermes/plugins/browser/<name>/`` (user).

Session metadata contract (legacy ``CloudBrowserProvider`` shape; ``tools.browser_tool`` needs no
translation). ``bb_session_id`` is a legacy key name kept verbatim — it holds the provider's session ID
regardless of provider::

    {
        "session_name": str,        # unique name for agent-browser --session
        "bb_session_id": str,       # provider session ID (for close/cleanup)
        "cdp_url": str,             # CDP websocket URL
        "expires_at": str,          # optional provider-authoritative ISO timestamp
        "features": dict,           # feature flags that were enabled
        "external_call_id": str,    # optional, managed-gateway billing key
    }
"""

from __future__ import annotations

import abc
from typing import Dict

from agent.provider_base import ProviderBase


class BrowserProvider(ProviderBase):
    """Abstract base class for a cloud browser backend.

    Subclasses implement :attr:`name` (the ``browser.cloud_provider`` value), :meth:`is_available`, and
    the lifecycle trio :meth:`create_session` / :meth:`close_session` / :meth:`emergency_cleanup`.
    ``get_setup_schema`` may add ``"post_setup"`` (e.g. ``"agent_browser"``) to trigger the install hook.
    """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this provider can service calls. Cheap check only (env var, token readable, dep
        importable) — must NOT make network calls; runs at tool-registration time and on every
        ``hermes tools`` paint."""

    @abc.abstractmethod
    def create_session(self, task_id: str) -> Dict[str, object]:
        """Create a cloud browser session and return the metadata dict from the module docstring.
        May raise ``ValueError`` (missing credentials) or ``RuntimeError`` (network / API failure);
        the dispatcher surfaces these to the user."""

    @abc.abstractmethod
    def close_session(self, session_id: str) -> bool:
        """Release a cloud session by provider session ID. Returns True on success, False on failure;
        should not raise (log and return False so the dispatcher's cleanup loop keeps moving)."""

    @abc.abstractmethod
    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort teardown from atexit / signal handlers. Must tolerate missing credentials and
        network errors; must not raise."""


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
