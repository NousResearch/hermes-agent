"""Abstract base for proxy upstream adapters.

An :class:`UpstreamAdapter` represents one OAuth-authenticated provider the
local proxy can forward requests to. The adapter is responsible for:

  - locating the user's auth state for that provider
  - refreshing/minting credentials when needed
  - reporting the resolved upstream base URL
  - declaring which request paths it accepts

The proxy server is otherwise provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class UpstreamCredential:
    """A resolved bearer + base URL ready to forward to."""

    bearer: str
    """Authorization header value to send upstream (token only, no ``Bearer`` prefix)."""

    base_url: str
    """Upstream base URL, e.g. ``https://inference-api.nousresearch.com/v1``."""

    token_type: str = "Bearer"
    """Auth scheme — currently always ``Bearer`` for supported providers."""

    expires_at: Optional[str] = None
    """ISO-8601 expiry timestamp for the bearer, when known. Informational."""


class UpstreamAdapter(ABC):
    """Contract for an upstream provider the proxy can forward to."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter key used on the CLI (e.g. ``"nous"``)."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name for logs and ``proxy status``."""

    @property
    @abstractmethod
    def allowed_paths(self) -> FrozenSet[str]:
        """Set of relative request paths the upstream accepts.

        Paths are relative to the proxy's ``/v1`` mount point. For example,
        ``"/chat/completions"`` corresponds to a client request to
        ``http://127.0.0.1:<port>/v1/chat/completions``. Requests to paths
        not in this set get a 404 with a helpful error body.
        """

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Return True if the user has usable credentials for this upstream.

        Should be cheap — no network calls. Used by ``proxy start`` for a
        clear up-front error before binding a port.
        """

    @abstractmethod
    def get_credential(self) -> UpstreamCredential:
        """Return a fresh credential, refreshing/minting if necessary.

        Implementations should:
          - refresh the access token if it's near expiry
          - mint/rotate the upstream bearer key if it's near expiry
          - persist any refreshed state back to disk

        Raises:
            RuntimeError: if the user isn't authenticated or the upstream
              refresh fails. The proxy will return 401 to the client.
        """

    def describe(self) -> str:
        """One-line status summary for ``proxy status``."""
        try:
            cred = self.get_credential()
        except Exception as exc:  # pragma: no cover - defensive
            return f"{self.display_name}: not ready ({exc})"
        ttl = f" (expires {cred.expires_at})" if cred.expires_at else ""
        return f"{self.display_name}: {cred.base_url}{ttl}"


__all__ = ["UpstreamAdapter", "UpstreamCredential"]
