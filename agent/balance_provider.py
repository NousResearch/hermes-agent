"""
Provider Balance/Credits ABC
=============================

Defines the pluggable interface for fetching balance/credit information from
an AI provider. Providers register subclasses via ``BalanceProviderRegistry``;
the Desktop frontend displays the active provider's balance via the
``balance.view`` RPC method.

Adding a new provider is one file + one registration call — no frontend changes
needed. See ``plugins/model-providers/kilocode/balance.py`` for an example.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderBalance:
    """Canonical balance data returned by every BalanceProvider.

    ``value`` is a float denominated in the provider's currency (usually USD).
    ``currency`` is the ISO 4217 code (default ``"USD"``). ``label`` is a
    human-friendly short string like ``"Kilo AI"``.

    ``is_depleted`` is a hint from the provider API (not derived from
    ``value > 0`` — some providers have a separate flag). ``fetched_at`` is
    set by the registry, not the provider subclass.

    When ``error`` is non-None the fetch failed and ``value`` should not
    be displayed.
    """

    provider_name: str
    label: str
    value: float
    currency: str = "USD"
    is_depleted: bool = False
    fetched_at: float = 0.0  # unix timestamp
    error: Optional[str] = None  # non-None → fetch failed

    def __str__(self) -> str:
        """Compact display string, e.g. ``"$6.61"`` or ``"125,000"``."""
        if self.error:
            return ""
        if self.currency == "USD":
            return f"${self.value:.2f}"
        return f"{self.value:.2f}"


@dataclass(frozen=True)
class BalanceConfig:
    """Per-provider config snippet from ``config.yaml providers.<name>.balance:``.

    Default values match the Kilo endpoint; providers override as needed.
    """

    endpoint: str = ""
    api_key_env: str = ""
    enabled: bool = True
    # How long (in seconds) cached balance data is considered fresh.
    cache_ttl_seconds: float = 60.0


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class BalanceProvider(ABC):
    """One registered balance provider.

    Subclass name must match the Hermes provider slug (e.g.
    ``KiloBalanceProvider`` lives in the ``kilocode`` plugin; its
    ``provider_slug`` is ``"kilocode"``).
    """

    # Must match the slug used in ``providers: {}`` config key and plugin name.
    provider_slug: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.provider_slug:
            raise TypeError(
                f"{cls.__name__} must define a non-empty 'provider_slug' class variable"
            )

    @abstractmethod
    def fetch(self, api_key: str, config: BalanceConfig) -> ProviderBalance:
        """Fetch current balance from the provider API.

        Called on a thread off the event loop (blocking I/O is fine). Must
        return a ``ProviderBalance`` — if the API is unreachable, set
        ``error`` on the return value (do not raise).
        """

    @classmethod
    def default_config(cls) -> BalanceConfig:
        """Default ``BalanceConfig`` when none is specified in config.yaml."""
        return BalanceConfig()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class BalanceProviderRegistry:
    """Holds all registered ``BalanceProvider`` subclasses.

    Registration is automatic via ``register()`` called alongside the existing
    ``register_provider()`` in each model provider plugin's ``__init__.py``.
    """

    _providers: dict[str, type[BalanceProvider]] = {}
    _cache: dict[str, ProviderBalance] = {}
    _last_fetch: dict[str, float] = {}  # provider_slug → unix timestamp
    _fetch_in_flight: set[str] = set()  # provider_slugs currently being fetched
    _lock = threading.Lock()

    @classmethod
    def register(cls, provider_cls: type[BalanceProvider]) -> None:
        """Register a ``BalanceProvider`` subclass."""
        slug = provider_cls.provider_slug
        if not slug:
            raise ValueError(f"{provider_cls.__name__} must set provider_slug")
        with cls._lock:
            cls._providers[slug] = provider_cls
        logger.info("BalanceProvider registered: %s", slug)

    @classmethod
    def registered_slugs(cls) -> frozenset[str]:
        """Return all registered provider slugs."""
        with cls._lock:
            return frozenset(cls._providers)

    @classmethod
    def get(cls, slug: str) -> type[BalanceProvider] | None:
        """Return the registered provider class for *slug*, or None."""
        with cls._lock:
            return cls._providers.get(slug)

    @classmethod
    def get_cached(cls, slug: str) -> ProviderBalance | None:
        """Return cached balance for *slug*, or None."""
        with cls._lock:
            return cls._cache.get(slug)

    @classmethod
    def cached_or_fetch(
        cls,
        slug: str,
        api_key: str,
        config: BalanceConfig,
        *,
        force: bool = False,
    ) -> tuple[ProviderBalance, bool]:
        """Return cached balance if fresh, else fetch and cache.

        Returns ``(ProviderBalance, was_cached)`` where ``was_cached`` is
        True when the result came from the in-memory cache (fresh enough
        per ``cache_ttl_seconds``).

        If another thread is already fetching the same slug, the second
        caller receives stale cache rather than issuing a duplicate HTTP
        request.
        """
        now = time.time()

        # Return cached if fresh enough and not forced.
        with cls._lock:
            if slug in cls._fetch_in_flight:
                # Another thread is already fetching; return stale cache.
                cached = cls._cache.get(slug)
                if cached:
                    return cached, True

            if not force:
                last = cls._last_fetch.get(slug, 0.0)
                cached = cls._cache.get(slug)
                if cached and (now - last) < config.cache_ttl_seconds:
                    return cached, True

            # Mark in-flight so concurrent callers see stale cache.
            cls._fetch_in_flight.add(slug)

        # Fetch (outside the lock to avoid holding during I/O).
        provider_cls = cls.get(slug)
        if provider_cls is None:
            with cls._lock:
                cls._fetch_in_flight.discard(slug)
            return ProviderBalance(
                provider_name=slug,
                label=slug,
                value=0.0,
                error=f"No BalanceProvider registered for '{slug}'",
            ), False

        try:
            balance = provider_cls().fetch(api_key, config)
            balance = ProviderBalance(
                provider_name=balance.provider_name,
                label=balance.label,
                value=balance.value,
                currency=balance.currency,
                is_depleted=balance.is_depleted,
                fetched_at=now,
                error=balance.error,
            )
        except Exception as exc:
            logger.debug("BalanceProvider '%s' fetch failed: %s", slug, exc)
            balance = ProviderBalance(
                provider_name=slug,
                label=slug,
                value=0.0,
                fetched_at=now,
                error=str(exc),
            )

        with cls._lock:
            cls._cache[slug] = balance
            cls._last_fetch[slug] = now
            cls._fetch_in_flight.discard(slug)

        return balance, False