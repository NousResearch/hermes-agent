"""Provider abstraction for the finance plugin.

A :class:`FinanceProvider` is the seam that keeps the finance feature
provider-agnostic. Plaid is the only concrete implementation today, but the
issue (#51697) explicitly calls for additional connectors (Teller, SnapTrade,
Salt Edge, TrueLayer, crypto exchanges) without redesigning the storage,
sync, or tool layers. Everything above this seam talks to accounts,
transactions, holdings, and liabilities in the local store; everything below
it talks to a specific aggregator's API.

Providers are resolved lazily by name so importing this module never pulls in
a backend's HTTP client (or its optional deps) until it is actually used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from plugins.finance.errors import FinanceConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plugins.finance.store import FinanceStore


class FinanceProvider(ABC):
    """Abstract financial-data provider.

    Concrete providers wrap a single aggregator. The contract is intentionally
    small: link/unlink an institution, pull fresh data into the store, and
    report status. All money/data persistence happens through the passed-in
    :class:`~plugins.finance.store.FinanceStore`; providers never own storage.
    """

    #: Stable, lowercase identifier used in config and the CLI (e.g. "plaid").
    name: str = ""

    @abstractmethod
    def connect(self, **kwargs: Any) -> Dict[str, Any]:
        """Begin or complete linking an institution.

        Returns a JSON-serializable dict describing the next step (e.g. a
        hosted-link URL to open) or the completed link (item id + institution).
        """

    @abstractmethod
    def sync(self, store: "FinanceStore", *, item_id: Optional[str] = None) -> Dict[str, Any]:
        """Pull fresh balances/transactions/holdings/liabilities into *store*.

        When *item_id* is given, only that item is refreshed; otherwise every
        linked item for this provider is synced. Returns a summary dict.
        """

    @abstractmethod
    def status(self, store: "FinanceStore") -> Dict[str, Any]:
        """Return linked-item + last-sync status for this provider."""

    @abstractmethod
    def disconnect(self, store: "FinanceStore", item_id: str, *, purge: bool = False) -> Dict[str, Any]:
        """Unlink an item, revoking its credential. Optionally purge its data."""


# ---------------------------------------------------------------------------
# Lazy provider registry
# ---------------------------------------------------------------------------

# Maps provider name -> zero-arg factory. Factories import their backend module
# only when invoked, so the default import path stays light and a provider's
# optional deps are never required unless the user selected it.
_PROVIDER_FACTORIES: Dict[str, Callable[[], FinanceProvider]] = {}


def register_provider(name: str, factory: Callable[[], FinanceProvider]) -> None:
    """Register a provider factory under *name* (last writer wins)."""
    _PROVIDER_FACTORIES[name.strip().lower()] = factory


def _default_plaid_factory() -> FinanceProvider:
    from plugins.finance.plaid.provider import PlaidProvider

    return PlaidProvider()


# Built-in providers. Plaid is the reference implementation.
register_provider("plaid", _default_plaid_factory)


def available_providers() -> List[str]:
    """Return the sorted names of all registered providers."""
    return sorted(_PROVIDER_FACTORIES)


def get_provider(name: str) -> FinanceProvider:
    """Instantiate a registered provider by name.

    Raises :class:`FinanceConfigError` for an unknown provider so the CLI and
    tools surface an actionable message instead of a raw KeyError.
    """
    key = (name or "").strip().lower()
    factory = _PROVIDER_FACTORIES.get(key)
    if factory is None:
        known = ", ".join(available_providers()) or "(none)"
        raise FinanceConfigError(
            f"Unknown finance provider '{name}'. Known providers: {known}."
        )
    return factory()
