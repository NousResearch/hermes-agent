"""hermes_storage.registry — thin façade over FactRetriever for multi-entity reasoning.

Exposes:
  - ReasonQuery  : typed dataclass that captures the inputs to a reason call.
  - MemoryRegistry : lightweight registry that holds a FactRetriever and
                     delegates ``reason`` queries to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.memory.holographic.retrieval import FactRetriever


@dataclass
class ReasonQuery:
    """Typed container for a multi-entity compositional reasoning request.

    Attributes
    ----------
    entities:
        One or more entity names to intersect structurally.  Must be
        non-empty; the registry will raise ``ValueError`` otherwise.
    category:
        Optional category filter forwarded to the underlying retriever.
    limit:
        Maximum number of facts to return (default 10).
    """

    entities: list[str]
    category: str | None = None
    limit: int = 10

    def __post_init__(self) -> None:
        if not self.entities:
            raise ValueError("ReasonQuery.entities must contain at least one entity.")
        if self.limit < 1:
            raise ValueError("ReasonQuery.limit must be a positive integer.")


class MemoryRegistry:
    """Registry that wraps a :class:`FactRetriever` and exposes a ``reason`` entry-point.

    Parameters
    ----------
    retriever:
        A fully-initialised ``FactRetriever`` instance.  The registry does
        not own the retriever's lifecycle — callers are responsible for
        opening / closing the underlying store.
    """

    def __init__(self, retriever: "FactRetriever") -> None:
        self._retriever = retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(self, query: ReasonQuery) -> list[dict]:
        """Execute a multi-entity compositional query via the retriever.

        Delegates directly to :py:meth:`FactRetriever.reason` using the
        fields of *query*.

        Parameters
        ----------
        query:
            A :class:`ReasonQuery` instance describing the request.

        Returns
        -------
        list[dict]
            Scored fact dicts, sorted by score descending, length ≤
            ``query.limit``.  Each dict contains at minimum the keys
            ``fact_id``, ``content``, ``category``, ``trust_score``, and
            ``score``.
        """
        return self._retriever.reason(
            entities=query.entities,
            category=query.category,
            limit=query.limit,
        )
