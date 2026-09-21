from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .models import Observation
from .policy import AccountPolicy

ReadAdapter = Callable[[str, str], Sequence[Mapping[str, str]]]


class ReadOnlyAdapterError(RuntimeError):
    pass


class MetadataClient:
    """Allowlisted boundary around an injected MCP read adapter.

    A controlled live MCP adapter is intentionally not wired here; callers must
    supply one after its read-only contract has been validated.
    """

    def __init__(self, policy: AccountPolicy, adapter: ReadAdapter) -> None:
        self._policy = policy
        self._adapter = adapter

    def call(self, operation: str, *, account: str) -> list[Observation]:
        if not account or account != self._policy.account:
            raise ReadOnlyAdapterError("every read requires the configured explicit account")
        if not self._policy.allows(operation):
            raise ReadOnlyAdapterError(f"operation is not allowlisted: {operation}")
        rows = self._adapter(operation, account)
        return [
            Observation(
                account=account,
                provider=self._policy.provider,
                message_id=str(row.get("id", "")),
                sender=str(row.get("from", "")),
                subject=str(row.get("subject", "")),
                received_at=str(row.get("received_at", "")),
                preview=str(row.get("preview", "")),
            )
            for row in rows
        ]

    def list_metadata(self, *, account: str) -> list[Observation]:
        return self.call("list_metadata", account=account)
