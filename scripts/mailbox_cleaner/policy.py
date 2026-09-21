from __future__ import annotations

from dataclasses import dataclass


class PolicyError(ValueError):
    pass


_READ_OPERATIONS = frozenset({"list_metadata", "get_metadata"})


@dataclass(frozen=True, slots=True)
class AccountPolicy:
    account: str
    provider: str
    allow_mutations: bool = False

    def __post_init__(self) -> None:
        if not self.account.strip() or not self.provider.strip():
            raise PolicyError("account and provider are required")
        if self.allow_mutations:
            raise PolicyError("scheduled mailbox policies are read-only")

    def allows(self, operation: str) -> bool:
        return operation in _READ_OPERATIONS
