"""Native update seam for the existing Hermes CLI updater.

The engine deliberately wraps only the exercised legacy update flow. On
platforms with the hardened POSIX lock backend, concurrent repository updates
are serialized. Unsupported platforms (notably native Windows) retain the
legacy updater behavior without attempting the POSIX-only lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from hermes_cli.update_lock import (
    acquire_shared_update_lock,
    mark_shared_update_lock_post_mutation,
    shared_update_lock_supported,
)


class UpdateMutationBoundary(str, Enum):
    """Whether an update failure is known to precede repository mutation."""

    PRE_MUTATION = "pre_mutation"
    POST_MUTATION_UNCERTAIN = "post_mutation_uncertain"


@dataclass(frozen=True)
class UpdateRequest:
    """Request passed into the native update seam."""

    args: Any
    gateway_mode: bool = False


@dataclass(frozen=True)
class UpdateResult:
    """Result produced by the update engine."""

    exit_code: int = 0
    message: str = ""


@dataclass(frozen=True)
class NativeUpdateEngine:
    """Delegate the existing updater, serializing it when the lock is supported."""

    legacy_runner: Callable[[UpdateRequest], UpdateResult]
    update_lock_identity: str | None = None
    update_lock_timeout_seconds: float = 30.0

    def run(self, request: UpdateRequest) -> UpdateResult:
        if self.update_lock_identity is None or not shared_update_lock_supported():
            return self.legacy_runner(request)
        with acquire_shared_update_lock(
            self.update_lock_identity,
            timeout_seconds=self.update_lock_timeout_seconds,
        ):
            # The legacy updater can mutate immediately after entry. Mark the
            # lock conservatively before handing over control.
            mark_shared_update_lock_post_mutation()
            return self.legacy_runner(request)
