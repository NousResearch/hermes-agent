"""Reference implementation of ``SessionRuntimeRepository`` in memory, every verb implemented.

The verbs live in ``runtime_meta`` (epoch, picks, policy refresh, viewer closes, retirement reads, the ledger-gated ends) and ``mutation`` (``prepare_mutation``, ``mutate``), with the command handlers in ``mutation_commands`` and the guards and local snapshot in ``mutation_guards``. Every public method runs under the backend-wide lock, so a verb is one atomic step. The local policy receipt and history-control bindings are the binding reference's own stored values, so both repositories read one store.

Reference limits: ``mutate`` with ``Reset``, ``Branch`` or ``Compress`` runs its fences and then refuses ``CapabilityUnsupported`` (see ``mutation_commands``); the rewind receipt's ``target_message`` is the reference's replay projection of the row.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .._authority import AuthorityState, authority_state, record_retirement
from ..backend import register_repository
from .mutation import RECEIPT_FAILPOINT, MutationMixin
from .runtime_meta import RuntimeMetaMixin

__all__ = ["RECEIPT_FAILPOINT", "InMemorySessionRuntimeRepository"]


class InMemorySessionRuntimeRepository(MutationMixin, RuntimeMetaMixin):
    """See ``persistence.authority.runtime.SessionRuntimeRepository`` for the contract each method implements."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def _state(self) -> AuthorityState:
        return authority_state(self._backend, self._backend.scope)

    def _tables(self) -> dict[str, Any]:
        return self._backend.scope_store(self._backend.scope)

    # ── fixture seeding seams ─────────────────────────────────────────────────

    def seed_local_receipt(self, session_id: str, receipt: Mapping[str, Any]) -> None:
        """Store ``receipt`` where the binding reference keeps a local policy receipt."""
        from .bindings import _RECEIPT_PREFIX

        with self._state().transaction() as staged:
            staged.authority["meta"][_RECEIPT_PREFIX + session_id] = copy.deepcopy(dict(receipt))

    def seed_retirement(self, session_id: str, *, expires: bool = False) -> None:
        """Record the retirement marker of ``session_id`` alone; ``expires`` makes it a sweep's marker."""
        with self._state().transaction() as staged:
            record_retirement(staged.authority, session_id, expires=expires)


register_repository("runtime", InMemorySessionRuntimeRepository, {"authority_fences": True})
