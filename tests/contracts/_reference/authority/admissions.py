"""Reference implementation of ``AdmissionRepository`` in memory, every verb implemented.

The verbs live in four mixins: ``admissions_admit`` (``admit``, ``retired_replay``), ``admissions_writes`` (the lifecycle writes), ``admissions_reads`` (the reads and the API payload compaction pair) and ``admissions_media`` (media holds and retirement candidates); ``admissions_rows`` holds the row shape and the shared steps, ``admissions_results`` the stored-result codecs and the discarded-turn closer. Every public method runs under the backend-wide lock, so a verb is one atomic step; writes stage through ``AuthorityState.transaction`` or, when they compose a sibling repository's verbs, run under ``AuthorityState.guarded``.

Reference limits: worker read-copy compaction at settlement belongs to the worker reference and is not run here; ``admit(hosted=)`` calls the hosted-room reference's ``hosted_member_check`` and refuses ``CapabilityUnsupported`` when that reference does not offer it.
"""

from __future__ import annotations

from typing import Any

from persistence.core.errors import AuthorityRefusal, NotFound

from .._authority import AuthorityState, authority_state
from . import admissions_rows as rows
from ..backend import register_repository
from .admissions_admit import AdmitMixin
from .admissions_media import AdmissionMediaMixin
from .admissions_reads import AdmissionReadsMixin
from .admissions_rows import _bound_id, _identity_key, _result_key, _terminal_key
from .admissions_writes import CLAIM_FAILPOINT, SETTLE_FAILPOINT, AdmissionWritesMixin

__all__ = [
    "CLAIM_FAILPOINT",
    "SETTLE_FAILPOINT",
    "InMemoryAdmissionRepository",
    "_bound_id",
    "_identity_key",
    "_result_key",
    "_terminal_key",
]


class InMemoryAdmissionRepository(AdmitMixin, AdmissionWritesMixin, AdmissionReadsMixin, AdmissionMediaMixin):
    """See ``persistence.authority.admissions.AdmissionRepository`` for the contract each method implements."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def _state(self) -> AuthorityState:
        return authority_state(self._backend, self._backend.scope)

    def _tables(self) -> dict[str, Any]:
        return self._backend.scope_store(self._backend.scope)

    @staticmethod
    def _admission(authority: dict[str, Any], admission_id: Any) -> dict[str, Any]:
        """The live row or its tombstone for the worker reference's receipt steps, which refuse an absent one as ``NotFound``."""
        try:
            return rows.admission(authority, admission_id)
        except AuthorityRefusal as exc:
            raise NotFound("admission", str(admission_id)) from exc


register_repository(
    "admissions",
    InMemoryAdmissionRepository,
    {"durable_admissions": True},
)
