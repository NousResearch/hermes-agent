"""Ares candidate GC façade; lifecycle approval remains mandatory."""

from hermes_cli.ares_candidate_store import CandidateStore


def gc_candidate(
    sealed_candidate_id: str, approval: dict, store: CandidateStore | None = None
) -> dict:
    return (store or CandidateStore()).gc(sealed_candidate_id, approval)
