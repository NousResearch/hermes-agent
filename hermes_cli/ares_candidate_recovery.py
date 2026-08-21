"""Ares candidate fresh-process recovery façade."""

from hermes_cli.ares_candidate_store import CandidateStore


def recover_candidates(store: CandidateStore | None = None) -> list[dict]:
    return (store or CandidateStore()).recover()
