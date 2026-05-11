"""Aggregation and ingestion helpers for the failure analysis subsystem.

Provides:
- Eval failure ingestion: converts failed eval case results into normalized failures
- Aggregation queries on top of FailureStore
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .classifier import classify_eval_case
from .storage import FailureStore
from .types import NormalizedFailure

logger = logging.getLogger(__name__)


def ingest_eval_failures(
    run_id: str,
    case_results: list[dict[str, Any]],
    store: Optional[FailureStore] = None,
    prior_results: Optional[dict[str, str]] = None,
) -> list[NormalizedFailure]:
    """Convert failed eval case results into normalized failures and persist them.

    Args:
        run_id: The eval run ID.
        case_results: List of case result dicts (from EvalStore.get_case_results).
        store: Optional FailureStore to persist into. If None, creates one.
        prior_results: Optional dict mapping case_id -> prior status for
                       regression detection.

    Returns:
        List of NormalizedFailure records created.
    """
    failures: list[NormalizedFailure] = []
    for cr in case_results:
        status = cr.get("status", "")
        if status in ("passed", "skipped"):
            continue

        prior_status = None
        if prior_results:
            prior_status = prior_results.get(cr.get("case_id", ""))

        nf = classify_eval_case(cr, run_id, prior_status=prior_status)
        failures.append(nf)

    if failures and store is not None:
        store.insert_many(failures)
        logger.info(
            "Ingested %d failure(s) from eval run %s", len(failures), run_id
        )

    return failures


def get_top_failures(
    store: FailureStore,
    window_days: int = 7,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the top recurring failure patterns."""
    return store.top_fingerprints(
        window_seconds=window_days * 86400,
        limit=limit,
    )


def get_failure_summary(store: FailureStore) -> dict[str, Any]:
    """Return a compact summary of failure patterns for reporting.

    Includes counts, top patterns, and type distribution.
    """
    total_24h = store.count_total(window_seconds=86400)
    total_7d = store.count_total(window_seconds=7 * 86400)
    total_all = store.count_total()
    top = store.top_fingerprints(window_seconds=7 * 86400, limit=5)

    return {
        "total_24h": total_24h,
        "total_7d": total_7d,
        "total_all": total_all,
        "top_patterns": top,
    }
