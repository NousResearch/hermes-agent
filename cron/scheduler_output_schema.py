"""Opt-in structured results for cron jobs (``output_schema`` on the job record).

The schema is fixed when the job is created or edited, so the contract appended to each fire's
prompt is the same bytes every run. Validation and the single correction turn are delegation's
(``tools.delegation_output_schema``); a fire whose answer still fails the schema fails the run
instead of delivering free text a consumer cannot parse.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from tools.delegation_output_schema import append_output_contract, parse_validated_output, settle_output

# Run-side verdict ``run_job`` stamps on its job copy; the bookkeeping tail moves it onto the
# execution ledger row (``cron.executions.finish_execution(result_json=...)``).
STRUCTURED_RESULT_FLAG = "_structured_result"


class OutputSchemaNotSatisfied(RuntimeError):
    """The final answer (after the one correction turn) does not validate against the schema."""

    def __init__(self, errors: list[str], retries: int) -> None:
        suffix = " (after 1 correction turn)" if retries else ""
        super().__init__(f"output_schema not satisfied{suffix}: " + "; ".join(errors[:5]))


def with_output_contract(job: dict[str, Any], prompt: str) -> str:
    """*prompt* plus the OUTPUT CONTRACT block when the job declares a schema."""
    schema = job.get("output_schema")
    return append_output_contract(prompt, schema) if isinstance(schema, dict) else prompt


_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def settle_cron_result(
    job: dict[str, Any], text: str, result: dict[str, Any],
    run_turn: Callable[[str, Optional[list]], dict[str, Any]], final_text: Callable[[dict[str, Any]], str],
) -> str:
    """Validate a schema job's final answer; returns the JSON text to deliver.

    ``run_turn(message, history)`` runs the correction turn in the same conversation and
    ``final_text`` is the same completion check the first turn passed (it raises on a failed
    turn, which fails the run). The value is secret-redacted BEFORE the final validation, so the
    stored and delivered object is exactly the one that satisfied the schema; an answer that
    misses (or that redaction broke) raises ``OutputSchemaNotSatisfied``. The correction turn's
    session token counters replace the first turn's on *result* (the fire's usage audit reads
    them). Schema-less jobs return *text* unchanged.
    """
    schema = job.get("output_schema")
    if not isinstance(schema, dict):
        return text
    history = result.get("messages") if isinstance(result.get("messages"), list) else None

    def _retry(message: str) -> Optional[str]:
        retried = run_turn(message, history)
        result.update({key: retried[key] for key in _USAGE_KEYS if key in retried})
        return final_text(retried)

    settled = settle_output(text, schema, None if result.get("interrupted") else _retry)
    if not settled.valid:
        raise OutputSchemaNotSatisfied(settled.errors, settled.retries)
    from tools.delegation_redaction import redact_child_value
    safe = json.dumps(redact_child_value(settled.value, force=True), ensure_ascii=False, indent=2)
    value, errors = parse_validated_output(safe, schema)
    if errors:
        raise OutputSchemaNotSatisfied(
            ["the secret-redacted answer no longer satisfies the schema"] + errors, settled.retries)
    job[STRUCTURED_RESULT_FLAG] = json.dumps(value, ensure_ascii=False)
    return safe
