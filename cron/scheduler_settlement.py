"""Trusted settlement shared by cron session booking and fire completion."""
from __future__ import annotations

import logging

from agent.runtime_policy import is_authoritative
from cron.executions import get_settlement_for_session, record_execution_settlement
from hermes_time import now

logger = logging.getLogger(__name__)


def trusted_terminal_status(job: dict, execution_id: str | None, session_id: str | None) -> str | None:
    """Only settled proof for this policy, execution and final session can classify a tool tail."""
    policy = job.get("runtime_policy")
    if not is_authoritative(policy) or not execution_id or not session_id:
        return None
    try:
        proof = get_settlement_for_session(session_id)
        if not isinstance(proof, dict) or (
            proof.get("execution_id"), proof.get("job_id"), proof.get("session_id")
        ) != (execution_id, job["id"], session_id):
            return None
        settlement = proof.get("settlement")
        if not isinstance(settlement, dict) or settlement.get("settled") is not True:
            return None
        outcome, receipt = settlement.get("outcome"), settlement.get("receipt")
        if not isinstance(outcome, dict) or not isinstance(receipt, dict):
            return None
        run_id = f"cron:{job['id']}:{execution_id}"
        if (outcome.get("policy"), outcome.get("run_id")) != (policy, run_id):
            return None
        if (receipt.get("status"), receipt.get("policy"), receipt.get("run_id"), receipt.get("session_id")) != (
            "finalized", policy, run_id, session_id
        ):
            return None
        if not isinstance(outcome.get("reason"), str) or not outcome["reason"]:
            return None
        status = outcome.get("status")
        return status if status in ("success", "failure") else None
    except Exception:
        logger.warning("Job '%s': trusted settlement could not be read", job.get("id"), exc_info=True)
        return None


def settle_run(agent, job: dict, execution_id: str | None, session_id: str, result: dict | None) -> None:
    """Required finalization precedes session teardown and durable completion classification."""
    from hermes_cli.lifecycle import finalize_session

    result = result or {}
    receipts = finalize_session(
        session_id=session_id, runtime_run_id=agent.runtime_task_id,
        runtime_policy=job.get("runtime_policy"), platform="cron",
        cron_job_id=job["id"], cron_job_name=job.get("name"), cron_max_turns=job.get("max_turns"),
        completed=result.get("completed") is True, failed=result.get("failed") is True,
        terminal_outcome=result.get("trusted_terminal_outcome"),
    )
    outcome = result.get("trusted_terminal_outcome")
    if outcome is None:
        return
    receipt = receipts[0] if receipts else None
    if not execution_id or not isinstance(outcome, dict) or not isinstance(receipt, dict):
        raise RuntimeError("trusted terminal outcome could not be settled durably")
    recorded = record_execution_settlement(
        execution_id, session_id=session_id,
        settlement={"outcome": outcome, "receipt": receipt, "settled": True,
                    "recorded_at": now().isoformat()},
    )
    if not recorded or trusted_terminal_status(job, execution_id, session_id) is None:
        raise RuntimeError("trusted terminal outcome could not be settled durably")


def classify_completion(job, execution_id, session_id, success, error, final_response):
    """Preserve runtime failures; validated trusted success may finish without assistant prose."""
    status = trusted_terminal_status(job, execution_id, session_id)
    if status == "failure":
        return False, error or "Authoritative runtime policy reported failure"
    if success and not final_response.strip() and status != "success":
        return False, "Agent completed but produced empty response (model error, timeout, or misconfiguration)"
    return success, error
