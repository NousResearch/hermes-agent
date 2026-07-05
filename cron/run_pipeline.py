"""Shared cron run pipeline orchestration."""

from __future__ import annotations

import logging
from typing import Callable, Optional


logger = logging.getLogger(__name__)


def run_one_job_pipeline(
    job: dict,
    *,
    run_job_fn: Callable[..., tuple[bool, str, str, Optional[str]]],
    save_job_output_fn: Callable[[str, str], object],
    mark_job_run_fn: Callable[..., object],
    claim_dispatch_fn: Callable[[str], bool],
    deliver_result_fn: Callable[..., Optional[str]],
    summarize_failure_fn: Callable[[dict, str | None], str],
    is_silence_response_fn: Callable[[str], bool],
    get_hermes_home_fn: Callable[[], object],
    teardown_agent_fn: Callable[[object, str], None],
    silent_marker: str,
    adapters=None,
    loop=None,
    verbose: bool = False,
    pipeline_logger: logging.Logger | None = None,
) -> bool:
    """Run one due job end-to-end: execute, save, deliver, mark."""
    log = pipeline_logger or logger
    try:
        if not claim_dispatch_fn(job["id"]):
            log.info(
                "Job '%s': one-shot dispatch limit reached — skipping",
                job.get("name", job["id"]),
            )
            return True

        from agent.secret_scope import (
            build_profile_secret_scope,
            reset_secret_scope,
            set_secret_scope,
        )

        scope_token = set_secret_scope(build_profile_secret_scope(get_hermes_home_fn()))
        deferred_agents: list = []
        try:
            try:
                success, output, final_response, error = run_job_fn(
                    job, defer_agent_teardown=deferred_agents
                )
            except BaseException:
                for deferred_agent in deferred_agents:
                    teardown_agent_fn(deferred_agent, job["id"])
                raise
        finally:
            reset_secret_scope(scope_token)

        delivery_error = None
        try:
            output_file = save_job_output_fn(job["id"], output)
            if verbose:
                log.info("Output saved to: %s", output_file)

            deliver_content = final_response if success else summarize_failure_fn(job, error)
            should_deliver = bool(deliver_content.strip())
            if should_deliver and success and is_silence_response_fn(deliver_content):
                log.info(
                    "Job '%s': agent returned %s — skipping delivery",
                    job["id"],
                    silent_marker,
                )
                should_deliver = False

            if should_deliver:
                try:
                    delivery_error = deliver_result_fn(job, deliver_content, adapters=adapters, loop=loop)
                except Exception as de:
                    delivery_error = str(de)
                    log.error("Delivery failed for job %s: %s", job["id"], de)
        finally:
            for deferred_agent in deferred_agents:
                teardown_agent_fn(deferred_agent, job["id"])

        if success and not final_response.strip():
            success = False
            error = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"

        mark_job_run_fn(job["id"], success, error, delivery_error=delivery_error)
        return True

    except Exception as exc:
        log.error("Error processing job %s: %s", job["id"], exc)
        mark_job_run_fn(job["id"], False, str(exc))
        return False
