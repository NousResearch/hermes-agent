"""Cron → workflow dispatch bridge.

Per the routine_workflow_arch (meta-202606-008): cron jobs created via
/api/jobs may carry ``metadata.workflow_ids: [...]``. When present, the
scheduler must delegate execution to the workflow engine instead of
spawning a plain LLM prompt subprocess.

This module owns the delegation step. It is intentionally minimal: the
workflow engine may be hosted externally; Hermes reaches it over the
loopback HTTP control channel.

Resolution order for the dispatch endpoint:

1. ``HERMES_WORKFLOW_DISPATCH_URL`` env var (full URL).
2. ``None`` — caller falls back to the legacy prompt-spawn path so
   pre-routine-workflow installs keep working unchanged.

Companion resolution (e.g. via companion base URL env) is patched in by
the adapter layer at startup so core stays free of companion specifics.

The dispatch payload is intentionally small (workflow_id + job_id +
optional trigger context). The engine side is responsible for
resolving the workflow definition, building the run, and persisting
state — this module never inspects workflow contents.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 30  # seconds — workflow start should return quickly; the run executes asynchronously.


def resolve_dispatch_url() -> Optional[str]:
    """Resolve the workflow-engine dispatch URL from env, or ``None``."""
    explicit = (os.getenv("HERMES_WORKFLOW_DISPATCH_URL") or "").strip()
    if explicit:
        return explicit
    # Companion support is injected via adapter patch
    # into this module's resolve_dispatch_url at register time.
    return None


def extract_workflow_ids(job: Dict[str, Any]) -> List[str]:
    """Return cron job workflow_ids as a list of clean strings, or ``[]``.

    Reads the canonical ``job["workflow_ids"]`` slot first
    (``cron.jobs._normalize_workflow_ids`` writes here for every job
    created via ``create_job`` / ``/api/jobs``). Falls back to
    ``job["metadata"]["workflow_ids"]`` for any forward-compat job
    shape that may stash routine associations under metadata.

    Tolerates non-list scalars (single string → one-element list) and
    drops empty / whitespace-only ids so an accidental ``[""]`` doesn't
    trip the non-empty check.
    """

    def _clean(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        cleaned: List[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text:
                cleaned.append(text)
        return cleaned

    top_level = _clean(job.get("workflow_ids"))
    if top_level:
        return top_level
    metadata = job.get("metadata")
    if isinstance(metadata, dict):
        return _clean(metadata.get("workflow_ids"))
    return []


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Tuple[bool, str]:
    """POST ``payload`` as JSON, return (success, body_or_error)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback control plane
            response_body = resp.read().decode("utf-8", errors="replace")
            return True, response_body
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return False, f"HTTP {exc.code}: {detail or exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"transport error: {exc}"


def dispatch_workflow_runs(
    job: Dict[str, Any],
    workflow_ids: List[str],
    *,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Tuple[bool, str, str]:
    """Start a workflow run for each id attached to a cron job.

    Returns ``(success, summary_doc, error_message_or_empty)``.

    ``success`` is True only when every workflow_id started cleanly.
    Partial successes are still reported as failures so the operator
    notices — partial-fire silently is exactly the failure mode
    routine_workflow_arch aims to prevent.

    Side effects: emits one POST per workflow_id. The workflow engine
    side is responsible for the actual execution; this
    function does NOT wait for the run to finish.
    """
    dispatch_url = resolve_dispatch_url()
    job_id = job.get("id") or "?"
    job_name = job.get("name") or job_id

    if not dispatch_url:
        err = (
            "no workflow dispatch URL configured — set "
            "HERMES_WORKFLOW_DISPATCH_URL (companion support via adapter)"
        )
        logger.warning("Cron job '%s': %s", job_name, err)
        return False, "", err

    results: List[str] = []
    errors: List[str] = []
    for workflow_id in workflow_ids:
        payload = {
            "workflow_id": workflow_id,
            "trigger": {
                "source": "hermes_cron",
                "job_id": str(job_id),
                "job_name": str(job_name),
            },
        }
        ok, body = _post_json(dispatch_url, payload, timeout=timeout)
        if ok:
            logger.info(
                "Cron job '%s': dispatched workflow_id=%s via %s",
                job_name, workflow_id, dispatch_url,
            )
            results.append(f"- {workflow_id}: started ({body[:120]})")
        else:
            logger.warning(
                "Cron job '%s': workflow_id=%s dispatch failed — %s",
                job_name, workflow_id, body,
            )
            errors.append(f"- {workflow_id}: {body}")

    doc_lines = [
        f"# Cron Job: {job_name}",
        "",
        f"**Job ID:** {job_id}",
        f"**Mode:** workflow-dispatch",
        f"**Dispatch URL:** {dispatch_url}",
        f"**Workflow IDs:** {', '.join(workflow_ids)}",
        "",
        "## Results",
        *results,
    ]
    if errors:
        doc_lines.extend(["", "## Errors", *errors])
    doc = "\n".join(doc_lines) + "\n"

    if errors:
        return False, doc, "; ".join(errors)
    return True, doc, ""
