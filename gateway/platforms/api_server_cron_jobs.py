"""Cron job edit door of the API server: ``PATCH /api/jobs/{job_id}`` and the shared job response.

Split out of ``gateway/platforms/api_server.py``. Store callables and helpers are read from the
facade at call time, so ``patch("gateway.platforms.api_server._cron_update")`` keeps intercepting.
"""
import logging

from aiohttp import web

logger = logging.getLogger("gateway.platforms.api_server")


def job_response(adapter, fn, job_id: str, *, notify: bool) -> web.Response:
    """Run ``fn(job_id)``: 404 when falsy, else ``{"job": ...}``; exceptions -> 500."""
    from gateway.platforms import api_server as facade
    try:
        job = fn(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        if notify:
            facade._notify_cron_provider_jobs_changed()
        return web.json_response({"job": job})
    except Exception as e:  # the HTTP boundary: a redacted 500, the traceback stays in the log
        logger.warning("Cron jobs API store call failed for job %s", job_id, exc_info=True)
        return adapter._cron_error_response(e)


async def handle_update_job(adapter, request: web.Request) -> web.Response:
    """PATCH /api/jobs/{job_id} — update a cron job."""
    from gateway.platforms import api_server as facade
    job_id, err = adapter._cron_request_guard(request, need_job_id=True)
    if err:
        return err
    try:
        body = await request.json()
        # Whitelist allowed fields to prevent arbitrary key injection
        sanitized = {k: v for k, v in body.items() if k in adapter._UPDATE_ALLOWED_FIELDS}
        if not sanitized:
            return web.json_response({"error": "No valid fields to update"}, status=400)
        if "name" in sanitized and len(sanitized["name"]) > adapter._MAX_NAME_LENGTH:
            return web.json_response({"error": f"Name must be ≤ {adapter._MAX_NAME_LENGTH} characters"}, status=400)
        if "prompt" in sanitized:
            prompt_err = adapter._validate_cron_prompt(sanitized["prompt"])
            if prompt_err:
                return prompt_err
    except Exception as e:  # health: allow BLE001 -- moved verbatim; the next commit narrows it
        return adapter._cron_error_response(e)
    return job_response(adapter, lambda jid: facade._cron_update(jid, sanitized), job_id, notify=True)
