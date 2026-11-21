"""Cron job doors of the API server: ``POST /api/jobs``, ``PATCH /api/jobs/{job_id}`` and the shared job response.

Split out of ``gateway/platforms/api_server.py``. Store callables and helpers are read from the
facade at call time, so ``patch("gateway.platforms.api_server._cron_update")`` keeps intercepting.
"""
import logging

from aiohttp import web

logger = logging.getLogger("gateway.platforms.api_server")


def job_response(adapter, fn, job_id: str, *, notify: bool, invalid_input=()) -> web.Response:
    """Run ``fn(job_id)``: 404 when falsy, else ``{"job": ...}``; an ``invalid_input`` exception
    -> 400 (the client's value was rejected), any other -> 500."""
    from gateway.platforms import api_server as facade
    try:
        job = fn(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        if notify:
            facade._notify_cron_provider_jobs_changed()
        return web.json_response({"job": job})
    except invalid_input as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:  # the HTTP boundary: a redacted 500, the traceback stays in the log
        logger.warning("Cron jobs API store call failed for job %s", job_id, exc_info=True)
        return adapter._cron_error_response(e)


def _invalid_update(adapter, sanitized: dict):
    """The 400 for a PATCH value the handler can judge itself, else None."""
    if not sanitized:
        return web.json_response({"error": "No valid fields to update"}, status=400)
    for key in ("name", "prompt", "schedule"):
        if key in sanitized and not isinstance(sanitized[key], str):
            return web.json_response({"error": f"{key.capitalize()} must be a string"}, status=400)
    if "name" in sanitized and len(sanitized["name"]) > adapter._MAX_NAME_LENGTH:
        return web.json_response({"error": f"Name must be ≤ {adapter._MAX_NAME_LENGTH} characters"}, status=400)
    return adapter._validate_cron_prompt(sanitized["prompt"]) if "prompt" in sanitized else None


async def handle_create_job(adapter, request: web.Request) -> web.Response:
    """POST /api/jobs: create a cron job; a body that is not a JSON object is a 400, never a 500."""
    from gateway.platforms import api_server as facade
    _, err = adapter._cron_request_guard(request)
    if err:
        return err
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        name = (body.get("name") or "").strip()
        schedule = (body.get("schedule") or "").strip()
        prompt = body.get("prompt", "")
        skills = body.get("skills")
        repeat = body.get("repeat")
        if not name:
            return web.json_response({"error": "Name is required"}, status=400)
        if len(name) > adapter._MAX_NAME_LENGTH:
            return web.json_response({"error": f"Name must be ≤ {adapter._MAX_NAME_LENGTH} characters"}, status=400)
        if not schedule:
            return web.json_response({"error": "Schedule is required"}, status=400)
        prompt_err = adapter._validate_cron_prompt(prompt)
        if prompt_err:
            return prompt_err
        if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
            return web.json_response({"error": "Repeat must be a positive integer"}, status=400)
        kwargs = {
            "prompt": prompt, "schedule": schedule, "name": name,
            "deliver": body.get("deliver", "local"),
            "origin": adapter._cron_origin_from_request(request)}
        for key in ("paused", "paused_reason", "timezone"):
            if key in body:
                kwargs[key] = body[key]
        if skills:
            kwargs["skills"] = skills
        if repeat is not None:
            kwargs["repeat"] = repeat
        return web.json_response({"job": facade._cron_create(**kwargs)})
    except facade._CronSchedulerRegistrationError as e:
        return web.json_response(e.to_dict(), status=424)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:  # health: allow BLE001 -- the API boundary: any store failure is a redacted 500
        return adapter._cron_error_response(e)


async def handle_update_job(adapter, request: web.Request) -> web.Response:
    """PATCH /api/jobs/{job_id}: a rejected value is a 400 ``{"error": ...}`` as on POST, never a 500."""
    from gateway.platforms import api_server as facade
    job_id, err = adapter._cron_request_guard(request, need_job_id=True)
    if err:
        return err
    try:
        body = await request.json()
    except ValueError as e:  # malformed JSON, as on POST
        return web.json_response({"error": str(e)}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)
    # Whitelist allowed fields to prevent arbitrary key injection
    sanitized = {k: v for k, v in body.items() if k in adapter._UPDATE_ALLOWED_FIELDS}
    invalid = _invalid_update(adapter, sanitized)
    if invalid is not None:
        return invalid
    # The store's ValueError is a rejected value (zone, schedule, repeat): 400 like POST.
    return job_response(adapter, lambda jid: facade._cron_update(jid, sanitized), job_id, notify=True,
                        invalid_input=ValueError)
