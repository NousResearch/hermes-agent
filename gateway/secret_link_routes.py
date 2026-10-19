"""aiohttp routes for one-time secret links, mounted on the gateway's existing listener
(api_server / webhook app) by ``mount_secret_link_routes`` — no port of their own.

Every response under ``/secret-link/`` (including 404, 405, 429 and 500) carries the no-store /
no-referrer / CSP headers.
Unknown, malformed, used and expired tokens share one 404 body. The secret value is read into one
local of ``_submit`` and passed only to the writer; error pages and logs are fixed strings.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from urllib.parse import parse_qsl

from aiohttp import web

from gateway import secret_link_store
from gateway.secret_link import (
    PATH_PREFIX, SubmitForm, after_use, find_conflicts, key_error, origin_of, public_base_url, redact_token_paths,
    resolve_profiles, restart_targets, spawn_followup, value_error, write_profiles)
from gateway.secret_link_page import (
    BAD_REQUEST_PAGE, ERROR_PAGE, FORBIDDEN_PAGE, NOT_FOUND_PAGE, TOO_MANY_PAGE, render_form, render_message, render_result)

logger = logging.getLogger(__name__)

ROUTE_NAMES = frozenset({"secret_link_get", "secret_link_post"})
MAX_BODY_BYTES = 16 * 1024
_FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"
_ROUTE = PATH_PREFIX + "{token:.*}"

SECRET_LINK_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "Expires": "0",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": ("default-src 'none'; style-src 'self' 'unsafe-inline'; form-action 'self'; "
                                "frame-ancestors 'none'; base-uri 'none'"),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=31536000",
}

class _TokenPathFilter(logging.Filter):
    """Rewrites ``/secret-link/<raw token>`` in aiohttp access-log records (request line and the
    Referer extra alike) to ``/secret-link/<token>``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_token_paths(record.getMessage())
        record.args = None
        for name, value in list(vars(record).items()):
            if isinstance(value, str) and PATH_PREFIX in value:
                setattr(record, name, redact_token_paths(value))
            elif isinstance(value, dict):
                setattr(record, name, {k: redact_token_paths(v) if isinstance(v, str) else v
                                       for k, v in value.items()})
        return True


def _install_access_log_filter() -> None:
    access = logging.getLogger("aiohttp.access")
    if not any(isinstance(f, _TokenPathFilter) for f in access.filters):
        access.addFilter(_TokenPathFilter())


def _page(status: int, html: str) -> web.Response:
    return web.Response(status=status, text=html, content_type="text/html", charset="utf-8",
                        headers=SECRET_LINK_HEADERS)


@web.middleware
async def _secret_link_middleware(request: web.Request, handler) -> web.StreamResponse:
    """Outermost middleware of the host listener, for ``/secret-link/`` paths only: the per-client
    rate limit, one fixed 500 page, and the security headers on EVERY response — also the ones the
    router (405) or the listener's own middlewares (CORS 403, body cap 413) answer with."""
    if not request.path.startswith(PATH_PREFIX):
        return await handler(request)
    try:
        if not secret_link_store.STORE.allow_client(request.remote or ""):
            return _page(429, TOO_MANY_PAGE)
        response = await handler(request)
    except web.HTTPException as exc:
        exc.headers.update(SECRET_LINK_HEADERS)
        raise
    except Exception as exc:  # health: allow BLE001 -- SR-35: fixed page, log the type only
        logger.error("Secret link request failed (%s)", type(exc).__name__)
        return _page(500, ERROR_PAGE)
    if not response.prepared:  # a submit result is already on the wire, with the headers from _page
        response.headers.update(SECRET_LINK_HEADERS)
    return response


def is_secret_link_route(request: web.Request) -> bool:
    """True for a request the secret-link routes matched: it is same-origin and checks its own
    Origin and body cap, so the host listener's CORS and body-size middlewares step aside."""
    return request.match_info.route.name in ROUTE_NAMES


async def _profile_names() -> list[str]:
    from hermes_cli.profiles import list_profile_names
    return await asyncio.to_thread(list_profile_names)


async def _handle_get(request: web.Request) -> web.StreamResponse:
    store = secret_link_store.STORE
    record = store.lookup(request.match_info["token"])
    if record is None:
        return _page(404, NOT_FOUND_PAGE)
    names = await _profile_names()
    return _page(200, render_form(nonce=store.rotate_nonce(record), key=record.suggested_key, profiles=names))


def _origin_matches(request: web.Request, base: str) -> bool:
    expected = origin_of(base)
    origin = request.headers.get("Origin")
    if origin is None:
        referer = request.headers.get("Referer")
        origin = origin_of(referer) if referer else None
    return bool(expected) and origin == expected


def _parse_form(body: bytes) -> tuple[Optional[SubmitForm], str]:
    """``(form, secret value)``; ``(None, "")`` when the body is not a valid form."""
    try:
        pairs = parse_qsl(body.decode("ascii"), keep_blank_values=True, errors="strict", max_num_fields=64)
    except ValueError:
        return None, ""
    fields: dict[str, list[str]] = {}
    for name, item in pairs:
        fields.setdefault(name, []).append(item)
    del pairs
    secrets_sent = fields.pop("secret", [])
    if len(secrets_sent) != 1 or len(fields.get("nonce", [])) != 1 or len(fields.get("key", [])) != 1:
        return None, ""
    form = SubmitForm(
        nonce=fields["nonce"][0], key=fields["key"][0].strip(), profiles=tuple(fields.get("profile", [])),
        all_profiles=bool(fields.get("all_profiles")), replace=bool(fields.get("replace")))
    return form, secrets_sent[0]


async def _read_body(request: web.Request) -> Optional[bytes]:
    if request.content_type != _FORM_CONTENT_TYPE:
        return None
    if (request.content_length or 0) > MAX_BODY_BYTES:
        return None
    body = bytearray()
    async for chunk in request.content.iter_any():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            return None
    return bytes(body)


async def _handle_post(request: web.Request) -> web.StreamResponse:
    body = await _read_body(request)
    if body is None:
        return _page(400, BAD_REQUEST_PAGE)
    base = public_base_url()
    if not base or not _origin_matches(request, base):
        return _page(403, FORBIDDEN_PAGE)
    store = secret_link_store.STORE
    record = store.lookup(request.match_info["token"])
    if record is None or not store.claim(record):
        return _page(404, NOT_FOUND_PAGE)
    form, value = _parse_form(body)
    del body
    try:
        return await _submit(request, record, form, value)
    finally:
        del value
        store.release(record)  # no-op once the use is final


async def _refuse(record, status: int, form: Optional[SubmitForm], message: str, *,
                  conflicts: list[str] = ()) -> web.Response:
    """A failed submit: count it, then show the form again (or the 404 once the cap expired it)."""
    store = secret_link_store.STORE
    store.release(record)
    if store.record_failure(record):
        return _page(status, render_message("Link closed", f"{message} This link is now closed after too many "
                                            "failed tries. Ask Hermes for a new link in Telegram.".lstrip()))
    if form is None or status == 403:
        return _page(status, FORBIDDEN_PAGE if status == 403 else BAD_REQUEST_PAGE)
    names = await _profile_names()
    html = render_form(nonce=store.rotate_nonce(record), key=form.key, profiles=names, selected=form.profiles,
                       all_selected=form.all_profiles, message=message, conflicts=conflicts)
    return _page(status, html)


async def _submit(request: web.Request, record, form: Optional[SubmitForm], value: str) -> web.StreamResponse:
    store = secret_link_store.STORE
    if form is None:
        return await _refuse(record, 400, None, "")
    if not store.nonce_matches(record, form.nonce):
        return await _refuse(record, 403, form, "")
    profiles, error = await asyncio.to_thread(resolve_profiles, form)
    error = error or key_error(form.key, profiles) or value_error(value)
    if error:
        return await _refuse(record, 400, form, error)
    if not form.replace:
        conflicts = await asyncio.to_thread(find_conflicts, form.key, profiles)
        if conflicts:
            return await _refuse(record, 409, form, "", conflicts=conflicts)
    report = await asyncio.to_thread(write_profiles, form.key, value, profiles)
    if not report.written:
        return await _refuse(record, 500, form, f"Write failed for profile {report.failed[0]}.")
    store.finish(record, not report.failed)
    targets = await asyncio.to_thread(restart_targets, report.written, record.owner_home, record.trigger)
    response = _page(500 if report.failed else 200, render_result(
        key=form.key, written=report.written, failed=report.failed, not_tried=report.not_tried, restarting=targets))
    # Response first, then the Telegram confirmation, then any restart (which may stop this process).
    await response.prepare(request)
    await response.write_eof()
    spawn_followup(after_use(record, form.key, report, targets))
    return response


def mount_secret_link_routes(app: web.Application) -> None:
    """Add the secret-link routes to an existing listener's app (before it is frozen)."""
    _install_access_log_filter()
    if _secret_link_middleware not in app.middlewares:
        app.middlewares.insert(0, _secret_link_middleware)
    app.router.add_get(_ROUTE, _handle_get, name="secret_link_get", allow_head=False)
    app.router.add_post(_ROUTE, _handle_post, name="secret_link_post")
