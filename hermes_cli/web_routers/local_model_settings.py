"""Native model-settings routes: per-artifact runtime settings for staged local models.

``GET/PUT /api/local-models/settings/{model_id}`` and ``POST .../reset``. Every
read/write goes through the :mod:`native_model_settings` domain adapter — the
identity gate (``expected_fingerprint``) runs BEFORE the CAS
(``expected_revision``), so a stale view can never adopt settings onto a
replaced artifact. Bodies are strict (unknown fields rejected) and queries
carry only the optional ``profile``; error details are structured, safe codes
— never absolute home/models paths.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, TypeVar

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from hermes_cli.local_runtime import catalog, native_model_settings as nms
from hermes_cli.local_runtime.model_settings import (
    InvalidSettingValueError,
    ModelSettingsError,
    RevisionConflictError,
    UnknownSettingError,
)
from hermes_cli.web_routers._common import log, scoped_to_thread

router = APIRouter()

# Only the optional profile selector may ride the query string; anything else
# (e.g. a caller-supplied ``root``) is rejected before the domain is touched.
_ALLOWED_QUERY_PARAMS = frozenset({"profile"})

_T = TypeVar("_T")


class SettingsPutBody(BaseModel):
    """Full-replacement save: CAS on the caller's revision + artifact identity."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)
    expected_fingerprint: str
    overrides: dict[str, Any]


class SettingsResetBody(BaseModel):
    """Drop overrides — the named schema keys, or all of them when null."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)
    expected_fingerprint: str
    keys: Optional[list[str]] = None


def _reject_unknown_query(request: Request) -> None:
    extra = sorted(set(request.query_params) - _ALLOWED_QUERY_PARAMS)
    if extra:
        raise HTTPException(status_code=422, detail={
            "error": "unexpected_query_params", "params": extra})


def _store_root() -> Path:
    """The effective profile home for this request — call INSIDE the profile
    scope, where ``get_hermes_home()`` is the named profile's root (or the
    current one when no profile was given)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _settings_http_error(exc: ModelSettingsError) -> HTTPException:
    """Safe structured detail for a domain refusal: codes only, no paths.

    The adapter raises lazily (identity first), so every store/artifact
    failure funnels through here; unexpected domain errors are logged and
    shipped as an opaque 500.
    """
    if isinstance(exc, nms.UnknownModelError):
        return HTTPException(status_code=404, detail={"error": "unknown_model"})
    if isinstance(exc, nms.ModelChangedError):
        return HTTPException(status_code=409, detail={
            "error": "model_changed", "current_fingerprint": exc.current_fingerprint})
    if isinstance(exc, RevisionConflictError):
        return HTTPException(status_code=409, detail={
            "error": "revision_conflict",
            "expected_revision": exc.expected_revision,
            "current_revision": exc.current_revision})
    if isinstance(exc, nms.InvalidModelIdError):
        return HTTPException(status_code=400, detail={"error": "invalid_model_id"})
    if isinstance(exc, UnknownSettingError):
        return HTTPException(status_code=400, detail={"error": "unknown_setting"})
    if isinstance(exc, InvalidSettingValueError):
        return HTTPException(status_code=400, detail={"error": "invalid_setting_value"})
    if isinstance(exc, nms.MissingBindingError):
        return HTTPException(status_code=400, detail={"error": "missing_fingerprint_binding"})
    log.exception("model-settings request failed")
    return HTTPException(status_code=500, detail={"error": "settings_store_unavailable"})


async def _run_scoped(profile: Optional[str], fn: Callable[[], _T]) -> _T:
    """``fn()`` on a worker thread inside the request's profile scope; domain
    refusals become their mapped HTTP status."""
    try:
        return await scoped_to_thread(profile, fn)
    except HTTPException:
        raise
    except ModelSettingsError as exc:
        raise _settings_http_error(exc) from None


@router.get("/api/local-models/settings/{model_id}")
async def get_local_model_settings(model_id: str, request: Request,
                                   profile: Optional[str] = None):
    """Full parent-UI contract for one staged model (identity, revision,
    overrides, effective+sources, typed fields, runtime + loading facts)."""
    _reject_unknown_query(request)
    # TTL-gated background fetch via the module attribute, so test seams on
    # catalog.refresh_catalog_soon keep winning: fresher catalog hints next call.
    catalog.refresh_catalog_soon()
    return await _run_scoped(profile, lambda: nms.read_model_settings(model_id, _store_root()))


@router.put("/api/local-models/settings/{model_id}")
async def put_local_model_settings(model_id: str, body: SettingsPutBody, request: Request,
                                   profile: Optional[str] = None):
    """Replace the saved overrides under the identity gate + revision CAS;
    the response is the same full contract as a fresh GET."""
    _reject_unknown_query(request)
    return await _run_scoped(profile, lambda: nms.save_model_settings(
        model_id, _store_root(), body.overrides, body.expected_revision,
        body.expected_fingerprint))


@router.post("/api/local-models/settings/{model_id}/reset")
async def reset_local_model_settings(model_id: str, body: SettingsResetBody, request: Request,
                                     profile: Optional[str] = None):
    """Drop the named schema keys' overrides (all of them when ``keys`` is
    null), under the same identity gate + revision CAS."""
    _reject_unknown_query(request)
    return await _run_scoped(profile, lambda: nms.reset_model_settings(
        model_id, _store_root(), body.expected_revision, body.expected_fingerprint,
        body.keys))
