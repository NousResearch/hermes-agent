"""What a status report says: contract v1, with ``status`` cut down to the fields a receiver grades."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, Mapping

from gateway.status import _get_process_start_time, read_runtime_status
from hermes_cli.auth import get_nous_session_validity
from hermes_cli.web_routers.status_public import redact_status_for_unverified

logger = logging.getLogger(__name__)

REPORT_VERSION = 1
ReportKind = Literal["beat", "goodbye"]

# ``starting``, ``running``, ``draining``: a lifecycle code, never prose.
_STATE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
# ``<platform>``, or ``<profile>:<platform>`` for a platform this gateway serves for another profile.
_PLATFORM_KEY_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}(?::[a-z0-9][a-z0-9_-]{0,63})?")
_PLATFORM_FIELDS = ("state", "error_code")
_SESSION_VALIDITY = frozenset({"valid", "terminal", "unknown"})


def read_status_source() -> dict[str, Any]:
    """This gateway's own state from what ``/api/status`` reads for it: the runtime record the
    gateway writes (``gateway_state.json``) and the local auth store, both under the bound home.
    Blocking: call it off the event loop."""
    runtime = read_runtime_status() or {}
    source: dict[str, Any] = {
        "gateway_running": True,  # the gateway process itself is asking
        "gateway_state": runtime.get("gateway_state"),
        "gateway_exit_reason": runtime.get("exit_reason"),
        "gateway_platforms": _own_platforms(runtime.get("platforms")),
    }
    try:
        source["nous_session_valid"] = get_nous_session_validity()
    except Exception:  # the field is optional, and an unreadable auth store must not stop the report
        logger.debug("status report: session validity unavailable", exc_info=True)
    return source


def _own_platforms(platforms: Any) -> dict[str, Any]:
    """The entries this gateway process stamped. A restart keeps the previous process's entries in
    the record, so a platform since removed would otherwise be reported as failing for good."""
    if not isinstance(platforms, dict):
        return {}
    pid = os.getpid()
    started = _get_process_start_time(pid)
    return {key: value for key, value in platforms.items()
            if isinstance(value, dict) and value.get("writer_pid") == pid
            and value.get("writer_start_time") == started}


def allowlisted_status(source: Mapping[str, Any]) -> dict[str, Any]:
    """The ``status`` of a report: only the keys below, each in the form an unverified
    ``/api/status`` caller gets. Whatever else *source* carries (install id, profile names, error
    text, counts, paths) is not copied, and a value of the wrong shape is dropped, never passed on."""
    state, platforms = source.get("gateway_state"), source.get("gateway_platforms")
    public: dict[str, Any] = {
        "gateway_state": state if isinstance(state, str) and _STATE_RE.fullmatch(state) else None,
        "gateway_exit_reason": source.get("gateway_exit_reason"),
        "gateway_platforms": {
            key: value for key, value in platforms.items()
            if isinstance(key, str) and _PLATFORM_KEY_RE.fullmatch(key) and isinstance(value, dict)
        } if isinstance(platforms, dict) else {},
    }
    redact_status_for_unverified(public)  # platform entries and the exit reason become bounded codes
    status: dict[str, Any] = {
        "gateway_running": source.get("gateway_running") is True,
        "gateway_state": public["gateway_state"],
        "gateway_exit_reason": public["gateway_exit_reason"],
        "gateway_platforms": {
            key: {name: entry[name] for name in _PLATFORM_FIELDS if name in entry}
            for key, entry in public["gateway_platforms"].items()
        },
    }
    if source.get("nous_session_valid") in _SESSION_VALIDITY:
        status["nous_session_valid"] = source["nous_session_valid"]
    return status


def build_report(kind: ReportKind, *, seq: int, boot_id: str, source: Mapping[str, Any]) -> bytes:
    """The JSON body of one report."""
    return json.dumps(
        {"v": REPORT_VERSION, "kind": kind, "seq": seq, "boot_id": boot_id, "status": allowlisted_status(source)},
        separators=(",", ":")).encode("utf-8")
