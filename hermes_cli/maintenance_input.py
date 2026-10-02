"""CLI-local maintenance carry: original authored text plus an optional sealed source."""

from __future__ import annotations

from typing import Any


class AuthoredInput(str):
    """A model-facing CLI payload that remembers the exact text its author typed.

    ``str`` semantics keep the existing queues, previews, and queue editors compatible;
    the two attributes let the TUI boundary mint one local source from ``raw`` while the
    chat layer records expanded/reference text only as ``extraction``.
    """

    def __new__(cls, text: str, *, raw: str | None = None, source: Any = None):
        value = str(text or "")
        obj = super().__new__(cls, value)
        obj.raw = str(raw) if isinstance(raw, str) and raw.strip() else value
        obj.maintenance_source = source
        return obj


def authored_raw(value: Any) -> str | None:
    raw = getattr(value, "raw", None)
    return raw if isinstance(raw, str) and raw.strip() else None


def authored_source(value: Any) -> Any:
    return getattr(value, "maintenance_source", None)


def _session_id(cli) -> str | None:
    value = getattr(cli, "session_id", None) or getattr(getattr(cli, "agent", None), "session_id", None)
    return value if isinstance(value, str) and value else None


def commit_control(cli, kind: str, raw: str, *, extraction: str | None = None, binding=None):
    """Mint and commit one local control source at the real CLI caller boundary."""
    session_id = _session_id(cli)
    if not session_id or not isinstance(raw, str) or not raw.strip():
        return None
    db = getattr(cli, "_session_db", None)
    if db is None:
        return None
    try:
        if db.get_session(session_id) is None:
            db.ensure_session(session_id, "cli", session_key=session_id)
    except Exception:
        return None
    from hermes_constants import get_hermes_home
    from hermes_maintenance_source import commit_local_control_source, mint_local_control
    home = get_hermes_home()
    try:
        receipt = mint_local_control(
            raw, home=home, session_id=session_id, kind=kind,
            extraction=extraction if extraction is not None else raw, binding=binding,
        )
        return receipt if commit_local_control_source(home, session_id, receipt) else None
    except (ValueError, OSError):
        return None


def revise_control(cli, base, kind: str, raw: str, *, extraction: str | None = None):
    """Commit an edited queued control as a new revision of the same occurrence."""
    session_id = _session_id(cli)
    if (not session_id or not isinstance(raw, str) or not raw.strip()
            or getattr(base, "intended_session_id", None) != session_id):
        return None
    from dataclasses import replace
    from hermes_constants import get_hermes_home
    from hermes_maintenance_source import (
        CLI_CONTROL_CONTRACT, commit_local_control_source, revise_original_input)
    if getattr(base, "contract", None) != CLI_CONTROL_CONTRACT:
        return None
    home = get_hermes_home()
    extracted = extraction if extraction is not None else raw
    try:
        revised = revise_original_input(base, raw, extraction=extracted)
        revised = replace(
            revised, contract=CLI_CONTROL_CONTRACT, input_kind=kind, control_kind=kind,
            control_binding_json=None, display=extracted,
        )
        return revised if commit_local_control_source(home, session_id, revised) else None
    except (ValueError, OSError):
        return None


def remember_steer(cli, text: str, source) -> None:
    """Track a CLI steer occurrence until the turn either absorbs it or requeues it."""
    if not isinstance(text, str) or not text.strip() or source is None:
        return
    pending = getattr(cli, "_maintenance_steer_inputs", None)
    if pending is None:
        pending = []
        setattr(cli, "_maintenance_steer_inputs", pending)
    pending.append((text.strip(), source))


def take_steer_inputs(cli, leftover: str):
    """Resolve a leftover agent steer back to one sealed CLI occurrence per input."""
    pending = list(getattr(cli, "_maintenance_steer_inputs", None) or [])
    setattr(cli, "_maintenance_steer_inputs", [])
    if not pending or not isinstance(leftover, str):
        return None if not pending else []
    texts = [text for text, _ in pending]
    if leftover == "\n".join(texts):
        return [AuthoredInput(text, raw=text, source=source) for text, source in pending]
    for text, source in pending:
        if leftover == text:
            return [AuthoredInput(text, raw=text, source=source)]
    for start in range(1, len(pending)):
        if leftover == "\n".join(texts[start:]):
            return [AuthoredInput(text, raw=text, source=source)
                    for text, source in pending[start:]]
    return []


def latest_retry_source_info(cli):
    """Return ``(receipt, blocked)`` for the latest projection.

    No projection is a legacy CLI turn and remains retryable. A projection owned by any
    non-local authority (for example a cloned platform transcript) fails closed.
    """
    session_id = _session_id(cli)
    if not session_id:
        return None, True
    from hermes_constants import get_hermes_home
    from hermes_maintenance_source import (
        admit_reused_local_source, read_committed_source, rehydrate_source_receipt,
        source_for_latest_user_message,
    )
    home = get_hermes_home()
    source_id = source_for_latest_user_message(home, session_id)
    if source_id is None:
        return None, False
    source = read_committed_source(home, source_id)
    if source is None or source.get("authority") != "cli-local":
        return None, True
    receipt = rehydrate_source_receipt(home, source_id)
    receipt = admit_reused_local_source(receipt, home=home, session_id=session_id)
    return receipt, receipt is None


def latest_retry_source(cli):
    receipt, blocked = latest_retry_source_info(cli)
    return None if blocked else receipt
