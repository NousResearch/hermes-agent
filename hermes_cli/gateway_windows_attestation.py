"""Neutral post-update diagnostics for Windows gateways without a trustworthy profile."""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_UNMAPPED_START_ATTESTATION_RELATIVE = ("state", "gateway.unmapped-start-attestation.json")


def _write_unmapped_start_attestation(pids: list[int], via: str) -> None:
    """Persist a diagnostic without assigning an unmapped gateway a profile."""
    from hermes_cli import gateway_windows as gw

    try:
        path = gw._hermes_home().joinpath(*_UNMAPPED_START_ATTESTATION_RELATIVE)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pids": [int(p) for p in pids], "via": via,
            "ts": datetime.now(UTC).isoformat(), "generation": uuid.uuid4().hex,
            "identity": "unmapped",
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to write unmapped gateway start attestation", exc_info=True)


def _unmapped_start_attestation_path() -> Path:
    """A neutral marker for a process that could not be assigned a profile/home."""
    from hermes_cli import gateway_windows as gw

    return gw._hermes_home().joinpath(*_UNMAPPED_START_ATTESTATION_RELATIVE)


def _read_unmapped_start_attestation() -> object | None:
    try:
        return json.loads(_unmapped_start_attestation_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _clear_unmapped_start_attestation() -> None:
    try:
        _unmapped_start_attestation_path().unlink(missing_ok=True)
    except OSError:
        pass


def check_unmapped_start_attestation(current_pids: list[int] | None = None) -> str | None:
    """Surface a one-shot diagnostic for a previously verified unmapped gateway.

    This intentionally does not use a profile's ledger or task hint: an
    unmapped process has no trustworthy home and must not be represented as
    the active/default profile.  It is diagnostic-only, unlike profile
    attestations which can participate in a cold-start plan.
    """
    data = _read_unmapped_start_attestation()
    if not isinstance(data, dict) or data.get("identity") != "unmapped":
        if data is not None:
            _clear_unmapped_start_attestation()
        return None
    from hermes_cli import gateway_windows as gw

    attested = gw._attested_pids_from(data)
    if not attested or not gw._attestation_within_horizon(data):
        _clear_unmapped_start_attestation()
        return None
    if current_pids is None:
        try:
            from hermes_cli.gateway import find_gateway_pids

            current_pids = list(find_gateway_pids(all_profiles=True))
        except Exception:
            logger.debug("Could not probe unmapped gateway PIDs", exc_info=True)
            return None
    _clear_unmapped_start_attestation()
    if set(attested) & set(current_pids):
        return None
    via = data.get("via") or "post-update recovery"
    ts = data.get("ts") or "unknown time"
    return (
        f"⚠ A previously verified unmapped gateway ({via}, {ts}) is no longer running "
        f"(PID {', '.join(map(str, attested))}). Its profile could not be determined."
    )
