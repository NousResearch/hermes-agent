"""External drain-control marker contract (dashboard → gateway).

Task 2.2 of the safe-shutdown plan (decisions.md Q-B, option A): the dashboard
has no way to call into a running gateway — there is no HTTP control channel
into the gateway process (guardrails: "there is NO external control channel
into a running gateway"). Restart/drain is driven only by the gateway reacting
to its own inputs: slash commands, process signals, and file markers it writes
itself (``.restart_notify.json``).

So the begin/cancel-drain dashboard endpoint communicates with the running
gateway the same way: it writes (or removes) a marker file, and a gateway
background watcher reacts to it. This module owns that marker contract so both
sides — the dashboard endpoint (writer) and the gateway watcher (reader) —
share one definition and can never disagree.

Contract (presence-based, mirroring ``.restart_notify.json``):

  * begin-drain  → write ``{HERMES_HOME}/.drain_request.json`` with
    ``{"action": "drain", "requested_at": <iso>, "principal": <str>,
    "epoch": <instantiation-epoch>, "suppress_notification": <bool>}``.
  * cancel-drain → remove the marker.
  * The gateway watcher treats **presence of a marker stamped with the current
    instantiation epoch** as "external drain active": flip
    ``gateway_state -> "draining"`` and stop accepting new turns. Absence (or a
    marker from a *prior* instantiation) means "not draining" (revert to
    ``running`` if we had flipped it).

Why the epoch (NS-570). ``HERMES_HOME`` is a **durable** store — on Hermes
Cloud it is a persistent Fly volume (``/opt/data``). A begin-drain marker
written there *survives a machine restart*. But the disruptive lifecycle
actions a drain protects (auto-update / image migrate / env edit / profile
change) all **restart the machine**, which is exactly the signal that the drain
is over. Without the epoch, a freshly-restarted gateway re-reads the orphaned
marker on boot and parks itself right back in ``draining`` forever (NS-570: an
auto-updated instance refused every turn for ~52 min). Stamping the marker with
an identity of *this* container/VM instantiation, and ignoring a marker whose
epoch doesn't match, makes "a deliberate restart clears the drain" true by
construction — while a marker written during the *current* instantiation (the
live drain) still matches, and an s6 respawn of just the gateway (PID 1 / init
unchanged) still honours an in-flight drain.

Reading the marker never raises: a malformed/half-written file reads as
"present but contentless", which the watcher still treats as drain-active
(fail-safe toward quiescing — a corrupt begin marker must not be ignored). The
epoch check is deliberately **lenient**: it ignores a marker only on a
*definite* epoch mismatch. A marker with no epoch (legacy/corrupt/contentless),
or an environment where the epoch cannot be computed (non-Linux, no ``/proc``),
both degrade to the original presence-only behaviour — never fail-closed.
"""
from __future__ import annotations

import functools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

_log = logging.getLogger(__name__)

_DRAIN_REQUEST_FILENAME = ".drain_request.json"


@functools.lru_cache(maxsize=1)
def current_instantiation_epoch() -> str:
    """Identity of THIS container / VM instantiation.

    Stable for the life of the PID-1 init process — so an s6 respawn of just
    the gateway keeps the same epoch and an in-flight drain is honoured — but
    changes when the machine/container is recreated (a fresh PID 1 → a fresh
    epoch). Composed from two ``/proc`` facts:

      * the kernel **boot id** (``/proc/sys/kernel/random/boot_id``) — changes
        on a VM / microVM reboot (e.g. a Fly Firecracker machine restart);
      * **PID 1's start time** (field 22 of ``/proc/1/stat``) — changes on a
        plain ``docker restart`` (the host kernel, hence boot_id, is unchanged,
        but ``/init`` is a brand-new process).

    Together they discriminate every restart mode that matters:

      | event                          | boot_id | pid1 start | epoch  | marker |
      |--------------------------------|---------|------------|--------|--------|
      | Fly microVM reboot (auto-upd.) | changes | changes    | NEW    | reject |
      | plain ``docker restart``       | same    | changes    | NEW    | reject |
      | s6 respawn of the gateway only | same    | same       | SAME   | honour |
      | host ``hermes gateway restart``| same    | same(init) | SAME   | honour |

    The last row is intentional: a host install has no durable-volume drain
    bug, and honouring a drain across a deliberate process restart is the
    intended reversible behaviour (D4a) — PID 1 there is the long-lived init
    (systemd/launchd), so the epoch is stable.

    Returns ``""`` when neither identity source is readable (non-Linux, no
    ``/proc``). An empty epoch disables the staleness check downstream,
    degrading to the released presence-only behaviour — never fail-closed.
    Memoised: the epoch is constant for the life of the process.
    """
    boot_id = ""
    try:
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        _log.debug("Suppressed exception", exc_info=True)

    pid1_start = ""
    try:
        # /proc/1/stat: "<pid> (<comm>) <state> ... <starttime@field22> ...".
        # comm can contain spaces and parens, so split on the LAST ')' and
        # index into the whitespace-delimited tail. starttime is field 22
        # (1-indexed); after the comm the tail starts at field 3, so it is the
        # tail's index 19.
        stat = Path("/proc/1/stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].split()
        pid1_start = tail[19]
    except (OSError, IndexError):
        _log.debug("Suppressed exception", exc_info=True)

    if not boot_id and not pid1_start:
        return ""
    return f"{boot_id}:{pid1_start}"


def drain_request_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the drain-request marker, respecting HERMES_HOME."""
    base = home if home is not None else get_hermes_home()
    return Path(base) / _DRAIN_REQUEST_FILENAME


def write_drain_request(
    *,
    principal: str = "drain-control",
    suppress_notification: bool = False,
    home: Optional[Path] = None,
) -> dict[str, Any]:
    """Write the begin-drain marker. Returns the payload written.

    Atomic write so the gateway watcher never reads a half-written file.
    Idempotent: re-writing while a drain is already in progress just refreshes
    ``requested_at`` (harmless — the watcher keys off presence, not content).

    Stamps the marker with :func:`current_instantiation_epoch` so a marker that
    later survives a machine restart on the durable HERMES_HOME volume can be
    recognised as stale and ignored (NS-570).

    ``suppress_notification`` is a generic "be quiet on the shutdown that ends
    this drain" flag. When the drain culminates in a process exit (e.g. NAS
    recreates the machine for an auto-update image migration), the gateway's
    shutdown path reads it via :func:`drain_notification_suppressed` and skips
    the *home-channel* "gateway shutting down" broadcast — the operator-flavoured
    ping that would otherwise fire on every routine auto-update, potentially
    dozens of times a day. It NEVER suppresses the per-active-session interrupt
    ping. The gateway stays agnostic about *why* the drain is quiet; the policy
    of which drain causes set the flag lives entirely in the caller (NAS). The
    field defaults False so legacy/operator drains behave exactly as before.
    """
    payload = {
        "action": "drain",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "principal": principal,
        "epoch": current_instantiation_epoch(),
        "suppress_notification": bool(suppress_notification),
    }
    atomic_json_write(drain_request_path(home), payload)
    return payload


def clear_drain_request(*, home: Optional[Path] = None) -> bool:
    """Remove the drain marker (cancel-drain). Returns True if one existed.

    Best-effort: a missing file is not an error (cancel is idempotent).
    """
    path = drain_request_path(home)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        _log.warning("drain-control: failed to remove %s: %s", path, e)
        return False


def _marker_epoch_is_stale(body: dict[str, Any]) -> bool:
    """True iff ``body``'s epoch is a *definite* mismatch with this process.

    Lenient by design — returns False (i.e. "not stale, honour it") whenever it
    can't be sure:
      * the current epoch can't be computed ("" fallback, no /proc), OR
      * the marker carries no epoch (legacy marker, or a corrupt/contentless
        ``{}`` body).
    Only a marker whose epoch is present AND differs from the current
    instantiation epoch is considered stale. This preserves the
    fail-safe-toward-quiescing contract for malformed markers.
    """
    current = current_instantiation_epoch()
    if not current:
        return False
    marker_epoch = body.get("epoch")
    if not marker_epoch:
        return False
    return marker_epoch != current


def drain_requested(*, home: Optional[Path] = None) -> bool:
    """True iff a begin-drain marker for THIS instantiation is present.

    A marker whose ``epoch`` does not match the current instantiation epoch is
    treated as absent: it survived a container/VM restart (HERMES_HOME is a
    durable Fly volume on Hermes Cloud) and the lifecycle action that triggered
    the drain has already completed — honouring it would wedge the
    freshly-restarted gateway in ``draining`` (NS-570). The staleness check is
    lenient (see :func:`_marker_epoch_is_stale`): a legacy/corrupt marker with
    no epoch, or an environment without ``/proc``, still reads as drain-active.
    """
    body = read_drain_request(home=home)
    if body is None:
        return False
    if _marker_epoch_is_stale(body):
        return False
    return True


def drain_notification_suppressed(*, home: Optional[Path] = None) -> bool:
    """True iff an ACTIVE drain marker asks to suppress the shutdown broadcast.

    "Active" means exactly what :func:`drain_requested` means — a marker present
    AND stamped with the current instantiation epoch. A stale (other-epoch)
    marker that survived a machine restart on the durable HERMES_HOME volume is
    ignored here just as it is for drain state (NS-570): we must never let an
    orphaned marker's flag silence a *fresh* gateway's legitimate shutdown
    broadcast.

    Only honours the flag when it is explicitly truthy in the marker body. A
    legacy marker without the field, a corrupt/contentless ``{}`` body, or an
    absent marker all read as "not suppressed" (False) — fail toward the louder,
    more-visible behaviour, consistent with :func:`read_drain_request`'s
    never-raise contract. The gateway's shutdown path uses this to skip ONLY the
    home-channel broadcast; the per-active-session interrupt ping is unaffected.
    """
    body = read_drain_request(home=home)
    if body is None:
        return False
    if _marker_epoch_is_stale(body):
        return False
    return bool(body.get("suppress_notification"))


def read_drain_request(*, home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Return the marker payload, or ``None`` if absent.

    A present-but-unparseable marker returns ``{}`` (truthy-presence preserved
    via :func:`drain_requested`; callers that need the body get an empty dict
    rather than an exception). Never raises.
    """
    path = drain_request_path(home)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        _log.warning("drain-control: failed to read %s: %s", path, e)
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
