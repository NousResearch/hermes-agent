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

Why the max-age (#85433). The epoch handles the restart case, but it bakes in
the assumption that the action a drain protects always ends in a machine
restart. When a drain-gated action completes *without* recreating the container
and the writer never cancels the drain (writer crash, forgotten cleanup), the
orphaned marker still carries the *current* epoch — so the epoch check honours
it and the gateway bounces every inbound message forever (observed in the
field: a cloud instance refused all Telegram turns for ~3 days). The marker's
``requested_at`` timestamp is therefore also checked: a marker older than
:data:`DRAIN_REQUEST_MAX_AGE_SECONDS` reads as stale. Same leniency contract as
the epoch — only a *definite* expiry (timestamp present, parseable, and too
old) is ignored; a missing/corrupt timestamp still reads as drain-active. A
deliberately long drain has a sanctioned keep-alive: re-calling
:func:`write_drain_request` refreshes ``requested_at``.
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

# Max-age fallback for a same-epoch orphaned marker (#85433). Drain-gated
# lifecycle actions complete in minutes; an hour is comfortably past any
# legitimate drain while still bounding the wedge a leaked marker can cause
# (vs. the unbounded outage observed in the field). Long-running drains
# refresh the marker via write_drain_request() (idempotent re-write bumps
# ``requested_at``) rather than raising this bound.
DRAIN_REQUEST_MAX_AGE_SECONDS = 3600.0

# Dedup guard for the expired-marker warning: the drain watcher re-reads the
# marker every second, and an expired orphan sits on disk until removed, so
# an unconditional warning would fire ~86k times/day. Keyed by the marker's
# ``requested_at`` string — a keep-alive re-write (new timestamp) that later
# expires again logs again, which is the desired behaviour.
_expiry_logged_for: Optional[str] = None


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
        pass

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
        pass

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
    Idempotent: re-writing while a drain is already in progress refreshes
    ``requested_at`` — the sanctioned keep-alive for a drain that legitimately
    needs longer than :data:`DRAIN_REQUEST_MAX_AGE_SECONDS`.

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


def _marker_is_expired(body: dict[str, Any]) -> bool:
    """True iff ``body``'s ``requested_at`` is *definitely* too old (#85433).

    The max-age fallback for a same-epoch orphan: a drain-gated action that
    completes WITHOUT a machine restart leaves a marker the epoch check cannot
    reject, and if the writer never cancels the drain the gateway is wedged in
    ``draining`` for the life of the container. Bounding the marker's lifetime
    by :data:`DRAIN_REQUEST_MAX_AGE_SECONDS` converts that unbounded outage
    into a self-healing one.

    Same leniency contract as :func:`_marker_epoch_is_stale` — returns False
    ("not expired, honour it") whenever it can't be sure:
      * the marker carries no ``requested_at`` (legacy/corrupt/contentless
        body), OR
      * the timestamp isn't a parseable ISO-8601 string.
    Only a timestamp that parses AND lies more than the max-age in the past is
    considered expired. A future-dated timestamp (clock skew) is honoured. The
    expiry is logged loudly — but once per marker, not per poll: the gateway's
    drain watcher re-reads the marker every second, and an expired orphan stays
    on disk until an operator or writer removes it, so an unconditional warning
    would repeat ~86k times/day. This path only fires when a writer leaked a
    marker, and the log is the operator's breadcrumb.
    """
    global _expiry_logged_for
    raw = body.get("requested_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        requested_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if requested_at.tzinfo is None:
        requested_at = requested_at.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - requested_at).total_seconds()
    if age <= DRAIN_REQUEST_MAX_AGE_SECONDS:
        return False
    if _expiry_logged_for != raw:
        _expiry_logged_for = raw
        _log.warning(
            "drain-control: ignoring expired drain marker (requested_at=%s, "
            "age=%.0fs > max %.0fs, principal=%s) — the drain that wrote it "
            "was never cancelled; treating as stale so the gateway keeps "
            "accepting turns.",
            raw,
            age,
            DRAIN_REQUEST_MAX_AGE_SECONDS,
            body.get("principal"),
        )
    return True


def _marker_is_stale(body: dict[str, Any]) -> bool:
    """True iff the marker is definitely from a drain that is already over.

    Two independent, individually-lenient signals (either suffices):
      * epoch mismatch — the marker survived a machine restart (NS-570);
      * expiry — a same-epoch orphan outlived any legitimate drain (#85433).
    """
    return _marker_epoch_is_stale(body) or _marker_is_expired(body)


def drain_requested(*, home: Optional[Path] = None) -> bool:
    """True iff a begin-drain marker for THIS instantiation is present.

    A marker whose ``epoch`` does not match the current instantiation epoch is
    treated as absent: it survived a container/VM restart (HERMES_HOME is a
    durable Fly volume on Hermes Cloud) and the lifecycle action that triggered
    the drain has already completed — honouring it would wedge the
    freshly-restarted gateway in ``draining`` (NS-570). A marker whose
    ``requested_at`` is older than :data:`DRAIN_REQUEST_MAX_AGE_SECONDS` is
    likewise treated as absent: it is a same-epoch orphan whose drain-gated
    action completed without a restart and was never cancelled (#85433). Both
    staleness checks are lenient (see :func:`_marker_epoch_is_stale` /
    :func:`_marker_is_expired`): a legacy/corrupt marker with no epoch and no
    timestamp, or an environment without ``/proc``, still reads as
    drain-active.
    """
    body = read_drain_request(home=home)
    if body is None:
        return False
    if _marker_is_stale(body):
        return False
    return True


def drain_notification_suppressed(*, home: Optional[Path] = None) -> bool:
    """True iff an ACTIVE drain marker asks to suppress the shutdown broadcast.

    "Active" means exactly what :func:`drain_requested` means — a marker present
    AND stamped with the current instantiation epoch AND not past its max-age.
    A stale (other-epoch) marker that survived a machine restart on the durable
    HERMES_HOME volume, or an expired same-epoch orphan (#85433), is
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
    if _marker_is_stale(body):
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
