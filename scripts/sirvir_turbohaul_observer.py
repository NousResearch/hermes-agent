#!/usr/bin/env python3
"""Sirvir's read-only Turbohaul/Turbofit pressure observer.

This process never controls model lifecycle.  It turns a runtime status
snapshot into a graduated, hysteretic Sirvir policy decision stream that can
be reviewed before a future, separately approved control plane exists.

Memory policy: the P12 256K policy (docs/adr/0012-p12-256k-memory-policy.md)
keeps model weights GPU0-only with no CPU/system-RAM offload on the normal
fast path, reserves huge-context KV-cache offload as explicitly opt-in, and
preserves GPU1 isolation.  This observer only reports pressure; it never
offloads weights or moves data to CPU/system RAM.

Backends (select with --backend; default remains the legacy Turbohaul
manager until cutover changes the explicit configuration):

- ``turbohaul`` — legacy manager on :11401.  /status carries per-GPU free
  VRAM, generation state and queue depths directly.

- ``turbofit`` — local gateway on :8091.  Its /status carries main/aux route
  resolution and is always HTTP 200; it has NO VRAM data and returns 503 on
  /health when intentionally idle.  Idle-vs-unhealthy is therefore derived
  from read-only runtime/controller state files:

    * runtime state (TURBOFIT_RUNTIME_STATE, default
      ~/.local/state/turbofit/runtime-state.json): selected profile + routes.
    * native state dir (TURBOFIT_NATIVE_STATE, default
      ~/.local/state/turbofit/native): ``main.json``/``aux.json`` owned
      runtime records, ``lifecycle-endpoint.json`` (controller presence) and
      ``lifecycle-state.json`` (leases, orphaned flag).

  Read-only classification:
    - ``residents_present`` — a role resolves to a ready/loading backend or
      an owned runtime record exists.
    - ``idle`` — everything down, no owned runtime, no leases, not orphaned:
      an intentionally drained system, not a failure.
    - ``stale`` — controller state claims a selection but the lifecycle
      endpoint file is missing/invalid, leases exist with no resident, or
      lifecycle state reports orphaned: operator reconciliation is required.
    - ``unknown`` — state files unreadable/absent with no other signal.

  The observer reads state files and issues exactly one GET per cycle.  It
  never POSTs to the lifecycle endpoint, never mutates routes or residency,
  and never performs inference.  Missing VRAM data is reported as
  ``vram_unavailable``: the contraction ladder is not evaluated, so absent
  telemetry can never fake a pressure emergency.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STATUS_URL = "http://127.0.0.1:11401/status"
TURBOFIT_DEFAULT_STATUS_URL = "http://127.0.0.1:8091/status"
DEFAULT_TURBOFIT_RUNTIME_STATE = "~/.local/state/turbofit/runtime-state.json"
DEFAULT_TURBOFIT_NATIVE_STATE = "~/.local/state/turbofit/native"

# Retained Sirvir contraction ladder, expressed as minimum free GiB on any GPU.
SHRINK_CONTEXT_BELOW_GIB = 6.0
EXPERT_OFFLOAD_BELOW_GIB = 4.0
SWAP_MODEL_BELOW_GIB = 3.0
STOP_AUX_BELOW_GIB = 2.0
API_SURVIVAL_BELOW_GIB = 1.0

# Recovery thresholds retain the original +4 GiB hysteresis.
RECOVER_CONTEXT_ABOVE_GIB = 10.0
RECOVER_EXPERTS_ABOVE_GIB = 8.0
RECOVER_SWAP_ABOVE_GIB = 7.0
RECOVER_AUX_ABOVE_GIB = 6.0
RECOVER_MAIN_ABOVE_GIB = 5.0

PROVEN_DARWIN_CONTEXT = 65536


class Snapshot:
    def __init__(self, free_mib, total_mib, generation_state, model_state, queue_depth,
                 backend="turbohaul", extra=None):
        self.free_mib = list(free_mib) if free_mib is not None else None
        self.total_mib = list(total_mib) if total_mib is not None else None
        self.generation_state = generation_state
        self.model_state = model_state
        self.queue_depth = queue_depth
        self.backend = backend
        self.extra = dict(extra or {})

    @property
    def minimum_free_gib(self):
        if self.free_mib is None:
            return None
        return min(self.free_mib) / 1024


def snapshot_from_status(payload):
    """Build a read-only policy snapshot from the legacy Turbohaul /status."""
    free_mib = payload.get("vram")
    total_mib = payload.get("vram_total_mib")
    if not isinstance(free_mib, list) or not free_mib:
        raise ValueError("Turbohaul status has no per-GPU free VRAM data")
    if not isinstance(total_mib, list) or len(total_mib) != len(free_mib):
        raise ValueError("Turbohaul status has invalid per-GPU total VRAM data")

    generation = payload.get("generation") or {}
    queue = payload.get("queue") or {}
    model_state = next(
        (
            name
            for name in ("active", "loading", "grace", "idle_hot")
            if payload.get(name) is not None
        ),
        "unloaded",
    )
    queue_depth = int(queue.get("acceptance_buffer_depth", 0)) + int(
        queue.get("staging_queue_depth", 0)
    )
    return Snapshot(
        free_mib=free_mib,
        total_mib=total_mib,
        generation_state=str(generation.get("state", "unknown")),
        model_state=model_state,
        queue_depth=queue_depth,
        backend="turbohaul",
    )


def _read_json_file(path):
    """Return parsed JSON, or None. Missing file is 'no signal'; a file that
    exists but is unreadable/corrupt is flagged so the observer can mirror the
    lifecycle owner's own 'corrupt history fails closed' behaviour."""
    try:
        with open(path, encoding="utf-8-sig") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return False  # present but unreadable/invalid: distinct from absent


def _role_state(role_payload):
    """Normalise one gateway /status role entry to (state, alias)."""
    if not isinstance(role_payload, dict):
        return "down", None
    alias = str(role_payload.get("alias") or "").strip() or None
    if alias is not None and alias.lower() == "none":
        alias = None
    if role_payload.get("type") == "down" or not alias:
        return "down", alias
    state = str(role_payload.get("state") or "down")
    if state not in ("ready", "loading"):
        return "down", alias
    return state, alias


def _owned_runtimes(native_state):
    """Read owned {main,aux}.json records without acting on them."""
    owned = {}
    for role in ("main", "aux"):
        record = _read_json_file(Path(native_state) / f"{role}.json")
        if not isinstance(record, dict):
            continue
        try:
            owned[role] = {
                "pid": int(record["pid"]),
                "alias": str(record["alias"]),
                "port": int(record["port"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return owned


def _lifecycle_owner_locked(native_state):
    """Observe the native controller's exclusive flock; never take a lock.

    Endpoint JSON survives crashes. The kernel releases this singleton lock
    when its owner dies. Unavailable Linux lock telemetry fails closed.
    This is a sampled ownership signal, not a proof of controller progress.
    """
    try:
        info = (Path(native_state) / "lifecycle.lock").stat()
        target = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
        for line in Path("/proc/locks").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 6 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
                continue
            try:
                major, minor, inode = fields[5].split(":")
                identity = (int(major, 16), int(minor, 16), int(inode))
                if identity == target and int(fields[4]) > 0:
                    return True
            except ValueError:
                continue
    except OSError:
        pass
    return False


def _lifecycle_view(native_state):
    """Read-only lifecycle controller view: presence, leases, orphan flag."""
    endpoint = _read_json_file(Path(native_state) / "lifecycle-endpoint.json")
    endpoint_valid = bool(
        isinstance(endpoint, dict)
        and endpoint.get("host") == "127.0.0.1"
        and type(endpoint.get("port")) is int
        and 0 < endpoint.get("port", 0) < 65536
    )
    state = _read_json_file(Path(native_state) / "lifecycle-state.json")
    orphaned = False
    leases = {}
    if state is False:
        # Present but unreadable/corrupt: mirror the lifecycle owner's own
        # 'do not reclaim potentially busy residents based on corrupt history'
        # rule by reporting orphaned (stale → operator reconciliation).
        orphaned = True
    elif isinstance(state, dict) and state.get("schema") == 2:
        orphaned = bool(state.get("orphaned"))
        raw_leases = state.get("leases")
        if isinstance(raw_leases, dict):
            for role in raw_leases.values():
                if role in ("main", "aux"):
                    leases[role] = leases.get(role, 0) + 1
                else:
                    leases["other"] = leases.get("other", 0) + 1
    return {
        "endpoint_present": endpoint_valid,
        "owner_locked": _lifecycle_owner_locked(native_state),
        "orphaned": orphaned,
        "leases": leases,
    }


def snapshot_from_turbofit_status(payload, runtime_state=None, native_state=None,
                                  controller_state=None):
    """Build a read-only snapshot from the Turbofit gateway /status.

    The gateway answers 200 even when intentionally idle (roles report
    ``{"alias": "none", "type": "down"}``), so health and residency come from
    the payload plus read-only state files, never from HTTP status alone.
    """
    main_state, main_alias = _role_state(payload.get("main"))
    aux_state, aux_alias = _role_state(payload.get("aux"))

    native_state = native_state or os.path.expanduser(DEFAULT_TURBOFIT_NATIVE_STATE)
    owned = _owned_runtimes(native_state)
    lifecycle = _lifecycle_view(native_state)

    runtime = _read_json_file(runtime_state) if runtime_state else None
    selection = None
    if isinstance(runtime, dict) and runtime.get("active"):
        selection = str(runtime.get("active"))
    controller = _read_json_file(controller_state) if controller_state else None
    if isinstance(controller, dict):
        selection = selection or str(controller.get("profile_id") or "") or selection

    roles_resident = any(state != "down" for state in (main_state, aux_state))
    leases_live = any(count > 0 for count in lifecycle["leases"].values())

    # Precedence: degraded controller state beats live residents for
    # attention; residents beat leftover bookkeeping; silence means idle.
    if lifecycle["orphaned"] or (
        selection and not (lifecycle["endpoint_present"] and lifecycle["owner_locked"])
    ):
        resident_state = "stale"
    elif roles_resident:
        resident_state = "residents_present"
    elif owned:
        resident_state = "stale"
    elif leases_live:
        resident_state = "stale"
    elif selection and not lifecycle["endpoint_present"]:
        resident_state = "stale"
    else:
        resident_state = "idle" if lifecycle["endpoint_present"] and lifecycle["owner_locked"] else "unknown"

    free_mib = payload.get("vram")
    total_mib = payload.get("vram_total_mib")
    if not (isinstance(free_mib, list) and free_mib
            and isinstance(total_mib, list) and len(total_mib) == len(free_mib)):
        free_mib = None
        total_mib = None

    return Snapshot(
        free_mib=free_mib,
        total_mib=total_mib,
        generation_state="unknown",
        model_state=main_state if main_state != "down" else aux_state,
        queue_depth=0,
        backend="turbofit",
        extra={
            "resident_state": resident_state,
            "selection": selection,
            "roles": {"main": {"state": main_state, "alias": main_alias},
                      "aux": {"state": aux_state, "alias": aux_alias}},
            "owned": owned,
            "lifecycle": lifecycle,
            "vram_source": "gateway_status" if free_mib is not None else "unavailable",
        },
    )


def target_level_for_free_gib(minimum_free_gib):
    if minimum_free_gib is None:
        return 0  # no VRAM telemetry: never fake a pressure emergency
    if minimum_free_gib < API_SURVIVAL_BELOW_GIB:
        return 5
    if minimum_free_gib < STOP_AUX_BELOW_GIB:
        return 4
    if minimum_free_gib < SWAP_MODEL_BELOW_GIB:
        return 3
    if minimum_free_gib < EXPERT_OFFLOAD_BELOW_GIB:
        return 2
    if minimum_free_gib < SHRINK_CONTEXT_BELOW_GIB:
        return 1
    return 0


def next_recovery_level(current_level, minimum_free_gib):
    if minimum_free_gib is None:
        return current_level  # no telemetry: hold, never fake recovery
    if current_level >= 5 and minimum_free_gib > RECOVER_MAIN_ABOVE_GIB:
        return 4
    if current_level >= 4 and minimum_free_gib > RECOVER_AUX_ABOVE_GIB:
        return 3
    if current_level >= 3 and minimum_free_gib > RECOVER_SWAP_ABOVE_GIB:
        return 2
    if current_level >= 2 and minimum_free_gib > RECOVER_EXPERTS_ABOVE_GIB:
        return 1
    if current_level >= 1 and minimum_free_gib > RECOVER_CONTEXT_ABOVE_GIB:
        return 0
    return current_level


def contraction_decision(level):
    return {
        1: f"would_hold_context_at_{PROVEN_DARWIN_CONTEXT}",
        2: "would_offload_moe_experts_if_eligible",
        3: "would_swap_to_validated_smaller_model",
        4: "would_stop_aux_models",
        5: "would_enter_api_survival_mode",
    }[level]


class Observer:
    def __init__(self, stability_checks=2):
        self.stability_checks = stability_checks
        self.level = 0
        self._pending_target = 0
        self._stable_polls = 0

    def evaluate(self, snapshot):
        minimum_free_gib = snapshot.minimum_free_gib
        target_level = target_level_for_free_gib(minimum_free_gib)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": snapshot.backend,
            "current_level": self.level,
            "target_level": target_level,
            "next_level": self.level,
            "decision": "healthy",
            "minimum_free_gib": round(minimum_free_gib, 3) if minimum_free_gib is not None else None,
            "free_mib": list(snapshot.free_mib) if snapshot.free_mib is not None else None,
            "generation_state": snapshot.generation_state,
            "model_state": snapshot.model_state,
            "queue_depth": snapshot.queue_depth,
        }
        event.update(snapshot.extra)

        if snapshot.backend == "turbofit" and snapshot.extra.get("resident_state") in {"stale", "unknown"}:
            event["decision"] = "controller_state_unavailable"
            self._pending_target = self._stable_polls = 0
            return event
        if snapshot.backend == "turbofit" and minimum_free_gib is None:
            event["decision"] = "vram_unavailable"
            self._pending_target = self._stable_polls = 0
            return event

        if target_level > self.level:
            if snapshot.generation_state not in {"idle", "unknown"}:
                event["decision"] = "defer_generation_in_flight"
                return event
            if target_level != self._pending_target:
                self._pending_target = target_level
                self._stable_polls = 1
            else:
                self._stable_polls += 1
            if self._stable_polls < self.stability_checks:
                event["decision"] = "await_stability"
                return event

            self.level = min(target_level, self.level + 1)
            self._pending_target = 0
            self._stable_polls = 0
            event["next_level"] = self.level
            event["decision"] = contraction_decision(self.level)
            return event

        if target_level < self.level:
            recovered_level = next_recovery_level(self.level, minimum_free_gib)
            if recovered_level < self.level:
                previous_level = self.level
                self.level = recovered_level
                event["next_level"] = self.level
                if previous_level == 1 and self.level == 0:
                    event["decision"] = f"would_keep_context_at_{PROVEN_DARWIN_CONTEXT}"
                else:
                    event["decision"] = f"would_recover_to_level_{self.level}"
                return event

        self._pending_target = 0
        self._stable_polls = 0
        return event


def fetch_status(status_url, timeout=5):
    request = urllib.request.Request(status_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description="Sirvir Turbohaul/Turbofit observe-only policy controller")
    parser.add_argument("--backend", choices=("turbohaul", "turbofit", "auto"), default="turbohaul")
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--turbofit-status-url", default=TURBOFIT_DEFAULT_STATUS_URL)
    parser.add_argument(
        "--turbofit-runtime-state",
        default=os.environ.get("TURBOFIT_RUNTIME_STATE", DEFAULT_TURBOFIT_RUNTIME_STATE),
    )
    parser.add_argument(
        "--turbofit-native-state",
        default=os.environ.get("TURBOFIT_NATIVE_STATE", DEFAULT_TURBOFIT_NATIVE_STATE),
    )
    parser.add_argument("--turbofit-controller-state", default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--stability-checks", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.stability_checks < 1:
        parser.error("interval must be positive and stability-checks must be at least one")

    turbofit_runtime_state = os.path.expanduser(args.turbofit_runtime_state)
    turbofit_native_state = os.path.expanduser(args.turbofit_native_state)

    def collect_once():
        if args.backend == "turbohaul":
            return snapshot_from_status(fetch_status(args.status_url))
        turbohaul_url = args.status_url if args.backend == "auto" else None
        return snapshot_for_urls(
            args.turbofit_status_url,
            turbohaul_url,
            runtime_state=turbofit_runtime_state,
            native_state=turbofit_native_state,
            controller_state=args.turbofit_controller_state,
        )

    observer = Observer(stability_checks=args.stability_checks)
    while True:
        try:
            snapshot = collect_once()
            print(json.dumps(observer.evaluate(snapshot), sort_keys=True), flush=True)
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
            print(json.dumps({"decision": "status_unavailable", "error": str(error)}), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


def snapshot_for_urls(turbofit_url, turbohaul_url, *, runtime_state=None,
                      native_state=None, controller_state=None):
    """One backend selection cycle against explicit URLs (test seam for main).

    ``turbohaul_url=None`` means Turbofit was explicitly selected: a fetch
    failure or unrecognised payload surfaces the error instead of silently
    probing the legacy manager.  With both URLs set, an unavailable or
    unrecognised Turbofit endpoint falls back to the legacy backend.
    """
    payload = None
    fetch_error = None
    try:
        payload = fetch_status(turbofit_url)
    except (OSError, ValueError, urllib.error.URLError) as error:
        fetch_error = error
    if isinstance(payload, dict) and str(payload.get("gateway", "")).startswith("turbofit-gateway"):
        return snapshot_from_turbofit_status(
            payload,
            runtime_state=runtime_state,
            native_state=native_state,
            controller_state=controller_state,
        )
    if turbohaul_url is None:
        raise fetch_error or ValueError("Turbofit gateway status payload not recognised")
    return snapshot_from_status(fetch_status(turbohaul_url))


if __name__ == "__main__":
    main()