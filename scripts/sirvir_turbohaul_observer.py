#!/usr/bin/env python3
"""Sirvir's read-only Turbohaul pressure observer.

This process never controls model lifecycle.  It turns Turbohaul's status
snapshot into a graduated, hysteretic Sirvir policy decision stream that can
be reviewed before a future, separately approved control plane exists.

Memory policy: the P12 256K policy (docs/adr/0012-p12-256k-memory-policy.md)
keeps model weights GPU0-only with no CPU/system-RAM offload on the normal
fast path, reserves huge-context KV-cache offload as explicitly opt-in, and
preserves GPU1 isolation.  This observer only reports pressure; it never
offloads weights or moves data to CPU/system RAM.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone

DEFAULT_STATUS_URL = "http://127.0.0.1:11401/status"

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
    def __init__(self, free_mib, total_mib, generation_state, model_state, queue_depth):
        self.free_mib = tuple(free_mib)
        self.total_mib = tuple(total_mib)
        self.generation_state = generation_state
        self.model_state = model_state
        self.queue_depth = queue_depth

    @property
    def minimum_free_gib(self):
        return min(self.free_mib) / 1024


def snapshot_from_status(payload):
    """Build a read-only policy snapshot from Turbohaul's /status response."""
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
    )


def target_level_for_free_gib(minimum_free_gib):
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
            "current_level": self.level,
            "target_level": target_level,
            "next_level": self.level,
            "decision": "healthy",
            "minimum_free_gib": round(minimum_free_gib, 3),
            "free_mib": list(snapshot.free_mib),
            "generation_state": snapshot.generation_state,
            "model_state": snapshot.model_state,
            "queue_depth": snapshot.queue_depth,
        }

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
    parser = argparse.ArgumentParser(description="Sirvir Turbohaul observe-only policy controller")
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--stability-checks", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.stability_checks < 1:
        parser.error("interval must be positive and stability-checks must be at least one")

    observer = Observer(stability_checks=args.stability_checks)
    while True:
        try:
            snapshot = snapshot_from_status(fetch_status(args.status_url))
            print(json.dumps(observer.evaluate(snapshot), sort_keys=True), flush=True)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(json.dumps({"decision": "status_unavailable", "error": str(error)}), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
