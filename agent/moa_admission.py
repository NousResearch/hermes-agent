"""Nonblocking MoA admission for the managed router's resident-model budget."""

import functools
import threading
from urllib.parse import urlsplit
from weakref import WeakValueDictionary


class _RouterGate:
    def __init__(self, capacity):
        self.slots = threading.BoundedSemaphore(capacity)


_gates = WeakValueDictionary()
_gate_lock = threading.Lock()


def _endpoint_key(url):
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
    except (TypeError, ValueError):
        return None
    if not parsed.scheme or not host:
        return None
    if host == "localhost":
        host = "127.0.0.1"
    return parsed.scheme, host, port, parsed.path.rstrip("/")


def _shared_gate(admission):
    key = (_endpoint_key(admission["base_url"]), admission["incarnation"])
    with _gate_lock:
        gate = _gates.get(key)
        if gate is None:
            gate = _RouterGate(admission["capacity"])
            _gates[key] = gate
        return gate


def _release(gate, wake, future):
    # Completion, cancellation and failure all release exactly once. An abandoned running
    # request keeps its lease until it actually completes, including after a user interrupt.
    gate.slots.release()
    wake.set()


class ReferenceQueue:
    def __init__(self, slots, resolve_runtime):
        from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES, managed_model_admission

        admission = managed_model_admission()
        endpoint = _endpoint_key(admission["base_url"]) if admission else None
        gate = _shared_gate(admission) if endpoint else None
        self.waiting = []
        self._wake = threading.Event()
        for index, slot in slots:
            runtime = resolve_runtime(slot) if admission else {}
            base = runtime.get("base_url")
            local = bool(endpoint) and (_endpoint_key(base) == endpoint if base else
                                       runtime.get("provider") in LLAMACPP_ALIASES)
            self.waiting.append((index, slot, gate if local else None))

    def submit_ready(self, submit):
        futures = {}
        waiting = []
        for index, slot, gate in self.waiting:
            if gate is not None and not gate.slots.acquire(blocking=False):
                waiting.append((index, slot, gate))
                continue
            try:
                future = submit(slot)
            except BaseException:
                if gate is not None:
                    gate.slots.release()
                raise
            if gate is not None:
                future.add_done_callback(functools.partial(_release, gate, self._wake))
            futures[future] = index
        self.waiting = waiting
        return futures

    def wait(self, pending, wait_futures, timeout):
        if pending:
            return wait_futures(pending, timeout=timeout)
        # Another fan-out may own every local lease. Poll without occupying a worker;
        # cloud references are submitted independently and interrupts still get observed.
        self._wake.wait(timeout)
        self._wake.clear()
        return set(), set()

    def interrupt_waiting(self, results, placeholder, note):
        for index, slot, _gate in self.waiting:
            results[index] = placeholder(slot, note)
        self.waiting.clear()
