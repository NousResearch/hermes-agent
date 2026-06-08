"""Validation — heartbeat waits + drift detection.

Drift states (see design doc, Pushback B):
    healthy        — registered AND heartbeat fresh (<N seconds)
    down           — registered AND heartbeat stale (>N seconds)
    never-came-up  — registered AND no heartbeat ever seen
    ghost          — NOT registered AND heartbeat present (unauthorized publisher)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import paho.mqtt.client as mqtt

if TYPE_CHECKING:
    from .provisioner import ControllerConfig


HEARTBEAT_FRESH_SECONDS = 60  # tunable via config later

DriftState = Literal["healthy", "down", "never-came-up", "ghost"]


@dataclass
class NodeStatus:
    host: str
    state: DriftState
    registered: bool
    last_heartbeat_age_seconds: float | None  # None = never seen
    metadata_mismatch: bool = False                # nodes.yaml vs retained registry topic
    mismatch_detail: str | None = None             # human-readable when mismatch is True


def wait_for_heartbeat(namespace: str, host: str, cfg: "ControllerConfig", timeout: int = 30) -> bool:
    """Block until <namespace>/<host>/alive is seen with NON-EMPTY payload, or timeout.

    Non-empty matters: an empty payload could be a stale LWT from a prior life.
    Returns True on success, False on timeout.
    """
    saw_live = threading.Event()
    topic = f"{namespace}/{host}/alive"

    def _on_connect(c, userdata, flags, rc, properties=None):
        c.subscribe(topic, qos=0)

    def _on_message(c, userdata, msg):
        if msg.payload:  # non-empty == alive (empty == LWT/dead marker)
            saw_live.set()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"hermes-mesh-{namespace}-heartbeat-wait-{host}-{int(time.time())}")
    c.username_pw_set(cfg.broker_user, str(cfg.broker_password))
    if cfg.ca_cert_path:
        c.tls_set(ca_certs=str(cfg.ca_cert_path))
    port = 8883 if cfg.ca_cert_path else 1883
    c.on_connect = _on_connect
    c.on_message = _on_message
    c.connect(cfg.broker, port, keepalive=15)
    c.loop_start()
    try:
        return saw_live.wait(timeout=timeout)
    finally:
        c.loop_stop()
        c.disconnect()


def drift_check(cfg: "ControllerConfig") -> list[NodeStatus]:
    """Three-source drift check.

    1. nodes.yaml (registered set)
    2. <namespace>/registry/+ retained topics (broker's view)
    3. <namespace>/+/alive heartbeat freshness

    metadata_mismatch is set when (1) and (2) disagree on a host's role/capabilities;
    DriftState is determined by (1) and (3). Both axes are surfaced independently.

    Multi-tenant: scans every namespace appearing in nodes.yaml plus the controller
    default. A node registered under namespace=test is checked against test/* topics,
    not the controller's default namespace.
    """
    from . import registry  # local import to avoid module-load cycle

    registered_nodes = {n["host"]: n for n in registry.list_nodes()}

    # Collect every namespace we need to scan: controller default + every per-node namespace.
    namespaces = {cfg.namespace} | {n.get("namespace", cfg.namespace) for n in registered_nodes.values()}

    retained_manifests: dict[str, dict] = {}
    heartbeats: dict[str, tuple[str, float]] = {}
    for ns in namespaces:
        retained_manifests.update(
            {h: m for h, m in registry.query_retained_registry(cfg, namespace=ns, timeout=3.0).items()}
        )
        heartbeats.update(_collect_heartbeat_freshness(cfg, namespace=ns, timeout=3.0))

    all_hosts = set(registered_nodes) | set(retained_manifests) | set(heartbeats)
    statuses: list[NodeStatus] = []
    for host in sorted(all_hosts):
        is_registered = host in registered_nodes
        hb = heartbeats.get(host)  # (payload_str, publish_epoch) | (None, None) | absent
        if hb is None:
            # No retained alive topic at all.
            heartbeat_status: Literal["absent", "dead", "fresh", "stale"] = "absent"
            age = None
        elif hb == (None, None):
            # Empty retained payload — broker confirms this node is dead.
            heartbeat_status = "dead"
            age = None
        else:
            age = time.time() - hb[1]
            heartbeat_status = "fresh" if age <= HEARTBEAT_FRESH_SECONDS else "stale"

        if not is_registered:
            if heartbeat_status in ("fresh", "stale"):
                # Heartbeat exists without registration → ghost (unauthorized publisher)
                statuses.append(NodeStatus(
                    host=host, state="ghost", registered=False,
                    last_heartbeat_age_seconds=age,
                ))
            # absent/dead unregistered host → skip (nothing to report)
            continue

        # Registered host: state is determined by heartbeat_status
        if heartbeat_status == "absent":
            state: DriftState = "never-came-up"
        elif heartbeat_status == "dead":
            state = "down"
        elif heartbeat_status == "stale":
            state = "down"
        else:  # fresh
            state = "healthy"

        # Metadata mismatch: registered yaml entry vs retained manifest
        mismatch = False
        detail: str | None = None
        yaml_entry = registered_nodes.get(host, {})
        retained = retained_manifests.get(host)
        if retained:
            for field in ("role", "capabilities"):
                if yaml_entry.get(field) != retained.get(field):
                    mismatch = True
                    detail = f"{field}: yaml={yaml_entry.get(field)!r} retained={retained.get(field)!r}"
                    break
        elif is_registered:
            # Registered but no retained manifest — usually transient (broker restart, etc).
            # Surface as mismatch so operator can investigate, but don't change state.
            mismatch = True
            detail = "registered in nodes.yaml but no retained registry topic on broker"

        statuses.append(NodeStatus(
            host=host,
            state=state,
            registered=True,
            last_heartbeat_age_seconds=age,
            metadata_mismatch=mismatch,
            mismatch_detail=detail,
        ))
    return statuses


def _collect_heartbeat_freshness(cfg: "ControllerConfig", namespace: str | None = None, timeout: float = 3.0) -> dict[str, tuple[str | None, float | None]]:
    """Subscribe to <namespace>/+/alive briefly. Returns per host:

      (iso_payload, publish_epoch)  — node alive at the payload-embedded timestamp
      (None, None)                  — empty retained payload (LWT fired or clean dead-publish)
      <not in dict>                 — no retained topic at all

    Retained alive payloads keep their last fresh ISO timestamp even after the
    publisher disconnects gracefully. So "received-at" is always ~now for retained
    messages; the only honest freshness signal is the timestamp inside the payload.
    """
    from datetime import datetime
    ns = namespace if namespace is not None else cfg.namespace
    found: dict[str, tuple[str | None, float | None]] = {}
    topic = f"{ns}/+/alive"

    def _on_connect(c, userdata, flags, rc, properties=None):
        c.subscribe(topic, qos=0)

    def _on_message(c, userdata, msg):
        host = msg.topic.split("/")[1]
        if not msg.payload:
            # Empty retained payload — node was up, is now dead. Mark as dead, not absent.
            found[host] = (None, None)
            return
        payload_str = msg.payload.decode(errors="replace")
        try:
            # Python 3.11+ fromisoformat accepts both `+00:00` and (with Z replaced) the Z form.
            ts = datetime.fromisoformat(payload_str.replace("Z", "+00:00"))
            found[host] = (payload_str, ts.timestamp())
        except ValueError:
            # Non-ISO payload (legacy hand-rolled phoebe_*.py emit free-text).
            # Treat as absent — these nodes need to migrate to the bare-template
            # ISO contract before drift detection will see them as live.
            pass

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"hermes-mesh-{ns}-drift-{int(time.time())}")
    c.username_pw_set(cfg.broker_user, str(cfg.broker_password))
    if cfg.ca_cert_path:
        c.tls_set(ca_certs=str(cfg.ca_cert_path))
    port = 8883 if cfg.ca_cert_path else 1883
    c.on_connect = _on_connect
    c.on_message = _on_message
    c.connect(cfg.broker, port, keepalive=15)
    c.loop_start()
    try:
        time.sleep(timeout)
    finally:
        c.loop_stop()
        c.disconnect()
    return found
