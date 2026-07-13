"""Registry — dual-source: nodes.yaml (controller truth) + retained MQTT topics (live view).

Source of truth: ~/.hermes/mesh/nodes.yaml (git-trackable, portable).
Live view: <namespace>/registry/<host> retained topics on the broker.
Drift detection: compare both + heartbeat freshness (see validation.py).
"""
from __future__ import annotations

import json
import ssl
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import paho.mqtt.client as mqtt
import yaml

from hermes_constants import get_hermes_home

if TYPE_CHECKING:
    from .provisioner import ControllerConfig, NodeSpec


def _nodes_yaml_path() -> Path:
    """Profile-aware nodes.yaml path under HERMES_HOME."""
    return get_hermes_home() / "mesh" / "nodes.yaml"


# Kept for backwards compatibility with external callers. Resolves at call
# time so profile overrides are honored.
NODES_YAML_PATH = _nodes_yaml_path()


def _mqtt_client(cfg: "ControllerConfig", client_id_suffix: str) -> mqtt.Client:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"hermes-mesh-{cfg.namespace}-{client_id_suffix}")
    c.username_pw_set(cfg.broker_user, str(cfg.broker_password))
    if cfg.ca_cert_path:
        if not Path(cfg.ca_cert_path).exists():
            raise SystemExit(
                "mesh: ca_cert_path does not exist: %s. Point it at a valid CA "
                "cert for TLS, or leave it empty for plaintext MQTT." % cfg.ca_cert_path
            )
        c.tls_set(ca_certs=str(cfg.ca_cert_path))
    return c


def _broker_port(cfg: "ControllerConfig") -> int:
    return 8883 if cfg.ca_cert_path else 1883


def publish_manifest(spec: "NodeSpec", cfg: "ControllerConfig") -> None:
    """Publish retained <namespace>/registry/<host> with capability manifest."""
    payload = json.dumps({
        "host": spec.host,
        "role": spec.role,
        "capabilities": spec.capabilities,
        "namespace": spec.namespace,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    })
    topic = f"{spec.namespace}/registry/{spec.host}"
    c = _mqtt_client(cfg, f"publish-{spec.host}")
    c.connect(cfg.broker, _broker_port(cfg), keepalive=15)
    c.loop_start()
    try:
        info = c.publish(topic, payload, qos=1, retain=True)
        info.wait_for_publish(timeout=10)
    finally:
        c.loop_stop()
        c.disconnect()


def unpublish_manifest(host: str, cfg: "ControllerConfig") -> None:
    """Clear the retained registry topic for a host (publish empty payload, retain=True)."""
    topic = f"{cfg.namespace}/registry/{host}"
    c = _mqtt_client(cfg, f"unpublish-{host}")
    c.connect(cfg.broker, _broker_port(cfg), keepalive=15)
    c.loop_start()
    try:
        info = c.publish(topic, "", qos=1, retain=True)
        info.wait_for_publish(timeout=10)
    finally:
        c.loop_stop()
        c.disconnect()


def append_to_nodes_yaml(spec: "NodeSpec", cfg: "ControllerConfig", node_user: str | None = None) -> None:
    """Append/update an entry in nodes.yaml. Idempotent ((namespace, host) = key).

    node_user: runtime user discovered via probe (ProbeResult.user). When None,
    falls back to spec.user (SSH login user). Threading the probed value
    keeps the manifest honest — `spec.user` is operator intent (often None
    = "use current user"), `node_user` is ground truth from `id -un` on the
    remote.
    """
    nodes_path = _nodes_yaml_path()
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    if nodes_path.exists():
        data = yaml.safe_load(nodes_path.read_text()) or {}
    else:
        data = {}
    data.setdefault("namespace", spec.namespace)
    data.setdefault("broker", spec.broker)
    data.setdefault("nodes", {})
    # Key by (namespace, host) so the same host in two different namespaces
    # doesn't overwrite each other. Within a single namespace, host alone
    # remains unique (re-provisioning the same node updates in place).
    node_key = f"{spec.namespace}:{spec.host}"
    data["nodes"][node_key] = {
        "role": spec.role,
        "host": spec.host,
        "user": node_user if node_user is not None else spec.user,
        "namespace": spec.namespace,  # per-node — multi-tenant ground truth (NOT the top-level controller default)
        "capabilities": spec.capabilities,
        "added": datetime.now(timezone.utc).date().isoformat(),
    }
    nodes_path.write_text(yaml.safe_dump(data, sort_keys=False))


def remove_from_nodes_yaml(host: str, namespace: str | None = None) -> None:
    """Remove a node entry. If namespace is given, only removes the entry
    for that namespace; otherwise removes ALL entries for the host across
    every namespace (preserving prior single-arg behavior for the CLI).
    """
    nodes_path = _nodes_yaml_path()
    if not nodes_path.exists():
        return
    data = yaml.safe_load(nodes_path.read_text()) or {}
    nodes = data.get("nodes", {})
    if namespace is not None:
        key = f"{namespace}:{host}"
        removed = key in nodes
        if removed:
            del nodes[key]
    else:
        # Remove every entry whose host matches (cross-namespace).
        to_del = [k for k, v in nodes.items() if v.get("host") == host]
        for k in to_del:
            del nodes[k]
        removed = bool(to_del)
    if removed:
        nodes_path.write_text(yaml.safe_dump(data, sort_keys=False))


def list_nodes() -> list[dict]:
    """Read nodes.yaml and return registered node entries."""
    nodes_path = _nodes_yaml_path()
    if not nodes_path.exists():
        return []
    data = yaml.safe_load(nodes_path.read_text()) or {}
    nodes = data.get("nodes", {})
    # Each value already carries its own "host" and "namespace" fields;
    # the dict key is the composite "namespace:host" used for dedup.
    return [{"key": k, **v} for k, v in nodes.items()]


def query_retained_registry(cfg: "ControllerConfig", namespace: str | None = None, timeout: float = 3.0) -> dict[str, dict]:
    """Subscribe to <namespace>/registry/+ briefly, collect retained manifests.

    namespace defaults to cfg.namespace; pass explicit value to scan a different tenant.
    Returns: {hostname: manifest_dict}. Empty payloads (unpublished) are skipped.
    """
    ns = namespace if namespace is not None else cfg.namespace
    found: dict[str, dict] = {}
    done = threading.Event()

    def _on_connect(c, userdata, flags, rc, properties=None):
        c.subscribe(f"{ns}/registry/+", qos=0)

    def _on_message(c, userdata, msg):
        if not msg.payload:
            return
        try:
            data = json.loads(msg.payload.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        host = msg.topic.rsplit("/", 1)[-1]
        found[host] = data

    c = _mqtt_client(cfg, "query-registry")
    c.on_connect = _on_connect
    c.on_message = _on_message
    c.connect(cfg.broker, _broker_port(cfg), keepalive=15)
    c.loop_start()
    try:
        done.wait(timeout=timeout)
    finally:
        c.loop_stop()
        c.disconnect()
    return found
