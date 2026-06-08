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

if TYPE_CHECKING:
    from .provisioner import ControllerConfig, NodeSpec


NODES_YAML_PATH = Path.home() / ".hermes" / "mesh" / "nodes.yaml"


def _mqtt_client(cfg: "ControllerConfig", client_id_suffix: str) -> mqtt.Client:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"hermes-mesh-{cfg.namespace}-{client_id_suffix}")
    c.username_pw_set(cfg.broker_user, str(cfg.broker_password))
    if cfg.ca_cert_path:
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
    """Append/update an entry in nodes.yaml. Idempotent (host = key).

    node_user: runtime user discovered via probe (ProbeResult.user). When None,
    falls back to spec.user (SSH login user). Threading the probed value
    keeps the manifest honest — `spec.user` is operator intent (often None
    = "use current user"), `node_user` is ground truth from `id -un` on the
    remote.
    """
    NODES_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    if NODES_YAML_PATH.exists():
        data = yaml.safe_load(NODES_YAML_PATH.read_text()) or {}
    else:
        data = {}
    data.setdefault("namespace", spec.namespace)
    data.setdefault("broker", spec.broker)
    data.setdefault("nodes", {})
    data["nodes"][spec.host] = {
        "role": spec.role,
        "host": spec.host,
        "user": node_user if node_user is not None else spec.user,
        "namespace": spec.namespace,  # per-node — multi-tenant ground truth (NOT the top-level controller default)
        "capabilities": spec.capabilities,
        "added": datetime.now(timezone.utc).date().isoformat(),
    }
    NODES_YAML_PATH.write_text(yaml.safe_dump(data, sort_keys=False))


def remove_from_nodes_yaml(host: str) -> None:
    if not NODES_YAML_PATH.exists():
        return
    data = yaml.safe_load(NODES_YAML_PATH.read_text()) or {}
    nodes = data.get("nodes", {})
    if host in nodes:
        del nodes[host]
        NODES_YAML_PATH.write_text(yaml.safe_dump(data, sort_keys=False))


def list_nodes() -> list[dict]:
    """Read nodes.yaml and return registered node entries."""
    if not NODES_YAML_PATH.exists():
        return []
    data = yaml.safe_load(NODES_YAML_PATH.read_text()) or {}
    nodes = data.get("nodes", {})
    return [{"host": k, **v} for k, v in nodes.items()]


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
