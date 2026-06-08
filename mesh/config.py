"""Controller config — load / save / construct.

Lives at ~/.hermes/mesh/config.yaml (chmod 600 — contains broker_password).

Schema:
    namespace: my-mesh
    broker: 10.0.0.5
    broker_user: my-mesh
    broker_password: <secret>
    ca_cert_path: /etc/my-mesh/ca.crt   # optional, enables TLS on :8883
    template_dirs:                       # optional; defaults computed if absent
      - ~/.hermes/mesh/roles             # user overrides win
      - <repo>/mesh/templates/roles      # shipped defaults
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .provisioner import ControllerConfig, _Redacted

CONFIG_DIR = Path.home() / ".hermes" / "mesh"
CONFIG_PATH = CONFIG_DIR / "config.yaml"


def _default_template_dirs() -> list[Path]:
    """User overrides first, then shipped defaults from this scaffold/repo."""
    user_dir = CONFIG_DIR / "roles"
    repo_default = Path(__file__).resolve().parent / "templates" / "roles"
    return [user_dir, repo_default]


def load() -> ControllerConfig:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No controller config at {CONFIG_PATH}. Run `hermes mesh init` first."
        )
    raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    tdirs_raw = raw.get("template_dirs")
    if tdirs_raw:
        template_dirs = [Path(p).expanduser() for p in tdirs_raw]
    else:
        template_dirs = _default_template_dirs()
    ca_cert = raw.get("ca_cert_path")
    return ControllerConfig(
        namespace=raw["namespace"],
        broker=raw["broker"],
        broker_user=raw["broker_user"],
        broker_password=_Redacted(raw["broker_password"]),
        ca_cert_path=Path(ca_cert).expanduser() if ca_cert else None,
        template_dirs=template_dirs,
    )


def save(cfg: ControllerConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {
        "namespace": cfg.namespace,
        "broker": cfg.broker,
        "broker_user": cfg.broker_user,
        # _Redacted is a str subclass; yaml dumps the actual value (intended here).
        "broker_password": str(cfg.broker_password),
        "ca_cert_path": str(cfg.ca_cert_path) if cfg.ca_cert_path else None,
    }
    # If the user customized template_dirs (i.e. not the default), persist them.
    if cfg.template_dirs and cfg.template_dirs != _default_template_dirs():
        payload["template_dirs"] = [str(p) for p in cfg.template_dirs]
    # Write with 0600 from creation so broker_password is never briefly
    # world-readable under the umask-default mode. The trailing chmod only
    # matters when the file already existed with looser permissions.
    fd = os.open(CONFIG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(yaml.safe_dump(payload, sort_keys=False))
    os.chmod(CONFIG_PATH, 0o600)


def init_interactive() -> ControllerConfig:
    """Interactive wizard — mirrors `hermes gateway setup` UX shape."""
    print("hermes mesh init — one-time controller setup")
    print("---------------------------------------------")
    namespace = input("Namespace (default: hermes): ").strip() or "hermes"
    broker = input("MQTT broker host (e.g. 10.0.0.5): ").strip()
    if not broker:
        raise ValueError("Broker host required")
    broker_user = input(f"MQTT username (default: {namespace}): ").strip() or namespace
    broker_password = input("MQTT password: ").strip()
    if not broker_password:
        raise ValueError("Broker password required")
    ca_cert = input("CA cert path for TLS [skip = plain MQTT]: ").strip() or None

    cfg = ControllerConfig(
        namespace=namespace,
        broker=broker,
        broker_user=broker_user,
        broker_password=_Redacted(broker_password),
        ca_cert_path=Path(ca_cert).expanduser() if ca_cert else None,
        template_dirs=_default_template_dirs(),
    )
    save(cfg)
    print(f"\n✓ Wrote {CONFIG_PATH} (chmod 600)")
    print(f"  namespace: {namespace}")
    print(f"  broker:    {broker}")
    print(f"  user:      {broker_user}")
    print(f"  tls:       {'yes' if ca_cert else 'no'}")
    return cfg


def init_noninteractive(
    namespace: str,
    broker: str,
    broker_user: str,
    broker_password: str,
    ca_cert_path: Optional[str] = None,
) -> ControllerConfig:
    """Non-interactive variant for tests + scripted setup."""
    cfg = ControllerConfig(
        namespace=namespace,
        broker=broker,
        broker_user=broker_user,
        broker_password=_Redacted(broker_password),
        ca_cert_path=Path(ca_cert_path).expanduser() if ca_cert_path else None,
        template_dirs=_default_template_dirs(),
    )
    save(cfg)
    return cfg
