"""Configuration for the Mobile Compute Node.

Loads from hermes config.yaml under ``mobile_compute`` section:

    mobile_compute:
      enabled: true
      server:
        host: "0.0.0.0"
        port: 8765
        tls:
          cert_path: "/path/to/cert.pem"
          key_path: "/path/to/key.pem"
          ca_path: "/path/to/ca.pem"
          require_client_cert: true
      client:
        default_endpoint: "http://localhost:8765"
        timeout_seconds: 30
        retry_attempts: 3
        retry_backoff_base: 1.5
        tls:
          cert_path: "/path/to/client.crt"
          key_path: "/path/to/client.key"
          ca_path: "/path/to/ca.pem"

If the section is absent, all methods return a no-op / disabled config.
This module does NOT require config.yaml to exist — it degrades gracefully.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home_override


@dataclasses.dataclass
class TLSConfig:
    """TLS/mTLS configuration for a server or client."""
    enabled: bool = False
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    require_client_cert: bool = False


@dataclasses.dataclass
class MobileComputeServerConfig:
    """Server (iPhone-side) network configuration."""
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8765
    tls: TLSConfig = dataclasses.field(default_factory=TLSConfig)


@dataclasses.dataclass
class MobileComputeClientConfig:
    """Client (Windows-side) configuration for talking to the iPhone server."""
    default_endpoint: str = "http://localhost:8765"
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_backoff_base: float = 1.5
    tls: TLSConfig = dataclasses.field(default_factory=TLSConfig)


@dataclasses.dataclass
class MobileComputeConfig:
    """Top-level mobile_compute config block."""
    enabled: bool = False
    server: MobileComputeServerConfig = dataclasses.field(default_factory=MobileComputeServerConfig)
    client: MobileComputeClientConfig = dataclasses.field(default_factory=MobileComputeClientConfig)
    # Directory for persistent state (registrations, task queues, logs)
    state_dir: str = ""

    @classmethod
    def load(cls, config_dict: Optional[Dict[str, Any]] = None) -> "MobileComputeConfig":
        """Load config from a config.yaml dict or return disabled defaults."""
        if config_dict is None:
            try:
                from hermes_cli.config import load_config_readonly
                raw = load_config_readonly()
                config_dict = raw.get("mobile_compute")
            except Exception:
                config_dict = None

        if not config_dict or not isinstance(config_dict, dict):
            return cls()

        mc = config_dict
        enabled = bool(mc.get("enabled", False))

        srv_raw = mc.get("server", {}) or {}
        clt_raw = mc.get("client", {}) or {}

        tls_s = srv_raw.get("tls", {}) or {}
        tls_c = clt_raw.get("tls", {}) or {}

        server = MobileComputeServerConfig(
            enabled=enabled,
            host=srv_raw.get("host", "0.0.0.0"),
            port=int(srv_raw.get("port", 8765)),
            tls=TLSConfig(
                enabled=bool(tls_s.get("enabled", False)),
                cert_path=tls_s.get("cert_path"),
                key_path=tls_s.get("key_path"),
                ca_path=tls_s.get("ca_path"),
                require_client_cert=bool(tls_s.get("require_client_cert", False)),
            ),
        )

        client = MobileComputeClientConfig(
            default_endpoint=clt_raw.get("default_endpoint", "http://localhost:8765"),
            timeout_seconds=int(clt_raw.get("timeout_seconds", 30)),
            retry_attempts=int(clt_raw.get("retry_attempts", 3)),
            retry_backoff_base=float(clt_raw.get("retry_backoff_base", 1.5)),
            tls=TLSConfig(
                enabled=bool(tls_c.get("enabled", False)),
                cert_path=tls_c.get("cert_path"),
                key_path=tls_c.get("key_path"),
                ca_path=tls_c.get("ca_path"),
            ),
        )

        hermes_home = get_hermes_home_override() or Path.home() / ".hermes"
        state_dir = str(Path(hermes_home) / "mobile_compute")

        return cls(
            enabled=enabled,
            server=server,
            client=client,
            state_dir=state_dir,
        )