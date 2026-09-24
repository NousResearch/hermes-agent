"""Production-safe, user-triggered Compose bootstrap for a profile state store."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import stat
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlparse

POSTGRES_IMAGE = "pgvector/pgvector:pg16"
_POSTGRES_USER = "hermes"
_POSTGRES_DATABASE = "hermes"


def _profile_identity(home: Path) -> str:
    return hashlib.sha256(str(home.resolve()).encode()).hexdigest()[:12]


def _find_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_bootstrap_password(existing_dsn: str | None, metadata: dict[str, object], *, project: str) -> str | None:
    """Return a password only when the DSN identifies this home's local service."""
    if not existing_dsn or metadata.get("project") != project:
        return None
    metadata_port = metadata.get("port")
    if not isinstance(metadata_port, (str, int)):
        return None
    try:
        port = int(metadata_port)
        parsed = urlparse(existing_dsn)
        dsn_port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.hostname != "127.0.0.1"
        or dsn_port != port
        or parsed.username != _POSTGRES_USER
        or parsed.path != f"/{_POSTGRES_DATABASE}"
    ):
        return None
    return parsed.password


def _write_compose(path: Path, *, project: str, port: int) -> None:
    compose = {
        "name": project,
        "services": {
            "postgres": {
                "image": POSTGRES_IMAGE,
                "restart": "unless-stopped",
                "environment": {
                    "POSTGRES_USER": _POSTGRES_USER,
                    "POSTGRES_DB": _POSTGRES_DATABASE,
                    "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}",
                },
                "ports": [f"127.0.0.1:{port}:5432"],
                "volumes": [f"{project}_data:/var/lib/postgresql/data"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"],
                    "interval": "5s", "timeout": "3s", "retries": 12, "start_period": "5s",
                },
            }
        },
        "volumes": {f"{project}_data": {"name": f"{project}_data"}},
    }
    import yaml
    path.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")
    os.chmod(path, 0o600)


def bootstrap_postgresql_state_store(
    hermes_home: Path, *, existing_dsn: str | None = None, runner=None, sleep=time.sleep, attempts: int = 30
) -> str:
    """Start/reuse the profile-local Compose service and return its DSN without printing it."""
    home = Path(hermes_home)
    state_dir = home / "state-store" / "postgresql"
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    metadata_path = state_dir / "bootstrap.json"
    identity = _profile_identity(home)
    project = f"hermes-state-store-{identity}"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata_is_secure = metadata_path.exists() and stat.S_IMODE(metadata_path.stat().st_mode) == 0o600
    port = int(metadata.get("port") or _find_loopback_port())
    compose_path = state_dir / "compose.yaml"
    _write_compose(compose_path, project=project, port=port)
    password = _local_bootstrap_password(
        existing_dsn,
        metadata if metadata_is_secure else {},
        project=project,
    ) or secrets.token_urlsafe(32)
    dsn = f"postgresql://{_POSTGRES_USER}:{quote(password, safe='')}@127.0.0.1:{port}/{_POSTGRES_DATABASE}"
    metadata_path.write_text(json.dumps({"port": port, "project": project}) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o600)
    invoke = runner or (lambda argv, *, env: subprocess.run(argv, env=env, check=False).returncode)
    env = {**os.environ, "POSTGRES_PASSWORD": password}
    prefix = ["docker", "compose", "-p", project, "-f", str(compose_path)]
    if invoke([*prefix, "up", "-d"], env=env) != 0:
        raise RuntimeError("PostgreSQL Compose startup failed; no state-store configuration was changed.")
    for _ in range(attempts):
        if invoke([*prefix, "exec", "-T", "postgres", "pg_isready", "-U", _POSTGRES_USER, "-d", _POSTGRES_DATABASE], env=env) == 0:
            return dsn
        sleep(1)
    raise RuntimeError("PostgreSQL did not become ready; no state-store configuration was changed.")
