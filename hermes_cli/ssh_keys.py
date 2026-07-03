"""SSH key management for the Hermes dashboard.

Keys are stored under ``$HERMES_HOME/.ssh/``. Private key material is never
returned by the HTTP API — only fingerprints and public keys.

Host aliases are stored in ``config.yaml`` under ``ssh.hosts`` and synced to
``$HERMES_HOME/.ssh/config`` for OpenSSH.
"""

from __future__ import annotations

import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hermes_cli.config import cfg_get, load_config, save_config
from hermes_constants import get_hermes_home

_SSH_KEY_NAME_RE = re.compile(
    r"^id_(?:ed25519|rsa|ecdsa)(?:_[A-Za-z0-9][A-Za-z0-9_-]{0,62})?$"
)
_HOST_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RESERVED_SSH_FILES = frozenset({"config", "known_hosts", "authorized_keys"})


class SshKeysError(ValueError):
    """User-facing SSH keys management error."""


def get_ssh_dir() -> Path:
    return get_hermes_home() / ".ssh"


def ensure_ssh_dir() -> Path:
    ssh_dir = get_ssh_dir()
    ssh_dir.mkdir(parents=True, exist_ok=True)
    ssh_dir.chmod(stat.S_IRWXU)  # 0700
    return ssh_dir


def validate_key_name(name: str) -> str:
    name = (name or "").strip()
    if not _SSH_KEY_NAME_RE.match(name):
        raise SshKeysError(
            "Invalid key name. Use id_ed25519, id_rsa, or id_ed25519_<label> "
            "(letters, numbers, underscores, hyphens only)."
        )
    if name in _RESERVED_SSH_FILES:
        raise SshKeysError(f"{name!r} is reserved.")
    return name


def validate_host_alias(alias: str) -> str:
    alias = (alias or "").strip()
    if not _HOST_ALIAS_RE.match(alias):
        raise SshKeysError(
            "Invalid host alias. Use letters, numbers, dots, underscores, or hyphens."
        )
    return alias


def _private_key_path(name: str) -> Path:
    return ensure_ssh_dir() / validate_key_name(name)


def _public_key_path(name: str) -> Path:
    return Path(str(_private_key_path(name)) + ".pub")


def _run_ssh_keygen(args: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["ssh-keygen", *args],
            capture_output=True,
            text=True,
            input=input_text,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SshKeysError(
            "ssh-keygen is not installed. Install OpenSSH client tools."
        ) from exc


def _secure_private_key(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600


def _secure_public_key(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)  # 0644


def _fingerprint_for_public_key(pub_path: Path) -> str:
    result = _run_ssh_keygen(["-lf", str(pub_path)])
    if result.returncode != 0:
        raise SshKeysError(result.stderr.strip() or "Could not read key fingerprint")
    parts = result.stdout.strip().split()
    return parts[1] if len(parts) > 1 else ""


def _key_type_for_public_key(pub_path: Path) -> str:
    result = _run_ssh_keygen(["-lf", str(pub_path)])
    if result.returncode != 0:
        return "unknown"
    text = result.stdout.strip()
    if "(" in text and text.endswith(")"):
        return text.rsplit("(", 1)[-1].rstrip(")")
    return "unknown"


def _key_info(priv_path: Path) -> dict[str, Any]:
    name = priv_path.name
    pub_path = Path(str(priv_path) + ".pub")
    has_public = pub_path.is_file()
    info: dict[str, Any] = {
        "name": name,
        "has_private": priv_path.is_file(),
        "has_public": has_public,
        "fingerprint": None,
        "key_type": None,
        "public_key": None,
        "comment": None,
    }
    if has_public:
        info["fingerprint"] = _fingerprint_for_public_key(pub_path)
        info["key_type"] = _key_type_for_public_key(pub_path)
        pub_line = pub_path.read_text(encoding="utf-8").strip()
        info["public_key"] = pub_line
        parts = pub_line.split()
        if len(parts) >= 3:
            info["comment"] = " ".join(parts[2:])
    return info


def list_ssh_keys() -> list[dict[str, Any]]:
    ssh_dir = get_ssh_dir()
    if not ssh_dir.is_dir():
        return []

    keys: list[dict[str, Any]] = []
    for entry in sorted(ssh_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file() or entry.suffix == ".pub":
            continue
        if entry.name in _RESERVED_SSH_FILES:
            continue
        if not _SSH_KEY_NAME_RE.match(entry.name):
            continue
        keys.append(_key_info(entry))
    return keys


def generate_ssh_key(name: str = "id_ed25519", comment: str = "hermes-agent") -> dict[str, Any]:
    priv_path = _private_key_path(name)
    if priv_path.exists():
        raise SshKeysError(f"Key {priv_path.name} already exists.")

    comment = (comment or "hermes-agent").strip()[:256] or "hermes-agent"
    result = _run_ssh_keygen(
        ["-t", "ed25519", "-f", str(priv_path), "-N", "", "-C", comment]
    )
    if result.returncode != 0:
        priv_path.unlink(missing_ok=True)
        Path(str(priv_path) + ".pub").unlink(missing_ok=True)
        raise SshKeysError(result.stderr.strip() or "ssh-keygen failed.")

    _secure_private_key(priv_path)
    pub_path = _public_key_path(priv_path.name)
    if pub_path.exists():
        _secure_public_key(pub_path)
    return _key_info(priv_path)


def import_ssh_key(
    name: str,
    private_key: str,
    *,
    public_key: str | None = None,
) -> dict[str, Any]:
    priv_path = _private_key_path(name)
    if priv_path.exists():
        raise SshKeysError(f"Key {priv_path.name} already exists.")

    material = (private_key or "").strip()
    if "BEGIN" not in material:
        raise SshKeysError("Invalid private key — expected PEM/OpenSSH private key block.")

    if not material.endswith("\n"):
        material += "\n"

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(material)
        tmp_path = Path(tmp.name)
    try:
        tmp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        probe = _run_ssh_keygen(["-y", "-f", str(tmp_path)])
        if probe.returncode != 0:
            raise SshKeysError("Invalid private key — ssh-keygen could not derive public key.")
        derived_public = probe.stdout.strip()
        if not derived_public:
            raise SshKeysError("Invalid private key — empty public key derivation.")

        priv_path.write_text(material, encoding="utf-8")
        _secure_private_key(priv_path)

        pub_line = (public_key or derived_public).strip()
        if not pub_line:
            pub_line = derived_public
        if not pub_line.endswith("\n"):
            pub_line += "\n"
        pub_path = _public_key_path(priv_path.name)
        pub_path.write_text(pub_line, encoding="utf-8")
        _secure_public_key(pub_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return _key_info(priv_path)


def delete_ssh_key(name: str) -> None:
    priv_path = _private_key_path(name)
    if not priv_path.exists():
        raise SshKeysError(f"Key {priv_path.name} not found.")
    pub_path = _public_key_path(priv_path.name)
    priv_path.unlink()
    pub_path.unlink(missing_ok=True)


def _load_hosts_from_config() -> list[dict[str, Any]]:
    cfg = load_config()
    raw = cfg_get(cfg, "ssh", "hosts", default=[]) or []
    if not isinstance(raw, list):
        return []
    hosts: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("alias") or "").strip()
        host_name = str(item.get("host_name") or item.get("hostname") or "").strip()
        if not alias or not host_name:
            continue
        hosts.append(
            {
                "alias": alias,
                "host_name": host_name,
                "user": str(item.get("user") or "").strip(),
                "port": int(item.get("port") or 22),
                "identity_file": str(item.get("identity_file") or "id_ed25519").strip(),
            }
        )
    return hosts


def _save_hosts_to_config(hosts: list[dict[str, Any]]) -> None:
    cfg = load_config()
    cfg.setdefault("ssh", {})["hosts"] = hosts
    save_config(cfg)
    _sync_ssh_config_file(hosts)


def _sync_ssh_config_file(hosts: list[dict[str, Any]]) -> None:
    ssh_dir = ensure_ssh_dir()
    config_path = ssh_dir / "config"
    lines = [
        "# Managed by Hermes Agent dashboard.\n",
        "# Host aliases configured here are written from config.yaml (ssh.hosts).\n\n",
    ]
    for host in hosts:
        alias = validate_host_alias(host["alias"])
        host_name = str(host["host_name"]).strip()
        if not host_name:
            continue
        identity = validate_key_name(str(host.get("identity_file") or "id_ed25519"))
        lines.append(f"Host {alias}\n")
        lines.append(f"    HostName {host_name}\n")
        user = str(host.get("user") or "").strip()
        if user:
            lines.append(f"    User {user}\n")
        port = int(host.get("port") or 22)
        if port != 22:
            lines.append(f"    Port {port}\n")
        lines.append(f"    IdentityFile {ssh_dir / identity}\n")
        lines.append("    IdentitiesOnly yes\n")
        lines.append("    StrictHostKeyChecking accept-new\n\n")
    config_path.write_text("".join(lines), encoding="utf-8")
    config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600


def list_ssh_hosts() -> list[dict[str, Any]]:
    return _load_hosts_from_config()


def upsert_ssh_host(
    *,
    alias: str,
    host_name: str,
    user: str = "",
    port: int = 22,
    identity_file: str = "id_ed25519",
) -> dict[str, Any]:
    alias = validate_host_alias(alias)
    host_name = (host_name or "").strip()
    if not host_name:
        raise SshKeysError("host_name is required.")
    identity_file = validate_key_name(identity_file)
    if not _private_key_path(identity_file).exists():
        raise SshKeysError(f"Identity key {identity_file} does not exist.")

    port = int(port or 22)
    if port < 1 or port > 65535:
        raise SshKeysError("port must be between 1 and 65535.")

    hosts = _load_hosts_from_config()
    entry = {
        "alias": alias,
        "host_name": host_name,
        "user": (user or "").strip(),
        "port": port,
        "identity_file": identity_file,
    }
    replaced = False
    for idx, existing in enumerate(hosts):
        if existing.get("alias") == alias:
            hosts[idx] = entry
            replaced = True
            break
    if not replaced:
        hosts.append(entry)
    hosts.sort(key=lambda h: str(h.get("alias", "")).lower())
    _save_hosts_to_config(hosts)
    return entry


def delete_ssh_host(alias: str) -> None:
    alias = validate_host_alias(alias)
    hosts = [h for h in _load_hosts_from_config() if h.get("alias") != alias]
    if len(hosts) == len(_load_hosts_from_config()):
        raise SshKeysError(f"Host alias {alias!r} not found.")
    _save_hosts_to_config(hosts)


def test_ssh_host(alias: str) -> dict[str, Any]:
    alias = validate_host_alias(alias)
    hosts = _load_hosts_from_config()
    if not any(h.get("alias") == alias for h in hosts):
        raise SshKeysError(f"Host alias {alias!r} not found.")

    ssh_dir = ensure_ssh_dir()
    config_path = ssh_dir / "config"
    if not config_path.is_file():
        raise SshKeysError("SSH config file is missing.")

    try:
        result = subprocess.run(
            [
                "ssh",
                "-F",
                str(config_path),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                alias,
                "echo",
                "ok",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SshKeysError("ssh client is not installed.") from exc

    ok = result.returncode == 0 and "ok" in (result.stdout or "")
    return {
        "ok": ok,
        "message": (result.stdout or result.stderr or "").strip() or (
            "Connection succeeded." if ok else "Connection failed."
        ),
    }
