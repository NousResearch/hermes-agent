"""Caddy site management for the Desktop setup flow.

The gateway owns the webapp listener on loopback; Caddy terminates TLS on
:80/:443 and reverse-proxies to it. This module renders and idempotently
upserts one site block into an existing Caddyfile (backup first), then
validates and reloads. It never creates a Caddyfile from nothing and never
touches other site blocks.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_CADDYFILE = Path("/etc/caddy/Caddyfile")


def render_site_block(host: str, port: int, *, internal_tls: bool = False) -> str:
    """One Caddyfile site block. ``internal_tls`` uses Caddy's local CA (for
    ``.local`` names public ACME cannot serve); the default is automatic
    Let's Encrypt HTTPS for a public domain."""
    lines = [f"{host} {{", f"\treverse_proxy 127.0.0.1:{port}", "\tencode zstd gzip"]
    if internal_tls:
        lines.append("\ttls internal")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_redirect_block(name: str, ip: str, port: int) -> str:
    """Local-mode friendly entry: ``http://<name>.local`` (port 80) redirects to
    the gateway's LAN origin. Nous Portal OAuth only accepts IP/real-domain
    callbacks, so sign-in must complete on the IP origin — the mDNS name is the
    friendly front door, not the OAuth origin. Explicit ``http://`` keeps Caddy
    from attempting ACME for a .local name."""
    return f"http://{name} {{\n\tredir http://{ip}:{port}{{uri}} permanent\n}}\n"


def _site_exists(text: str, host: str) -> bool:
    return re.search(rf"^(?:http://)?{re.escape(host)}\s*\{{", text, re.MULTILINE) is not None


def upsert_block(caddyfile: Path, host: str, block: str) -> tuple[bool, Optional[Path]]:
    """Append *block* when no site for *host* exists yet. Returns
    ``(changed, backup_path)``; idempotent, backup first, other blocks kept."""
    text = caddyfile.read_text(encoding="utf-8-sig")
    if _site_exists(text, host):
        return False, None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = caddyfile.with_name(caddyfile.name + f".bak-{stamp}")
    shutil.copy2(caddyfile, backup)
    caddyfile.write_text(text.rstrip("\n") + "\n\n" + block, encoding="utf-8")
    return True, backup


def upsert_site_block(caddyfile: Path, host: str, port: int, *, internal_tls: bool = False,
                      ) -> tuple[bool, Optional[Path]]:
    """Append the reverse-proxy site block when missing (see upsert_block)."""
    return upsert_block(caddyfile, host, render_site_block(host, port, internal_tls=internal_tls))


def validate(caddyfile: Path, *, run: Optional[Callable] = None) -> bool:
    """``caddy validate`` against the file; False on any failure."""
    if shutil.which("caddy") is None:
        return False
    runner = run or subprocess.run
    try:
        result = runner(["caddy", "validate", "--config", str(caddyfile), "--adapter", "caddyfile"],
                        capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("caddy validate failed: %s", exc)
        return False


def restore_backup(caddyfile: Path, backup: Path) -> None:
    shutil.copy2(backup, caddyfile)


def reload_caddy(*, run: Optional[Callable] = None) -> bool:
    """``systemctl reload caddy`` (restart as fallback). Reload engages the TLS
    listener and triggers certificate issuance after a cold start."""
    runner = run or subprocess.run
    for verb in ("reload", "restart"):
        try:
            result = runner(["systemctl", verb, "caddy"], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("systemctl %s caddy failed: %s", verb, exc)
    return False


def caddy_installed() -> bool:
    return shutil.which("caddy") is not None