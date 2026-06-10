"""Public webhook exposure: ngrok and Tailscale serve/funnel tunnels.

Port of OpenClaw's ``src/tunnel.ts``. Resolution order lives in
``runtime._resolve_public_url``: explicit ``public_url`` wins; otherwise
``tunnel.provider`` selects one of these. The returned handle's
``public_url`` is the public *base* URL (no path) — providers append
``serve.path`` themselves.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from .config import VoiceCallConfig

logger = logging.getLogger(__name__)

NGROK_STARTUP_TIMEOUT_S = 30.0
TAILSCALE_TIMEOUT_S = 10.0


@dataclass
class TunnelHandle:
    public_url: str
    provider: str
    _process: Optional[asyncio.subprocess.Process] = None
    _stop_args: Optional[List[str]] = None  # tailscale off command

    async def stop(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None
        if self._stop_args:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self._stop_args,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (OSError, asyncio.TimeoutError):
                logger.debug("tailscale tunnel off failed", exc_info=True)
            self._stop_args = None


async def _run_command(args: List[str], timeout_s: float = 15.0) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"command timed out: {args[0]}")
    if proc.returncode != 0:
        detail = (stderr or stdout or b"").decode("utf-8", "replace")[:400]
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {detail}")
    return (stdout or b"").decode("utf-8", "replace")


def parse_ngrok_log_line(line: str) -> Optional[str]:
    """Extract the public URL from one ngrok JSON log line, if present."""
    try:
        log = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(log, dict):
        return None
    url = log.get("url")
    if url and (log.get("msg") == "started tunnel" or log.get("addr")):
        return str(url)
    return None


async def start_ngrok_tunnel(config: VoiceCallConfig) -> TunnelHandle:
    """Spawn the ngrok CLI and parse the public URL from its JSON logs."""
    auth_token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if auth_token:
        await _run_command(["ngrok", "config", "add-authtoken", auth_token])

    args = [
        "ngrok", "http", str(config.serve.port),
        "--log", "stdout", "--log-format", "json",
    ]
    if config.tunnel.ngrok_domain:
        args += ["--domain", config.tunnel.ngrok_domain]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _await_url() -> str:
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                stderr = b""
                if proc.stderr is not None:
                    stderr = await proc.stderr.read()
                detail = stderr.decode("utf-8", "replace")[:400]
                raise RuntimeError(f"ngrok exited before reporting a URL: {detail}")
            url = parse_ngrok_log_line(raw.decode("utf-8", "replace").strip())
            if url:
                return url

    try:
        public_url = await asyncio.wait_for(
            _await_url(), timeout=NGROK_STARTUP_TIMEOUT_S
        )
    except (asyncio.TimeoutError, RuntimeError):
        if proc.returncode is None:
            proc.terminate()
        raise
    logger.info("voice_call: ngrok tunnel active at %s", public_url)
    return TunnelHandle(
        public_url=public_url.rstrip("/"), provider="ngrok", _process=proc
    )


async def get_tailscale_dns_name() -> Optional[str]:
    try:
        output = await _run_command(["tailscale", "status", "--json"])
        status = json.loads(output)
        dns_name = (status.get("Self") or {}).get("DNSName", "")
        return dns_name.rstrip(".") or None
    except (RuntimeError, json.JSONDecodeError, OSError):
        return None


async def start_tailscale_tunnel(config: VoiceCallConfig, mode: str) -> TunnelHandle:
    """Expose the local port via ``tailscale serve`` or ``tailscale funnel``."""
    dns_name = await get_tailscale_dns_name()
    if not dns_name:
        raise RuntimeError(
            "could not get Tailscale DNS name — is Tailscale running?"
        )
    path = config.serve.path
    local_url = f"http://127.0.0.1:{config.serve.port}{path}"
    await _run_command(
        ["tailscale", mode, "--bg", "--yes", "--set-path", path, local_url],
        timeout_s=TAILSCALE_TIMEOUT_S,
    )
    # Tunnel base URL excludes the path; providers re-append serve.path.
    public_url = f"https://{dns_name}"
    logger.info("voice_call: tailscale %s active at %s%s", mode, public_url, path)
    return TunnelHandle(
        public_url=public_url,
        provider=f"tailscale-{mode}",
        _stop_args=["tailscale", mode, "off", path],
    )


async def start_tunnel(config: VoiceCallConfig) -> TunnelHandle:
    provider = config.tunnel.provider
    if provider == "ngrok":
        return await start_ngrok_tunnel(config)
    if provider == "tailscale-serve":
        return await start_tailscale_tunnel(config, "serve")
    if provider == "tailscale-funnel":
        return await start_tailscale_tunnel(config, "funnel")
    raise ValueError(f"unknown tunnel provider {provider!r}")
