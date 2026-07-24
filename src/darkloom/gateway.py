"""Hermes gateway integration — Tor lifecycle + proxy injection.

This module bridges darkloom with the Hermes messaging gateway.
The gateway already has a centralized proxy resolver at
gateway.platforms.base.resolve_proxy_url() that checks:
  1. Platform-specific env var (TELEGRAM_PROXY, DISCORD_PROXY, etc.)
  2. HTTPS_PROXY / HTTP_PROXY / ALL_PROXY (and lowercase variants)
  3. macOS system proxy

By setting ALL_PROXY=socks5://127.0.0.1:9050 BEFORE the gateway starts,
every platform adapter that calls resolve_proxy_url() automatically
routes through Tor.

Usage:
    from darkloom.gateway import start_tor_for_gateway

    # Start Tor and inject ALL_PROXY before gateway connects
    mgr = start_tor_for_gateway()

    # ... gateway starts, all platforms route through Tor ...

    # Shutdown
    mgr.stop()

Or as a standalone pre-start wrapper:
    python -m darkloom.gateway -- hermes gateway run
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional
from urllib.parse import urlsplit

from darkloom.constants import (
    BRIDGES_PATH,
    DEFAULT_SOCKS_PORT,
    is_tor_installed,
)
if TYPE_CHECKING:
    from darkloom.manager import TorManager

from darkloom.policy import authorize_subprocess

from darkloom.secure_files import atomic_private_write, private_lock

from darkloom.privacy import get_logger

logger = get_logger(__name__)

# Environment variables injected for gateway-wide Tor routing.
# ALL_PROXY is the catch-all that resolve_proxy_url() checks after
# platform-specific vars. Setting it means every platform adapter
# that calls resolve_proxy_url() picks up the SOCKS5 proxy.
_GENERIC_PROXY_NAMES = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "TOR_PROXY")
_PLATFORM_PROXY_NAMES = (
    "TELEGRAM_PROXY", "DISCORD_PROXY", "MATRIX_PROXY", "PHOTON_PROXY",
    "WHATSAPP_PROXY", "SLACK_PROXY", "GRPC_PROXY",
)
PROXY_ENV_VARS = tuple(
    dict.fromkeys(
        name
        for upper in (*_GENERIC_PROXY_NAMES, *_PLATFORM_PROXY_NAMES)
        for name in (upper, upper.lower())
    )
)
NO_PROXY_ENV_VARS = ("NO_PROXY", "no_proxy")
_MISSING = object()
_environment_snapshot: Optional[dict[str, object]] = None
_environment_lock = threading.RLock()


@dataclass(frozen=True)
class ProxyPolicy:
    """One immutable, validated routing decision for the gateway process."""

    url: str
    strict: bool
    loopback_bypass: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")


class ProxyPolicyError(RuntimeError):
    """The process environment cannot satisfy the requested proxy policy."""


def _strict_mode(environment: Mapping[str, str]) -> bool:
    return environment.get("TOR_STRICT_MODE", "").strip().lower() in {"1", "true", "yes"}


def _validate_no_proxy(value: str, name: str, policy: ProxyPolicy) -> None:
    if not value.strip() or not policy.strict:
        return
    entries = {part.strip().strip("[]").lower() for part in value.split(",") if part.strip()}
    if not entries.issubset(policy.loopback_bypass):
        raise ProxyPolicyError(
            f"{name} may contain only {', '.join(policy.loopback_bypass)} in strict mode"
        )


def _validate_proxy_value(name: str, value: str, policy: ProxyPolicy) -> None:
    candidate = value.strip()
    if not candidate:
        raise ProxyPolicyError(f"{name} is set but empty")
    if candidate.lower() in {"direct", "direct://", "none", "off"}:
        raise ProxyPolicyError(f"{name} disables proxy routing")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"socks5", "socks5h"} or not parsed.hostname or parsed.port is None:
        raise ProxyPolicyError(f"{name} has an unsupported proxy value: {candidate!r}")
    if candidate != policy.url:
        raise ProxyPolicyError(f"{name} conflicts with immutable proxy policy {policy.url!r}")


def establish_proxy_policy(
    socks_port: int = DEFAULT_SOCKS_PORT,
    *,
    strict: Optional[bool] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> ProxyPolicy:
    """Build and validate the policy before any network client is imported.

    In strict mode every pre-existing, platform-specific proxy setting is
    considered authoritative input: empty, direct, unsupported, and conflicting
    settings fail closed rather than being silently overwritten.
    """
    env = os.environ if environment is None else environment
    if not 1 <= socks_port <= 65535:
        raise ProxyPolicyError(f"invalid SOCKS port: {socks_port}")
    policy = ProxyPolicy(
        url=f"socks5://127.0.0.1:{socks_port}",
        strict=_strict_mode(env) if strict is None else strict,
    )
    if policy.strict:
        for name in NO_PROXY_ENV_VARS:
            if name in env:
                _validate_no_proxy(env[name], name, policy)
        for name in PROXY_ENV_VARS:
            if name in env:
                _validate_proxy_value(name, env[name], policy)
    return policy


def _is_http_proxy(value: str) -> bool:
    """Return whether a proxy URL is suitable for HTTP-only integrations."""
    parsed = urlsplit(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _policy_environment(
    policy: ProxyPolicy, environment: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    values = {name: policy.url for name in PROXY_ENV_VARS}
    # Slack's SDK rejects SOCKS URLs. In non-strict mode, retain an explicitly
    # configured HTTP-to-SOCKS bridge (for example, a local Privoxy instance)
    # rather than replacing it with the generic Tor SOCKS endpoint.
    env = os.environ if environment is None else environment
    if not policy.strict:
        for name in ("SLACK_PROXY", "slack_proxy"):
            if name in env and _is_http_proxy(env[name]):
                values[name] = env[name]
    values.update({"TOR_PROXY": policy.url, "tor_proxy": policy.url, "TOR_ENABLED": "1"})
    # Preserve loopback-only communication with local gateway sidecars.
    bypass = ",".join(policy.loopback_bypass)
    values.update({"NO_PROXY": bypass, "no_proxy": bypass})
    return values


GATEWAY_ENV_VARS = _policy_environment(
    ProxyPolicy(f"socks5://127.0.0.1:{DEFAULT_SOCKS_PORT}", strict=False)
)
# Add TOR_HEALTH so clients can observe gateway-level Tor health state.
GATEWAY_ENV_VARS["TOR_HEALTH"] = "1"

# TOR_SKIP_LLM is a signal for LLM integrations to bypass Tor on their own
# client.  Generic proxy variables must remain in place: they are also the
# gateway-wide routing mechanism for platform adapters and tools.
# OpenAI, Anthropic, and their CDNs (Cloudflare) block known Tor exit IPs
# with 403/429/CAPTCHA. The API key already identifies your account — Tor
# for LLM calls provides IP privacy but not account anonymity. Bypassing
# Tor for LLM calls preserves streaming performance (TTFT) while keeping
# all other traffic (messaging platforms, web tools, subagents) through Tor.


def inject_gateway_env(
    socks_port: int = DEFAULT_SOCKS_PORT, *, policy: Optional[ProxyPolicy] = None
) -> ProxyPolicy:
    """Set ALL_PROXY + HTTPS_PROXY + HTTP_PROXY for gateway-wide Tor routing.

    Must be called BEFORE the Hermes gateway initializes any platform
    connections. The gateway loads ~/.hermes/.env at startup, so
    writing ALL_PROXY to .env is an alternative to runtime injection.

    Platform adapters that use resolve_proxy_url() will automatically
    pick up ALL_PROXY and route through Tor:
      - Telegram:   ✅ TELEGRAM_PROXY > ALL_PROXY > HTTPS_PROXY
      - Discord:    ✅ DISCORD_PROXY  > ALL_PROXY > HTTPS_PROXY
      - Matrix:     ✅ MATRIX_PROXY   > ALL_PROXY > HTTPS_PROXY
      - Slack:      ⚠️ HTTP proxy only (SOCKS rejected by Slack SDK)
      - Photon:     ✅ After applying 0001-photon-proxy.patch
      - WhatsApp:   ✅ After applying 0002-whatsapp-proxy.patch
      - Email:      ❌ Raw SMTP/IMAP — no HTTP proxy support
    """
    global _environment_snapshot
    policy = policy or establish_proxy_policy(socks_port)
    if policy.url != f"socks5://127.0.0.1:{socks_port}":
        raise ProxyPolicyError("supplied policy does not match requested SOCKS port")
    values = _policy_environment(policy)
    with _environment_lock:
        if _environment_snapshot is None:
            _environment_snapshot = {
                key: os.environ.get(key, _MISSING) for key in values
            }
        os.environ.update(values)
    logger.info(
        "Gateway Tor environment injected: ALL_PROXY=%s, TOR_ENABLED=1, "
        "%d env vars set",
        policy.url,
        len(values),
    )
    return policy


def clear_gateway_env():
    """Restore the exact environment which preceded proxy injection."""
    global _environment_snapshot
    with _environment_lock:
        if _environment_snapshot is None:
            return
        items = _environment_snapshot.items()
        if os.name == "nt":
            # Windows os.environ is case-insensitive — ALL_PROXY and all_proxy
            # are the same key.  Process lowercase variants first so uppercase
            # restorations overwrite them instead of being popped away.
            items = sorted(items, key=lambda kv: kv[0].isupper())
        for key, previous in items:
            if previous is _MISSING:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(previous)
        _environment_snapshot = None
    logger.info("Gateway Tor environment restored")


def block_gateway_env():
    """Fail closed before recovery; never expose clients to a direct fallback.

    Sets all proxy vars to a dead SOCKS5 endpoint and signals TOR_HEALTH=0
    so downstream clients know Tor is unavailable.
    """
    blocked = "socks5://127.0.0.1:1"
    for name in PROXY_ENV_VARS:
        os.environ[name] = blocked
    os.environ["TOR_ENABLED"] = "0"
    os.environ["TOR_HEALTH"] = "0"
    logger.warning("Gateway Tor blocked — all proxy vars set to dead endpoint")


def create_httpx_client(*, policy: ProxyPolicy, asynchronous: bool = False, **kwargs):
    """Create an HTTPX client with a verified explicit SOCKS transport.

    ``trust_env=False`` prevents a later environment change from altering the
    immutable routing decision.  Callers must use this factory in strict mode;
    clients with no explicit, testable proxy hook are unsupported.
    """
    import httpx

    transport_type = httpx.AsyncHTTPTransport if asynchronous else httpx.HTTPTransport
    client_type = httpx.AsyncClient if asynchronous else httpx.Client
    kwargs.pop("proxy", None)
    kwargs.pop("transport", None)
    return client_type(
        transport=transport_type(proxy=policy.url), trust_env=False, **kwargs
    )


def require_verified_proxy_clients(policy: ProxyPolicy, *client_names: str) -> None:
    """Fail closed for clients for which this module has no explicit transport."""
    if not policy.strict:
        return
    unsupported = sorted(set(client_names) - {"httpx"})
    if unsupported:
        raise ProxyPolicyError(
            "strict mode has no verified explicit proxy integration for: "
            + ", ".join(unsupported)
        )


def skip_llm_proxy():
    """Request a Tor bypass from integrations that create LLM clients.

    Call this when LLM providers block Tor exit nodes (403/429 errors).
    This deliberately leaves the process-wide proxy variables untouched:
    platform adapters, web tools, and subprocesses rely on those variables
    for Tor routing.  The LLM integration must interpret TOR_SKIP_LLM and
    disable environment-proxy inheritance only for its own HTTP client.

    Only meaningful when TOR_ENABLED=1. Has no effect otherwise.
    """
    if os.environ.get("TOR_ENABLED", "").lower() not in ("1", "true", "yes"):
        return
    if _strict_mode(os.environ):
        raise ProxyPolicyError("LLM proxy bypass is unsupported in strict mode")
    os.environ["TOR_SKIP_LLM"] = "1"
    logger.warning(
        "TOR_SKIP_LLM=1 — requesting a Tor bypass for LLM clients; "
        "gateway-wide proxy variables remain enabled."
    )


def is_llm_skipped() -> bool:
    return os.environ.get("TOR_SKIP_LLM", "").lower() in ("1", "true", "yes")


def _dotenv_assignment(line: str) -> Optional[str]:
    """Return the key for a simple dotenv assignment, if present."""
    candidate = line.lstrip()
    if candidate.startswith("export "):
        candidate = candidate[7:].lstrip()
    key, separator, _ = candidate.partition("=")
    return key.strip() if separator else None


def write_gateway_env_file(
    socks_port: int = DEFAULT_SOCKS_PORT,
    env_path: Optional[Path] = None,
    healthy: bool = True,
):
    """Persist proxy vars in the environment file loaded by Hermes.

    Args:
        socks_port: SOCKS5 port (default: 9050)
        env_path: Path to .env file (default: ~/.hermes/.env)
        healthy: If False, TOR_ENABLED and TOR_HEALTH are set to 0
    """
    if env_path is None:
        env_path = Path.home() / ".hermes" / ".env"

    proxy_url = f"socks5://127.0.0.1:{socks_port}"

    tor_vars = _policy_environment(
        ProxyPolicy(proxy_url, strict=False), environment={}
    )
    tor_vars["TOR_HEALTH"] = "1" if healthy else "0"
    if not healthy:
        tor_vars["TOR_ENABLED"] = "0"
    # Preserve credentials, comments, quoting, and ordering byte-for-byte. Only
    # replace assignments owned by this integration, then append fresh values.
    with private_lock(env_path):
        if env_path.is_symlink():
            raise OSError(f"refusing symbolic link: {env_path}")
        existing_lines = (
            env_path.read_text().splitlines(keepends=True) if env_path.exists() else []
        )
        retained = [
            line for line in existing_lines if _dotenv_assignment(line) not in tor_vars
        ]
        if retained and not retained[-1].endswith(("\n", "\r")):
            retained[-1] += "\n"
        content = "".join(retained) + "".join(
            f"{key}={value}\n" for key, value in sorted(tor_vars.items())
        )
        atomic_private_write(env_path, content)
    logger.info("Gateway Tor config written to %s (%d vars)", env_path, len(tor_vars))


def remove_gateway_env_file(env_path: Optional[Path] = None):
    """Remove Tor-owned settings from the Hermes environment file."""
    if env_path is None:
        env_path = Path.home() / ".hermes" / ".env"

    with private_lock(env_path):
        if env_path.is_symlink():
            raise OSError(f"refusing symbolic link: {env_path}")
        if env_path.exists():
            managed = set(
                _policy_environment(
                    ProxyPolicy(
                        f"socks5://127.0.0.1:{DEFAULT_SOCKS_PORT}", strict=False
                    ),
                    environment={},
                )
            ) | {"TOR_HEALTH"}
            content = "".join(
                line
                for line in env_path.read_text().splitlines(keepends=True)
                if _dotenv_assignment(line) not in managed
            )
            atomic_private_write(env_path, content)
    logger.info("Gateway Tor config removed from %s", env_path)


# ═══════════════════════════════════════════════════════════════
# Self-Healing Tor Watchdog
# ═══════════════════════════════════════════════════════════════

class TorWatchdog:
    """Background thread that monitors Tor health and auto-restarts on failure.

    Self-healing: if Tor dies (process crash, OOM kill, port conflict),
    the watchdog detects it, kills any stale state, restarts the daemon,
    and re-injects proxy env vars. Gateway platform adapters will pick up
    the new connection on their next reconnect cycle.

    Circuit rotation: periodically sends NEWNYM to get fresh Tor circuits,
    preventing long-lived circuit fingerprinting.
    """

    def __init__(
        self,
        manager: TorManager,
        check_interval: float = 15.0,
        circuit_rotate_interval: float = 600.0,  # 10 minutes
        max_restart_attempts: int = 5,
        restart_backoff: float = 10.0,
    ):
        self._mgr = manager
        self._check_interval = check_interval
        self._circuit_rotate_interval = circuit_rotate_interval
        self._max_restart_attempts = max_restart_attempts
        self._restart_backoff = restart_backoff

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._restart_count = 0
        self._last_restart_time: float = 0
        self._last_circuit_rotation: float = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def start(self):
        """Start the watchdog background thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._last_circuit_rotation = time.time()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="tor-watchdog")
        self._thread.start()
        logger.info(
            "Tor watchdog started (health every %.0fs, circuit rotate every %.0fs)",
            self._check_interval, self._circuit_rotate_interval,
        )

    def stop(self):
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Tor watchdog stopped (restarts: %d)", self._restart_count)

    def _watchdog_loop(self):
        """Main loop: check health, rotate circuits, restart on failure."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._check_interval)
            if self._stop_event.is_set():
                break

            try:
                self._check_and_heal()
            except Exception:
                logger.exception("Watchdog check failed — will retry")

    def _check_and_heal(self):
        """Check Tor health. Restart if dead. Rotate circuit if due."""
        status = self._mgr.status()

        if status.state.name == "RUNNING" and status.healthy:
            self._restart_count = 0  # Reset counter on stable state

            # Circuit rotation
            now = time.time()
            if now - self._last_circuit_rotation > self._circuit_rotate_interval:
                self._rotate_circuit()
                self._last_circuit_rotation = now

        elif status.state.name == "ERROR" or not status.healthy:
            logger.warning(
                "Tor layered health check failed (state=%s, process=%s, socks=%s, bootstrap=%s, route=%s)",
                status.state.name, status.process_healthy, status.socks_healthy,
                status.bootstrap_complete, status.external_route_verified,
            )
            self._restart_tor()

        elif status.state.name == "STOPPED":
            logger.warning("Tor daemon stopped unexpectedly — restarting")
            self._restart_tor()

    def _restart_tor(self):
        """Restart Tor daemon with exponential backoff."""
        # Block new connections before stopping Tor or waiting through backoff.
        block_gateway_env()
        write_gateway_env_file(1, healthy=False)
        if self._restart_count >= self._max_restart_attempts:
            logger.error(
                "Tor restart limit reached (%d/%d) — watchdog giving up. "
                "Manual intervention required.",
                self._restart_count, self._max_restart_attempts,
            )
            return

        self._restart_count += 1
        delay = self._restart_backoff * (2 ** (self._restart_count - 1))
        logger.warning(
            "Restarting Tor (attempt %d/%d, delay %.0fs)...",
            self._restart_count, self._max_restart_attempts, delay,
        )

        # Stop any stale daemon
        try:
            self._mgr.stop()
        except Exception:
            pass

        # Small delay before restart
        time.sleep(delay)

        # Restart
        try:
            status = self._mgr.start(timeout=60)
            status = self._mgr.status(verify_route=True)
            if status.state.name == "RUNNING" and status.healthy:
                logger.info("Tor restarted successfully (attempt %d)", self._restart_count)
                self._restart_count = 0

                # Re-inject env vars so new connections pick up the fresh proxy
                inject_gateway_env(self._mgr.socks_port)
                write_gateway_env_file(self._mgr.socks_port)

                self._last_restart_time = time.time()
            else:
                logger.error("Tor restart failed: %s", status.error)
        except Exception:
            logger.exception("Tor restart raised exception")

    def _rotate_circuit(self):
        """Request a new Tor circuit (fresh exit node).

        Uses daemon's cookie-authenticated NEWNYM via ControlPort.
        Falls back to daemon restart for circuit rotation.
        """
        daemon = self._mgr._daemon
        if daemon and daemon.signal_newnym():
            logger.info("Tor circuit rotated via authenticated NEWNYM signal")
            return
        logger.debug("Authenticated NEWNYM failed, restarting daemon for fresh circuit")

        # Fallback: restart daemon for fresh circuit
        logger.info("Restarting Tor daemon for fresh circuit...")
        self._restart_tor()


def start_tor_for_gateway(
    socks_port: int = DEFAULT_SOCKS_PORT,
    bootstrap_timeout: float = 60.0,
    write_env: bool = True,
    *,
    hermes_root=None,
    runtime_probes=None,
) -> TorManager:
    """Start Tor and inject gateway-wide proxy environment.

    This is the primary entry point for gateway integration.
    Call this BEFORE starting the Hermes gateway.

    Args:
        socks_port: SOCKS5 port (default: 9050)
        bootstrap_timeout: Max seconds to wait for Tor bootstrap
        write_env: If True, persist ALL_PROXY to ~/.hermes/.env
        hermes_root: Hermes checkout to verify (defaults to automatic discovery)
        runtime_probes: Mapping of control IDs to zero-argument verification
            callables. Strict mode requires probes for controls that cannot be
            established from the installed files alone.

    Returns:
        TorManager instance (call .stop() to shut down)

    Raises:
        TorDaemonError: If Tor fails to bootstrap
    """
    # Freeze and validate routing before importing TorManager: its downloader
    # is the first module in this path which imports an HTTP network client.
    policy = establish_proxy_policy(socks_port)
    if policy.strict and not is_tor_installed():
        raise ProxyPolicyError(
            "strict mode requires a preinstalled Tor binary; run "
            "`darkloom download` before enabling TOR_STRICT_MODE"
        )

    # Environment variables alone are not a verifiable transport. None of the
    # gateway platform adapters are constructed in this package, so strict mode
    # must reject them until each adapter is wired to an explicit proxy client.
    require_verified_proxy_clients(
        policy, *(name.removesuffix("_PROXY").lower() for name in _PLATFORM_PROXY_NAMES)
    )

    # Validate Hermes integrations before Tor bootstrap.
    # Strict mode fails closed before Hermes can establish any connections.
    from darkloom.hardening import verify_compatibility

    compatibility = verify_compatibility(
        hermes_root, strict=None, runtime_probes=runtime_probes)
    for result in compatibility:
        logger.info("Hermes control %s: %s (%s)", result.control.id,
                    result.status.value, result.evidence.value)

    from darkloom.manager import TorManager

    mgr = TorManager(auto_download=not policy.strict, socks_port=socks_port)

    # Load bridges if available
    bridge_count = mgr.load_bridges()
    if bridge_count == 0:
        logger.warning(
            "No bridges configured — Tor will use public relays. "
            "Get bridges from @GetBridgesBot on Telegram and save to %s",
            BRIDGES_PATH,
        )

    # Start Tor
    logger.info("Starting Tor for gateway (timeout=%ds)...", bootstrap_timeout)
    status = mgr.start(timeout=bootstrap_timeout)

    if status.state.name != "RUNNING":
        raise RuntimeError(f"Tor failed to start: {status.error}")

    # No client is enabled and no persistent proxy is written until all four
    # layers (process, SOCKS, bootstrap, route) have independently passed.
    status = mgr.status(verify_route=True)
    if not status.healthy:
        block_gateway_env()
        mgr.stop()
        raise RuntimeError(f"Tor verification failed: {status.verification_error or status}")

    # Inject environment
    inject_gateway_env(socks_port, policy=policy)

    # Persist to .env for gateway restarts
    if write_env:
        write_gateway_env_file(socks_port)

    logger.info(
        "Tor ready for gateway — SOCKS5 %s, bootstrap %s%%, route verified %s, bridges %d, uptime %.1fs",
        status.socks_proxy_url,
        status.bootstrap_percent,
        status.external_route_verified,
        status.bridge_count,
        status.uptime_seconds or 0,
    )

    # Start self-healing watchdog
    watchdog = TorWatchdog(
        manager=mgr,
        check_interval=15.0,           # health check every 15s
        circuit_rotate_interval=600.0,  # fresh circuit every 10 min
        max_restart_attempts=5,
        restart_backoff=10.0,
    )
    watchdog.start()

    # Attach watchdog to manager so caller can stop it
    mgr._watchdog = watchdog

    return mgr


def _is_proxy_aware_gateway_command(command: list[str]) -> bool:
    """Return whether *command* is the installed Hermes gateway launcher.

    Arbitrary executables may ignore proxy environment variables, so strict
    mode only authorizes the known Hermes gateway entry point.
    """
    if len(command) < 3 or command[1:3] != ["gateway", "run"]:
        return False
    executable = shutil.which(command[0])
    if executable is None:
        return False
    try:
        launcher = Path(executable).read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeError):
        return False
    return "hermes_cli" in launcher and "import main" in launcher


# ── CLI entry point ────────────────────────────────────────────

def main():
    """Pre-start wrapper: start Tor, then exec the gateway.

    Usage:
        python -m darkloom.gateway -- hermes gateway run
        python -m darkloom.gateway --timeout 90 -- hermes gateway run

    The -- separator divides darkloom flags from gateway flags.
    Everything after -- is passed verbatim to the gateway process.
    """
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(
        description="Start Tor, then launch Hermes gateway with ALL_PROXY set"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_SOCKS_PORT,
        help=f"SOCKS5 port (default: {DEFAULT_SOCKS_PORT})",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Tor bootstrap timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-env-file", action="store_true",
        help="Skip writing ALL_PROXY to ~/.hermes/.env",
    )
    parser.add_argument(
        "gateway_args", nargs=argparse.REMAINDER,
        help="Arguments to pass to the gateway (after -- separator)",
    )

    args = parser.parse_args()

    # Strip the leading '--' separator if present
    gateway_cmd = args.gateway_args
    if gateway_cmd and gateway_cmd[0] == "--":
        gateway_cmd = gateway_cmd[1:]

    if not gateway_cmd:
        print("Usage: python -m darkloom.gateway -- hermes gateway run", file=sys.stderr)
        print("       python -m darkloom.gateway --timeout 90 -- hermes gateway run", file=sys.stderr)
        sys.exit(1)

    # Start Tor
    print(f"[darkloom] Starting Tor daemon (port {args.port}, timeout {args.timeout}s)...")
    try:
        mgr = start_tor_for_gateway(
            socks_port=args.port,
            bootstrap_timeout=args.timeout,
            write_env=not args.no_env_file,
        )
    except Exception as e:
        print(f"[darkloom] FATAL: Tor failed to start: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[darkloom] Tor running — SOCKS5 on 127.0.0.1:{args.port}")
    print(f"[darkloom] ALL_PROXY injected — all gateway platforms will route through Tor")
    print(f"[darkloom] Self-healing watchdog active (health every 15s, circuit rotate every 10min)")
    print("[darkloom] Launching gateway (command arguments redacted)")
    print()

    # Exec the gateway
    try:
        authorize_subprocess(proxy_aware=_is_proxy_aware_gateway_command(gateway_cmd))
        result = subprocess.run(gateway_cmd)
        sys.exit(result.returncode)
    finally:
        print("[darkloom] Gateway exited — stopping Tor daemon...")
        # Stop watchdog first
        watchdog = getattr(mgr, '_watchdog', None)
        if watchdog:
            watchdog.stop()
        mgr.stop()
        clear_gateway_env()
        print("[darkloom] Tor stopped.")


if __name__ == "__main__":
    main()
