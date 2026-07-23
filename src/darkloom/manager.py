"""Unified TorManager — the main entry point for darkloom.

Ties together download, daemon, bridges, and verification into a
single API with a state machine.

Usage:
    from darkloom.manager import TorManager

    mgr = TorManager(auto_download=True)
    mgr.load_bridges()  # from ~/.hermes/tor/bridges.txt
    mgr.start()
    print(mgr.socks_proxy_url)  # socks5://127.0.0.1:9050
    mgr.verify()  # check.torproject.org
    mgr.stop()
"""
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from darkloom.constants import (
    DATA_DIR,
    BRIDGES_PATH,
    DEFAULT_SOCKS_PORT,
    DEFAULT_CONTROL_PORT,
    is_tor_installed,
    get_tor_binary_path,
)
from darkloom.daemon import TorDaemon, TorDaemonError
from darkloom.bridges import (
    load_bridges_from_file,
    save_bridges_to_file,
    format_bridges_for_torrc,
    Bridge,
    parse_bridge_line,
)
from darkloom.downloader import (
    DownloadError,
    download_tor_binary,
    validate_installed_binary,
)
from darkloom.verifier import TorVerifier, VerificationResult
from darkloom.socks_support import require_socks_support

from darkloom.privacy import classify_error, get_logger, private_diagnostic

logger = get_logger(__name__)


class TorState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass
class TorStatus:
    """Snapshot of current Tor state."""

    state: TorState
    socks_proxy_url: str | None = None
    process_healthy: bool = False
    socks_healthy: bool = False
    bootstrap_percent: int | None = None
    bootstrap_complete: bool = False
    external_route_verified: bool = False
    bridge_count: int = 0
    exit_ip: str | None = None
    error: str | None = None
    error_code: str | None = None
    uptime_seconds: float | None = None
    verification_error: str | None = None

    @property
    def healthy(self) -> bool:
        return (self.process_healthy and self.socks_healthy and
                self.bootstrap_complete and self.external_route_verified)


@dataclass(frozen=True)
class AddBridgeResult:
    """Typed result from an attempted bridge addition."""

    added: bool
    total_bridges: int
    error: str | None = None


class TorManager:
    """Unified Tor lifecycle manager for Hermes agents.

    Handles the full lifecycle: download → configure bridges →
    start daemon → verify anonymity → stop.

    State machine:
        STOPPED → STARTING → RUNNING → STOPPING → STOPPED
                       ↓         ↓
                      ERROR ←────┘
    """

    _VALID_TRANSITIONS = {
        TorState.STOPPED: {TorState.STARTING},
        TorState.STARTING: {TorState.RUNNING, TorState.ERROR},
        TorState.RUNNING: {TorState.STOPPING, TorState.ERROR},
        TorState.STOPPING: {TorState.STOPPED, TorState.ERROR},
        TorState.ERROR: {TorState.STOPPED, TorState.STARTING},
    }

    def __init__(
        self,
        data_dir: Path | None = None,
        socks_port: int = DEFAULT_SOCKS_PORT,
        control_port: int = DEFAULT_CONTROL_PORT,
        bridges: list[str] | None = None,
        auto_download: bool = True,
        strict: bool = True,
    ):
        self.data_dir = data_dir or DATA_DIR
        self.socks_port = socks_port
        self.control_port = control_port
        self._bridges: list[str] = []
        for line in bridges or []:
            bridge = parse_bridge_line(line)
            if bridge is None:
                raise ValueError("invalid initial bridge line")
            self._bridges.append(bridge.line)
        self.auto_download = auto_download
        self.strict = strict

        self._daemon: Optional[TorDaemon] = None
        self._state = TorState.STOPPED
        self._verifier = TorVerifier(f"socks5://127.0.0.1:{socks_port}")
        self._start_time: Optional[float] = None

    # ── Public API ─────────────────────────────────────────────

    @property
    def state(self) -> TorState:
        return self._state

    @property
    def socks_proxy_url(self) -> str | None:
        if self._daemon and self._daemon.is_running:
            return self._daemon.socks_proxy_url
        return None

    def ensure_installed(self) -> Path:
        """Ensure Tor binary is available; download if needed.

        Returns path to tor binary.
        Raises TorDaemonError if download fails or auto_download is False.
        """
        if not is_tor_installed():
            if not self.auto_download:
                raise TorDaemonError(
                    "Tor not installed and auto_download=False. "
                    "Run: darkloom download  or set auto_download=True."
                )
            logger.info("Tor not installed — downloading...")
            return download_tor_binary()
        try:
            verified = validate_installed_binary(strict=self.strict)
        except DownloadError as exc:
            raise TorDaemonError(f"Tor installation failed authentication: {exc}") from exc
        if not verified:
            logger.warning("Using an unverified legacy Tor installation (strict=False)")
        return get_tor_binary_path()

    def load_bridges(self, path: Path | None = None) -> int:
        """Load bridges from file. Returns number of bridges loaded.

        Default path: ~/.hermes/tor/bridges.txt
        """
        bridge_path = path or BRIDGES_PATH
        bridges = load_bridges_from_file(bridge_path)
        self._bridges = [b.line for b in bridges]
        logger.info("Loaded %d bridges", len(self._bridges))
        return len(self._bridges)

    def add_bridge(self, bridge_line: str) -> AddBridgeResult:
        """Validate and persist a bridge before changing in-memory state."""
        bridge = parse_bridge_line(bridge_line)
        if bridge is None:
            return AddBridgeResult(False, len(self._bridges), "invalid bridge line")
        line = bridge.line
        if line in self._bridges:
            return AddBridgeResult(False, len(self._bridges))
        try:
            save_bridges_to_file(BRIDGES_PATH, [line], append=True)
        except OSError as exc:
            return AddBridgeResult(False, len(self._bridges), f"could not persist bridge: {exc}")
        self._bridges.append(line)
        return AddBridgeResult(True, len(self._bridges))

    def start(self, timeout: float = 60.0) -> TorStatus:
        """Start Tor daemon. Blocks until bootstrapped.

        If no bridges are configured, Tor will use public relays —
        which works but defeats bridge-based circumvention.
        """
        self._transition(TorState.STARTING)

        if not self._bridges:
            logger.warning(
                "No bridges configured. Tor will use public relays. "
                "Get bridges from @GetBridgesBot on Telegram and "
                "save them to %s, then restart.",
                BRIDGES_PATH,
            )

        try:
            # Validate the client-side transport before downloading or starting
            # Tor. This is a local check and cannot accidentally connect direct.
            require_socks_support(f"socks5://127.0.0.1:{self.socks_port}")
            tor_binary = self.ensure_installed()
            self._daemon = TorDaemon(
                tor_binary=tor_binary,
                data_dir=self.data_dir / "tor-data",
                bridges=self._bridges,
                socks_port=self.socks_port,
                control_port=self.control_port,
            )
            self._start_time = time.time()
            self._daemon.start(timeout=timeout)
            self._transition(TorState.RUNNING)
            return self.status()
        except Exception as e:
            self._transition(TorState.ERROR)
            private_diagnostic("manager.start", e)
            public = classify_error(e)
            return TorStatus(state=TorState.ERROR, error=public.message, error_code=public.code)

    def stop(self) -> TorStatus:
        """Stop Tor daemon gracefully."""
        self._transition(TorState.STOPPING)
        try:
            if self._daemon:
                self._daemon.stop()
        finally:
            self._transition(TorState.STOPPED)
            self._daemon = None
            self._start_time = None
        return self.status()

    def status(self, verify_route: bool = True) -> TorStatus:
        """Return independently measured process, SOCKS, control, and route state."""
        running = bool(self._daemon and self._daemon.is_running)
        process_healthy = self._daemon.process_health() if running else False
        socks_healthy = self._daemon.health_check() if process_healthy else False
        bootstrap, control_error = self._daemon.bootstrap_status() if process_healthy else (None, None)
        verification = self._verifier.verify() if verify_route and socks_healthy and bootstrap == 100 else None

        uptime = None
        if self._start_time and running:
            uptime = time.time() - self._start_time

        return TorStatus(
            state=self._state,
            socks_proxy_url=self.socks_proxy_url,
            process_healthy=process_healthy,
            socks_healthy=socks_healthy,
            bootstrap_percent=bootstrap,
            bootstrap_complete=bootstrap == 100,
            external_route_verified=bool(verification and verification.using_tor),
            exit_ip=verification.exit_ip if verification else None,
            bridge_count=len(self._bridges),
            uptime_seconds=uptime,
            verification_error=(verification.error if verification else control_error),
        )

    def verify(self) -> VerificationResult:
        """Verify traffic routes through Tor via check.torproject.org."""
        if not self.socks_proxy_url:
            return VerificationResult(False, error="Tor not running")
        return self._verifier.verify()

    def restart(self) -> TorStatus:
        """Stop then start Tor (e.g., after adding bridges)."""
        if self._state == TorState.RUNNING:
            self.stop()
        return self.start()

    # ── Context manager ────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False

    # ── Internals ──────────────────────────────────────────────

    def _transition(self, to: TorState):
        valid = self._VALID_TRANSITIONS.get(self._state, set())
        if to not in valid and self._state != to:
            logger.debug(
                "State transition %s → %s (valid from %s: %s)",
                self._state, to, self._state, valid,
            )
        self._state = to
