"""Strict parsing and canonical serialization of Tor bridge lines.

Bridges are user-provided. Sources:
  1. Telegram: @GetBridgesBot (send /bridges)
  2. Web: https://bridges.torproject.org/bridges?transport=obfs4
  3. Email: bridges@torproject.org (from Gmail/Riseup, body: "get transport obfs4")

Bridges are stored one-per-line in ~/.hermes/tor/bridges.txt.
"""
import ipaddress
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from darkloom.secure_files import atomic_private_write, private_lock, secure_read

logger = logging.getLogger(__name__)

MAX_BRIDGE_LINE_LENGTH = 4096
_FINGERPRINT_RE = re.compile(r"[0-9A-Fa-f]{40}\Z", re.ASCII)
_OPTION_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_-]*)=(?P<value>[A-Za-z0-9._~:/?&=+,%@-]+)\Z", re.ASCII)
_OBFS4_CERT_RE = re.compile(r"[A-Za-z0-9+/=_-]+\Z", re.ASCII)

# BridgeDB's obfs4 result format: one valid bridge per line.
# fullmatch ensures a valid prefix cannot disguise trailing junk.
OBFS4_RESULT_RE = re.compile(
    r"obfs4\s+[\d.]+:(?:[1-9]\d{0,4})\s+[A-Fa-f0-9]{40}"
    r"\s+cert=\S+\s+iat-mode=[01]"
)


@dataclass(frozen=True)
class Bridge:
    """A validated bridge.  Only structured, encoder-safe data is retained."""

    transport: str
    address: str
    fingerprint: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.transport not in {"vanilla", "obfs4", "snowflake"}:
            raise ValueError("unsupported bridge transport")
        if _address(self.address) != self.address or _FINGERPRINT_RE.fullmatch(self.fingerprint) is None:
            raise ValueError("invalid bridge address or fingerprint")
        if any(
            _OPTION_RE.fullmatch(f"{key}={value}") is None
            for key, value in self.parameters
        ):
            raise ValueError("invalid bridge parameter")
        values = dict(self.parameters)
        if len(values) != len(self.parameters):
            raise ValueError("duplicate bridge parameter")
        if self.transport == "vanilla" and values:
            raise ValueError("vanilla bridges do not accept parameters")
        if self.transport == "obfs4" and (
            set(values) != {"cert", "iat-mode"}
            or _OBFS4_CERT_RE.fullmatch(values.get("cert", "")) is None
            or values.get("iat-mode") not in {"0", "1", "2"}
        ):
            raise ValueError("invalid obfs4 parameters")

    @property
    def line(self) -> str:
        prefix = "" if self.transport == "vanilla" else f"{self.transport} "
        options = "".join(f" {key}={value}" for key, value in self.parameters)
        return f"{prefix}{self.address} {self.fingerprint}{options}"


def _safe_input(line: object) -> str | None:
    if not isinstance(line, str) or not line or len(line) > MAX_BRIDGE_LINE_LENGTH:
        return None
    # Accept only ordinary ASCII spaces as separators.  This rejects CR/LF/NUL,
    # tabs, Unicode line/paragraph separators, and other control characters.
    if line != line.strip(" ") or any(
        ch != " " and (ch.isspace() or unicodedata.category(ch).startswith("C"))
        for ch in line
    ):
        return None
    if "#" in line or "  " in line:
        return None
    return line


def _address(value: str) -> str | None:
    try:
        if value.startswith("["):
            end = value.index("]")
            if end + 1 >= len(value) or value[end + 1] != ":":
                return None
            host, port_text = value[1:end], value[end + 2 :]
        else:
            host, port_text = value.rsplit(":", 1)
        ip = ipaddress.ip_address(host)
        port = int(port_text) if port_text.isascii() and port_text.isdigit() else 0
        if not 1 <= port <= 65535 or port_text != str(port):
            return None
        rendered_host = f"[{ip.compressed}]" if ip.version == 6 else ip.compressed
        return f"{rendered_host}:{port}"
    except (ValueError, IndexError):
        return None


def parse_bridge_line(line: str) -> Optional[Bridge]:
    """Parse an entire supported bridge line, returning ``None`` on any error."""
    safe = _safe_input(line)
    if safe is None:
        return None
    tokens = safe.split(" ")
    transport = tokens[0] if tokens[0] in {"obfs4", "snowflake"} else "vanilla"
    offset = 1 if transport != "vanilla" else 0
    if len(tokens) < offset + 2:
        return None
    address = _address(tokens[offset])
    fingerprint = tokens[offset + 1]
    if address is None or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
        return None

    parameters: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in tokens[offset + 2 :]:
        match = _OPTION_RE.fullmatch(token)
        if match is None or match["key"] in seen:
            return None
        seen.add(match["key"])
        parameters.append((match["key"], match["value"]))

    if transport == "vanilla" and parameters:
        return None
    if transport == "obfs4":
        values = dict(parameters)
        if set(values) != {"cert", "iat-mode"}:
            return None
        if not values["cert"] or _OBFS4_CERT_RE.fullmatch(values["cert"]) is None:
            return None
        if values["iat-mode"] not in {"0", "1", "2"}:
            return None
        parameters = [("cert", values["cert"]), ("iat-mode", values["iat-mode"])]
    # Snowflake's configuration evolves, but every option is still a single,
    # validated key/value token and is serialized from its parsed components.
    return Bridge(transport, address, fingerprint.upper(), tuple(parameters))


def validate_bridge(line: str) -> bool:
    return parse_bridge_line(line) is not None


def parse_bridge_set(text: str, *, transport: str | None = None) -> list[Bridge]:
    """Parse an all-or-nothing set of bridge lines from an external source.

    Unlike :func:`parse_bridge_line`, which is intentionally tolerant when
    reading a user-managed file, this rejects comments, markup, unknown
    transports, and mixed malformed content.  Callers can therefore validate
    an entire response before replacing a known-good configuration.
    """
    lines = text.splitlines()
    if not lines:
        raise ValueError("bridge result is empty")

    bridges: list[Bridge] = []
    for line in lines:
        if not line.strip():
            continue
        bridge = parse_bridge_line(line)
        if (
            bridge is None
            or bridge.transport == "unknown"
            or (transport is not None and bridge.transport != transport)
        ):
            raise ValueError("bridge result contains an invalid line")
        if bridge.transport == "obfs4" and not OBFS4_RESULT_RE.fullmatch(bridge.line):
            raise ValueError("bridge result contains a malformed obfs4 line")
        if bridge.transport == "obfs4":
            host, port = bridge.address.rsplit(":", 1)
            try:
                ipaddress.IPv4Address(host)
            except ipaddress.AddressValueError as exc:
                raise ValueError("bridge result contains an invalid address") from exc
            if not 1 <= int(port) <= 65535:
                raise ValueError("bridge result contains an invalid port")
        bridges.append(bridge)

    if not bridges:
        raise ValueError("bridge result contains no bridges")
    return bridges


def load_bridges_from_file(path: Path) -> list[Bridge]:
    if not path.exists():
        logger.warning("No bridges file at %s — Tor will use public relays", path)
        return []

    from darkloom.secure_files import private_lock, secure_read

    with private_lock(path):
        content = secure_read(path)

    bridges: list[Bridge] = []
    for line in content.splitlines():
        if not line or line.startswith("#"):
            continue
        bridge = parse_bridge_line(line)
        if bridge is not None:
            bridges.append(bridge)
        else:
            logger.warning("Ignoring invalid bridge entry in %s", path)
    return bridges


def save_bridges_to_file(path: Path, bridge_lines: list[str], append: bool = False):
    """Validate and canonically encode bridge lines before writing them.

    Args:
        path: File path.
        bridge_lines: List of full bridge lines.
        append: If True, append to existing file instead of overwriting.
    """
    from darkloom.secure_files import private_lock, secure_read, atomic_private_write

    parsed = [parse_bridge_line(line) for line in bridge_lines]
    if any(bridge is None for bridge in parsed):
        raise ValueError("invalid bridge line")

    with private_lock(path):
        previous = secure_read(path) if append and path.exists() else ""
        content = previous + "".join(
            bridge.line + "\n" for bridge in parsed  # type: ignore[union-attr]
        )
        atomic_private_write(path, content)

    logger.info("Wrote %d bridges to %s (append=%s)", len(parsed), path, append)


def format_bridges_for_torrc(bridges: list[Bridge]) -> list[str]:
    return [f"Bridge {bridge.line}" for bridge in bridges]
