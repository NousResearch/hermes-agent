"""Non-destructive bridge allocation for multiplexed secondary profiles."""
import socket
from contextlib import suppress
from pathlib import Path

# 3000 stays the default profile's historical port; secondaries allocate above it. 999 ports
# covers any realistic number of paired phones on one host.
SECONDARY_PORT_FIRST = 3001
SECONDARY_PORT_LAST = 3999


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _read_port_record(record: Path) -> int:
    try:
        return int(record.read_text(encoding="utf-8-sig").strip())
    except ValueError:
        raise ValueError(f"Bridge port record {record} is not a port number; delete it to allocate a new port") from None


def _sibling_claims(home: Path) -> set:
    """Ports other profiles' records claim. A sibling whose bridge is down leaves its port unbound, so a
    bind probe alone would hand it to this profile and the sibling would hit a conflict on its next start."""
    claims = set()
    for record in home.parent.glob("*/platforms/whatsapp/bridge_port"):
        if record.parents[2] != home:
            with suppress(OSError, ValueError):
                claims.add(_read_port_record(record))
    return claims


def secondary_bridge_port(home: Path, explicit) -> int:
    record = home / "platforms/whatsapp/bridge_port"
    if explicit is not None:
        port = int(explicit)
    elif record.exists():
        port = _read_port_record(record)
    else:
        claimed = _sibling_claims(home)
        port = next((p for p in range(SECONDARY_PORT_FIRST, SECONDARY_PORT_LAST + 1)
                     if p not in claimed and port_is_free(p)), None)
        if port is None:
            raise ValueError(f"No free bridge port in {SECONDARY_PORT_FIRST}..{SECONDARY_PORT_LAST}")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(str(port), encoding="utf-8")
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid bridge port {port}")
    return port


def standalone_bridge_port(home: Path, explicit) -> int:
    """Resolve a port for an out-of-process send without allocating one.

    Startup is responsible for creating the secondary record. A standalone
    sender may only consume an explicit value or an already persisted record;
    an unknown profile keeps the historical default port.
    """
    if explicit is not None:
        port = int(explicit)
    else:
        record = home / "platforms/whatsapp/bridge_port"
        if not record.exists():
            return 3000
        port = _read_port_record(record)
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid bridge port {port}")
    return port


def check_secondary_ownership(session: Path, port: int) -> str:
    """``"free"`` when the port is unbound, ``"ours"`` when it is bound and this profile's own
    ``bridge.pid`` (pid plus kernel start time, the same fingerprint the default path trusts) names
    a live process, so a bridge orphaned by a gateway crash can be adopted or reaped by identity.
    Anything else bound on the port belongs to someone else and is a fatal for this profile: no HTTP
    probe is safe before ownership is known, a healthy listener can still be another phone, and a
    secondary never signals a process it cannot prove is its own."""
    if port_is_free(port):
        return "free"
    from .adapter import _bridge_pid_is_ours, _read_bridge_pidfile
    pid, recorded_start = _read_bridge_pidfile(session)
    if pid is not None and _bridge_pid_is_ours(pid, recorded_start):
        return "ours"
    raise ValueError(f"Bridge port {port} is bound by a process this profile does not own")
