"""Terminal QR rendering + advertised-URL resolution (shared helpers).

The canonical home for the half-block terminal QR renderer that previously
lived (duplicated) in platform onboarding modules. Used by ``hermes dashboard
--qr`` to make the Web UI scannable from a phone, and importable by any
onboarding flow that needs a QR in the terminal.
"""

import logging
import socket

logger = logging.getLogger(__name__)


def ensure_qrcode_installed() -> bool:
    """Try to import qrcode; if missing, auto-install it via pip/uv."""
    try:
        import qrcode  # noqa: F401
        return True
    except ImportError:
        pass

    import subprocess

    from hermes_cli.tools_config import _pip_install

    try:
        result = _pip_install(["-q", "qrcode"], timeout=120)
        if result.returncode == 0:
            import qrcode  # noqa: F401,F811
            return True
    except (subprocess.SubprocessError, ImportError, OSError):
        pass
    return False


def render_qr_to_terminal(url: str) -> bool:
    """Render *url* as a compact QR code in the terminal.

    Returns True if the QR code was printed, False if the library is missing.
    """
    try:
        import qrcode
    except ImportError:
        return False

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Use half-block characters for compact rendering (2 rows per character)
    matrix = qr.get_matrix()
    rows = len(matrix)
    lines: list[str] = []

    TOP_HALF = "▀"      # ▀
    BOTTOM_HALF = "▄"   # ▄
    FULL_BLOCK = "█"    # █
    EMPTY = " "

    for r in range(0, rows, 2):
        line_chars: list[str] = []
        for c in range(len(matrix[r])):
            top = matrix[r][c]
            bottom = matrix[r + 1][c] if r + 1 < rows else False
            if top and bottom:
                line_chars.append(FULL_BLOCK)
            elif top:
                line_chars.append(TOP_HALF)
            elif bottom:
                line_chars.append(BOTTOM_HALF)
            else:
                line_chars.append(EMPTY)
        lines.append("    " + "".join(line_chars))

    print("\n".join(lines), flush=True)
    return True


def lan_ip() -> "str | None":
    """Best-effort LAN IP of this machine (no traffic is actually sent).

    The UDP connect trick: connecting a datagram socket selects the outbound
    interface without sending a packet, and getsockname() reveals its address.
    Returns None when the machine has no route (offline, airplane mode).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0)
            probe.connect(("10.254.254.254", 1))
            address = probe.getsockname()[0]
        return address if address and not address.startswith("127.") else None
    except OSError:
        return None


def resolve_advertised_url(host: str, port: int, public_url: str = "") -> str:
    """The URL another device should use to reach a server bound to host:port.

    ``public_url`` (a tunnel hostname) wins outright. A wildcard bind
    advertises the machine's LAN IP so the URL is scannable from a phone on
    the same network; a concrete bind advertises itself. Loopback binds are
    returned as-is — callers should warn that loopback is not phone-reachable.
    """
    if public_url:
        return public_url.rstrip("/")
    if host in ("0.0.0.0", "::", ""):
        address = lan_ip()
        if address:
            return f"http://{address}:{port}"
        logger.debug("wildcard bind but no LAN IP resolvable; advertising bind host")
    return f"http://{host}:{port}"
