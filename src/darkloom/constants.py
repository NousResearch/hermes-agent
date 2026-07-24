"""Platform-specific constants for darkloom.

All URLs and paths verified against live Tor Expert Bundle 15.0.19
(downloaded and inspected 2026-07-22).
"""
import platform
import sys
from pathlib import Path

# ── Tor Expert Bundle ──────────────────────────────────────────
# Pinned version for deterministic behavior.
# Verified bundles: windows-x86_64 (22MB), linux-x86_64 (32MB).
TOR_VERSION = "15.0.19"
TOR_BASE_URL = "https://archive.torproject.org/tor-package-archive/torbrowser"

TOR_PLATFORM_MAP = {
    "win32": "windows",
    "linux": "linux",
}

# Tor Browser release signing key.  The key itself is shipped in
# ``darkloom/keys/tor-browser-developers.asc``; accepting a key obtained at
# download time would merely move trust to the network.
TOR_RELEASE_SIGNING_FINGERPRINTS = frozenset({
    "EF6E286DDA85EA2A4BA7DE684E2C6E8793298290",
})
TOR_RELEASE_SIGNING_KEY = Path(__file__).with_name("keys") / "tor-browser-developers.asc"

# Every artifact we advertise has an explicitly associated piece of signed
# metadata. Tor publishes detached OpenPGP signatures alongside expert bundles.
TOR_PLATFORM_ARTIFACTS = {
    ("windows", "x86_64"): "tor-expert-bundle-windows-x86_64-{version}.tar.gz",
    ("linux", "x86_64"): "tor-expert-bundle-linux-x86_64-{version}.tar.gz",
    ("linux", "aarch64"): "tor-expert-bundle-linux-aarch64-{version}.tar.gz",
}
TOR_PLATFORM_SIGNATURES = {
    platform_artifact: artifact + ".asc"
    for platform_artifact, artifact in TOR_PLATFORM_ARTIFACTS.items()
}

TOR_ARCH_MAP = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x86": "i686",
    "i686": "i686",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}

# ── Default ports ──────────────────────────────────────────────
DEFAULT_SOCKS_PORT = 9050
DEFAULT_CONTROL_PORT = 9051

# ── Filesystem paths ───────────────────────────────────────────
# Everything lives under ~/.hermes/tor/ to stay inside the
# Hermes home directory — no system-wide state.
DATA_DIR = Path.home() / ".hermes" / "tor"
TOR_BINARY_DIR = DATA_DIR / "tor-bin"
TOR_DATA_DIR = DATA_DIR / "tor-data"
BRIDGES_PATH = DATA_DIR / "bridges.txt"

# ── Platform detection ─────────────────────────────────────────
CURRENT_PLATFORM = sys.platform  # "win32" or "linux"
CURRENT_ARCH = platform.machine().lower()


def get_tor_binary_path() -> Path:
    """Platform-appropriate tor binary path.

    Windows: tor-bin/Tor/tor.exe  (uppercase Tor from the tarball)
    Linux:   tor-bin/tor/tor      (lowercase tor from the tarball)

    Note: The 15.0.19 Windows tarball extracts to a 'Tor' directory
    (capital T), while Linux extracts to 'tor'. We handle both.
    """
    if CURRENT_PLATFORM == "win32":
        # The Windows tarball uses "Tor" (capital T)
        return TOR_BINARY_DIR / "Tor" / "tor.exe"
    return TOR_BINARY_DIR / "tor" / "tor"


def get_lyrebird_path(tor_binary_dir: Path | None = None) -> Path:
    """Path to lyrebird (obfs4proxy successor) bundled in the Tor Expert Bundle.

    lyrebird handles: meek_lite, obfs2, obfs3, obfs4, scramblesuit,
    snowflake, webtunnel. It is the Tor Project's unified pluggable
    transport binary — no separate obfs4proxy download needed.
    """
    base = tor_binary_dir or TOR_BINARY_DIR
    # Match the tarball's directory case
    if CURRENT_PLATFORM == "win32":
        exe_name = "lyrebird.exe"
        # The Windows tarball uses "Tor" (capital T)
        return base / "Tor" / "pluggable_transports" / exe_name
    return base / "tor" / "pluggable_transports" / "lyrebird"


def get_geoip_paths(tor_binary_dir: Path | None = None) -> tuple[Path, Path]:
    """Return (geoip_path, geoip6_path) for the bundled GeoIP databases.

    Tor needs these for country-based path selection.
    Both are bundled in the Expert Bundle.
    """
    base = tor_binary_dir or TOR_BINARY_DIR
    return base / "data" / "geoip", base / "data" / "geoip6"


def is_tor_installed() -> bool:
    """Check if Tor binary exists at the expected path."""
    return get_tor_binary_path().exists()


def get_download_url() -> str:
    """Verified download URL for the current platform/arch.

    URL pattern (verified 2026-07-22):
      {TOR_BASE_URL}/{version}/tor-expert-bundle-{platform}-{arch}-{version}.tar.gz

    Example:
      https://archive.torproject.org/tor-package-archive/torbrowser/15.0.19/
        tor-expert-bundle-windows-x86_64-15.0.19.tar.gz
    """
    plat = TOR_PLATFORM_MAP.get(CURRENT_PLATFORM)
    arch = TOR_ARCH_MAP.get(CURRENT_ARCH)
    if not plat or not arch:
        raise RuntimeError(
            f"Unsupported platform: {CURRENT_PLATFORM}/{CURRENT_ARCH}. "
            f"Available: windows/x86_64, linux/x86_64, linux/aarch64"
        )
    template = TOR_PLATFORM_ARTIFACTS.get((plat, arch))
    if template is None:
        raise RuntimeError(f"No signed Tor artifact for {plat}/{arch}")
    filename = template.format(version=TOR_VERSION)
    return f"{TOR_BASE_URL}/{TOR_VERSION}/{filename}"


def get_signature_url() -> str:
    """Return the detached OpenPGP signature location for this artifact."""
    plat = TOR_PLATFORM_MAP.get(CURRENT_PLATFORM)
    arch = TOR_ARCH_MAP.get(CURRENT_ARCH)
    template = TOR_PLATFORM_SIGNATURES.get((plat, arch))
    if template is None:
        raise RuntimeError(f"No signed Tor metadata for {plat}/{arch}")
    filename = template.format(version=TOR_VERSION)
    return f"{TOR_BASE_URL}/{TOR_VERSION}/{filename}"


# ── Bridge sourcing ────────────────────────────────────────────
# Bridges must be USER-PROVIDED. The built-in bridges in the Tor
# Expert Bundle's pt_config.json are shared across millions of
# Tor Browser users and are frequently blocked or rate-limited.
#
# Users get bridges from:
#   1. Telegram: @GetBridgesBot (send /bridges)
#   2. Web:      https://bridges.torproject.org/bridges?transport=obfs4
#   3. Email:    bridges@torproject.org (from Gmail/Riseup, body: "get transport obfs4")
#
# Bridges are stored one-per-line in ~/.hermes/tor/bridges.txt
# and added via the `tor_add_bridge` MCP tool or by editing the file.
#
# Without user-provided bridges, Tor will attempt to connect via
# public relays — which works but defeats the bridge-based
# censorship-circumvention purpose.
