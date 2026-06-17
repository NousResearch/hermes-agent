"""iron-proxy (`ironsh/iron-proxy`) integration for credential-injecting egress control.

Why
---

Remote terminal sandboxes (Docker, Modal, SSH) currently see real upstream
API credentials.  A prompt-injected agent inside one of these sandboxes can
``cat ~/.config/openrouter/auth.json`` or ``printenv | grep -i key`` and
exfiltrate them.

iron-proxy is a TLS-intercepting egress firewall (Apache-2.0, Go binary, by
ironsh).  It sits between the sandbox and the internet, enforces a default-deny
allowlist on outbound hosts, and *swaps proxy tokens for real credentials*
on the way out.  The sandbox only ever holds opaque proxy tokens — leaking
them is useless, since they only work from behind the proxy.

Design summary
--------------

* The ``iron-proxy`` binary is auto-installed into ``<hermes_home>/bin/iron-proxy``
  on first use.  Hermes pins one upstream version (``_IRON_PROXY_VERSION``)
  and downloads the matching tar.gz from the official GitHub Releases page,
  verifying the SHA-256 against a source-pinned digest
  (``_IRON_PROXY_PINNED_SHA256``); the release's ``checksums.txt`` is only a
  defence-in-depth cross-check.

* A long-lived CA at ``<hermes_home>/proxy/ca.{crt,key}`` is generated on
  first ``hermes egress setup``.  Sandboxes trust this CA so iron-proxy can
  terminate TLS and rewrite headers.

* The proxy config lives at ``<hermes_home>/proxy/proxy.yaml``.  It enumerates
  the per-provider allowlists and the ``secrets`` transform that does the
  Authorization-header swap.

* Token mappings (proxy token -> real credential lookup) live alongside the
  config.  The real credential is **never** written to the config — iron-proxy
  reads it from its own environment via ``{type: env, var: NAME}``.  When
  Bitwarden Secrets Manager is configured, the real value is pulled there
  at proxy startup instead.

* The proxy runs as a managed subprocess (``hermes egress start``), pidfile
  at ``<hermes_home>/proxy/iron-proxy.pid``.  Daemon output (including
  per-request records on v0.39) goes to ``<hermes_home>/proxy/iron-proxy.log``;
  ``audit.log`` is pre-created but reserved for a future pin that supports
  ``log.audit_path``.

* Failures (binary missing, port collision, bad config) emit a one-line
  warning and do *not* block agent startup.  The Docker backend refuses to
  start a sandbox with the proxy enabled-but-down, with a clear error.

This module is intentionally subprocess-driven rather than depending on any
iron-proxy Python bindings — a single cross-platform binary is easier to
lazy-install than a wheels-with-extension dependency, and we keep maintenance
to a "bump the pinned version" loop.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import platform
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Pinned upstream version.  Bump in a follow-up PR — never auto-resolve "latest"
# because upstream YAML schema is allowed to change between releases and we
# want updates to be deliberate.
_IRON_PROXY_VERSION = "0.39.0"

_IRON_PROXY_RELEASE_BASE = (
    f"https://github.com/ironsh/iron-proxy/releases/download/v{_IRON_PROXY_VERSION}"
)
_IRON_PROXY_CHECKSUM_NAME = "checksums.txt"

# weklund: pin the expected per-platform SHA-256 directly in source, next to
# the version. The release ``checksums.txt`` is fetched from the same URL as
# the binary, so an attacker who controls the download path (corporate proxy,
# DNS hijack, poisoned CDN cache) could serve a malicious binary AND a matching
# checksums file, making that verification a no-op. Pinning the digest here
# means the only way a new hash reaches users is via a code-reviewed commit,
# regardless of the download channel. We still cross-check checksums.txt when
# present (defence-in-depth), but the source-pinned value is authoritative.
#
# Hashes for iron-proxy v0.39.0 (verified against upstream checksums.txt).
# When bumping _IRON_PROXY_VERSION, update every entry in the same commit.
_IRON_PROXY_PINNED_SHA256: dict[str, str] = {
    "iron-proxy_0.39.0_darwin_amd64.tar.gz": "9c278663d2e9c04fb3f6b1c18588fc2095974b42347c7885d10014835fc4b24a",
    "iron-proxy_0.39.0_darwin_arm64.tar.gz": "6aa2d3d78ba87bdfacaef8f8465ffd289f8d23ab4d3f433e8c0b2b77f887632a",
    "iron-proxy_0.39.0_linux_amd64.tar.gz": "0ab2e031074a01410caecaa2769c0b38f69cbdf4b6333bce498fe73919f6f9f6",
    "iron-proxy_0.39.0_linux_arm64.tar.gz": "1316cb4f20fcfba5042b06d07cd523ffa617de9de2999cd5f1f44474777c48ad",
}

# How long to wait for HTTP downloads and subprocess interactions, in seconds.
_DOWNLOAD_TIMEOUT = 120  # binary is ~16MB
_RUN_TIMEOUT = 30
_STARTUP_GRACE_SECONDS = 5

# Default listen ports.  HTTPS_PROXY semantics use a single CONNECT tunnel,
# so we expose only the tunnel listener for v1 — no need to put the sandbox
# DNS at the iron-proxy IP.  This greatly simplifies wiring.
_DEFAULT_TUNNEL_PORT = 9090

# Hosts allowed by default for AI inference traffic.  Anything else is 403'd.
_DEFAULT_ALLOWED_HOSTS: Tuple[str, ...] = (
    "openrouter.ai",
    "*.openrouter.ai",
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.x.ai",
    "api.mistral.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.deepseek.com",
    "inference.nousresearch.com",
)

# Provider env-var name -> upstream host (or list of hosts) on which the
# Authorization Bearer token should be swapped.  Only includes providers
# whose API uses a plain "Authorization: Bearer <key>" header — providers
# with custom auth (x-api-key, query params, signatures) get added as we
# write per-provider rules.
_BEARER_PROVIDERS: Dict[str, Tuple[str, ...]] = {
    "OPENROUTER_API_KEY": ("openrouter.ai", "*.openrouter.ai"),
    "OPENAI_API_KEY": ("api.openai.com",),
    "GROQ_API_KEY": ("api.groq.com",),
    "TOGETHER_API_KEY": ("api.together.xyz",),
    "DEEPSEEK_API_KEY": ("api.deepseek.com",),
    "MISTRAL_API_KEY": ("api.mistral.ai",),
    "XAI_API_KEY": ("api.x.ai",),
    "NOUS_API_KEY": ("inference.nousresearch.com",),
}


# Providers whose env-var names we recognize but whose API uses a non-bearer
# auth scheme (x-api-key, AAD/OAuth, SigV4, custom signatures).  When any of
# these env vars are present at proxy-start time AND
# ``proxy.fail_on_uncovered_providers`` is true (which is OFF by default),
# ``start_proxy`` refuses to start.  Without this list the sandbox would
# still hold real credentials for these providers and silently bypass the
# proxy.
#
# The default is False because many of these env vars (AWS_*,
# GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_API_KEY) are present on most
# developer laptops for reasons unrelated to LLM API access — defaulting to
# refuse-start would force everyone using terraform / gcloud / aws-cli
# alongside Hermes to either unset their cloud auth or set the flag in
# config.yaml.  The wizard surfaces uncovered providers at setup time and
# `hermes egress status` keeps them visible; operators who want hard
# enforcement opt in via ``proxy.fail_on_uncovered_providers: true``.
#
# Bare strings here are env-var names; the proxy doesn't try to wire them up,
# only flags their presence so the operator knows isolation is incomplete.
_NON_BEARER_PROVIDERS: Tuple[str, ...] = (
    # Anthropic native uses x-api-key, not Authorization: Bearer.
    "ANTHROPIC_API_KEY",
    # Azure OpenAI: api-key header + optional AAD bearer.
    "AZURE_OPENAI_API_KEY",
    # AWS Bedrock / SageMaker: SigV4-signed requests.
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    # GCP Vertex AI: OAuth bearer from gcloud SDK, not a static env key.
    "GOOGLE_APPLICATION_CREDENTIALS",
    # Google AI Studio (Gemini): x-goog-api-key OR query param.
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


# Tier of `_NON_BEARER_PROVIDERS` that's LLM-specific enough that any
# accidental sandbox bypass is a real isolation failure.  When
# ``fail_on_uncovered_providers`` is true, only env vars in this tier
# cause refuse-start; the rest are warn-only via `_NON_BEARER_PROVIDERS`.
# Splitting this avoids tripping every operator with `AWS_PROFILE` set
# for unrelated cloud work.
_LLM_SPECIFIC_NON_BEARER_PROVIDERS: Tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GEMINI_API_KEY",
)


# Default SSRF-protection deny list applied to the proxy's outbound traffic.
# Mirrors the public docs promise ("cloud metadata IPs are refused by default
# regardless of allowlist").  Tests / dev setups that need loopback can pass
# an explicit override (e.g. [] to disable, or a smaller subset).
_DEFAULT_UPSTREAM_DENY_CIDRS: Tuple[str, ...] = (
    "127.0.0.0/8",        # IPv4 loopback
    "::1/128",            # IPv6 loopback
    "169.254.0.0/16",     # IPv4 link-local incl. AWS/GCP/Azure IMDS
    "fe80::/10",          # IPv6 link-local
    "10.0.0.0/8",         # RFC1918
    "172.16.0.0/12",      # RFC1918
    "192.168.0.0/16",     # RFC1918
    "fc00::/7",           # IPv6 ULA
    # IPv4-mapped IPv6 (``::ffff:0:0/96``) covers the dual-stack case
    # where an upstream resolves to e.g. ``::ffff:169.254.169.254`` and
    # the kernel hands the v4-mapped form to the socket — that would
    # otherwise be a clean SSRF bypass to IMDS through the v6 path.
    "::ffff:0:0/96",
    # RFC6598 / CGNAT — used by AWS VPC for shared services, K8s pod
    # networks, many cloud overlays.  Not strictly RFC1918 but operators
    # universally want it denied for the same reasons.
    "100.64.0.0/10",
    # RFC2544 benchmark range — rare in practice but occasionally used
    # for internal services and never legitimate as an upstream.
    "198.18.0.0/15",
)


# Min env vars the iron-proxy subprocess actually needs.  Everything else
# is stripped — see ``_build_proxy_subprocess_env`` for the rationale.
_PROXY_SUBPROCESS_ENV_ALLOWLIST: Tuple[str, ...] = (
    "PATH",
    "HOME",
    "TMPDIR",
    "TZ",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NO_COLOR",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",  # Windows
    "USERPROFILE",  # Windows
)


# Env vars that must be stripped from the subprocess env even if they're on
# the allowlist or named in mappings — these would either recurse the proxy
# back through itself or send its traffic through a corporate proxy.
_PROXY_SUBPROCESS_ENV_STRIP: Tuple[str, ...] = (
    "HTTPS_PROXY", "https_proxy",
    "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy",
    "NO_PROXY", "no_proxy",
)


# SIGKILL doesn't exist on Windows.  We fall back to SIGTERM there, which the
# OS treats as a hard terminate via TerminateProcess() — equivalent semantics.
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


# Cached ``iron-proxy --version`` output keyed by binary path.  ``get_status``
# is invoked per Docker-container-create; the version string is constant for
# a given binary so a one-shot subprocess call is plenty.
_VERSION_CACHE: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProxyStatus:
    """Snapshot of the iron-proxy installation + runtime state."""

    enabled: bool = False
    binary_path: Optional[Path] = None
    binary_version: Optional[str] = None
    config_path: Optional[Path] = None
    ca_cert_path: Optional[Path] = None
    pid: Optional[int] = None
    listening: bool = False
    tunnel_port: int = _DEFAULT_TUNNEL_PORT
    warnings: List[str] = field(default_factory=list)

    @property
    def installed(self) -> bool:
        return self.binary_path is not None and self.binary_path.exists()

    @property
    def configured(self) -> bool:
        return (
            self.config_path is not None
            and self.config_path.exists()
            and self.ca_cert_path is not None
            and self.ca_cert_path.exists()
        )


@dataclass
class TokenMapping:
    """Map a sandbox-visible proxy token to a real upstream credential lookup.

    ``real_env_name`` is the env-var name iron-proxy reads at egress time.
    When Bitwarden is configured as the credential source for the proxy,
    iron-proxy's *own* environment is populated from bws on startup — the
    sandbox still sees only ``proxy_token``.
    """

    proxy_token: str
    real_env_name: str
    upstream_hosts: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _hermes_bin_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "bin"


def _proxy_state_dir_ro() -> Path:
    """Return the proxy state dir without creating it.

    Read-only callers (status probes, pidfile reads, version queries) use
    this — there's no reason to materialize ``~/.hermes/proxy/`` just to
    check whether a pidfile exists.
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "proxy"


def _proxy_state_dir() -> Path:
    """Return the proxy state dir, creating it with 0o700 if absent.

    Writable callers (CA gen, config write, mappings write, start_proxy)
    use this.  We force 0o700 — the dir holds the CA signing key, audit
    log, and pidfile, so traversal by other local users is undesirable.
    The chmod is unconditional so a pre-existing dir with a slack umask
    gets tightened on first access.
    """
    d = _proxy_state_dir_ro()
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)
    except OSError:
        # On Windows the chmod is a no-op for POSIX modes; on shared
        # filesystems we may not own the dir.  Don't fail here — the
        # individual files still get explicit perms.
        pass
    return d


def _platform_binary_name() -> str:
    return "iron-proxy.exe" if platform.system() == "Windows" else "iron-proxy"


def _platform_asset_name() -> str:
    """Map (uname, arch) → upstream release asset filename.

    iron-proxy ships ``iron-proxy_<version>_<os>_<arch>.tar.gz``.
    Windows builds aren't published upstream as of v0.39.0; we raise a
    clear error for callers on Windows.
    """

    system = platform.system()
    machine = platform.machine().lower()

    if system == "Linux":
        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        return f"iron-proxy_{_IRON_PROXY_VERSION}_linux_{arch}.tar.gz"
    if system == "Darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        return f"iron-proxy_{_IRON_PROXY_VERSION}_darwin_{arch}.tar.gz"
    if system == "Windows":
        raise RuntimeError(
            "iron-proxy does not ship native Windows binaries as of "
            f"v{_IRON_PROXY_VERSION}. Run the proxy on a Linux/macOS host, "
            "or inside WSL."
        )

    raise RuntimeError(
        f"Unsupported platform for iron-proxy auto-install: {system} {machine}"
    )


# ---------------------------------------------------------------------------
# Binary discovery + lazy install
# ---------------------------------------------------------------------------


def find_iron_proxy(*, install_if_missing: bool = False) -> Optional[Path]:
    """Return a path to a usable ``iron-proxy`` binary, or None.

    Resolution order:
      1. ``<hermes_home>/bin/iron-proxy``  (our managed copy — preferred)
      2. ``shutil.which("iron-proxy")``    (system PATH)

    When ``install_if_missing`` is True and neither resolves, calls
    :func:`install_iron_proxy` to download and verify the pinned version.
    """

    managed = _hermes_bin_dir() / _platform_binary_name()
    if managed.exists() and os.access(managed, os.X_OK):
        return managed

    system = shutil.which("iron-proxy")
    if system:
        return Path(system)

    if install_if_missing:
        try:
            return install_iron_proxy()
        except Exception as exc:  # noqa: BLE001 — never block startup
            logger.warning("iron-proxy auto-install failed: %s", exc)
            return None
    return None


def install_iron_proxy(*, force: bool = False) -> Path:
    """Download, verify, and install the pinned ``iron-proxy`` binary.

    Returns the path to the installed executable.  Raises on any failure
    (network, checksum, extraction).  Callers in the auto-install path catch
    these; the user-facing ``hermes proxy install`` surface lets them
    propagate so the wizard can show a clear error.
    """

    bin_dir = _hermes_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / _platform_binary_name()

    if target.exists() and not force:
        return target

    asset_name = _platform_asset_name()
    asset_url = f"{_IRON_PROXY_RELEASE_BASE}/{asset_name}"
    checksum_url = f"{_IRON_PROXY_RELEASE_BASE}/{_IRON_PROXY_CHECKSUM_NAME}"

    with tempfile.TemporaryDirectory(prefix="hermes-iron-proxy-") as tmpdir:
        tmp = Path(tmpdir)
        archive_path = tmp / asset_name
        checksum_path = tmp / _IRON_PROXY_CHECKSUM_NAME

        logger.info("Downloading %s", asset_url)
        _http_download(asset_url, archive_path)

        actual = _sha256_file(archive_path)

        # Source-pinned digest is authoritative (see _IRON_PROXY_PINNED_SHA256).
        # This is the trust anchor: it can't be influenced by whoever served
        # the download.
        pinned = _IRON_PROXY_PINNED_SHA256.get(asset_name)
        if pinned is None:
            raise RuntimeError(
                f"No source-pinned SHA-256 for {asset_name}. Refusing to "
                "install an unpinned iron-proxy build. Add the digest to "
                "_IRON_PROXY_PINNED_SHA256 in a reviewed commit."
            )
        if pinned.lower() != actual.lower():
            raise RuntimeError(
                f"Checksum mismatch for {asset_name}: expected (pinned) "
                f"{pinned}, got {actual}. The downloaded binary does not "
                "match the source-pinned digest — aborting."
            )

        # Defence-in-depth: if the release ships a checksums.txt, cross-check
        # it too and warn loudly on any disagreement with the pinned value.
        # A mismatch here (when the pinned check already passed) means the
        # upstream release content changed — worth surfacing.
        try:
            _http_download(checksum_url, checksum_path)
            published = _expected_sha256(checksum_path, asset_name)
            if published.lower() != pinned.lower():
                logger.warning(
                    "iron-proxy %s: upstream checksums.txt (%s) disagrees "
                    "with the source-pinned digest (%s) for %s. Installed the "
                    "pinned, verified binary; investigate the upstream release.",
                    _IRON_PROXY_VERSION, published, pinned, asset_name,
                )
        except RuntimeError as exc:
            # Missing/unreadable checksums.txt is non-fatal now that the
            # pinned digest is authoritative.
            logger.debug("checksums.txt cross-check skipped: %s", exc)

        with tarfile.open(archive_path, "r:gz") as tf:
            member = _pick_tar_member(tf, _platform_binary_name())
            # PEP 706 data filter — strips ownership/mode replay (we set
            # chmod explicitly below) AND rejects symlink/hardlink members
            # that escape the extraction dir.  Required on 3.12+ to silence
            # the deprecation warning and on 3.14+ to opt into the
            # tarbomb-rejecting default.
            try:
                tf.extract(member, tmp, filter="data")  # noqa: S202
            except TypeError:
                # Python < 3.12 — filter kw didn't exist yet; the
                # _pick_tar_member sanitization already rejects path
                # traversal so this is acceptable.
                tf.extract(member, tmp)  # noqa: S202
            extracted = tmp / member.name

        # Stage into the final directory then atomically rename so the new
        # binary is never visible half-written.
        fd, staged = tempfile.mkstemp(dir=str(bin_dir), prefix=".iron-proxy_")
        os.close(fd)
        shutil.copy2(extracted, staged)
        os.chmod(
            staged,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            | stat.S_IRGRP | stat.S_IXGRP
            | stat.S_IROTH | stat.S_IXOTH,
        )
        os.replace(staged, target)

    # Invalidate the version cache so a freshly-installed binary
    # re-probes ``--version`` on the next ``get_status()`` call instead
    # of returning the pre-upgrade string.  Long-lived processes that
    # bump the pinned version via ``force=True`` need this.
    _VERSION_CACHE.pop(str(target), None)

    logger.info("Installed iron-proxy %s at %s", _IRON_PROXY_VERSION, target)
    return target


def _http_download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent"})
    try:
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _expected_sha256(checksum_file: Path, asset_name: str) -> str:
    """Parse the standard ``sha256sum`` output: ``<hex>  <filename>``."""

    text = checksum_file.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1] == asset_name:
            return parts[0]
    raise RuntimeError(
        f"No checksum entry for {asset_name} in {checksum_file.name}"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_tar_member(tf: tarfile.TarFile, binary_name: str) -> tarfile.TarInfo:
    """Find the binary inside the upstream tar.

    iron-proxy's archive is typically flat (binary at root) but we tolerate
    a top-level directory.  Members must be regular files with a leaf name
    matching ``binary_name``, no absolute paths, and no ``..`` traversal.
    """

    candidates: List[tarfile.TarInfo] = []
    for member in tf.getmembers():
        if not member.isfile():
            continue
        if member.name.startswith("/") or ".." in Path(member.name).parts:
            continue
        if Path(member.name).name == binary_name:
            candidates.append(member)
    if not candidates:
        raise RuntimeError(
            f"Could not find {binary_name} inside downloaded archive "
            f"(members: {[m.name for m in tf.getmembers()[:5]]}...)"
        )
    candidates.sort(key=lambda m: len(m.name))
    return candidates[0]


def iron_proxy_version(binary: Path) -> str:
    """Return ``iron-proxy --version`` output, stripped.  Empty on failure.

    Cached by binary path: ``get_status`` is called per Docker container
    create, but the version string is constant for a given binary.  A
    single subprocess invocation is plenty.
    """

    key = str(binary)
    cached = _VERSION_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        res = subprocess.run(  # noqa: S603 — binary path is trusted
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    out = (res.stdout or res.stderr or "").strip()
    # Don't cache empty output — that would poison ``hermes egress
    # status`` for the lifetime of the process if the first probe hit a
    # corrupt binary or a flag-rename in a newer upstream.  Re-probe on
    # the next call instead.
    if out:
        _VERSION_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# CA cert generation
# ---------------------------------------------------------------------------


def ensure_ca_cert(*, force: bool = False) -> Tuple[Path, Path]:
    """Generate (or return existing) iron-proxy CA cert + key.

    Uses the host's ``openssl`` binary.  We don't try to bind to a Python
    crypto library — openssl is universally available on the platforms we
    support, and it sidesteps cryptography-package licensing/distribution
    surface.
    """

    state = _proxy_state_dir()
    ca_crt = state / "ca.crt"
    ca_key = state / "ca.key"

    if ca_crt.exists() and ca_key.exists() and not force:
        return ca_crt, ca_key

    if shutil.which("openssl") is None:
        raise RuntimeError(
            "openssl not found on PATH. Install OpenSSL (apt: `openssl`, "
            "brew: `openssl`) to generate the iron-proxy CA cert."
        )

    # weklund: 2-year CA validity. iron-proxy mints short-lived leaf certs
    # from this CA, so a long-lived CA isn't load-bearing for normal operation
    # — but a 10-year cert means a locally-compromised CA key stays trusted for
    # a decade. A 2-year window is a natural forcing function to revisit the
    # proxy CA, and rotation is seamless (`hermes egress rotate-ca`, or
    # `ensure_ca_cert(force=True)`).
    with tempfile.TemporaryDirectory(prefix="hermes-proxy-ca-") as tmpdir:
        tmp = Path(tmpdir)
        tmp_key = tmp / "ca.key"
        tmp_crt = tmp / "ca.crt"

        subprocess.run(  # noqa: S603 — openssl path is trusted PATH lookup
            ["openssl", "genrsa", "-out", str(tmp_key), "4096"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(  # noqa: S603
            [
                "openssl", "req", "-x509", "-new", "-nodes",
                "-key", str(tmp_key),
                "-sha256", "-days", "730",
                "-subj", "/CN=hermes iron-proxy CA",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,keyCertSign",
                "-out", str(tmp_crt),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

        # Move into place with private permissions.  CRITICAL: the key
        # has to be created with 0o600 from the very first byte — a
        # ``shutil.copy2`` followed by ``os.chmod`` leaves a TOCTOU window
        # where the private key is world-readable on multi-user hosts.
        key_bytes = tmp_key.read_bytes()
        crt_bytes = tmp_crt.read_bytes()

        # Stage with explicit 0o600, then atomically rename into place.
        # O_NOFOLLOW guards against a symlink at ca_key (defence-in-depth
        # — the state dir is 0o700-owned but a malicious local user with
        # the same uid could pre-create one).
        key_staged = ca_key.with_suffix(ca_key.suffix + ".staged")
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        # O_NOFOLLOW exists on POSIX; on Windows we just rely on the
        # default semantics.
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        # Best-effort: pre-unlink any existing staged file so the open
        # with O_CREAT is always against a fresh inode.
        try:
            key_staged.unlink()
        except FileNotFoundError:
            pass
        fd = os.open(str(key_staged), open_flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(key_bytes)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(key_staged, ca_key)

        # Cert is public — 0o644 is fine and matches typical PEM layout.
        ca_crt.write_bytes(crt_bytes)
        os.chmod(ca_crt, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    logger.info("Generated iron-proxy CA at %s", ca_crt)
    return ca_crt, ca_key


# ---------------------------------------------------------------------------
# Proxy config + token mapping generation
# ---------------------------------------------------------------------------


def mint_proxy_token(prefix: str = "hermes-proxy") -> str:
    """Mint a fresh opaque token to hand to the sandbox.

    The token has no internal structure beyond a recognizable prefix —
    iron-proxy matches on exact equality.  We use a 128-bit random suffix
    (32 hex chars from a SHA-256 of 32 bytes of os.urandom).  At that
    entropy the birthday-bound collision probability is below 2^-64 for
    up to 2^32 tokens, which is plenty for a proxy-scoped namespace.
    """

    return f"{prefix}-{hashlib.sha256(os.urandom(32)).hexdigest()[:32]}"


def _default_http_listen(tunnel_port: int) -> List[str]:
    """Build the single host:port bind the proxy should listen on.

    iron-proxy v0.39 supports exactly ONE ``proxy.http_listen`` bind per
    daemon process, so this returns a one-element list and the choice of
    host matters:

    * **Linux:** bind the docker bridge gateway (``172.17.0.1`` by
      default).  Sandboxes reach the proxy via
      ``host.docker.internal:host-gateway``, which Docker resolves to
      exactly this bridge gateway IP on Linux — a loopback-only bind is
      unreachable from inside containers there.  The bridge IP is still
      host-local (it's an address on the host's ``docker0`` interface),
      so host-side tooling and the status probe can reach it too.  When
      no docker bridge is detected (docker not installed / not started),
      fall back to loopback — there are no sandboxes to serve in that
      state, and the operator gets a warning.
    * **macOS / Windows Docker Desktop:** ``host.docker.internal``
      resolves via VPNkit to the host, so a loopback bind is reachable
      from containers and is the least-exposed choice.

    We never bind ``0.0.0.0`` — that would expose the proxy (and, with a
    leaked sandbox token, the user's API quota) to anyone on the local
    network.  The bridge-gateway bind is reachable by other containers
    on the default bridge network, which is unavoidable given v0.39's
    single-bind limit; requests still require a minted proxy token and
    an allowlisted upstream.
    """

    if platform.system() == "Linux":
        bridge_ip = _detect_docker_bridge_ip()
        if bridge_ip and bridge_ip != "127.0.0.1":
            return [f"{bridge_ip}:{tunnel_port}"]
        logger.warning(
            "No docker bridge (docker0) detected — binding iron-proxy to "
            "loopback only.  Docker sandboxes will NOT be able to reach "
            "the proxy until it is restarted with docker running."
        )
    return [f"127.0.0.1:{tunnel_port}"]


def _detect_docker_bridge_ip() -> Optional[str]:
    """Return the docker0 bridge IPv4, if present, else None.

    Best-effort: we try ``ip -4 addr show docker0`` first.  Anything that
    fails, doesn't parse as a strict IPv4, or parses as an address we
    must NOT bind to (unspecified, loopback, multicast, reserved, public)
    returns None — callers handle that as "no bridge bind".

    SECURITY: a hostile ``ip`` shim earlier on the operator's PATH used
    to be able to inject ``0.0.0.0`` here and re-open INADDR_ANY binding
    that the rest of the bind-policy work explicitly closed.  We
    validate via :mod:`ipaddress` and reject anything that isn't
    plausibly a docker bridge IP (private + non-special).
    """

    candidate: Optional[str] = None
    try:
        res = subprocess.run(  # noqa: S603 — ip is a system binary
            ["ip", "-4", "-o", "addr", "show", "docker0"],
            capture_output=True, text=True, timeout=2,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = line.split()
                # Expected: "<n>: docker0  inet 172.17.0.1/16 ..."
                for i, tok in enumerate(parts):
                    if tok == "inet" and i + 1 < len(parts):
                        candidate = parts[i + 1].split("/")[0]
                        break
                if candidate is not None:
                    break
    except (OSError, subprocess.TimeoutExpired):
        return None

    if not candidate:
        return None

    # Stdlib validation: rejects garbage strings AND special-purpose
    # addresses that must not be used as a bind target.
    try:
        addr = ipaddress.IPv4Address(candidate)
    except (ipaddress.AddressValueError, ValueError):
        return None
    # Reject:
    # - 0.0.0.0 / INADDR_ANY  (is_unspecified)
    # - 127.0.0.0/8           (is_loopback — already in deny list)
    # - 224.0.0.0/4           (is_multicast)
    # - 240.0.0.0/4           (is_reserved)
    # - 169.254.0.0/16        (is_link_local — IMDS range, never docker0)
    # - global / public IPs   (is_global — docker0 must be RFC1918)
    if (
        addr.is_unspecified
        or addr.is_loopback
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_link_local
        or addr.is_global
    ):
        logger.warning(
            "Refusing suspicious docker bridge IP %s reported by `ip`; "
            "skipping bridge bind.", candidate,
        )
        return None

    return str(addr)


def build_proxy_config(
    *,
    mappings: List[TokenMapping],
    ca_cert: Path,
    ca_key: Path,
    tunnel_port: int = _DEFAULT_TUNNEL_PORT,
    audit_log: Optional[Path] = None,
    allowed_hosts: Optional[List[str]] = None,
    upstream_deny_cidrs: Optional[List[str]] = None,
    http_listen: Optional[List[str]] = None,
) -> Dict:
    """Build the iron-proxy YAML config (as a dict) for a given mapping set.

    The dict is YAML-serializable via ``yaml.safe_dump``.  iron-proxy reads
    real secrets from its OWN environment via ``source: {type: env, var: ...}``;
    the sandbox never sees them.

    Bind policy: the sandbox-facing listeners (``tunnel_listen`` on
    ``tunnel_port``, plain-HTTP ``http_listen`` on ``tunnel_port + 1``)
    bind the docker bridge gateway on Linux (``172.17.0.1`` or whatever
    ``docker0`` resolves to — that's what ``host.docker.internal``
    resolves to inside containers there) and loopback on macOS / Windows
    Docker Desktop.  We do NOT bind ``0.0.0.0`` — a LAN peer with a
    leaked sandbox token could otherwise spend the operator's API quota
    against any allowlisted upstream.

    SSRF policy: ``upstream_deny_cidrs`` defaults to a conservative deny
    list covering loopback, link-local (incl. AWS/GCP/Azure IMDS at
    169.254.169.254), and RFC1918.  Pass an explicit ``[]`` to opt out of
    the deny list entirely (only sensible in hermetic tests).

    Schema mirrors the official iron-proxy schema as of v0.39.0.  Notable
    points:

    * The ``dns`` section is required by the binary even when we only use the
      CONNECT tunnel.  We point it at loopback so it doesn't conflict with
      anything else and disable the listener.
    * The ``proxy.tunnel_listen`` is what sandboxes hit via ``HTTPS_PROXY``.
      ``http_listen`` / ``https_listen`` are present (loopback only) so the
      proxy boots; sandboxes never route directly to them.
    * ``allowlist`` transform takes ``domains:`` and ``cidrs:``, not ``hosts:``.
    * ``secrets`` transform takes ``secrets:`` (plural), each with a
      ``source``, a ``replace.proxy_value`` (the sandbox-visible token), and
      a list of ``rules`` saying which hosts the swap should fire on.
    """

    hosts: List[str] = list(allowed_hosts or _DEFAULT_ALLOWED_HOSTS)
    for m in mappings:
        for h in m.upstream_hosts:
            if h not in hosts:
                hosts.append(h)

    secrets_rules = []
    for m in mappings:
        secrets_rules.append({
            "source": {"type": "env", "var": m.real_env_name},
            "replace": {
                "proxy_value": m.proxy_token,
                "match_headers": ["Authorization"],
                # The token is also accepted as a bearer query param in case
                # the sandbox passes it that way.  Body matching is off — we
                # don't want body inspection forced for every request.
                "match_query": True,
                "match_body": False,
            },
            "rules": [{"host": h} for h in m.upstream_hosts],
        })

    # SSRF protection: default-deny cloud metadata + loopback + RFC1918.
    # Callers can pass [] to opt out entirely (hermetic tests need this for
    # talking to a loopback upstream).  None means "use the default".
    deny_cidrs: List[str]
    if upstream_deny_cidrs is None:
        deny_cidrs = list(_DEFAULT_UPSTREAM_DENY_CIDRS)
    else:
        deny_cidrs = list(upstream_deny_cidrs)

    # Listen addresses.  iron-proxy v0.39 takes a single string per
    # listener field — there is no plural ``http_listens`` form, despite
    # earlier drafts of this module claiming v0.39 accepts both.  An
    # empirical strings(1) audit + a live "start the binary and observe
    # the YAML unmarshal error" confirms the singular form is the only
    # one the binary accepts.
    #
    # LISTENER ROLES (verified live against the v0.39 binary):
    # * ``tunnel_listen`` is the CONNECT + MITM listener.  HTTPS through
    #   ``HTTPS_PROXY`` issues CONNECT — this is the listener sandboxes
    #   must reach.  A CONNECT sent to ``http_listen`` is NOT terminated:
    #   v0.39 forwards it upstream as a regular request and the upstream
    #   responds 400.
    # * ``http_listen`` is the absolute-form plain-HTTP forward listener
    #   (``HTTP_PROXY`` for ``http://`` URLs).  Transforms fire here too.
    # Both get the sandbox-facing bind host: tunnel on ``tunnel_port``,
    # plain HTTP on ``tunnel_port + 1``.
    #
    # The bind host comes from _default_http_listen: the docker bridge
    # gateway on Linux (containers reach the proxy via
    # host.docker.internal, which maps to the bridge gateway there —
    # loopback would be unreachable from inside sandboxes) and loopback
    # on macOS/Windows Docker Desktop (where host.docker.internal routes
    # to the host via VPNkit).
    listens = list(http_listen) if http_listen else _default_http_listen(tunnel_port)
    primary_listen = listens[0] if listens else f"127.0.0.1:{tunnel_port}"
    bind_host = primary_listen.rsplit(":", 1)[0] or "127.0.0.1"
    plain_http_listen = f"{bind_host}:{tunnel_port + 1}"

    log_block: Dict = {"level": "info"}
    # NOTE: ``log.audit_path`` is NOT a field in iron-proxy v0.39's
    # ``config.Log`` struct — the binary rejects it with
    # ``field audit_path not found in type config.Log``.  Per-request
    # audit records are written to the same log destination as
    # everything else at this binary version; the operator-facing
    # ``audit.log`` file we pre-create is still useful as a sentinel
    # for monitoring (logrotate target, downstream tail watchers) but
    # the daemon does not write to it directly.  The kwarg is kept so
    # we're forward-compatible with a future v0.40+ that adds the
    # field; if you upgrade _IRON_PROXY_VERSION and the upstream gains
    # ``log.audit_path``, re-enable the line below.
    # if audit_log is not None:
    #     log_block["audit_path"] = str(audit_log)
    _ = audit_log  # consumed by ensure_audit_log() / docs only on v0.39

    return {
        # DNS section is required by the binary's config parser, but we run
        # in tunnel-only mode so the DNS listener never binds an exposed port.
        # Sandboxes reach the proxy via HTTPS_PROXY/CONNECT, not via DNS
        # redirection.
        "dns": {
            "listen": "127.0.0.1:0",   # ephemeral loopback — effectively disabled
            "proxy_ip": "127.0.0.1",
        },
        "proxy": {
            # tunnel_listen is the CONNECT/MITM listener — what sandboxes
            # hit via `HTTPS_PROXY=http://host:tunnel_port` for HTTPS
            # upstreams (curl/requests/node issue CONNECT through it).
            # http_listen handles absolute-form plain-HTTP forwards
            # (`HTTP_PROXY` for http:// URLs) on tunnel_port+1.  Both
            # bind the docker bridge gateway on Linux / loopback on
            # Docker Desktop — NEVER 0.0.0.0.  LAN peers with a leaked
            # sandbox token would otherwise be able to spend the
            # operator's API quota against any allowlisted upstream.
            "tunnel_listen": primary_listen,
            "http_listen": plain_http_listen,
            # The HTTPS-listener (direct TLS termination, no CONNECT)
            # gets a loopback ephemeral port — we don't expose it.
            "https_listen": "127.0.0.1:0",
            "max_request_body_bytes": 16 * 1024 * 1024,
            "max_response_body_bytes": 0,
            "upstream_response_header_timeout": "120s",
            # SSRF protection: deny outbound to cloud metadata + loopback by
            # default.  An empty list opts out entirely.
            "upstream_deny_cidrs": deny_cidrs,
        },
        # iron-proxy v0.39 starts a Prometheus-style metrics server by
        # default on ``:9090`` — which is the SAME port as our default
        # ``tunnel_port: 9090``, causing a guaranteed bind collision on
        # startup.  Pin the metrics listener to an ephemeral loopback
        # port (``127.0.0.1:0``) so the metrics binding can't collide
        # with the proxy listener regardless of what tunnel_port the
        # operator chose.  NOTE: ``:0`` means the kernel picks a fresh
        # random port each start and nothing records it — metrics are
        # effectively disabled/undiscoverable at this pin.  If we want
        # scrapable metrics later, allocate a fixed port and surface it
        # in ``ProxyStatus`` / ``hermes egress status``.
        "metrics": {
            "listen": "127.0.0.1:0",
        },
        "tls": {
            "ca_cert": str(ca_cert),
            "ca_key": str(ca_key),
            "cert_cache_size": 1000,
            "leaf_cert_expiry_hours": 168,
        },
        "transforms": [
            {
                "name": "allowlist",
                "config": {"domains": hosts},
            },
            {
                "name": "secrets",
                "config": {"secrets": secrets_rules},
            },
        ],
        "log": log_block,
    }


def ensure_audit_log(audit_path: Path) -> None:
    """Create the audit log file with private permissions (0o600).

    Called from the wizard right before ``start_proxy``.  On the pinned
    v0.39 the daemon never writes this file (no ``log.audit_path``
    config field), so the pre-create is purely forward-compat: when the
    pin moves to a version that supports a dedicated audit stream, the
    file already exists with tight permissions and the daemon inherits
    them instead of creating it under the default umask.

    Raises :class:`RuntimeError` on any OSError (planted symlink,
    immutable parent dir, full disk) so the caller can decide how to
    surface it.  The wizard treats this as a WARNING on v0.39 — the
    file is non-load-bearing until the version bump — but the qualified
    message keeps operators from wiring monitoring to a path that can't
    exist.
    """

    try:
        # Use os.open + O_CREAT to avoid races on the chmod.
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        fd = os.open(str(audit_path), open_flags, 0o600)
        try:
            # Tighten perms even if the file already existed under a
            # slacker umask.
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
    except OSError as exc:
        raise RuntimeError(
            f"Refusing to start: could not pre-create audit log "
            f"{audit_path} with restrictive permissions ({exc}).  "
            f"Move or chmod any existing file at that path and retry."
        ) from exc


def write_proxy_config(config: Dict) -> Path:
    """Serialize the config dict to ``<hermes_home>/proxy/proxy.yaml``.

    Uses ``yaml.safe_dump`` so we never emit Python tags.
    """

    try:
        import yaml  # PyYAML is already a Hermes dep
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to write the iron-proxy config but is not "
            "installed."
        ) from exc

    state = _proxy_state_dir()
    out = state / "proxy.yaml"
    tmp_path = state / ".proxy.yaml.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp_path, out)
    os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
    return out


def write_mappings(mappings: List[TokenMapping]) -> Path:
    """Persist the sandbox-visible proxy tokens to ``mappings.json``.

    The Docker backend reads this file to inject the right tokens as env
    vars when starting a sandbox.  The file is NOT read by iron-proxy
    itself — the mapping is already baked into ``proxy.yaml``.
    """

    state = _proxy_state_dir()
    out = state / "mappings.json"
    payload = {
        "version": 1,
        "tokens": [
            {
                "proxy_token": m.proxy_token,
                "env_name": m.real_env_name,
                "upstream_hosts": list(m.upstream_hosts),
            }
            for m in mappings
        ],
    }
    tmp_path = state / ".mappings.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out)
    os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
    return out


def load_mappings() -> List[TokenMapping]:
    """Read mappings.json, if it exists.  Empty list on any error."""

    state = _proxy_state_dir()
    f = state / "mappings.json"
    if not f.exists():
        return []
    try:
        payload = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read iron-proxy mappings.json: %s", exc)
        return []
    out: List[TokenMapping] = []
    for item in payload.get("tokens", []):
        try:
            out.append(TokenMapping(
                proxy_token=item["proxy_token"],
                real_env_name=item["env_name"],
                upstream_hosts=tuple(item.get("upstream_hosts") or ()),
            ))
        except (KeyError, TypeError):
            continue
    return out


def discover_provider_mappings(
    *,
    available_env_names: Optional[List[str]] = None,
) -> List[TokenMapping]:
    """Mint a TokenMapping for every known provider whose env var is set.

    Pass ``available_env_names`` to override the lookup source (used by the
    Bitwarden adapter so we mint mappings for keys that *will* be in the
    proxy's environment even if they aren't in the host process env right
    now).
    """

    if available_env_names is not None:
        names = set(available_env_names)
    else:
        names = {k for k, v in os.environ.items() if v}

    mappings: List[TokenMapping] = []
    for env_name, hosts in _BEARER_PROVIDERS.items():
        if env_name not in names:
            continue
        mappings.append(TokenMapping(
            proxy_token=mint_proxy_token(prefix=env_name.lower().replace("_api_key", "")),
            real_env_name=env_name,
            upstream_hosts=hosts,
        ))
    return mappings


def discover_uncovered_providers(
    *,
    available_env_names: Optional[List[str]] = None,
) -> List[str]:
    """Return env-var names for providers we recognize but can't proxy.

    Anthropic native (x-api-key), AWS Bedrock (SigV4), Azure OpenAI
    (api-key), etc.  When any of these are configured, the sandbox is
    holding real credentials that the proxy can't strip — the isolation
    guarantee is incomplete for those providers.

    The wizard uses this to print a warning at setup time; ``start_proxy``
    can be configured to refuse to start when ``fail_on_uncovered_providers``
    is true (see :func:`discover_blocked_providers` for the strict tier
    that actually blocks).
    """

    if available_env_names is not None:
        names = set(available_env_names)
    else:
        names = {k for k, v in os.environ.items() if v}

    return [n for n in _NON_BEARER_PROVIDERS if n in names]


def discover_blocked_providers(
    *,
    available_env_names: Optional[List[str]] = None,
) -> List[str]:
    """Return env-var names for non-bearer providers that BLOCK start.

    Subset of :func:`discover_uncovered_providers` that's LLM-specific
    enough to refuse-start when ``proxy.fail_on_uncovered_providers`` is
    true.  Excludes generic cloud creds (AWS_*, GCP application-default)
    that are usually present for unrelated tooling.
    """

    if available_env_names is not None:
        names = set(available_env_names)
    else:
        names = {k for k, v in os.environ.items() if v}

    return [n for n in _LLM_SPECIFIC_NON_BEARER_PROVIDERS if n in names]


def merge_mappings(
    *,
    existing: List[TokenMapping],
    discovered: List[TokenMapping],
    rotate: bool = False,
) -> List[TokenMapping]:
    """Combine an existing mapping set with freshly discovered providers.

    By default this PRESERVES tokens for providers already in ``existing`` —
    re-running ``hermes egress setup`` should not invalidate the tokens
    baked into containers that are already running.  Only newly added
    providers get freshly minted tokens.

    When ``rotate=True``, every token in the result is freshly minted
    regardless of overlap.  The wizard exposes this via ``--rotate-tokens``
    for the rare case where the operator wants to roll all tokens
    deliberately (e.g. after a suspected token leak).

    Providers that are in ``existing`` but no longer in ``discovered``
    (operator removed the env var since last setup) are dropped.
    """

    by_name = {m.real_env_name: m for m in existing}
    out: List[TokenMapping] = []
    for d in discovered:
        prior = by_name.get(d.real_env_name)
        if prior is not None and not rotate:
            # Preserve the token, refresh the host list in case we added
            # new upstreams since last setup.
            out.append(TokenMapping(
                proxy_token=prior.proxy_token,
                real_env_name=prior.real_env_name,
                upstream_hosts=d.upstream_hosts,
            ))
        else:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------


def _pidfile() -> Path:
    return _proxy_state_dir() / "iron-proxy.pid"


def _read_pid() -> Optional[int]:
    # Use the read-only path: don't create the proxy dir just to read the
    # pidfile.  If neither pid file nor dir exists, the daemon is plainly
    # not running.
    pf = _proxy_state_dir_ro() / "iron-proxy.pid"
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


# Nonce env-var set in the iron-proxy subprocess at start_proxy time.  Used
# by ``_pid_alive`` to confirm a candidate PID still refers to *our* managed
# binary even across PID recycling (a fresh process can't inherit our
# arbitrary env value).
_HERMES_IRON_PROXY_NONCE_ENV = "HERMES_IRON_PROXY_NONCE"
_proxy_nonce: Optional[str] = None


def _pid_proc_starttime(pid: int) -> Optional[str]:
    """Return /proc/<pid>/stat[21] (starttime) on Linux, else None.

    Comparing starttime is the standard cheap way to detect PID recycling
    without relying on cmdline scanning.  When None, callers fall back to
    the cmdline + nonce check.
    """
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # /proc/<pid>/stat: pid (comm-with-parens) state ppid ... fields[21]=starttime
    # The "comm" field can contain spaces and parens, so split from the
    # right parenthesis instead of using shlex.
    rparen = text.rfind(")")
    if rparen < 0:
        return None
    fields = text[rparen + 1:].split()
    # field index in the post-")" tail: original 3..n become fields[0..n-3]
    # starttime is original field 22 (1-indexed) → tail index 22-3 = 19
    if len(fields) <= 19:
        return None
    return fields[19]


def _persisted_nonce_path() -> Path:
    """Path to the on-disk sibling of the pidfile that stores the nonce.

    Written by ``_write_pidfile_safely`` after ``start_proxy`` plants
    the nonce in the iron-proxy child env, read by ``_pid_alive`` in a
    later CLI invocation (``stop`` / ``status``) so cross-process
    PID-recycling defense holds.
    """
    return _proxy_state_dir_ro() / "iron-proxy.nonce"


def _read_persisted_nonce() -> Optional[str]:
    """Read the on-disk nonce written next to the pidfile.

    Returns None when the file is missing, unreadable, or empty —
    callers fall back to argv0 basename matching in that case.
    """
    p = _persisted_nonce_path()
    try:
        # O_NOFOLLOW: defence-in-depth against a planted symlink at the
        # nonce path; same-uid required to plant one but worth defending
        # since the nonce read here decides whether stop_proxy will
        # SIGKILL a candidate PID.
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(p), flags)
    except OSError:
        return None
    try:
        # Ownership check — if the file isn't owned by us, ignore it.
        # Same threat model as the pidfile uid check.
        try:
            st = os.fstat(fd)
            if hasattr(os, "getuid") and st.st_uid != os.getuid():
                return None
        except AttributeError:
            pass
        data = os.read(fd, 256).decode("utf-8", errors="ignore").strip()
        return data or None
    finally:
        os.close(fd)


def _pid_alive(pid: int) -> bool:
    """Return True iff ``pid`` is alive AND is an iron-proxy process.

    Defends against PID reuse via three signals (in priority order):
    1. ``/proc/<pid>/environ`` contains our nonce  (most reliable, Linux)
    2. ``/proc/<pid>/cmdline`` basename matches the managed binary
    3. ``ps -p <pid>`` command line contains the binary path

    The legacy ``"iron-proxy" in cmdline`` match was loose enough to match
    ``tail iron-proxy.log`` or an editor with that file open.  We tighten
    on argv[0] basename plus an in-process nonce instead.
    """

    if pid <= 0:
        return False
    try:
        # Use psutil.pid_exists when available — it's a no-op on Windows
        # whereas os.kill(pid, 0) on Windows is actually a hard kill
        # (CTRL_C_EVENT to the target's console process group).  See
        # bpo-14484.  windows-footgun: ok — we explicitly skip the
        # os.kill probe on Windows below.
        import psutil  # type: ignore
        if not psutil.pid_exists(pid):
            return False
    except ImportError:
        if platform.system() == "Windows":
            # On Windows without psutil we can't safely probe — assume
            # the pidfile content is fresh and confirm via the cmdline
            # path below.  os.kill(pid, 0) is NOT safe here.
            pass
        else:
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only branch
            except (ProcessLookupError, PermissionError, OSError):
                return False

    # Strong proof: nonce env var matches.  /proc/<pid>/environ is null-
    # separated KEY=VALUE pairs; substring search is safe.
    #
    # The nonce can come from either:
    #   1. the module-global ``_proxy_nonce`` set during this process's
    #      own ``start_proxy`` call (same-process case);
    #   2. the on-disk ``iron-proxy.nonce`` file written by
    #      ``_write_pidfile_safely``, used when ``start`` and ``stop``
    #      run in separate CLI invocations (cross-process case).
    # Either source provides the same defeat-PID-recycling guarantee.
    nonce_candidates: List[str] = []
    if _proxy_nonce:
        nonce_candidates.append(_proxy_nonce)
    on_disk = _read_persisted_nonce()
    if on_disk and on_disk not in nonce_candidates:
        nonce_candidates.append(on_disk)
    if nonce_candidates:
        try:
            env_bytes = Path(f"/proc/{pid}/environ").read_bytes()
            for nonce in nonce_candidates:
                needle = f"{_HERMES_IRON_PROXY_NONCE_ENV}={nonce}".encode()
                if needle in env_bytes:
                    return True
        except OSError:
            pass

    # Fallback: cmdline basename match.  argv[0] is the first null-
    # separated token in /proc/<pid>/cmdline.
    try:
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if cmdline_path.exists():
            tokens = cmdline_path.read_bytes().split(b"\x00")
            if tokens:
                argv0 = tokens[0].decode("utf-8", errors="ignore")
                argv0_base = os.path.basename(argv0)
                if argv0_base.startswith("iron-proxy"):
                    return True
            return False
    except OSError:
        pass

    # macOS / non-Linux fallback: ``ps`` command basename.
    try:
        res = subprocess.run(  # noqa: S603
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True, text=True, timeout=2,
        )
        if res.returncode == 0:
            comm = (res.stdout or "").strip()
            return os.path.basename(comm).startswith("iron-proxy")
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Exotic platforms: be conservative — if the OS says alive we believe
    # it.  This restores the previous behaviour for non-Linux/non-macOS.
    return True


def start_proxy(
    *,
    binary: Optional[Path] = None,
    config_path: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
    install_if_missing: bool = True,
    refresh_secrets_from_bitwarden: bool = False,
    bitwarden_config: Optional[Dict] = None,
) -> ProxyStatus:
    """Spawn iron-proxy as a managed background subprocess.

    Idempotent — if the proxy is already running with the expected PID,
    just returns the live status.

    ``refresh_secrets_from_bitwarden=True`` re-fetches upstream secrets
    via ``bws secret list`` at startup and injects them into the child
    env.  This delivers the rotation promise that distinguishes
    ``credential_source: bitwarden`` from ``credential_source: env``.
    Without this flag (or with ``bitwarden_config=None``) the proxy still
    starts but uses whatever the host process env happens to contain.
    """

    global _proxy_nonce

    existing = _read_pid()
    if existing and _pid_alive(existing):
        return get_status()

    bin_path = binary or find_iron_proxy(install_if_missing=install_if_missing)
    if bin_path is None:
        raise RuntimeError(
            "iron-proxy binary not available — run `hermes egress install`."
        )

    cfg = config_path or (_proxy_state_dir() / "proxy.yaml")
    if not cfg.exists():
        raise RuntimeError(
            f"iron-proxy config not found at {cfg}. "
            "Run `hermes egress setup` first."
        )

    # Build a minimal subprocess env.  os.environ.copy() would ship every
    # secret in the operator's shell to the proxy — /proc/<pid>/environ
    # would then expose OPENAI_API_KEY, AWS keys, etc. to any same-uid
    # local process.  Defeats the threat model the proxy exists to
    # mitigate.
    env = _build_proxy_subprocess_env(
        extra_env=extra_env,
        refresh_from_bitwarden=refresh_secrets_from_bitwarden,
        bitwarden_config=bitwarden_config,
    )

    # Plant a per-start nonce in the child env so ``_pid_alive`` can
    # confirm a candidate PID still refers to *our* binary across PID
    # recycling.  Module-global is fine — only one managed proxy per
    # Hermes process.
    _proxy_nonce = hashlib.sha256(os.urandom(16)).hexdigest()
    env[_HERMES_IRON_PROXY_NONCE_ENV] = _proxy_nonce

    log_path = _proxy_state_dir() / "iron-proxy.log"
    # Keep ownership of the fd tight: open with explicit 0o600 so the
    # log doesn't get world-readable under a slack umask, then close it
    # immediately after Popen (the child has its own dup).  Without the
    # close-on-success path, every restart leaked one fd in the Hermes
    # process.
    #
    # O_NOFOLLOW (defence-in-depth, same threat model as the pidfile
    # path): a same-uid attacker who plants ``iron-proxy.log`` as a
    # symlink to e.g. ``~/.ssh/authorized_keys`` would otherwise cause
    # every restart to append daemon diagnostics to that file.
    log_open_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        log_open_flags |= os.O_NOFOLLOW
    try:
        log_fd = os.open(str(log_path), log_open_flags, 0o600)
    except OSError as exc:
        # ELOOP from a planted symlink — refuse with a clear error.
        raise RuntimeError(
            f"Refusing to write iron-proxy log {log_path}: {exc}.  "
            "Remove that path manually and retry."
        ) from exc
    try:
        os.fchmod(log_fd, 0o600)  # tighten if file pre-existed
    except OSError:
        pass
    # Verify ownership — same st_uid check the pidfile uses.
    try:
        st = os.fstat(log_fd)
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            os.close(log_fd)
            raise RuntimeError(
                f"iron-proxy log {log_path} has unexpected owner "
                f"uid={st.st_uid}; refusing to write."
            )
    except AttributeError:
        pass  # Windows

    try:
        # Use the fd directly via the dup mechanism; Popen will dup() it
        # into the child so we can close ours unconditionally below.
        # NOTE: on Windows ``start_new_session`` is invalid; we don't
        # support Windows for the proxy (the binary itself doesn't ship)
        # but the kwarg is POSIX-only and silently ignored on Win.
        popen_kwargs: Dict = dict(
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
        )
        if platform.system() != "Windows":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(  # noqa: S603 — binary path is trusted
            [str(bin_path), "-config", str(cfg)],
            **popen_kwargs,
        )
    except OSError as exc:
        os.close(log_fd)
        raise RuntimeError(f"failed to spawn iron-proxy: {exc}") from exc
    finally:
        # Close our copy of the fd whether Popen raised or succeeded.
        # The child has its own dup via Popen, so it's still writing.
        try:
            os.close(log_fd)
        except OSError:
            pass

    # Write the pidfile IMMEDIATELY after Popen, BEFORE the listening
    # verification.  If the parent dies during the poll loop (SIGINT,
    # OOM, kernel pause), the pidfile is still on disk so the next
    # ``hermes egress stop`` can clean up the orphan.  Failure paths
    # below unlink the pidfile when they kill the child.
    pidfile = _pidfile()
    try:
        _write_pidfile_safely(pidfile, proc.pid)
    except RuntimeError:
        # Kill the orphan so we don't leave a daemon nobody can stop.
        _kill_and_wait(proc, grace_seconds=2)
        raise

    # Poll-with-timeout instead of an unconditional 5s sleep.  The Go
    # binary normally comes up in <200ms; falling through within 100ms
    # of liveness keeps Docker container creation snappy.
    #
    # We scope a Ctrl-C handler around the poll loop so an operator who
    # hits Ctrl-C while waiting for ``hermes egress start`` doesn't leak
    # an orphan with the port bound.
    #
    # Probe the CONFIGURED bind host, not loopback unconditionally — on
    # Linux the daemon binds the docker bridge gateway, where a loopback
    # connect never succeeds and we'd kill a healthy daemon as "never
    # came up".
    listen_hp = _read_http_listen_from_config()
    if listen_hp is not None:
        probe_host, tunnel_port = listen_hp
    else:
        probe_host, tunnel_port = "127.0.0.1", _DEFAULT_TUNNEL_PORT
    listening = False

    def _interrupt_handler(_signum, _frame):  # pragma: no cover - signal path
        # Kill the child and unlink the pidfile, then re-raise so the
        # caller sees the interrupt.
        _kill_and_wait(proc, grace_seconds=2)
        try:
            pidfile.unlink()
        except FileNotFoundError:
            pass
        raise KeyboardInterrupt()

    prev_sigint = None
    prev_sigterm = None
    install_handlers = (
        platform.system() != "Windows"
        and threading.current_thread() is threading.main_thread()
    )
    if install_handlers:
        prev_sigint = signal.signal(signal.SIGINT, _interrupt_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, _interrupt_handler)
    try:
        deadline = time.time() + _STARTUP_GRACE_SECONDS
        # Do-while shape: check listening at least once even when the
        # grace window is 0 (test harness / synchronous fast-path).
        while True:
            if proc.poll() is not None:
                tail = _tail_log(log_path, lines=20)
                try:
                    pidfile.unlink()
                except FileNotFoundError:
                    pass
                raise RuntimeError(
                    f"iron-proxy exited immediately (code {proc.returncode}). "
                    f"Last log lines:\n{tail}"
                )
            if _port_listening(probe_host, tunnel_port):
                listening = True
                break
            if time.time() >= deadline:
                break
            time.sleep(0.1)
    finally:
        if install_handlers:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

    # Final exit check — process may have died right at deadline.
    if proc.poll() is not None:
        tail = _tail_log(log_path, lines=20)
        try:
            pidfile.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"iron-proxy exited immediately (code {proc.returncode}). "
            f"Last log lines:\n{tail}"
        )

    # The previous version of this code treated "process still alive at
    # deadline" as success.  That left iron-proxy running but
    # non-listening on the port, with a pidfile pointing at it —
    # subsequent restarts would fail with "address in use" because the
    # orphan still held the port.  Require port-listening for success.
    if not listening:
        tail = _tail_log(log_path, lines=20)
        _kill_and_wait(proc, grace_seconds=2)
        try:
            pidfile.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"iron-proxy did not bind {probe_host}:{tunnel_port} within "
            f"{_STARTUP_GRACE_SECONDS}s.  Process was killed.  "
            f"Last log lines:\n{tail}"
        )

    logger.info("Started iron-proxy pid=%s config=%s", proc.pid, cfg)
    return get_status()


def _write_pidfile_safely(pidfile: Path, pid: int) -> None:
    """Write ``pid`` to ``pidfile`` with O_EXCL + O_NOFOLLOW + ownership check.

    O_EXCL means "another start is in progress" if the file already
    exists with a live owner — we cleanly fail rather than racing.  When
    the existing pidfile points at a dead pid (stale crash), we
    explicitly unlink it before retrying once.

    Side effect: also persists the in-process nonce to disk so
    cross-CLI-invocation ``_pid_alive`` checks (start in one process,
    stop in another) can still defeat PID recycling.
    """
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(pidfile), open_flags, 0o600)
    except FileExistsError:
        # Pidfile already exists.  If it points at a live iron-proxy,
        # caller's _read_pid + _pid_alive at the top of start_proxy
        # should already have returned.  Reaching here means EITHER
        # the previous _pid_alive check raced (rare; another start in
        # flight), OR a stale pidfile survived a crash.  Discriminate
        # and retry once with O_TRUNC if stale.
        existing_pid = _read_pid()
        if existing_pid and _pid_alive(existing_pid):
            raise RuntimeError(
                f"Another iron-proxy start appears to be in progress "
                f"(pidfile {pidfile} -> pid {existing_pid}).  "
                f"Run `hermes egress stop` if that proxy is stuck."
            )
        # Stale — unlink and retry.
        try:
            pidfile.unlink()
        except FileNotFoundError:
            pass
        fd = os.open(str(pidfile), open_flags, 0o600)
    except OSError as exc:
        # ELOOP from a planted symlink at the pidfile path.
        raise RuntimeError(
            f"Refusing to write pidfile {pidfile}: {exc}.  "
            "Remove that path manually and retry."
        ) from exc

    try:
        # Ownership check — same st_uid pattern the log file uses.
        try:
            st = os.fstat(fd)
            if hasattr(os, "getuid") and st.st_uid != os.getuid():
                raise RuntimeError(
                    f"pidfile {pidfile} has unexpected owner uid={st.st_uid}"
                )
        except AttributeError:
            pass  # Windows
        os.write(fd, str(pid).encode("utf-8"))
    finally:
        os.close(fd)

    # Persist the nonce next to the pidfile (sibling, 0o600).
    # ``stop_proxy`` in a separate CLI invocation can read this and use
    # it to confirm the pid still refers to our binary even though the
    # module-global ``_proxy_nonce`` is fresh in the new process.
    if _proxy_nonce:
        noncefile = pidfile.with_suffix(".nonce")
        nfd = -1
        try:
            nopen = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                nopen |= os.O_NOFOLLOW
            nfd = os.open(str(noncefile), nopen, 0o600)
            os.write(nfd, _proxy_nonce.encode("utf-8"))
        except OSError:
            # Best-effort.  Without the nonce file we fall back to
            # argv0-basename matching, which is what we did before.
            pass
        finally:
            if nfd >= 0:
                try:
                    os.close(nfd)
                except OSError:
                    pass


def _kill_and_wait(proc: "subprocess.Popen", *, grace_seconds: int = 2) -> None:
    """Best-effort SIGTERM → wait → SIGKILL for a child we own."""
    try:
        proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass


def _build_proxy_subprocess_env(
    *,
    extra_env: Optional[Dict[str, str]] = None,
    refresh_from_bitwarden: bool = False,
    bitwarden_config: Optional[Dict] = None,
) -> Dict[str, str]:
    """Construct the minimal env for the iron-proxy subprocess.

    Allowlists infrastructure vars (PATH, HOME, locale) plus the env vars
    named in ``load_mappings()`` (the real upstream secrets the proxy
    needs to do the swap).  Everything else is stripped — see
    ``_PROXY_SUBPROCESS_ENV_STRIP`` for proxy chain protection.

    When ``refresh_from_bitwarden=True`` AND ``bitwarden_config`` is
    populated, fetches upstream secrets via the BSM SDK at startup and
    merges them in.  This is what delivers the rotation guarantee
    promised by ``credential_source: bitwarden`` — without it, rotating
    a key in the Bitwarden web app doesn't reach the proxy.
    """

    env: Dict[str, str] = {}
    parent = os.environ
    for name in _PROXY_SUBPROCESS_ENV_ALLOWLIST:
        if name in parent:
            env[name] = parent[name]

    # The proxy reads the real upstream secrets from its OWN env, indexed
    # by ``m.real_env_name`` in the YAML config's ``secrets.source.var``
    # field.  Forward those — but only those.
    needed = {m.real_env_name for m in load_mappings()}
    for name in needed:
        if name in parent:
            env[name] = parent[name]

    # Optional Bitwarden refresh path.  Pulled lazily so the proxy module
    # doesn't hard-depend on the bitwarden module being importable in
    # every install.
    if refresh_from_bitwarden and bitwarden_config:
        try:
            from agent.secret_sources import bitwarden as bw
            access_token_name = bitwarden_config.get(
                "access_token_env", "BWS_ACCESS_TOKEN"
            )
            access_token = parent.get(access_token_name, "").strip()
            project_id = bitwarden_config.get("project_id", "")
            if access_token and project_id:
                secrets, warnings = bw.fetch_bitwarden_secrets(
                    access_token=access_token,
                    project_id=project_id,
                    cache_ttl_seconds=0,
                    use_cache=False,
                )
                # Only inject env names we have a mapping for — extra
                # secrets in the BW project shouldn't leak into the proxy
                # process unless they're going to be used by the swap.
                missing = sorted(needed - set(secrets))
                for n in needed:
                    if n in secrets:
                        env[n] = secrets[n]
                if missing:
                    # stephenschoettler #1: don't silently keep stale
                    # host-env values when BWS mode was explicitly
                    # selected.  An operator on credential_source=bitwarden
                    # picked it specifically to get rotation; falling back
                    # to parent env reintroduces the bug class the mode
                    # is supposed to defeat.  ``allow_env_fallback`` is the
                    # documented, deliberate opt-out — honor it here exactly
                    # as the empty-token branch below does (the error
                    # message tells operators to set it, so it must work).
                    if not (bitwarden_config or {}).get("allow_env_fallback"):
                        raise RuntimeError(
                            f"Bitwarden refresh did not return secrets for "
                            f"{missing}.  Either add the secrets to your BWS "
                            f"project, switch to credential_source: env via "
                            f"`hermes egress setup --no-bitwarden`, or set "
                            f"`proxy.allow_env_fallback: true` in config.yaml "
                            f"to opt into the legacy host-env fallback."
                        )
                    logger.warning(
                        "Bitwarden refresh did not return secrets for %s — "
                        "falling back to host env for those names "
                        "(allow_env_fallback=true).",
                        missing,
                    )
                # bws warnings are non-secret status messages (e.g. "no
                # project found", "rate limited"), but the taint analyzer
                # can't tell that — log the count and let the operator
                # rerun under verbose if they need detail.
                if warnings:
                    logger.warning(
                        "Bitwarden refresh produced %d warning(s); "
                        "run `hermes secrets bitwarden status` for detail.",
                        len(warnings),
                    )
            else:
                # NOTE: deliberately do not interpolate access_token_name
                # in the log message — CodeQL's taint analyzer treats
                # bitwarden_config values as secret-tainted (it can't
                # distinguish the env-var NAME from the env-var VALUE).
                # The name is non-secret but logging it just trips the
                # check for no real benefit.
                if not (bitwarden_config or {}).get("allow_env_fallback"):
                    raise RuntimeError(
                        "credential_source=bitwarden but the access-token "
                        "env or project_id is empty.  Either set both, "
                        "switch to credential_source: env, or set "
                        "`proxy.allow_env_fallback: true` to opt into "
                        "the legacy fallback behaviour."
                    )
                logger.warning(
                    "credential_source=bitwarden but access-token env or "
                    "project_id is empty — proxy will fall back to parent env "
                    "(allow_env_fallback=true).",
                )
        except (ImportError,) as exc:
            # Truly unrecoverable import failure: log + fall through to
            # the env path so we don't completely block start.  Callers
            # that want strict mode can pre-check at the wizard layer.
            logger.warning(
                "Bitwarden refresh module unavailable at proxy start, "
                "falling back to parent env: %s", exc,
            )

    # Caller-supplied overrides win.  This is intentionally last so the
    # wizard can inject ad-hoc test secrets without recomputing the BW
    # path.
    if extra_env:
        env.update(extra_env)

    # Strip proxy-recursion-risk vars regardless of how they got in.
    for name in _PROXY_SUBPROCESS_ENV_STRIP:
        env.pop(name, None)

    env.setdefault("NO_COLOR", "1")
    return env


def stop_proxy() -> bool:
    """Stop the managed iron-proxy.  Returns True if it was running."""

    global _proxy_nonce

    def _cleanup_state_files() -> None:
        """Best-effort cleanup of pidfile + persisted nonce."""
        _pidfile().unlink(missing_ok=True)
        try:
            _persisted_nonce_path().unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        _cleanup_state_files()
        _proxy_nonce = None
        return False

    # Capture starttime BEFORE signalling so we can compare after the
    # grace window — if the pid got recycled mid-wait, the starttime
    # changes and we abort the SIGKILL.
    starttime_before = _pid_proc_starttime(pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _cleanup_state_files()
        _proxy_nonce = None
        return False

    # Wait up to 5s for graceful exit, then SIGKILL.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        # Verify the pid hasn't been recycled before delivering SIGKILL.
        # Two checks:
        #   1. /proc/<pid>/stat starttime is unchanged (Linux)
        #   2. _pid_alive() still says it's an iron-proxy process
        starttime_after = _pid_proc_starttime(pid)
        recycled = (
            starttime_before is not None
            and starttime_after is not None
            and starttime_before != starttime_after
        ) or not _pid_alive(pid)
        if recycled:
            logger.warning(
                "iron-proxy pid=%s appears recycled before SIGKILL; "
                "not killing.", pid,
            )
        else:
            try:
                os.kill(pid, _KILL_SIGNAL)
            except ProcessLookupError:
                pass

    _cleanup_state_files()
    _proxy_nonce = None
    logger.info("Stopped iron-proxy pid=%s", pid)
    return True


def get_status() -> ProxyStatus:
    """Snapshot the current proxy state — does NOT start anything.

    Crucially, this is called per Docker-container-create when egress
    enforcement is on.  It must not have side-effects (no mkdir, no
    binary version subprocess that takes 30s on a hung binary).  The
    state dir is read-only here.
    """

    status = ProxyStatus()
    listen_hp = _read_http_listen_from_config()
    if listen_hp is not None:
        probe_host, status.tunnel_port = listen_hp
    else:
        probe_host = "127.0.0.1"
        status.tunnel_port = _DEFAULT_TUNNEL_PORT

    binary = find_iron_proxy(install_if_missing=False)
    if binary:
        status.binary_path = binary
        # Cached — see iron_proxy_version().  First call still costs one
        # subprocess; subsequent calls in the same process are dict
        # lookups.
        status.binary_version = iron_proxy_version(binary)

    state = _proxy_state_dir_ro()
    cfg = state / "proxy.yaml"
    ca = state / "ca.crt"
    if cfg.exists():
        status.config_path = cfg
    if ca.exists():
        status.ca_cert_path = ca

    pid = _read_pid()
    if pid and _pid_alive(pid):
        status.pid = pid
        # Probe the configured bind host — on Linux that's the docker
        # bridge gateway, where a loopback connect would report a healthy
        # daemon as "not listening".
        status.listening = _port_listening(probe_host, status.tunnel_port)

    return status


def _read_tunnel_port_from_config() -> Optional[int]:
    listen = _read_http_listen_from_config()
    if listen is None:
        return None
    return listen[1]


def _read_http_listen_from_config() -> Optional[Tuple[str, int]]:
    """Return ``(host, port)`` of the configured sandbox-facing listener.

    Reads ``proxy.tunnel_listen`` — the CONNECT/MITM listener sandboxes
    hit via ``HTTPS_PROXY`` — falling back to ``proxy.http_listen`` for
    configs written before the tunnel/http listener-role split.

    The bind host matters for liveness probes: on Linux the daemon binds
    the docker bridge gateway (e.g. ``172.17.0.1``), where a loopback
    connect would report "not listening" for a perfectly healthy daemon.
    """

    cfg = _proxy_state_dir_ro() / "proxy.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    proxy_block = (data or {}).get("proxy") or {}
    # The CLI/Docker side calls this "the tunnel port" because that's how
    # sandboxes use it (HTTPS_PROXY) — on the iron-proxy side it's the
    # tunnel_listen (CONNECT + MITM).  http_listen is the plain-HTTP
    # forward listener on tunnel_port+1.
    listen = proxy_block.get("tunnel_listen") or proxy_block.get("http_listen") or ""
    if not isinstance(listen, str) or ":" not in listen:
        return None
    host, _, port_s = listen.rpartition(":")
    try:
        port = int(port_s)
    except ValueError:
        return None
    return (host or "127.0.0.1", port)


def _port_listening(host: str, port: int) -> bool:
    """Cheap TCP connect probe — True iff something accepts on host:port."""

    import socket

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _tail_log(path: Path, *, lines: int = 20) -> str:
    if not path.exists():
        return "(no log file)"
    try:
        data = path.read_bytes()[-8192:]
        return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"(could not read log: {exc})"


# ---------------------------------------------------------------------------
# Test hook
# ---------------------------------------------------------------------------


def _reset_for_tests() -> None:
    """Clear module-level caches so tests get a fresh start.

    This module owns two mutable globals that need reset between tests:
    - ``_VERSION_CACHE`` — subprocess output cache keyed by binary path.
    - ``_proxy_nonce`` — the strong-proof token written by ``start_proxy``
      and read by ``_pid_alive`` to defeat PID recycling.

    Today the repo's tests run each file in its own subprocess (per
    AGENTS.md) so leakage is bounded, but any in-process caller
    (notebooks, ad-hoc scripts, ``pytest -p no:xdist``) would otherwise
    see whichever values were probed first regardless of subsequent
    ``install_iron_proxy(force=True)`` or ``start_proxy`` calls.
    """

    global _proxy_nonce
    _VERSION_CACHE.clear()
    _proxy_nonce = None


# Make a small set of symbols available without underscored access.
__all__ = [
    "ProxyStatus",
    "TokenMapping",
    "build_proxy_config",
    "discover_provider_mappings",
    "discover_blocked_providers",
    "discover_uncovered_providers",
    "ensure_audit_log",
    "ensure_ca_cert",
    "find_iron_proxy",
    "get_status",
    "install_iron_proxy",
    "iron_proxy_version",
    "load_mappings",
    "merge_mappings",
    "mint_proxy_token",
    "start_proxy",
    "stop_proxy",
    "write_mappings",
    "write_proxy_config",
]
