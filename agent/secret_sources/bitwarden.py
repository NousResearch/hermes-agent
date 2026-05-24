"""Bitwarden Secrets Manager (`bws` CLI) integration.

Hermes pulls API keys from Bitwarden Secrets Manager at process startup
so they don't have to live in plaintext in ``~/.hermes/.env``.

Design summary
--------------

* The ``bws`` binary is auto-installed into ``<hermes_home>/bin/bws`` on
  first use.  Hermes pins one version (``_BWS_VERSION``) and downloads
  the matching asset from the official GitHub Releases page, verifying
  the SHA-256 against the release's published checksum file.
* The access token is stored in ``~/.hermes/.env`` as
  ``BWS_ACCESS_TOKEN`` (or whatever name the user picked in
  ``secrets.bitwarden.access_token_env``).  This is the one
  bootstrap secret — every other provider key can live in Bitwarden.
* Pulling secrets is a single ``bws secret list <project_id>
  --output json`` call.  We cache the result in-process for
  ``cache_ttl_seconds`` so back-to-back ``hermes`` invocations don't
  hammer the API.
* Failures NEVER block Hermes startup.  Missing binary, no network,
  expired token, etc. all emit a one-line warning and continue with
  whatever credentials ``.env`` already had.

The module is intentionally subprocess-driven rather than going through
the ``bitwarden-sdk-secrets`` Python package: one cross-platform binary
is easier to lazy-install than a wheels-with-Rust-extension dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Pinned upstream version.  Bump in a follow-up PR — never auto-resolve
# "latest" because upstream release shape (asset names, CLI flags) is
# allowed to change between majors and we want updates to be deliberate.
_BWS_VERSION = "2.0.0"

_BWS_RELEASE_BASE = (
    f"https://github.com/bitwarden/sdk-sm/releases/download/bws-v{_BWS_VERSION}"
)
_BWS_CHECKSUM_NAME = f"bws-sha256-checksums-{_BWS_VERSION}.txt"

# How long to wait for bws subprocesses and HTTP downloads, in seconds.
_BWS_DOWNLOAD_TIMEOUT = 60
_BWS_RUN_TIMEOUT = 30

# In-process cache so repeated load_hermes_dotenv() calls (CLI startup,
# gateway hot-reload, test suites) don't re-fetch from BSM.
_CacheKey = Tuple[str, str, str]  # (access_token_fingerprint, project_id, server_url)
_CACHE: Dict[_CacheKey, "_CachedFetch"] = {}


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of a single BSM pull."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)   # set into os.environ
    skipped: List[str] = field(default_factory=list)   # already set, not overridden
    warnings: List[str] = field(default_factory=list)  # non-fatal issues
    error: Optional[str] = None                        # fatal: nothing was fetched
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Binary discovery + lazy install
# ---------------------------------------------------------------------------


def _hermes_bin_dir() -> Path:
    """Where Hermes stores its managed binaries.  Profile-aware."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "bin"


def find_bws(*, install_if_missing: bool = False) -> Optional[Path]:
    """Return a path to a usable ``bws`` binary, or None.

    Resolution order:
      1. ``<hermes_home>/bin/bws``  (our managed copy — preferred)
      2. ``shutil.which("bws")``    (system PATH)

    When ``install_if_missing`` is True and neither resolves, this calls
    :func:`install_bws` to download and verify the pinned version.
    """
    managed = _hermes_bin_dir() / _platform_binary_name()
    if managed.exists() and os.access(managed, os.X_OK):
        return managed

    system = shutil.which("bws")
    if system:
        return Path(system)

    if install_if_missing:
        try:
            return install_bws()
        except Exception as exc:  # noqa: BLE001 — never block startup
            logger.warning("bws auto-install failed: %s", exc)
            return None
    return None


def _platform_binary_name() -> str:
    return "bws.exe" if platform.system() == "Windows" else "bws"


def _platform_asset_name() -> str:
    """Map (uname, arch, libc) → the upstream asset filename.

    Asset names follow Rust's target triple convention.  Linux defaults
    to gnu (glibc); we switch to musl only if ldd --version says so.
    """
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        # Universal binary works on both Intel and Apple Silicon — no
        # need to pick a per-arch asset.
        return f"bws-macos-universal-{_BWS_VERSION}.zip"

    if system == "Windows":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return f"bws-{arch}-pc-windows-msvc-{_BWS_VERSION}.zip"

    if system == "Linux":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        libc = "gnu"
        # ldd --version writes to stderr on glibc, stdout on musl.  We
        # don't need bullet-proof detection — getting it wrong falls
        # back to a clear error from the binary loader, which we catch.
        try:
            res = subprocess.run(
                ["ldd", "--version"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if "musl" in (res.stdout + res.stderr).lower():
                libc = "musl"
        except (OSError, subprocess.TimeoutExpired):
            pass
        return f"bws-{arch}-unknown-linux-{libc}-{_BWS_VERSION}.zip"

    raise RuntimeError(
        f"Unsupported platform for bws auto-install: {system} {machine}"
    )


def install_bws(*, force: bool = False) -> Path:
    """Download, verify, and install the pinned ``bws`` binary.

    Returns the path to the installed executable.  Raises on any
    failure (network, checksum, extraction) — callers in the auto-install
    path catch these; the user-facing ``hermes secrets bitwarden setup``
    surface lets them propagate so the wizard can show a clear error.
    """
    bin_dir = _hermes_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / _platform_binary_name()

    if target.exists() and not force:
        return target

    asset_name = _platform_asset_name()
    asset_url = f"{_BWS_RELEASE_BASE}/{asset_name}"
    checksum_url = f"{_BWS_RELEASE_BASE}/{_BWS_CHECKSUM_NAME}"

    with tempfile.TemporaryDirectory(prefix="hermes-bws-") as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / asset_name
        checksum_path = tmp / _BWS_CHECKSUM_NAME

        logger.info("Downloading %s", asset_url)
        _http_download(asset_url, zip_path)
        _http_download(checksum_url, checksum_path)

        expected = _expected_sha256(checksum_path, asset_name)
        actual = _sha256_file(zip_path)
        if expected.lower() != actual.lower():
            raise RuntimeError(
                f"Checksum mismatch for {asset_name}: "
                f"expected {expected}, got {actual}"
            )

        with zipfile.ZipFile(zip_path) as zf:
            member = _pick_zip_member(zf, _platform_binary_name())
            zf.extract(member, tmp)
            extracted = tmp / member

        # Move into place atomically.  We write to a sibling tempfile in
        # the final directory so the rename can't cross filesystems.
        fd, staged = tempfile.mkstemp(dir=str(bin_dir), prefix=".bws_")
        os.close(fd)
        shutil.copy2(extracted, staged)
        os.chmod(
            staged,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            | stat.S_IRGRP | stat.S_IXGRP
            | stat.S_IROTH | stat.S_IXOTH,
        )
        os.replace(staged, target)

    logger.info("Installed bws %s at %s", _BWS_VERSION, target)
    return target


def _http_download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent"})
    try:
        with urllib.request.urlopen(req, timeout=_BWS_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _expected_sha256(checksum_file: Path, asset_name: str) -> str:
    """Parse the upstream ``bws-sha256-checksums-X.Y.Z.txt`` file.

    Format is the standard ``sha256sum`` output: ``<hex>  <filename>``,
    one per line.
    """
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


def _pick_zip_member(zf: zipfile.ZipFile, binary_name: str) -> str:
    """Find the binary inside the upstream zip.

    Historically the archive has been flat (``bws`` at the root) but we
    tolerate a top-level directory just in case upstream changes.
    """
    candidates = [n for n in zf.namelist() if n.split("/")[-1] == binary_name]
    if not candidates:
        raise RuntimeError(
            f"Could not find {binary_name} inside downloaded archive "
            f"(members: {zf.namelist()[:5]}...)"
        )
    # Prefer the shortest path (i.e. root over nested) for determinism.
    candidates.sort(key=len)
    return candidates[0]


# ---------------------------------------------------------------------------
# Secret fetch + apply
# ---------------------------------------------------------------------------


def _token_fingerprint(token: str) -> str:
    """SHA-256 prefix used as a cache key — never logged, never displayed."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def fetch_bitwarden_secrets(
    *,
    access_token: str,
    project_id: str,
    binary: Optional[Path] = None,
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    server_url: str = "",
) -> Tuple[Dict[str, str], List[str]]:
    """Pull the secrets for ``project_id`` from Bitwarden Secrets Manager.

    Returns ``(secrets_dict, warnings_list)``.

    Set ``server_url`` to point at a non-default Bitwarden region or a
    self-hosted instance — e.g. ``https://vault.bitwarden.eu`` for EU
    Cloud accounts.  When empty, ``bws`` uses its built-in default
    (``https://vault.bitwarden.com``, US Cloud).  This is plumbed into
    the subprocess as ``BWS_SERVER_URL``.

    Raises :class:`RuntimeError` for fatal conditions (missing binary,
    auth failure, unparseable output).  Callers in the env_loader path
    catch this and emit a single warning; callers in the user-facing
    setup wizard let it propagate.
    """
    if not access_token:
        raise RuntimeError("Bitwarden access token is empty")
    if not project_id:
        raise RuntimeError("Bitwarden project_id is empty")

    cache_key = (_token_fingerprint(access_token), project_id, server_url or "")
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return cached.secrets, []

    bws = binary or find_bws(install_if_missing=True)
    if bws is None:
        raise RuntimeError(
            "bws binary not available — auto-install failed and `bws` is "
            "not on PATH.  Install manually from "
            "https://github.com/bitwarden/sdk-sm/releases or re-run "
            "`hermes secrets bitwarden setup`."
        )

    secrets, warnings = _run_bws_list(bws, access_token, project_id, server_url)
    _CACHE[cache_key] = _CachedFetch(secrets=secrets, fetched_at=time.time())
    return secrets, warnings


def _run_bws_list(
    bws: Path, access_token: str, project_id: str, server_url: str = ""
) -> Tuple[Dict[str, str], List[str]]:
    cmd = [str(bws), "secret", "list", project_id, "--output", "json"]
    env = os.environ.copy()
    env["BWS_ACCESS_TOKEN"] = access_token
    # Make sure we're not echoing telemetry / colour codes into json.
    env.setdefault("NO_COLOR", "1")
    # Region / self-hosted support.  bws defaults to https://vault.bitwarden.com
    # (US Cloud); EU Cloud users need https://vault.bitwarden.eu, and
    # self-hosted users need their own URL.  When unset, fall back to whatever
    # BWS_SERVER_URL the caller already had in their shell env (preserved by
    # the copy above) so manual overrides keep working too.
    if server_url:
        env["BWS_SERVER_URL"] = server_url

    try:
        proc = subprocess.run(  # noqa: S603 — bws path is trusted
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=_BWS_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"bws timed out after {_BWS_RUN_TIMEOUT}s fetching secrets"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke bws: {exc}") from exc

    if proc.returncode != 0:
        # bws writes auth/network errors to stderr in plain English.
        # Strip ANSI just in case and surface the first 200 chars.
        err = (proc.stderr or proc.stdout or "").strip().replace("\x1b", "")
        raise RuntimeError(
            f"bws exited {proc.returncode}: {err[:200]}"
        )

    raw = proc.stdout.strip()
    if not raw:
        return {}, ["bws returned no output (empty project?)"]

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bws returned non-JSON output: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError(
            f"bws returned unexpected shape: {type(payload).__name__}"
        )

    secrets: Dict[str, str] = {}
    warnings: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value = item.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not _is_valid_env_name(key):
            warnings.append(
                f"Skipping secret {key!r}: not a valid env-var name"
            )
            continue
        secrets[key] = value
    return secrets, warnings


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


# ---------------------------------------------------------------------------
# Public entry point — called from hermes_cli.env_loader
# ---------------------------------------------------------------------------


def apply_bitwarden_secrets(
    *,
    enabled: bool,
    access_token_env: str = "BWS_ACCESS_TOKEN",
    project_id: str = "",
    override_existing: bool = False,
    cache_ttl_seconds: float = 300,
    auto_install: bool = True,
    server_url: str = "",
) -> FetchResult:
    """Pull secrets from BSM and set them on ``os.environ``.

    This is the function ``load_hermes_dotenv()`` calls after the .env
    files have loaded.  It is intentionally defensive — any failure
    returns a :class:`FetchResult` with ``error`` set; it never raises.

    ``server_url`` selects the Bitwarden region or self-hosted endpoint
    (e.g. ``https://vault.bitwarden.eu`` for EU Cloud).  Empty string
    means use ``bws``'s default (US Cloud).

    Parameters mirror the ``secrets.bitwarden.*`` config keys so the
    caller can just splat the dict in.
    """
    result = FetchResult()

    if not enabled:
        return result

    access_token = os.environ.get(access_token_env, "").strip()
    if not access_token:
        result.error = (
            f"secrets.bitwarden.enabled is true but {access_token_env} is "
            "not set.  Run `hermes secrets bitwarden setup`."
        )
        return result

    if not project_id:
        result.error = (
            "secrets.bitwarden.project_id is empty.  "
            "Run `hermes secrets bitwarden setup`."
        )
        return result

    binary = find_bws(install_if_missing=auto_install)
    result.binary_path = binary
    if binary is None:
        result.error = (
            "bws binary not available and auto-install is disabled.  "
            "Run `hermes secrets bitwarden setup` to install."
        )
        return result

    try:
        secrets, warnings = fetch_bitwarden_secrets(
            access_token=access_token,
            project_id=project_id,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
            server_url=server_url,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings)

    for key, value in secrets.items():
        if key == access_token_env:
            # Don't let BSM clobber the very token we used to fetch
            # itself — that would be a footgun if someone stored the
            # token as a BSM secret too.
            result.skipped.append(key)
            continue
        if not override_existing and os.environ.get(key):
            result.skipped.append(key)
            continue
        os.environ[key] = value
        result.applied.append(key)

    return result


# ---------------------------------------------------------------------------
# Test hook — used by hermetic tests to flush the cache between cases.
# ---------------------------------------------------------------------------


def _reset_cache_for_tests() -> None:
    _CACHE.clear()
