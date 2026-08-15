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

import base64
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from agent.secret_sources._cache import (
    CachedFetch as _CachedFetch,
    DiskCache,
    FetchResult,
    is_valid_env_name as _is_valid_env_name,
)
from agent.secret_sources.base import ErrorKind, SecretSource
from agent.secret_sources.base import get_source_environment

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
_CachePolicy = Tuple[bool, bool, str]  # (encrypted_mode, transition_pending, cache_path)
_CACHE: Dict[_CacheKey, _CachedFetch] = {}
_CACHE_POLICY: Dict[_CacheKey, _CachePolicy] = {}
_ENCRYPTED_CACHE_INVALIDATIONS: set[str] = set()

# Disk-persisted cache so back-to-back CLI invocations (e.g. `hermes chat -q ...`
# called from scripts, cron, the gateway forking new agents) don't each pay the
# ~380ms `bws secret list` tax. The in-process _CACHE above only saves repeated
# fetches WITHIN one process; this saves repeated fetches ACROSS processes.
#
# Layout: one JSON object per cache key, written atomically with mode 0600 in
# <hermes_home>/cache/bws_cache.json. The file holds only the secret VALUES,
# never the access token. It's plaintext-equivalent to ~/.hermes/.env (which
# we already accept) but kept out of the .env file so users editing it won't
# accidentally commit BSM-sourced secrets. The atomic-write/0600/TTL mechanics
# live in agent.secret_sources._cache.DiskCache, shared with the other backends.
_DISK_CACHE_BASENAME = "bws_cache.json"
_ENCRYPTED_CACHE_BASENAME = "bws_cache.enc.json"
_ENCRYPTED_CACHE_INVALIDATION_BASENAME = ".bws_cache.enc.invalidated"
_ENCRYPTED_CACHE_VERSION = 2
_ENCRYPTED_CACHE_LEGACY_VERSION = 1
_ENCRYPTED_CACHE_INVALIDATED_VERSION = 0
_ENCRYPTED_CACHE_LEGACY_INFO = b"hermes-bws-encrypted-cache-v1"
_ENCRYPTED_CACHE_LEGACY_UNBOUND_SERVER = (
    "__hermes_legacy_inherited_endpoint_unbound__"
)
# Version 2 fixes these scrypt parameters as part of the file format.  N=2^15,
# r=8 uses about 32 MiB per derivation, adding meaningful offline-guess cost
# without making an opt-in cache read impractical on supported machines.
_ENCRYPTED_CACHE_SCRYPT_N = 2**15
_ENCRYPTED_CACHE_SCRYPT_R = 8
_ENCRYPTED_CACHE_SCRYPT_P = 1


class _BwsFetchError(RuntimeError):
    """A displayed fetch error with taxonomy bound before decoration."""

    def __init__(self, message: str, error_kind: ErrorKind) -> None:
        super().__init__(message)
        self.error_kind = error_kind


class _PlaintextCleanupResult(Enum):
    """Outcome of removing a plaintext predecessor during migration."""

    COMPLETE = auto()
    NEWER_PLAINTEXT = auto()
    FAILED = auto()


def _cache_put(
    cache_key: _CacheKey,
    entry: _CachedFetch,
    *,
    encrypted_mode: bool,
    transition_pending: bool = False,
    home_path: Optional[Path] = None,
) -> None:
    """Store an L1 entry together with its storage-policy state."""
    _CACHE[cache_key] = entry
    _CACHE_POLICY[cache_key] = (
        encrypted_mode,
        transition_pending,
        str(_disk_cache_path(home_path)),
    )


def _cache_drop(cache_key: _CacheKey) -> None:
    """Evict an L1 entry when the caller disables fresh caching."""
    _CACHE.pop(cache_key, None)
    _CACHE_POLICY.pop(cache_key, None)


def _cache_drop_encrypted_home(home_path: Optional[Path]) -> None:
    """Evict every encrypted L1 entry sharing a home-scoped cache file."""
    disk_path = str(_disk_cache_path(home_path))
    for cache_key, policy in list(_CACHE_POLICY.items()):
        if policy[0] and policy[2] == disk_path:
            _cache_drop(cache_key)


def _encrypted_cache_identity(
    cache_key: _CacheKey,
    home_path: Optional[Path],
) -> str:
    """Identify the shared encrypted cache file, independent of route."""
    del cache_key
    return str(_encrypted_disk_cache_path(home_path))


def _encrypted_cache_invalidation_marker_path(
    home_path: Optional[Path],
) -> Path:
    return _encrypted_disk_cache_path(home_path).with_name(
        _ENCRYPTED_CACHE_INVALIDATION_BASENAME
    )


def _encrypted_cache_invalidation_marker_paths(
    home_path: Optional[Path],
) -> Tuple[Path, ...]:
    """Return cache-local then home-level durable veto locations."""
    primary = _encrypted_cache_invalidation_marker_path(home_path)
    fallback = primary.parent.parent / _ENCRYPTED_CACHE_INVALIDATION_BASENAME
    return (primary,) if fallback == primary else (primary, fallback)


def _persist_encrypted_cache_invalidation_marker(
    home_path: Optional[Path],
) -> None:
    """Best-effort durable veto for a failed shared-cache invalidation.

    The atomic path is preferred.  If replacement itself is unavailable,
    write the non-secret marker in place so a fresh process still fails
    closed; malformed or interrupted marker contents are treated as a veto
    by the reader.
    """
    payload = {"version": 1, "invalidated": True}
    for path in _encrypted_cache_invalidation_marker_paths(home_path):
        tmp: Optional[str] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".bws_cache_enc_marker_",
                suffix=".tmp",
                dir=str(path.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
            return
        except Exception:  # noqa: BLE001 — try the home-level fallback
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o600)
            return
        except Exception:  # noqa: BLE001 — try the next durable location
            continue


def _remove_encrypted_cache_invalidation_markers(
    home_path: Optional[Path],
) -> bool:
    removed = True
    for marker_path in _encrypted_cache_invalidation_marker_paths(home_path):
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            removed = False
    return removed


def _read_encrypted_cache_invalidation_marker(
    home_path: Optional[Path],
) -> bool:
    for marker_path in _encrypted_cache_invalidation_marker_paths(home_path):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError, json.JSONDecodeError):
            return True
        if isinstance(marker, dict) and marker.get("invalidated") is True:
            return True
    return False


def _mark_encrypted_cache_invalidated(
    cache_key: _CacheKey,
    home_path: Optional[Path],
) -> None:
    _ENCRYPTED_CACHE_INVALIDATIONS.add(
        _encrypted_cache_identity(cache_key, home_path)
    )
    _cache_drop_encrypted_home(home_path)
    _persist_encrypted_cache_invalidation_marker(home_path)


def _clear_encrypted_cache_invalidated(
    cache_key: _CacheKey,
    home_path: Optional[Path],
) -> bool:
    removed = True
    for marker_path in _encrypted_cache_invalidation_marker_paths(home_path):
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            removed = False
    identity = _encrypted_cache_identity(cache_key, home_path)
    if removed:
        _ENCRYPTED_CACHE_INVALIDATIONS.discard(identity)
    else:
        _ENCRYPTED_CACHE_INVALIDATIONS.add(identity)
    return removed


def _encrypted_cache_was_invalidated(
    cache_key: _CacheKey,
    home_path: Optional[Path],
) -> bool:
    identity = _encrypted_cache_identity(cache_key, home_path)
    if identity in _ENCRYPTED_CACHE_INVALIDATIONS:
        return True
    return _read_encrypted_cache_invalidation_marker(home_path)


def _cache_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key to a stable string for JSON storage."""
    token_fp, project_id, server_url = cache_key
    return f"{token_fp}|{project_id}|{server_url}"


def _encrypted_cache_context(cache_key: _CacheKey) -> Dict[str, str]:
    """Return the non-secret routing context persisted by encrypted v2."""
    _, project_id, server_url = cache_key
    return {"project_id": project_id, "server_url": server_url}


def _encrypted_cache_aad(cache_key: _CacheKey) -> bytes:
    """Bind encrypted v2 entries to their format and routing context."""
    context = _encrypted_cache_context(cache_key)
    return json.dumps(
        {
            "format": "hermes-bws-encrypted-cache-v2",
            **context,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


_DISK_CACHE: DiskCache = DiskCache(
    _DISK_CACHE_BASENAME, key_serializer=_cache_key_str
)


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Return the disk cache path under hermes_home/cache/.

    Thin wrapper over the shared DiskCache, kept for tests and any direct
    callers; falls back to `$HERMES_HOME` / `~/.hermes` when home is None.
    """
    return _DISK_CACHE.path(home_path)


def _encrypted_disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Return the encrypted disk cache path under hermes_home/cache/."""
    from agent.secret_sources._cache import resolve_cache_home

    return resolve_cache_home(home_path) / "cache" / _ENCRYPTED_CACHE_BASENAME


def _encrypted_cache_invalidation_warning(home_path: Optional[Path]) -> str:
    """Describe an encrypted-cache invalidation that could not be persisted."""
    return (
        "Encrypted Bitwarden disk-cache invalidation did not complete; "
        "stale encrypted secrets may remain at "
        f"{_encrypted_disk_cache_path(home_path)}"
    )


def _snapshot_bws_source_environment() -> Dict[str, str]:
    """Freeze the environment used for token, endpoint, and child routing."""
    source_env = get_source_environment()
    if source_env is os.environ:
        from tools.environments.local import build_subprocess_env

        return build_subprocess_env(
            scrub_secrets=False,
            inherit_profile_home=False,
        )
    return dict(source_env)


def _effective_server_url(
    server_url: str,
    source_env: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve the endpoint used by both bws and every cache identity."""
    explicit = str(server_url or "").strip()
    if explicit:
        return explicit
    if source_env is None:
        source_env = _snapshot_bws_source_environment()
    inherited = source_env.get("BWS_SERVER_URL", "")
    return str(inherited or "").strip()


def _invalidate_encrypted_disk_cache(home_path: Optional[Path] = None) -> bool:
    """Drop an opposite-mode encrypted cache after a plaintext write.

    If unlink is denied, replace the payload with a non-secret tombstone so a
    later encrypted-mode read cannot resurrect the old ciphertext.
    """
    path = _encrypted_disk_cache_path(home_path)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        tmp: Optional[str] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".bws_cache_enc_invalidated_",
                suffix=".tmp",
                dir=str(path.parent),
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "version": _ENCRYPTED_CACHE_INVALIDATED_VERSION,
                        "invalidated": True,
                    },
                    handle,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
            return True
        except Exception:  # noqa: BLE001 — cache invalidation is best-effort
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return False


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
                text=True, encoding='utf-8', errors='replace',
                timeout=2,
                stdin=subprocess.DEVNULL,
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
            # Zip-slip guard: a malicious archive can carry member names like
            # ``../../etc/cron.d/x`` or absolute paths.  ``ZipFile.extract``
            # joins the member onto ``tmp`` without verifying the result stays
            # inside it, so validate containment before touching the disk.
            extracted = _safe_extract_member(zf, member, tmp)

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


def _safe_extract_member(
    zf: zipfile.ZipFile, member: str, dest_dir: Path
) -> Path:
    """Extract a single archive member, refusing path traversal.

    ``ZipFile.extract`` will happily honour member names containing
    ``../`` or absolute paths, letting a malicious archive write outside
    ``dest_dir`` (a "zip-slip").  We resolve the would-be target and
    confirm it stays within ``dest_dir`` before extracting.
    """
    dest_root = os.path.realpath(dest_dir)
    target = os.path.realpath(os.path.join(dest_root, member))
    # ``commonpath`` raises ValueError for e.g. different drives on
    # Windows; treat that as an escape too.
    try:
        contained = os.path.commonpath([dest_root, target]) == dest_root
    except ValueError:
        contained = False
    if not contained or target == dest_root:
        raise RuntimeError(
            f"Refusing to extract unsafe archive member {member!r}: "
            f"it escapes the extraction directory"
        )
    zf.extract(member, dest_root)
    return Path(target)


# ---------------------------------------------------------------------------
# Secret fetch + apply
# ---------------------------------------------------------------------------


def _token_fingerprint(token: str) -> str:
    """SHA-256 prefix used as a cache key — never logged, never displayed."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def _derive_encrypted_cache_key(access_token: str, salt: bytes) -> bytes:
    """Stretch the bootstrap BWS token into a local cache encryption key."""
    return Scrypt(
        salt=salt,
        length=32,
        n=_ENCRYPTED_CACHE_SCRYPT_N,
        r=_ENCRYPTED_CACHE_SCRYPT_R,
        p=_ENCRYPTED_CACHE_SCRYPT_P,
    ).derive(access_token.encode("utf-8"))


def _derive_legacy_encrypted_cache_key(access_token: str, salt: bytes) -> bytes:
    """Derive a version-1 key solely to migrate an existing cache."""
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=_ENCRYPTED_CACHE_LEGACY_INFO,
    ).derive(access_token.encode("utf-8"))


def _encrypted_cache_transition_warning(home_path: Optional[Path]) -> str:
    """Describe an incomplete plaintext-to-encrypted cache transition."""
    return (
        "Encrypted Bitwarden disk-cache transition did not complete; "
        "legacy plaintext secrets may remain at "
        f"{_disk_cache_path(home_path)}"
    )


def _encrypted_cache_newer_plaintext_warning(home_path: Optional[Path]) -> str:
    """Explain why a newer plaintext predecessor remains intentionally."""
    return (
        "A newer plaintext Bitwarden cache remains at "
        f"{_disk_cache_path(home_path)}; encrypted cache data was not served"
    )


def _encrypted_cache_write_warning(home_path: Optional[Path]) -> str:
    """Describe a cache write failure without an incomplete transition."""
    return (
        "Encrypted Bitwarden disk cache could not be written; "
        "live secrets were returned without disk caching at "
        f"{_encrypted_disk_cache_path(home_path)}"
    )


def _encrypted_cache_legacy_endpoint_warning(home_path: Optional[Path]) -> str:
    """Explain why an inherited-endpoint v1 cache requires a live binding."""
    return (
        "A legacy Bitwarden v1 cache used an inherited endpoint and was "
        "re-encrypted without serving its origin-ambiguous secrets; a live "
        "fetch is required to bind the endpoint at "
        f"{_encrypted_disk_cache_path(home_path)}"
    )


def _remove_plaintext_cache_predecessor(
    *,
    cache_key: _CacheKey,
    home_path: Optional[Path],
    encrypted_fetched_at: Optional[float] = None,
) -> _PlaintextCleanupResult:
    """Remove this entry's plaintext predecessor, preserving known others."""
    path = _disk_cache_path(home_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _PlaintextCleanupResult.COMPLETE
    except (OSError, UnicodeError, json.JSONDecodeError):
        # An unreadable or malformed Bitwarden plaintext cache cannot be
        # identified safely.  Preserve the encrypted-mode invariant by
        # requiring its conservative removal before serving encrypted data.
        payload = None
    if isinstance(payload, dict):
        plaintext_key = payload.get("key")
        legacy_inherited_route_alias = (
            bool(cache_key[2])
            and plaintext_key
            == _cache_key_str((cache_key[0], cache_key[1], ""))
        )
        if (
            isinstance(plaintext_key, str)
            and plaintext_key != _cache_key_str(cache_key)
            and not legacy_inherited_route_alias
        ):
            return _PlaintextCleanupResult.COMPLETE
        plaintext_fetched_at = payload.get("fetched_at")
        if (
            encrypted_fetched_at is not None
            and isinstance(plaintext_fetched_at, (int, float))
            and float(plaintext_fetched_at) > encrypted_fetched_at
        ):
            # A newer plaintext write wins the cross-mode race.  Preserve it
            # and make the older encrypted entry unavailable to callers.
            return _PlaintextCleanupResult.NEWER_PLAINTEXT
    try:
        path.unlink()
    except FileNotFoundError:
        return _PlaintextCleanupResult.COMPLETE
    except OSError:
        return _PlaintextCleanupResult.FAILED
    return _PlaintextCleanupResult.COMPLETE


def _write_encrypted_disk_cache(
    *,
    cache_key: _CacheKey,
    access_token: str,
    entry: _CachedFetch,
    home_path: Optional[Path] = None,
    transition_warnings: Optional[List[str]] = None,
    transition_pending_out: Optional[List[bool]] = None,
) -> bool:
    """Persist an encrypted last-good cache entry atomically.

    Best-effort by design: cache write failure must never block a fresh BWS
    fetch.  The raw BWS access token is not stored; it only derives the AES key.
    Return whether the complete encrypted-storage transition succeeded.
    """
    path = _encrypted_disk_cache_path(home_path)
    installed = False
    encrypted_predecessor = path.exists()
    transition_pending = (
        encrypted_predecessor or _disk_cache_path(home_path).exists()
    )
    if transition_pending_out is not None:
        transition_pending_out[:] = [transition_pending]
    try:
        cache_dir = path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(cache_dir, 0o700)
        except OSError:
            pass
        salt = os.urandom(16)
        nonce = os.urandom(12)
        context = _encrypted_cache_context(cache_key)
        aad = _encrypted_cache_aad(cache_key)
        key = _derive_encrypted_cache_key(access_token, salt)
        plaintext = json.dumps(
            {"secrets": entry.secrets, "fetched_at": entry.fetched_at},
            separators=(",", ":"),
        ).encode("utf-8")
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
        payload = {
            "version": _ENCRYPTED_CACHE_VERSION,
            "context": context,
            "salt": _b64e(salt),
            "nonce": _b64e(nonce),
            "ciphertext": _b64e(ciphertext),
        }
        fd, tmp = tempfile.mkstemp(
            prefix=".bws_cache_enc_", suffix=".tmp", dir=str(cache_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
            installed = True
            # Remove the plaintext predecessor even when a durable encrypted
            # veto cannot be cleared; marker failure must not leave readable
            # same-route secrets at rest after the ciphertext is installed.
            cleanup_result = _remove_plaintext_cache_predecessor(
                cache_key=cache_key,
                home_path=home_path,
                encrypted_fetched_at=entry.fetched_at,
            )
            markers_cleared = _clear_encrypted_cache_invalidated(
                cache_key, home_path
            )
            if (
                not markers_cleared
                or cleanup_result is not _PlaintextCleanupResult.COMPLETE
            ):
                transition_pending = True
                if transition_pending_out is not None:
                    transition_pending_out[:] = [True]
                if transition_warnings is not None:
                    if not markers_cleared:
                        marker_warning = _encrypted_cache_invalidation_warning(
                            home_path
                        )
                        if marker_warning not in transition_warnings:
                            transition_warnings.append(marker_warning)
                    if cleanup_result is not _PlaintextCleanupResult.COMPLETE:
                        cleanup_warning = (
                            _encrypted_cache_newer_plaintext_warning(home_path)
                            if cleanup_result
                            is _PlaintextCleanupResult.NEWER_PLAINTEXT
                            else _encrypted_cache_transition_warning(home_path)
                        )
                        if cleanup_warning not in transition_warnings:
                            transition_warnings.append(cleanup_warning)
                return False
            if transition_pending_out is not None:
                transition_pending_out[:] = [False]
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception:  # noqa: BLE001 — best-effort cache only
        if encrypted_predecessor and not installed:
            # Every failed attempt to supersede readable ciphertext must veto
            # that older generation, including failures before os.replace().
            _mark_encrypted_cache_invalidated(cache_key, home_path)
            if transition_warnings is not None:
                warning = _encrypted_cache_invalidation_warning(home_path)
                if warning not in transition_warnings:
                    transition_warnings.append(warning)
        return False


def _retire_legacy_encrypted_cache(
    *,
    cache_key: _CacheKey,
    home_path: Optional[Path],
    transition_warnings: Optional[List[str]],
) -> None:
    """Remove or durably veto every recognized weak v1 artifact."""
    invalidated = _invalidate_encrypted_disk_cache(home_path)
    markers_cleared = (
        _clear_encrypted_cache_invalidated(cache_key, home_path)
        if invalidated
        else False
    )
    if invalidated and markers_cleared:
        return
    _mark_encrypted_cache_invalidated(cache_key, home_path)
    warning = _encrypted_cache_invalidation_warning(home_path)
    if transition_warnings is not None and warning not in transition_warnings:
        transition_warnings.append(warning)


def _read_legacy_encrypted_disk_cache(
    *,
    payload: Dict[str, object],
    cache_key: _CacheKey,
    access_token: str,
    max_age_seconds: float,
    home_path: Optional[Path],
    transition_warnings: Optional[List[str]],
) -> Optional[_CachedFetch]:
    """Authenticate one v1 artifact, harden it to v2, and retire on failure."""
    serialized_key = _cache_key_str(cache_key)
    payload_key = payload.get("key")
    legacy_cache_key = cache_key
    inherited_endpoint_unbound = False
    if payload_key != serialized_key:
        empty_endpoint_key = (cache_key[0], cache_key[1], "")
        if cache_key[2] and payload_key == _cache_key_str(empty_endpoint_key):
            # Base v1 bound its AAD to the configured value, not to an
            # inherited BWS_SERVER_URL. Authenticate and harden it, but keep
            # the origin-ambiguous value outside every routable cache identity.
            legacy_cache_key = empty_endpoint_key
            inherited_endpoint_unbound = True
        else:
            _retire_legacy_encrypted_cache(
                cache_key=cache_key,
                home_path=home_path,
                transition_warnings=transition_warnings,
            )
            return None

    try:
        salt = _b64d(str(payload.get("salt", "")))
        nonce = _b64d(str(payload.get("nonce", "")))
        ciphertext = _b64d(str(payload.get("ciphertext", "")))
        if len(salt) != 16 or len(nonce) != 12 or len(ciphertext) < 16:
            raise ValueError("invalid legacy encrypted-cache field length")
        key = _derive_legacy_encrypted_cache_key(access_token, salt)
        aad = _cache_key_str(legacy_cache_key).encode("utf-8")
        raw = AESGCM(key).decrypt(nonce, ciphertext, aad)
        inner = json.loads(raw.decode("utf-8"))
        if not isinstance(inner, dict):
            raise ValueError("invalid legacy encrypted-cache payload")
        secrets = inner.get("secrets")
        inner_fetched_at = inner.get("fetched_at")
        if (
            not isinstance(secrets, dict)
            or not all(
                isinstance(name, str) and isinstance(value, str)
                for name, value in secrets.items()
            )
            or isinstance(inner_fetched_at, bool)
            or not isinstance(inner_fetched_at, (int, float))
            or not math.isfinite(float(inner_fetched_at))
        ):
            raise ValueError("invalid legacy encrypted-cache schema")
        fetched_at = float(inner_fetched_at)
        entry = _CachedFetch(secrets=dict(secrets), fetched_at=fetched_at)
        migration_cache_key = (
            (cache_key[0], cache_key[1], _ENCRYPTED_CACHE_LEGACY_UNBOUND_SERVER)
            if inherited_endpoint_unbound
            else cache_key
        )
        warning_count = (
            len(transition_warnings) if transition_warnings is not None else 0
        )
        if not _write_encrypted_disk_cache(
            cache_key=migration_cache_key,
            access_token=access_token,
            entry=entry,
            home_path=home_path,
            transition_warnings=transition_warnings,
        ):
            if (
                transition_warnings is not None
                and len(transition_warnings) == warning_count
            ):
                transition_warnings.append(
                    _encrypted_cache_transition_warning(home_path)
                )
            path = _encrypted_disk_cache_path(home_path)
            legacy_artifact_remains = False
            try:
                current_payload = json.loads(path.read_text(encoding="utf-8"))
                legacy_artifact_remains = (
                    isinstance(current_payload, dict)
                    and current_payload.get("version")
                    == _ENCRYPTED_CACHE_LEGACY_VERSION
                )
            except FileNotFoundError:
                pass
            except (OSError, UnicodeError, json.JSONDecodeError):
                legacy_artifact_remains = path.exists()
            if legacy_artifact_remains:
                _retire_legacy_encrypted_cache(
                    cache_key=cache_key,
                    home_path=home_path,
                    transition_warnings=transition_warnings,
                )
            return None
    except Exception:  # noqa: BLE001 — every recognized v1 failure retires v1
        _retire_legacy_encrypted_cache(
            cache_key=cache_key,
            home_path=home_path,
            transition_warnings=transition_warnings,
        )
        return None

    if inherited_endpoint_unbound:
        warning = _encrypted_cache_legacy_endpoint_warning(home_path)
        if transition_warnings is not None and warning not in transition_warnings:
            transition_warnings.append(warning)
        return None
    entry_age = time.time() - entry.fetched_at
    in_window = (
        math.isfinite(max_age_seconds)
        and math.isfinite(entry_age)
        and entry_age >= 0
        and max_age_seconds > 0
        and entry_age <= max_age_seconds
    )
    if not in_window:
        return None
    return entry


def _read_encrypted_disk_cache(
    *,
    cache_key: _CacheKey,
    access_token: str,
    max_age_seconds: float,
    home_path: Optional[Path] = None,
    transition_warnings: Optional[List[str]] = None,
) -> Optional[_CachedFetch]:
    """Authenticate, migrate legacy data, and return only in-window entries."""
    path = _encrypted_disk_cache_path(home_path)
    if _encrypted_cache_was_invalidated(cache_key, home_path):
        warning = _encrypted_cache_invalidation_warning(home_path)
        if transition_warnings is not None and warning not in transition_warnings:
            transition_warnings.append(warning)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unrecognized files are ordinary misses
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    if version == _ENCRYPTED_CACHE_LEGACY_VERSION:
        return _read_legacy_encrypted_disk_cache(
            payload=payload,
            cache_key=cache_key,
            access_token=access_token,
            max_age_seconds=max_age_seconds,
            home_path=home_path,
            transition_warnings=transition_warnings,
        )
    if version != _ENCRYPTED_CACHE_VERSION:
        return None
    context = payload.get("context")
    legacy_unbound_context = {
        "project_id": cache_key[1],
        "server_url": _ENCRYPTED_CACHE_LEGACY_UNBOUND_SERVER,
    }
    if context == legacy_unbound_context:
        warning = _encrypted_cache_legacy_endpoint_warning(home_path)
        if transition_warnings is not None and warning not in transition_warnings:
            transition_warnings.append(warning)
        return None
    if context != _encrypted_cache_context(cache_key):
        return None
    try:
        salt = _b64d(str(payload.get("salt", "")))
        nonce = _b64d(str(payload.get("nonce", "")))
        ciphertext = _b64d(str(payload.get("ciphertext", "")))
        key = _derive_encrypted_cache_key(access_token, salt)
        raw = AESGCM(key).decrypt(nonce, ciphertext, _encrypted_cache_aad(cache_key))
        inner = json.loads(raw.decode("utf-8"))
        if not isinstance(inner, dict):
            return None
        secrets = inner.get("secrets")
        inner_fetched_at = inner.get("fetched_at")
        if not isinstance(secrets, dict) or not isinstance(inner_fetched_at, (int, float)):
            return None
        entry_age = time.time() - float(inner_fetched_at)
        in_window = (
            math.isfinite(max_age_seconds)
            and math.isfinite(entry_age)
            and entry_age >= 0
            and max_age_seconds > 0
            and entry_age <= max_age_seconds
        )
        typed = {
            k: v for k, v in secrets.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        entry = _CachedFetch(secrets=typed, fetched_at=float(inner_fetched_at))
        # Cleanup is an encrypted-mode invariant independent of freshness:
        # stale/future payloads are not served, but their plaintext predecessor
        # must not remain silently at rest.
        cleanup_result = _remove_plaintext_cache_predecessor(
            cache_key=cache_key,
            home_path=home_path,
            encrypted_fetched_at=entry.fetched_at,
        )
        if cleanup_result is not _PlaintextCleanupResult.COMPLETE:
            warning = (
                _encrypted_cache_newer_plaintext_warning(home_path)
                if cleanup_result is _PlaintextCleanupResult.NEWER_PLAINTEXT
                else _encrypted_cache_transition_warning(home_path)
            )
            if (
                transition_warnings is not None
                and warning not in transition_warnings
            ):
                transition_warnings.append(warning)
            return None
        if not in_window:
            return None
        return entry
    except Exception:  # noqa: BLE001 — v2 parse/decrypt/I/O errors are misses
        return None


def _with_encrypted_cache_transition_warning(
    error: str,
    transition_warnings: List[str],
) -> str:
    """Append the canonical transition warning to a fatal live-fetch error."""
    if transition_warnings:
        return "; ".join([error, *transition_warnings])
    return error


def fetch_bitwarden_secrets(
    *,
    access_token: str,
    project_id: str,
    binary: Optional[Path] = None,
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    server_url: str = "",
    home_path: Optional[Path] = None,
    encrypted_cache_enabled: bool = False,
    encrypted_cache_max_stale_seconds: float = 0,
    source_env: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Pull the secrets for ``project_id`` from Bitwarden Secrets Manager.

    Returns ``(secrets_dict, warnings_list)``.

    Set ``server_url`` to point at a non-default Bitwarden region or a
    self-hosted instance — e.g. ``https://vault.bitwarden.eu`` for EU
    Cloud accounts.  When empty, Hermes inherits ``BWS_SERVER_URL`` from the
    source environment when set; otherwise ``bws`` uses its built-in default
    (``https://vault.bitwarden.com``, US Cloud).  The effective endpoint is
    plumbed into the subprocess and encrypted-cache identity.

    ``cache_ttl_seconds`` controls the normal fresh cache.  When
    ``encrypted_cache_enabled`` is true, fresh cache entries are written as
    AES-GCM encrypted JSON instead of plaintext, and a last-good encrypted
    entry may be used after NETWORK/TIMEOUT failures for up to
    ``encrypted_cache_max_stale_seconds``.  This stale fallback is separate
    from the fresh-cache TTL so operators can set ``cache_ttl_seconds: 0``
    while still keeping an encrypted break-glass cache for offline startup.

    Raises :class:`RuntimeError` for fatal conditions (missing binary,
    auth failure, unparseable output).  Callers in the env_loader path
    catch this and emit a single warning; callers in the user-facing
    setup wizard let it propagate.
    """
    if not access_token:
        raise RuntimeError("Bitwarden access token is empty")
    if not project_id:
        raise RuntimeError("Bitwarden project_id is empty")

    frozen_source_env = (
        _snapshot_bws_source_environment()
        if source_env is None
        else dict(source_env)
    )
    server_url = _effective_server_url(server_url, frozen_source_env)
    cache_key = (_token_fingerprint(access_token), project_id, server_url or "")
    encrypted_cache_read_warnings: List[str] = []
    if use_cache and cache_ttl_seconds > 0:
        cached = _CACHE.get(cache_key)
        policy = _CACHE_POLICY.get(cache_key)
        if (
            cached
            and policy
            == (
                encrypted_cache_enabled,
                False,
                str(_disk_cache_path(home_path)),
            )
            and cached.is_fresh(cache_ttl_seconds)
        ):
            return cached.secrets, []
        # L2: disk cache. ~5ms on cache hit vs ~380ms for `bws secret list`.
        if encrypted_cache_enabled:
            disk_cached = _read_encrypted_disk_cache(
                cache_key=cache_key,
                access_token=access_token,
                max_age_seconds=cache_ttl_seconds,
                home_path=home_path,
                transition_warnings=encrypted_cache_read_warnings,
            )
        else:
            disk_cached = _DISK_CACHE.read(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            # Promote into in-process cache so subsequent fetches in the
            # same process skip the disk read too.
            _cache_put(
                cache_key,
                disk_cached,
                encrypted_mode=encrypted_cache_enabled,
                home_path=home_path,
            )
            return disk_cached.secrets, []

    bws = binary or find_bws(install_if_missing=True)
    if bws is None:
        encrypted_inspection_age = max(
            cache_ttl_seconds,
            encrypted_cache_max_stale_seconds,
        )
        if use_cache and encrypted_cache_enabled:
            # Missing-binary failures cannot enter the network-error stale
            # fallback below.  Always inspect recognized encrypted state to
            # retire v1 and surface blocked predecessor cleanup, even when no
            # cache window permits serving; never consume the discarded entry.
            _read_encrypted_disk_cache(
                cache_key=cache_key,
                access_token=access_token,
                max_age_seconds=encrypted_inspection_age,
                home_path=home_path,
                transition_warnings=encrypted_cache_read_warnings,
            )
        error = (
            "bws binary not available — auto-install failed and `bws` is "
            "not on PATH.  Install manually from "
            "https://github.com/bitwarden/sdk-sm/releases or re-run "
            "`hermes secrets bitwarden setup`."
        )
        raise _BwsFetchError(
            _with_encrypted_cache_transition_warning(
                error,
                encrypted_cache_read_warnings,
            ),
            ErrorKind.BINARY_MISSING,
        )

    try:
        secrets, warnings = _run_bws_list(
            bws,
            access_token,
            project_id,
            server_url,
            source_env=frozen_source_env,
        )
    except RuntimeError as exc:
        # Live fetch failed. Fall back to a stale disk cache ONLY for
        # transport-level failures (network down, DNS error, transient BWS
        # outage / timeout) — never for AUTH_FAILED or a malformed-output
        # INTERNAL error, where serving old secrets would mask a real
        # config/credential problem the caller needs to see.  Without this
        # fallback a fleet of bots sharing one BWS project all stop working
        # on a single network blip.
        #
        # Two fallback tiers share the transport-only gate:
        # * encrypted cache (opt-in) — AES-GCM payload keyed off the
        #   bootstrap token, with its own max_stale_seconds window.  When
        #   enabled it is the ONLY fallback consulted: the whole point is
        #   that the at-rest payload is never plaintext, so we don't
        #   quietly serve the plaintext file alongside it.
        # * plaintext disk cache (default) — the ordinary DiskCache file.
        #   `cache_ttl_seconds <= 0` means the caller opted out of caching
        #   entirely (DiskCache.read/write both short-circuit on it) —
        #   honor that on the fallback path too.  `ttl_seconds=inf` on the
        #   read bypasses freshness (we explicitly want a stale hit); the
        #   caller's real TTL gates whether we even attempt the read.
        kind = _classify_bws_error(str(exc))
        if use_cache and kind in (ErrorKind.NETWORK, ErrorKind.TIMEOUT):
            if encrypted_cache_enabled:
                stale = _read_encrypted_disk_cache(
                    cache_key=cache_key,
                    access_token=access_token,
                    max_age_seconds=encrypted_cache_max_stale_seconds,
                    home_path=home_path,
                    transition_warnings=encrypted_cache_read_warnings,
                )
                if stale is not None:
                    age = max(0.0, time.time() - stale.fetched_at)
                    _cache_put(
                        cache_key,
                        stale,
                        encrypted_mode=True,
                        home_path=home_path,
                    )
                    return stale.secrets, [
                        f"bws live fetch failed ({exc}); falling back to "
                        f"stale ENCRYPTED disk cache ({int(age)}s old)"
                    ]
            elif cache_ttl_seconds > 0:
                stale = _DISK_CACHE.read(cache_key, float("inf"), home_path)
                if stale is not None:
                    age = max(0.0, time.time() - stale.fetched_at)
                    _cache_put(
                        cache_key,
                        stale,
                        encrypted_mode=False,
                        home_path=home_path,
                    )
                    return stale.secrets, [
                        f"bws live fetch failed ({exc}); "
                        f"falling back to stale disk cache ({int(age)}s old)"
                    ]
        audit_age = max(
            cache_ttl_seconds,
            encrypted_cache_max_stale_seconds,
        )
        if (
            use_cache
            and encrypted_cache_enabled
            and kind not in (ErrorKind.NETWORK, ErrorKind.TIMEOUT)
            and not encrypted_cache_read_warnings
        ):
            # Auth/internal failures must not consume stale secrets, but they
            # still need a discard-only transition audit for TTL-zero setups.
            _read_encrypted_disk_cache(
                cache_key=cache_key,
                access_token=access_token,
                max_age_seconds=audit_age,
                home_path=home_path,
                transition_warnings=encrypted_cache_read_warnings,
            )
        if encrypted_cache_read_warnings:
            raise _BwsFetchError(
                _with_encrypted_cache_transition_warning(
                    str(exc),
                    encrypted_cache_read_warnings,
                ),
                kind,
            ) from exc
        raise
    entry = _CachedFetch(secrets=secrets, fetched_at=time.time())
    if use_cache:
        if encrypted_cache_enabled:
            # Encryption is the storage policy; max_stale_seconds only controls
            # whether an outage may consume the last-good entry.  Never fall
            # back to the plaintext cache just because stale fallback is off.
            warning_count = len(warnings)
            transition_state: List[bool] = []
            transition_complete = _write_encrypted_disk_cache(
                cache_key=cache_key,
                access_token=access_token,
                entry=entry,
                home_path=home_path,
                transition_warnings=warnings,
                transition_pending_out=transition_state,
            )
            if not transition_complete and len(warnings) == warning_count:
                warnings.append(
                    _encrypted_cache_transition_warning(home_path)
                    if transition_state and transition_state[0]
                    else _encrypted_cache_write_warning(home_path)
                )
            transition_pending = bool(transition_state and transition_state[0])
            if cache_ttl_seconds > 0:
                _cache_put(
                    cache_key,
                    entry,
                    encrypted_mode=True,
                    transition_pending=transition_pending,
                    home_path=home_path,
                )
            else:
                # A TTL-zero live fetch supersedes any older process-local
                # value even though it deliberately does not retain a fresh
                # L1 entry for the current call.
                _cache_drop(cache_key)
        else:
            invalidation_complete = _invalidate_encrypted_disk_cache(home_path)
            markers_cleared = (
                _clear_encrypted_cache_invalidated(cache_key, home_path)
                if invalidation_complete
                else False
            )
            if not invalidation_complete or not markers_cleared:
                _mark_encrypted_cache_invalidated(cache_key, home_path)
                warning = _encrypted_cache_invalidation_warning(home_path)
                if warning not in warnings:
                    warnings.append(warning)
            if cache_ttl_seconds > 0:
                _cache_put(cache_key, entry, encrypted_mode=False, home_path=home_path)
                _DISK_CACHE.write(cache_key, entry, cache_ttl_seconds, home_path)
            else:
                _cache_drop(cache_key)
    return secrets, warnings


def _summarize_bws_stderr(raw: str) -> str:
    """Reduce a bws (Rust color-eyre) error dump to its cause line(s).

    bws failures look like::

        Error:
           0: Received error message from server: [400 Bad Request] {"error":"invalid_client"}

        Location:
           crates/bws/src/main.rs:108
        ...

    Everything from ``Location:`` on is diagnostic noise for a Hermes
    user.  Keep the numbered cause lines (joined), drop the rest, and
    fall back to the stripped raw text when the shape is unrecognized.
    """
    text = raw.replace("\x1b", "").strip()
    if not text:
        return text
    causes: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Location:", "Backtrace omitted", "Run with ")):
            break
        if stripped in ("", "Error:"):
            continue
        # Cause lines are numbered "0: ...", "1: ..." — strip the index.
        stripped = re.sub(r"^\d+:\s*", "", stripped)
        if stripped:
            causes.append(stripped)
    return "; ".join(causes) if causes else text


def _run_bws_list(
    bws: Path,
    access_token: str,
    project_id: str,
    server_url: str = "",
    *,
    source_env: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    cmd = [str(bws), "secret", "list", project_id, "--output", "json"]
    # bws child intentionally receives the access token.  Under a profile-local
    # fetch it must not inherit sibling credentials from process-global env.
    env = (
        _snapshot_bws_source_environment()
        if source_env is None
        else dict(source_env)
    )
    env["BWS_ACCESS_TOKEN"] = access_token
    # Make sure we're not echoing telemetry / colour codes into json.
    env.setdefault("NO_COLOR", "1")
    # Region / self-hosted support.  bws defaults to https://vault.bitwarden.com
    # (US Cloud); EU Cloud users need https://vault.bitwarden.eu, and
    # self-hosted users need their own URL.  The caller already froze the
    # effective route, so an empty value must remove a later inherited value.
    if server_url:
        env["BWS_SERVER_URL"] = server_url
    else:
        env.pop("BWS_SERVER_URL", None)

    try:
        proc = subprocess.run(  # noqa: S603 — bws path is trusted
            cmd,
            env=env,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=_BWS_RUN_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"bws timed out after {_BWS_RUN_TIMEOUT}s fetching secrets"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke bws: {exc}") from exc

    if proc.returncode != 0:
        # bws writes auth/network errors to stderr as a Rust error-report
        # dump (color-eyre): an "Error:" header, indented cause lines, then
        # "Location:" / "Backtrace omitted" noise.  Strip ANSI and boil it
        # down to the meaningful cause line(s) before surfacing.
        err = _summarize_bws_stderr(proc.stderr or proc.stdout or "")
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
    home_path: Optional[Path] = None,
    encrypted_cache_enabled: bool = False,
    encrypted_cache_max_stale_seconds: float = 0,
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

    frozen_source_env = _snapshot_bws_source_environment()
    access_token = frozen_source_env.get(access_token_env, "").strip()
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

    server_url = _effective_server_url(server_url, frozen_source_env)
    binary = find_bws(install_if_missing=auto_install)
    result.binary_path = binary
    if binary is None:
        error = (
            "bws binary not available and auto-install is disabled.  "
            "Run `hermes secrets bitwarden setup` to install."
        )
        transition_warnings: List[str] = []
        encrypted_inspection_age = max(
            cache_ttl_seconds,
            encrypted_cache_max_stale_seconds,
        )
        if encrypted_cache_enabled:
            # This public entry point owns binary discovery, so preserve the
            # discard-only transition audit before returning its usual
            # non-raising FetchResult contract, including at zero windows.
            _read_encrypted_disk_cache(
                cache_key=(
                    _token_fingerprint(access_token),
                    project_id,
                    server_url or "",
                ),
                access_token=access_token,
                max_age_seconds=encrypted_inspection_age,
                home_path=home_path,
                transition_warnings=transition_warnings,
            )
        result.error = _with_encrypted_cache_transition_warning(
            error,
            transition_warnings,
        )
        return result

    try:
        secrets, warnings = fetch_bitwarden_secrets(
            access_token=access_token,
            project_id=project_id,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
            server_url=server_url,
            home_path=home_path,
            encrypted_cache_enabled=encrypted_cache_enabled,
            encrypted_cache_max_stale_seconds=encrypted_cache_max_stale_seconds,
            source_env=frozen_source_env,
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
# SecretSource adapter — the registry-facing wrapper around this module.
# ---------------------------------------------------------------------------


class BitwardenSource(SecretSource):
    """Bitwarden Secrets Manager as a registered secret source.

    Thin adapter over the module's fetch machinery.  ``fetch()`` only
    *fetches* — precedence, override semantics, conflict warnings, and
    the ``os.environ`` writes are the orchestrator's job
    (see ``agent.secret_sources.registry.apply_all``).

    Bitwarden is a **bulk** source: it injects every secret in the
    configured BSM project, so explicit per-var bindings from mapped
    sources (e.g. the 1Password ``env:`` map) outrank it.
    """

    name = "bitwarden"
    label = "Bitwarden Secrets Manager"
    shape = "bulk"
    scheme = "bws"

    def override_existing(self, cfg: dict) -> bool:
        # Default True (matches DEFAULT_CONFIG): the point of BSM is
        # centralized rotation — if .env had the final say, rotating a
        # key in Bitwarden wouldn't take effect until the stale .env
        # line was also deleted.
        return bool(isinstance(cfg, dict) and cfg.get("override_existing", True))

    def protected_env_vars(self, cfg: dict):
        token_env = "BWS_ACCESS_TOKEN"
        if isinstance(cfg, dict):
            token_env = str(cfg.get("access_token_env") or token_env)
        return frozenset({token_env})

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "access_token_env": {
                "description": "Env var holding the machine-account access token",
                "default": "BWS_ACCESS_TOKEN",
            },
            "project_id": {"description": "BSM project UUID", "default": ""},
            "cache_ttl_seconds": {
                "description": "Fresh disk+memory cache TTL; 0 disables fresh-cache reuse",
                "default": 300,
            },
            "encrypted_cache": {
                "description": "Encrypted last-good cache for network/timeout fallback",
                "default": {
                    "enabled": False,
                    "max_stale_seconds": 0,
                },
            },
            "override_existing": {
                "description": "BSM values overwrite .env/shell values",
                "default": True,
            },
            "auto_install": {
                "description": "Auto-download the pinned bws binary",
                "default": True,
            },
            "server_url": {
                "description": "Region / self-hosted endpoint (empty = US Cloud)",
                "default": "",
            },
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()
        frozen_source_env = _snapshot_bws_source_environment()

        access_token_env = str(cfg.get("access_token_env") or "BWS_ACCESS_TOKEN")
        access_token = frozen_source_env.get(access_token_env, "").strip()
        if not access_token:
            result.error = (
                f"secrets.bitwarden.enabled is true but {access_token_env} is "
                "not set.  Run `hermes secrets bitwarden setup`."
            )
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        project_id = str(cfg.get("project_id") or "")
        if not project_id:
            result.error = (
                "secrets.bitwarden.project_id is empty.  "
                "Run `hermes secrets bitwarden setup`."
            )
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        try:
            ttl = float(cfg.get("cache_ttl_seconds", 300))
        except (TypeError, ValueError):
            ttl = 300.0

        encrypted_cfg = cfg.get("encrypted_cache")
        encrypted_cfg = encrypted_cfg if isinstance(encrypted_cfg, dict) else {}
        encrypted_enabled = bool(encrypted_cfg.get("enabled", False))
        try:
            encrypted_max_stale = float(encrypted_cfg.get("max_stale_seconds", 0))
        except (TypeError, ValueError):
            encrypted_max_stale = 0.0
        server_url = _effective_server_url(
            str(cfg.get("server_url", "") or ""),
            frozen_source_env,
        )

        auto_install = bool(cfg.get("auto_install", True))
        binary = find_bws(install_if_missing=auto_install)
        result.binary_path = binary
        if binary is None:
            error = (
                "bws binary not available and auto-install is disabled.  "
                "Run `hermes secrets bitwarden setup` to install."
            )
            transition_warnings: List[str] = []
            encrypted_inspection_age = max(ttl, encrypted_max_stale)
            if encrypted_enabled:
                # Authenticate and gate the encrypted entry without serving it.
                # The adapter owns binary discovery, so its early return must
                # still retire v1 and expose blocked predecessor cleanup even
                # when neither configured cache window permits serving.
                _read_encrypted_disk_cache(
                    cache_key=(
                        _token_fingerprint(access_token),
                        project_id,
                        server_url,
                    ),
                    access_token=access_token,
                    max_age_seconds=encrypted_inspection_age,
                    home_path=home_path,
                    transition_warnings=transition_warnings,
                )
            result.error = _with_encrypted_cache_transition_warning(
                error,
                transition_warnings,
            )
            result.error_kind = ErrorKind.BINARY_MISSING
            return result

        try:
            secrets, warnings = fetch_bitwarden_secrets(
                access_token=access_token,
                project_id=project_id,
                binary=binary,
                cache_ttl_seconds=ttl,
                server_url=server_url,
                home_path=home_path,
                encrypted_cache_enabled=encrypted_enabled,
                encrypted_cache_max_stale_seconds=encrypted_max_stale,
                source_env=frozen_source_env,
            )
        except RuntimeError as exc:
            result.error = str(exc)
            if isinstance(exc, _BwsFetchError):
                result.error_kind = exc.error_kind
            else:
                result.error_kind = _classify_bws_error(str(exc))
            if result.error_kind == ErrorKind.AUTH_FAILED:
                # Translate the raw OAuth reject into what it actually means
                # for the user before the mechanics.
                result.error = (
                    "Bitwarden rejected the machine-account access token "
                    f"({access_token_env}) — it was likely revoked, expired, "
                    f"or belongs to another region.  ({result.error})"
                )
            return result

        result.secrets = secrets
        result.warnings.extend(warnings)
        return result

    def remediation(self, kind, cfg: dict) -> str:
        if kind in (ErrorKind.AUTH_FAILED, ErrorKind.AUTH_EXPIRED):
            return (
                "Run `hermes secrets bitwarden token` to paste a fresh access "
                "token (create one in the Bitwarden web app: Secrets Manager → "
                "Machine accounts → Access tokens).  Wrong region?  Re-run "
                "`hermes secrets bitwarden setup` and pick EU/self-hosted."
            )
        return super().remediation(kind, cfg)


def _classify_bws_error(message: str) -> ErrorKind:
    """Best-effort mapping of bws failure text onto the shared taxonomy."""
    lowered = message.lower()
    if "timed out" in lowered:
        return ErrorKind.TIMEOUT
    if "binary not available" in lowered or "failed to invoke" in lowered:
        return ErrorKind.BINARY_MISSING
    if any(tok in lowered for tok in ("unauthorized", "invalid token",
                                      "access token", "401", "403",
                                      # The BSM identity endpoint rejects a
                                      # revoked/expired/deleted machine-account
                                      # token with an OAuth-style
                                      # `[400 Bad Request] {"error":"invalid_client"}`.
                                      "invalid_client", "invalid_grant",
                                      "400 bad request")):
        return ErrorKind.AUTH_FAILED
    if any(tok in lowered for tok in ("network", "connection", "resolve",
                                      "download", "dns")):
        return ErrorKind.NETWORK
    return ErrorKind.INTERNAL


# ---------------------------------------------------------------------------
# Test hook — used by hermetic tests to flush the cache between cases.
# ---------------------------------------------------------------------------


def clear_caches(home_path: Optional[Path] = None) -> None:
    """Drop in-process AND disk caches (plaintext and encrypted).

    Used after a token rotation (`hermes secrets bitwarden token`) so the
    next startup fetches fresh with the new credential instead of serving
    a pull cached under the old token's fingerprint.  The encrypted cache
    is keyed off the old token too, so it must go as well.
    """
    encrypted_path = _encrypted_disk_cache_path(home_path)
    identity = str(encrypted_path)
    _CACHE.clear()
    _CACHE_POLICY.clear()
    _DISK_CACHE.clear(home_path)
    try:
        encrypted_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Clearing must not erase the durable veto while the old ciphertext
        # remains.  Preserve a home-scoped marker for the next process.
        _ENCRYPTED_CACHE_INVALIDATIONS.add(identity)
        _persist_encrypted_cache_invalidation_marker(home_path)
        return
    if _remove_encrypted_cache_invalidation_markers(home_path):
        _ENCRYPTED_CACHE_INVALIDATIONS.discard(identity)
    else:
        _ENCRYPTED_CACHE_INVALIDATIONS.add(identity)


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the cache
    writer itself.
    """
    clear_caches(home_path)
