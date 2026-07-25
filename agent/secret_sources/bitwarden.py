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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from agent.secret_sources._cache import (
    CachedFetch as _CachedFetch,
    DiskCache,
    FetchResult,
    is_valid_env_name as _is_valid_env_name,
)
from agent.secret_sources.base import ErrorKind, SecretSource

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
_CACHE: Dict[_CacheKey, _CachedFetch] = {}

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
_ENCRYPTED_CACHE_VERSION = 1
_ENCRYPTED_CACHE_INFO = b"hermes-bws-encrypted-cache-v1"


def _cache_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key to a stable string for JSON storage."""
    token_fp, project_id, server_url = cache_key
    return f"{token_fp}|{project_id}|{server_url}"


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
    """Derive the local cache encryption key from the bootstrap BWS token."""
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=_ENCRYPTED_CACHE_INFO,
    ).derive(access_token.encode("utf-8"))


def _write_encrypted_disk_cache(
    *,
    cache_key: _CacheKey,
    access_token: str,
    entry: _CachedFetch,
    home_path: Optional[Path] = None,
) -> None:
    """Persist an encrypted last-good cache entry atomically.

    Best-effort by design: cache write failure must never block a fresh BWS
    fetch.  The raw BWS access token is not stored; it only derives the AES key.
    """
    path = _encrypted_disk_cache_path(home_path)
    try:
        cache_dir = path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(cache_dir, 0o700)
        except OSError:
            pass
        salt = os.urandom(16)
        nonce = os.urandom(12)
        serialized_key = _cache_key_str(cache_key)
        key = _derive_encrypted_cache_key(access_token, salt)
        plaintext = json.dumps(
            {"secrets": entry.secrets, "fetched_at": entry.fetched_at},
            separators=(",", ":"),
        ).encode("utf-8")
        ciphertext = AESGCM(key).encrypt(
            nonce, plaintext, serialized_key.encode("utf-8")
        )
        payload = {
            "version": _ENCRYPTED_CACHE_VERSION,
            "key": serialized_key,
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
            # A successful encrypted write completes migration; remove the
            # legacy plaintext cache so stale secrets cannot remain on disk.
            try:
                _disk_cache_path(home_path).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:  # noqa: BLE001 — best-effort cache only
        return


def _read_encrypted_disk_cache(
    *,
    cache_key: _CacheKey,
    access_token: str,
    max_age_seconds: float,
    home_path: Optional[Path] = None,
) -> Optional[_CachedFetch]:
    """Return a decrypted encrypted-cache entry if it matches and is in-window."""
    if max_age_seconds <= 0:
        return None
    path = _encrypted_disk_cache_path(home_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        serialized_key = _cache_key_str(cache_key)
        if payload.get("version") != _ENCRYPTED_CACHE_VERSION:
            return None
        if payload.get("key") != serialized_key:
            return None
        salt = _b64d(str(payload.get("salt", "")))
        nonce = _b64d(str(payload.get("nonce", "")))
        ciphertext = _b64d(str(payload.get("ciphertext", "")))
        key = _derive_encrypted_cache_key(access_token, salt)
        raw = AESGCM(key).decrypt(
            nonce, ciphertext, serialized_key.encode("utf-8")
        )
        inner = json.loads(raw.decode("utf-8"))
        if not isinstance(inner, dict):
            return None
        secrets = inner.get("secrets")
        inner_fetched_at = inner.get("fetched_at")
        if not isinstance(secrets, dict) or not isinstance(inner_fetched_at, (int, float)):
            return None
        entry_age = time.time() - float(inner_fetched_at)
        if entry_age < 0 or entry_age > max_age_seconds:
            return None
        typed = {
            k: v for k, v in secrets.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        return _CachedFetch(secrets=typed, fetched_at=float(inner_fetched_at))
    except Exception:  # noqa: BLE001 — cache miss on parse/decrypt/I/O errors
        return None


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
) -> Tuple[Dict[str, str], List[str]]:
    """Pull the secrets for ``project_id`` from Bitwarden Secrets Manager.

    Returns ``(secrets_dict, warnings_list)``.

    Set ``server_url`` to point at a non-default Bitwarden region or a
    self-hosted instance — e.g. ``https://vault.bitwarden.eu`` for EU
    Cloud accounts.  When empty, ``bws`` uses its built-in default
    (``https://vault.bitwarden.com``, US Cloud).  This is plumbed into
    the subprocess as ``BWS_SERVER_URL``.

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

    cache_key = (_token_fingerprint(access_token), project_id, server_url or "")
    if use_cache and cache_ttl_seconds > 0:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return cached.secrets, []
        # L2: disk cache. ~5ms on cache hit vs ~380ms for `bws secret list`.
        if encrypted_cache_enabled:
            disk_cached = _read_encrypted_disk_cache(
                cache_key=cache_key,
                access_token=access_token,
                max_age_seconds=cache_ttl_seconds,
                home_path=home_path,
            )
        else:
            disk_cached = _DISK_CACHE.read(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            # Promote into in-process cache so subsequent fetches in the
            # same process skip the disk read too.
            _CACHE[cache_key] = disk_cached
            return disk_cached.secrets, []

    bws = binary or find_bws(install_if_missing=True)
    if bws is None:
        raise RuntimeError(
            "bws binary not available — auto-install failed and `bws` is "
            "not on PATH.  Install manually from "
            "https://github.com/bitwarden/sdk-sm/releases or re-run "
            "`hermes secrets bitwarden setup`."
        )

    try:
        secrets, warnings = _run_bws_list(bws, access_token, project_id, server_url)
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
                )
                if stale is not None:
                    age = max(0.0, time.time() - stale.fetched_at)
                    _CACHE[cache_key] = stale
                    return stale.secrets, [
                        f"bws live fetch failed ({exc}); falling back to "
                        f"stale ENCRYPTED disk cache ({int(age)}s old)"
                    ]
            elif cache_ttl_seconds > 0:
                stale = _DISK_CACHE.read(cache_key, float("inf"), home_path)
                if stale is not None:
                    age = max(0.0, time.time() - stale.fetched_at)
                    _CACHE[cache_key] = stale
                    return stale.secrets, [
                        f"bws live fetch failed ({exc}); "
                        f"falling back to stale disk cache ({int(age)}s old)"
                    ]
        raise
    entry = _CachedFetch(secrets=secrets, fetched_at=time.time())
    if use_cache:
        if cache_ttl_seconds > 0:
            _CACHE[cache_key] = entry
        if encrypted_cache_enabled:
            # Encryption is the storage policy; max_stale_seconds only controls
            # whether an outage may consume the last-good entry.  Never fall
            # back to the plaintext cache just because stale fallback is off.
            _write_encrypted_disk_cache(
                cache_key=cache_key,
                access_token=access_token,
                entry=entry,
                home_path=home_path,
            )
        elif cache_ttl_seconds > 0:
            _DISK_CACHE.write(cache_key, entry, cache_ttl_seconds, home_path)
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
            home_path=home_path,
            encrypted_cache_enabled=encrypted_cache_enabled,
            encrypted_cache_max_stale_seconds=encrypted_cache_max_stale_seconds,
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

        access_token_env = str(cfg.get("access_token_env") or "BWS_ACCESS_TOKEN")
        access_token = os.environ.get(access_token_env, "").strip()
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

        auto_install = bool(cfg.get("auto_install", True))
        binary = find_bws(install_if_missing=auto_install)
        result.binary_path = binary
        if binary is None:
            result.error = (
                "bws binary not available and auto-install is disabled.  "
                "Run `hermes secrets bitwarden setup` to install."
            )
            result.error_kind = ErrorKind.BINARY_MISSING
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

        try:
            secrets, warnings = fetch_bitwarden_secrets(
                access_token=access_token,
                project_id=project_id,
                binary=binary,
                cache_ttl_seconds=ttl,
                server_url=str(cfg.get("server_url", "") or "").strip(),
                home_path=home_path,
                encrypted_cache_enabled=encrypted_enabled,
                encrypted_cache_max_stale_seconds=encrypted_max_stale,
            )
        except RuntimeError as exc:
            result.error = str(exc)
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
    _CACHE.clear()
    _DISK_CACHE.clear(home_path)
    try:
        _encrypted_disk_cache_path(home_path).unlink()
    except (FileNotFoundError, OSError):
        pass


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the cache
    writer itself.
    """
    clear_caches(home_path)
