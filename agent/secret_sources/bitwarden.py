"""Bitwarden Secrets Manager (`bws` CLI) integration.

Pulls API keys from BSM at startup so they need not live in ``~/.hermes/.env``.
``bws`` is auto-installed into ``<hermes_home>/bin/bws`` (one pinned version,
SHA-256-verified against the published checksum). The one bootstrap secret is
the access token in ``.env``; every other key can live in BSM. One
``bws secret list <project_id>`` call per fetch, cached in-process and on disk
for ``cache_ttl_seconds``. Failures NEVER block startup. Subprocess-driven on
purpose: one cross-platform binary beats the ``bitwarden-sdk-secrets`` Rust wheel.
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
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.secret_sources._cache import (
    CachedFetch as _CachedFetch, SecretCache, atomic_write_json, entry_from_payload,
    fingerprint as _token_fingerprint, resolve_cache_home,
)
from agent.secret_sources.base import (
    ErrorKind, FetchResult, SecretSource, classify_cli_error, coerce_float,
    is_valid_env_name as _is_valid_env_name, get_source_environment, run_cli, source_child_env,
)

logger = logging.getLogger(__name__)

# Pinned upstream version — never auto-resolve "latest": release shape (asset
# names, CLI flags) may change between majors and updates must be deliberate.
_BWS_VERSION = "2.0.0"
_BWS_RELEASE_BASE = f"https://github.com/bitwarden/sdk-sm/releases/download/bws-v{_BWS_VERSION}"
_BWS_CHECKSUM_NAME = f"bws-sha256-checksums-{_BWS_VERSION}.txt"
_BWS_DOWNLOAD_TIMEOUT = 60
_BWS_RUN_TIMEOUT = 30

# <hermes_home>/cache/bws_cache.json holds only secret VALUES (never the access
# token); kept out of .env so users editing .env don't commit BSM-sourced secrets.
_CacheKey = Tuple[str, str, str]  # (access_token_fingerprint, project_id, server_url)
_DISK_CACHE_BASENAME = "bws_cache.json"
_ENCRYPTED_CACHE_BASENAME = "bws_cache.enc.json"
_ENCRYPTED_CACHE_VERSION = 1
_ENCRYPTED_CACHE_INFO = b"hermes-bws-encrypted-cache-v1"


def _cache_key_str(cache_key: _CacheKey) -> str:
    return "|".join(cache_key)


_STORE: SecretCache[_CacheKey] = SecretCache(_DISK_CACHE_BASENAME, key_serializer=_cache_key_str)
# Test seams: L1 dict, L2 DiskCache, and its path.
_CACHE = _STORE.memory
_DISK_CACHE = _STORE.disk
_disk_cache_path = _DISK_CACHE.path


def _encrypted_disk_cache_path(home_path: Optional[Path] = None) -> Path:
    return resolve_cache_home(home_path) / "cache" / _ENCRYPTED_CACHE_BASENAME


# First matching rule wins. The BSM identity endpoint rejects a revoked /
# expired machine-account token with an OAuth-style
# `[400 Bad Request] {"error":"invalid_client"}`, hence those AUTH tokens.
_BWS_ERROR_RULES = (
    (ErrorKind.TIMEOUT, ("timed out",)),
    (ErrorKind.BINARY_MISSING, ("binary not available", "failed to invoke")),
    (ErrorKind.AUTH_FAILED, ("unauthorized", "invalid token", "access token", "401", "403",
                             "invalid_client", "invalid_grant", "400 bad request")),
    (ErrorKind.NETWORK, ("network", "connection", "resolve", "download", "dns")),
)


def _classify_bws_error(message: str) -> ErrorKind:
    return classify_cli_error(message, _BWS_ERROR_RULES)


# --- Binary discovery + lazy install ----------------------------------------


def _hermes_bin_dir() -> Path:
    """Where Hermes stores its managed binaries. Profile-aware."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "bin"


def find_bws(*, install_if_missing: bool = False) -> Optional[Path]:
    """Managed ``<hermes_home>/bin/bws`` first, then PATH, then optional auto-install."""
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


def _platform_binary_name() -> str:
    return "bws.exe" if platform.system() == "Windows" else "bws"


def _platform_asset_name() -> str:
    """Map (uname, arch, libc) → upstream asset filename (Rust target-triple style)."""
    system = platform.system()
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"

    if system == "Darwin":  # universal binary covers Intel + Apple Silicon
        return f"bws-macos-universal-{_BWS_VERSION}.zip"
    if system == "Windows":
        return f"bws-{arch}-pc-windows-msvc-{_BWS_VERSION}.zip"
    if system == "Linux":
        # glibc default; musl only if ldd says so (a wrong guess surfaces as a loader error).
        libc = "gnu"
        try:
            res = subprocess.run(["ldd", "--version"], capture_output=True, text=True, encoding='utf-8',
                                 errors='replace', timeout=2, stdin=subprocess.DEVNULL)
            if "musl" in (res.stdout + res.stderr).lower():
                libc = "musl"
        except (OSError, subprocess.TimeoutExpired):
            pass
        return f"bws-{arch}-unknown-linux-{libc}-{_BWS_VERSION}.zip"

    raise RuntimeError(f"Unsupported platform for bws auto-install: {system} {machine}")


def install_bws(*, force: bool = False) -> Path:
    """Download, verify, and install the pinned ``bws`` binary; raises on any failure
    (the auto-install path catches; the setup wizard shows the error)."""
    bin_dir = _hermes_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / _platform_binary_name()
    if target.exists() and not force:
        return target

    asset_name = _platform_asset_name()
    with tempfile.TemporaryDirectory(prefix="hermes-bws-") as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / asset_name
        checksum_path = tmp / _BWS_CHECKSUM_NAME

        logger.info("Downloading %s", f"{_BWS_RELEASE_BASE}/{asset_name}")
        _http_download(f"{_BWS_RELEASE_BASE}/{asset_name}", zip_path)
        _http_download(f"{_BWS_RELEASE_BASE}/{_BWS_CHECKSUM_NAME}", checksum_path)

        expected = _expected_sha256(checksum_path, asset_name)
        actual = _sha256_file(zip_path)
        if expected.lower() != actual.lower():
            raise RuntimeError(f"Checksum mismatch for {asset_name}: expected {expected}, got {actual}")

        with zipfile.ZipFile(zip_path) as zf:
            member = _pick_zip_member(zf, _platform_binary_name())
            extracted = _safe_extract_member(zf, member, tmp)

        # Stage in the final directory so the rename can't cross filesystems.
        fd, staged = tempfile.mkstemp(dir=str(bin_dir), prefix=".bws_")
        os.close(fd)
        shutil.copy2(extracted, staged)
        os.chmod(staged, 0o755)
        os.replace(staged, target)

    logger.info("Installed bws %s at %s", _BWS_VERSION, target)
    return target


def _http_download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent"})
    try:
        with urllib.request.urlopen(req, timeout=_BWS_DOWNLOAD_TIMEOUT) as resp, open(dest, "wb") as f:  # noqa: S310
            shutil.copyfileobj(resp, f)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _expected_sha256(checksum_file: Path, asset_name: str) -> str:
    """Parse standard ``sha256sum`` output (``<hex>  <filename>`` per line)."""
    for line in checksum_file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[-1] == asset_name:
            return parts[0]
    raise RuntimeError(f"No checksum entry for {asset_name} in {checksum_file.name}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_zip_member(zf: zipfile.ZipFile, binary_name: str) -> str:
    """Find the binary in the zip; tolerate a top-level dir, prefer the shortest path."""
    candidates = [n for n in zf.namelist() if n.split("/")[-1] == binary_name]
    if not candidates:
        raise RuntimeError(f"Could not find {binary_name} inside downloaded archive "
                           f"(members: {zf.namelist()[:5]}...)")
    return min(candidates, key=len)


def _safe_extract_member(zf: zipfile.ZipFile, member: str, dest_dir: Path) -> Path:
    """Extract one member, refusing zip-slip: ``ZipFile.extract`` never verifies the
    joined path stays inside ``dest_dir``, so containment is checked here first."""
    dest_root = os.path.realpath(dest_dir)
    target = os.path.realpath(os.path.join(dest_root, member))
    try:  # commonpath raises for e.g. different Windows drives — treat as escape
        contained = os.path.commonpath([dest_root, target]) == dest_root and target != dest_root
    except ValueError:
        contained = False
    if not contained:
        raise RuntimeError(f"Refusing to extract unsafe archive member {member!r}: "
                           f"it escapes the extraction directory")
    zf.extract(member, dest_root)
    return Path(target)


# --- Encrypted last-good cache (opt-in) -------------------------------------


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _derive_encrypted_cache_key(access_token: str, salt: bytes) -> bytes:
    """HKDF the local cache key from the bootstrap BWS token. cryptography is imported
    lazily: eagerly mapping ``_rust.pyd`` on Windows blocks the updater replacing it."""
    # Keep the native cryptography extension lazy. Most CLI commands import this module while building
    # argparse, even though only encrypted-cache reads/writes need it. See #73381.
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt,
                info=_ENCRYPTED_CACHE_INFO).derive(access_token.encode("utf-8"))


def _write_encrypted_disk_cache(*, cache_key: _CacheKey, access_token: str, entry: _CachedFetch,
                                home_path: Optional[Path] = None) -> None:
    """Persist an AES-GCM encrypted last-good entry atomically (best-effort). The raw
    token only derives the key; a successful write removes the legacy plaintext cache."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        salt = os.urandom(16)
        nonce = os.urandom(12)
        serialized_key = _cache_key_str(cache_key)
        key = _derive_encrypted_cache_key(access_token, salt)
        plaintext = json.dumps(
            {"secrets": entry.secrets, "fetched_at": entry.fetched_at}, separators=(",", ":"),
        ).encode("utf-8")
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, serialized_key.encode("utf-8"))
        payload = {"version": _ENCRYPTED_CACHE_VERSION, "key": serialized_key,
                   "salt": _b64e(salt), "nonce": _b64e(nonce), "ciphertext": _b64e(ciphertext)}
        atomic_write_json(_encrypted_disk_cache_path(home_path), payload, tmp_prefix=".bws_cache_enc_")
        _STORE.disk.clear(home_path)
    except Exception:  # noqa: BLE001 — best-effort cache only
        return


def _read_encrypted_disk_cache(*, cache_key: _CacheKey, access_token: str, max_age_seconds: float,
                               home_path: Optional[Path] = None) -> Optional[_CachedFetch]:
    """Decrypted encrypted-cache entry if it matches ``cache_key`` and is in-window."""
    if max_age_seconds <= 0:
        return None
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        payload = json.loads(_encrypted_disk_cache_path(home_path).read_text(encoding="utf-8"))
        serialized_key = _cache_key_str(cache_key)
        if (not isinstance(payload, dict)
                or payload.get("version") != _ENCRYPTED_CACHE_VERSION
                or payload.get("key") != serialized_key):
            return None
        salt, nonce, ciphertext = (base64.b64decode(str(payload.get(k, "")).encode("ascii"), validate=True)
                                   for k in ("salt", "nonce", "ciphertext"))
        key = _derive_encrypted_cache_key(access_token, salt)
        entry = entry_from_payload(json.loads(
            AESGCM(key).decrypt(nonce, ciphertext, serialized_key.encode("utf-8")).decode("utf-8")
        ))
        if entry is None:
            return None
        entry_age = time.time() - entry.fetched_at
        return None if entry_age < 0 or entry_age > max_age_seconds else entry
    except Exception:  # noqa: BLE001 — cache miss on parse/decrypt/I/O errors
        return None


# --- Secret fetch -----------------------------------------------------------


def fetch_bitwarden_secrets(
    *, access_token: str, project_id: str, binary: Optional[Path] = None,
    cache_ttl_seconds: float = 300, use_cache: bool = True, server_url: str = "",
    home_path: Optional[Path] = None, encrypted_cache_enabled: bool = False,
    encrypted_cache_max_stale_seconds: float = 0,
) -> Tuple[Dict[str, str], List[str]]:
    """Pull the secrets for ``project_id`` from BSM → ``(secrets, warnings)``.

    ``server_url``: region / self-hosted instance (empty = US Cloud). With
    ``encrypted_cache_enabled`` fresh entries are written AES-GCM encrypted and a
    last-good entry may be served after NETWORK/TIMEOUT failures for up to
    ``encrypted_cache_max_stale_seconds`` — independent of the fresh TTL, so
    ``cache_ttl_seconds: 0`` can coexist with a break-glass offline cache.
    Raises ``RuntimeError`` on fatal conditions (missing binary, auth failure,
    unparseable output); env_loader catches, the setup wizard lets it propagate.
    """
    if not access_token:
        raise RuntimeError("Bitwarden access token is empty")
    if not project_id:
        raise RuntimeError("Bitwarden project_id is empty")

    cache_key = (_token_fingerprint(access_token), project_id, server_url or "")

    def _read_encrypted(max_age: float) -> Optional[_CachedFetch]:
        return _read_encrypted_disk_cache(cache_key=cache_key, access_token=access_token,
                                          max_age_seconds=max_age, home_path=home_path)

    if use_cache and cache_ttl_seconds > 0:
        # L2 (~5ms) vs ~380ms for `bws secret list`.
        cached = _STORE.lookup(
            cache_key, cache_ttl_seconds, home_path,
            read_disk=(lambda: _read_encrypted(cache_ttl_seconds)) if encrypted_cache_enabled else None,
        )
        if cached is not None:
            return cached.secrets, []

    bws = binary or find_bws(install_if_missing=True)
    if bws is None:
        raise RuntimeError("bws binary not available — auto-install failed and `bws` is "
                           "not on PATH.  Install manually from "
                           "https://github.com/bitwarden/sdk-sm/releases or re-run "
                           "`hermes secrets bitwarden setup`.")

    try:
        secrets, warnings = _run_bws_list(bws, access_token, project_id, server_url)
    except RuntimeError as exc:
        # Stale fallback ONLY for transport failures — never AUTH_FAILED / INTERNAL,
        # where old secrets would mask a real problem (without it a fleet sharing
        # one project all stops on a network blip). With the encrypted cache on it
        # is the ONLY fallback (at-rest payload must never be plaintext); else the
        # plain DiskCache is read with ttl=inf, but only when the real TTL > 0.
        if use_cache and _classify_bws_error(str(exc)) in (ErrorKind.NETWORK, ErrorKind.TIMEOUT):
            stale = label = None
            if encrypted_cache_enabled:
                stale = _read_encrypted(encrypted_cache_max_stale_seconds)
                label = "stale ENCRYPTED disk cache"
            elif cache_ttl_seconds > 0:
                stale = _STORE.disk.read(cache_key, float("inf"), home_path)
                label = "stale disk cache"
            if stale is not None:
                age = max(0.0, time.time() - stale.fetched_at)
                _STORE.memory[cache_key] = stale
                return stale.secrets, [
                    f"bws live fetch failed ({exc}); falling back to {label} ({int(age)}s old)"
                ]
        raise
    entry = _CachedFetch(secrets=secrets, fetched_at=time.time())
    if use_cache:
        if cache_ttl_seconds > 0:
            _STORE.memory[cache_key] = entry
        if encrypted_cache_enabled:  # storage policy; max_stale only gates outage reads
            _write_encrypted_disk_cache(cache_key=cache_key, access_token=access_token,
                                        entry=entry, home_path=home_path)
        else:
            _STORE.disk.write(cache_key, entry, cache_ttl_seconds, home_path)
    return secrets, warnings


def _summarize_bws_stderr(raw: str) -> str:
    """Reduce a bws (color-eyre) error dump to its numbered cause lines joined with
    ``; `` (dropping ``Location:``/``Backtrace`` on); raw text if unrecognized."""
    text = raw.replace("\x1b", "").strip()
    causes: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Location:", "Backtrace omitted", "Run with ")):
            break
        if stripped not in ("", "Error:") and (cause := re.sub(r"^\d+:\s*", "", stripped)):
            causes.append(cause)
    return "; ".join(causes) if causes else text


def _run_bws_list(bws: Path, access_token: str, project_id: str, server_url: str = "") -> Tuple[Dict[str, str], List[str]]:
    cmd = [str(bws), "secret", "list", project_id, "--output", "json"]
    # The bws child intentionally receives the access token; a profile-local
    # fetch must not inherit sibling credentials (source_child_env).
    env = source_child_env()
    env["BWS_ACCESS_TOKEN"] = access_token
    env.setdefault("NO_COLOR", "1")
    if server_url:  # empty keeps whatever BWS_SERVER_URL the shell already had
        env["BWS_SERVER_URL"] = server_url

    proc = run_cli(cmd, env=env, timeout=_BWS_RUN_TIMEOUT, label="bws",
                   timeout_message=f"bws timed out after {_BWS_RUN_TIMEOUT}s fetching secrets")

    if proc.returncode != 0:
        err = _summarize_bws_stderr(proc.stderr or proc.stdout or "")
        raise RuntimeError(f"bws exited {proc.returncode}: {err[:200]}")

    raw = proc.stdout.strip()
    if not raw:
        return {}, ["bws returned no output (empty project?)"]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bws returned non-JSON output: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"bws returned unexpected shape: {type(payload).__name__}")

    secrets: Dict[str, str] = {}
    warnings: List[str] = []
    for item in payload:
        key, value = (item.get("key"), item.get("value")) if isinstance(item, dict) else (None, None)
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if _is_valid_env_name(key):
            secrets[key] = value
        else:
            warnings.append(f"Skipping secret {key!r}: not a valid env-var name")
    return secrets, warnings


class BitwardenSource(SecretSource):
    """Bitwarden Secrets Manager as a registered **bulk** source (injects every
    secret in the project, so explicit mapped bindings outrank it)."""

    name = "bitwarden"
    label = "Bitwarden Secrets Manager"
    shape = "bulk"
    scheme = "bws"
    token_env_key = "access_token_env"
    default_token_env = "BWS_ACCESS_TOKEN"
    # override_existing defaults True: the point of BSM is centralized rotation
    # — a stale .env line must not have the final say.
    override_existing_default = True
    _AUTH_HINT = (
        "Run `hermes secrets bitwarden token` to paste a fresh access "
        "token (create one in the Bitwarden web app: Secrets Manager → "
        "Machine accounts → Access tokens).  Wrong region?  Re-run "
        "`hermes secrets bitwarden setup` and pick EU/self-hosted."
    )
    remediation_hints = {ErrorKind.AUTH_FAILED: _AUTH_HINT, ErrorKind.AUTH_EXPIRED: _AUTH_HINT}

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "access_token_env": {"description": "Env var holding the machine-account access token",
                                 "default": "BWS_ACCESS_TOKEN"},
            "project_id": {"description": "BSM project UUID", "default": ""},
            "cache_ttl_seconds": {"description": "Fresh disk+memory cache TTL; 0 disables fresh-cache reuse",
                                  "default": 300},
            "encrypted_cache": {"description": "Encrypted last-good cache for network/timeout fallback",
                                "default": {"enabled": False, "max_stale_seconds": 0}},
            "override_existing": {"description": "BSM values overwrite .env/shell values", "default": True},
            "auto_install": {"description": "Auto-download the pinned bws binary", "default": True},
            "server_url": {"description": "Region / self-hosted endpoint (empty = US Cloud)", "default": ""},
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        access_token_env = self.token_env(cfg)
        access_token = get_source_environment().get(access_token_env, "").strip()
        if not access_token:
            return result.fail(f"secrets.bitwarden.enabled is true but {access_token_env} is "
                               "not set.  Run `hermes secrets bitwarden setup`.", ErrorKind.NOT_CONFIGURED)
        project_id = str(cfg.get("project_id") or "")
        if not project_id:
            return result.fail("secrets.bitwarden.project_id is empty.  Run `hermes secrets bitwarden setup`.",
                               ErrorKind.NOT_CONFIGURED)
        binary = find_bws(install_if_missing=bool(cfg.get("auto_install", True)))
        result.binary_path = binary
        if binary is None:
            return result.fail("bws binary not available and auto-install is disabled.  "
                               "Run `hermes secrets bitwarden setup` to install.", ErrorKind.BINARY_MISSING)

        encrypted_cfg = cfg.get("encrypted_cache")
        encrypted_cfg = encrypted_cfg if isinstance(encrypted_cfg, dict) else {}

        try:
            secrets, warnings = fetch_bitwarden_secrets(
                access_token=access_token, project_id=project_id, binary=binary,
                cache_ttl_seconds=coerce_float(cfg.get("cache_ttl_seconds", 300), 300.0),
                server_url=str(cfg.get("server_url", "") or "").strip(), home_path=home_path,
                encrypted_cache_enabled=bool(encrypted_cfg.get("enabled", False)),
                encrypted_cache_max_stale_seconds=coerce_float(encrypted_cfg.get("max_stale_seconds", 0), 0.0),
            )
        except RuntimeError as exc:
            result.fail(str(exc), _classify_bws_error(str(exc)))
            if result.error_kind == ErrorKind.AUTH_FAILED:  # say what the raw OAuth reject means first
                result.error = ("Bitwarden rejected the machine-account access token "
                                f"({access_token_env}) — it was likely revoked, expired, "
                                f"or belongs to another region.  ({result.error})")
            return result

        result.secrets = secrets
        result.warnings.extend(warnings)
        return result


def clear_caches(home_path: Optional[Path] = None) -> None:
    """Drop in-process AND disk caches (plaintext and encrypted), e.g. after a token rotation."""
    _STORE.clear(home_path)
    try:
        _encrypted_disk_cache_path(home_path).unlink()
    except (FileNotFoundError, OSError):
        pass


_reset_cache_for_tests = clear_caches


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import stat  # noqa: F401,E402

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


_PLUGIN_COMPAT_LAZY = {
    'DiskCache': ('agent.secret_sources._cache', 'DiskCache'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
