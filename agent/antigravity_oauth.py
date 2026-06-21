"""Google OAuth PKCE flow for the Antigravity (google-antigravity) provider.

Tokens are stored separately from the existing ``google-gemini-cli`` provider so
development and production credentials do not accidentally bleed across:

    ~/.hermes/auth/antigravity_oauth.json

The on-disk schema matches ``agent.google_oauth`` so the runtime resolver can
share the same refresh/project-id packing convention.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import logging
import os
import re
import secrets
import shutil
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

ENV_CLIENT_ID = "HERMES_ANTIGRAVITY_CLIENT_ID"
ENV_CLIENT_SECRET = "HERMES_ANTIGRAVITY_CLIENT_SECRET"
ENV_CLI_PATH = "HERMES_ANTIGRAVITY_CLI_PATH"

# Public Antigravity CLI desktop OAuth client. Like Google's gemini-cli
# credentials (see agent/google_oauth.py), this is a DESKTOP OAuth client and
# its "secret" is not confidential — installed-app clients have no
# secret-keeping requirement (PKCE provides the security), and these creds are
# baked into every copy of the Antigravity CLI. Shipping them as a fallback
# lets users without `agy` installed authenticate directly. Split into parts
# with explicit comments per the convention in google_oauth.py.
_PUBLIC_CLIENT_ID_PROJECT_NUM = "1071006060591"
_PUBLIC_CLIENT_ID_HASH = "tmhssin2h21lcre235vtolojh4g403ep"
_PUBLIC_CLIENT_SECRET_SUFFIX = "K58FWR486LdLJ1mLB8sXC4z6qDAf"

_DEFAULT_CLIENT_ID = (
    f"{_PUBLIC_CLIENT_ID_PROJECT_NUM}-{_PUBLIC_CLIENT_ID_HASH}"
    ".apps.googleusercontent.com"
)
_DEFAULT_CLIENT_SECRET = f"GOCSPX-{_PUBLIC_CLIENT_SECRET_SUFFIX}"

# Fallback project ID when Code Assist project discovery fails entirely.
DEFAULT_PROJECT_ID = "rising-fact-p41fc"

_CLIENT_ID_PATTERN = re.compile(
    r"([0-9]{8,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com)"
)
_CLIENT_SECRET_PATTERN = re.compile(r"(GOCSPX-[A-Za-z0-9_-]{20,80})")
_DISCOVERY_MAX_FILE_BYTES = 25 * 1024 * 1024
_DISCOVERY_MAX_AGY_BINARY_BYTES = 220 * 1024 * 1024
_DISCOVERY_MAX_FILES = 600
_DISCOVERY_EXTENSIONS = {
    "",
    ".cjs",
    ".exe",
    ".js",
    ".json",
    ".mjs",
    ".node",
    ".ts",
}
_DISCOVERY_SKIP_DIRS = {
    ".system_generated",
    "brain",
    "conversations",
    "log",
    "logs",
    "scratch",
}

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v1/userinfo"

OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform "
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile "
    "https://www.googleapis.com/auth/cclog "
    "https://www.googleapis.com/auth/experimentsandconfigs"
)

DEFAULT_REDIRECT_PORT = 51121
REDIRECT_HOST = "localhost"
CALLBACK_PATH = "/oauth-callback"
REFRESH_SKEW_SECONDS = 60
TOKEN_REQUEST_TIMEOUT_SECONDS = 20.0
CALLBACK_WAIT_SECONDS = 300
LOCK_TIMEOUT_SECONDS = 30.0


class AntigravityOAuthError(RuntimeError):
    def __init__(self, message: str, *, code: str = "antigravity_oauth_error") -> None:
        super().__init__(message)
        self.code = code


def _credentials_path() -> Path:
    return get_hermes_home() / "auth" / "antigravity_oauth.json"


def _lock_path() -> Path:
    return _credentials_path().with_suffix(".json.lock")


_lock_state = threading.local()


@contextlib.contextmanager
def _credentials_lock(timeout_seconds: float = LOCK_TIMEOUT_SECONDS):
    depth = getattr(_lock_state, "depth", 0)
    if depth > 0:
        _lock_state.depth = depth + 1
        try:
            yield
        finally:
            _lock_state.depth -= 1
        return

    lock_file_path = _lock_path()
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_file_path), os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        try:
            import fcntl
        except ImportError:
            fcntl = None

        if fcntl is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out acquiring Antigravity OAuth credentials lock at {lock_file_path}."
                        )
                    time.sleep(0.05)
        else:
            try:
                import msvcrt  # type: ignore[import-not-found]

                deadline = time.monotonic() + max(0.0, float(timeout_seconds))
                while True:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        acquired = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out acquiring Antigravity OAuth credentials lock at {lock_file_path}."
                            )
                        time.sleep(0.05)
            except ImportError:
                acquired = True

        _lock_state.depth = 1
        yield
    finally:
        try:
            if acquired:
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                except ImportError:
                    try:
                        import msvcrt  # type: ignore[import-not-found]

                        try:
                            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                        except OSError:
                            pass
                    except ImportError:
                        pass
        finally:
            os.close(fd)
            _lock_state.depth = 0


_discovered_creds_cache: Dict[str, Any] = {}


def _secret_candidates(raw: str) -> list[str]:
    candidates: list[str] = []
    for length in (35, 34, 36, 33, 37, 38, 39, 40, 41, 42):
        if len(raw) >= length:
            candidates.append(raw[:length])
    candidates.append(raw)
    return list(dict.fromkeys(candidates))


def _candidate_discovery_roots() -> list[Path]:
    roots: list[Path] = []

    explicit = (os.getenv(ENV_CLI_PATH) or "").strip()
    if explicit:
        roots.append(Path(explicit))

    for command in ("agy", "agy.exe", "antigravity", "antigravity.exe"):
        found = shutil.which(command)
        if found:
            roots.append(Path(found))

    for env_key in ("LOCALAPPDATA", "APPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        base = os.getenv(env_key)
        if not base:
            continue
        base_path = Path(base)
        roots.extend((
            base_path / "agy",
            base_path / "agy" / "bin" / "agy.exe",
            base_path / "Programs" / "Antigravity",
            base_path / "Programs" / "Antigravity CLI",
            base_path / "Google" / "Antigravity",
            base_path / "Google" / "Antigravity CLI",
        ))

    home = Path.home()
    for root in (
        home / ".gemini" / "antigravity-cli",
        home / ".antigravitycli",
        home / ".antigravity",
    ):
        roots.append(root)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.expanduser().resolve())
        except OSError:
            key = str(root.expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _iter_discovery_files() -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if len(files) >= _DISCOVERY_MAX_FILES:
            return
        if path.suffix.lower() not in _DISCOVERY_EXTENSIONS:
            return
        try:
            stat_info = path.stat()
            max_bytes = (
                _DISCOVERY_MAX_AGY_BINARY_BYTES
                if path.name.lower() in {"agy", "agy.exe", "antigravity", "antigravity.exe"}
                else _DISCOVERY_MAX_FILE_BYTES
            )
            if not path.is_file() or stat_info.st_size > max_bytes:
                return
            key = str(path.resolve())
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        files.append(path)

    for root in _candidate_discovery_roots():
        if len(files) >= _DISCOVERY_MAX_FILES:
            break
        try:
            if root.is_file():
                add(root)
                continue
            if not root.is_dir():
                continue
        except OSError:
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in _DISCOVERY_SKIP_DIRS and not d.startswith(".git")
            ]
            for filename in filenames:
                add(Path(dirpath) / filename)
                if len(files) >= _DISCOVERY_MAX_FILES:
                    break
            if len(files) >= _DISCOVERY_MAX_FILES:
                break
    return files


def _extract_client_credential_candidates_from_text(content: str) -> list[Tuple[str, str]]:
    client_ids = list(dict.fromkeys(match.group(1) for match in _CLIENT_ID_PATTERN.finditer(content)))
    secrets: list[str] = []
    for match in _CLIENT_SECRET_PATTERN.finditer(content):
        secrets.extend(_secret_candidates(match.group(1)))
    secrets = list(dict.fromkeys(secrets))
    return [(client_id, secret) for client_id in client_ids for secret in secrets]


def _discover_client_credentials() -> Tuple[str, str]:
    if _discovered_creds_cache.get("resolved"):
        return (
            _discovered_creds_cache.get("client_id", ""),
            _discovered_creds_cache.get("client_secret", ""),
        )

    for path in _iter_discovery_files():
        try:
            content = path.read_bytes().decode("utf-8", errors="ignore")
        except OSError:
            continue
        candidates = _extract_client_credential_candidates_from_text(content)
        if candidates:
            client_id, client_secret = candidates[0]
            _discovered_creds_cache.update({
                "client_id": client_id,
                "client_secret": client_secret,
                "candidates": candidates,
                "resolved": "1",
            })
            logger.info("Discovered Antigravity OAuth client credentials from %s", path)
            return client_id, client_secret

    _discovered_creds_cache["resolved"] = "1"
    return "", ""


def _get_client_id() -> str:
    env_val = (os.getenv(ENV_CLIENT_ID) or "").strip()
    if env_val:
        return env_val
    discovered, _ = _discover_client_credentials()
    if discovered:
        return discovered
    return _DEFAULT_CLIENT_ID


def _get_client_secret() -> str:
    env_val = (os.getenv(ENV_CLIENT_SECRET) or "").strip()
    if env_val:
        return env_val
    _, discovered = _discover_client_credentials()
    if discovered:
        return discovered
    return _DEFAULT_CLIENT_SECRET


def _iter_client_credential_candidates() -> list[Tuple[str, str]]:
    env_id = (os.getenv(ENV_CLIENT_ID) or "").strip()
    env_secret = (os.getenv(ENV_CLIENT_SECRET) or "").strip()
    if env_id and env_secret:
        return [(env_id, env_secret)]

    _discover_client_credentials()
    cached = _discovered_creds_cache.get("candidates")
    candidates: list[Tuple[str, str]] = []
    if isinstance(cached, list):
        candidates = [
            (str(client_id), str(client_secret))
            for client_id, client_secret in cached
            if client_id and client_secret
        ]
    else:
        client_id = str(_discovered_creds_cache.get("client_id") or "")
        client_secret = str(_discovered_creds_cache.get("client_secret") or "")
        if client_id and client_secret:
            candidates = [(client_id, client_secret)]

    # Always include the public baked-in default as a last-resort candidate so
    # users without `agy` installed can still authenticate. De-dupe in case
    # discovery already surfaced the same client.
    default_pair = (_DEFAULT_CLIENT_ID, _DEFAULT_CLIENT_SECRET)
    if default_pair not in candidates:
        candidates.append(default_pair)
    return candidates


def _require_client_id() -> str:
    client_id = _get_client_id()
    if not client_id:
        raise AntigravityOAuthError(
            "Antigravity OAuth client ID is not available. Install Antigravity CLI "
            "so Hermes can discover its desktop OAuth client, set "
            f"{ENV_CLI_PATH} to the agy executable, or set {ENV_CLIENT_ID} and "
            f"{ENV_CLIENT_SECRET} in ~/.hermes/.env.",
            code="antigravity_oauth_client_id_missing",
        )
    return client_id


def _require_client_secret() -> str:
    client_secret = _get_client_secret()
    if not client_secret:
        raise AntigravityOAuthError(
            "Antigravity OAuth client secret is not available. Install Antigravity CLI "
            "so Hermes can discover its desktop OAuth client, set "
            f"{ENV_CLI_PATH} to the agy executable, or set {ENV_CLIENT_ID} and "
            f"{ENV_CLIENT_SECRET} in ~/.hermes/.env.",
            code="antigravity_oauth_client_secret_missing",
        )
    return client_secret


def _require_client_credentials() -> Tuple[str, str]:
    candidates = _iter_client_credential_candidates()
    if not candidates:
        _require_client_id()
        _require_client_secret()
    return candidates[0]


def _generate_pkce_pair() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


@dataclass
class RefreshParts:
    refresh_token: str
    project_id: str = ""
    managed_project_id: str = ""

    @classmethod
    def parse(cls, packed: str) -> "RefreshParts":
        if not packed:
            return cls(refresh_token="")
        parts = packed.split("|", 2)
        return cls(
            refresh_token=parts[0],
            project_id=parts[1] if len(parts) > 1 else "",
            managed_project_id=parts[2] if len(parts) > 2 else "",
        )

    def format(self) -> str:
        if not self.refresh_token:
            return ""
        if not self.project_id and not self.managed_project_id:
            return self.refresh_token
        return f"{self.refresh_token}|{self.project_id}|{self.managed_project_id}"


@dataclass
class AntigravityCredentials:
    access_token: str
    refresh_token: str
    expires_ms: int
    email: str = ""
    project_id: str = ""
    managed_project_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refresh": RefreshParts(
                refresh_token=self.refresh_token,
                project_id=self.project_id,
                managed_project_id=self.managed_project_id,
            ).format(),
            "access": self.access_token,
            "expires": int(self.expires_ms),
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AntigravityCredentials":
        parts = RefreshParts.parse(str(data.get("refresh", "") or ""))
        return cls(
            access_token=str(data.get("access", "") or ""),
            refresh_token=parts.refresh_token,
            expires_ms=int(data.get("expires", 0) or 0),
            email=str(data.get("email", "") or ""),
            project_id=parts.project_id,
            managed_project_id=parts.managed_project_id,
        )

    def access_token_expired(self, skew_seconds: int = REFRESH_SKEW_SECONDS) -> bool:
        if not self.access_token or not self.expires_ms:
            return True
        return (time.time() + max(0, skew_seconds)) * 1000 >= self.expires_ms


def load_credentials() -> Optional[AntigravityCredentials]:
    path = _credentials_path()
    if not path.exists():
        return None
    try:
        with _credentials_lock():
            raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError, IOError) as exc:
        logger.warning("Failed to read Antigravity OAuth credentials at %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    creds = AntigravityCredentials.from_dict(data)
    if not creds.access_token:
        return None
    return creds


def save_credentials(creds: AntigravityCredentials) -> Path:
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    payload = json.dumps(creds.to_dict(), indent=2, sort_keys=True) + "\n"
    with _credentials_lock():
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
        try:
            fd = os.open(
                str(tmp_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            atomic_replace(tmp_path, path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
    return path


def clear_credentials() -> None:
    path = _credentials_path()
    with _credentials_lock():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to remove Antigravity OAuth credentials at %s: %s", path, exc)


def _post_form(url: str, data: Dict[str, str], timeout: float) -> Dict[str, Any]:
    body = urllib.parse.urlencode(data).encode("ascii")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        code = "antigravity_oauth_token_http_error"
        if "invalid_grant" in detail.lower():
            code = "antigravity_oauth_invalid_grant"
        elif "invalid_client" in detail.lower():
            code = "antigravity_oauth_invalid_client"
        raise AntigravityOAuthError(
            f"Antigravity OAuth token endpoint returned HTTP {exc.code}: {detail or exc.reason}",
            code=code,
        ) from exc
    except urllib.error.URLError as exc:
        raise AntigravityOAuthError(
            f"Antigravity OAuth token request failed: {exc}",
            code="antigravity_oauth_token_network_error",
        ) from exc


def exchange_code(
    code: str,
    verifier: str,
    redirect_uri: str,
    *,
    timeout: float = TOKEN_REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    last_error: Optional[AntigravityOAuthError] = None
    candidates = _iter_client_credential_candidates()
    if not candidates:
        candidates = [_require_client_credentials()]
    for client_id, client_secret in candidates:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }
        try:
            return _post_form(TOKEN_ENDPOINT, data, timeout)
        except AntigravityOAuthError as exc:
            last_error = exc
            if exc.code != "antigravity_oauth_invalid_client":
                raise
    if last_error is not None:
        raise last_error
    raise AntigravityOAuthError(
        "Antigravity OAuth client credentials are unavailable.",
        code="antigravity_oauth_client_missing",
    )


def refresh_access_token(
    refresh_token: str,
    *,
    timeout: float = TOKEN_REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if not refresh_token:
        raise AntigravityOAuthError(
            "Cannot refresh: refresh_token is empty. Re-run OAuth login.",
            code="antigravity_oauth_refresh_token_missing",
        )
    last_error: Optional[AntigravityOAuthError] = None
    candidates = _iter_client_credential_candidates()
    if not candidates:
        candidates = [_require_client_credentials()]
    for client_id, client_secret in candidates:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        try:
            return _post_form(TOKEN_ENDPOINT, data, timeout)
        except AntigravityOAuthError as exc:
            last_error = exc
            if exc.code not in {
                "antigravity_oauth_invalid_client",
                "antigravity_oauth_invalid_grant",
            }:
                raise
    if last_error is not None:
        raise last_error
    raise AntigravityOAuthError(
        "Antigravity OAuth client credentials are unavailable.",
        code="antigravity_oauth_client_missing",
    )


def _fetch_user_email(access_token: str, timeout: float = TOKEN_REQUEST_TIMEOUT_SECONDS) -> str:
    try:
        request = urllib.request.Request(
            USERINFO_ENDPOINT + "?alt=json",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return str(data.get("email", "") or "")
    except Exception as exc:
        logger.debug("Antigravity userinfo fetch failed (non-fatal): %s", exc)
        return ""


_refresh_inflight: Dict[str, threading.Event] = {}
_refresh_inflight_lock = threading.Lock()


def get_valid_access_token(*, force_refresh: bool = False) -> str:
    creds = load_credentials()
    if creds is None:
        raise AntigravityOAuthError(
            "No Antigravity OAuth credentials found. Run `hermes login --provider google-antigravity` first.",
            code="antigravity_oauth_not_logged_in",
        )
    if not force_refresh and not creds.access_token_expired():
        return creds.access_token

    rt = creds.refresh_token
    with _refresh_inflight_lock:
        event = _refresh_inflight.get(rt)
        if event is None:
            event = threading.Event()
            _refresh_inflight[rt] = event
            owner = True
        else:
            owner = False

    if not owner:
        event.wait(timeout=LOCK_TIMEOUT_SECONDS)
        fresh = load_credentials()
        if fresh is not None and not fresh.access_token_expired():
            return fresh.access_token

    try:
        try:
            resp = refresh_access_token(rt)
        except AntigravityOAuthError as exc:
            if exc.code == "antigravity_oauth_invalid_grant":
                clear_credentials()
            raise
        new_access = str(resp.get("access_token", "") or "").strip()
        if not new_access:
            raise AntigravityOAuthError(
                "Refresh response did not include an access_token.",
                code="antigravity_oauth_refresh_empty",
            )
        creds.access_token = new_access
        creds.refresh_token = str(resp.get("refresh_token", "") or "").strip() or creds.refresh_token
        expires_in = int(resp.get("expires_in", 0) or 0)
        creds.expires_ms = int((time.time() + max(60, expires_in)) * 1000)
        save_credentials(creds)
        return creds.access_token
    finally:
        if owner:
            with _refresh_inflight_lock:
                _refresh_inflight.pop(rt, None)
            event.set()


def update_project_ids(project_id: str = "", managed_project_id: str = "") -> None:
    creds = load_credentials()
    if creds is None:
        return
    if project_id:
        creds.project_id = project_id
    if managed_project_id:
        creds.managed_project_id = managed_project_id
    save_credentials(creds)


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    expected_state: str = ""
    captured_code: Optional[str] = None
    captured_error: Optional[str] = None
    ready: Optional[threading.Event] = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, N802
        logger.debug("Antigravity OAuth callback: " + format, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed.query)
        state = (params.get("state") or [""])[0]
        error = (params.get("error") or [""])[0]
        code = (params.get("code") or [""])[0]

        handler_cls = type(self)
        if state != self.expected_state:
            handler_cls.captured_error = "OAuth state mismatch."
        elif error:
            handler_cls.captured_error = error
        elif not code:
            handler_cls.captured_error = "OAuth callback did not include a code."
        else:
            handler_cls.captured_code = code

        ok = not handler_cls.captured_error
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        msg = "Antigravity OAuth complete. You can return to Hermes." if ok else handler_cls.captured_error
        self.wfile.write(f"<html><body><p>{msg}</p></body></html>".encode("utf-8"))
        if handler_cls.ready is not None:
            handler_cls.ready.set()


class _ReusableHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True


def resolve_project_id_from_env() -> str:
    for key in ("HERMES_ANTIGRAVITY_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT_ID"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def start_oauth_flow(
    *,
    force_relogin: bool = False,
    open_browser: bool = True,
    port: int = DEFAULT_REDIRECT_PORT,
    project_id: str = "",
) -> AntigravityCredentials:
    if not force_relogin:
        existing = load_credentials()
        if existing and not existing.access_token_expired():
            return existing

    verifier, challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(24)
    client_id, _ = _require_client_credentials()

    ready = threading.Event()
    handler_cls = type("AntigravityOAuthCallbackHandler", (_OAuthCallbackHandler,), {})
    handler_cls.expected_state = state
    handler_cls.captured_code = None
    handler_cls.captured_error = None
    handler_cls.ready = ready

    try:
        server = _ReusableHTTPServer((REDIRECT_HOST, int(port)), handler_cls)
    except OSError:
        server = _ReusableHTTPServer((REDIRECT_HOST, 0), handler_cls)
    actual_port = int(server.server_address[1])
    redirect_uri = f"http://{REDIRECT_HOST}:{actual_port}{CALLBACK_PATH}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": OAUTH_SCOPES,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        auth_url = AUTH_ENDPOINT + "?" + urllib.parse.urlencode(params)
        print("Open this URL to authorize Antigravity OAuth:")
        print(auth_url)
        if open_browser:
            webbrowser.open(auth_url)
        if not ready.wait(timeout=CALLBACK_WAIT_SECONDS):
            raise AntigravityOAuthError(
                "Timed out waiting for Antigravity OAuth callback.",
                code="antigravity_oauth_callback_timeout",
            )
        if handler_cls.captured_error:
            raise AntigravityOAuthError(
                handler_cls.captured_error,
                code="antigravity_oauth_callback_error",
            )
        code = handler_cls.captured_code or ""
        token = exchange_code(code, verifier, redirect_uri)
    finally:
        server.shutdown()
        server.server_close()

    access_token = str(token.get("access_token", "") or "").strip()
    refresh_token = str(token.get("refresh_token", "") or "").strip()
    if not access_token or not refresh_token:
        raise AntigravityOAuthError(
            "Antigravity OAuth response did not include both access_token and refresh_token.",
            code="antigravity_oauth_missing_token",
        )
    expires_in = int(token.get("expires_in", 0) or 0)
    creds = AntigravityCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_ms=int((time.time() + max(60, expires_in)) * 1000),
        email=_fetch_user_email(access_token),
        project_id=project_id,
    )
    save_credentials(creds)
    return creds


def run_antigravity_oauth_login_pure() -> Dict[str, Any]:
    creds = start_oauth_flow(
        force_relogin=True,
        project_id=resolve_project_id_from_env(),
    )
    return {
        "access_token": creds.access_token,
        "refresh_token": creds.refresh_token,
        "expires_at_ms": creds.expires_ms,
        "email": creds.email,
        "project_id": creds.project_id,
    }
