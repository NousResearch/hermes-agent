"""Safe profile cloning between authenticated Hermes API gateways.

A bot clone carries the bot's runnable definition (SOUL, config, skills,
plugins, cron, and desktop appearance), never credentials, memories, sessions,
or runtime state.  A persistent UUID travels with every clone so a target can
reject a second copy even when the first copy was renamed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote, urlsplit

import httpx

from hermes_cli.archive_safe import archive_root_dirs, make_targz, safe_extract_targz

MAX_BOT_CLONE_BYTES = 10_000_000
MAX_BOT_CLONE_MEMBERS = 10_000
BOT_ID_FILENAME = ".hermes-bot-id"

# Deliberately excludes USER/MEMORY, sessions, memories, databases, logs,
# credentials, and caches.  This is a shareable agent definition, not a backup.
BOT_CLONE_ROOTS = frozenset(
    {
        BOT_ID_FILENAME,
        ".cursorrules",
        ".no-bundled-skills",
        "AGENTS.md",
        "CLAUDE.md",
        "SOUL.md",
        "config.yaml",
        "cron",
        "desktop.json",
        "mcp.json",
        "plugins",
        "profile.yaml",
        "scripts",
        "skills",
        "system_prompt.md",
    }
)


class BotTransferError(RuntimeError):
    """A remote clone request failed without changing the target profile."""


def _valid_bot_id(value: str) -> str:
    try:
        return str(uuid.UUID(str(value).strip()))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Bot identity is missing or invalid.") from exc


def get_profile_bot_id(profile_dir: Path) -> Optional[str]:
    """Return a profile's stable clone identity, or ``None`` when unassigned."""
    path = profile_dir / BOT_ID_FILENAME
    if not path.is_file():
        return None
    return _valid_bot_id(path.read_text(encoding="utf-8"))


def ensure_profile_bot_id(profile_dir: Path) -> str:
    """Atomically assign and return the profile's stable clone identity."""
    existing = get_profile_bot_id(profile_dir)
    if existing:
        return existing

    candidate = str(uuid.uuid4())
    path = profile_dir / BOT_ID_FILENAME
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # A concurrent exporter won the identity race.
        return _valid_bot_id(path.read_text(encoding="utf-8"))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(candidate + "\n")
    return candidate


def set_profile_cloneable(name: str, allowed: bool) -> str:
    """Set the per-bot pull policy and return its stable bot UUID."""
    from hermes_cli.profiles import get_profile_dir, write_profile_meta

    profile_dir = get_profile_dir(name)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist.")
    bot_id = ensure_profile_bot_id(profile_dir)
    write_profile_meta(profile_dir, cloneable=bool(allowed))
    return bot_id


def profile_is_cloneable(name: str) -> bool:
    from hermes_cli.profiles import get_profile_dir, read_profile_meta

    profile_dir = get_profile_dir(name)
    return profile_dir.is_dir() and bool(read_profile_meta(profile_dir).get("cloneable"))


def _reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Bot clone cannot include symbolic link: {path.relative_to(root)}")


def _reset_owner_policies(staged: Path) -> None:
    """Do not transfer the source owner's inbound sharing decisions."""
    from hermes_cli.profiles import write_profile_meta

    write_profile_meta(staged, cloneable=False)
    config_path = staged / "config.yaml"
    if not config_path.is_file():
        return
    try:
        from hermes_cli.config import read_user_config_raw

        config = read_user_config_raw(config_path)
        gateway = config.get("gateway") if isinstance(config, dict) else None
        if not isinstance(gateway, dict) or "bot_sharing" not in gateway:
            return
        gateway.pop("bot_sharing", None)
        from utils import atomic_yaml_write

        atomic_yaml_write(config_path, config, sort_keys=False)
    except Exception as exc:
        raise ValueError("Could not remove owner-only sharing policy from bot clone.") from exc


def export_bot_profile(name: str, output_path: str) -> tuple[Path, str]:
    """Create a bounded, credential-free bot clone archive."""
    from hermes_cli.profiles import (
        _scrub_export_secrets,
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    profile_dir = get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile '{canon}' does not exist.")
    bot_id = ensure_profile_bot_id(profile_dir)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = str(output).removesuffix(".tar.gz").removesuffix(".tgz")
    with tempfile.TemporaryDirectory(prefix="hermes_bot_export_") as tmpdir:
        staged = Path(tmpdir) / canon
        staged.mkdir()
        for entry in sorted(BOT_CLONE_ROOTS):
            source = profile_dir / entry
            if not source.exists():
                continue
            if source.is_symlink():
                raise ValueError(f"Bot clone cannot include symbolic link: {entry}")
            target = staged / entry
            if source.is_dir():
                _reject_symlinks(source)
                shutil.copytree(source, target)
            elif source.is_file():
                shutil.copy2(source, target)
        _reset_owner_policies(staged)
        _scrub_export_secrets(staged)
        result = Path(make_targz(base, tmpdir, canon))

    if result.stat().st_size > MAX_BOT_CLONE_BYTES:
        result.unlink(missing_ok=True)
        raise ValueError(
            f"Bot clone exceeds the {MAX_BOT_CLONE_BYTES // 1_000_000} MB transfer limit."
        )
    return result, bot_id


def _validate_bot_archive(archive: Path) -> tuple[str, str]:
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if archive.stat().st_size > MAX_BOT_CLONE_BYTES:
        raise ValueError(
            f"Bot clone exceeds the {MAX_BOT_CLONE_BYTES // 1_000_000} MB transfer limit."
        )
    roots = archive_root_dirs(archive, max_members=MAX_BOT_CLONE_MEMBERS)
    if len(roots) != 1:
        raise ValueError("Bot clone must contain exactly one top-level directory.")
    archive_root = next(iter(roots))
    with tempfile.TemporaryDirectory(prefix="hermes_bot_validate_") as tmpdir:
        staging = Path(tmpdir)
        safe_extract_targz(
            archive,
            staging,
            max_bytes=MAX_BOT_CLONE_BYTES,
            max_members=MAX_BOT_CLONE_MEMBERS,
        )
        extracted = staging / archive_root
        unexpected = {path.name for path in extracted.iterdir()} - BOT_CLONE_ROOTS
        if unexpected:
            raise ValueError(
                "Bot clone contains disallowed profile data: " + ", ".join(sorted(unexpected))
            )
        bot_id = get_profile_bot_id(extracted)
        if not bot_id:
            raise ValueError("Bot clone has no stable identity.")
    return archive_root, bot_id


def _iter_profile_dirs() -> Iterator[Path]:
    from hermes_cli.profiles import _get_default_hermes_home, _get_profiles_root

    default = _get_default_hermes_home()
    if default.is_dir():
        yield default
    root = _get_profiles_root()
    if root.is_dir():
        for path in root.iterdir():
            if path.is_dir():
                yield path


@contextmanager
def _clone_import_lock(timeout: float = 30.0) -> Iterator[None]:
    """Cross-process exclusion for identity/name checks plus profile install."""
    from hermes_cli.profiles import _get_profiles_root

    root = _get_profiles_root()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".bot-clone.lock"
    handle = open(lock, "a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while not acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, OSError):
                pass
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for another bot clone import.")
            time.sleep(0.05)
        yield
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def import_bot_profile(archive_path: str, name: Optional[str] = None) -> tuple[Path, str]:
    """Install a bot clone without overwriting a name or duplicate bot UUID."""
    from hermes_cli.profiles import import_profile

    archive = Path(archive_path)
    archive_root, bot_id = _validate_bot_archive(archive)
    with _clone_import_lock():
        for profile_dir in _iter_profile_dirs():
            try:
                existing_id = get_profile_bot_id(profile_dir)
            except (OSError, ValueError):
                continue
            if existing_id == bot_id:
                raise FileExistsError(
                    f"Bot {bot_id} already exists as profile '{profile_dir.name}'."
                )
        profile_dir = import_profile(
            str(archive),
            name=name or archive_root,
            max_extract_bytes=MAX_BOT_CLONE_BYTES,
            max_archive_members=MAX_BOT_CLONE_MEMBERS,
        )
    return profile_dir, bot_id


def _remote_settings(remote: Optional[str]) -> tuple[str, str]:
    from agent.secret_scope import UnscopedSecretError, get_secret
    from hermes_cli.config import cfg_get, load_config

    configured = cfg_get(load_config(), "gateway", "proxy_url", default="")
    base = str(remote or configured or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if not base:
        raise ValueError("Remote gateway URL is required (--from/--to or gateway.proxy_url).")
    if not base.startswith(("http://", "https://")):
        raise ValueError("Remote gateway URL must start with http:// or https://.")
    parsed = urlsplit(base)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Remote gateway URL cannot contain credentials, a query, or a fragment.")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Remote gateway URL must use HTTPS (HTTP is allowed only on loopback).")
    try:
        key = str(get_secret("GATEWAY_PROXY_KEY", "") or "").strip()
    except UnscopedSecretError:
        key = os.getenv("GATEWAY_PROXY_KEY", "").strip()
    if not key:
        raise ValueError("GATEWAY_PROXY_KEY is required for bot cloning.")
    return base, key


def _response_error(response: httpx.Response) -> BotTransferError:
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("code") or "")
        else:
            detail = str(error or payload.get("detail") or "") if isinstance(payload, dict) else ""
    except Exception:
        detail = ""
    return BotTransferError(detail or f"Remote gateway returned HTTP {response.status_code}.")


def pull_bot_profile(
    remote_profile: str,
    *,
    remote: Optional[str] = None,
    name: Optional[str] = None,
) -> tuple[Path, str]:
    """Pull one clone-enabled remote bot into a new local profile."""
    base, key = _remote_settings(remote)
    url = f"{base}/v1/bots/{quote(remote_profile, safe='')}/clone"
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/gzip"}
    with tempfile.TemporaryDirectory(prefix="hermes_bot_pull_") as tmpdir:
        archive = Path(tmpdir) / "bot.tar.gz"
        with httpx.Client(timeout=120.0, follow_redirects=False) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    response.read()
                    raise _response_error(response)
                total = 0
                with archive.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > MAX_BOT_CLONE_BYTES:
                            raise BotTransferError("Remote bot clone exceeds the 10 MB transfer limit.")
                        handle.write(chunk)
        return import_bot_profile(str(archive), name=name)


def push_bot_profile(
    local_profile: str,
    *,
    remote: Optional[str] = None,
    name: Optional[str] = None,
) -> tuple[str, str]:
    """Push one local bot into a new profile on an opt-in remote gateway."""
    base, key = _remote_settings(remote)
    with tempfile.TemporaryDirectory(prefix="hermes_bot_push_") as tmpdir:
        archive, _ = export_bot_profile(local_profile, str(Path(tmpdir) / "bot.tar.gz"))
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/gzip",
            "Accept": "application/json",
        }
        params = {"name": name} if name else None
        with archive.open("rb") as body, httpx.Client(timeout=120.0, follow_redirects=False) as client:
            response = client.post(f"{base}/v1/bots/clone", headers=headers, params=params, content=body)
        if response.status_code != 201:
            raise _response_error(response)
        payload = response.json()
        return str(payload["name"]), _valid_bot_id(payload["bot_id"])
