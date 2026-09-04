"""Safe profile cloning between authenticated Hermes API gateways.

A bot clone carries the bot's runnable definition (SOUL, config, skills,
plugins, cron, and desktop appearance), never credentials, memories, sessions,
or runtime state.  A persistent UUID travels with every clone so a target can
reject a second copy even when the first copy was renamed.
"""

from __future__ import annotations

import json
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
BOT_CLONE_DENIED_NAMES = frozenset(
    {".env", ".netrc", "auth.json", "credentials.json", "secrets.json"}
)
BOT_CLONE_DENIED_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
BOT_CLONE_SAFE_BINARY_SUFFIXES = frozenset(
    {
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".otf",
        ".png",
        ".ttf",
        ".webp",
        ".woff",
        ".woff2",
    }
)

# Cron clones carry runnable definitions, not the source gateway's scheduler,
# delivery, filesystem, or execution state. Keep this allowlist aligned with
# create_job's caller-owned inputs rather than its persisted runtime record.
BOT_CLONE_CRON_FIELDS = frozenset(
    {
        "id",
        "name",
        "prompt",
        "skills",
        "skill",
        "model",
        "provider",
        "base_url",
        "script",
        "no_agent",
        "monitor_script",
        "monitor_url",
        "context_from",
        "schedule",
        "schedule_display",
        "enabled_toolsets",
        "reasoning_effort",
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
    fd, temporary = tempfile.mkstemp(
        dir=profile_dir, prefix=f".{BOT_ID_FILENAME}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(candidate + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
            return candidate
        except FileExistsError:
            return _valid_bot_id(path.read_text(encoding="utf-8"))
    finally:
        Path(temporary).unlink(missing_ok=True)


def set_profile_cloneable(name: str, allowed: bool) -> str:
    """Set the per-bot pull policy and return its stable bot UUID."""
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
        write_profile_meta,
    )

    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    profile_dir = get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist.")
    bot_id = ensure_profile_bot_id(profile_dir)
    write_profile_meta(profile_dir, cloneable=bool(allowed))
    return bot_id


def profile_is_cloneable(name: str) -> bool:
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        read_profile_meta,
        validate_profile_name,
    )

    canon = normalize_profile_name(name)
    try:
        validate_profile_name(canon)
    except (TypeError, ValueError):
        return False
    profile_dir = get_profile_dir(canon)
    return profile_dir.is_dir() and bool(read_profile_meta(profile_dir).get("cloneable"))


def _reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Bot clone cannot include symbolic link: {path.relative_to(root)}")


def _check_clone_source_budget(profile_dir: Path) -> None:
    """Reject a clone definition that cannot satisfy the import contract."""
    members = 1  # The archive's profile root directory.
    total_bytes = 0
    for entry in sorted(BOT_CLONE_ROOTS):
        source = profile_dir / entry
        if not source.exists():
            continue
        candidates = [source]
        if source.is_dir():
            candidates.extend(source.rglob("*"))
        for path in candidates:
            relative = path.relative_to(profile_dir)
            if (
                path.name.lower() in BOT_CLONE_DENIED_NAMES
                or path.suffix.lower() in BOT_CLONE_DENIED_SUFFIXES
                or "__pycache__" in relative.parts
                or path.name.endswith((".sock", ".tmp"))
                or (
                    relative.parts[0] == "cron"
                    and len(relative.parts) > 1
                    and relative.parts[1] != "jobs.json"
                )
            ):
                continue
            if path.is_symlink():
                raise ValueError(
                    f"Bot clone cannot include symbolic link: {path.relative_to(profile_dir)}"
                )
            if not (path.is_dir() or path.is_file()):
                raise ValueError(
                    f"Bot clone cannot include special file: {path.relative_to(profile_dir)}"
                )
            members += 1
            if members > MAX_BOT_CLONE_MEMBERS:
                raise ValueError(
                    f"Bot clone exceeds the {MAX_BOT_CLONE_MEMBERS} member limit."
                )
            if path.is_file():
                total_bytes += path.stat().st_size
                if total_bytes > MAX_BOT_CLONE_BYTES:
                    raise ValueError(
                        "Bot clone exceeds the 10 MB expanded-size limit."
                    )


def _copy_clone_tree(source: Path, target: Path, *, cron_root: bool = False) -> None:
    """Copy a definition tree while excluding credentials and runtime state."""

    def _ignore(directory: str, contents: list[str]) -> set[str]:
        ignored = {
            name
            for name in contents
            if name.lower() in BOT_CLONE_DENIED_NAMES
            or Path(name).suffix.lower() in BOT_CLONE_DENIED_SUFFIXES
            or name == "__pycache__"
            or name.endswith((".sock", ".tmp"))
        }
        if cron_root and Path(directory) == source:
            ignored.update(name for name in contents if name != "jobs.json")
        return ignored

    shutil.copytree(source, target, ignore=_ignore)


def _scrub_clone_files(staged: Path) -> None:
    """Redact every text file and reject unclassified binary profile data."""
    from agent.redact import redact_sensitive_text

    for path in staged.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in BOT_CLONE_DENIED_NAMES or suffix in BOT_CLONE_DENIED_SUFFIXES:
            raise ValueError(f"Bot clone contains credential file: {path.relative_to(staged)}")
        if suffix in BOT_CLONE_SAFE_BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Bot clone contains unsupported binary file: {path.relative_to(staged)}"
            ) from exc
        if "\x00" in text:
            raise ValueError(
                f"Bot clone contains unsupported binary file: {path.relative_to(staged)}"
            )
        redacted = redact_sensitive_text(text, force=True)
        if redacted != text:
            path.write_text(redacted, encoding="utf-8")


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


def _normalize_clone_script_path(
    value: object,
    *,
    scripts_root: Path,
    allow_absolute: bool,
) -> str:
    """Return a portable path for a regular file owned by ``scripts_root``."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("cron script path must be a non-empty string")
    try:
        raw = Path(value).expanduser()
        if raw.is_absolute() and not allow_absolute:
            raise ValueError("imported cron script path must be relative")
        root = scripts_root.resolve()
        resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("cron script path is invalid") from exc
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("cron script path resolves outside the scripts directory") from exc
    if not resolved.is_file():
        raise ValueError("cron script path does not identify a regular file")
    return relative.as_posix()


def _sanitize_clone_cron_jobs(
    staged: Path, *, source_scripts_root: Optional[Path] = None
) -> None:
    """Retain cron definitions while dropping source-owned state and routing."""
    jobs_path = staged / "cron" / "jobs.json"
    if not jobs_path.is_file():
        return
    try:
        payload = json.loads(jobs_path.read_text(encoding="utf-8-sig"))
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else payload
        if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
            raise ValueError("jobs must be a list of objects")

        sanitized = []
        for job in jobs:
            clone = {key: job[key] for key in BOT_CLONE_CRON_FIELDS if key in job}
            scripts_root = source_scripts_root or staged / "scripts"
            for field in ("script", "monitor_script"):
                if clone.get(field) is None:
                    continue
                relative = _normalize_clone_script_path(
                    clone[field],
                    scripts_root=scripts_root,
                    allow_absolute=source_scripts_root is not None,
                )
                if source_scripts_root is not None and not (
                    staged / "scripts" / Path(relative)
                ).is_file():
                    raise ValueError("cron script file is excluded from the bot clone")
                clone[field] = relative
            repeat = job.get("repeat")
            if isinstance(repeat, dict):
                clone["repeat"] = {"times": repeat.get("times"), "completed": 0}
            clone.update(
                {
                    "enabled": False,
                    "state": "paused",
                    "paused_at": None,
                    "paused_reason": "Imported bot clone requires review.",
                    "next_run_at": None,
                    "deliver": "local",
                }
            )
            sanitized.append(clone)
        jobs_path.write_text(
            json.dumps({"jobs": sanitized}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Bot clone contains invalid cron job definitions.") from exc


def _prepare_imported_clone(staged: Path) -> None:
    """Enforce receiver-owned policy and data boundaries before publication."""
    cron_dir = staged / "cron"
    if cron_dir.is_dir():
        for entry in cron_dir.iterdir():
            if entry.name == "jobs.json":
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    _sanitize_clone_cron_jobs(staged)
    _scrub_clone_files(staged)
    _reset_owner_policies(staged)


def export_bot_profile(name: str, output_path: str) -> tuple[Path, str]:
    """Create a bounded, credential-free bot clone archive."""
    from hermes_cli.profiles import (
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
    _check_clone_source_budget(profile_dir)

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
                _copy_clone_tree(source, target, cron_root=entry == "cron")
            elif source.is_file():
                shutil.copy2(source, target)
        _sanitize_clone_cron_jobs(
            staged, source_scripts_root=profile_dir / "scripts"
        )
        _reset_owner_policies(staged)
        _scrub_clone_files(staged)
        result = Path(make_targz(base, tmpdir, canon))

    try:
        archive_root_dirs(
            result,
            max_bytes=MAX_BOT_CLONE_BYTES,
            max_members=MAX_BOT_CLONE_MEMBERS,
        )
        if result.stat().st_size > MAX_BOT_CLONE_BYTES:
            raise ValueError(
                f"Bot clone exceeds the {MAX_BOT_CLONE_BYTES // 1_000_000} MB transfer limit."
            )
    except BaseException:
        result.unlink(missing_ok=True)
        raise
    return result, bot_id


def _validate_bot_archive(archive: Path) -> tuple[str, str]:
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if archive.stat().st_size > MAX_BOT_CLONE_BYTES:
        raise ValueError(
            f"Bot clone exceeds the {MAX_BOT_CLONE_BYTES // 1_000_000} MB transfer limit."
        )
    roots = archive_root_dirs(
        archive,
        max_bytes=MAX_BOT_CLONE_BYTES,
        max_members=MAX_BOT_CLONE_MEMBERS,
    )
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
            prepare_staged=_prepare_imported_clone,
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
        try:
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
                                raise BotTransferError(
                                    "Remote bot clone exceeds the 10 MB transfer limit."
                                )
                            handle.write(chunk)
        except httpx.HTTPError as exc:
            raise BotTransferError(f"Remote bot clone request failed: {exc}") from exc
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
        try:
            with archive.open("rb") as body, httpx.Client(
                timeout=120.0, follow_redirects=False
            ) as client:
                response = client.post(
                    f"{base}/v1/bots/clone",
                    headers=headers,
                    params=params,
                    content=body,
                )
        except httpx.HTTPError as exc:
            raise BotTransferError(f"Remote bot clone request failed: {exc}") from exc
        if response.status_code != 201:
            raise _response_error(response)
        try:
            payload = response.json()
            remote_name = payload["name"]
            if not isinstance(remote_name, str) or not remote_name:
                raise ValueError("missing name")
            return remote_name, _valid_bot_id(payload["bot_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BotTransferError(
                "Remote gateway returned an invalid bot clone response."
            ) from exc
