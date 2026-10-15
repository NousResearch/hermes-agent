"""Own Webex listener dependency installers through cancellation and retries."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import TypeVar
from weakref import WeakValueDictionary

from hermes_constants import find_node_executable, with_hermes_node_path

logger = logging.getLogger(__name__)

_INSTALL_TIMEOUT_SECONDS = 300
_INSTALL_COMPLETION_FILE = ".hermes-install-complete"
_INSTALL_LOCKS: WeakValueDictionary[
    tuple[asyncio.AbstractEventLoop, Path], asyncio.Lock
] = WeakValueDictionary()
_T = TypeVar("_T")


def listener_installation_complete(listener_dir: Path) -> bool:
    """Require a completed install for the current lockfile, beyond partial deps."""
    try:
        completed_digest = (listener_dir / _INSTALL_COMPLETION_FILE).read_text(
            encoding="ascii"
        )
        current_digest = hashlib.sha256(
            (listener_dir / "package-lock.json").read_bytes()
        ).hexdigest()
    except (OSError, UnicodeError):
        return False
    return completed_digest == current_digest


def _begin_install(listener_dir: Path) -> str:
    (listener_dir / _INSTALL_COMPLETION_FILE).unlink(missing_ok=True)
    return hashlib.sha256(
        (listener_dir / "package-lock.json").read_bytes()
    ).hexdigest()


def _complete_install(listener_dir: Path, lock_digest: str) -> None:
    (listener_dir / _INSTALL_COMPLETION_FILE).write_text(lock_digest, encoding="ascii")


async def _finish_owned_task(task: asyncio.Future[_T]) -> _T:
    """Finish owned cleanup before propagating a caller's cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def install_listener_dependencies(listener_dir: Path) -> bool:
    """Install without leaving an npm writer behind a cancelled connection."""
    runtime_path = await asyncio.to_thread(listener_dir.resolve)
    key = (asyncio.get_running_loop(), runtime_path)
    lock = _INSTALL_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _INSTALL_LOCKS[key] = lock

    # Gateway deadlines detach cancelled connects; a retry must await their reap.
    async with lock:
        npm = find_node_executable("npm")
        if not npm:
            logger.error("[Webex] npm is required to install listener dependencies")
            return False
        try:
            lock_digest = await _finish_owned_task(
                asyncio.create_task(asyncio.to_thread(_begin_install, listener_dir))
            )
            process = await asyncio.create_subprocess_exec(
                npm,
                "ci",
                "--omit=optional",
                # Locked packages work without install scripts. Keep cancellation
                # ownership to npm, without lifecycle children writing after it dies.
                "--ignore-scripts",
                cwd=str(listener_dir),
                env=with_hermes_node_path(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.error("[Webex] Listener dependency install failed: %s", exc)
            return False

        try:
            await asyncio.wait_for(process.wait(), timeout=_INSTALL_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.error("[Webex] Listener dependency install timed out")
            return False
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                # health: allow HX006 -- killed npm must be reaped before releasing the directory lock
                reap = asyncio.create_task(process.wait())
                await _finish_owned_task(reap)

        if process.returncode != 0:
            logger.error(
                "[Webex] Listener dependency install exited with code %s",
                process.returncode,
            )
            return False
        try:
            await _finish_owned_task(
                asyncio.create_task(
                    asyncio.to_thread(_complete_install, listener_dir, lock_digest)
                )
            )
        except OSError as exc:
            logger.error("[Webex] Could not record completed listener install: %s", exc)
            return False
        return True
