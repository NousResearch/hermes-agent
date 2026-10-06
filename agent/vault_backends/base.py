"""Login-backend contract + registry for the browser credential vault.

A ``LoginBackend`` lists login metadata (never secrets) and resolves ONE
password at fill time. External managers (1Password, Bitwarden) additionally
need a per-session unlock; ``resolve_password`` raises ``UnlockRequired``
while locked so the tool can ask the surface to prompt. Handles are
namespaced by ``prefix`` so ``backend_for_handle`` needs no lookup table.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

from agent.vault_store import VaultItemMeta

logger = logging.getLogger(__name__)


class UnlockRequired(Exception):
    """The backend is locked for this session; the surface must prompt for the master password."""

    def __init__(self, backend: "LoginBackend"):
        super().__init__(f"{backend.display_name} is locked")
        self.backend = backend


class LoginBackend(ABC):
    name: str                # config key: local | onepassword | bitwarden
    display_name: str        # user-facing
    prefix: str              # handle prefix ("vault_", "op:", "bw:")
    needs_unlock: bool = False

    def owns(self, handle: str) -> bool:
        return handle.startswith(self.prefix)

    def is_unlocked(self) -> bool:
        return True

    def try_secretless_unlock(self) -> bool:
        """Unlock without a typed secret by letting the manager's own desktop UI approve it (``bw``'s
        Touch ID / Windows Hello path). Returns True once a session token is stored. The base returns
        False, so the surface prompts for the master password as before. Only ever reached where a
        human can answer — see ``unlock.can_prompt_here``."""
        return False

    @abstractmethod
    def list_items(self) -> list[VaultItemMeta]:
        """Metadata only. Locked external backends return [] (the agent sees a lock hint instead)."""

    @abstractmethod
    def get_meta(self, handle: str) -> Optional[VaultItemMeta]: ...

    @abstractmethod
    def resolve_password(self, handle: str) -> str:
        """Server-side only; raises ``UnlockRequired`` when locked."""

    def resolve_otp(self, handle: str) -> Optional[str]:
        """Current one-time code for a login that stores a TOTP seed, else None (the user is asked).
        Server-side only, like resolve_password."""
        return None

    def resolve_secret(self, handle: str) -> dict[str, str]:
        """Full payload of a payment/address item (server-side only). External managers list only
        logins, so the base returns the password-only shape."""
        return {"password": self.resolve_password(handle)}


def run_with_stdin_secret(argv: Sequence[str], *, env: dict[str, str], secret: str, timeout: float,
                          label: str) -> subprocess.CompletedProcess:
    """Run a manager CLI feeding *secret* on stdin (never argv, never env). Spawn/timeout → RuntimeError."""
    try:
        return subprocess.run(
            list(argv), env=env, input=secret + "\n", capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} unlock timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke {label}: {exc}") from exc


def run_with_secret_env(argv: Sequence[str], *, env: dict[str, str], secret_env: str, secret: str, timeout: float,
                        label: str) -> subprocess.CompletedProcess:
    """Run a manager CLI whose non-interactive contract reads the secret from a named env var.
    The variable is set on the child's environment only (never argv, never our process)."""
    child_env = dict(env)
    child_env[secret_env] = secret
    try:
        return subprocess.run(
            list(argv), env=child_env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} unlock timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke {label}: {exc}") from exc


_CLEANUP_TIMEOUT_S = 5.0


def _kill_process_tree(proc: subprocess.Popen, pgid: Optional[int]) -> bool:
    """Kill a manager CLI and the helpers it spawned; True when nothing survived.

    ``bw`` starts a ``desktop_proxy`` child to reach the desktop app. Walking the tree after the fact
    is not enough on its own: if the CLI exited first, its helper is reparented and disappears from
    that tree, so the process group captured at spawn is the additional guarantee on POSIX. The CLI is
    always signalled directly, and reaped through its own handle, because a bare ``communicate()``
    blocks while a grandchild holds the pipe. One deadline covers the whole cleanup, so a probe cannot
    quietly outlive its cap through two sequential waits.
    """
    import psutil  # lazy, like the rest of the tree; psutil is a pinned core dependency

    deadline = time.monotonic() + _CLEANUP_TIMEOUT_S
    if pgid is not None:
        try:
            # The probe owns its session, so this group holds the CLI and every helper it spawned,
            # including one already reparented away from it.
            # windows-footgun: ok -- pgid is captured only on POSIX (os.getpgid in the runner)
            os.killpg(pgid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass

    try:
        parent = psutil.Process(proc.pid)
        victims = [*parent.children(recursive=True), parent]
    except psutil.Error:
        victims = []
    alive = []
    if victims:
        _gone, alive = psutil.wait_procs(victims, timeout=max(0.0, deadline - time.monotonic()))
    try:
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return False
    return not alive


def run_for_secretless_unlock(argv: Sequence[str], *, env: dict[str, str], timeout: float,
                              label: str) -> Optional[subprocess.CompletedProcess]:
    """Run a manager CLI that mints a session token from a desktop-app approval instead of a secret we
    hand it. stdin is closed, so a CLI falling back to its own password prompt fails fast instead of
    hanging; a timeout or spawn failure returns None and the caller prompts as usual. The child gets
    its own process group, and a timed-out probe is killed with that group before we move on."""
    try:
        proc = subprocess.Popen(  # argv list, no shell
            list(argv), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            start_new_session=True)  # its own group, so a timeout can kill the whole probe
    except OSError as exc:
        logger.debug("%s secretless unlock could not start: %s", label, exc)
        return None
    pgid: Optional[int] = None
    if os.name == "posix":
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = None
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if not _kill_process_tree(proc, pgid):
            logger.warning("%s approval probe left a process behind after timing out; continuing", label)
        logger.debug("%s secretless unlock timed out after %.0fs; prompting instead", label, timeout)
        return None
    return subprocess.CompletedProcess(list(argv), proc.returncode, out, err)


def _cfg() -> dict:
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("vault") or {}
    return cfg if isinstance(cfg, dict) else {}


def external_backend_classes():
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    return (OnePasswordLoginBackend, BitwardenLoginBackend)


def is_installed(name: str) -> bool:
    """Is the manager CLI reachable — honouring a configured ``binary_path`` over PATH."""
    import shutil
    section = _cfg().get(name) or {}
    explicit = str(section.get("binary_path") or "") if isinstance(section, dict) else ""
    if explicit:
        return Path(explicit).is_file()
    if name == "onepassword":
        from agent.secret_sources.onepassword import find_op
        return find_op() is not None
    return shutil.which("bw") is not None


def is_enabled(name: str) -> bool:
    """An installed manager is a login source unless the user opted out (``vault.<name>.enabled: false``).
    Zero-config on purpose: a user with ``bw``/``op`` on PATH should never have to discover a toggle."""
    section = _cfg().get(name) or {}
    if isinstance(section, dict) and section.get("enabled") is False:
        return False
    return is_installed(name)


def enabled_backends() -> list[LoginBackend]:
    """Local first (always on), then every detected external manager the user has not turned off."""
    from agent.vault_backends.local import LocalLoginBackend

    cfg = _cfg()
    out: list[LoginBackend] = [LocalLoginBackend()]
    for cls in external_backend_classes():
        if is_enabled(cls.name):
            section = cfg.get(cls.name) or {}
            out.append(cls(section if isinstance(section, dict) else {}))
    return out


def backend_for_handle(handle: str) -> Optional[LoginBackend]:
    return next((b for b in enabled_backends() if b.owns(handle)), None)
