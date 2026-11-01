"""Login-backend contract + registry for the browser credential vault.

A ``LoginBackend`` lists login metadata (never secrets) and resolves ONE
password at fill time. External managers (1Password, Bitwarden) additionally
need a per-session unlock; ``resolve_password`` raises ``UnlockRequired``
while locked so the tool can ask the surface to prompt. Handles are
namespaced by ``prefix`` so ``backend_for_handle`` needs no lookup table.
"""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from agent.vault_store import VaultItemMeta


class UnlockRequired(Exception):
    """The backend is locked for this session; the surface must prompt for the master password."""

    def __init__(self, backend: LoginBackend):
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

    def fill_context(self) -> AbstractContextManager:
        """Optional profile-exclusive context, entered after the origin pre-check: covers the in-context
        metadata re-check, secret resolution and the CDP write."""
        return nullcontext()

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
            encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} unlock timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke {label}: {exc}") from exc


def run_with_secret_env(argv: Sequence[str], *, env: dict[str, str], secret_env: str, secret: str, timeout: float,
                        label: str, answered: Callable[[str], bool]) -> subprocess.CompletedProcess:
    """Run a manager CLI whose non-interactive contract reads the secret from a named env var.
    The variable is set on the child's environment only (never argv, never our process)."""
    child_env = dict(env)
    child_env[secret_env] = secret
    return run_until_answered(argv, env=child_env, timeout=timeout, label=label, answered=answered,
                              timeout_message=f"{label} unlock timed out after {timeout:.0f}s")


_ANSWER_QUIET_S = 0.5


def run_until_answered(argv: Sequence[str], *, env: dict[str, str], timeout: float, label: str,
                       timeout_message: str, answered: Callable[[str], bool],
                       quiet: float = _ANSWER_QUIET_S) -> subprocess.CompletedProcess:
    """Run a manager CLI that may print its whole answer and then never exit.

    ``bw`` (2026.9+) leaves a keep-alive TLS socket in Node's event loop after any network command
    and its success path never calls ``process.exit()`` (#145121), so waiting for exit turns a
    delivered session token or JSON document into a timeout. A child that exits is returned as
    ``subprocess.run`` would. A child still running once stdout satisfies *answered* and has been
    quiet for *quiet* seconds has finished its work: it is killed and its output returned with
    returncode 0. Spawn failure or the *timeout* deadline -> ``RuntimeError``."""
    try:
        proc = subprocess.Popen(list(argv), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"failed to invoke {label}: {exc}") from exc
    out, err = bytearray(), bytearray()
    last_out = [time.monotonic()]

    def pump(stream, buf: bytearray, stamp: bool) -> None:
        for chunk in iter(lambda: stream.read1(65536), b""):
            buf.extend(chunk)
            if stamp:
                last_out[0] = time.monotonic()

    pipes = ((proc.stdout, out, True), (proc.stderr, err, False))
    readers = [threading.Thread(target=pump, args=pipe, daemon=True)  # health: allow HX012 -- pipe byte pump, reads no profile state
               for pipe in pipes]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                returncode = proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if out and now - last_out[0] >= quiet and answered(_decode(out)):
                returncode = 0
                break
            if now >= deadline:
                raise RuntimeError(timeout_message)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for reader in readers:
            reader.join(timeout=2)
    return subprocess.CompletedProcess(list(argv), returncode, _decode(out), _decode(err))


def _decode(buf: bytearray) -> str:
    return bytes(buf).decode("utf-8", errors="replace")


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
    from agent.vault_backend_registry import enabled_plugin_backends
    return out + enabled_plugin_backends(cfg)


def backend_for_handle(handle: str) -> Optional[LoginBackend]:
    return next((b for b in enabled_backends() if b.owns(handle)), None)
