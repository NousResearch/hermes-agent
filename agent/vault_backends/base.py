"""Login-backend contract + registry for the browser credential vault.

A ``LoginBackend`` lists login metadata (never secrets) and resolves ONE
password at fill time. External managers (1Password, Bitwarden) additionally
need a per-session unlock; ``resolve_password`` raises ``UnlockRequired``
while locked so the tool can ask the surface to prompt. Handles are
namespaced by ``prefix`` so ``backend_for_handle`` needs no lookup table.
"""

from __future__ import annotations

import logging
import socket
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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

    @abstractmethod
    def list_items(self) -> List[VaultItemMeta]:
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

    def resolve_secret(self, handle: str) -> Dict[str, str]:
        """Full payload of a payment/address item (server-side only). External managers list only
        logins, so the base returns the password-only shape."""
        return {"password": self.resolve_password(handle)}


def run_with_stdin_secret(argv: Sequence[str], *, env: Dict[str, str], secret: str, timeout: float,
                          label: str) -> subprocess.CompletedProcess:
    """Run a manager CLI feeding *secret* on stdin (never argv, never env). Spawn/timeout → RuntimeError."""
    try:
        return subprocess.run(  # noqa: S603 — argv list, no shell
            list(argv), env=env, input=secret + "\n", capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} unlock timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke {label}: {exc}") from exc


def run_with_secret_env(argv: Sequence[str], *, env: Dict[str, str], secret_env: str, secret: str, timeout: float,
                        label: str) -> subprocess.CompletedProcess:
    """Run a manager CLI whose non-interactive contract reads the secret from a named env var.
    The variable is set on the child's environment only (never argv, never our process)."""
    child_env = dict(env)
    child_env[secret_env] = secret
    try:
        return subprocess.run(  # noqa: S603 — argv list, no shell
            list(argv), env=child_env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} unlock timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke {label}: {exc}") from exc


def _cfg() -> Dict:
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("vault") or {}
    return cfg if isinstance(cfg, dict) else {}


def external_backend_classes():
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    return (OnePasswordLoginBackend, BitwardenLoginBackend)


class SourceStatus(str, Enum):
    """The four states a login source can be in, kept distinct so the surface can say which.

    ``not_installed`` the manager CLI is not on the owning host; ``disconnected`` it is present
    but the host cannot be reached / the CLI failed to answer; ``auth_required`` it is reachable
    but this session is locked (sign in ON THE OWNING HOST); ``available`` ready to resolve.
    """

    not_installed = "not_installed"
    disconnected = "disconnected"
    auth_required = "auth_required"
    available = "available"


@dataclass(frozen=True)
class SourceProbe:
    """One login source's discovery result. ``host`` names the machine the manager lives on, so a
    message about it can be acted on; it is never a credential or vault contents."""

    name: str
    installed: bool
    status: SourceStatus
    host: str
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"installed": self.installed, "status": self.status.value,
                "host": self.host, "reason": self.reason}


# The two external managers' CLI names, for binary discovery.
_MANAGER_BINARIES = {"onepassword": "op", "bitwarden": "bw"}


def owning_host(name: str = "") -> str:
    """The machine a manager's CLI runs on.

    A manager CLI is resolved and executed on whatever host runs the agent's terminal backend;
    with a non-local backend (``ssh``) that is the REMOTE machine, and naming the gateway host
    in a status message would point the user at the wrong box. Per the root
    "capability is a property of the SESSION" rule, this reads the session's terminal policy
    (``TERMINAL_SSH_HOST``), never a hardcoded local identity.
    """
    from tools.terminal_scope import terminal_env

    try:
        backend = (terminal_env("TERMINAL_ENV", "") or "").strip().lower()
        if backend == "ssh" and (host := (terminal_env("TERMINAL_SSH_HOST", "") or "").strip()):
            return host
    except Exception:
        logger.debug("terminal policy unavailable while naming the host for %s", name or "vault", exc_info=True)
    return socket.gethostname() or "this host"


def find_manager_binary(name: str) -> Optional[Path]:
    """The manager CLI's path on the owning host, or None. A configured ``binary_path`` is
    honoured first; otherwise the shared resolver checks PATH AND the common install prefixes a
    service PATH omits, so a launchd/systemd gateway detects what a login shell detects."""
    from agent.secret_sources.base import resolve_cli_binary

    section = _cfg().get(name) or {}
    explicit = str(section.get("binary_path") or "") if isinstance(section, dict) else ""
    binary = _MANAGER_BINARIES.get(name)
    if not binary:
        return None
    return resolve_cli_binary(binary, explicit)


def probe(name: str) -> SourceProbe:
    """Discover one manager's real state on its owning host: installed, and if so whether it is
    available, needs authentication there, or is unreachable.

    Metadata only — this never resolves, reads or logs a secret value; a locked manager is
    reported as ``auth_required`` and nothing is read from it.
    """
    host = owning_host(name)
    binary = find_manager_binary(name)
    if binary is None:
        section = _cfg().get(name) or {}
        explicit = str(section.get("binary_path") or "") if isinstance(section, dict) else ""
        reason = (f"vault.{name}.binary_path is set but is not an executable on {host}" if explicit
                  else f"the {name} CLI was not found on {host}")
        return SourceProbe(name=name, installed=False, status=SourceStatus.not_installed, host=host, reason=reason)
    cls = next((c for c in external_backend_classes() if c.name == name), None)
    if cls is None:
        return SourceProbe(name=name, installed=True, status=SourceStatus.available, host=host)
    section = _cfg().get(name) or {}
    try:
        backend = cls(section if isinstance(section, dict) else {})
        unlocked = backend.is_unlocked()
    except Exception as exc:
        # The CLI is installed but could not be interrogated (unreachable host, unreadable
        # manager config, profile scope failure). Name the host so the message is actionable;
        # the exception text is manager metadata, never a value.
        return SourceProbe(name=name, installed=True, status=SourceStatus.disconnected, host=host,
                           reason=f"{cls.display_name} on {host} could not be reached: {str(exc)[:200]}")
    if not unlocked:
        return SourceProbe(name=name, installed=True, status=SourceStatus.auth_required, host=host,
                           reason=f"{cls.display_name} is locked on {host} — authenticate there to use it")
    return SourceProbe(name=name, installed=True, status=SourceStatus.available, host=host)


def is_installed(name: str) -> bool:
    """Is the manager CLI reachable on its owning host — honouring a configured ``binary_path``.
    Thin wrapper over :func:`probe` so every caller sees one discovery answer."""
    return probe(name).installed


def is_opted_out(name: str) -> bool:
    """Has the user turned this manager off (``vault.<name>.enabled: false``)?

    Separate from discovery on purpose: a source can be detected-but-off, and the surface must be
    able to say so without re-running a probe.
    """
    section = _cfg().get(name) or {}
    return isinstance(section, dict) and section.get("enabled") is False


def is_enabled(name: str) -> bool:
    """An installed manager is a login source unless the user opted out (``vault.<name>.enabled: false``).
    Zero-config on purpose: a user with ``bw``/``op`` on PATH should never have to discover a toggle."""
    return not is_opted_out(name) and is_installed(name)


def enabled_backends() -> List[LoginBackend]:
    """Local first (always on), then every detected external manager the user has not turned off."""
    from agent.vault_backends.local import LocalLoginBackend

    cfg = _cfg()
    out: List[LoginBackend] = [LocalLoginBackend()]
    for cls in external_backend_classes():
        if is_enabled(cls.name):
            section = cfg.get(cls.name) or {}
            out.append(cls(section if isinstance(section, dict) else {}))
    return out


def backend_for_handle(handle: str) -> Optional[LoginBackend]:
    return next((b for b in enabled_backends() if b.owns(handle)), None)
