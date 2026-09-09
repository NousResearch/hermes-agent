"""Login-backend contract + registry for the browser credential vault.

A ``LoginBackend`` lists login metadata (never secrets) and resolves ONE
password at fill time. External managers (1Password, Bitwarden) additionally
need a per-session unlock; ``resolve_password`` raises ``UnlockRequired``
while locked so the tool can ask the surface to prompt. Handles are
namespaced by ``prefix`` so ``backend_for_handle`` needs no lookup table.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agent.vault_store import VaultItemMeta


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


def enabled_backends() -> List[LoginBackend]:
    """Local first (always on), then each enabled external manager, in config order."""
    from agent.vault_backends.local import LocalLoginBackend

    cfg = _cfg()
    out: List[LoginBackend] = [LocalLoginBackend()]
    for cls in external_backend_classes():
        section = cfg.get(cls.name) or {}
        if isinstance(section, dict) and section.get("enabled"):
            out.append(cls(section))
    return out


def backend_for_handle(handle: str) -> Optional[LoginBackend]:
    return next((b for b in enabled_backends() if b.owns(handle)), None)
