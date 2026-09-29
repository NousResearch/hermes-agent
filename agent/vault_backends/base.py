"""Login-backend interface and discovery for browser autofill.

Backends list metadata and retrieve credentials only for server-side filling.
Interactive managers raise ``UnlockRequired`` when they need a masked prompt.
Handle prefixes route requests to the owning backend.
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from copy import deepcopy
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
    name: str                # configuration key under vault
    display_name: str        # user-facing
    prefix: str              # handle prefix ("vault_", "op:", "bw:")
    needs_unlock: bool = False

    @classmethod
    def is_available(cls, config: dict[str, object]) -> bool:
        """Check local prerequisites without authentication or credential access.

        Plugins override this class method. Built-in detection uses the CLI resolver.
        """
        return False

    def owns(self, handle: str) -> bool:
        return handle.startswith(self.prefix)

    def is_unlocked(self) -> bool:
        return True

    def matches_origin(self, meta: VaultItemMeta, origin: str) -> bool:
        """Match a normalized password-fill destination using metadata only.

        The default accepts exact saved origins. Overrides may run on the
        supervisor thread. Only literal True authorizes a destination. The fill
        engine checks the selected origin and inspected fields at write time.
        """
        allowed = meta.allowed_origins or ((meta.origin,) if meta.origin else ())
        return bool(origin) and origin in allowed

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


def external_backend_classes() -> tuple[type[LoginBackend], ...]:
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    from agent.vault_backends.registry import list_backend_classes

    return (OnePasswordLoginBackend, BitwardenLoginBackend, *list_backend_classes())


def is_installed(name: str) -> bool:
    """Check built-in CLI paths or the plugin's availability probe."""
    import shutil
    section = _cfg().get(name) or {}
    explicit = str(section.get("binary_path") or "") if isinstance(section, dict) else ""
    if name == "onepassword":
        if explicit:
            return Path(explicit).is_file()
        from agent.secret_sources.onepassword import find_op
        return find_op() is not None
    if name == "bitwarden":
        if explicit:
            return Path(explicit).is_file()
        return shutil.which("bw") is not None
    cls = next((candidate for candidate in external_backend_classes() if candidate.name == name), None)
    if cls is None:
        return False
    try:
        return cls.is_available(deepcopy(section) if isinstance(section, dict) else {}) is True
    except Exception:  # noqa: BLE001 — a broken plugin must not break vault discovery
        logger.warning("Login backend '%s' availability check failed; skipping", name)
        return False


def is_enabled(name: str) -> bool:
    """Require availability and explicit opt-in for third-party backends."""
    section = _cfg().get(name)
    if name in {"onepassword", "bitwarden"}:
        if isinstance(section, dict) and section.get("enabled") is False:
            return False
    elif not isinstance(section, dict) or section.get("enabled") is not True:
        return False
    return is_installed(name)


def enabled_backends() -> List[LoginBackend]:
    """Local first, then available external managers enabled for this profile."""
    from agent.vault_backends.local import LocalLoginBackend

    cfg = _cfg()
    out: List[LoginBackend] = [LocalLoginBackend()]
    for cls in external_backend_classes():
        if is_enabled(cls.name):
            section = cfg.get(cls.name) or {}
            try:
                out.append(cls(deepcopy(section) if isinstance(section, dict) else {}))
            except Exception:
                # Provider exceptions may carry credentials, even during initialization.
                logger.warning("Login backend '%s' initialization failed; skipping", cls.name)
    return out


def backend_for_handle(handle: str) -> Optional[LoginBackend]:
    return next((b for b in enabled_backends() if b.owns(handle)), None)
