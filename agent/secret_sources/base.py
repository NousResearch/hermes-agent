"""Secret-source contract: the ABC every secret backend implements.

A *secret source* resolves credentials from an external secret manager into
env-var-shaped values at process startup, AFTER ``~/.hermes/.env`` has loaded
and BEFORE the rest of Hermes reads ``os.environ``. The contract is deliberately
narrow: read-only; startup-time and synchronous (one ``fetch()`` per process per
HERMES_HOME, under a registry-enforced wall-clock timeout, no background
refreshers); never raises, never prompts (errors go in ``FetchResult.error``
with an :class:`ErrorKind`; interactive auth belongs in the CLI ``setup`` flow);
sources fetch, the orchestrator (``registry.apply_all``) applies.

``SECRET_SOURCE_API_VERSION`` gates plugin compatibility: additive optional
hooks with defaults do NOT bump it; required-signature changes do.
"""

from __future__ import annotations

import os
import re
import subprocess
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, MutableMapping, Optional, Sequence, Tuple

SECRET_SOURCE_API_VERSION = 1

# Generous: a first run may include a one-time CLI auto-install (bws download).
DEFAULT_FETCH_TIMEOUT_SECONDS = 120.0
DEFAULT_CLI_TIMEOUT_SECONDS = 30.0

_SOURCE_ENVIRONMENT: ContextVar[Optional[MutableMapping[str, str]]]
_SOURCE_ENVIRONMENT = ContextVar("hermes_secret_source_environment", default=None)


def set_source_environment(environ: MutableMapping[str, str]) -> Token:
    """Install a per-fetch environment view without changing ``os.environ``."""
    return _SOURCE_ENVIRONMENT.set(environ)


def reset_source_environment(token: Token) -> None:
    _SOURCE_ENVIRONMENT.reset(token)


def get_source_environment() -> MutableMapping[str, str]:
    """Return the active per-fetch environment, or the process environment."""
    environ = _SOURCE_ENVIRONMENT.get()
    return environ if environ is not None else os.environ


def source_child_env() -> Dict[str, str]:
    """Environment for a helper child that legitimately needs the caller's env:
    full process env (minus the terminal blocklist) in single-profile startup;
    ONLY the per-fetch view under multiplex, so no sibling profile's secrets leak."""
    source_env = get_source_environment()
    if source_env is os.environ:
        from tools.environments.local import build_subprocess_env

        return build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    return dict(source_env)


class ErrorKind(str, Enum):
    """Failure taxonomy for :class:`FetchResult.error`; lets the orchestrator apply
    kind-dependent policy once (stale-cache fallback on NETWORK/TIMEOUT, never AUTH_FAILED)."""

    NOT_CONFIGURED = "not_configured"    # enabled but missing token/project/map
    BINARY_MISSING = "binary_missing"    # helper CLI not found / not installed
    AUTH_FAILED = "auth_failed"          # bad credentials
    AUTH_EXPIRED = "auth_expired"        # credentials were valid, aren't now
    REF_INVALID = "ref_invalid"          # a secret reference failed validation
    NETWORK = "network"                  # transport-level failure
    EMPTY_VALUE = "empty_value"          # backend returned nothing for a ref
    TIMEOUT = "timeout"                  # fetch exceeded its wall-clock budget
    INTERNAL = "internal"                # anything else (bug, unexpected shape)


# Ordered (kind, substrings) rules for mapping CLI failure text onto ErrorKind;
# first rule whose substring appears (case-insensitive) wins.
ErrorRules = Sequence[Tuple[ErrorKind, Sequence[str]]]


def classify_cli_error(message: str, rules: ErrorRules) -> ErrorKind:
    """Best-effort mapping of helper-CLI failure text onto the taxonomy."""
    lowered = message.lower()
    for kind, tokens in rules:
        if any(tok in lowered for tok in tokens):
            return kind
    return ErrorKind.INTERNAL


def coerce_float(value: Any, default: float) -> float:
    """``float(value)`` with ``default`` for malformed config values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class FetchResult:
    """Outcome of one source's fetch. ``secrets`` is what the source *would*
    contribute; ``applied``/``skipped`` serve the legacy fetch-and-apply entry
    points and stay empty in ``fetch()``."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    error_kind: Optional[ErrorKind] = None
    # Helper binary used (CLI-driven sources); surfaced by status commands.
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def fail(self, error: str, kind: ErrorKind) -> "FetchResult":
        self.error, self.error_kind = error, kind
        return self


_GENERIC_REMEDIATION = {
    ErrorKind.NOT_CONFIGURED: "Run `hermes secrets {name} setup` to finish configuration.",
    ErrorKind.BINARY_MISSING: "Run `hermes secrets {name} setup` to install the helper CLI.",
    ErrorKind.AUTH_FAILED: "Credentials rejected — run `hermes secrets {name} setup` to re-authenticate.",
    ErrorKind.AUTH_EXPIRED: "Credentials expired — run `hermes secrets {name} setup` to re-authenticate.",
    ErrorKind.NETWORK: "Network problem reaching the secrets backend — check connectivity and retry.",
    ErrorKind.TIMEOUT: "Backend was slow — raise secrets.{name}.timeout_seconds if this recurs.",
}


class SecretSource(ABC):
    """One external secret backend. Subclasses set attributes + ``fetch``.

    ``name``: config-section key under ``secrets:`` (``[a-z0-9_]+``) and the
    provenance label. ``shape``: ``"mapped"`` (user binds env-var names to refs)
    or ``"bulk"`` (backend injects whole projects); mapped beats bulk because an
    explicit binding is stronger intent. ``scheme``: URI scheme this source owns
    for refs, unique across sources. ``token_env_key`` / ``default_token_env``:
    config key naming the bootstrap-auth env var and its default; drives
    :meth:`protected_env_vars` so a vault holding its own access token can't
    clobber the credential used to reach it. ``remediation_hints``: per-kind
    overrides of the generic remediation text (``{name}`` / ``{token_env}``).
    """

    api_version: int = SECRET_SOURCE_API_VERSION
    name: str = ""
    label: str = ""
    shape: str = "mapped"  # "mapped" | "bulk"
    scheme: Optional[str] = None
    token_env_key: Optional[str] = None
    default_token_env: str = ""
    override_existing_default: bool = False
    remediation_hints: Dict[ErrorKind, str] = {}

    @abstractmethod
    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        """Resolve this source's secrets. MUST NOT raise or prompt; ``cfg`` is the
        raw ``secrets.<name>`` section and may be malformed."""

    def is_enabled(self, cfg: dict) -> bool:
        return bool(isinstance(cfg, dict) and cfg.get("enabled"))

    def override_existing(self, cfg: dict) -> bool:
        """May this source overwrite vars .env / the shell already set? Never extends
        to vars claimed by another source (a config error the orchestrator warns about)."""
        return bool(isinstance(cfg, dict)
                    and cfg.get("override_existing", self.override_existing_default))

    def token_env(self, cfg: dict) -> str:
        """Name of the env var holding this source's bootstrap credential."""
        if isinstance(cfg, dict) and self.token_env_key:
            return str(cfg.get(self.token_env_key) or self.default_token_env)
        return self.default_token_env

    def protected_env_vars(self, cfg: dict) -> FrozenSet[str]:
        """Env vars the orchestrator must never let ANY source overwrite."""
        return frozenset({self.token_env(cfg)}) if self.token_env_key else frozenset()

    def fetch_timeout_seconds(self, cfg: dict) -> float:
        """Wall-clock budget the orchestrator enforces around fetch()."""
        val = coerce_float((cfg or {}).get("timeout_seconds", DEFAULT_FETCH_TIMEOUT_SECONDS),
                           DEFAULT_FETCH_TIMEOUT_SECONDS)
        return val if val > 0 else DEFAULT_FETCH_TIMEOUT_SECONDS

    def config_schema(self) -> dict:
        """Informational ``{key: {"description": str, "default": Any}}`` for setup UIs."""
        return {}

    def remediation(self, kind: Optional["ErrorKind"], cfg: dict) -> str:
        """One-line actionable next step for a failed fetch (pure); "" suppresses the hint."""
        if kind is None:
            return ""
        template = self.remediation_hints.get(kind) or _GENERIC_REMEDIATION.get(kind, "")
        return template.format(name=self.name, token_env=self.token_env(cfg))


# --- Shared helpers — use these instead of hand-rolling per backend ---------

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Deliberately NOT tools.ansi_strip.strip_ansi: the optional terminator here
# also strips *unterminated* OSC sequences (a CLI killed mid-write), which
# strip_ansi leaves untouched.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?)")


def is_valid_env_name(name: str) -> bool:
    """True when ``name`` is a legal environment-variable name."""
    return bool(name) and bool(_ENV_NAME_RE.match(name))


def scrub_ansi(text: str) -> str:
    """Strip ANSI escape sequences (whole CSI/OSC sequences, not just ESC)."""
    return _ANSI_RE.sub("", text or "")


def run_cli(argv: Sequence[str], *, env: Dict[str, str], timeout: float, label: str,
            timeout_message: str, stdin: Any = subprocess.DEVNULL) -> subprocess.CompletedProcess:
    """``subprocess.run`` an argv list (never a shell), capturing utf-8 text; timeout
    and spawn failure become ``RuntimeError``. Callers own returncode interpretation."""
    try:
        return subprocess.run(  # noqa: S603 — argv list, no shell
            list(argv), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, stdin=stdin,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(timeout_message) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke {label}: {exc}") from exc


def run_secret_cli(argv: Sequence[str], *, allow_env: Sequence[str] = (), extra_env: Optional[Dict[str, str]] = None,
                   timeout: float = DEFAULT_CLI_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    """Run a secret-manager helper CLI with a minimal, allowlisted env (never the
    full post-dotenv ``os.environ``): PATH/HOME/locale basics plus ``allow_env``
    and ``extra_env``. ``NO_COLOR=1`` + ANSI-scrubbed stderr; stdin is /dev/null so
    a prompting helper fails fast. Pass user refs AFTER a ``--`` terminator."""
    base_keep = ("PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TMPDIR", "TEMP",
                 "LANG", "LC_ALL", "XDG_CONFIG_HOME", "XDG_DATA_HOME")
    env = {k: os.environ[k] for k in (*base_keep, *allow_env) if k in os.environ}
    if extra_env:
        env.update(extra_env)
    env.setdefault("NO_COLOR", "1")

    name = Path(str(argv[0])).name
    proc = run_cli(argv, env=env, timeout=timeout, label=name,
                   timeout_message=f"{name} timed out after {timeout:.0f}s")
    proc.stdout = proc.stdout or ""
    proc.stderr = scrub_ansi(proc.stderr or "")
    return proc
