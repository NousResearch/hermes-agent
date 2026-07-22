"""Secret-source contract: the ABC every secret backend implements.

A *secret source* resolves credentials from an external secret manager
(Bitwarden Secrets Manager, 1Password, an OS keystore, a user script, ...)
into environment-variable-shaped values at process startup, AFTER
``~/.hermes/.env`` has loaded and BEFORE the rest of Hermes reads
``os.environ``.

Scope of the contract (deliberate, please do not widen):

* **Read-only.**  Sources resolve refs → values.  There is no write-back
  ("save this key to your vault"), no arbitrary secret objects, and no
  mid-session secret API.  If a future need for rotation/refresh appears
  it will arrive as a versioned optional hook — do not bolt it on.
* **Startup-time, synchronous.**  ``fetch()`` is called once per process
  (per HERMES_HOME) by the orchestrator in
  :mod:`agent.secret_sources.registry`, which enforces a wall-clock
  timeout around it.  Sources must not spawn background refreshers.
* **Never raises, never prompts.**  ``fetch()`` returns a
  :class:`FetchResult` — errors go in ``result.error`` with a
  machine-readable :class:`ErrorKind`.  Interactive auth belongs in the
  source's CLI ``setup`` flow, never on the startup path (non-TTY
  gateway/cron startup must never block on stdin).
* **Sources fetch; the orchestrator applies.**  A source returns the
  name→value mapping it *would* contribute.  Precedence (mapped-beats-bulk,
  first-wins, ``override_existing``, protected vars), conflict warnings,
  provenance tracking, and the actual ``os.environ`` writes are owned by
  the orchestrator so no backend can get them wrong.

Versioning: ``SECRET_SOURCE_API_VERSION`` gates plugin compatibility.
New *optional* hooks with default implementations do not bump it;
required-signature changes do, and the registry skips (with a warning)
sources built against a different major version instead of crashing
startup.
"""

from __future__ import annotations

import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence

# Bump ONLY for breaking changes to the required contract surface
# (abstract-method signatures, FetchResult required fields).  Additive
# optional hooks must ship with defaults and must NOT bump this.
SECRET_SOURCE_API_VERSION = 1

# Timeout the orchestrator enforces around fetch() when the source's
# config section doesn't override it.  Generous because a first run may
# include a one-time CLI binary auto-install (e.g. bws download+verify).
DEFAULT_FETCH_TIMEOUT_SECONDS = 120.0

# Default timeout for run_secret_cli() subprocess invocations.
DEFAULT_CLI_TIMEOUT_SECONDS = 30.0


class ErrorKind(str, Enum):
    """Machine-readable failure taxonomy for :class:`FetchResult.error`.

    A fixed vocabulary keeps startup warnings and ``hermes secrets status``
    uniform across backends, and lets the orchestrator implement
    kind-dependent policy (e.g. a future stale-cache fallback on
    ``NETWORK``/``TIMEOUT`` but not on ``AUTH_FAILED``) exactly once.
    """

    NOT_CONFIGURED = "not_configured"    # enabled but missing token/project/map
    BINARY_MISSING = "binary_missing"    # helper CLI not found / not installed
    AUTH_FAILED = "auth_failed"          # bad credentials
    AUTH_EXPIRED = "auth_expired"        # credentials were valid, aren't now
    REF_INVALID = "ref_invalid"          # a secret reference failed validation
    NETWORK = "network"                  # transport-level failure
    EMPTY_VALUE = "empty_value"          # backend returned nothing for a ref
    TIMEOUT = "timeout"                  # fetch exceeded its wall-clock budget
    INTERNAL = "internal"                # anything else (bug, unexpected shape)


@dataclass
class FetchResult:
    """Outcome of one source's fetch.

    ``secrets`` holds what the source *would* contribute; whether each
    var is actually applied is the orchestrator's decision.  ``applied``
    and ``skipped`` exist for backward compatibility with the original
    Bitwarden fetch-and-apply entry point and are left empty by
    conforming ``fetch()`` implementations.
    """

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    error_kind: Optional[ErrorKind] = None
    # Path of the helper binary used, when the source is CLI-driven.
    # Surfaced by status commands; None for SDK/API-driven sources.
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class SecretSource(ABC):
    """One external secret backend.

    Subclasses set the class attributes and implement :meth:`fetch`.
    Everything else has a sensible default.

    Attributes:
        name: Config-section key under ``secrets:`` in config.yaml.
            Lowercase ``[a-z0-9_]+``.  Also the provenance label stored
            for every var this source supplies.
        label: Human-readable name used in startup messages and
            ``hermes secrets status`` (e.g. ``"Bitwarden Secrets Manager"``).
        shape: ``"mapped"`` when the user explicitly binds env-var names
            to refs (1Password ``env:`` map, command source) or
            ``"bulk"`` when the backend injects whole projects/folders
            of secrets implicitly (Bitwarden BSM).  The orchestrator
            gives mapped sources precedence over bulk sources: an
            explicit binding is stronger intent than a project dump.
        scheme: Optional URI scheme this source owns for secret
            references (``"op"`` for ``op://...``).  Must be unique
            across registered sources — refs may eventually appear
            outside the ``secrets:`` block (e.g. credential-pool
            ``api_key`` fields), so scheme collisions are rejected at
            registration time to keep that future possible.
        api_version: Contract version this source was built against.
    """

    api_version: int = SECRET_SOURCE_API_VERSION
    name: str = ""
    label: str = ""
    shape: str = "mapped"  # "mapped" | "bulk"
    scheme: Optional[str] = None

    # -- required ----------------------------------------------------------

    @abstractmethod
    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        """Resolve this source's secrets. MUST NOT raise or prompt.

        ``cfg`` is the source's raw config section (``secrets.<name>``)
        from config.yaml — treat every field defensively, the section
        may be malformed.  ``home_path`` is the resolved HERMES_HOME.
        """

    # -- optional hooks (defaults are correct for most sources) ------------

    def is_enabled(self, cfg: dict) -> bool:
        """Whether the user turned this source on."""
        return bool(isinstance(cfg, dict) and cfg.get("enabled"))

    def override_existing(self, cfg: dict) -> bool:
        """May this source overwrite vars that .env / the shell already set?

        This NEVER extends to vars claimed by another secret source in the
        same startup pass — cross-source overrides are a config error the
        orchestrator warns about, not a knob.
        """
        return bool(isinstance(cfg, dict) and cfg.get("override_existing", False))

    def protected_env_vars(self, cfg: dict) -> FrozenSet[str]:
        """Env vars the orchestrator must never let ANY source overwrite.

        Typically the source's own bootstrap-auth var (e.g.
        ``BWS_ACCESS_TOKEN``) so a vault that contains its own access
        token can't clobber the credential used to reach it.
        """
        return frozenset()

    def fetch_timeout_seconds(self, cfg: dict) -> float:
        """Wall-clock budget the orchestrator enforces around fetch()."""
        try:
            val = float((cfg or {}).get("timeout_seconds", DEFAULT_FETCH_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            return DEFAULT_FETCH_TIMEOUT_SECONDS
        return val if val > 0 else DEFAULT_FETCH_TIMEOUT_SECONDS

    def config_schema(self) -> dict:
        """Optional description of this source's config keys.

        Shape: ``{key: {"description": str, "default": Any}}``.  Used by
        setup surfaces to render config without hardcoding per-source
        knowledge.  Purely informational.
        """
        return {}

    def remediation(self, kind: Optional["ErrorKind"], cfg: dict) -> str:
        """One-line, actionable next step for a failed fetch.

        Called by the startup status printer (and ``hermes secrets ...
        status``) right after a fetch error is surfaced, so the user sees
        *what to run* next to fix it — not just what broke.  Sources
        should override this to point at their own CLI verbs (e.g.
        ``hermes secrets bitwarden token`` for AUTH_FAILED).  Return an
        empty string to suppress the hint.

        Must never raise and must not perform I/O — it's a pure
        kind→string mapping on the startup path.
        """
        generic = {
            ErrorKind.NOT_CONFIGURED: (
                f"Run `hermes secrets {self.name} setup` to finish configuration."
            ),
            ErrorKind.BINARY_MISSING: (
                f"Run `hermes secrets {self.name} setup` to install the helper CLI."
            ),
            ErrorKind.AUTH_FAILED: (
                f"Credentials rejected — run `hermes secrets {self.name} setup` "
                "to re-authenticate."
            ),
            ErrorKind.AUTH_EXPIRED: (
                f"Credentials expired — run `hermes secrets {self.name} setup` "
                "to re-authenticate."
            ),
            ErrorKind.NETWORK: (
                "Network problem reaching the secrets backend — check "
                "connectivity and retry."
            ),
            ErrorKind.TIMEOUT: (
                f"Backend was slow — raise secrets.{self.name}.timeout_seconds "
                "if this recurs."
            ),
        }
        return generic.get(kind, "") if kind is not None else ""


# ---------------------------------------------------------------------------
# Shared helpers — use these instead of hand-rolling per backend
# ---------------------------------------------------------------------------


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ANSI CSI/OSC escape sequences — helper-CLI stderr often carries color
# codes that must not reach Hermes' own startup output.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?)")


def is_valid_env_name(name: str) -> bool:
    """True when ``name`` is a legal environment-variable name."""
    return bool(name) and bool(_ENV_NAME_RE.match(name))


def scrub_ansi(text: str) -> str:
    """Strip ANSI escape sequences (whole CSI/OSC sequences, not just ESC)."""
    return _ANSI_RE.sub("", text or "")


def run_secret_cli(
    argv: Sequence[str],
    *,
    allow_env: Sequence[str] = (),
    extra_env: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_CLI_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    """Run a secret-manager helper CLI with a minimal, allowlisted env.

    Security posture shared by every subprocess-driven backend:

    * argv list only — never ``shell=True``.  Callers pass user-supplied
      reference strings AFTER a ``--`` option terminator in their argv.
    * The child gets ``PATH``/``HOME``/locale basics plus only the env
      vars named in ``allow_env`` (auth/session vars) and ``extra_env``
      — never a copy of the full post-dotenv ``os.environ``, which by
      this point holds every credential Hermes knows about.
    * ``NO_COLOR=1`` is set and stderr/stdout are ANSI-scrubbed so
      helper diagnostics can't smuggle escape sequences into Hermes
      output.
    * stdin is ``/dev/null`` so a helper that decides to prompt fails
      fast instead of hanging startup.

    Raises ``RuntimeError`` on spawn failure or timeout (message safe to
    surface); returns the completed process otherwise — callers own
    returncode interpretation.
    """
    base_keep = ("PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TMPDIR", "TEMP",
                 "LANG", "LC_ALL", "XDG_CONFIG_HOME", "XDG_DATA_HOME")
    env: Dict[str, str] = {}
    for key in (*base_keep, *allow_env):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    if extra_env:
        env.update(extra_env)
    env.setdefault("NO_COLOR", "1")

    try:
        proc = subprocess.run(  # noqa: S603 — argv list, no shell
            list(argv),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{Path(str(argv[0])).name} timed out after {timeout:.0f}s"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"failed to invoke {Path(str(argv[0])).name}: {exc}"
        ) from exc

    proc.stdout = proc.stdout or ""
    proc.stderr = scrub_ansi(proc.stderr or "")
    return proc
