"""Terminal Environment Provider ABC.

Pluggable-backend interface for terminal execution environments (cloud sandboxes, remote
runners). Providers register via :meth:`PluginContext.register_terminal_environment_provider`;
:func:`tools.terminal_tool_backends._create_environment` consults the registry for any ``TERMINAL_ENV``
/ ``terminal.backend`` value that is not a built-in (built-ins stay in ``tools/environments/``;
third-party sandbox vendors do NOT have to live in core). :meth:`create_environment` returns
any ``BaseEnvironment`` duck type (``execute()``, ``cleanup()`` …); the factory stamps
``_hermes_backend_name`` on the result so file-path resolution can identify plugin backends.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

from agent.provider_base import ProviderBase


class TerminalEnvironmentProvider(ProviderBase):
    """Abstract base class for a pluggable terminal execution backend.

    :attr:`name` is the ``terminal.backend`` / ``TERMINAL_ENV`` value (``[a-z0-9_]``); the
    registry rejects built-in backend names.

    Classification contract — each core policy decision that was historically a frozenset of
    built-in names is a declarative attribute so a new backend cannot silently miss one:

    * ``is_remote`` — commands run off-host: suppresses host OS/home/cwd hints in the system
      prompt, the host Python probe, and remote-aware skill env handling.
    * ``is_container`` — own filesystem rooted away from the host: container resource config
      passed through, host-looking cwds sanitized, file tools use container path resolution.
    * ``skip_container_guards`` — sandbox disposable enough to skip dangerous-command approval
      prompts. Defaults to ``is_container``; backends that can mount host paths override to False.
    * ``cache_path_base`` — where auto-synced ``~/.hermes/cache`` files land inside the backend
      (``"~/.hermes"``, ``"/root/.hermes"``), or ``None`` when host paths remain correct.
    * ``strip_env_keys`` — vendor credential env vars, stripped from every subprocess the agent
      spawns so a model-authored command can never read them.
    * ``session_isolated_when_nonpersistent`` — non-persistent mode gives each session its own
      sandbox identity; opt in when a shared name would let two ephemeral runs destroy each other.
    """

    is_remote: bool = True
    is_container: bool = True
    session_isolated_when_nonpersistent: bool = False

    @property
    def description(self) -> str:
        """One-line description shown in backend pickers."""
        return f"Run commands in a {self.display_name} environment."

    @property
    def skip_container_guards(self) -> bool:
        return self.is_container

    @property
    def cache_path_base(self) -> Optional[str]:
        return None

    @property
    def strip_env_keys(self) -> frozenset:
        return frozenset()

    @property
    def env_description(self) -> str:
        """Prompt-builder fallback for where commands run when the live probe fails (e.g. ``"a Daytona workspace (Linux)"``)."""
        return f"a {self.display_name} environment (likely Linux)"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this backend can service commands. Cheap, NO network calls: runs during UI paints."""

    def check_requirements(self, config: Dict[str, Any]) -> bool:
        """Full requirements check with the merged terminal env config; log actionable errors before returning False."""
        return self.is_available()

    def probe(self) -> Tuple[str, str]:
        """Dashboard picker health ``(status, detail)``; status ``ready``/``needs_setup``/``unavailable``. Never raise; <~2s."""
        return ("ready", "") if self.is_available() else ("needs_setup", f"{self.display_name} is not configured.")

    def setup_instructions(self) -> List[str]:
        """Lines printed by ``hermes setup`` after selection (the wizard persists ``terminal.backend`` itself)."""
        return []

    def post_setup(self) -> None:
        """Optional interactive hook run by ``hermes setup`` after selection (prompt for tokens, install SDKs)."""

    def doctor_checks(self) -> List[Tuple[bool, str, str]]:
        """``hermes doctor`` rows ``(ok, label, detail)``; default reflects :meth:`is_available`."""
        try:
            ok = bool(self.is_available())
        except Exception:
            ok = False
        return [(ok, f"{self.display_name} backend", "(configured)" if ok else "(not configured — see setup instructions)")]

    @abc.abstractmethod
    def create_environment(
        self, *, cwd: str, timeout: int, task_id: str = "default", image: Optional[str] = None,
        container_config: Optional[Dict[str, Any]] = None, **kwargs: Any,
    ):
        """Create an execution environment (``BaseEnvironment`` duck type). MUST accept ``**kwargs`` and ignore
        unknown keys so the factory can evolve without breaking older plugins. ``task_id`` keys reuse/persistence;
        ``container_config`` carries ``container_cpu/memory/disk/persistent`` when :attr:`is_container`."""
