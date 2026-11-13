"""Terminal Environment Provider ABC.

Pluggable-backend interface for terminal execution environments (cloud sandboxes, remote
runners). Providers register via :meth:`PluginContext.register_terminal_environment_provider`;
:func:`hermes_agent.tools.terminal_tool_backends._create_environment` consults the registry for any ``TERMINAL_ENV``
/ ``terminal.backend`` value that is not a built-in (built-ins stay in ``hermes_agent/tools/environments/``;
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
    * ``default_cwd`` — where commands start when ``terminal.cwd`` is unset or unusable there: an
      absolute in-sandbox path, or ``"~"`` for each sandbox's own home. With ``"~"``,
      ``create_environment`` receives ``cwd="~"`` (never expanded on the host) and the environment sets
      ``self.cwd`` to the absolute home it resolved; commands may also arrive with ``cwd="~"``.
      ``None`` keeps the generic ``/root``.
    * ``supports_streams`` — environments implement ``exec_prefix(*, user=None, interactive=True)``:
      a local argv that runs its remainder (``bash -c <script>``, passed through as separate words)
      inside the sandbox with stdin/stdout open, or ``None`` when it cannot right now. Bot Desktop,
      ``computer_use`` and the browser then follow the terminal into this backend, as they do for
      docker/ssh, instead of refusing.

    * ``serves_hosted_scopes`` — the backend runs a hosted tenant's agent: every tool of a hosted scope
      (terminal, code execution, background processes, file tools) runs in its environment, and a
      hosted scope refuses any backend without it (built-ins included). See below.

    ``default_cwd`` home resolution and ``exec_prefix`` need the environment object itself, so they
    work for in-process plugins only; under ``plugins.isolation: host`` the environment reaches Hermes
    as an opaque placeholder and Bot Desktop placement refuses a ``supports_streams`` backend.

    **Hosted scopes** (``hermes_agent/tools/environments/sandbox_scope.py``). With a hosted
    :class:`hermes_scope.Scope` bound, the sandbox is the agent's: its identity is
    ``SandboxIdentity(scope.tenant, scope.agent)`` and nothing else, so one agent has one sandbox
    whatever task or session a tool call carries. :meth:`create_environment` then also receives
    ``scope=`` (the bound scope) and ``sandbox=`` (the identity), and ``task_id`` is the identity's
    key. The environment it returns is wrapped by core in ``ScopedEnvironment``, which refuses any call
    made under another agent's scope (or under none) before it reaches the backend, and passes the
    bound scope as ``scope=`` to every operation. The environment must implement the hosted operation
    interface (``sandbox_scope.require_hosted_interface``: ``execute``, ``cleanup``, ``stop_idle`` (suspend:
    the workspace is kept and resumes on the next call; an idle hosted sandbox is never destroyed instead)
    and the long-lived process methods, plus ``fetch_file``/``fetch_realpath``/``get_temp_dir`` when
    present, each accepting ``scope=``) or it is refused at creation. The streams core hands back are proxies
    that refuse every pipe and ``poll`` operation under another agent's scope. The long-lived process
    methods are:

    * ``start_process(name, argv, *, child_env=None, scope)`` → ``ProcessStream`` with stdin and stdout
      open as byte pipes; ``attach_process(name, *, scope)`` → the running process's stream, or
      ``None``; ``stop_process(name, *, scope)`` → whether one was stopped. ``name`` is unique per
      sandbox. Suspending the sandbox once such a process is idle-stopped, and network isolation
      between sandboxes, are the backend's.

    Credentials reach a hosted sandbox as broker references (``CredentialRef``), never values from
    the secret store; no broker exists yet, so core passes none.
    """

    is_remote: bool = True
    is_container: bool = True
    session_isolated_when_nonpersistent: bool = False
    supports_streams: bool = False
    serves_hosted_scopes: bool = False

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
    def default_cwd(self) -> Optional[str]:
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

    def check_requirements(self, config: dict[str, Any]) -> bool:
        """Full requirements check with the merged terminal env config; log actionable errors before returning False."""
        return self.is_available()

    def probe(self) -> tuple[str, str]:
        """Dashboard picker health ``(status, detail)``; status ``ready``/``needs_setup``/``unavailable``. Never raise; <~2s."""
        return ("ready", "") if self.is_available() else ("needs_setup", f"{self.display_name} is not configured.")

    def setup_instructions(self) -> list[str]:
        """Lines printed by ``hermes setup`` after selection (the wizard persists ``terminal.backend`` itself)."""
        return []

    def post_setup(self) -> None:
        """Optional interactive hook run by ``hermes setup`` after selection (prompt for tokens, install SDKs)."""

    def doctor_checks(self) -> list[tuple[bool, str, str]]:
        """``hermes doctor`` rows ``(ok, label, detail)``; default reflects :meth:`is_available`."""
        try:
            ok = bool(self.is_available())
        except Exception:
            ok = False
        return [(ok, f"{self.display_name} backend", "(configured)" if ok else "(not configured — see setup instructions)")]

    @abc.abstractmethod
    def create_environment(
        self, *, cwd: str, timeout: int, task_id: str = "default", image: Optional[str] = None,
        container_config: Optional[dict[str, Any]] = None, **kwargs: Any,
    ):
        """Create an execution environment (``BaseEnvironment`` duck type). MUST accept ``**kwargs`` and ignore
        unknown keys so the factory can evolve without breaking older plugins. ``task_id`` keys reuse/persistence;
        ``container_config`` carries ``container_cpu/memory/disk/persistent`` when :attr:`is_container`.
        Inside a hosted scope (only for :attr:`serves_hosted_scopes`) it also receives ``scope`` and
        ``sandbox``; the sandbox to create or resume is ``sandbox`` alone."""
