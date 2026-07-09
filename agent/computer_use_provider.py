"""Computer Use Provider ABC
==========================

Defines the pluggable-backend interface for computer-use providers that
supply a per-task :class:`tools.computer_use.backend.ComputerUseBackend`
(typically a cua-driver instance bound to a per-task display / container).

The default backend (``CuaDriverBackend`` spawned on the host, single
display) lives in :mod:`tools.computer_use.cua_backend` and is used when no
provider is configured. A provider is selected via ``computer_use.provider``
in ``config.yaml``; the active provider services every ``computer_use`` tool
call by returning a backend bound to the caller's ``task_id``.

This ABC mirrors :class:`agent.browser_provider.BrowserProvider` (PR #25214)
and :class:`agent.web_search_provider.WebSearchProvider` (PR #25182) — same
shape, same registration flow. Providers live in their own plugin
directories (e.g. ``~/.hermes/plugins/computer_use/<name>/``) and
self-register via :func:`agent.computer_use_registry.register_provider` at
plugin-discovery time.

Why an ABC and not another core edit: per-thread OS-level keyboard/mouse
requires running cua-driver against a per-task display that lives somewhere
the host can't reach directly (a container's ``:1``, a remote VNC, …). Wiring
that into the singleton backend would special-case one runtime (webtop
containers) into core. Instead, the core gains one small generic hook —
"resolve a per-task backend from a registered provider" — and every concrete
runtime ships as a plugin that implements this ABC. That keeps the core
footprint minimal and lets third-party runtimes (someone else's product)
live outside the tree, per AGENTS.md ("widen the generic plugin surface,
don't special-case it in core"; "third-party products … do NOT land under
plugins/").
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # avoid agent -> tools layering inversion at runtime
    from tools.computer_use.backend import ComputerUseBackend


class ComputerUseProvider(abc.ABC):
    """Abstract base class for a per-task computer-use backend provider.

    Subclasses must implement :meth:`name`, :meth:`is_available`, and the
    three lifecycle methods: :meth:`get_backend`, :meth:`close_backend`,
    :meth:`emergency_cleanup`.

    Contract:

    * :meth:`get_backend` must return a **started** backend (the provider
      owns ``backend.start()`` and any LRU/cap policy over live backends).
      It is called on every ``computer_use`` tool dispatch with the current
      ``task_id``; the provider is expected to return the same backend
      instance for a given ``task_id`` across calls (stable per-task
      binding) and to evict/stop backends for evicted tasks.
    * The returned object must implement the
      :class:`tools.computer_use.backend.ComputerUseBackend` interface.
      The return type is string-annotated (``TYPE_CHECKING``) so this
      ``agent/`` module has no runtime dependency on ``tools/``.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in the ``computer_use.provider``
        config key.

        Lowercase, hyphens permitted. Examples: ``webtop-pool``,
        ``remote-vnc``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``. Defaults to ``name``."""
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Typically a cheap check (Docker reachable, container image present,
        optional Python dep importable). Must NOT make network calls that
        block — this runs at tool-registration time and on every
        ``hermes tools`` paint. Mirrors :meth:`BrowserProvider.is_available`.
        """

    @abc.abstractmethod
    def get_backend(self, task_id: str) -> "ComputerUseBackend":
        """Return a started backend bound to ``task_id``.

        The provider owns the per-task backend cache + LRU/cap policy and
        calls ``backend.start()`` before returning. Repeated calls with the
        same ``task_id`` must return the same backend instance. When the
        provider evicts a task (e.g. its container was LRU-reclaimed), it
        must ``stop()`` that backend.

        May raise ``RuntimeError`` (backend spawn / display setup failure);
        the dispatcher in :mod:`tools.computer_use.tool` surfaces these to
        the user as a ``computer_use backend unavailable`` error.
        """

    @abc.abstractmethod
    def close_backend(self, task_id: str) -> bool:
        """Release a task's backend by ``task_id``.

        Returns True on success, False on failure / unknown task. Should not
        raise — log and return False so the dispatcher's cleanup keeps
        moving.
        """

    @abc.abstractmethod
    def emergency_cleanup(self) -> None:
        """Best-effort teardown of all live backends during process exit.

        Called from atexit / signal handlers. Must tolerate missing
        resources, errors, etc. — log and move on. Must not raise.
        """

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``hermes tools`` picker.

        Default: minimal entry derived from :attr:`display_name`. Override
        to expose prerequisite prompts, badges, and post-setup hooks.
        """
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }
