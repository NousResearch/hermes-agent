"""
Abstract base class for session-orchestration agent adapters.

All concrete adapters (claude-code, omp, …) must subclass ``AgentAdapter``
and implement every abstract method.  The watcher discovers adapters via
the registry; the relay calls ``drive()``; the Discord command flow calls
``launch()``.

Lifecycle contract
------------------
1. ``launch()`` spawns the tmux session and returns a ``SessionHandle``.
   The caller stores the handle in the registry.
2. ``drive()`` delivers a user prompt to the running session (pane).
   It must acquire and release the per-session lock itself or coordinate
   with the caller.
3. ``detect()`` inspects the current pane state and returns a
   ``SessionLifecycle`` value.  It is called by the watcher on every
   tick and must be safe to call concurrently with ``drive()`` (the watcher
   skips capture while the per-session lock is held by ``drive()``).
4. ``resume()`` performs a ``/clear``+re-inject cycle when the session
   is in ``PAUSED_HANDOFF`` state.  It must be idempotent: calling it
   on a non-handoff session must be a safe no-op.

Capability assertion
--------------------
At watcher startup the watcher calls ``capabilities()`` on each adapter
and smoke-tests the declared values against observed behaviour.  A mismatch
logs an error for that adapter and marks it disabled; it does NOT crash the
watcher.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle


class TuiNotReadyError(RuntimeError):
    """Raised by ``drive(type_ahead=True)`` when the TUI composer is not present.

    Signals the caller (the Discord-drive path) that a type-ahead delivery could
    not land — the pane is booting or dead — so it should fall back to the
    persistent pending-drive queue instead of dropping the message.
    """


class AgentAdapter(ABC):
    """Abstract base for all session-orchestration agent adapters.

    Subclasses must implement ``launch``, ``drive``, ``detect``, and
    ``resume``.  They should also override ``capabilities()`` to return
    an accurate ``Capabilities`` descriptor; the default returns all-False
    capabilities with no regex or dialog handlers.
    """

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Return a descriptor of what this adapter supports.

        Override in subclasses to declare accurate values.  The default
        returns a conservative all-False descriptor (no print mode, no
        hooks, no RPC/JSON mode, no idle regex, no dialog handlers).
        """
        return Capabilities()

    # ------------------------------------------------------------------
    # Abstract lifecycle methods
    # ------------------------------------------------------------------

    @abstractmethod
    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        """Spawn a new tmux session and start the agent in ``workdir``.

        Parameters
        ----------
        workdir:
            Absolute path to the working directory for the agent session.
        prompt:
            The initial prompt to inject once the agent is ready.

        Returns
        -------
        SessionHandle
            A fully-populated handle (session_id, tmux_session, pane,
            launch_ts) that the caller stores in the registry.
        """

    @abstractmethod
    def drive(
        self,
        handle: SessionHandle,
        message: str,
        *,
        pre_keys: list[str] | None = None,
        type_ahead: bool = False,
    ) -> None:
        """Deliver ``message`` to the running session.

        Implementations must perform a prompt-readiness check before
        pasting to avoid injecting into a mid-render pane.  The preferred
        mechanism is ``load-buffer`` / ``paste-buffer`` (not ``send-keys``)
        to avoid shell-metacharacter expansion.

        Parameters
        ----------
        handle:
            The ``SessionHandle`` returned by ``launch()``.
        message:
            The user message to send to the agent.
        pre_keys:
            Optional tmux key names (e.g. ``["Escape"]``) to send before the
            readiness check + paste. Used to cancel a selection menu so a
            natural-language answer can be pasted into the freed composer.
        type_ahead:
            When True, deliver as soon as the TUI composer is present, WITHOUT
            waiting for the agent to be idle — the composer queues the turn
            while the agent works. If the composer/TUI is not present at all
            (booting / dead pane), raise :class:`TuiNotReadyError` so the caller
            can fall back to the persistent queue. When False (default), the
            strict "ready and not busy" readiness gate applies.
        """

    @abstractmethod
    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        """Inspect the current pane and return a lifecycle state.

        This method is called by the watcher on every tick.  It must be
        safe to call without holding the per-session lock; the watcher
        skips ``detect`` (or uses a stale cached value) when the lock is
        held by ``drive()``.

        Parameters
        ----------
        handle:
            The ``SessionHandle`` for the session to inspect.

        Returns
        -------
        SessionLifecycle
            Exactly one of ``RUNNING | WAITING_USER | PAUSED_HANDOFF |
            STALLED | DONE | ERROR``.
        """

    @abstractmethod
    def resume(self, handle: SessionHandle, prompt: str, *, force: bool = False) -> None:
        """Perform a ``/clear``+re-inject cycle for a handoff session.

        Called by the relay when ``detect()`` returns ``PAUSED_HANDOFF``, and by
        the watcher when a ``handoff_continue`` marker fires (``force=True``).
        Must be idempotent: without ``force``, calling on a session that is NOT
        in handoff state is a safe no-op (log a warning; do not raise).

        Parameters
        ----------
        handle:
            The ``SessionHandle`` for the session to resume.
        prompt:
            The resume command / prompt to inject after clearing. When empty,
            only ``/clear`` is sent (no follow-up drive).
        force:
            When True, skip the ``detect()==PAUSED_HANDOFF`` gate — the caller
            (the watcher, acting on an authoritative ``handoff_continue`` marker)
            has already decided a resume is due, independent of pane state.
        """

    @abstractmethod
    def terminate(self, handle: SessionHandle) -> None:
        """Kill the agent's tmux process.

        Must be safe to call on an already-dead session (``check=False``
        on tmux kill).  Concrete impls live in the claude-code and omp
        adapters (T003/T004).

        Parameters
        ----------
        handle:
            The ``SessionHandle`` for the session to kill.
        """
