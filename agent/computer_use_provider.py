"""Computer Use provider ABC.

``computer_use`` drives a real keyboard and mouse, and which machine's
keyboard and mouse that is has turned into the subsystem's central question:
the host display (today's default), a per-task container display, a leased
cloud sandbox, or the desktop client on the other end of a remote gateway.
Each of those is somebody's concrete runtime, and wiring any one of them into
the dispatcher would special-case a third-party product in core.

So the dispatcher gains one generic seam instead — "ask the active provider
for a backend" — and every concrete runtime ships as a plugin implementing
this ABC. Mirrors :class:`agent.browser_provider.BrowserProvider` and
:class:`agent.web_search_provider.WebSearchProvider`: same shape, same
registration flow, same config-key selection.

Providers are **factories, not caches.** ``tools.computer_use.tool`` already
owns one backend per Hermes session, its call lock, its recorded permission
mode, and the release path that stops it (``release_computer_use_session``).
A provider that kept its own per-task cache would shadow all of that and
reintroduce the cross-session bleed the session cache exists to prevent — so
a provider only answers "build me one", and the caller owns its life.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # avoid an agent -> tools layering inversion at runtime
    from tools.computer_use.backend import ComputerUseBackend


class ComputerUseProvider(abc.ABC):
    """A source of computer-use backends.

    Subclasses implement :attr:`name`, :meth:`is_available`, and
    :meth:`create_backend`. Everything else has a working default, so adding
    a method here later cannot break a provider that already shipped.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier matched against ``computer_use.provider``.

        Lowercase, hyphens permitted. Examples: ``local``, ``webtop-pool``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label for ``hermes tools``. Defaults to :attr:`name`."""
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this provider can service calls right now.

        Must be cheap and must not block on the network: this runs on every
        tool-registration pass and every ``hermes tools`` paint. A provider
        that supplies its own displays should answer for those rather than
        for the host's — that is what lets the tool appear on a headless
        gateway.
        """

    @abc.abstractmethod
    def create_backend(self, session_id: str, permission_mode: str) -> "ComputerUseBackend":
        """Build an **unstarted** backend for one Hermes session.

        The caller starts it, caches it, locks around its calls, and stops it
        on release — see the module docstring. Return a fresh instance per
        call; handing back a shared one puts two sessions on the same driver
        target namespace.

        ``permission_mode`` is the already-resolved cua-driver mode
        (``standard`` / ``bounded`` / ``unrestricted``) for this session. It
        cannot change after driver startup, which is why it is a construction
        argument rather than something the backend reads for itself.

        Raise here when the runtime is known to be gone — a container pool
        that is down, an expired lease. Nothing has been spawned yet, so the
        dispatcher reports the cause as ``computer_use backend unavailable``
        instead of the symptom a start() timeout would produce minutes later.
        Answering that question is each provider's own, because the cheap
        check that is right for a leased sandbox is wrong for the host: an
        absent cua-driver binary already gates the tool out of the schema,
        and re-checking it here would only refuse calls the caller has
        deliberately pointed at a backend of their own.
        """

    def routing_identity(self) -> str:
        """Which machine a backend from this provider is pointed at, as a key.

        The dispatcher caches one started backend per Hermes session and
        rebuilds it when this string changes, so a provider that can retarget —
        a lease that moved, a Desktop that reconnected as somebody else — must
        fold whatever it retargets on into the answer. Providers with exactly
        one target (the host's display) need not override.
        """
        return self.name

    def unavailable_reason(self) -> str:
        """Why :meth:`is_available` said no, in one sentence for the model.

        The dispatcher refuses the call before building a backend, and the
        generic default names only the provider. A provider whose absence has
        a specific, actionable cause — nobody has connected a desktop, the
        lease expired — should say so, because that string is the whole of
        what the model gets to reason about.
        """
        return f"computer_use provider {self.name!r} is not available"

    def get_status(self) -> Dict[str, Any]:
        """Readiness detail for the Desktop / dashboard Computer Use card.

        Defaults to the host's own report. A provider that drives some other
        machine must override this — a container pool answering with the
        gateway's Accessibility grants describes a display nobody is looking
        at. Payload shape: :func:`tools.computer_use.permissions.computer_use_status`.
        """
        from tools.computer_use.permissions import computer_use_status

        return computer_use_status()

    def emergency_cleanup(self) -> None:
        """Best-effort teardown of anything the provider owns outside a backend.

        Called from the process ``atexit`` hook, after every live backend has
        already been stopped. Leased sandboxes and spawned containers outlive
        a backend object, so this is where they get released. Must not raise.
        """
        return None

    def get_setup_schema(self) -> Dict[str, Any]:
        """Provider metadata for the ``hermes tools`` picker."""
        return {"name": self.display_name, "badge": "", "tag": "", "env_vars": []}
