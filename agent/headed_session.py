"""Headed cloud session interfaces.

Product: one remote display shared by PTY, session-scoped WebVNC, and
``computer_use`` (CUA). This module is the contract only — providers live
behind :mod:`agent.headed_session_registry`. Do not add a second X11/VNC/CUA
stack here.

Compose existing seams instead of reminting them:

* ``computer_use`` provider ABC (#90380) with Linux display (#61310)
* Bot Desktop / hermes-desktop noVNC (#97859, #17258)
* gateway target invariant (#90374) — fail closed on fingerprint mismatch
* browser login handoff (#92524) stays ``host_kind=browser_handoff`` and is
  rejected for CUA on a headed session
* headless Cursor Cloud Agents skill (#107480) is a different product
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

BROWSER_HANDOFF_HOST_KIND = "browser_handoff"


@dataclass(frozen=True)
class DisplayTarget:
    """Stable identity for the headed session's single display.

    ``fingerprint`` is the #90374 invariant: every surface (PTY, WebVNC,
    CUA) must name the same value. Empty fingerprints are invalid.
    """

    fingerprint: str
    host_kind: str = "headed_session"
    label: str = ""

    def __post_init__(self) -> None:
        fp = (self.fingerprint or "").strip()
        if not fp:
            raise ValueError("DisplayTarget.fingerprint must be a non-empty string")
        object.__setattr__(self, "fingerprint", fp)
        kind = (self.host_kind or "").strip()
        if not kind:
            raise ValueError("DisplayTarget.host_kind must be a non-empty string")
        object.__setattr__(self, "host_kind", kind)


@dataclass(frozen=True)
class HeadedSessionSurfaces:
    """The three operator/agent surfaces bound to one :class:`DisplayTarget`."""

    display: DisplayTarget
    pty_endpoint: str
    webvnc_endpoint: str

    def __post_init__(self) -> None:
        if not (self.pty_endpoint or "").strip():
            raise ValueError("pty_endpoint must be a non-empty string")
        if not (self.webvnc_endpoint or "").strip():
            raise ValueError("webvnc_endpoint must be a non-empty string")


class HeadedCloudSessionProvider(abc.ABC):
    """Pluggable backend that provisions a headed cloud session.

    Implementations register via :mod:`agent.headed_session_registry`.
    Core must not implement a second desktop; providers should wrap the
    existing terminal, WebVNC, and ``ComputerUseBackend`` seams.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier (lowercase, hyphens allowed)."""

    @property
    def display_name(self) -> str:
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True when this provider can provision a session."""

    @abc.abstractmethod
    def create_session(self) -> HeadedSessionSurfaces:
        """Provision PTY + WebVNC + CUA on one :class:`DisplayTarget`."""

    @abc.abstractmethod
    def close_session(self, surfaces: HeadedSessionSurfaces) -> None:
        """Tear down a session previously returned by :meth:`create_session`."""


def assert_computer_use_targets_session(
    backend,
    surfaces: HeadedSessionSurfaces,
) -> None:
    """Fail closed unless CUA is bound to ``surfaces.display``.

    * Missing or empty ``bound_display_fingerprint()`` → ``ValueError``
      (#90374: do not silently click the local / wrong display).
    * Fingerprint mismatch → ``ValueError``.
    * ``host_kind=browser_handoff`` (#92524) is refused for CUA. Browser
      takeover stays a browser-only path, not headed-session computer use.
    """
    if surfaces.display.host_kind == BROWSER_HANDOFF_HOST_KIND:
        raise ValueError(
            "computer_use refuses host_kind=browser_handoff; "
            "issue #92524 is browser-only login handoff, not headed CUA"
        )

    bound: Optional[str] = None
    getter = getattr(backend, "bound_display_fingerprint", None)
    if getter is not None:
        bound = getter() if callable(getter) else getter
    if not isinstance(bound, str) or not bound.strip():
        raise ValueError(
            "computer_use is not bound to a headed-session display fingerprint "
            "(#90374 fail-closed)"
        )
    if bound.strip() != surfaces.display.fingerprint:
        raise ValueError(
            f"computer_use fingerprint {bound.strip()!r} does not match "
            f"headed session {surfaces.display.fingerprint!r} (#90374)"
        )
