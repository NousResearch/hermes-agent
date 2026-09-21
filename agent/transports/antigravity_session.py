"""Stateful adapter for an Antigravity client conversation.

The client protocol is deliberately small: it owns transport/protocol details while this
adapter owns one external conversation id, the workspace cwd and interruption lifecycle.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AntigravityTurnResult:
    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    usage: Any = None
    interrupted: bool = False
    error: Optional[str] = None
    conversation_id: Optional[str] = None
    should_retire: bool = False


class AntigravitySession:
    """One external Antigravity conversation, owned by one ``AIAgent``."""

    def __init__(self, *, cwd: str, model: Optional[str] = None, client: Any = None,
                 client_factory: Optional[Callable[..., Any]] = None,
                 projector_factory: Optional[Callable[[], Any]] = None, event_callback: Optional[Callable[..., Any]] = None):
        self.cwd = cwd
        self.model = model
        self._client = client
        self._client_factory = client_factory
        self._projector_factory = projector_factory
        self._event_callback = event_callback
        self.conversation_id: Optional[str] = None
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._busy = False
        self._closed = False

    def _get_client(self):
        if self._client is None:
            factory = self._client_factory
            if factory is None:
                from agent.transports.antigravity_cli import AntigravityClient
                factory = AntigravityClient
            self._client = factory(cwd=self.cwd)
        return self._client

    def request_interrupt(self) -> None:
        """Cancel the in-flight client request; a stop while idle is a no-op."""
        with self._state_lock:
            if not self._busy:
                return
            self._cancel_event.set()
        client = self._client
        cancel = getattr(client, "cancel", None) if client is not None else None
        if callable(cancel):
            cancel()

    def run_turn(self, prompt: Any) -> AntigravityTurnResult:
        if self._closed:
            return AntigravityTurnResult(error="Antigravity session is closed", should_retire=True)
        with self._state_lock:
            if self._busy:
                return AntigravityTurnResult(error="Antigravity session is already busy", should_retire=False)
            self._busy = True
            self._cancel_event.clear()
        projector_factory = self._projector_factory
        if projector_factory is None:
            from agent.transports.antigravity_event_projector import AntigravityEventProjector
            projector_factory = AntigravityEventProjector
        projector = projector_factory()
        protocol_error: Optional[str] = None

        def on_event(event: Any) -> None:
            nonlocal protocol_error
            projected = projector.feed(event) if projector is not None else None
            if projected is not None and getattr(projected, "error", None):
                protocol_error = str(projected.error)
                self.request_interrupt()
            if self._event_callback is not None:
                if getattr(self._event_callback, "_accepts_projected", False):
                    self._event_callback(event, projected)
                else:
                    self._event_callback(event)

        try:
            response = self._get_client().run_turn(
                prompt, conversation_id=self.conversation_id, model=self.model,
                event_callback=on_event, cancel_event=self._cancel_event,
            )
        except Exception as exc:
            from agent.transports.antigravity_cli import AntigravityCancelled
            projected_messages = list(getattr(projector, "projected_messages", []) or [])
            if protocol_error:
                return AntigravityTurnResult(
                    projected_messages=projected_messages, error=protocol_error,
                    conversation_id=self.conversation_id, should_retire=True,
                )
            if isinstance(exc, AntigravityCancelled):
                return AntigravityTurnResult(
                    projected_messages=projected_messages, interrupted=True,
                    conversation_id=self.conversation_id,
                )
            return AntigravityTurnResult(
                projected_messages=projected_messages, error=str(exc), should_retire=True,
                conversation_id=self.conversation_id,
            )
        finally:
            with self._state_lock:
                self._busy = False
                self._cancel_event.clear()
        conversation_id = getattr(response, "conversation_id", None)
        if conversation_id:
            self.conversation_id = str(conversation_id)
        interrupted = bool(getattr(response, "interrupted", False))
        error = getattr(response, "error", None)
        final_text = getattr(response, "final_text", None) or getattr(projector, "final_text", "") or ""
        return AntigravityTurnResult(
            final_text=str(final_text), projected_messages=list(getattr(projector, "projected_messages", []) or []),
            usage=getattr(response, "usage", None), interrupted=interrupted, error=str(error) if error else None,
            conversation_id=self.conversation_id, should_retire=bool(error),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        close = getattr(client, "close", None) if client is not None else None
        if callable(close):
            close()
