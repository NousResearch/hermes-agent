"""Staged credential-client replacement owned by the client lifecycle."""

from __future__ import annotations

import logging
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


logger = logging.getLogger("run_agent")


class CredentialSwapTicketState(Enum):
    PREPARED = auto()
    COMMITTED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class _CredentialSwapTarget:
    entry_id: Any
    runtime_key: str
    runtime_base: Any
    api_mode: str
    actual_route: bool
    route_changed: bool
    client_kwargs: Optional[dict[str, Any]]
    anthropic_oauth: Optional[bool] = None

    @property
    def resource_key(self) -> tuple[Any, ...]:
        return (
            self.runtime_key,
            self.runtime_base,
            self.api_mode,
            self.actual_route,
            self.client_kwargs,
        )


def _derive_target(owner: Any, entry: Any) -> Optional[_CredentialSwapTarget]:
    runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    runtime_base = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "base_url", None)
        or owner.base_url
    )
    from hermes_cli.providers import is_actual_route

    actual_route = is_actual_route(getattr(owner, "provider", ""), runtime_base)
    if actual_route:
        from hermes_cli.auth import normalize_actual_base_url

        runtime_base = normalize_actual_base_url(runtime_base)
    stripped_base = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
    from hermes_cli.anon_auth import route_can_serve_model

    if not route_can_serve_model(
        getattr(owner, "provider", None), stripped_base, getattr(owner, "model", None)
    ):
        logger.info(
            "Credential %s skipped: its route cannot serve model %s",
            getattr(entry, "id", "?"),
            owner.model,
        )
        return None

    api_mode = "chat_completions" if actual_route else owner.api_mode
    from hermes_cli.route_identity import normalize_route_base_url

    route_changed = normalize_route_base_url(owner.base_url) != normalize_route_base_url(
        stripped_base
    )
    client_kwargs = None
    if api_mode != "anthropic_messages":
        client_kwargs = owner._derive_credential_swap_client_kwargs(
            runtime_key=runtime_key,
            runtime_base=stripped_base,
            api_mode=api_mode,
            route_changed=route_changed,
        )
    return _CredentialSwapTarget(
        entry_id=getattr(entry, "id", None),
        runtime_key=runtime_key,
        runtime_base=stripped_base,
        api_mode=api_mode,
        actual_route=actual_route,
        route_changed=route_changed,
        client_kwargs=client_kwargs,
    )


def _close_candidate(owner: Any, resource: Any, api_mode: str) -> None:
    if resource is None:
        return
    if api_mode == "anthropic_messages":
        with suppress(Exception):
            resource.close()
        return
    owner._close_openai_client(
        resource, reason="credential_rotation_abort", shared=False,
    )


def _build_candidate(owner: Any, target: _CredentialSwapTarget) -> tuple[Any, _CredentialSwapTarget]:
    resource = None
    try:
        if target.api_mode == "anthropic_messages":
            resource = owner._build_direct_anthropic_client(
                target.runtime_key, target.runtime_base,
            )
            target = _CredentialSwapTarget(
                **{
                    **target.__dict__,
                    "anthropic_oauth": owner._derive_anthropic_oauth_flag(
                        target.runtime_key, target.runtime_base,
                    ),
                }
            )
        else:
            resource = owner._create_openai_client(
                target.client_kwargs or {}, reason="credential_rotation", shared=True,
            )
        if resource is None:
            raise RuntimeError("credential replacement client builder returned no client")
        return resource, target
    except Exception:
        _close_candidate(owner, resource, target.api_mode)
        raise


class CredentialSwapTicket:
    """Replacement client built off to the side and published at commit."""

    def __init__(self, owner: Any, target: _CredentialSwapTarget, resource: Any) -> None:
        self._owner = owner
        self._target = target
        self._candidate_resource = resource
        self._candidate_closed = False
        self._installed = False
        self.state = CredentialSwapTicketState.PREPARED

    def _close_candidate(self) -> None:
        if self._candidate_closed or self._candidate_resource is None or self._installed:
            return
        _close_candidate(self._owner, self._candidate_resource, self._target.api_mode)
        self._candidate_closed = True

    def commit(self, entry: Any) -> bool:
        if self.state is CredentialSwapTicketState.COMMITTED:
            return True
        if self.state is CredentialSwapTicketState.ABORTED:
            raise RuntimeError("cannot commit an aborted credential swap ticket")

        final_target = _derive_target(self._owner, entry)
        if final_target is None or final_target.entry_id != self._target.entry_id:
            return False
        if final_target.resource_key != self._target.resource_key:
            try:
                final_resource, final_target = _build_candidate(self._owner, final_target)
            except Exception as exc:
                logger.warning("Failed to rebuild shared primary client (credential_rotation): %s", exc)
                self._close_candidate()
                return False
            self._close_candidate()
            self._candidate_resource = final_resource
            self._candidate_closed = False
            self._target = final_target

        owner = self._owner
        target = self._target
        lock_factory = getattr(owner, "_openai_client_lock", None)
        lock = lock_factory() if callable(lock_factory) else nullcontext()
        try:
            with lock:
                if target.api_mode == "anthropic_messages":
                    original = getattr(owner, "_anthropic_client", None)
                    owner._anthropic_api_key = target.runtime_key
                    owner._anthropic_base_url = target.runtime_base
                    owner._anthropic_client = self._candidate_resource
                    owner._is_anthropic_oauth = bool(target.anthropic_oauth)
                    owner.api_key = target.runtime_key
                    owner.base_url = target.runtime_base
                else:
                    original = getattr(owner, "client", None)
                    owner.api_mode = target.api_mode
                    owner._credential_pool_entry_id = target.entry_id
                    owner.api_key = target.runtime_key
                    owner.base_url = target.runtime_base
                    owner._client_kwargs = dict(target.client_kwargs or {})
                    owner.client = self._candidate_resource
                    if target.actual_route and hasattr(owner, "_transport_cache"):
                        owner._transport_cache.clear()
                owner._credential_pool_entry_id = target.entry_id
        except Exception:
            self._close_candidate()
            raise

        self._installed = True
        self.state = CredentialSwapTicketState.COMMITTED
        if target.api_mode == "anthropic_messages":
            with suppress(Exception):
                original.close()
        else:
            try:
                owner._retire_shared_openai_client(
                    original, reason="replace:credential_rotation",
                )
            except Exception:
                logger.debug("Credential replacement retirement failed", exc_info=True)
        return True

    def abort(self) -> None:
        if self.state is CredentialSwapTicketState.ABORTED:
            return
        self._close_candidate()
        self.state = CredentialSwapTicketState.ABORTED


def prepare_credential_swap(owner: Any, entry: Any) -> Optional[CredentialSwapTicket]:
    target = _derive_target(owner, entry)
    if target is None:
        return None
    try:
        resource, target = _build_candidate(owner, target)
    except Exception as exc:
        logger.warning("Failed to build replacement client (credential_rotation): %s", exc)
        return None
    return CredentialSwapTicket(owner, target, resource)
