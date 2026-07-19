"""Immutable per-message capability boundary for trusted gateway adapters.

A :class:`SecurityContext` is data, not provenance.  The gateway binds it to an
opaque, per-adapter capability at intake and tool execution revalidates through
that same capability.  Platform strings and mutable registry entries are never
used as proof of origin.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hashlib
import json
import re
import threading
import weakref
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,255}$")
_SAFE_HASH = re.compile(r"^[a-f0-9]{64}$")
_CURRENT_SECURITY_CONTEXT: contextvars.ContextVar["SecurityContext | None"] = contextvars.ContextVar(
    "gateway_security_context", default=None
)


class SecurityContextError(PermissionError):
    """Raised when a capability envelope is absent, malformed or exceeded."""


SecurityRevalidator = Callable[["SecurityContext", str], Awaitable[None]]
_CAPABILITY_SENTINEL = object()


DEFAULT_REVALIDATION_TIMEOUT_SECONDS = 10.0
MAX_SECURITY_CONTEXT_AGE_SECONDS = 300.0
MAX_SECURITY_CONTEXT_CLOCK_SKEW_SECONDS = 30.0



def require_fresh_security_context(
    context: "SecurityContext", *, now: datetime | None = None
) -> None:
    """Reject stale or implausibly future-dated capability evidence."""
    current = now or datetime.now(timezone.utc)
    age = (current - context.authenticated_at).total_seconds()
    if age > MAX_SECURITY_CONTEXT_AGE_SECONDS:
        raise SecurityContextError("security_context_expired")
    if age < -MAX_SECURITY_CONTEXT_CLOCK_SKEW_SECONDS:
        raise SecurityContextError("security_context_from_future")


class AdapterSecurityCapability:
    """Opaque identity binding one live adapter to its gateway revalidator."""

    __slots__ = (
        "_adapter_ref", "_platform", "_loop", "_revalidator", "_active",
        "_revalidation_timeout_seconds",
    )

    def __init__(
        self,
        sentinel: object,
        adapter: Any,
        loop: asyncio.AbstractEventLoop,
        revalidator: SecurityRevalidator,
        revalidation_timeout_seconds: float,
    ) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise TypeError("AdapterSecurityCapability cannot be constructed directly")
        self._adapter_ref = weakref.ref(adapter)
        self._platform = str(getattr(getattr(adapter, "platform", None), "value", ""))
        self._loop = loop
        self._revalidator = revalidator
        self._active = True
        self._revalidation_timeout_seconds = revalidation_timeout_seconds

    def belongs_to(self, adapter: Any) -> bool:
        return self._active and self._adapter_ref() is adapter

    def belongs_to_adapter_source(self, source: Any) -> bool:
        return bool(
            self._active
            and self._adapter_ref() is not None
            and self._platform
            == str(getattr(getattr(source, "platform", None), "value", ""))
        )

    def revoke(self) -> None:
        self._active = False

    async def revalidate_async(
        self, context: "SecurityContext", dispatch_name: str
    ) -> None:
        require_fresh_security_context(context)
        if not self._active or self._adapter_ref() is None or self._loop.is_closed():
            raise SecurityContextError("security_adapter_unavailable")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not self._loop:
            raise SecurityContextError("security_revalidation_off_gateway_loop")
        try:
            await asyncio.wait_for(
                self._revalidator(context, dispatch_name),
                timeout=self._revalidation_timeout_seconds,
            )
            require_fresh_security_context(context)
        except asyncio.TimeoutError as exc:
            raise SecurityContextError("security_revalidation_timeout") from exc
        except PermissionError:
            raise
        except Exception as exc:
            raise SecurityContextError("security_revalidation_failed") from exc

    def revalidate_sync(self, context: "SecurityContext", tool_name: str) -> None:
        require_fresh_security_context(context)
        if not self._active or self._adapter_ref() is None or self._loop.is_closed():
            raise SecurityContextError("security_adapter_unavailable")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            # Tool dispatch is intentionally synchronous and must run in the
            # gateway executor. Blocking its own loop would deadlock.
            raise SecurityContextError("security_revalidation_on_gateway_loop")
        future = asyncio.run_coroutine_threadsafe(
            self._revalidator(context, tool_name), self._loop
        )
        try:
            future.result(timeout=self._revalidation_timeout_seconds)
            require_fresh_security_context(context)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise SecurityContextError("security_revalidation_timeout") from exc
        except PermissionError:
            raise
        except Exception as exc:
            raise SecurityContextError("security_revalidation_failed") from exc


def issue_adapter_security_capability(
    adapter: Any,
    loop: asyncio.AbstractEventLoop,
    revalidator: SecurityRevalidator,
    *,
    revalidation_timeout_seconds: float = DEFAULT_REVALIDATION_TIMEOUT_SECONDS,
) -> AdapterSecurityCapability:
    """Issue an opaque capability for one exact adapter instance.

    GatewayRunner calls this while wiring a reviewed adapter.  The returned
    object is checked by identity; names, registry entries and event metadata
    cannot substitute for it.
    """
    if (
        adapter is None
        or not callable(revalidator)
        or loop is None
        or isinstance(revalidation_timeout_seconds, bool)
        or not isinstance(revalidation_timeout_seconds, (int, float))
        or revalidation_timeout_seconds <= 0
    ):
        raise SecurityContextError("invalid_security_adapter_binding")
    return AdapterSecurityCapability(
        _CAPABILITY_SENTINEL,
        adapter,
        loop,
        revalidator,
        float(revalidation_timeout_seconds),
    )


def bind_context_to_adapter(
    context: "SecurityContext", capability: AdapterSecurityCapability
) -> "SecurityContext":
    if not isinstance(context, SecurityContext):
        raise SecurityContextError("untrusted_security_context_type")
    if not isinstance(capability, AdapterSecurityCapability):
        raise SecurityContextError("missing_adapter_security_capability")
    require_fresh_security_context(context)
    existing = context._adapter_capability
    if existing is not None and existing is not capability:
        raise SecurityContextError("security_context_provenance_mismatch")
    return context if existing is capability else replace(context, _adapter_capability=capability)


def _clean_set(values: Iterable[str], *, field_name: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise SecurityContextError(f"{field_name}_must_be_sequence")
    result: set[str] = set()
    for raw in values:
        value = str(raw)
        if value != value.strip() or not value or not _SAFE_ID.fullmatch(value):
            raise SecurityContextError(f"invalid_{field_name}")
        result.add(value)
    return frozenset(result)


def canonical_capability_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SecurityContext:
    principal_id: str
    tenant_id: str
    platform: str
    authority: str
    domains: frozenset[str] = field(default_factory=frozenset)
    capability_bundle: str = "denied"
    capability_hash: str = ""
    allowed_toolsets: frozenset[str] = field(default_factory=frozenset)
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    allowed_actions: frozenset[str] = field(default_factory=frozenset)
    expose_private_context: bool = False
    expose_memory: bool = False
    expose_identity: bool = True
    policy_revision: str = ""
    authenticated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence_source: str = "live"
    _adapter_capability: AdapterSecurityCapability | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "principal_id", "tenant_id", "platform", "authority",
            "capability_bundle", "policy_revision", "evidence_source",
        ):
            value = str(getattr(self, name))
            if value != value.strip() or not value or not _SAFE_ID.fullmatch(value):
                raise SecurityContextError(f"invalid_{name}")
        for name in (
            "domains", "allowed_toolsets", "allowed_tools", "denied_tools",
            "allowed_actions",
        ):
            object.__setattr__(self, name, _clean_set(getattr(self, name), field_name=name))
        if self.allowed_tools & self.denied_tools:
            raise SecurityContextError("tool_allow_deny_overlap")
        for name in (
            "expose_private_context", "expose_memory", "expose_identity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SecurityContextError(f"invalid_{name}")
        when = self.authenticated_at
        if not isinstance(when, datetime) or when.tzinfo is None:
            raise SecurityContextError("authenticated_at_must_be_timezone_aware")
        expected = self.derive_capability_hash()
        supplied = str(self.capability_hash or "")
        if supplied and (not _SAFE_HASH.fullmatch(supplied) or supplied != expected):
            raise SecurityContextError("capability_hash_mismatch")
        object.__setattr__(self, "capability_hash", expected)

    def authority_payload(self) -> Mapping[str, Any]:
        """Canonical projection of deterministic effective authority.

        Authentication timestamps and evidence labels are freshness metadata.
        They are verified independently and intentionally do not split sessions.
        """
        return {
            "principal_id": self.principal_id,
            "tenant_id": self.tenant_id,
            "platform": self.platform,
            "authority": self.authority,
            "domains": sorted(self.domains),
            "capability_bundle": self.capability_bundle,
            "allowed_toolsets": sorted(self.allowed_toolsets),
            "allowed_tools": sorted(self.allowed_tools),
            "denied_tools": sorted(self.denied_tools),
            "allowed_actions": sorted(self.allowed_actions),
            "expose_private_context": self.expose_private_context,
            "expose_memory": self.expose_memory,
            "expose_identity": self.expose_identity,
            "policy_revision": self.policy_revision,
        }

    def derive_capability_hash(self) -> str:
        return canonical_capability_hash(self.authority_payload())

    def verify(self) -> None:
        if self.capability_hash != self.derive_capability_hash():
            raise SecurityContextError("capability_hash_mismatch")

    @property
    def denied(self) -> bool:
        return self.authority == "denied" or self.capability_bundle == "denied"

    def permits_tool(self, tool_name: str) -> bool:
        name = str(tool_name or "")
        return bool(not self.denied and name in self.allowed_tools and name not in self.denied_tools)

    def permits_action(self, action: str) -> bool:
        name = str(action or "")
        return bool(not self.denied and name and ("*" in self.allowed_actions or name in self.allowed_actions))

    def public_summary(self) -> Mapping[str, Any]:
        return {
            "principal_id": self.principal_id,
            "tenant_id": self.tenant_id,
            "platform": self.platform,
            "authority": self.authority,
            "domains": sorted(self.domains),
            "capability_bundle": self.capability_bundle,
            "capability_hash": self.capability_hash,
            "expose_private_context": self.expose_private_context,
            "expose_memory": self.expose_memory,
            "expose_identity": self.expose_identity,
            "policy_revision": self.policy_revision,
            "evidence_source": self.evidence_source,
        }


def agent_context_options(context: SecurityContext | None) -> dict[str, bool]:
    """Translate explicit exposure capabilities into ``AIAgent`` options.

    A missing security context is the legacy gateway path and retains the
    agent's existing context, identity, and memory defaults.
    """
    if context is None:
        return {
            "skip_context_files": False,
            "load_soul_identity": False,
            "skip_memory": False,
        }
    if not isinstance(context, SecurityContext):
        raise SecurityContextError("untrusted_security_context_type")
    context.verify()
    return {
        "skip_context_files": not context.expose_private_context,
        "load_soul_identity": context.expose_identity,
        "skip_memory": not context.expose_memory,
    }


def bind_security_context(context: SecurityContext | None) -> contextvars.Token:
    if context is not None:
        if not isinstance(context, SecurityContext):
            raise SecurityContextError("untrusted_security_context_type")
        context.verify()
    return _CURRENT_SECURITY_CONTEXT.set(context)


def reset_security_context(token: contextvars.Token) -> None:
    _CURRENT_SECURITY_CONTEXT.reset(token)


def current_security_context(*, required: bool = False) -> SecurityContext | None:
    context = _CURRENT_SECURITY_CONTEXT.get()
    if required and context is None:
        raise SecurityContextError("missing_security_context")
    return context


def require_action(action: str, *, domains: Iterable[str] = ()) -> SecurityContext:
    context = current_security_context(required=True)
    assert context is not None
    context.verify()
    if not context.permits_action(action):
        raise SecurityContextError("action_denied")
    if not _clean_set(domains, field_name="required_domains").issubset(context.domains):
        raise SecurityContextError("domain_denied")
    return context


async def require_action_for_context(
    context: SecurityContext | None,
    action: str,
    *,
    domains: Iterable[str] = (),
) -> SecurityContext | None:
    """Authorize an exact control action and live-revalidate provenance.

    ``None`` is a legacy context and preserves existing gateway behaviour.
    Secure contexts never inherit wildcard command authority.
    """
    if context is None:
        return None
    if not isinstance(context, SecurityContext):
        raise SecurityContextError("untrusted_security_context_type")
    context.verify()
    name = str(action or "")
    if not name or context.denied or name not in context.allowed_actions:
        raise SecurityContextError("action_denied")
    if not _clean_set(domains, field_name="required_domains").issubset(context.domains):
        raise SecurityContextError("domain_denied")
    capability = context._adapter_capability
    if not isinstance(capability, AdapterSecurityCapability):
        raise SecurityContextError("missing_adapter_security_capability")
    await capability.revalidate_async(context, f"action:{name}")
    return context


def filter_tool_schemas(
    schemas: Sequence[Mapping[str, Any]], context: SecurityContext
) -> list[dict[str, Any]]:
    context.verify()
    if context.denied:
        return []
    filtered: list[dict[str, Any]] = []
    for schema in schemas or ():
        try:
            name = str(schema["function"]["name"])
        except (KeyError, TypeError):
            continue
        if context.permits_tool(name):
            filtered.append(dict(schema))
    return filtered


def build_security_context_prompt(context: SecurityContext) -> str:
    context.verify()
    summary = context.public_summary()
    return (
        "## Authenticated Channel Security Boundary\n\n"
        f"Authority: `{summary['authority']}`. "
        f"Capability bundle: `{summary['capability_bundle']}`. "
        f"Domains: `{', '.join(summary['domains']) or 'none'}`. "
        f"Policy revision: `{summary['policy_revision']}`.\n\n"
        "The gateway has already filtered context and tools. Do not claim access "
        "outside this boundary. Explain denials naturally. Never request, reveal, "
        "or transport credentials or secret values through this channel."
    )


def apply_security_context_to_agent(agent: Any, context: SecurityContext) -> None:
    """Bind/revalidate a capability-scoped agent and narrow its current tools."""
    context.verify()
    existing = getattr(agent, "_security_context", None)
    if existing is not None:
        if not isinstance(existing, SecurityContext):
            raise SecurityContextError("invalid_agent_security_context")
        existing.verify()
        if existing.capability_hash != context.capability_hash:
            raise SecurityContextError("cached_agent_capability_mismatch")
        if existing._adapter_capability is not context._adapter_capability:
            raise SecurityContextError("cached_agent_provenance_mismatch")
    agent.tools = filter_tool_schemas(getattr(agent, "tools", ()) or (), context)
    agent.valid_tool_names = {
        str(item["function"]["name"]) for item in agent.tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    agent._security_context = context
    agent._capability_allowed_tools = frozenset(agent.valid_tool_names)


def enforce_agent_tool_dispatch(agent: Any, tool_name: str) -> None:
    """Exact check and live revalidation immediately before dispatch."""
    context = getattr(agent, "_security_context", None)
    if context is None:
        return
    if not isinstance(context, SecurityContext):
        raise SecurityContextError("invalid_agent_security_context")
    context.verify()
    if context.denied:
        raise SecurityContextError("context_denied")
    if tool_name not in getattr(agent, "_capability_allowed_tools", frozenset()):
        raise SecurityContextError("tool_denied")
    if not context.permits_tool(tool_name):
        raise SecurityContextError("tool_denied")
    capability = context._adapter_capability
    if not isinstance(capability, AdapterSecurityCapability):
        raise SecurityContextError("missing_adapter_security_capability")
    capability.revalidate_sync(context, tool_name)


def reapply_security_context_after_tool_refresh(agent: Any) -> None:
    """Re-filter a rebuilt tool snapshot before it is published to a turn."""
    context = getattr(agent, "_security_context", None)
    if context is not None:
        apply_security_context_to_agent(agent, context)
