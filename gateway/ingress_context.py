"""Turn-local live ingress adapter binding for plugin platform actions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional
import weakref


@dataclass(frozen=True)
class IngressAdapterBinding:
    adapter_ref: weakref.ReferenceType
    platform: str
    transport_profile: Optional[str]

    def adapter(self) -> Any:
        return self.adapter_ref()


_CURRENT_INGRESS: ContextVar[Optional[IngressAdapterBinding]] = ContextVar(
    "gateway_ingress_adapter", default=None
)


def current_ingress_binding(platform: str) -> Optional[IngressAdapterBinding]:
    """Return the ingress binding only when its platform matches exactly."""
    binding = _CURRENT_INGRESS.get()
    if binding is None or binding.platform != str(platform).strip().lower():
        return None
    return binding


def current_ingress_adapter(platform: str) -> Any:
    """Return the bound receiving adapter only when its platform matches exactly."""
    binding = current_ingress_binding(platform)
    return binding.adapter() if binding is not None else None


@contextmanager
def bind_ingress_adapter(source: Any) -> Iterator[Optional[IngressAdapterBinding]]:
    """Bind the live adapter that received *source* for this context and its async children."""
    from gateway.session_identity import identity_of

    identity = identity_of(source)
    adapter = identity.adapter() if identity is not None else None
    if adapter is None:
        ref = getattr(source, "_transport_adapter_ref", None)
        adapter = ref() if callable(ref) else None
    platform = getattr(
        getattr(source, "platform", None), "value", getattr(source, "platform", "")
    )
    binding = None
    if adapter is not None and platform:
        binding = IngressAdapterBinding(
            adapter_ref=weakref.ref(adapter),
            platform=str(platform).strip().lower(),
            transport_profile=getattr(identity, "transport_profile", None),
        )
    token = _CURRENT_INGRESS.set(binding)
    try:
        yield binding
    finally:
        _CURRENT_INGRESS.reset(token)
