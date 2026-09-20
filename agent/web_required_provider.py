"""Fail-closed, profile-local mandatory mediation for native web tools.

This is request routing, not a sandbox or authentication scheme. Trusted startup
must supply the policy and plugin. Binding survives plugin unload; a config or
instance change requires a new process, never a silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading

from agent.web_search_provider import WebSearchProvider
from hermes_constants import hermes_home_key


class RequiredWebProviderError(RuntimeError):
    """No native web dispatch may continue after a mandatory-policy failure."""


@dataclass
class _Binding:
    name: str
    provider: WebSearchProvider | None = None


_bindings: dict[str, _Binding] = {}
_lock = threading.RLock()
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


def required_provider_name() -> str | None:
    """Read effective policy without permissive recovery and pin it once seen.

    An absent setting on a never-managed profile keeps legacy behavior. Removing
    it after binding is an error, including removal of the whole config file.
    Binding metadata contains only a name/reference, not request content.
    """
    from hermes_cli.config_effective import load_user_config_effective

    try:
        config = load_user_config_effective(fail_closed=True)
    except Exception as exc:
        raise RequiredWebProviderError(
            "required_web_provider: cannot read effective policy"
        ) from exc
    web = config.get("web")
    if web is None:
        web = {}
    if not isinstance(web, dict):
        raise RequiredWebProviderError(
            "required_web_provider: web policy must be a mapping"
        )
    name = web.get("required_provider")
    scope = hermes_home_key()
    with _lock:
        bound = _bindings.get(scope)
        if bound is not None and name != bound.name:
            raise RequiredWebProviderError(
                "required_web_provider: policy changed; restart required"
            )
        if name is None:
            return None
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise RequiredWebProviderError(
                "required_web_provider: invalid provider name"
            )
        _bindings.setdefault(scope, _Binding(name))
    if any(
        web.get(key) not in (None, "", name)
        for key in ("backend", "search_backend", "extract_backend")
    ):
        raise RequiredWebProviderError(
            "required_web_provider: conflicting backend selection"
        )
    return name


def get_required_provider(capability: str) -> WebSearchProvider | None:
    """Resolve the exact profile slot, never a global/cross-profile fallback.

    In-flight calls already handed to a provider belong to that provider; unload
    is not cancellation or proof that their external effect did not occur.
    """
    name = required_provider_name()
    if name is None:
        return None
    from agent.web_search_registry import snapshot_registration

    scope = hermes_home_key()
    provider = snapshot_registration(name, scope=scope)
    if provider is None:
        raise RequiredWebProviderError(
            "required_web_provider: configured plugin is missing or unloaded"
        )
    with _lock:
        bound = _bindings[scope]
        if bound.provider is not None and provider is not bound.provider:
            raise RequiredWebProviderError(
                "required_web_provider: provider replaced; restart required"
            )
        bound.provider = provider
    try:
        version = provider.request_policy_version
        if type(version) is not int or version != 1:
            raise RequiredWebProviderError(
                "required_web_provider: unsupported request policy version"
            )
        if (
            capability not in ("search", "extract")
            or getattr(provider, f"supports_{capability}")() is not True
        ):
            raise RequiredWebProviderError(
                "required_web_provider: unsupported capability"
            )
        if provider.is_available() is not True:
            raise RequiredWebProviderError("required_web_provider: service unavailable")
    except RequiredWebProviderError:
        raise
    except Exception as exc:
        raise RequiredWebProviderError(
            "required_web_provider: provider readiness failed"
        ) from exc
    return provider


def requires_direct_dispatch(provider: WebSearchProvider, capability: str) -> bool:
    """Whether to call the mediator without native cache, rescue, or bucketing."""
    required = get_required_provider(capability)
    if required is None:
        return False
    if required is not provider:
        raise RequiredWebProviderError(
            "required_web_provider: attempted alternate provider"
        )
    return True
