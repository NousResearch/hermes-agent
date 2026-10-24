"""Profile-scoped, leased browser-vault backends for trusted in-process plugins."""
import logging
import re
from agent.provider_registry import ProviderRegistry
from agent.vault_backends.base import LoginBackend

_registry = ProviderRegistry(label="Login backend", provider_cls=LoginBackend,
                             logger=logging.getLogger(__name__))
_registry.export(globals())


def register_provider(provider, *, scope=None):
    if not isinstance(provider, LoginBackend):
        raise ValueError("login backend must subclass LoginBackend")
    name, prefix = getattr(provider, "name", ""), getattr(provider, "prefix", "")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("invalid login backend name")
    if not isinstance(prefix, str) or not re.fullmatch(r"[a-z][a-z0-9_]*:", prefix):
        raise ValueError("invalid login backend prefix")
    if name in {"local", "onepassword", "bitwarden"} or prefix in {"op:", "bw:"}:
        raise ValueError("builtin login backend namespace is reserved")
    for existing in _registry.list_providers(scope=scope):
        if existing.name == name and existing.prefix != prefix:
            raise ValueError("login backend prefix cannot change during replacement")
        if existing.name != name and existing.prefix == prefix:
            raise ValueError("login backend prefix is already registered")
    _registry.register(provider, scope=scope)
