"""Profile-scoped browser-vault login backends registered by trusted in-process plugins.

Registration (``ctx.register_login_backend``) is refused for: a name that is not a lowercase
identifier, a handle prefix other than ``<name>:``, a name or prefix in a built-in backend's namespace,
a name another plugin already holds in this profile, a backend that wants Hermes' master-password
prompt (``needs_unlock``), and ``name``/``prefix`` computed by a descriptor or an attribute-access hook
(they could change after validation, and reading them would run plugin code; :func:`static_name` judges
them before anything reads them). ``enabled_plugin_backends`` re-checks every stored key and wraps each backend in
:class:`agent.vault_backends.plugin.PluginLoginBackend`, the only form the vault tool ever sees.
"""

import inspect
import logging
import re

from agent.provider_registry import ProviderRegistry
from agent.vault_backends.base import LoginBackend

_registry = ProviderRegistry(label="Login backend", provider_cls=LoginBackend,
                             logger=logging.getLogger(__name__))
_registry.export(globals())

_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _builtin_namespaces() -> tuple[set[str], tuple[str, ...]]:
    from agent.vault_backends.base import external_backend_classes
    from agent.vault_backends.local import LocalLoginBackend

    classes = (LocalLoginBackend, *external_backend_classes())
    names = {cls.name for cls in classes} | {cls.prefix.rstrip(":_") for cls in classes}
    return names, tuple(cls.prefix for cls in classes)


def _namespace_error(name) -> str:
    """Why ``name`` cannot be a plugin backend's namespace, or "" when it can."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        return "login backend name must be a lowercase identifier"
    names, prefixes = _builtin_namespaces()
    if name in names or f"{name}:".startswith(prefixes):
        return "builtin login backend namespace is reserved"
    return ""


def static_name(provider) -> str:
    """The backend's name, judged without running plugin code (``ValueError`` with a fixed message when refused).

    Registration reads ``provider.name``; a property, a ``__getattribute__`` hook or a ``__class__`` override
    would run the plugin's code there, outside any guard, and its exception text would reach the loader log.
    """
    cls = type(provider)
    if not issubclass(cls, LoginBackend):
        raise ValueError("login backend must subclass LoginBackend")
    if cls.__getattribute__ is not object.__getattribute__ or any("__class__" in vars(k) for k in cls.__mro__[:-1]):
        raise ValueError("login backend must not override attribute access")
    name, prefix = inspect.getattr_static(provider, "name", None), inspect.getattr_static(provider, "prefix", None)
    if type(name) is not str or type(prefix) is not str:
        raise ValueError("login backend name and prefix must be plain string attributes")
    error = _namespace_error(name) or ("" if prefix == f"{name}:" else "login backend prefix must be '<name>:'")
    if error:
        raise ValueError(error)
    if inspect.getattr_static(provider, "needs_unlock", False) is not False:
        raise ValueError("plugin login backends cannot use the Hermes master-password prompt (needs_unlock)")
    return name


def register_provider(provider, *, scope=None):
    name = static_name(provider)
    if name in _registry.merged(scope):
        raise ValueError(f"login backend name {name!r} is already registered")
    _registry.register(provider, scope=scope)


def enabled_plugin_backends(cfg: dict) -> list[LoginBackend]:
    """Registered backends the profile turned on with ``vault.<name>.enabled: true``, wrapped."""
    from agent.vault_backends.plugin import PluginLoginBackend

    out: list[LoginBackend] = []
    for name, provider in sorted(_registry.merged().items()):
        section = cfg.get(name)
        if not _namespace_error(name) and isinstance(section, dict) and section.get("enabled") is True:
            out.append(PluginLoginBackend(name, provider))
    return out
