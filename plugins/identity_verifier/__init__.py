"""Identity verifier provider discovery: bundled ``plugins/identity_verifier/<name>/`` then user
``$HERMES_HOME/plugins/<name>/`` (bundled wins on collision). Hosted boot resolves the configured
verifier here and refuses to serve without it (``hermes_cli.hosted_mode``); there is no local
backend, because a local install verifies nothing.

Providers load in-process only: under ``plugins.isolation: host`` a user provider is refused
(:class:`IdentityVerifierRefused`) rather than returned as a host proxy, whose checks would run
outside this process and whose verdicts could not be bound as this process's scope.

A user provider's module is named after its canonical directory, so same-named providers in two
profiles (and their sibling modules) never share a module or its import-time state.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from plugins import plugin_loader as _loader

if TYPE_CHECKING:
    from hermes_identity_verifier import IdentityVerifier

logger = logging.getLogger(__name__)


class IdentityVerifierRefused(RuntimeError):
    """An identity verifier provider that exists but may not be loaded here (the reason is the
    message)."""


_PLUGINS_DIR = Path(__file__).parent
# Synthetic parent package for user-installed providers (keeps them out of the bundled namespace).
_USER_NAMESPACE = "_hermes_user_identity_verifier"


def _is_provider_dir(path: Path) -> bool:
    """Cheap text heuristic: ``__init__.py`` mentions the identity verifier contract."""
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8-sig")[:8192]
    except OSError:
        return False
    return "register_identity_verifier" in source or "IdentityVerifier" in source


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory (bundled first, then user-installed)."""
    bundled = _PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    user_dir = _loader.user_plugins_dir()
    user = user_dir / name if user_dir else None
    return user if user and user.is_dir() and _is_provider_dir(user) else None


def load_identity_verifier(name: str) -> Optional[IdentityVerifier]:
    """Load a provider instance by name; ``None`` if not found or it fails to load. Raises
    :class:`IdentityVerifierRefused` for a user provider a hosted process (or one under
    ``plugins.isolation: host``) must not import — the reason is the message."""
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug("Identity verifier provider '%s' not found in bundled or user plugins", name)
        return None
    if not _is_bundled(provider_dir):
        from hermes_cli.plugin_isolation import in_process_import_refusal
        refusal = in_process_import_refusal(f"identity verifier provider {name!r}")
        if not refusal:
            # A hosted process resolves named providers only from the bundled tree; the shared
            # loader refuses a user dir before any import, and the boot reports the reason.
            refusal = _loader.hosted_user_dir_refusal(provider_dir, bundled_root=_PLUGINS_DIR)
        if refusal:
            logger.error("%s", refusal)
            raise IdentityVerifierRefused(refusal)
    return _loader.load_named(name, provider_dir, _load_provider_from_dir,
                              kind="Identity verifier provider", noun="provider", logger=logger,
                              bundled_root=_PLUGINS_DIR)


def _is_bundled(provider_dir: Path) -> bool:
    return provider_dir.parent == _PLUGINS_DIR


def _user_namespace(provider_dir: Path) -> str:
    """The synthetic package a user provider loads under: one per canonical provider directory,
    so profile A's ``<name>`` and profile B's ``<name>`` are different modules (a directory alias
    of the same profile resolves to the same one)."""
    digest = hashlib.sha256(str(provider_dir.resolve()).encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"{_USER_NAMESPACE}.p{digest}"


def _load_provider_from_dir(provider_dir: Path) -> Optional[IdentityVerifier]:
    """Import a provider module in-process and extract its provider (``register(ctx)`` or a
    subclass). ``load_plugin_module`` refuses a user module under host isolation as well."""
    from hermes_identity_verifier import IdentityVerifier
    name = provider_dir.name
    namespace = None if _is_bundled(provider_dir) else _user_namespace(provider_dir)
    module_name = f"plugins.identity_verifier.{name}" if namespace is None else f"{namespace}.{name}"
    if namespace is not None:
        _loader.register_synthetic_package(_USER_NAMESPACE, [])
    mod = _loader.load_plugin_module(
        module_name, provider_dir, parents=("plugins", "plugins.identity_verifier"), logger=logger,
        synthetic_namespace=namespace)
    return mod and _loader.instance_from_module(
        mod, collector=_ProviderCollector(), collected_attr="provider",
        base_cls=IdentityVerifier, name=name, logger=logger)


class _ProviderCollector(_loader.NoopPluginContext):
    """Fake plugin context that captures ``register_identity_verifier`` calls."""

    def __init__(self):
        self.provider = None

    def register_identity_verifier(self, provider):
        self.provider = provider
