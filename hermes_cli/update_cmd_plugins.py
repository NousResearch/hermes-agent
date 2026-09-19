"""Post-``hermes update`` refresh of plugin-declared ``pip_dependencies``.

``hermes update`` re-applies the core install, the active lazy backends, and the active memory
provider's dependencies — but a web provider plugin's declared pip dependency was applied once, by
the ``hermes tools`` post_setup hook, which installs only when the package is missing. Nothing
checked it afterwards, so an install kept whatever version landed first, forever, with no signal that
a newer one existed. See #108711.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("hermes_cli.update_cmd")

# One resolver pass per active provider, inside the update the user already waits on.
_INSTALL_TIMEOUT_SECS = 240

# A provider is selected via ``web.<capability>_backend`` or the shared ``web.backend``.
_WEB_BACKEND_KEYS = ("search_backend", "extract_backend", "backend")

# Failure reasons are rendered on one line of update output.
_REASON_LIMIT = 240


def _normalize_provider_name(name: str) -> str:
    """``brave-free`` and ``brave_free`` name the same provider (see ``agent.web_search_registry``)."""
    return name.strip().lower().replace("-", "_")


@dataclass(frozen=True)
class WebProviderPlugin:
    """A web provider plugin manifest that declares pip dependencies."""

    label: str  # manifest key, e.g. "web/ddgs"
    providers: tuple[str, ...]  # provides_web_providers entries
    dependencies: tuple[str, ...]  # pip_dependencies specs

    def is_selected(self, configured: set[str]) -> bool:
        """True when config names one of this plugin's providers as a web backend."""
        return any(_normalize_provider_name(name) in configured for name in self.providers)


def _manifest_data(plugin_dir: Path) -> dict:
    """Parsed ``plugin.yaml`` for a discovered plugin dir; ``{}`` when absent or unreadable."""
    try:
        import yaml
    except Exception:  # pragma: no cover — yaml ships with the CLI extras
        return {}
    for filename in ("plugin.yaml", "plugin.yml"):
        path = plugin_dir / filename
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.debug("Could not read %s: %s", path, exc)
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _declared_specs(raw) -> tuple[str, ...]:
    """Normalized ``pip_dependencies`` entries; blank and non-string values are dropped."""
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(spec for spec in (str(item).strip() for item in raw) if spec)


def _web_provider_plugin_dependencies() -> tuple[WebProviderPlugin, ...]:
    """``provides_web_providers`` plugins that declare ``pip_dependencies``, in discovery order.

    Manifests come from the loader's own import-free sweep — never by importing plugin code, since
    several providers pull heavy SDKs and an update must not run them.
    """
    from hermes_cli.plugins_discovery import collect_directory_manifests
    from hermes_cli.plugins_manifest import manifest_key

    found: list[WebProviderPlugin] = []
    for manifest in collect_directory_manifests():
        data = _manifest_data(Path(manifest.path))
        specs = _declared_specs(data.get("pip_dependencies"))
        raw_providers = data.get("provides_web_providers")
        if not specs or not isinstance(raw_providers, (list, tuple)):
            continue
        providers = tuple(name for name in (str(entry).strip() for entry in raw_providers) if name)
        if providers:
            found.append(WebProviderPlugin(manifest_key(manifest), providers, specs))
    return tuple(found)


def _configured_web_backends() -> set[str]:
    """``web.search_backend`` / ``web.extract_backend`` / ``web.backend``, normalized.

    Empty when config is unreadable — then only already-installed plugins are refreshed, never
    activated.
    """
    try:
        from hermes_cli.config import load_config_readonly

        web = (load_config_readonly() or {}).get("web")
    except Exception as exc:
        logger.debug("Could not read web backend config: %s", exc)
        return set()
    if not isinstance(web, dict):
        return set()
    configured = set()
    for key in _WEB_BACKEND_KEYS:
        value = web.get(key)
        if isinstance(value, str) and value.strip():
            configured.add(_normalize_provider_name(value))
    return configured


def _active_web_provider_plugins(
    plugins: tuple[WebProviderPlugin, ...], configured: set[str]
) -> tuple[WebProviderPlugin, ...]:
    """The plugins this update must keep current: selected in config, or already installed.

    An installed dependency is kept current because that package is what makes the provider
    resolvable at all (a user who never set ``web.backend`` is routed to it by the availability
    walk). A plugin the user never selected and never installed is skipped on purpose — installing
    its package would add it to that walk and could flip which provider
    ``agent.web_search_registry`` resolves.
    """
    from tools import lazy_deps

    return tuple(
        plugin for plugin in plugins
        if plugin.is_selected(configured) or any(lazy_deps.spec_installed(spec) for spec in plugin.dependencies)
    )


def _refresh_active_web_provider_dependencies() -> dict[str, str]:
    """Re-apply declared ``pip_dependencies`` for web providers that are in use.

    Returns ``{plugin label: "refreshed" | "skipped: <reason>" | "failed: <reason>"}`` (empty when
    nothing is in use), mirroring the lazy-backend refresh statuses. Never raises: a failed optional
    dependency must not abort the update.
    """
    try:
        plugins = _web_provider_plugin_dependencies()
    except Exception as exc:
        logger.debug("Web provider dependency refresh skipped (discovery failed): %s", exc)
        return {}
    active = _active_web_provider_plugins(plugins, _configured_web_backends())
    if not active:
        return {}

    from tools import lazy_deps

    print()
    print(f"→ Refreshing {len(active)} active web provider dependency set(s)...")
    results: dict[str, str] = {}
    for plugin in active:
        specs = list(plugin.dependencies)
        try:
            outcome = lazy_deps.install_specs(specs, upgrade=True, timeout=_INSTALL_TIMEOUT_SECS)
        except Exception as exc:  # install_specs is never-raise by contract; defend anyway
            results[plugin.label] = f"failed: {exc}"
            print(f"  ⚠ {plugin.label} dependencies failed to refresh: {str(exc)[:_REASON_LIMIT]}")
            continue
        if outcome.ok:
            results[plugin.label] = "refreshed"
            print(f"  ↑ {plugin.label}: {', '.join(specs)}")
        elif outcome.blocked:
            # Usually security.allow_lazy_installs=false; informational, not an error.
            results[plugin.label] = f"skipped: {outcome.reason}"
            print(f"  · {plugin.label} skipped ({outcome.reason}): {', '.join(specs)}")
        else:
            reason = ((outcome.stderr or outcome.stdout).strip() or "install error")[:_REASON_LIMIT]
            results[plugin.label] = f"failed: {reason}"
            print(f"  ⚠ {plugin.label} dependencies failed to refresh: {reason}")
    return results
