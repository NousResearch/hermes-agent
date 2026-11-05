"""The ``register_<category>_provider`` table behind :class:`hermes_cli.plugins.PluginContext`.

Every ``register_<category>_provider`` shares one body (:meth:`PluginContext._register_scoped_provider`):
type-check, register in the scope-keyed process-global registry, lease the slot so unload restores
the displaced entry. ``hermes_cli.plugins`` generates one ctx method per row. Rows: (method, kind,
registry module, base-class module:attr, label, docstring, options). ``normalize``: ``strip``
(default), ``lower`` (strip+lowercase) or ``None`` (raw name). ``admit``: ``module:function`` taking
the plugin's manifest and returning why it may not register this kind ("" = it may), judged under the
manager's home before the registry is touched. ``identity``: a registry function judging the provider's
name without running its code, before the shared body reads ``provider.name`` (``ValueError`` = refused,
logged with ``reject_message``). Object-taking rows also need a
``plugin_isolation.HOST_OBJECT_BASES`` entry so the provider works under ``plugins.isolation: host``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional

from hermes_cli.plugins_loader import _plugin_home_scope, _serialized_replacement

logger = logging.getLogger(__name__)

_SCOPED_PROVIDER_REGISTRARS: tuple[tuple[str, str, str, str, str, str, dict[str, Any]], ...] = (
    ("register_login_backend", "login_backend", "agent.vault_backend_registry",
     "agent.vault_backends.base:LoginBackend", "login backend",
     "Register a trusted in-process browser-vault LoginBackend; explicit vault.<name>.enabled "
     "is required. Registration is profile-scoped and removed on plugin unload. "
     "Backend instances must read active-profile configuration lazily. Accepted only from bundled "
     "plugins and catalog installs at tier verified or official.",
     {"normalize": None, "reject_message": "Plugin '%s' login backend rejected: %s",
      "admit": "hermes_cli.plugins_cmd_catalog:login_backend_provenance_error", "identity": "static_name"}),
    ("register_browser_page_backend", "browser_page_backend", "agent.browser_page_registry",
     "agent.browser_page_backend:BrowserPageBackend", "browser page backend",
     "Register an :class:`agent.browser_page_backend.BrowserPageBackend` so ``browser_vault`` inspects and "
     "fills the pages of a browser with no CDP endpoint (Camofox). Profile-scoped, removed on plugin "
     "unload; accepted only from bundled plugins and catalog installs at tier verified or official.",
     {"reject_message": "Plugin '%s' browser page backend rejected: %s",
      "admit": "hermes_cli.plugins_cmd_catalog:browser_page_backend_provenance_error"}),
    ("register_image_gen_provider", "image_gen_provider", "agent.image_gen_registry",
     "agent.image_gen_provider:ImageGenProvider", "image_gen provider",
     "Register an :class:`agent.image_gen_provider.ImageGenProvider`; "
     "``provider.name`` is matched by ``image_gen.provider``.", {"article": "an"}),
    ("register_video_gen_provider", "video_gen_provider", "agent.video_gen_registry",
     "agent.video_gen_provider:VideoGenProvider", "video_gen provider",
     "Register an :class:`agent.video_gen_provider.VideoGenProvider`; "
     "``provider.name`` is matched by ``video_gen.provider``.", {}),
    ("register_web_search_provider", "web_search_provider", "agent.web_search_registry",
     "agent.web_search_provider:WebSearchProvider", "web provider",
     "Register an :class:`agent.web_search_provider.WebSearchProvider`; "
     "``provider.name`` is matched by ``web.search_backend`` / ``web.extract_backend`` / ``web.backend``.",
     {}),
    ("register_browser_provider", "browser_provider", "agent.browser_registry",
     "agent.browser_provider:BrowserProvider", "browser provider",
     "Register an :class:`agent.browser_provider.BrowserProvider`; "
     "``provider.name`` is matched by ``browser.cloud_provider`` (consulted by "
     "``tools.browser_session_cloud.get_cloud_provider``).", {}),
    ("register_terminal_environment_provider", "terminal_environment_provider",
     "agent.terminal_env_registry", "agent.terminal_env_provider:TerminalEnvironmentProvider",
     "terminal environment provider",
     "Register a :class:`agent.terminal_env_provider.TerminalEnvironmentProvider`; ``provider.name`` "
     "is matched by ``terminal.backend`` when no built-in backend has that name. Built-in names (local, "
     "docker, modal, daytona, vercel_sandbox, ssh) are rejected — plugins never shadow "
     "in-tree backends.",
     {"normalize": "lower", "reject_message": "Plugin '%s' terminal environment provider rejected: %s"}),
    ("register_secret_source", "secret_source", "agent.secret_sources.registry",
     "agent.secret_sources.base:SecretSource", "secret source",
     "Register a :class:`agent.secret_sources.base.SecretSource`, run by ``load_hermes_dotenv()`` "
     "(after ``~/.hermes/.env``, before credentials are read) when ``secrets.<name>`` is enabled. The "
     "orchestrator owns ordering/precedence/provenance; the source only fetches. Since dotenv usually "
     "loads before discovery, the manager re-pulls enabled plugin sources afterwards.",
     {"normalize": None, "register": "register_source", "param": "source"}),
    ("register_tts_provider", "tts_provider", "agent.tts_registry",
     "agent.tts_provider:TTSProvider", "TTS provider",
     "Register an :class:`agent.tts_provider.TTSProvider`; ``provider.name`` is matched by "
     "``tts.provider`` unless it is a built-in name (rejected with a warning) or a "
     "``tts.providers.<name>: type: command`` entry shares it (command-providers win).",
     {"normalize": "lower"}),
    ("register_transcription_provider", "transcription_provider", "agent.transcription_registry",
     "agent.transcription_provider:TranscriptionProvider", "transcription provider",
     "Register an :class:`agent.transcription_provider.TranscriptionProvider`; ``provider.name`` is "
     "matched by ``stt.provider`` unless it is a built-in name (rejected) or a ``stt.providers.<name>: "
     "type: command`` entry shares it (command-providers win).", {"normalize": "lower"}),
    ("register_delegation_provider", "delegation_provider", "agent.delegation_registry",
     "agent.delegation_provider:DelegationProvider", "delegation provider",
     "Register an :class:`agent.delegation_provider.DelegationProvider`; ``provider.name`` is matched by "
     "``delegation.backend``. It decides each ``delegate_task`` task's route (model/provider or a Hermes "
     "profile) and may add per-task schema parameters; core keeps the subagent lifecycle.",
     {"normalize": "lower"}),
)

_NAME_NORMALIZERS: dict[Optional[str], Optional[Callable[[str], str]]] = {
    "strip": lambda n: n.strip(), "lower": lambda n: n.strip().lower(), None: None,
}


def _resolve(ref: str) -> Any:
    module, attr = ref.split(":")
    return getattr(importlib.import_module(module), attr)


def make_scoped_provider_registrar(method_name, kind, registry_mod, base_ref, label, doc, options):
    """Build one ``register_<category>_provider`` method from a ``_SCOPED_PROVIDER_REGISTRARS`` row."""
    normalize_fn = _NAME_NORMALIZERS[options.get("normalize", "strip")]
    register_name = options.get("register")
    admit, identity = options.get("admit"), options.get("identity")

    def register(self, provider):
        if admit:
            with _plugin_home_scope(self._manager.home_path):
                refusal = _resolve(admit)(self.manifest)
            if refusal:
                logger.warning(options["reject_message"], self.manifest.name, refusal)
                return None
        registry = importlib.import_module(registry_mod)
        if identity:
            try:
                getattr(registry, identity)(provider)
            except ValueError as exc:
                logger.warning(options["reject_message"], self.manifest.name, exc)
                return None
        return self._register_scoped_provider(
            provider, kind=kind, base_class=_resolve(base_ref),
            registry=registry, label=label, article=options.get("article", "a"), normalize=normalize_fn,
            register=getattr(registry, register_name) if register_name else None,
            reject_message=options.get("reject_message"),
        )

    def register_source(self, source):  # secret sources: ``source``
        return register(self, source)

    method = register_source if options.get("param") == "source" else register
    method.__name__, method.__qualname__, method.__doc__ = method_name, f"PluginContext.{method_name}", doc
    return _serialized_replacement(method)
