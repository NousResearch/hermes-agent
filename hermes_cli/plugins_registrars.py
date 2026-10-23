"""The ``register_<category>_provider`` table behind :class:`hermes_cli.plugins.PluginContext`.

Every ``register_<category>_provider`` shares one body (:meth:`PluginContext._register_scoped_provider`):
type-check, register in the scope-keyed process-global registry, lease the slot so unload restores
the displaced entry. ``hermes_cli.plugins`` generates one ctx method per row. Rows: (method, kind,
registry module, base-class module:attr, label, docstring, options). ``normalize``: ``strip``
(default), ``lower`` (strip+lowercase) or ``None`` (raw name). Object-taking rows also need a
``plugin_isolation.HOST_OBJECT_BASES`` entry so the provider works under ``plugins.isolation: host``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

_SCOPED_PROVIDER_REGISTRARS: tuple[tuple[str, str, str, str, str, str, dict[str, Any]], ...] = (
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
