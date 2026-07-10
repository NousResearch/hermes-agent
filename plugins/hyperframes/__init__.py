"""Hermes plugin bridge for heygen-com/hyperframes."""

from __future__ import annotations

from . import core
from .cli import hyperframes_command, register_cli


_ALWAYS_AVAILABLE = {
    "hyperframes_status",
    "hyperframes_setup",
    "hyperframes_install",
}

_TOOLS = (
    ("hyperframes_status", core.STATUS_SCHEMA, core.handle_status, "🎬"),
    ("hyperframes_setup", core.SETUP_SCHEMA, core.handle_setup, "🛠️"),
    ("hyperframes_install", core.INSTALL_SCHEMA, core.handle_install, "📦"),
    ("hyperframes_init", core.INIT_SCHEMA, core.handle_init, "🆕"),
    ("hyperframes_validate", core.VALIDATE_SCHEMA, core.handle_validate, "✅"),
    ("hyperframes_render", core.RENDER_SCHEMA, core.handle_render, "🎞️"),
    ("hyperframes_preview", core.PREVIEW_SCHEMA, core.handle_preview, "👁️"),
    ("hyperframes_capture", core.CAPTURE_SCHEMA, core.handle_capture, "🌐"),
    ("hyperframes_audio", core.AUDIO_SCHEMA, core.handle_audio, "🔊"),
)


def register(ctx) -> None:
    """Register HyperFrames tools, slash command, and CLI."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset=core.TOOLSET,
            schema=schema,
            handler=handler,
            check_fn=(lambda: True) if name in _ALWAYS_AVAILABLE else core.check_available,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "hyperframes",
        handler=core.handle_slash,
        description="HyperFrames HTML-to-video status and quick commands.",
        args_hint="[status|install|init <name>|lint <dir>|render <dir>]",
    )
    ctx.register_cli_command(
        name="hyperframes",
        help="Install and operate heygen-com/hyperframes HTML video projects",
        setup_fn=register_cli,
        handler_fn=hyperframes_command,
        description=(
            "Link the bundled hyperframes skill, install the npm hyperframes CLI, "
            "scaffold HTML video projects, lint/validate/inspect, preview, render, "
            "capture websites, and run TTS/transcribe helpers."
        ),
    )

    if _auto_install_enabled():

        def _on_session_start(**_: object) -> None:
            payload = core.status()
            env = payload.get("environment") or {}
            skill = payload.get("hermes_skill") or {}
            if env.get("ready") and skill.get("present"):
                return
            core.install(skip_vendor=True, auto_prereqs=True)

        ctx.register_hook("on_session_start", _on_session_start)


def _auto_install_enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        plugins = cfg.get("plugins", {}) if isinstance(cfg, dict) else {}
        entries = plugins.get("entries", {}) if isinstance(plugins, dict) else {}
        for key in core.CONFIG_ALIASES:
            entry = entries.get(key)
            if isinstance(entry, dict) and entry.get("auto_install") is False:
                return False
        return True
    except Exception:
        return True
