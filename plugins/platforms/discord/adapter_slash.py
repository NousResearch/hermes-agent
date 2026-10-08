# fmt: off
"""Discord adapter support: the native slash-command registry.

Kept out of ``adapter.py`` so the platform facade cannot outgrow its
code-health cap (``scripts/code_health/config.py``); ``adapter.py`` imports
from here. Moved verbatim from the adapter -- zero behavior change.
"""

from __future__ import annotations

from typing import Any

from agent.i18n import t
from gateway.platforms.base import _prefix_within_utf16_limit


_REQUIRED = object()
# Text slots hold ``platform.discord.command.*`` / ``slash.*`` catalog keys; ``_native_slash_commands()``
# resolves them for the active language (a language change re-syncs: the fingerprint carries it).
_NATIVE_SLASH_COMMAND_SPECS: tuple = (
    ("new", "platform.discord.command.new.description", (), "/reset", "platform.discord.command.new.followup"),
    ("reset", "platform.discord.command.reset.description", (), "/reset", "platform.discord.command.reset.followup"),
    ("model", "platform.discord.command.model.description",
     (("name", str, "", "platform.discord.command.model.arg_name", None),),
     "/model {name}", None),
    ("reasoning", "platform.discord.command.reasoning.description",
     (("effort", str, "", "platform.discord.command.reasoning.arg_effort",
       # One `/reasoning <arg>` handler; Discord has no free-text subcommand, so list every value.
       # Choice labels are (key-or-literal, value); bare level names are identifiers, not prose.
       (("platform.discord.command.reasoning.choice_none", "none"), ("minimal", "minimal"), ("low", "low"),
        ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"), ("max", "max"),
        ("platform.discord.command.reasoning.choice_ultra", "ultra"), ("platform.discord.command.reasoning.choice_reset", "reset"),
        ("platform.discord.command.reasoning.choice_show", "show"), ("platform.discord.command.reasoning.choice_hide", "hide"))),),
     "/reasoning {effort}", None),
    ("personality", "platform.discord.command.personality.description",
     (("name", str, "", "platform.discord.command.personality.arg_name", None),),
     "/personality {name}", None),
    ("retry", "platform.discord.command.retry.description", (), "/retry", "platform.discord.command.retry.followup"),
    ("undo", "platform.discord.command.undo.description", (), "/undo", None),
    ("status", "platform.discord.command.status.description", (), "/status", "platform.discord.command.status.followup"),
    ("sethome", "slash.sethome.description", (), "/sethome", None),
    ("stop", "platform.discord.command.stop.description", (), "/stop", "platform.discord.command.stop.followup"),
    ("steer", "platform.discord.command.steer.description",
     (("prompt", str, _REQUIRED, "platform.discord.command.steer.arg_prompt", None),),
     "/steer {prompt}", None),
    ("plan", "platform.discord.command.plan.description",
     (("task", str, "", "platform.discord.command.plan.arg_task", None),),
     "/plan {task}", None),
    ("compress", "platform.discord.command.compress.description", (), "/compress", None),
    ("title", "platform.discord.command.title.description",
     (("name", str, "", "platform.discord.command.title.arg_name", None),),
     "/title {name}", None),
    ("resume", "slash.resume.description",
     (("name", str, "", "platform.discord.command.resume.arg_name", None),),
     "/resume {name}", None),
    ("usage", "platform.discord.command.usage.description", (), "/usage", None),
    ("help", "platform.discord.command.help.description", (), "/help", None),
    ("insights", "slash.insights.description",
     (("days", int, 7, "platform.discord.command.insights.arg_days", None),),
     "/insights {days}", None),
    ("reload-mcp", "slash.reload_mcp.description", (), "/reload-mcp", None),
    ("reload-skills", "platform.discord.command.reload_skills.description", (), "/reload-skills", None),
    ("voice", "platform.discord.command.voice.description",
     (("mode", str, "", "platform.discord.command.voice.arg_mode",
       # `join` and `channel` both hit _handle_voice_channel_join; expose both to match docs.
       (("platform.discord.command.voice.choice_join", "join"), ("platform.discord.command.voice.choice_channel", "channel"),
        ("platform.discord.command.voice.choice_leave", "leave"), ("platform.discord.command.voice.choice_mode_on", "on"),
        ("platform.discord.command.voice.choice_tts", "tts"), ("platform.discord.command.voice.choice_mode_off", "off"),
        ("platform.discord.command.voice.choice_status", "status"))),),
     "/voice {mode}", None),
    ("update", "slash.update.description", (), "/update", "platform.discord.command.update.followup"),
    ("restart", "platform.discord.command.restart.description", (), "/restart", "platform.discord.command.restart.followup"),
    ("approve", "slash.approve.description",
     (("scope", str, "", "platform.discord.command.approve.arg_scope", None),),
     "/approve {scope}", None),
    ("deny", "platform.discord.command.deny.description",
     (("scope", str, "", "platform.discord.command.deny.arg_scope", None),),
     "/deny {scope}", None),
    # /thread: template None -> registered by _register_thread_slash (auth-gated defer).
    ("thread", "platform.discord.command.thread.description", (), None, None),
    ("queue", "platform.discord.command.queue.description",
     (("prompt", str, _REQUIRED, "platform.discord.command.queue.arg_prompt", None),),
     "/queue {prompt}", "platform.discord.command.queue.followup"),
    ("bg", "slash.bg.description",
     (("prompt", str, _REQUIRED, "platform.discord.command.bg.arg_prompt", None),),
     "/bg {prompt}", "platform.discord.command.bg.followup"),
    ("btw", "platform.discord.command.btw.description",
     (("question", str, _REQUIRED, "platform.discord.command.btw.arg_question", None),),
     "/btw {question}", "platform.discord.command.btw.followup"),
)
# Discord rejects the whole bulk sync (error 50035) when ONE description / parameter description /
# Choice name exceeds 100 UTF-16 units, so every localized slot is cut at the cap.
_DISCORD_APP_COMMAND_TEXT_LIMIT = 100


def _t_discord(key: str, limit: int, **kwargs: Any) -> str:
    """``t()`` cut to a Discord field cap (UTF-16 units)."""
    return _truncate_discord_component_text(t(key, **kwargs), limit)



def _native_slash_commands() -> tuple:
    """``_NATIVE_SLASH_COMMAND_SPECS`` with descriptions, parameter descriptions and Choice names
    resolved for the active language: ``(name, description, args, template, followup_key)``.
    Follow-ups stay KEYS — ``_run_simple_slash`` resolves them when the command actually runs."""
    def _text(key: str) -> str:
        return _t_discord(key, _DISCORD_APP_COMMAND_TEXT_LIMIT)

    def _choice_label(label: str) -> str:
        return _text(label) if "." in label else label

    out = []
    for name, description_key, args, template, followup_key in _NATIVE_SLASH_COMMAND_SPECS:
        localized_args = tuple(
            (arg_name, arg_type, default, _text(desc_key),
             tuple((_choice_label(lbl), val) for lbl, val in choices) if choices else None)
            for arg_name, arg_type, default, desc_key, choices in args)
        out.append((name, _text(description_key), localized_args, template, followup_key))
    return tuple(out)




def _truncate_discord_component_text(text: str, limit: int) -> str:
    """Return text within Discord's UTF-16 component field budget."""
    return _prefix_within_utf16_limit(str(text or ""), max(0, limit))


# fmt: on
