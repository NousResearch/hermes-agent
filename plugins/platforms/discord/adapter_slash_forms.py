"""Slash commands whose grammar is split into separate Discord fields (``/thread``, ``/loop``).

Discord ``/loop`` as separate slash-command fields instead of one ``args`` string.

The generic auto-registered ``/loop`` offered a single ``args`` box whose hint was the whole CLI
grammar (``[interval] <prompt> [--times N] [--until <condition>] | status | ...``). Here each part of
that grammar is its own Discord option, and the options are rendered back into the exact text
``hermes_cli.loops.dispatch_loop_command`` already parses, so the loop core is unchanged.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from agent.i18n import t

try:
    import discord
except ImportError:  # the adapter does not start without discord.py
    discord = None

# (name, type, default, description catalog key, choices) — the ``_NATIVE_SLASH_COMMAND_SPECS`` shape.
_KEY = "platform.discord.command.loop."
_FLAG_RE = re.compile(r"\s--(?:times|until)(?:\s|$)")
LOOP_SLASH_ARGS: tuple = (
    ("prompt", str, "", _KEY + "arg_prompt", None),
    ("every", str, "", _KEY + "arg_every", None),
    ("times", int, 0, _KEY + "arg_times", None),
    ("until", str, "", _KEY + "arg_until", None),
    ("action", str, "", _KEY + "arg_action",
     # Bare labels are the control words themselves (identifiers, like /reasoning's "low").
     (("status", "status"), ("pause", "pause"), ("resume", "resume"), ("stop", "stop"), ("help", "help"))),
)


class LoopSlashInputError(ValueError):
    """A field value the loop parser would silently misread; carries the catalog key to show."""

    def __init__(self, key: str, **kwargs: Any):
        super().__init__(key)
        self.key, self.kwargs = key, kwargs


def render_loop_command(prompt: str = "", every: str = "", times: int = 0, until: str = "",
                        action: str = "") -> str:
    """Fields -> ``/loop ...`` text. ``action`` wins; an empty form is ``/loop`` (status)."""
    from hermes_cli.loops import parse_interval_token

    if action:
        if prompt.strip() or every.strip() or times or until.strip():
            raise LoopSlashInputError(_KEY + "error_action_with_fields")
        return f"/loop {action}"
    prompt, every, until = prompt.strip(), every.strip(), until.strip()
    if not prompt:
        if every or times or until:
            raise LoopSlashInputError(_KEY + "error_missing_prompt")
        return "/loop"
    # A bare "5" is not an interval to the parser; it would quietly become part of the prompt
    # and the loop would run self-paced. Refuse it here, where the field is still separate.
    if every and parse_interval_token(every) is None:
        raise LoopSlashInputError(_KEY + "error_bad_interval", every=every)
    # The parser reads " --times"/" --until" anywhere in the line, so the same words inside a field
    # would be re-read as a flag (or split the prompt). Refuse rather than misparse.
    for value in (prompt, until):
        if _FLAG_RE.search(" " + value):
            raise LoopSlashInputError(_KEY + "error_flag_in_text")
    parts = ["/loop"] + ([every] if every else []) + [prompt]
    if times:
        parts.append(f"--times {times}")
    if until:
        parts.append(f"--until {until}")  # last: --until consumes to end of line
    return " ".join(parts)


class DiscordSlashFormsMixin:
    _check_slash_authorization: Any
    _handle_thread_create_slash: Any

    async def _render_slash_text(self, interaction: Any, name: str,
                                 template: "str | Callable[..., str]", kwargs: dict) -> Optional[str]:
        """A slash proxy's command text, or None after a private hint for an unreadable field form.
        The hint goes only to an authorized user, so the form never answers anyone the gate refuses."""
        if not callable(template):
            return template.format(**kwargs)
        try:
            return template(**kwargs)
        except LoopSlashInputError as exc:
            if await self._check_slash_authorization(interaction, f"/{name}"):
                await interaction.response.send_message(t(exc.key, **exc.kwargs), ephemeral=True)
            return None

    def _register_thread_slash(self, tree, name: str, description: str) -> None:
        from plugins.platforms.discord.adapter import _DISCORD_APP_COMMAND_TEXT_LIMIT as _LIMIT, _t_discord as _t

        @tree.command(name=name, description=description)
        @discord.app_commands.describe(
            name=_t("platform.discord.command.thread.arg_name", _LIMIT),
            message=_t("platform.discord.command.thread.arg_message", _LIMIT),
            auto_archive_duration=_t("platform.discord.command.thread.arg_auto_archive", _LIMIT),
        )
        async def slash_thread(
            interaction: discord.Interaction, name: str, message: str = "",
            auto_archive_duration: int = 1440,
        ):
            # defer() happens inside the handler *after* the auth gate.
            await self._handle_thread_create_slash(interaction, name, message, auto_archive_duration)
