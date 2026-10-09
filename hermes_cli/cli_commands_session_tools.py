"""Shared print/translate helpers + session-info slash handlers (``cli_commands_mixin`` split).

Moved verbatim from ``hermes_cli/cli_commands_mixin.py`` (file-line ratchet #68779): the
lazy-binding contract carries over unchanged — ``cli.py``-internal symbols are imported at
call time via ``from cli import ...``, so patch seams on the ``cli`` facade still intercept.
``cli_commands_mixin`` re-imports every helper moved here, so existing ``from
cli_commands_mixin import _t`` style imports keep working.
"""

from __future__ import annotations

import argparse
import importlib
import io
import shlex
from contextlib import redirect_stdout
from io import StringIO

from agent.i18n import t  # noqa: F401  module-global seam: _t/_tn/_gt close over it


def _cp(*lines: str) -> None:
    """``_cprint`` each line (lazy import: cli.py imports this module)."""
    from cli import _cprint
    for line in lines:
        _cprint(line)


def _pr(*lines: str) -> None:
    """print() each line."""
    for line in lines:
        print(line)


def _save(key: str, value) -> bool:
    """cli.save_config_value, resolved lazily (cli.py imports this module)."""
    from cli import save_config_value
    return save_config_value(key, value)


def _dim(text: str) -> str:
    """Wrap ``text`` in the dim ANSI escape."""
    from cli import _DIM, _RST
    return f"{_DIM}{text}{_RST}"


def _dim_line(text: str) -> str:
    """Two-space indented dim line (the standard slash-command hint shape)."""
    return "  " + _dim(text)


def _accent(text: str) -> str:
    """Wrap ``text`` in the skin accent ANSI escape."""
    from cli import _ACCENT, _RST
    return f"{_ACCENT}{text}{_RST}"


def _accent_line(text: str) -> str:
    """Two-space indented accent line (the standard slash-command headline shape)."""
    return f"  {_accent(text)}"


def _t(key: str, **kwargs) -> str:
    """Catalog text for this module: ``cli.commands.<key>`` in the active language."""
    return t(f"cli.commands.{key}", **kwargs)


def _tn(key: str, count: int, **kwargs) -> str:
    """Plural-aware ``_t``: ``<key>_one`` when ``count == 1``, else ``<key>_other``."""
    return _t(f"{key}_{'one' if count == 1 else 'other'}", count=count, **kwargs)


def _gt(key: str, **kwargs) -> str:
    """A ``gateway.<key>`` catalog entry the CLI shares verbatim with the messaging gateway."""
    return t(f"gateway.{key}", **kwargs)


def _lines(text: str, pad: str = "  ") -> list:
    """Each line of a (possibly multi-line) catalog value prefixed with ``pad`` (blank lines kept)."""
    return [f"{pad}{line}" if line else "" for line in text.splitlines()]


def _probe(module: str, name: str, default, *args):
    """``<module>.<name>(*args)`` or ``default`` when the import or the call fails (optional
    subsystems: browser backends, async delegations, wake word, ...)."""
    try:
        return getattr(importlib.import_module(module), name)(*args)
    except Exception:
        return default


class CLICommandsSessionToolsMixin:
    """``/stop``, ``/agents``, ``/journey``, ``/paste``, ``/copy``, ``/image``, ``/tools``,
    ``/profile`` handlers (moved from ``CLICommandsMixin``)."""


    # ---- /stop, /agents -------------------------------------------------------------------
    def _handle_stop_command(self):
        """Handle /stop — kill all running background processes and background (async) delegations.
        Separate from interrupt (stop the current turn), as in Codex.

        See #14602.
        """
        from tools.process_registry import process_registry
        running = [p for p in process_registry.list_sessions() if p.get("status") == "running"]
        # Background subagents live in their own registry, not the process registry.
        n_async = _probe("tools.async_delegation", "active_count", 0)
        if not running and not n_async:
            return print(f"  {_t('stop.none_running')}")
        if running:
            print(f"  {_t('stop.stopping', count=len(running))}")
            print(f"  {_t('stop.stopped', count=process_registry.kill_all(source='cli.stop'))}")
        if n_async:
            from tools.async_delegation import interrupt_all
            print(f"  {_t('stop.interrupted_delegations', count=interrupt_all(reason='/stop'))}")

    def _handle_agents_command(self):
        """Handle /agents — show background processes and agent status."""
        from tools.process_registry import format_uptime_short, process_registry
        processes = process_registry.list_sessions()
        running = [p for p in processes if p.get("status") == "running"]
        finished = [p for p in processes if p.get("status") != "running"]
        _cp(f"  {_t('agents.running_processes', count=len(running))}")
        for p in running:
            up = format_uptime_short(p.get("uptime_seconds", 0))
            _cp(f"    {p.get('session_id', '?')} · {up} · {p.get('command', '')[:80]}")
        if finished:
            _cp(f"  {_t('agents.recently_finished', count=len(finished))}")
        # Background (async) delegations — delegate_task(background=true)
        delegations = _probe("tools.async_delegation", "list_async_delegations", [])
        if delegations:
            running_d = [d for d in delegations if d.get("status") in ("running", "stalling")]
            _cp(f"  {_t('agents.background_delegations', count=len(running_d))}")
            for d in delegations:
                status = d.get("status", "?")
                line = f"    {d.get('delegation_id', '?')} · {status} · {(d.get('goal') or '')[:60]}"
                # Live-status detail for in-flight delegations.
                # See #51690.
                if status == "stalling":
                    quiet = d.get("stalled_after_quiet_seconds")
                    if quiet is not None:
                        line += _t("agents.no_progress", seconds=f"{quiet:.0f}")
                elif status == "running":
                    quiet = d.get("seconds_since_progress")
                    if quiet is not None and quiet >= 60:
                        line += _t("agents.quiet", seconds=f"{quiet:.0f}")
                _cp(line)
                for i, child in enumerate(d.get("children_activity") or []):
                    if not isinstance(child, dict):
                        continue
                    tool = child.get("current_tool")
                    doing = _t("agents.in_tool", tool=tool) if tool else _t("agents.between_turns")
                    part = "      " + _t("agents.child_line", index=i + 1,
                                         api_calls=child.get("api_calls", "?"), doing=doing)
                    idle = child.get("seconds_since_activity")
                    if idle is not None:
                        part += _t("agents.last_activity", seconds=f"{idle:.0f}")
                    _cp(part)
        agent_running = getattr(self, "_agent_running", False)
        _cp(f"  {_t('agents.agent_running') if agent_running else _t('agents.agent_idle')}")

    # ---- /journey, /paste, /copy, /image --------------------------------------------------
    def _handle_journey_command(self, cmd_original: str) -> None:
        """Handle /journey — the learning timeline (see `hermes journey`). Read-only views render
        Rich color that patch_stdout would swallow, so capture with forced ANSI and re-emit via
        ``_cprint``; ``delete``/``edit`` are interactive and keep real stdio."""
        from hermes_cli.journey import register_cli
        parser = argparse.ArgumentParser(prog="/journey", add_help=False)
        register_cli(parser)
        try:
            args = parser.parse_args(shlex.split(cmd_original)[1:])
        except SystemExit:
            return
        try:
            if getattr(args, "journey_action", None) in ("delete", "edit"):
                args.func(args)
                return
            args.force_color = True
            buf = io.StringIO()
            with redirect_stdout(buf):
                args.func(args)
            _cp(buf.getvalue().rstrip("\n"))
        except Exception as exc:
            _cp(f"  {_t('journey.failed', error=exc)}")

    def _handle_paste_command(self):
        """Handle /paste — explicitly check clipboard for an image.

        This is the reliable fallback for terminals where BracketedPaste
        doesn't fire for image-only clipboard content (e.g., VSCode terminal,
        Windows Terminal with WSL2).
        """
        from hermes_cli.clipboard import has_clipboard_image
        if not has_clipboard_image():
            _cp(_dim_line(_t("paste.no_image")))
        elif self._try_attach_clipboard_image():
            _cp(f"  {_t('paste.attached', index=len(self._attached_images))}")
        else:
            _cp(_dim_line(_t("paste.extract_failed")))

    def _handle_copy_command(self, cmd_original: str) -> None:
        """Handle /copy [number] — copy assistant output to clipboard."""
        from cli import _assistant_copy_text
        arg = _command_arg(cmd_original)
        assistant = [m for m in self.conversation_history if m.get("role") == "assistant"]
        if not assistant:
            return _cp(f"  {_t('copy.nothing_yet')}")
        if arg:
            try:
                idx = int(arg) - 1
            except ValueError:
                return _cp(f"  {_t('copy.usage')}")
            if idx < 0 or idx >= len(assistant):
                return _cp(f"  {_t('copy.invalid_number', max=len(assistant))}")
        else:  # latest response that has copyable text
            idx = next((i for i in range(len(assistant) - 1, -1, -1)
                        if _assistant_copy_text(assistant[i].get("content"))), -1)
            if idx < 0:
                return _cp(f"  {_t('copy.nothing_in_responses')}")
        text = _assistant_copy_text(assistant[idx].get("content"))
        if not text:
            return _cp(f"  {_t('copy.nothing_in_response')}")
        try:
            from hermes_cli.clipboard import is_remote_shell_session, write_clipboard_text
            # Over SSH native tools write the REMOTE clipboard; OSC 52 reaches the user's terminal.
            # Locally, OSC 52 is the fallback when native tools are unavailable/fail (SSH/tmux).
            if is_remote_shell_session() or not write_clipboard_text(text):
                # Fixes #31528.
                self._write_osc52_clipboard(text)
                _cp(f"  {_t('copy.copied_osc52', index=idx + 1)}")
            else:
                _cp(f"  {_t('copy.copied', index=idx + 1)}")
        except Exception as e:
            _cp(f"  {_t('copy.failed', error=e)}")

    def _handle_image_command(self, cmd_original: str):
        """Handle /image <path> — attach a local image file for the next prompt."""
        from cli import _DIM, _IMAGE_EXTENSIONS, _RST, _cprint, _resolve_attachment_path, _split_path_input
        raw_args = (cmd_original.split(None, 1)[1].strip() if " " in cmd_original else "")
        if not raw_args:
            hint = "/path/to/image.png"
            _cprint(f"  {_DIM}{_t('image.usage', example=hint)}{_RST}")
            return

        path_token, _remainder = _split_path_input(raw_args)
        image_path = _resolve_attachment_path(path_token)
        if image_path is None:
            return _cp(_dim_line(_t("image.not_found", path=path_token)))
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _cp(_dim_line(_t("image.unsupported", name=image_path.name)))
        self._attached_images.append(image_path)
        _cp(f"  {_t('image.attached', name=image_path.name)}")
        if _remainder:
            _cprint(f"  {_DIM}{_t('image.now_type_prompt', text=_remainder)}{_RST}")

    # ---- /tools, /profile -----------------------------------------------------------------
    def _handle_tools_command(self, cmd: str):
        """Handle /tools [list|disable|enable]. Bare shows the tool list; ``list`` shows per-toolset
        status; disable/enable save to config and reset the session so the new tool set takes
        effect cleanly (no prompt-cache breakage mid-conversation)."""
        parts = _shlex_args(cmd)
        subcommand = parts[0] if parts else ""
        if subcommand not in {"list", "disable", "enable"}:
            return self.show_tools()
        if subcommand == "list":
            return self._run_tools_config(tools_action="list", platform="cli")
        names = parts[1:]
        if not names:
            return _pr(_t("tools.usage", subcommand=subcommand),
                       f"  {_t('tools.example_toolset', subcommand=subcommand)}",
                       f"  {_t('tools.example_mcp', subcommand=subcommand)}")
        # Typing the command is consent. Do NOT use input() — it hangs in prompt_toolkit's loop.
        _cp(_accent(_t("tools.disabling" if subcommand == "disable" else "tools.enabling",
                       names=", ".join(names))))
        self._run_tools_config(tools_action=subcommand, names=names, platform="cli")
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_cli.config import load_config
        self.enabled_toolsets = _get_platform_tools(load_config(), "cli")
        self.new_session()
        _cp(_dim(_t("tools.session_reset")))

    def _run_tools_config(self, **ns) -> None:
        """Run ``tools_disable_enable_command``. Inside the interactive TUI its ANSI print() output
        is captured (isatty=True so colors still render) and re-emitted through _cprint so
        patch_stdout's StdoutProxy doesn't garble the escapes; standalone/tests call straight through."""
        from argparse import Namespace
        from hermes_cli.tools_config import tools_disable_enable_command
        if getattr(self, "_app", None) is None:
            return tools_disable_enable_command(Namespace(**ns))
        buf = _TTYBuf()
        with redirect_stdout(buf):
            tools_disable_enable_command(Namespace(**ns))
        _cp(*buf.getvalue().splitlines())

    def _handle_profile_command(self):
        """Display active profile name and home directory."""
        from hermes_cli.slash_exec import CommandContext, execute_command
        reply = execute_command("profile", CommandContext(surface="cli"))
        _say_block(f"  {_t('profile.profile', profile=reply.data['profile'])}",
                   f"  {_t('profile.home', home=reply.data['home'])}")


class _TTYBuf(StringIO):
    """StringIO that claims to be a TTY so ``hermes_cli.colors.color()`` still emits ANSI escapes."""
    def isatty(self) -> bool:
        return True


def _say_block(*lines: str) -> None:
    """print() the lines framed by a blank line above and below (the /browser output style)."""
    _pr("", *lines, "")


def _command_arg(cmd: str, *, lower: bool = False) -> str:
    """Everything after the slash-command word, stripped (optionally lowercased)."""
    parts = (cmd or "").strip().split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""
    return arg.lower() if lower else arg


def _shlex_args(cmd: str) -> list:
    """Tokens after the command word; falls back to whitespace split on unbalanced quotes."""
    try:
        return shlex.split(cmd)[1:] if cmd else []
    except ValueError:
        return (cmd or "").split()[1:]
