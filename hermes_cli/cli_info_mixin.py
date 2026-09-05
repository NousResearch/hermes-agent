"""Informational views and reload flows for the interactive CLI: banner, help, tools, usage,
insights, MCP/skills reload, bang shell.

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal symbols are
imported LAZILY inside each method — the mixin never imports ``cli`` at module load time (cycle).
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import threading
import time

from hermes_constants import is_termux as _is_termux_environment
from rich.markup import escape as _escape
from utils import base_url_hostname

from hermes_cli.cli_modal_mixin import _gated_confirm
from hermes_cli.colors import Colors as _Colors

CONFIG_WATCH_INTERVAL = 5.0  # seconds between config.yaml stat() calls

_TOOL_PROGRESS_CYCLE = ["off", "new", "all", "verbose"]
# Raw ANSI (not Rich markup): _cprint routes through prompt_toolkit's renderer, while Rich markup
# written to stdout gets mangled by patch_stdout's StdoutProxy ('?[33mTool progress: NEW?[0m').
_TOOL_PROGRESS_LABELS = {
    # Use raw ANSI codes via _cprint so the output is routed through prompt_toolkit's renderer.
    # self.console.print() with Rich markup writes directly to stdout which patch_stdout's StdoutProxy
    # mangles into garbled sequences like '?[33mTool progress: NEW?[0m' (#2262).
    "off": f"{_Colors.DIM}Tool progress: OFF{_Colors.RESET} — silent mode, just the final response.",
    "new": f"{_Colors.YELLOW}Tool progress: NEW{_Colors.RESET} — show each new tool (skip repeats).",
    "all": f"{_Colors.GREEN}Tool progress: ALL{_Colors.RESET} — show every tool call.",
    "verbose": f"{_Colors.BOLD}{_Colors.GREEN}Tool progress: VERBOSE{_Colors.RESET} — full args, results, and think blocks.",
}

_RELOAD_MCP_CHOICES = [
    ("once", "Approve Once", "reload now"),
    ("always", "Always Approve", "reload now and silence this prompt permanently"),
    ("cancel", "Cancel", "leave MCP tools unchanged")]
_RELOAD_MCP_DETAIL = (
    "Reloading MCP servers rebuilds the tool set for this session and\n"
    "invalidates the provider prompt cache. The next message will\n"
    "re-send full input tokens (can be expensive on long-context or\n"
    "high-reasoning models).")


def _ascii_box(title: str, width: int) -> None:
    """Print the kawaii ``+---+ | title | +---+`` header used by /tools and /toolsets."""
    pad = width - len(title)
    print("+" + "-" * width + "+")
    print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
    print("+" + "-" * width + "+")


def _toolset_map(tools, availability, get_toolset_for_tool) -> dict:
    """tool name → toolset id, including tools of unavailable toolsets (banner snapshot)."""
    tmap = {t["function"]["name"]: get_toolset_for_tool(t["function"]["name"]) for t in tools}
    for item in availability.get("unavailable_toolsets", []):
        for name in item.get("tools", []):
            tmap.setdefault(name, item.get("id", item.get("name", "")))
    return tmap


def _skill_line(item: dict) -> str:
    nm = item.get("name", "")
    desc = item.get("description", "")
    return f"    - {nm}: {desc}" if desc else f"    - {nm}"


class CLIInfoMixin:
    """Informational views and reload flows for the interactive CLI: banner, help, tools, usage,
    insights, MCP/skills reload, bang shell."""

    def _show_plugin_compat_notice(self) -> None:
        """One yellow block under the banner when an enabled external plugin imports paths scheduled for
        removal (red once the date has passed and the plugin was skipped). Never raises."""
        try:
            from hermes_cli.plugin_compat import compat_report, removal_in_effect, summary_lines
            lines = summary_lines(compat_report())
        except Exception:
            return
        if not lines:
            return
        colour = "bold red" if removal_in_effect() else "bold yellow"
        self._console_print()
        self._console_print(f"[{colour}]⚠  {lines[0]}[/]")
        self._console_print(f"[dim]   {lines[1]}[/]")

    def show_banner(self):
        """Display the welcome banner in Claude Code style."""
        from cli import _build_compact_banner, get_tool_definitions, logger
        from hermes_cli.banner import build_welcome_banner
        self.console.clear()
        ctx_len = None
        if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'context_compressor'):
            ctx_len = self.agent.context_compressor.context_length

        # Auto-compact for narrow terminals — the full banner needs ~80 columns to avoid wrapping.
        if self.compact or shutil.get_terminal_size().columns < 80:
            self._console_print(_build_compact_banner())
            self._show_status()
        else:
            # Warm-launch fast path: replay last launch's tool panel when the snapshot fingerprint
            # (config.yaml + .env + checkout rev + toolsets) is unchanged, skipping the ~0.5-0.9s
            # cold get_tool_definitions walk. The agent's REAL tool list is still computed fresh at
            # first message; a background refresh re-verifies the snapshot so drift self-heals.
            from hermes_cli.banner import (
                compute_toolset_availability, load_banner_snapshot, save_banner_snapshot)
            try:
                snapshot = load_banner_snapshot(self.enabled_toolsets)
            except Exception:
                snapshot = None
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())  # where commands will execute
            banner_kw = dict(
                console=self.console, model=self.model, cwd=cwd,
                enabled_toolsets=self.enabled_toolsets, session_id=self.session_id,
                context_length=ctx_len, provider=self.provider)

            if snapshot is not None:
                self._defer_tool_warnings = True
                toolset_map = snapshot["toolset_map"]
                build_welcome_banner(
                    tools=snapshot["tools"],
                    get_toolset_for_tool=lambda name: toolset_map.get(name),
                    availability=snapshot["availability"],
                    skills_by_category=snapshot.get("skills_by_category"),
                    **banner_kw)

                def _refresh_banner_snapshot() -> None:
                    try:
                        from model_tools import get_toolset_for_tool
                        tools = get_tool_definitions(
                            enabled_toolsets=self.enabled_toolsets,
                            disabled_toolsets=self.disabled_toolsets, quiet_mode=True)
                        availability = compute_toolset_availability(self.enabled_toolsets)
                        tmap = _toolset_map(tools, availability, get_toolset_for_tool)
                        save_banner_snapshot(tools, self.enabled_toolsets, availability, tmap)
                    except Exception:
                        logger.debug("banner snapshot refresh failed", exc_info=True)

                threading.Thread(
                    target=_refresh_banner_snapshot, name="banner-snapshot-refresh", daemon=True,
                ).start()
            else:
                # Cold path: compute live, then persist the snapshot for the next launch.
                from model_tools import get_toolset_for_tool
                tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets,
                                             disabled_toolsets=self.disabled_toolsets, quiet_mode=True)
                availability = compute_toolset_availability(self.enabled_toolsets)
                build_welcome_banner(tools=tools, availability=availability, **banner_kw)
                try:
                    tmap = _toolset_map(tools, availability, get_toolset_for_tool)
                    save_banner_snapshot(tools, self.enabled_toolsets, availability, tmap)
                except Exception:
                    logger.debug("banner snapshot save failed", exc_info=True)

        # Tool discovery is deferred on the Termux bare prompt path (warnings show once tools
        # init). On the snapshot fast path the check walks every check_fn (~180ms) — run it in
        # the background and let its output land above the prompt (patch_stdout-safe).
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            if getattr(self, "_defer_tool_warnings", False):
                threading.Thread(
                    target=self._show_tool_availability_warnings,
                    name="tool-availability-warnings",
                    daemon=True).start()
            else:
                self._show_tool_availability_warnings()

        # Low context warning — tied to the runtime guard so guidance cannot drift.
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        self._show_plugin_compat_notice()
        if ctx_len and ctx_len < MINIMUM_CONTEXT_LENGTH:
            self._console_print()
            self._console_print(
                f"[yellow]⚠️  Context length is only {ctx_len:,} tokens — "
                f"this is likely too low for agent use with tools.[/]")
            self._console_print(
                f"[dim]   Hermes needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens. Tool schemas + system prompt use a large fixed prefix.[/]"
            )
            base_url = getattr(self, "base_url", "") or ""
            from urllib.parse import urlparse as _urlparse
            try:
                _port = _urlparse(base_url if "://" in base_url else f"//{base_url}").port
            except ValueError:
                _port = None
            if _port == 11434 or "ollama" in base_url_hostname(base_url):
                fix = f"Ollama fix: OLLAMA_CONTEXT_LENGTH={MINIMUM_CONTEXT_LENGTH} ollama serve"
            elif _port == 1234:
                fix = "LM Studio fix: Set context length in model settings → reload model"
            else:
                fix = "Fix: Set model.context_length in config.yaml, or increase your server's context setting"
            self._console_print(f"[dim]   {fix}[/]")

        from hermes_cli.model_switch import is_nous_hermes_non_agentic
        if is_nous_hermes_non_agentic(getattr(self, "model", "") or ""):
            self._console_print()
            self._console_print(
                "[bold yellow]⚠  Nous Research Hermes 3 & 4 models are NOT agentic and are not "
                "designed for use with Hermes Agent.[/]")
            self._console_print(
                "[dim]   They lack tool-calling capabilities required for agent workflows. "
                "Consider using an agentic model (Claude, GPT, Gemini, DeepSeek, etc.).[/]")
            self._console_print("[dim]   Switch with: /model sonnet  or  /model gpt5[/]")

        # Project-local skills one-liner: trusted → count; untrusted-with-skills → point at
        # `hermes skills trust`. Never raises.
        try:
            from agent.skill_utils import (
                get_project_skills_dirs, get_untrusted_project_skills_root, iter_skill_index_files)
            _proj_dirs = get_project_skills_dirs()
            if _proj_dirs:
                _n = sum(sum(1 for _ in iter_skill_index_files(d, "SKILL.md")) for d in _proj_dirs)
                if _n:
                    self._console_print(f"[dim]◆ {_n} project skill(s) loaded from this repo[/]")
            else:
                _untrusted = get_untrusted_project_skills_root()
                if _untrusted is not None:
                    _root, _n = _untrusted
                    self._console_print(
                        f"[yellow]◆ {_n} project skill(s) found in {_root} but not "
                        f"loaded — run `hermes skills trust` to enable them.[/]")
        except Exception:
            logger.debug("project skills banner notice failed", exc_info=True)

        self._console_print()

    def _fast_command_available(self) -> bool:
        try:
            from hermes_cli.models import model_supports_fast_mode
        except Exception:
            return False
        agent = getattr(self, "agent", None)
        return model_supports_fast_mode(getattr(agent, "model", None) or getattr(self, "model", None))

    def _command_available(self, slash_command: str) -> bool:
        if slash_command == "/fast":
            return self._fast_command_available()
        return True

    def show_help(self, arg: str = ""):
        """Display help. Bare /help shows categorized core commands with the skill list collapsed
        to one line; /help skills lists all skill commands; /help <query> filters by substring."""
        from cli import (
            ChatConsole, _BOLD, _DIM, _RST, _accent_hex, _cprint, _ensure_skill_commands,
            _termux_example_image_path, get_skill_bundles)
        from hermes_cli.commands import COMMANDS_BY_CATEGORY, HELP_SESSION_SUBGROUPS

        arg = (arg or "").strip()
        skill_commands = _ensure_skill_commands()

        def _row(cmd: str, desc: str, width: int = 15) -> None:
            ChatConsole().print(
                f"    [bold {_accent_hex()}]{cmd:<{width}}[/] [dim]-[/] {_escape(desc)}")

        # /help skills — the full list, kept out of the default view so core commands don't
        # scroll off screen.
        if arg.lower() in ("skills", "skill"):
            if not skill_commands:
                _cprint("\n  No skill commands installed.\n")
                return
            _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} ({len(skill_commands)} installed):")
            for cmd, info in sorted(skill_commands.items()):
                _row(cmd, info['description'], 22)
            _cprint("")
            return

        query = arg.lower() if arg else ""

        try:
            from hermes_cli.skin_engine import get_active_help_header
            header = get_active_help_header("(^_^)? Available Commands")
        except Exception:
            header = "(^_^)? Available Commands"
        header = ((header or "").strip() or "(^_^)? Available Commands")[:55]
        _cprint(f"\n{_BOLD}+{'-' * 55}+{_RST}")
        _cprint(f"{_BOLD}|{header:^55}|{_RST}")
        _cprint(f"{_BOLD}+{'-' * 55}+{_RST}")

        def _section(title: str, rows) -> None:
            """Print available/matching rows under a `── title ──` header (omitted if empty)."""
            printed_header = False
            for cmd, desc in rows:
                if not self._command_available(cmd):
                    continue
                if query and query not in cmd.lower() and query not in desc.lower():
                    continue
                if not printed_header:
                    _cprint(f"\n  {_BOLD}── {title} ──{_RST}")
                    printed_header = True
                _row(cmd, desc)

        for category, commands in COMMANDS_BY_CATEGORY.items():
            if category != "Session":
                _section(category, commands.items())
                continue
            # The oversized Session category renders as sub-groups
            # (Session / Context / Background & Automation).
            sub_of = {f"/{n}": sub for sub, names in HELP_SESSION_SUBGROUPS.items() for n in names}
            buckets: dict[str, list[tuple[str, str]]] = {"Session": []}
            for _sub in HELP_SESSION_SUBGROUPS:
                buckets[_sub] = []
            for cmd, desc in commands.items():
                buckets[sub_of.get(cmd, "Session")].append((cmd, desc))
            for _sub in ("Session", *HELP_SESSION_SUBGROUPS.keys()):
                _section(_sub, buckets.get(_sub) or [])

        # Skill commands collapse to a one-line pointer by default so 60+ entries don't bury the
        # core reference; filter mode includes matching skill commands inline.
        if query:
            matched_skills = [
                (cmd, info) for cmd, info in sorted(skill_commands.items())
                if query in cmd.lower() or query in (info.get("description", "").lower())]
            if matched_skills:
                _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} (matching '{arg}'):")
                for cmd, info in matched_skills:
                    _row(cmd, info['description'], 22)
        elif skill_commands:
            _cprint(
                f"\n  ⚡ {_BOLD}Skill Commands{_RST}: {len(skill_commands)} installed "
                f"— {_DIM}/help skills{_RST} to list them")

        _bundles_now = get_skill_bundles()
        if _bundles_now and not query:
            _cprint(f"\n  ▣ {_BOLD}Skill Bundles{_RST} ({len(_bundles_now)} installed):")
            for cmd, info in sorted(_bundles_now.items()):
                skill_count = len(info.get("skills", []))
                desc = info.get("description") or f"Load {skill_count} skills"
                ChatConsole().print(
                    f"    [bold {_accent_hex()}]{cmd:<22}[/] [dim]-[/] "
                    f"{_escape(desc)} [dim]({skill_count} skills)[/]")

        quick_commands = self.config.get("quick_commands", {})
        if quick_commands and not query:
            _cprint(f"\n  ⚡ {_BOLD}Quick Commands{_RST} ({len(quick_commands)} configured):")
            for name, qcmd in sorted(quick_commands.items()):
                _row('/' + name, qcmd.get("description", qcmd.get("type", "")), 22)

        if query:
            _cprint(f"\n  {_DIM}Filtered by '{arg}' — run /help for the full list.{_RST}\n")
            return

        _cprint(f"\n  {_DIM}Tip: /help skills lists skill commands · /help <text> filters · Ctrl+P opens the command palette{_RST}")
        _cprint(f"  {_DIM}Multi-line: Ctrl+J, Alt+Enter, or \\\\+Enter for a new line{_RST}")
        _cprint(f"  {_DIM}Draft editor: Ctrl+G (Alt+G in VSCode/Cursor){_RST}")
        if _is_termux_environment():
            _cprint(f"  {_DIM}Attach image: /image {_termux_example_image_path()} or start your prompt with a local image path{_RST}\n")
        else:
            _cprint(f"  {_DIM}Paste image: Alt+V (or /paste){_RST}\n")

    def show_tools(self):
        """Display available tools with kawaii ASCII art."""
        from cli import get_tool_definitions
        from model_tools import get_toolset_for_tool
        # Pre-assembly list: /tools is a discovery surface, so it must show the full catalog
        # including tools deferred behind the tool_search bridge (users verify MCP installs here).
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets,
                                     disabled_toolsets=self.disabled_toolsets, quiet_mode=True,
                                     skip_tool_search_assembly=True)
        if not tools:
            print("(;_;) No tools available")
            return

        print()
        _ascii_box("(^_^)/ Available Tools", 78)
        print()

        toolsets: dict[str, list] = {}
        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            toolset = get_toolset_for_tool(name) or "unknown"
            desc = tool["function"].get("description", "").split("\n")[0]
            # First sentence: split on ". " (period+space) so "e.g." / "v2.0" stay intact.
            if ". " in desc:
                desc = desc[:desc.index(". ") + 1]
            toolsets.setdefault(toolset, []).append((name, desc))

        for toolset in sorted(toolsets.keys()):
            print(f"  [{toolset}]")
            for name, desc in toolsets[toolset]:
                print(f"    * {name:<20} - {desc}")
            print()

        print(f"  Total: {len(tools)} tools  ヽ(^o^)ノ")
        print()

    def show_toolsets(self):
        """Display available toolsets with kawaii ASCII art."""
        from toolsets import get_all_toolsets, get_toolset_info
        all_toolsets = get_all_toolsets()

        print()
        _ascii_box("(^_^)b Available Toolsets", 58)
        print()

        for name in sorted(all_toolsets.keys()):
            info = get_toolset_info(name)
            if info:
                marker = "(*)" if self.enabled_toolsets and name in self.enabled_toolsets else "   "
                print(f"  {marker} {name:<18} [{info['tool_count']:>2} tools] - {info['description']}")

        print()
        print("  (*) = currently enabled")
        print()
        print("  Tip: Use 'all' or '*' to enable all toolsets")
        print("  Example: python cli.py --toolsets web,terminal")
        print()

    def _handle_whoami_command(self):
        """Display slash-command access for the local CLI surface."""
        import getpass
        try:
            user_name = getpass.getuser() or "?"
        except Exception:
            user_name = "?"
        print()
        print("  You:            cli (local terminal)")
        print(f"  User:           {user_name}")
        print("  Tier:           unrestricted")
        print("  Slash commands: all available")
        print()

    def _busy_inline_command(self, text: str, has_images: bool, names: tuple) -> bool:
        """True when ``text`` is a slash command in ``names`` typed while the agent is running.

        Such commands MUST bypass the normal ``_pending_input`` → ``process_loop`` path: the loop
        is blocked inside ``self.chat()`` for the whole run, so by the time the queued command is
        pulled, ``_agent_running`` has flipped back to False and it would be delivered as a
        next-turn message. Dispatching inline on the UI thread acts mid-run (``agent.steer()`` is
        thread-safe; ``/bg`` / ``/btw`` start their side session without touching the foreground turn).
        """
        from cli import _looks_like_slash_command
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        if not getattr(self, "_agent_running", False):
            return False
        try:
            from hermes_cli.commands import resolve_command
            cmd = resolve_command(text.split(None, 1)[0].lower().lstrip('/'))
            return bool(cmd and cmd.name in names)
        except Exception:
            return False

    def _should_handle_steer_command_inline(self, text: str, has_images: bool = False) -> bool:
        """Return True when /steer should be dispatched immediately while the agent is running."""
        return self._busy_inline_command(text, has_images, ("steer",))

    def _should_handle_background_command_inline(
        self, text: str, has_images: bool = False) -> bool:
        """Return True when /bg or /btw should be dispatched while the agent runs (their
        ``CommandDef`` entries declare ``busy_policy="dispatch"``; the classic CLI honours it here)."""
        return self._busy_inline_command(text, has_images, ("bg", "btw"))

    def handle_bang_shell(self, text: str) -> bool:
        """Run a ``!<command>`` submission. Returns True when it was handled.

        Dispatched from the input loop BEFORE slash routing and before anything is queued for the
        agent, so a bang command never becomes a turn: nothing touches ``conversation_history``,
        zero tokens, role alternation / prompt caching untouched by construction
        (tests/cli/test_bang_shell_mode.py). Returns False when the text is not a bang command or
        bang mode is disabled for this context (gateway/cron), so the caller routes normally.
        """
        from cli import _rich_text_from_ansi
        from hermes_cli.bang_shell import (
            USAGE_HINT, bang_shell_enabled, check_bang_approval, is_bang_command,
            parse_bang_command, resolve_bang_cwd, run_bang_command)

        if not is_bang_command(text):
            return False
        if not bang_shell_enabled():
            # Gateway / cron / API: no composer, no human at a keyboard, and those users already
            # have shells — route normally rather than becoming remote execution.
            return False

        command = parse_bang_command(text)
        if not command:  # bare `!` — show what the feature does
            self._console_print(f"[dim]{USAGE_HINT}[/]")
            return True

        approval = check_bang_approval(command)
        if not approval.get("approved"):
            message = approval.get("message") or (
                f"Command denied: {approval.get('description', 'flagged as dangerous')}")
            self._console_print(f"[bold red]{_escape(str(message))}[/]")
            return True

        exit_code = run_bang_command(
            command,
            cwd=resolve_bang_cwd(getattr(self, "session_id", None)),
            writer=lambda line: self._console_print(_rich_text_from_ansi(line)))
        if exit_code:
            self._console_print(f"[dim]! exited {exit_code}[/]")
        return True

    def _show_gateway_status(self):
        """Show status of the gateway and connected messaging platforms."""
        from hermes_constants import display_hermes_home
        from gateway.config import load_gateway_config, Platform

        print()
        print("+" + "-" * 60 + "+")
        print("|" + " " * 15 + "(✿◠‿◠) Gateway Status" + " " * 17 + "|")
        print("+" + "-" * 60 + "+")
        print()

        try:
            config = load_gateway_config()
            print("  Messaging Platform Configuration:")
            print("  " + "-" * 55)
            platform_status = {
                Platform.TELEGRAM: ("Telegram", "TELEGRAM_BOT_TOKEN"),
                Platform.DISCORD: ("Discord", "DISCORD_BOT_TOKEN"),
                Platform.SLACK: ("Slack", "SLACK_BOT_TOKEN"),
                Platform.WHATSAPP: ("WhatsApp", "WHATSAPP_ENABLED")}
            for platform, (name, env_var) in platform_status.items():
                pconfig = config.platforms.get(platform)
                if pconfig and pconfig.enabled:
                    home = config.get_home_channel(platform)
                    home_str = f" → {home.name}" if home else ""
                    print(f"    ✓ {name:<12} Enabled{home_str}")
                else:
                    print(f"    ○ {name:<12} Not configured ({env_var})")

            print()
            print("  Session Reset Policy:")
            print("  " + "-" * 55)
            policy = config.default_reset_policy
            print(f"    Mode: {policy.mode}")
            print(f"    Daily reset at: {policy.at_hour}:00")
            print(f"    Idle timeout: {policy.idle_minutes} minutes")
            print()
            print("  To start the gateway:")
            print("    python cli.py --gateway")
            print()
            print(f"  Configuration file: {display_hermes_home()}/config.yaml")
            print()
        except Exception as e:
            print(f"  Error loading gateway config: {e}")
            print()
            print("  To configure the gateway:")
            print("    1. Set environment variables:")
            print("       TELEGRAM_BOT_TOKEN=your_token")
            print("       DISCORD_BOT_TOKEN=your_token")
            print(f"    2. Or configure settings in {display_hermes_home()}/config.yaml")
            print()

    def _print_random_tip(self) -> None:
        """Best-effort discovery tip (startup + /clear); never raises."""
        try:
            from hermes_cli.tips import get_random_tip
            _tip = get_random_tip()
            try:
                from hermes_cli.skin_engine import get_active_skin
                _tip_color = get_active_skin().get_color("banner_dim", "#B8860B")
            except Exception:
                _tip_color = "#B8860B"
            self._console_print(f"[dim {_tip_color}]✦ Tip: {_tip}[/]")
        except Exception:
            pass

    def _toggle_verbose(self):
        """Cycle tool progress mode: off → new → all → verbose → off.

        Tool-progress display is INDEPENDENT of global DEBUG logging: this never changes
        ``self.verbose`` or the agent's ``verbose_logging`` / ``quiet_mode`` (those belong to
        ``-v`` and ``/verbose-logging``).
        """
        from cli import _cprint, save_config_value
        try:
            idx = _TOOL_PROGRESS_CYCLE.index(self.tool_progress_mode)
        except ValueError:
            idx = 2  # default to "all"
        self.tool_progress_mode = _TOOL_PROGRESS_CYCLE[(idx + 1) % len(_TOOL_PROGRESS_CYCLE)]

        # /verbose is the explicit tool-progress control, so cycling it takes ownership of the
        # mode back from focus view (else a "focus" badge + hidden-line counts would show while
        # tool lines visibly print). Display-only state change.
        if getattr(self, "_focus_view_enabled", False):
            self._focus_view_enabled = False
            self._focus_saved_tool_progress = None
            self._focus_hidden_lines = 0
            self._focus_last_counted_tool = None
            try:
                from hermes_cli.focus_view import FOCUS_CONFIG_KEY
                save_config_value(FOCUS_CONFIG_KEY, False)
            except Exception:
                pass

        if self.agent:
            self.agent.reasoning_callback = self._current_reasoning_callback()
            # Sync the live agent so tool_executor rendering reflects the new mode this turn.
            self.agent.tool_progress_mode = self.tool_progress_mode

        _cprint(_TOOL_PROGRESS_LABELS.get(self.tool_progress_mode, ""))

    def _handle_usage_command(self, cmd_original: str):
        """Dispatch `/usage [reset [--force]]`: bare `/usage` is the classic display; `reset`
        redeems one banked Codex rate-limit reset credit (refuses unless exhausted or --force)."""
        parts = cmd_original.split()
        args = [p.lower() for p in parts[1:]]
        if args and args[0] == "reset":
            self._usage_reset(force="--force" in args[1:])
            return
        if args:
            print(f"  Unknown /usage subcommand: {' '.join(parts[1:])}. Try /usage or /usage reset [--force].")
            return
        self._show_usage()

    def _agent_or_self(self, name: str):
        """Provider-ish attribute from the live agent, falling back to the CLI's own value."""
        return (getattr(self.agent, name, None) if self.agent else None) or getattr(self, name, None)

    def _usage_reset(self, force: bool = False):
        """`/usage reset [--force]` — redeem one banked Codex reset credit."""
        if str(self._agent_or_self("provider") or "").strip().lower() != "openai-codex":
            print("  Banked usage resets are only available on the openai-codex provider.")
            print("  Switch with `/model` or `hermes auth` first.")
            return
        from agent.account_usage import redeem_codex_reset_credit

        print("  ⏳ Checking banked reset credits...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            try:
                result = _pool.submit(
                    redeem_codex_reset_credit, base_url=self._agent_or_self("base_url"),
                    api_key=self._agent_or_self("api_key"), force=force,
                ).result(timeout=45.0)
            except concurrent.futures.TimeoutError:
                print("  ❌ Timed out talking to the Codex backend — try again shortly.")
                return
        print(f"  {result.message}")

    def _show_context_breakdown(self, cmd_original: str = ""):
        """`/context [all]` — 5×20 glyph grid (cell ≈ 1% of the window) plus an estimated
        per-category table; `all` appends per-skill / per-toolset costs. Read-only: same chars/4
        engine as the desktop popover (agent.context_breakdown) — no provider calls, no cache impact."""
        if not self.agent:
            print("  (._.) No active agent -- send a message first.")
            return

        args = cmd_original.split(maxsplit=1)[1].strip().lower() if " " in cmd_original else ""
        expanded = args in {"all", "full", "details"}

        from agent.context_breakdown import (
            compute_context_details, compute_session_context_breakdown,
            render_context_breakdown_lines)
        try:
            payload = compute_session_context_breakdown(self.agent, self.conversation_history)
        except Exception as e:
            print(f"  (._.) Could not compute context breakdown: {e}")
            return

        details = None
        if expanded:
            try:
                details = compute_context_details(self.agent)
            except Exception:
                details = {"skills": [], "toolsets": []}

        print()
        print(f"  🧠 Context Usage — {payload.get('model') or self.model}")
        print()
        for line in render_context_breakdown_lines(payload, details=details, grid=True):
            print(f"  {line}")
        print()

    def _show_usage(self):
        """Rate limits + session token usage (when a live agent exists) + Nous credits.

        The Nous credits block is agent-independent (portal fetch), so it runs even with no live
        agent — the TUI's /usage slash-worker resumes the session WITHOUT building an agent.
        """
        from cli import datetime, format_duration_compact

        def _credits_or(fallback: str) -> None:
            if self._print_nous_credits_block():
                self._print_usage_cta()
            else:
                print(fallback)

        if not self.agent:
            _credits_or("(._.) No active agent -- send a message first.")
            return
        agent = self.agent
        calls = agent.session_api_calls
        if calls == 0:
            _credits_or("(._.) No API calls made yet in this session.")
            return

        rl_state = agent.get_rate_limit_state()
        if rl_state and rl_state.has_data:
            from agent.rate_limit_tracker import format_rate_limit_display
            print()
            print(format_rate_limit_display(rl_state))
            print()

        input_tokens = getattr(agent, "session_input_tokens", 0) or 0
        output_tokens = getattr(agent, "session_output_tokens", 0) or 0
        reasoning_tokens = getattr(agent, "session_reasoning_tokens", 0) or 0
        compressor = agent.context_compressor
        last_prompt = compressor.last_prompt_tokens if compressor.last_prompt_tokens > 0 else 0
        ctx_len = compressor.context_length
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        elapsed = format_duration_compact((datetime.now() - self.session_start).total_seconds())

        print("  📊 Session Token Usage")
        print(f"  {'─' * 40}")
        print(f"  Model:                     {agent.model}")
        print(f"  Input tokens:              {input_tokens:>10,}")
        print(f"  Output tokens:             {output_tokens:>10,}")
        if reasoning_tokens:
            print(f"  ↳ Reasoning (subset):      {reasoning_tokens:>10,}")
        print(f"  Prompt tokens (total):     {agent.session_prompt_tokens:>10,}")
        print(f"  Completion tokens:         {agent.session_completion_tokens:>10,}")
        print(f"  Total tokens:              {agent.session_total_tokens:>10,}")
        print(f"  API calls:                 {calls:>10,}")
        print(f"  Session duration:          {elapsed:>10}")
        print(f"  {'─' * 40}")
        print(f"  Current context:  {last_prompt:,} / {ctx_len:,} ({pct:.0f}%)")
        print(f"  Messages:         {len(self.conversation_history)}")
        print(f"  Compressions:     {compressor.compression_count}")

        # Account limits — fetched off-thread with a hard timeout so slow provider APIs don't
        # hang the prompt. Lazy import: pulls the OpenAI SDK chain.
        provider = self._agent_or_self("provider")
        from agent.account_usage import fetch_account_usage, render_account_usage_lines
        account_snapshot = None
        if provider:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                try:
                    account_snapshot = _pool.submit(
                        fetch_account_usage, provider, base_url=self._agent_or_self("base_url"),
                        api_key=self._agent_or_self("api_key"),
                    ).result(timeout=10.0)
                except (concurrent.futures.TimeoutError, Exception):
                    account_snapshot = None
        account_lines = [f"  {line}" for line in render_account_usage_lines(account_snapshot)]
        if account_lines:
            print()
            for line in account_lines:
                print(line)

        if self._print_nous_credits_block():
            self._print_usage_cta()

        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            for noisy in ('openai', 'openai._base_client', 'httpx', 'httpcore', 'asyncio', 'hpack', 'grpc', 'modal'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)

    def _show_insights(self, command: str = "/insights"):
        """Show usage insights and analytics from session history (`--days N` / `N`, `--source`)."""
        parts = command.split()
        days = 30
        source = None
        i = 1
        while i < len(parts):
            if parts[i] == "--days" and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    print(f"  Invalid --days value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--source" and i + 1 < len(parts):
                source = parts[i + 1]
                i += 2
            else:
                if parts[i].isdigit():
                    days = int(parts[i])
                i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine
            db = SessionDB()
            try:
                engine = InsightsEngine(db)
                print(engine.format_terminal(engine.generate(days=days, source=source)))
            finally:
                db.close()
        except Exception as e:
            print(f"  Error generating insights: {e}")

    def _check_config_mcp_changes(self) -> None:
        """Detect mcp_servers changes in config.yaml (polled from process_loop every
        CONFIG_WATCH_INTERVAL seconds) and react.

        Default (``mcp.auto_reload_on_config_change: true``) auto-triggers ``_reload_mcp()``.
        When opted out it only notifies and points at ``/reload-mcp`` — every reload rebuilds the
        tool surface and INVALIDATES the provider prompt cache (next message re-sends the full
        prefix), so silent reloads are wrong when external tooling rewrites config.yaml often.

        Instead it notifies the user that the config changed and that they can apply it with ``/reload-mcp``
        — while warning that ``/reload-mcp`` rebuilds the tool surface and **invalidates the provider prompt
        cache** (the next message re-sends the full input prefix, expensive on long-context / high-reasoning
        models). See #1474.
        """
        import yaml as _yaml

        now = time.monotonic()
        if now - self._last_config_check < CONFIG_WATCH_INTERVAL:
            return
        self._last_config_check = now

        from hermes_cli.config import get_config_path as _get_config_path
        cfg_path = _get_config_path()
        if not cfg_path.exists():
            return
        try:
            mtime = cfg_path.stat().st_mtime
        except OSError:
            return
        if mtime == self._config_mtime:
            return  # unchanged — fast path

        self._config_mtime = mtime
        try:
            with open(cfg_path, encoding="utf-8") as f:
                new_cfg = _yaml.safe_load(f) or {}
        except Exception:
            return

        # Expand ${VAR} templates so the comparison matches the init snapshot (populated from the
        # deep-merged + expanded config); otherwise any save_config_value() rewrite of an
        # unrelated key would false-positive on "${POWERMEM_API_KEY}" vs its expanded value.
        from hermes_cli.config import _expand_env_vars
        new_mcp = _expand_env_vars(new_cfg.get("mcp_servers") or {})
        if new_mcp == self._config_mcp_servers:
            return  # some other section was edited

        # Read the toggle from the config just parsed so the user can flip it in the same edit;
        # missing key means default-on.
        _mcp_cfg = new_cfg.get("mcp")
        _auto = _mcp_cfg.get("auto_reload_on_config_change", True) if isinstance(_mcp_cfg, dict) else True
        self._config_mcp_servers = new_mcp

        if not _auto:
            print()
            print("🔄 MCP server config changed — reload skipped (auto-reload disabled).")
            print("   New settings are NOT applied yet. To apply them now, run:")
            print("     /reload-mcp")
            print("   ⚠️  Note: /reload-mcp rebuilds the tool set and invalidates the")
            print("   provider prompt cache (next message re-sends full input tokens).")
            return

        # Separate thread so a hung MCP server can't block process_loop (freezing the TUI).
        print()
        print("🔄 MCP server config changed — reloading connections...")
        threading.Thread(target=self._reload_mcp, daemon=True).start()

    def _confirm_and_reload_mcp(self, cmd_original: str = "") -> None:
        """Interactive /reload-mcp — confirm (Approve Once / Always Approve / Cancel, gated by
        ``approvals.mcp_reload_confirm``, default on), then reload. The config watcher's
        auto-reload calls ``_reload_mcp`` directly. Reloading invalidates the provider prompt cache
        (tool schemas are baked into the system prompt), hence the warning."""
        choice = _gated_confirm(
            self, "reload-mcp", "mcp_reload_confirm",
            title="⚠️  /reload-mcp — Prompt cache invalidation warning",
            detail=_RELOAD_MCP_DETAIL,
            choices=_RELOAD_MCP_CHOICES,
            unchanged="MCP tools unchanged.",
            always_msg="🔒 Future /reload-mcp calls will run without confirmation.",
            once_verb="reloading")
        if choice is None:
            return
        with self._busy_command(self._slow_command_status(cmd_original)):
            self._reload_mcp()

    def _reload_mcp(self):
        """Reload MCP servers: disconnect all, re-read config.yaml, reconnect, then refresh the
        agent's tool list so the model sees the updated tools on the next turn."""
        try:
            from tools.mcp_tool_lifecycle import shutdown_mcp_servers
            from tools.mcp_tool_discovery import discover_mcp_tools
            from tools.mcp_tool_agent import reprobe_tool_availability
            from tools.mcp_tool import _servers, _lock
            with _lock:
                old_servers = set(_servers.keys())
            if not self._command_running:
                print("🔄 Reloading MCP servers...")

            shutdown_mcp_servers()
            reprobe_tool_availability()  # explicit reload also re-probes check_fn availability
            new_tools = discover_mcp_tools()  # reads config.yaml fresh

            with _lock:
                connected_servers = set(_servers.keys())
            diff = {
                "Added": connected_servers - old_servers,
                "Removed": old_servers - connected_servers,
                "Reconnected": connected_servers & old_servers}
            for label, icon in (("Reconnected", "♻️ "), ("Added", "➕"), ("Removed", "➖")):
                if diff[label]:
                    print(f"  {icon} {label}: {', '.join(sorted(diff[label]))}")
            if not connected_servers:
                print("  No MCP servers connected.")
            else:
                print(f"  🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Route through the shared helper so this path stays in lockstep with the TUI RPC /
            # gateway reload / late-binding paths (name-diff, thread-safe, additive-preserving so
            # memory-provider and context-engine tools survive the rebuild).
            if self.agent is not None:
                from tools.mcp_tool_agent import refresh_agent_mcp_tools
                # Pick up servers ENABLED in config this session: enabled_toolsets was resolved at
                # startup, so merge now-connected names in (unless `all`/`*` is pinned) so a
                # freshly-added server isn't filtered out. Mirrors startup (see __init__).
                enabled_override = None
                et = self.enabled_toolsets
                if et and "all" not in et and "*" not in et:
                    merged = list(et)
                    for _name in sorted(connected_servers):
                        if _name not in merged:
                            merged.append(_name)
                    enabled_override = merged
                refresh_agent_mcp_tools(self.agent, enabled_override=enabled_override, quiet_mode=True)
                if enabled_override is not None:
                    self.enabled_toolsets = enabled_override

            # Tell the model tools changed — appended at the END so the prefix cache survives.
            change_parts = [
                f"{label} servers: {', '.join(sorted(names))}" for label, names in diff.items() if names
            ]
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            self.conversation_history.append({
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            })

            # Persist now so the session log reflects the refreshed tools list (best-effort).
            if self.agent is not None:
                try:
                    self.agent._persist_session(self.conversation_history, self.conversation_history)
                except Exception:
                    pass

            print(f"  ✅ Agent updated — {len(self.agent.tools if self.agent else [])} tool(s) available")
        except Exception as e:
            print(f"  ❌ MCP reload failed: {e}")

    def _reload_skills(self) -> None:
        """Reload skills: rescan ~/.hermes/skills/ and queue a note for the next user turn.

        Skills are invoked at runtime (``/skill-name``, ``skills_list``, ``skill_view``), not from
        the system prompt, so this does NOT clear the prompt cache. If anything was added/removed
        a one-shot note is prepended to the NEXT user message (``_pending_skills_reload_note``,
        same pattern as ``_pending_model_switch_note``) — nothing is written to
        conversation_history, so message alternation stays intact.
        """
        try:
            from agent.skill_commands import reload_skills, get_skill_commands
            if not self._command_running:
                print("🔄 Reloading skills...")
            result = reload_skills()

            # Sync cli.py's module-level _skill_commands so help / dispatch / Tab-completion see
            # the updated dict without a restart.
            import cli as _cli
            _cli._skill_commands = get_skill_commands()
            added = result.get("added", [])      # [{"name", "description"}, ...]
            removed = result.get("removed", [])
            total = result.get("total", 0)

            if not added and not removed:
                print("  No new skills detected.")
                print(f"  📚 {total} skill(s) available")
                return

            if added:
                print("  ➕ Added Skills:")
                for item in added:
                    print(f"  {_skill_line(item)}")
            if removed:
                print("  ➖ Removed Skills:")
                for item in removed:
                    print(f"  {_skill_line(item)}")
            print(f"  📚 {total} skill(s) available")

            # Same shape as the system prompt's skill catalog (``    - name: description``).
            sections = ["[USER INITIATED SKILLS RELOAD:"]
            if added:
                sections += ["", "Added Skills:", *(_skill_line(item) for item in added)]
            if removed:
                sections += ["", "Removed Skills:", *(_skill_line(item) for item in removed)]
            sections += ["", "Use skills_list to see the updated catalog.]"]
            self._pending_skills_reload_note = "\n".join(sections)
        except Exception as e:
            print(f"  ❌ Skills reload failed: {e}")
