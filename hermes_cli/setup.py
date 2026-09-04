"""Interactive setup wizard for Hermes Agent (config lives in ~/.hermes/).

Independently-runnable sections: Model & Provider, Terminal Backend, Agent Settings, Messaging
Platforms, Tools. Section bodies live in sibling setup_* modules and are re-exported here; they
resolve shared prompt/config helpers lazily through this module so test patches on
``hermes_cli.setup.<name>`` keep working.
"""

import importlib.util
import logging
import os
import re
import sys
import copy
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

from hermes_cli.curses_ui import MenuNavigationEvent, MenuNavigationStart
# Config helpers are re-exported (tests patch them on this module). display_hermes_home is
# imported lazily at call sites (stale-module safety during hermes update).
from hermes_cli.config import (
    cfg_get, DEFAULT_CONFIG, get_hermes_home, get_config_path, get_env_path, load_config, save_config,
    save_env_value, remove_env_value, get_env_value, ensure_hermes_home,
)
from hermes_cli.colors import Colors, color
from hermes_cli.cli_output import print_error, print_info, print_success, print_warning
from hermes_cli.secret_prompt import masked_secret_prompt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_DOCS_BASE = "https://hermes-agent.nousresearch.com/docs"
_BRACKETED_PASTE_PATTERN = re.compile(r"\x1b\[\s*200~|\x1b\[\s*201~")


def print_header(title: str, *, gap: bool = False):
    """Print a section header (``gap`` adds an extra blank line before it)."""
    if gap:
        print()
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def _info(*lines: str | None) -> None:
    """print_info each line in order; ``None`` emits a bare blank ``print()``."""
    for line in lines:
        print() if line is None else print_info(line)


def _sub_dict(parent: dict, key: str) -> dict:
    """``parent[key]`` as a dict, replacing a missing or non-dict value with ``{}``."""
    child = parent.get(key)
    if not isinstance(child, dict):
        child = parent[key] = {}
    return child


def _current_reasoning_effort(config: dict) -> str:
    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    return ""


def _set_reasoning_effort(config: dict, effort: str) -> None:
    _sub_dict(config, "agent")["reasoning_effort"] = effort


def is_interactive_stdin() -> bool:
    """Return True when stdin looks like a usable interactive TTY."""
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def print_noninteractive_setup_guidance(reason: str | None = None) -> None:
    """Print guidance for headless/non-interactive setup flows."""
    print()
    print(color("⚕ Hermes Setup — Non-interactive mode", Colors.CYAN, Colors.BOLD))
    print()
    if reason:
        print_info(reason)
    _info("The interactive wizard cannot be used here.", None,
          "Configure Hermes using environment variables or config commands:",
          "  hermes config set model.provider custom",
          "  hermes config set model.base_url http://localhost:8080/v1",
          "  hermes config set model.default your-model-name", None,
          "Or set OPENROUTER_API_KEY / OPENAI_API_KEY in your environment.",
          "Run 'hermes setup' in an interactive terminal to use the full wizard.", None)


def _sanitize_pasted_input(value: str) -> str:
    """Strip terminal bracketed-paste control markers from pasted text."""
    return _BRACKETED_PASTE_PATTERN.sub("", value) if isinstance(value, str) and value else value


def prompt(question: str, default: str = None, password: bool = False) -> str:
    """Prompt for input with optional default."""
    display = color(f"{question} [{default}]: " if default else f"{question}: ", Colors.YELLOW)
    try:
        if password:
            value = masked_secret_prompt(display)
        else:
            from hermes_cli.cli_output import line_input
            value = line_input(display)
        return _sanitize_pasted_input(value).strip() or default or ""
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)


# ── Setup navigation (Escape cancels, Left arrow goes back): a ContextVar state machine shared
# with the curses menus. ──


class _SetupControlFlow(BaseException):
    """Bypass provider error handlers that intentionally catch ``Exception`` so navigation reaches
    the outer state machine unchanged and it can replay the prior prompt."""


class _SetupCancelled(_SetupControlFlow):
    """Internal control flow for cancelling the interactive setup wizard."""


class _SetupGoBack(_SetupControlFlow):
    """Internal control flow for returning to an earlier setup choice."""

    def __init__(self, prompt_index: int):
        super().__init__(prompt_index)
        self.prompt_index = prompt_index


class _SetupNavigationState:
    """Per-invocation navigation state for the synchronous setup wizard."""

    def __init__(self, *, section_index: int = -1, prompt_index: int = 0):
        self.reset(section_index)
        self.prompt_index = prompt_index

    def reset(self, section_index: int = -1, replay: list | None = None) -> None:
        """Rewind per-section counters (entering a section, or leaving the wizard)."""
        self.section_index = section_index
        self.prompt_index = 0
        self.active_prompt_index = -1
        self.resolved_choices: list[object] = []
        self.replay_choices: list[object] = copy.deepcopy(replay or [])


_SETUP_NAVIGATION: ContextVar[_SetupNavigationState | None] = ContextVar("hermes_setup_navigation", default=None)


def _handle_setup_menu_navigation(event: MenuNavigationEvent, value: object = None) -> MenuNavigationStart | None:
    """Translate shared curses menu events into setup control flow."""
    state = _SETUP_NAVIGATION.get()
    if state is None:
        return None
    if event is MenuNavigationEvent.BEGIN:
        if state.section_index < 0:
            state.active_prompt_index = -1
            return MenuNavigationStart()
        idx = state.active_prompt_index = state.prompt_index
        state.prompt_index += 1
        allow_back = state.section_index > 0 or idx > 0
        if idx < len(state.replay_choices):
            return MenuNavigationStart(allow_back=allow_back, replay_value=copy.deepcopy(state.replay_choices[idx]))
        return MenuNavigationStart(allow_back=allow_back)
    if event is MenuNavigationEvent.RESOLVE:
        prompt_index = state.active_prompt_index
        if prompt_index >= 0:  # replace this answer and drop every later one
            state.resolved_choices[prompt_index:] = [copy.deepcopy(value)]
        return None
    if event is MenuNavigationEvent.CANCEL:
        raise _SetupCancelled()
    if event is MenuNavigationEvent.BACK:
        raise _SetupGoBack(state.active_prompt_index)
    return None


@contextmanager
def _setup_navigation_scope():
    """Install and reliably restore the setup menu navigation context."""
    from hermes_cli.curses_ui import reset_menu_navigation_handler, set_menu_navigation_handler
    token = _SETUP_NAVIGATION.set(_SetupNavigationState())
    menu_token = set_menu_navigation_handler(_handle_setup_menu_navigation)
    try:
        yield
    finally:
        reset_menu_navigation_handler(menu_token)
        _SETUP_NAVIGATION.reset(token)


def _run_setup_steps(steps: list[tuple[str, Callable[[], None]]]) -> None:
    """Run setup sections with left-arrow navigation: at a section's first choice it returns to
    the previous section; from a later choice it replays earlier selections invisibly and reopens
    only the preceding prompt."""
    state = _SETUP_NAVIGATION.get()
    section_index = 0
    answers_by_section: dict[int, list[object]] = {}
    replay_by_section: dict[int, list[object]] = {}

    def _record_answers() -> None:
        if state is not None:
            answers_by_section[section_index] = copy.deepcopy(state.resolved_choices)

    try:
        while section_index < len(steps):
            label, action = steps[section_index]
            if state is not None:
                state.reset(section_index, replay_by_section.pop(section_index, []))
            try:
                action()
            except _SetupGoBack as navigation:
                _record_answers()
                if navigation.prompt_index > 0:
                    previous_index = section_index
                    target_prompt = navigation.prompt_index - 1
                else:
                    previous_index = max(0, section_index - 1)
                    target_prompt = max(0, len(answers_by_section.get(previous_index, [])) - 1)
                replay_by_section[previous_index] = copy.deepcopy(
                    answers_by_section.get(previous_index, [])[:target_prompt])
                print()
                if previous_index == section_index:
                    print_info(f"Returning to the previous choice in {label}...")
                else:
                    print_info(f"Returning to {steps[previous_index][0]}...")
                section_index = previous_index
                continue
            _record_answers()
            section_index += 1
    finally:
        if state is not None:
            state.reset()


def run_setup_action_with_navigation(
    label: str, action: Callable[[], None], *, cancelled_message: str = "Setup cancelled."
) -> None:
    """Run a setup-style menu flow with Escape and nested Left navigation — for commands such as
    ``hermes model`` that use the wizard's pickers outside ``run_setup_wizard``."""
    with _setup_navigation_scope():
        try:
            _run_setup_steps([(label, action)])
        except _SetupCancelled:
            _info(None, cancelled_message)


# ── Prompt primitives ──


def _curses_prompt_choice(question: str, choices: list, default: int = 0, description: str | None = None) -> int:
    """Single-select menu using curses. Delegates to curses_radiolist."""
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(question, choices, selected=default, cancel_returns=-1, description=description)


def prompt_choice(question: str, choices: list, default: int = 0, description: str | None = None) -> int:
    """Prompt for a choice from a list with arrow key navigation. Escape cancels an active setup
    wizard; outside setup it keeps the default (the curses component owns its own numbered
    fallback, so a cancel result must never open another prompt). Ctrl+C exits the wizard."""
    idx = _curses_prompt_choice(question, choices, default, description=description)
    if idx < 0:
        return default
    if idx == default:
        _info("  Skipped (keeping current)", None)
        return default
    print()
    return idx


def is_noninteractive() -> bool:
    """True when no human is available to answer a prompt: the dashboard/desktop spawn CLI actions
    with ``stdin=DEVNULL`` and ``HERMES_NONINTERACTIVE=1`` (``hermes_cli/web_server.py``), where a
    prompt that aborts on EOF would kill the spawned action — callers fall back to their default."""
    return os.environ.get("HERMES_NONINTERACTIVE", "").strip().lower() in {"1", "true", "yes", "on"}


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no. Ctrl+C exits; empty input, ``HERMES_NONINTERACTIVE=1`` or a
    closed/redirected stdin return ``default`` instead of aborting the whole process."""
    if is_noninteractive():
        return default
    # Inside setup, route binary selections through the curses menu so ESC and left-arrow work
    # consistently; every other caller keeps the traditional line prompt.
    if _SETUP_NAVIGATION.get() is not None:
        return _curses_prompt_choice(question, ["Yes", "No"], 0 if default else 1) == 0
    default_str = "Y/n" if default else "y/N"
    while True:
        try:
            value = input(color(f"{question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
        except KeyboardInterrupt:
            print()
            sys.exit(1)
        except EOFError:
            # No stdin (closed/redirected, e.g. stdin=DEVNULL): accept the default so the caller
            # proceeds unattended instead of failing the whole command.
            print()
            return default
        answer = {"": default, "y": True, "yes": True, "n": False, "no": False}.get(value)
        if answer is not None:
            return answer
        print_error("Please enter 'y' or 'n'")


def prompt_checklist(title: str, items: list, pre_selected: list = None) -> list:
    """Multi-select checklist; returns the sorted indices of selected items. ``pre_selected``
    start checked; Space toggles, Enter confirms, cancel keeps the pre-selection."""
    from hermes_cli.curses_ui import curses_checklist
    pre = set(pre_selected or [])
    return sorted(curses_checklist(title, items, pre, cancel_returns=pre))


def _section_rule(title: str) -> None:
    """Blank-padded cyan ``─── title ───`` divider used by the key-entry screens."""
    print()
    print(color(f"  ─── {title} ───", Colors.CYAN))
    print()


def _prompt_api_key(var: dict):
    """Display a nicely formatted API key input screen for a single env var."""
    tools = var.get("tools", [])
    tools_str = ", ".join(tools[:3])
    if len(tools) > 3:
        tools_str += f", +{len(tools) - 3} more"
    _section_rule(var.get("description", var["name"]))
    if tools_str:
        print_info(f"  Enables: {tools_str}")
    if var.get("url"):
        print_info(f"  Get your key at: {var['url']}")
    print()
    _prompt_and_save_env_var(var, "  ✓ Saved", "  Skipped (configure later with 'hermes setup')")


def _prompt_and_save_env_var(var: dict, saved_msg: str, skipped_msg: str) -> None:
    """Prompt for one env-var value (masked when secret); persist and confirm, or report the skip."""
    value = prompt(f"  {var.get('prompt', var['name'])}", password=bool(var.get("password")))
    if value:
        save_env_value(var["name"], value)
        print_success(saved_msg)
    else:
        print_warning(skipped_msg)


def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _print_banner(*lines: str) -> None:
    """Print the magenta box banner: top border, the given body lines, bottom border."""
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    for line in lines:
        print(color(line, Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))


    # Web tools (Exa, Parallel, Firecrawl, Tavily, or Keenable)
    if subscription_features.web.managed_by_nous:
        tool_status.append(("Web Search & Extract (Nous subscription)", True, None))
    elif subscription_features.web.available:
        label = "Web Search & Extract"
        if subscription_features.web.current_provider:
            label = f"Web Search & Extract ({subscription_features.web.current_provider})"
        tool_status.append((label, True, None))
    else:
        tool_status.append(("Web Search & Extract", False, "EXA_API_KEY, PARALLEL_API_KEY, FIRECRAWL_API_KEY/FIRECRAWL_API_URL, TAVILY_API_KEY, KEENABLE_API_KEY, or SEARXNG_URL"))

    # Browser tools (local Chromium, Camofox, Browserbase, Browser Use, or Firecrawl)
    browser_provider = subscription_features.browser.current_provider
    if subscription_features.browser.managed_by_nous:
        tool_status.append(("Browser Automation (Nous Browser Use)", True, None))
    elif subscription_features.browser.available:
        label = "Browser Automation"
        if browser_provider:
            label = f"Browser Automation ({browser_provider})"
        tool_status.append((label, True, None))
    else:
        missing_browser_hint = "npm install -g agent-browser, set CAMOFOX_URL, or configure Browser Use or Browserbase"
        if browser_provider == "Browserbase":
            missing_browser_hint = (
                "npm install -g agent-browser and set "
                "BROWSERBASE_API_KEY/BROWSERBASE_PROJECT_ID"
            )
        elif browser_provider == "Browser Use":
            missing_browser_hint = (
                "npm install -g agent-browser and set BROWSER_USE_API_KEY"
            )
        elif browser_provider == "Camofox":
            missing_browser_hint = "CAMOFOX_URL"
        elif browser_provider == "Local browser":
            missing_browser_hint = (
                "npm install -g agent-browser && agent-browser install --with-deps"
            )
        tool_status.append(
            ("Browser Automation", False, missing_browser_hint)
        )

    # Image generation — FAL (direct or via Nous), or any plugin-registered
    # provider (OpenAI, etc.)
    if subscription_features.image_gen.managed_by_nous:
        tool_status.append(("Image Generation (Nous subscription)", True, None))
    elif subscription_features.image_gen.available:
        tool_status.append(("Image Generation", True, None))
    else:
        # Fall back to probing plugin-registered providers so OpenAI-only
        # setups don't show as "missing FAL_KEY".
        _img_backend = None
        try:
            from agent.image_gen_registry import list_providers
            from hermes_cli.plugins import _ensure_plugins_discovered

            _ensure_plugins_discovered()
            for _p in list_providers():
                if _p.name == "fal":
                    continue
                try:
                    if _p.is_available():
                        _img_backend = _p.display_name
                        break
                except Exception:
                    continue
        except Exception:
            pass
        if _img_backend:
            tool_status.append((f"Image Generation ({_img_backend})", True, None))
        else:
            tool_status.append(("Image Generation", False, "FAL_KEY or OPENAI_API_KEY"))

    # Video generation — opt-in via `hermes tools` → Video Generation.
    # Only show the row when a plugin reports available so we don't badger
    # users who don't care about video gen with a "missing" status line.
    if subscription_features.video_gen.managed_by_nous:
        tool_status.append(("Video Generation (FAL via Nous subscription)", True, None))
    else:
        try:
            from agent.video_gen_registry import list_providers as _list_video_providers
            from hermes_cli.plugins import _ensure_plugins_discovered as _ensure_plugins
            _ensure_plugins()
            _video_backend = None
            for _vp in _list_video_providers():
                try:
                    if _vp.is_available():
                        _video_backend = _vp.display_name
                        break
                except Exception:
                    continue
        except Exception:
            _video_backend = None
        if _video_backend:
            tool_status.append((f"Video Generation ({_video_backend})", True, None))

    # TTS — show configured provider
    tts_provider = cfg_get(config, "tts", "provider", default="edge")
    if subscription_features.tts.managed_by_nous:
        tool_status.append(("Text-to-Speech (OpenAI via Nous subscription)", True, None))
    elif tts_provider == "elevenlabs" and get_env_value("ELEVENLABS_API_KEY"):
        tool_status.append(("Text-to-Speech (ElevenLabs)", True, None))
    elif tts_provider == "openai" and (
        get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY")
    ):
        tool_status.append(("Text-to-Speech (OpenAI)", True, None))
    elif tts_provider == "minimax" and get_env_value("MINIMAX_API_KEY"):
        tool_status.append(("Text-to-Speech (MiniMax)", True, None))
    elif tts_provider == "mistral" and get_env_value("MISTRAL_API_KEY"):
        tool_status.append(("Text-to-Speech (Mistral Voxtral)", True, None))
    elif tts_provider == "gemini" and (get_env_value("GEMINI_API_KEY") or get_env_value("GOOGLE_API_KEY")):
        tool_status.append(("Text-to-Speech (Google Gemini)", True, None))
    elif tts_provider == "neutts":
        try:
            neutts_ok = importlib.util.find_spec("neutts") is not None
        except Exception:
            neutts_ok = False
        if neutts_ok:
            tool_status.append(("Text-to-Speech (NeuTTS local)", True, None))
        else:
            tool_status.append(("Text-to-Speech (NeuTTS — not installed)", False, "run 'hermes setup tts'"))
    elif tts_provider == "kittentts":
        try:
            kittentts_ok = importlib.util.find_spec("kittentts") is not None
        except Exception:
            kittentts_ok = False
        if kittentts_ok:
            tool_status.append(("Text-to-Speech (KittenTTS local)", True, None))
        else:
            tool_status.append(("Text-to-Speech (KittenTTS — not installed)", False, "run 'hermes setup tts'"))
    else:
        tool_status.append(("Text-to-Speech (Edge TTS)", True, None))

    # STT — show configured provider
    stt_provider = cfg_get(config, "stt", "provider", default="local") or "local"
    _stt_feature = subscription_features.features.get("stt")
    if _stt_feature is not None and _stt_feature.managed_by_nous:
        tool_status.append(("Speech-to-Text (OpenAI via Nous subscription)", True, None))
    elif stt_provider == "openai" and (
        get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY")
    ):
        tool_status.append(("Speech-to-Text (OpenAI)", True, None))
    elif stt_provider == "groq" and get_env_value("GROQ_API_KEY"):
        tool_status.append(("Speech-to-Text (Groq Whisper)", True, None))
    elif stt_provider == "elevenlabs" and get_env_value("ELEVENLABS_API_KEY"):
        tool_status.append(("Speech-to-Text (ElevenLabs Scribe)", True, None))
    elif stt_provider == "xai":
        tool_status.append(("Speech-to-Text (xAI)", True, None))
    elif stt_provider == "deepinfra" and get_env_value("DEEPINFRA_API_KEY"):
        tool_status.append(("Speech-to-Text (DeepInfra)", True, None))
    else:
        try:
            fw_ok = importlib.util.find_spec("faster_whisper") is not None
        except Exception:
            fw_ok = False
        if fw_ok:
            tool_status.append(("Speech-to-Text (Local Whisper)", True, None))
        else:
            tool_status.append(
                ("Speech-to-Text (Local Whisper — not installed)", False, "run 'hermes tools' → Speech-to-Text")
            )

    if subscription_features.modal.managed_by_nous:
        tool_status.append(("Modal Execution (Nous subscription)", True, None))
    elif cfg_get(config, "terminal", "backend") == "modal":
        if subscription_features.modal.direct_override:
            tool_status.append(("Modal Execution (direct Modal)", True, None))
        else:
            tool_status.append(("Modal Execution", False, "run 'hermes setup terminal'"))
    elif managed_nous_tools_enabled() and subscription_features.nous_auth_present:
        tool_status.append(("Modal Execution (optional via Nous subscription)", True, None))

    # Home Assistant
    if get_env_value("HASS_TOKEN"):
        tool_status.append(("Smart Home (Home Assistant)", True, None))

    # Spotify (OAuth via hermes auth spotify — check auth.json, not env vars)
    try:
        from hermes_cli.auth import get_provider_auth_state
        _spotify_state = get_provider_auth_state("spotify") or {}
        if _spotify_state.get("access_token") or _spotify_state.get("refresh_token"):
            tool_status.append(("Spotify (PKCE OAuth)", True, None))
    except Exception:
        pass

    # Skills Hub
    if get_env_value("GITHUB_TOKEN"):
        tool_status.append(("Skills Hub (GitHub)", True, None))
    else:
        tool_status.append(("Skills Hub (GitHub)", False, "GITHUB_TOKEN"))

    # Terminal (always available if system deps met)
    tool_status.append(("Terminal/Commands", True, None))

    # Task planning (always available, in-memory)
    tool_status.append(("Task Planning (todo)", True, None))

    # Skills (always available -- bundled skills + user-created skills)
    tool_status.append(("Skills (view, create, edit)", True, None))

    # Print status
    available_count = sum(1 for _, avail, _ in tool_status if avail)
    total_count = len(tool_status)

    print_info(f"{available_count}/{total_count} tool categories available:")
    print()

    for name, available, missing_var in tool_status:
        if available:
            print(f"   {color('✓', Colors.GREEN)} {name}")
        else:
            print(
                f"   {color('✗', Colors.RED)} {name} {color(f'(missing {missing_var})', Colors.DIM)}"
            )

    print()

    disabled_tools = [(name, var) for name, avail, var in tool_status if not avail]
    if disabled_tools:
        print_warning(
            "Some tools are disabled. Run 'hermes setup tools' to configure them,"
        )
        from hermes_constants import display_hermes_home as _dhh
        print_warning(f"or edit {_dhh()}/.env directly to add the missing API keys.")
        print()

    # Done banner
    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐", Colors.GREEN
        )
    )
    print(
        color(
            "│              ✓ Setup Complete!                          │", Colors.GREEN
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘", Colors.GREEN
        )
    )
    print()

    # Show file locations prominently
    from hermes_constants import display_hermes_home as _dhh
    print(color(f"📁 All your files are in {_dhh()}/:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('Settings:', Colors.YELLOW)}  {get_config_path()}")
    print(f"   {color('API Keys:', Colors.YELLOW)}  {get_env_path()}")
    print(
        f"   {color('Data:', Colors.YELLOW)}      {hermes_home}/cron/, sessions/, logs/"
    )
    print()

    print(color("─" * 60, Colors.DIM))
    print()
    print(color("📝 To edit your configuration:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('hermes setup', Colors.GREEN)}          Re-run the full wizard")
    print(f"   {color('hermes setup model', Colors.GREEN)}    Change model/provider")
    print(f"   {color('hermes setup terminal', Colors.GREEN)} Change terminal backend")
    print(f"   {color('hermes setup gateway', Colors.GREEN)}  Configure messaging")
    print(f"   {color('hermes setup tools', Colors.GREEN)}    Configure tool providers")
    print()
    print(f"   {color('hermes config', Colors.GREEN)}         View current settings")
    print(
        f"   {color('hermes config edit', Colors.GREEN)}    Open config in your editor"
    )
    print(f"   {color('hermes config set <key> <value>', Colors.GREEN)}")
    print("                          Set a specific value")
    print()
    print("   Or edit the files directly:")
    print(f"   {color(f'nano {get_config_path()}', Colors.DIM)}")
    print(f"   {color(f'nano {get_env_path()}', Colors.DIM)}")
    print()

    print(color("─" * 60, Colors.DIM))
    print()
    print(color("🚀 Ready to go!", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('hermes', Colors.GREEN)}              Start chatting")
    print(f"   {color('hermes gateway', Colors.GREEN)}      Start messaging gateway")
    print(f"   {color('hermes doctor', Colors.GREEN)}       Check for issues")
    print()


def _prompt_container_resources(config: dict):
    """Prompt for container resource settings (Docker, Singularity, Modal, Daytona)."""
    terminal = config.setdefault("terminal", {})

    print()
    print_info("Container Resource Settings:")

    # Persistence
    current_persist = terminal.get("container_persistent", True)
    persist_label = "yes" if current_persist else "no"
    print_info("  Persistent filesystem keeps files between sessions.")
    print_info("  Set to 'no' for ephemeral sandboxes that reset each time.")
    persist_str = prompt(
        "  Persist filesystem across sessions? (yes/no)", persist_label
    )
    terminal["container_persistent"] = persist_str.lower() in {"yes", "true", "y", "1"}

    # CPU
    current_cpu = terminal.get("container_cpu", 1)
    cpu_str = prompt("  CPU cores", str(current_cpu))
    try:
        terminal["container_cpu"] = float(cpu_str)
    except ValueError:
        pass

    # Memory
    current_mem = terminal.get("container_memory", 5120)
    mem_str = prompt("  Memory in MB (5120 = 5GB)", str(current_mem))
    try:
        terminal["container_memory"] = int(mem_str)
    except ValueError:
        pass

    # Disk
    current_disk = terminal.get("container_disk", 51200)
    disk_str = prompt("  Disk in MB (51200 = 50GB)", str(current_disk))
    try:
        terminal["container_disk"] = int(disk_str)
    except ValueError:
        pass


def _prompt_vercel_sandbox_settings(config: dict):
    """Prompt for Vercel Sandbox settings without exposing unsupported disk sizing."""
    terminal = config.setdefault("terminal", {})

    print()
    print_info("Vercel Sandbox settings:")
    print_info("  Filesystem persistence uses Vercel snapshots.")
    print_info("  Snapshots restore files only; live processes do not continue after sandbox recreation.")

    from tools.terminal_tool import _SUPPORTED_VERCEL_RUNTIMES

    current_runtime = terminal.get("vercel_runtime") or "node24"
    supported_label = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
    runtime = prompt(f"  Runtime ({supported_label})", current_runtime).strip() or current_runtime
    if runtime not in _SUPPORTED_VERCEL_RUNTIMES:
        print_warning(f"Unsupported Vercel runtime '{runtime}', keeping {current_runtime}.")
        runtime = current_runtime if current_runtime in _SUPPORTED_VERCEL_RUNTIMES else "node24"
    terminal["vercel_runtime"] = runtime
    save_env_value("TERMINAL_VERCEL_RUNTIME", runtime)

    current_persist = terminal.get("container_persistent", True)
    persist_label = "yes" if current_persist else "no"
    terminal["container_persistent"] = prompt(
        "  Persist filesystem with snapshots? (yes/no)", persist_label
    ).lower() in {"yes", "true", "y", "1"}

    current_cpu = terminal.get("container_cpu", 1)
    cpu_str = prompt("  CPU cores", str(current_cpu))
    try:
        terminal["container_cpu"] = float(cpu_str)
    except ValueError:
        pass

    current_mem = terminal.get("container_memory", 5120)
    mem_str = prompt("  Memory in MB (5120 = 5GB)", str(current_mem))
    try:
        terminal["container_memory"] = int(mem_str)
    except ValueError:
        pass

    if terminal.get("container_disk", 51200) not in {0, 51200}:
        print_warning("Vercel Sandbox does not support custom disk sizing; resetting container_disk to 51200.")
    terminal["container_disk"] = 51200

    print()
    print_info("Vercel authentication:")
    print_info("  Use a long-lived Vercel access token plus project/team IDs.")
    linked_project = _read_nearest_vercel_project()
    if linked_project:
        print_info("  Found defaults in nearest .vercel/project.json.")

    remove_env_value("VERCEL_OIDC_TOKEN")
    token = prompt("    Vercel access token", get_env_value("VERCEL_TOKEN") or "", password=True)
    project = prompt(
        "    Vercel project ID",
        get_env_value("VERCEL_PROJECT_ID") or linked_project.get("projectId", ""),
    )
    team = prompt(
        "    Vercel team ID",
        get_env_value("VERCEL_TEAM_ID") or linked_project.get("orgId", ""),
    )
    if token:
        save_env_value("VERCEL_TOKEN", token)
    if project:
        save_env_value("VERCEL_PROJECT_ID", project)
    if team:
        save_env_value("VERCEL_TEAM_ID", team)


def _read_nearest_vercel_project(start: Path | None = None) -> dict[str, str]:
    """Read project/team defaults from the nearest Vercel link file."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        project_file = directory / ".vercel" / "project.json"
        if not project_file.exists():
            continue
        try:
            data = json.loads(project_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            key: value
            for key, value in {
                "projectId": data.get("projectId"),
                "orgId": data.get("orgId"),
            }.items()
            if isinstance(value, str) and value.strip()
        }
    return {}


# Tool categories and provider config are now in tools_config.py (shared
# between `hermes tools` and `hermes setup tools`).


# =============================================================================
# Section 1: Model & Provider Configuration
# =============================================================================



def setup_model_provider(config: dict, *, quick: bool = False):
    """Configure the inference provider and default model via the ``hermes model`` flow (one code
    path — any provider added there is available here). *quick* is accepted for the first-time
    quick setup caller; rotation, vision and TTS keep safe defaults either way."""
    from hermes_cli.config import load_config, save_config
    print_header("Inference Provider")
    _info("Choose how to connect to your main chat model.",
          f"   Guide: {_DOCS_BASE}/integrations/providers", None)
    from hermes_cli.main import select_provider_and_model
    try:
        select_provider_and_model()
    except (SystemExit, KeyboardInterrupt):
        _info(None, "Provider setup skipped.")
    except Exception as exc:
        logger.debug("select_provider_and_model error during setup: %s", exc)
        print_warning(f"Provider setup encountered an error: {exc}")
        print_info("You can try again later with: hermes model")

    # Re-sync from disk in place: cmd_model saved via its own load/save cycle and the wizard's
    # final save_config(config) must not clobber it with stale values. Rotation, vision and TTS
    # keep safe defaults (configure via `hermes auth add` / `hermes setup tts`).
    config.clear()
    config.update(load_config())
    save_config(config)


# =============================================================================
# Section 1b: TTS Provider Configuration


def _check_espeak_ng() -> bool:
    """Check if espeak-ng is installed."""
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _install_neutts_deps() -> bool:
    """Install NeuTTS dependencies with user approval. Returns True on success."""
    import subprocess
    import sys

    # Check espeak-ng
    if not _check_espeak_ng():
        print()
        print_warning("NeuTTS requires espeak-ng for phonemization.")
        if sys.platform == "darwin":
            print_info("Install with: brew install espeak-ng")
        elif sys.platform == "win32":
            print_info("Install with: choco install espeak-ng")
        else:
            print_info("Install with: sudo apt install espeak-ng")
        print()
        if prompt_yes_no("Install espeak-ng now?", True):
            try:
                if sys.platform == "darwin":
                    subprocess.run(["brew", "install", "espeak-ng"], check=True)
                elif sys.platform == "win32":
                    subprocess.run(["choco", "install", "espeak-ng", "-y"], check=True)
                else:
                    subprocess.run(["sudo", "apt", "install", "-y", "espeak-ng"], check=True)
                print_success("espeak-ng installed")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print_warning(f"Could not install espeak-ng automatically: {e}")
                print_info("Please install it manually and re-run setup.")
                return False
        else:
            print_warning("espeak-ng is required for NeuTTS. Install it manually before using NeuTTS.")

    # Install neutts Python package
    print()
    print_info("Installing neutts Python package...")
    print_info("This will also download the TTS model (~300MB) on first use.")
    print()

    # Route through the canonical uv → pip → ensurepip ladder so pip-less
    # venvs (Ubuntu 25.10 `python -m venv`, `uv venv`) work out of the box.
    from hermes_cli.tools_config import _pip_install

    try:
        result = _pip_install(["-U", "neutts[all]", "--quiet"], timeout=300)
    except Exception as e:
        print_error(f"Failed to install neutts: {e}")
        print_info("Try manually: uv pip install -U 'neutts[all]'")
        return False
    if result.returncode == 0:
        print_success("neutts installed successfully")
        return True
    err = (result.stderr or "").strip()
    print_error(f"Failed to install neutts: {err[:300] if err else 'install failed'}")
    print_info("Try manually: uv pip install -U 'neutts[all]'")
    return False


def _install_kittentts_deps() -> bool:
    """Install KittenTTS dependencies with user approval. Returns True on success."""

    wheel_url = (
        "https://github.com/KittenML/KittenTTS/releases/download/"
        "0.8.1/kittentts-0.8.1-py3-none-any.whl"
    )
    print()
    print_info("Installing kittentts Python package (~25-80MB model downloaded on first use)...")
    print()

    from hermes_cli.tools_config import _pip_install

    try:
        result = _pip_install(["-U", wheel_url, "soundfile", "--quiet"], timeout=300)
    except Exception as e:
        print_error(f"Failed to install kittentts: {e}")
        print_info(f"Try manually: uv pip install -U '{wheel_url}' soundfile")
        return False
    if result.returncode == 0:
        print_success("kittentts installed successfully")
        return True
    err = (result.stderr or "").strip()
    print_error(f"Failed to install kittentts: {err[:300] if err else 'install failed'}")
    print_info(f"Try manually: uv pip install -U '{wheel_url}' soundfile")
    return False


def _xai_oauth_logged_in_for_setup() -> bool:
    """True iff xAI Grok OAuth credentials are already stored locally.

    Lets TTS / STT setup skip the API-key prompt for users who logged in
    through ``hermes model`` -> xAI Grok OAuth (SuperGrok / Premium+).
    """
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status

        return bool(get_xai_oauth_auth_status().get("logged_in"))
    except Exception:
        return False


def _run_xai_oauth_login_from_setup() -> bool:
    """Run the xAI Grok OAuth device-code login from inside the setup wizard.

    Saves OAuth tokens only. Does **not** switch the active inference
    provider or rewrite ``model.provider`` — callers (TTS setup, tools
    config) only need credentials for side tools.

    Returns True on success, False on any failure (the caller falls back
    to whatever the user picked next, e.g. Edge TTS).
    """
    try:
        from hermes_cli.auth import (
            _is_remote_session,
            _save_xai_oauth_tokens,
            _xai_oauth_device_code_login,
            unsuppress_credential_source,
        )
    except Exception as exc:
        print_warning(f"xAI Grok OAuth helpers unavailable: {exc}")
        return False

    open_browser = not _is_remote_session()
    print()
    print_info("Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    try:
        creds = _xai_oauth_device_code_login(open_browser=open_browser)
        _save_xai_oauth_tokens(
            creds["tokens"],
            discovery=creds.get("discovery"),
            redirect_uri=creds.get("redirect_uri", ""),
            last_refresh=creds.get("last_refresh"),
            auth_mode="oauth_device_code",
            set_active=False,
        )
        # Mirror model/dashboard re-login: clear device_code suppression so
        # the pool can seed from the singleton after a prior `auth remove`.
        unsuppress_credential_source("xai-oauth", "device_code")
        return True
    except Exception as exc:
        print_warning(f"xAI Grok OAuth login failed: {exc}")
        return False


def _setup_tts_provider(config: dict):
    """Interactive TTS provider selection with install flow for NeuTTS."""
    tts_config = config.get("tts", {})
    current_provider = tts_config.get("provider", "edge")
    subscription_features = get_nous_subscription_features(config)

    provider_labels = {
        "edge": "Edge TTS",
        "elevenlabs": "ElevenLabs",
        "openai": "OpenAI TTS",
        "xai": "xAI TTS",
        "minimax": "MiniMax TTS",
        "mistral": "Mistral Voxtral TTS",
        "gemini": "Google Gemini TTS",
        "neutts": "NeuTTS",
        "kittentts": "KittenTTS",
    }
    current_label = provider_labels.get(current_provider, current_provider)

    print()
    print_header("Text-to-Speech Provider (optional)")
    print_info(f"Current: {current_label}")
    print()

    choices = []
    providers = []
    if managed_nous_tools_enabled() and subscription_features.nous_auth_present:
        choices.append("Nous Subscription (managed OpenAI TTS, billed to your subscription)")
        providers.append("nous-openai")
    choices.extend(
        [
            "Edge TTS (free, cloud-based, no setup needed)",
            "ElevenLabs (premium quality, needs API key)",
            "OpenAI TTS (good quality, needs API key)",
            "xAI TTS (Grok voices — OAuth login or API key)",
            "MiniMax TTS (high quality with voice cloning, needs API key)",
            "Mistral Voxtral TTS (multilingual, native Opus, needs API key)",
            "Google Gemini TTS (30 prebuilt voices, prompt-controllable, needs API key)",
            "NeuTTS (local on-device, free, ~300MB model download)",
            "KittenTTS (local on-device, free, lightweight ~25-80MB ONNX)",
        ]
    )
    providers.extend(["edge", "elevenlabs", "openai", "xai", "minimax", "mistral", "gemini", "neutts", "kittentts"])
    choices.append(f"Keep current ({current_label})")
    keep_current_idx = len(choices) - 1
    idx = prompt_choice("Select TTS provider:", choices, keep_current_idx)

    if idx == keep_current_idx:
        return

    selected = providers[idx]
    selected_via_nous = selected == "nous-openai"
    if selected == "nous-openai":
        selected = "openai"
        print_info("OpenAI TTS will use the managed Nous gateway and bill to your subscription.")
        if get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY"):
            print_warning(
                "Direct OpenAI credentials are still configured and may take precedence until removed from ~/.hermes/.env."
            )

    if selected == "neutts":
        # Check if already installed
        try:
            already_installed = importlib.util.find_spec("neutts") is not None
        except Exception:
            already_installed = False

        if already_installed:
            print_success("NeuTTS is already installed")
        else:
            print()
            print_info("NeuTTS requires:")
            print_info("  • Python package: neutts (~50MB install + ~300MB model on first use)")
            print_info("  • System package: espeak-ng (phonemizer)")
            print()
            if prompt_yes_no("Install NeuTTS dependencies now?", True):
                if not _install_neutts_deps():
                    print_warning("NeuTTS installation incomplete. Falling back to Edge TTS.")
                    selected = "edge"
            else:
                print_info("Skipping install. Set tts.provider to 'neutts' after installing manually.")
                selected = "edge"

    elif selected == "elevenlabs":
        existing = get_env_value("ELEVENLABS_API_KEY")
        if not existing:
            print()
            api_key = prompt("ElevenLabs API key", password=True)
            if api_key:
                save_env_value("ELEVENLABS_API_KEY", api_key)
                print_success("ElevenLabs API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "openai" and not selected_via_nous:
        existing = get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY")
        if not existing:
            print()
            api_key = prompt("OpenAI API key for TTS", password=True)
            if api_key:
                save_env_value("VOICE_TOOLS_OPENAI_KEY", api_key)
                print_success("OpenAI TTS API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "xai":
        # Resolution order: existing OAuth tokens (free for SuperGrok subscribers
        # via the Hermes auth store) > existing XAI_API_KEY > prompt the user.
        # When neither is configured, offer both options instead of forcing the
        # API-key path — xAI TTS works fine with OAuth bearer tokens too.
        oauth_logged_in = _xai_oauth_logged_in_for_setup()
        existing_api_key = get_env_value("XAI_API_KEY")

        if oauth_logged_in:
            print_success(
                "xAI TTS will use your xAI Grok OAuth (SuperGrok / Premium+) "
                "credentials"
            )
        elif existing_api_key:
            print_success("xAI TTS will use your existing XAI_API_KEY")
        else:
            print()
            choice_idx = prompt_choice(
                "How do you want xAI TTS to authenticate?",
                choices=[
                    "Sign in with xAI Grok OAuth (SuperGrok / Premium+) — browser login",
                    "Paste an xAI API key (console.x.ai)",
                    "Skip → fallback to Edge TTS",
                ],
                default=0,
            )
            if choice_idx == 0:
                if _run_xai_oauth_login_from_setup():
                    print_success(
                        "Logged in — xAI TTS will use these OAuth credentials"
                    )
                else:
                    print_warning(
                        "xAI Grok OAuth login did not complete. "
                        "Falling back to Edge TTS."
                    )
                    selected = "edge"
            elif choice_idx == 1:
                api_key = prompt("xAI API key for TTS", password=True)
                if api_key:
                    save_env_value("XAI_API_KEY", api_key)
                    print_success("xAI TTS API key saved")
                else:
                    from hermes_constants import display_hermes_home as _dhh
                    print_warning(
                        "No xAI API key provided for TTS. Configure XAI_API_KEY "
                        f"via hermes setup model or {_dhh()}/.env to use xAI TTS. "
                        "Falling back to Edge TTS."
                    )
                    selected = "edge"
            else:
                print_warning("xAI TTS skipped. Falling back to Edge TTS.")
                selected = "edge"

        if selected == "xai":
            print()
            voice_id = prompt("xAI voice_id (Enter for 'eve', or paste a custom voice ID)")
            if voice_id and voice_id.strip():
                config.setdefault("tts", {}).setdefault("xai", {})["voice_id"] = voice_id.strip()
                print_success(f"xAI voice_id set to: {voice_id.strip()}")


    elif selected == "minimax":
        existing = get_env_value("MINIMAX_API_KEY")
        if not existing:
            print()
            api_key = prompt("MiniMax API key for TTS", password=True)
            if api_key:
                save_env_value("MINIMAX_API_KEY", api_key)
                print_success("MiniMax TTS API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "mistral":
        existing = get_env_value("MISTRAL_API_KEY")
        if not existing:
            print()
            api_key = prompt("Mistral API key for TTS", password=True)
            if api_key:
                save_env_value("MISTRAL_API_KEY", api_key)
                print_success("Mistral TTS API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "gemini":
        existing = get_env_value("GEMINI_API_KEY") or get_env_value("GOOGLE_API_KEY")
        if not existing:
            print()
            print_info("Get a free API key at https://aistudio.google.com/app/apikey")
            api_key = prompt("Gemini API key for TTS", password=True)
            if api_key:
                save_env_value("GEMINI_API_KEY", api_key)
                print_success("Gemini TTS API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "kittentts":
        # Check if already installed
        try:
            already_installed = importlib.util.find_spec("kittentts") is not None
        except Exception:
            already_installed = False

        if already_installed:
            print_success("KittenTTS is already installed")
        else:
            print()
            print_info("KittenTTS is lightweight (~25-80MB, CPU-only, no API key required).")
            print_info("Voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo")
            print()
            if prompt_yes_no("Install KittenTTS now?", True):
                if not _install_kittentts_deps():
                    print_warning("KittenTTS installation incomplete. Falling back to Edge TTS.")
                    selected = "edge"
            else:
                print_info("Skipping install. Set tts.provider to 'kittentts' after installing manually.")
                selected = "edge"

    # Save the selection
    if "tts" not in config:
        config["tts"] = {}
    config["tts"]["provider"] = selected
    save_config(config)
    print_success(f"TTS provider set to: {provider_labels.get(selected, selected)}")


def setup_tts(config: dict):
    """Standalone TTS setup (for 'hermes setup tts')."""
    _setup_tts_provider(config)


# =============================================================================
# Section 2: Terminal Backend Configuration
# =============================================================================


def setup_terminal_backend(config: dict):
    """Configure the terminal execution backend."""
    import platform as _platform
    print_header("Terminal Backend")
    print_info("Choose where Hermes runs shell commands and code.")
    print_info("This affects tool execution, file access, and isolation.")
    print_info(f"   Guide: {_DOCS_BASE}/user-guide/configuration#terminal-backend-configuration")
    print()

    current_backend = cfg_get(config, "terminal", "backend", default="local")
    is_linux = _platform.system() == "Linux"

    # Build backend choices with descriptions
    terminal_choices = [
        "Local - run directly on this machine (default)",
        "Docker - isolated container with configurable resources",
        "Modal - serverless cloud sandbox",
        "SSH - run on a remote machine",
        "Daytona - persistent cloud development environment",
        "Vercel Sandbox - cloud microVM with snapshot filesystem persistence",
    ]
    idx_to_backend = {0: "local", 1: "docker", 2: "modal", 3: "ssh", 4: "daytona", 5: "vercel_sandbox"}
    backend_to_idx = {"local": 0, "docker": 1, "modal": 2, "ssh": 3, "daytona": 4, "vercel_sandbox": 5}

    next_idx = 6
    if is_linux:
        terminal_choices.append("Singularity/Apptainer - HPC-friendly container")
        idx_to_backend[next_idx] = "singularity"
        backend_to_idx["singularity"] = next_idx
        next_idx += 1

    # Plugin-registered terminal backends (standalone plugin repos installed
    # under ~/.hermes/plugins/). Fail-soft: a broken plugin must not take the
    # setup wizard down.
    plugin_backend_names = []
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()  # idempotent — plugin state may not be loaded yet
        from agent.terminal_env_registry import list_providers

        for _provider in list_providers():
            _pname = _provider.name.strip().lower()
            terminal_choices.append(f"{_provider.display_name} - {_provider.description}")
            idx_to_backend[next_idx] = _pname
            backend_to_idx[_pname] = next_idx
            plugin_backend_names.append(_pname)
            next_idx += 1
    except Exception:
        pass

    # Add keep current option
    keep_current_idx = next_idx
    terminal_choices.append(f"Keep current ({current_backend})")
    idx_to_backend[keep_current_idx] = current_backend

    terminal_idx = prompt_choice(
        "Select terminal backend:", terminal_choices, keep_current_idx
    )

    selected_backend = idx_to_backend.get(terminal_idx)

    if terminal_idx == keep_current_idx:
        print_info(f"Keeping current backend: {current_backend}")
        return

    config.setdefault("terminal", {})["backend"] = selected_backend

    if selected_backend == "local":
        print_success("Terminal backend: Local")
        print_info("Commands run directly on this machine.")
        # Gateway working directory defaults to home; sudo stays off. Both are
        # configurable later via `hermes setup terminal` / config.yaml.
        config["terminal"].setdefault("cwd", str(Path.home()))

    elif selected_backend == "docker":
        print_success("Terminal backend: Docker")

        # Check if Docker is available
        docker_bin = shutil.which("docker")
        if not docker_bin:
            print_warning("Docker not found in PATH!")
            print_info("Install Docker: https://docs.docker.com/get-docker/")
        else:
            print_info(f"Docker found: {docker_bin}")

        # Image and resource limits use defaults; tune via `hermes setup terminal`.
        config["terminal"].setdefault(
            "docker_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )
        print()
        print_info("Docker sandboxes can be protected with the egress credential firewall.")
        print_info(
            "It routes sandbox traffic through iron-proxy so containers receive "
            "proxy tokens instead of real API keys."
        )
        print_info(
            "   Docker only for now; Modal, SSH, Daytona, and Singularity are not wired yet."
        )
        if prompt_yes_no("  Enable egress firewall for Docker sandboxes?", False):
            proxy_cfg = config.setdefault("proxy", {})
            proxy_cfg["enabled"] = True
            proxy_cfg.setdefault("enforce_on_docker", True)
            print_success("Egress firewall enabled in config")
            print_info(
                "Run `hermes egress setup` then `hermes egress start` to mint "
                "tokens and launch the proxy."
            )
        else:
            print_info(
                "Skipping egress firewall. You can enable it later with `hermes egress setup`."
            )

    elif selected_backend == "singularity":
        print_success("Terminal backend: Singularity/Apptainer")

        # Check if singularity/apptainer is available
        sing_bin = shutil.which("apptainer") or shutil.which("singularity")
        if not sing_bin:
            print_warning("Singularity/Apptainer not found in PATH!")
            print_info(
                "Install: https://apptainer.org/docs/admin/main/installation.html"
            )
        else:
            print_info(f"Found: {sing_bin}")

        # Image and resource limits use defaults; tune via `hermes setup terminal`.
        config["terminal"].setdefault(
            "singularity_image",
            "docker://nikolaik/python-nodejs:python3.11-nodejs20",
        )

    elif selected_backend == "modal":
        print_success("Terminal backend: Modal")
        print_info("Serverless cloud sandboxes. Each session gets its own container.")
        from tools.managed_tool_gateway import is_managed_tool_gateway_ready
        from tools.tool_backend_helpers import normalize_modal_mode

        managed_modal_available = bool(
            managed_nous_tools_enabled()
            and
            get_nous_subscription_features(config).nous_auth_present
            and is_managed_tool_gateway_ready("modal")
        )
        modal_mode = normalize_modal_mode(cfg_get(config, "terminal", "modal_mode"))
        use_managed_modal = False
        if managed_modal_available:
            modal_choices = [
                "Use my Nous subscription",
                "Use my own Modal account",
            ]
            if modal_mode == "managed":
                default_modal_idx = 0
            elif modal_mode == "direct":
                default_modal_idx = 1
            else:
                default_modal_idx = 1 if get_env_value("MODAL_TOKEN_ID") else 0
            modal_mode_idx = prompt_choice(
                "Select how Modal execution should be billed:",
                modal_choices,
                default_modal_idx,
            )
            use_managed_modal = modal_mode_idx == 0

        if use_managed_modal:
            config["terminal"]["modal_mode"] = "managed"
            print_info("Modal execution will use the managed Nous gateway and bill to your subscription.")
            if get_env_value("MODAL_TOKEN_ID") or get_env_value("MODAL_TOKEN_SECRET"):
                print_info(
                    "Direct Modal credentials are still configured, but this backend is pinned to managed mode."
                )
        else:
            config["terminal"]["modal_mode"] = "direct"
            print_info("Requires a Modal account: https://modal.com")

            # Check if modal SDK is installed
            try:
                __import__("modal")
            except ImportError:
                print_info("Installing modal SDK...")
                from hermes_cli.tools_config import _pip_install

                result = _pip_install(["modal"])
                if result.returncode == 0:
                    print_success("modal SDK installed")
                else:
                    print_warning("Install failed — run manually: uv pip install modal")

            # Modal token
            print()
            print_info("Modal authentication:")
            print_info("  Get your token at: https://modal.com/settings")
            existing_token = get_env_value("MODAL_TOKEN_ID")
            if existing_token:
                print_info("  Modal token: already configured")
                if prompt_yes_no("  Update Modal credentials?", False):
                    token_id = prompt("    Modal Token ID", password=True)
                    token_secret = prompt("    Modal Token Secret", password=True)
                    if token_id:
                        save_env_value("MODAL_TOKEN_ID", token_id)
                    if token_secret:
                        save_env_value("MODAL_TOKEN_SECRET", token_secret)
            else:
                token_id = prompt("    Modal Token ID", password=True)
                token_secret = prompt("    Modal Token Secret", password=True)
                if token_id:
                    save_env_value("MODAL_TOKEN_ID", token_id)
                if token_secret:
                    save_env_value("MODAL_TOKEN_SECRET", token_secret)

    elif selected_backend == "daytona":
        print_success("Terminal backend: Daytona")
        print_info("Persistent cloud development environments.")
        print_info("Each session gets a dedicated sandbox with filesystem persistence.")
        print_info("Sign up at: https://daytona.io")

        # Check if daytona SDK is installed
        try:
            __import__("daytona")
        except ImportError:
            print_info("Installing daytona SDK...")
            from hermes_cli.tools_config import _pip_install

            result = _pip_install(["daytona"])
            if result.returncode == 0:
                print_success("daytona SDK installed")
            else:
                print_warning("Install failed — run manually: uv pip install daytona")
                if result.stderr:
                    print_info(f"  Error: {result.stderr.strip().splitlines()[-1]}")

        # Daytona API key
        print()
        existing_key = get_env_value("DAYTONA_API_KEY")
        if existing_key:
            print_info("  Daytona API key: already configured")
            if prompt_yes_no("  Update API key?", False):
                api_key = prompt("    Daytona API key", password=True)
                if api_key:
                    save_env_value("DAYTONA_API_KEY", api_key)
                    print_success("    Updated")
        else:
            api_key = prompt("    Daytona API key", password=True)
            if api_key:
                save_env_value("DAYTONA_API_KEY", api_key)
                print_success("    Configured")

        # Image and resource limits use defaults; tune via `hermes setup terminal`.
        config["terminal"].setdefault(
            "daytona_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )

    elif selected_backend == "vercel_sandbox":
        print_success("Terminal backend: Vercel Sandbox")
        print_info("Cloud microVM sandboxes with snapshot-backed filesystem persistence.")
        print_info("Requires the optional SDK: pip install 'hermes-agent[vercel]'")

        try:
            __import__("vercel")
        except ImportError:
            print_info("Installing vercel SDK...")
            import subprocess

            # Managed uv first: the store is never on PATH, so a bare
            # which() misses the uv Hermes installed. Realizing one is
            # welcome here — this is the interactive setup wizard, already
            # mid-install, and the alternative tier is a pip that a `uv venv`
            # venv may not even have.
            import pm

            uv_bin, uv_env = pm.uv()
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "--python", sys.executable, "vercel"],
                    capture_output=True,
                    text=True,
                    env=uv_env,
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "vercel"],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                print_success("vercel SDK installed")
            else:
                print_warning("Install failed — run manually: pip install 'hermes-agent[vercel]'")
                if result.stderr:
                    print_info(f"  Error: {result.stderr.strip().splitlines()[-1]}")

        _prompt_vercel_sandbox_settings(config)

    elif selected_backend in plugin_backend_names:
        try:
            from agent.terminal_env_registry import get_provider

            _provider = get_provider(selected_backend)
            print_success(f"Terminal backend: {_provider.display_name}")
            for _line in _provider.setup_instructions():
                print_info(_line)
            _provider.post_setup()
        except Exception as exc:
            print_warning(f"Backend plugin setup hook failed: {exc}")

    elif selected_backend == "ssh":
        print_success("Terminal backend: SSH")
        print_info("Run commands on a remote machine via SSH.")

        # SSH host
        current_host = get_env_value("TERMINAL_SSH_HOST") or ""
        host = prompt("  SSH host (hostname or IP)", current_host)
        if host:
            save_env_value("TERMINAL_SSH_HOST", host)

        # SSH user
        current_user = get_env_value("TERMINAL_SSH_USER") or ""
        user = prompt("  SSH user", current_user or os.getenv("USER", ""))
        if user:
            save_env_value("TERMINAL_SSH_USER", user)

        # SSH port
        current_port = get_env_value("TERMINAL_SSH_PORT") or "22"
        port = prompt("  SSH port", current_port)
        if port and port != "22":
            save_env_value("TERMINAL_SSH_PORT", port)

        # SSH key
        current_key = get_env_value("TERMINAL_SSH_KEY") or ""
        default_key = str(Path.home() / ".ssh" / "id_rsa")
        ssh_key = prompt("  SSH private key path", current_key or default_key)
        if ssh_key:
            save_env_value("TERMINAL_SSH_KEY", ssh_key)

        # Test connection
        if host and prompt_yes_no("  Test SSH connection?", True):
            print_info("  Testing connection...")
            import subprocess

            ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
            if ssh_key:
                ssh_cmd.extend(["-i", ssh_key])
            if port and port != "22":
                ssh_cmd.extend(["-p", port])
            ssh_cmd.append(f"{user}@{host}" if user else host)
            ssh_cmd.append("echo ok")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            if result.returncode == 0:
                print_success("  SSH connection successful!")
            else:
                print_warning(f"  SSH connection failed: {result.stderr.strip()}")
                print_info("  Check your SSH key and host settings.")

    # Sync terminal backend to .env so terminal_tool picks it up directly.
    # config.yaml is the source of truth, but terminal_tool reads TERMINAL_ENV.
    save_env_value("TERMINAL_ENV", selected_backend)
    if selected_backend == "modal":
        save_env_value("TERMINAL_MODAL_MODE", config["terminal"].get("modal_mode", "auto"))
    if selected_backend == "vercel_sandbox":
        save_env_value("TERMINAL_VERCEL_RUNTIME", config["terminal"].get("vercel_runtime", "node24"))
    save_config(config)
    print()
    print_success(f"Terminal backend set to: {selected_backend}")


# =============================================================================
# Section 3: Agent Settings
# =============================================================================


def _apply_default_agent_settings(config: dict):
    """Apply recommended defaults for all agent settings without prompting."""
    config.setdefault("agent", {})["max_turns"] = 150
    # config.yaml is authoritative for max_turns (the gateway bridges it into HERMES_MAX_ITERATIONS);
    # a stale .env entry silently shadowing it caused the 60-vs-500 bug, so drop it.
    remove_env_value("HERMES_MAX_ITERATIONS")
    config.setdefault("display", {})["tool_progress"] = "all"
    config.setdefault("compression", {})["enabled"] = True
    config["compression"]["threshold"] = 0.50
    # Never auto-reset (the gateway default); written explicitly so it is visible in config.yaml.
    config.setdefault("session_reset", {})["mode"] = "none"
    save_config(config)
    print_success("Applied recommended defaults:")
    _info("  Max iterations: 150", "  Tool progress: all", "  Compression threshold: 0.50",
          "  Session reset: never (use /reset or compression)",
          "  Run `hermes setup agent` later to customize.")


def _prompt_number(label: str, current, cast=int):
    """Prompt for a number; ``None`` when the answer does not parse."""
    try:
        return cast(prompt(label, str(current)))
    except ValueError:
        return None


def _prompt_int_setting(section: dict, key: str, label: str, current, accept) -> None:
    """Prompt for an int; store it under *key* only when it parses and *accept* holds."""
    value = _prompt_number(label, current)
    if value is not None and accept(value):
        section[key] = value


_TOOL_PROGRESS_HELP = (
    "Tool Progress Display", "Controls how much tool activity is shown (CLI and messaging).",
    "  off     — Silent, just the final response",
    "  new     — Show tool name only when it changes (less noise)",
    "  all     — Show every tool call with a short preview",
    "  verbose — Full args, results, and debug logs",
    "  log     — Silent in chat; write every tool call to ~/.hermes/logs/tool_calls.log (gateway only)",
)
_SESSION_RESET_HELP = (
    "Messaging sessions (Telegram, Discord, etc.) accumulate context over time.",
    "Each message adds to the conversation history, which means growing API costs.", "",
    "To manage this, sessions can automatically reset after a period of inactivity",
    "or at a fixed time each day. When a reset happens, the agent saves important",
    "things to its persistent memory first — but the conversation context is cleared.", "",
    "You can also manually reset anytime by typing /reset in chat.", "",
)
_SESSION_RESET_CHOICES = [
    "Inactivity + daily reset (reset whichever comes first)",
    "Inactivity only (reset after N minutes of no messages)",
    "Daily only (reset at a fixed hour each day)",
    "Never auto-reset (recommended - context lives until /reset or context compression)",
    "Keep current settings",
]
_SESSION_RESET_MODES = ("both", "idle", "daily", "none")  # index 4 = keep current


def setup_agent_settings(config: dict):
    """Configure agent behavior: iterations, progress display, compression, session reset."""
    print_header("Agent Settings")
    _info(f"   Guide: {_DOCS_BASE}/user-guide/configuration", None)

    # ── Max Iterations ── (config.yaml is authoritative; never surface a stale legacy .env value)
    # If a legacy .env entry is still around (from pre-PR#18413 setups), prefer the config value so we don't
    # surface a stale number to the user.
    current_max = str(cfg_get(config, "agent", "max_turns", default=90))
    _info("Maximum tool-calling iterations per conversation.",
          "Higher = more complex tasks, but costs more tokens.",
          f"Press Enter to keep {current_max}. Use 90 for most tasks or 150+ for open exploration.")
    max_iter = _prompt_number("Max iterations", current_max)
    if max_iter is None:
        print_warning("Invalid number, keeping current value")
    elif max_iter > 0:
        # config.yaml only; gateway/run.py derives HERMES_MAX_ITERATIONS from agent.max_turns.
        config.setdefault("agent", {})["max_turns"] = max_iter
        config.pop("max_turns", None)
        remove_env_value("HERMES_MAX_ITERATIONS")
        print_success(f"Max iterations set to {max_iter}")

    # ── Tool Progress Display ──
    _info("", *_TOOL_PROGRESS_HELP)
    current_mode = cfg_get(config, "display", "tool_progress", default="all")
    mode = prompt("Tool progress mode", current_mode)
    if mode.lower() in {"off", "new", "all", "verbose", "log"}:
        config.setdefault("display", {})["tool_progress"] = mode.lower()
        save_config(config)
        print_success(f"Tool progress set to: {mode.lower()}")
    else:
        print_warning(f"Unknown mode '{mode}', keeping '{current_mode}'")

    # ── Context Compression ──
    print_header("Context Compression")
    _info("Automatically summarizes old messages when context gets too long.",
          "Higher threshold = compress later (use more context). Lower = compress sooner.")
    config.setdefault("compression", {})["enabled"] = True
    current_threshold = cfg_get(config, "compression", "threshold", default=0.50)
    threshold = _prompt_number("Compression threshold (0.5-0.95)", current_threshold, float)
    if threshold is not None and 0.5 <= threshold <= 0.95:
        config["compression"]["threshold"] = threshold
    print_success(f"Context compression threshold set to {config['compression'].get('threshold', 0.50)}")

    # ── Session Reset Policy ──
    print_header("Session Reset Policy")
    _info(*_SESSION_RESET_HELP)
    _prompt_session_reset(config.setdefault("session_reset", {}))
    save_config(config)


def _prompt_session_reset(reset_cfg: dict) -> None:
    """Pick the session reset mode and its idle/daily parameters in place."""
    current_mode = reset_cfg.get("mode", "none")
    current_idle, current_hour = reset_cfg.get("idle_minutes", 1440), reset_cfg.get("at_hour", 4)
    default_reset = _SESSION_RESET_MODES.index(current_mode) if current_mode in _SESSION_RESET_MODES else 3
    reset_idx = prompt_choice("Session reset mode:", _SESSION_RESET_CHOICES, default_reset)
    mode = _SESSION_RESET_MODES[reset_idx] if 0 <= reset_idx < len(_SESSION_RESET_MODES) else None
    if mode is None:  # keep current settings
        return
    reset_cfg["mode"] = mode
    if mode in ("both", "idle"):
        _prompt_int_setting(reset_cfg, "idle_minutes", "  Inactivity timeout (minutes)", current_idle, lambda v: v > 0)
    if mode in ("both", "daily"):
        _prompt_int_setting(reset_cfg, "at_hour", "  Daily reset hour (0-23, local time)", current_hour, lambda v: 0 <= v <= 23)
    idle_now, hour_now = reset_cfg.get("idle_minutes", 1440), reset_cfg.get("at_hour", 4)
    if mode == "none":
        print_info("Sessions will never auto-reset. Context is managed only by compression.")
        print_warning("Long conversations will grow in cost. Use /reset manually when needed.")
    else:
        print_success({
            "both": f"Sessions reset after {idle_now} min idle or daily at {hour_now}:00",
            "idle": f"Sessions reset after {idle_now} min of inactivity",
            "daily": f"Sessions reset daily at {hour_now}:00",
        }[mode])


# ── Section 5: Tool Configuration (delegates to unified tools_config.py) ──


def setup_tools(config: dict, first_install: bool = False):
    """`hermes setup tools` == `hermes tools`: platform selection → toolset toggles → provider keys.
    ``first_install`` selects the simplified flow (no platform menu, prompts for all missing keys)."""
    from hermes_cli.tools_config import tools_command
    tools_command(first_install=first_install, config=config)


# ── Shared Metrics ──


_SEND_CONSENT_EXPLAINER = (
    "", "Sending uploads each daily package to the Nous telemetry",
    "service. Packages carry your profile-scoped install ID, a",
    "stable random UUID that identifies this profile across days",
    "(it contains no personal information and is reset by deleting",
    "the shared-metrics directory). Only packages whose entire",
    "collection period falls inside a recorded consent window are",
    "ever sent — data from before you opt in, or from any gap",
    "while sending was off, stays on this machine. Sending can be", "turned off again at any time.",
)


def setup_telemetry(config: dict):
    """Configure the local shared-metrics subscriber and optional sending."""
    print_header("Shared Metrics")
    _info("Shared metrics contain only bounded counters and histograms.",
          "Collection is local. Sending them to Nous is a separate opt-in.")
    shared_metrics = _sub_dict(_sub_dict(config, "telemetry"), "shared_metrics")
    current = shared_metrics.get("enabled") is True
    shared_metrics["enabled"] = prompt_yes_no("Enable local shared metrics?", default=current)
    if not shared_metrics["enabled"]:
        print_info("Local shared metrics disabled.")
        # Sending cannot outlive collection (send=true would log an error every run, never send).
        if shared_metrics.get("send") is True:
            shared_metrics["send"] = False
            print_info("Sending shared metrics disabled as well.")
        # Turning collection off withdraws send consent too. Recorded unconditionally: the send
        # key may already be false while the consent window is still open, and it must close.
        _record_send_consent_change(enabled=False)
        return
    print_success("Local shared metrics enabled.")
    _info(*_SEND_CONSENT_EXPLAINER)
    shared_metrics["send"] = prompt_yes_no("Send shared metrics to Nous?", default=shared_metrics.get("send") is True)
    _record_send_consent_change(enabled=shared_metrics["send"])
    if shared_metrics["send"]:
        print_success("Sending shared metrics enabled.")
    else:
        print_info("Sending shared metrics disabled (collection stays local).")


def _record_send_consent_change(*, enabled: bool) -> None:
    """Reconcile consent windows at the moment the user decides — same single writer as the relay
    and the sender, so wizard, relay and mid-pass callers cannot disagree."""
    try:
        from hermes_cli.observability.shared_metrics import SharedMetricsStore
        from hermes_cli.observability.shared_metrics_sender import reconcile_send_consent
        from hermes_cli.sqlite_util import write_txn
        with SharedMetricsStore()._connection() as connection, write_txn(connection):
            reconcile_send_consent(connection, enabled)
    except Exception:
        # Never block the wizard on telemetry bookkeeping; the relay reconciles on the next hook.
        logger.debug("Unable to record shared-metrics consent change", exc_info=True)


# Extracted sections, re-exported so callers and test patches keep resolving through
# hermes_cli.setup. They import this module lazily inside bodies, so this is cycle-free.

from hermes_cli.setup_tts import setup_tts  # noqa: E402
from hermes_cli.setup_terminal import setup_terminal_backend  # noqa: E402
from hermes_cli.setup_platforms import setup_gateway  # noqa: E402
from hermes_cli.setup_summary import _print_setup_summary  # noqa: E402,F401
from hermes_cli.setup_migration import _offer_openclaw_migration, _skip_configured_section  # noqa: E402
from hermes_cli.setup_quick import _run_portal_one_shot, _run_quick_setup  # noqa: E402


# ── Main Wizard Orchestrator ──

SETUP_SECTIONS = [
    ("model", "Model & Provider", setup_model_provider),
    ("tts", "Text-to-Speech", setup_tts),
    ("terminal", "Terminal Backend", setup_terminal_backend),
    ("gateway", "Messaging Platforms (Gateway)", setup_gateway),
    ("tools", "Tools", setup_tools),
    ("telemetry", "Shared Metrics", setup_telemetry),
    ("agent", "Agent Settings", setup_agent_settings),
]


def run_setup_wizard(args):
    """Run setup with navigation control scoped to this invocation."""
    with _setup_navigation_scope():
        try:
            return _run_setup_wizard_impl(args)
        except _SetupCancelled:
            _info(None, "Setup cancelled. Remaining sections were not changed.")
            return None


def _backup_config_file(config_path: Path) -> Path | None:
    """Back up config.yaml before setup modifies it; None when absent or copy fails."""
    if not config_path.exists():
        return None
    import shutil
    from datetime import datetime
    backup_path = config_path.with_suffix(f".yaml.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        shutil.copy2(config_path, backup_path)
        return backup_path
    except Exception:
        return None


def _run_setup_section(config: dict, section: str) -> None:
    """``hermes setup <section>``: run one SETUP_SECTIONS entry under the banner."""
    entry = next(((label, func) for key, label, func in SETUP_SECTIONS if key == section), None)
    if entry is None:
        print_error(f"Unknown setup section: {section}")
        print_info(f"Available sections: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")
        return
    label, func = entry
    _print_banner(f"│     ⚕ Hermes Setup — {label:<34s} │")
    _run_setup_steps([(label, lambda: func(config))])
    save_config(config)
    print()
    print_success(f"{label} configuration complete!")


def _run_full_setup(config: dict, hermes_home, *, is_existing: bool, migration_ran: bool) -> None:
    """Full Setup — run all sections, honoring post-migration skips."""
    print_header("Configuration Location")
    _info(f"Config file:  {get_config_path()}", f"Secrets file: {get_env_path()}",
          f"Data folder:  {hermes_home}", f"Install dir:  {PROJECT_ROOT}", None,
          "You can edit these files directly or use 'hermes config edit'")
    if migration_ran:
        _info(None, "Settings were imported from OpenClaw.",
              "Each section below will show what was imported — press Enter to keep,",
              "or choose to reconfigure if needed.")

    # Agent Settings are not prompted: first installs get defaults, existing keep theirs.
    if not is_existing:
        _apply_default_agent_settings(config)

    def _skip(key: str, label: str) -> bool:
        return migration_ran and _skip_configured_section(config, key, label)

    def _gateway_step() -> None:
        if not _skip("gateway", "Messaging Platforms"):
            setup_gateway(config)
            return
        # A skipped (migrated) gateway section still needs its service so imported platforms
        # and cron jobs become active.
        from hermes_cli.gateway import ensure_gateway_service
        ensure_gateway_service(context="setup")

    def _step(key: str, label: str, run) -> tuple:
        return label, lambda: None if _skip(key, label) else run()

    _run_setup_steps([
        _step("model", "Model & Provider", lambda: setup_model_provider(config)),
        _step("terminal", "Terminal Backend", lambda: setup_terminal_backend(config)),
        ("Messaging Platforms", _gateway_step),
        _step("tools", "Tools", lambda: setup_tools(config, first_install=not is_existing))])


# First-time mode picker: (menu label, setup_quick runner name) — None falls through to Full Setup.
_FIRST_TIME_MODES = (
    ("Quick Setup (Nous Portal) — free OAuth login, no API keys, model + tools (recommended)",
     "_run_first_time_quick_setup"),
    ("Full setup — configure every provider, tool & option yourself (bring your own keys)", None),
    ("Blank Slate — everything off except the bare minimum; opt in to each capability", "_run_blank_slate_setup"),
)


def _run_setup_wizard_impl(args):
    """Run the interactive setup wizard: full/quick (auto-detected), ``--portal``, or one
    ``hermes setup <section>`` from SETUP_SECTIONS."""
    from hermes_cli.config import is_managed, managed_error
    if is_managed():
        managed_error("run setup wizard")
        return
    ensure_hermes_home()
    if getattr(args, "reset", False):
        save_config(copy.deepcopy(DEFAULT_CONFIG))
        print_success("Configuration reset to defaults.")
    reconfigure_requested = bool(getattr(args, "reconfigure", False))
    quick_requested = bool(getattr(args, "quick", False))
    config = load_config()
    hermes_home = get_hermes_home()
    # Back up existing config before setup modifies it (#3522)
    config_path = get_config_path()
    _backup_path = _backup_config_file(config_path)

    # Non-interactive environments (headless SSH, Docker, CI/CD)
    if getattr(args, 'non_interactive', False) or not is_interactive_stdin():
        print_noninteractive_setup_guidance("Running in a non-interactive environment (no TTY detected).")
        return
    if getattr(args, "portal", False):  # one-shot Nous Portal setup; skips the rest
        _run_portal_one_shot(config)
        return
    section = getattr(args, "section", None)
    if section:
        _run_setup_section(config, section)
        return

    # Existing installation == a provider is configured
    from hermes_cli.auth import get_active_provider
    is_existing = bool(get_env_value("OPENROUTER_API_KEY") or get_env_value("OPENAI_BASE_URL")
                       or get_active_provider() is not None)
    _print_banner("│             ⚕ Hermes Agent Setup Wizard                │",
                  "├─────────────────────────────────────────────────────────┤",
                  "│  Let's configure your Hermes Agent installation.       │",
                  "│  Press Ctrl+C at any time to exit.                     │")
    migration_ran = False
    if is_existing:
        # Full reconfigure wizard is the default (Enter keeps each current value); `--quick`
        # narrows it to missing items (partial OpenClaw import, cleared key). --reconfigure is a
        # backwards-compatible no-op here.
        if quick_requested:
            _run_setup_steps([("Quick Setup", lambda: _run_quick_setup(config, hermes_home))])
            return
        print_header("Reconfigure", gap=True)
        print_success("You already have Hermes configured.")
        _info("Running the full wizard — each prompt shows your current value.",
              "Press Enter to keep it, or type a new value to change it.", "",
              "Tip: jump straight to a section with 'hermes setup model|terminal|",
              "     gateway|tools|agent', or fill only missing items with --quick.")
    else:
        # First-time setup (--reconfigure / --quick are meaningless here; fall through)
        print()
        if reconfigure_requested or quick_requested:
            _info("No existing configuration found — running first-time setup.", None)
        migration_ran = _offer_openclaw_migration(hermes_home)  # before configuration begins
        if migration_ran:
            config = load_config()
        setup_mode = prompt_choice("How would you like to set up Hermes?", [label for label, _ in _FIRST_TIME_MODES], 0)
        label, runner = _FIRST_TIME_MODES[setup_mode]
        if runner is not None:
            from hermes_cli import setup_quick
            _run_setup_steps([(label, lambda: getattr(setup_quick, runner)(config, hermes_home, is_existing))])
            return
    _run_full_setup(config, hermes_home, is_existing=is_existing, migration_ran=migration_ran)

    # Save and show summary
    save_config(config)
    if _backup_path and _backup_path.exists():
        _info(f"Previous config backed up to: {_backup_path}",
              "If setup changed a value you customized, restore it with:",
              f"  cp {_backup_path} {config_path}")
    _print_setup_summary(config, hermes_home)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
from typing import Dict  # noqa: F401,E402
from typing import Optional  # noqa: F401,E402
import json  # noqa: F401,E402
import shutil  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'get_nous_subscription_features': ('hermes_cli.nous_subscription', 'get_nous_subscription_features'),
    'get_optional_skills_dir': ('hermes_constants', 'get_optional_skills_dir'),
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
