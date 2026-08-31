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
from typing import Optional, Dict, Any, Callable

from hermes_cli.curses_ui import MenuNavigationEvent, MenuNavigationStart
from hermes_cli.nous_subscription import get_nous_subscription_features
from tools.tool_backend_helpers import managed_nous_tools_enabled
from hermes_constants import get_optional_skills_dir

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


def _supports_same_provider_pool_setup(provider: str) -> bool:
    if not provider or provider == "custom":
        return False
    if provider == "openrouter":
        return True
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY.get(provider)
    if not pconfig:
        return False
    return pconfig.auth_type in {"api_key", "oauth_device_code"}


# Default model lists per provider — used as fallback when the live
# /models endpoint can't be reached.
_DEFAULT_PROVIDER_MODELS = {
    "copilot-acp": [
        "copilot-acp",
    ],
    "copilot": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-opus-4.6",
        "claude-sonnet-5",
        "claude-sonnet-4.6",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-2.5-pro",
    ],
    "gemini": [
        "gemini-3.1-pro-preview", "gemini-3-pro-preview",
        "gemini-3.6-flash", "gemini-3.1-flash-lite-preview",
    ],
    "vertex": [
        "google/gemini-3.1-pro-preview", "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview", "google/gemini-3.1-flash-lite-preview",
        "google/gemini-2.5-pro", "google/gemini-2.5-flash",
    ],
    "zai": ["glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5", "glm-4.7", "glm-4.5", "glm-4.5-flash"],
    "kimi-coding": ["kimi-k3", "kimi-k2.6", "kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview"],
    "kimi-coding-cn": ["kimi-k3", "kimi-k2.6", "kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview"],
    "stepfun": ["step-3.5-flash", "step-3.5-flash-2603"],
    "arcee": ["trinity-large-thinking", "trinity-large-preview", "trinity-mini"],
    "minimax": ["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"],
    "minimax-cn": ["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"],
    "ai-gateway": ["anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6", "openai/gpt-5", "google/gemini-3-flash"],
    "kilocode": ["anthropic/claude-sonnet-5", "anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "google/gemini-3-pro-preview", "google/gemini-3-flash-preview"],
    "opencode-zen": ["x-preview-f-free", "gpt-5.6-sol", "gpt-5.4", "gpt-5.3-codex", "claude-opus-5", "claude-sonnet-5", "gemini-3.7-flash", "glm-5.2", "kimi-k3", "minimax-m3"],
    "opencode-free": ["deepseek-v4-flash-free", "hy3-free", "mimo-v2.5-free", "laguna-s-2.1-free", "nemotron-3-ultra-free", "nemotron-3.5-lightning-free", "muse-spark-1.2-contributor-free"],
    "opencode-go": ["kimi-k3", "kimi-k2.7-code", "kimi-k2.6", "gpt-5.6-luna", "grok-4.5", "glm-5.3", "glm-5.3-flash", "glm-5.2", "mimo-v2.5-pro", "mimo-v2.5", "minimax-m3", "minimax-m2.7", "qwen3.8-max", "qwen3.7-max", "deepseek-v4-pro", "hy3"],
    "huggingface": [
        "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct", "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-V3.2", "moonshotai/Kimi-K2.5",
    ],
}


def _current_reasoning_effort(config: Dict[str, Any]) -> str:
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
    print(color("☤ Hermes Setup — Non-interactive mode", Colors.CYAN, Colors.BOLD))
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

            value = line_input(color(display, Colors.YELLOW))

        cleaned = _sanitize_pasted_input(value)
        return cleaned.strip() or default or ""
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)


class _SetupControlFlow(BaseException):
    """Bypass provider error handlers that intentionally catch ``Exception``.

    Provider setup contains broad compatibility boundaries around network,
    plugin, and credential integrations. Navigation must cross those layers
    unchanged so the outer setup state machine can replay the prior prompt.
    """


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
        self.section_index = section_index
        self.prompt_index = prompt_index
        self.active_prompt_index = -1
        self.resolved_choices: list[object] = []
        self.replay_choices: list[object] = []


_SETUP_NAVIGATION: ContextVar[_SetupNavigationState | None] = ContextVar(
    "hermes_setup_navigation", default=None
)


def _handle_setup_menu_navigation(
    event: MenuNavigationEvent,
    value: object = None,
) -> MenuNavigationStart | None:
    """Translate shared curses menu events into setup control flow."""
    state = _SETUP_NAVIGATION.get()
    if state is None:
        return None
    if event is MenuNavigationEvent.BEGIN:
        if state.section_index < 0:
            state.active_prompt_index = -1
            return MenuNavigationStart()
        state.active_prompt_index = state.prompt_index
        state.prompt_index += 1
        allow_back = state.section_index > 0 or state.active_prompt_index > 0
        if state.active_prompt_index < len(state.replay_choices):
            return MenuNavigationStart(
                allow_back=allow_back,
                replay_value=copy.deepcopy(
                    state.replay_choices[state.active_prompt_index]
                ),
            )
        return MenuNavigationStart(allow_back=allow_back)
    if event is MenuNavigationEvent.RESOLVE:
        prompt_index = state.active_prompt_index
        if prompt_index < 0:
            return None
        resolved = copy.deepcopy(value)
        if prompt_index < len(state.resolved_choices):
            state.resolved_choices[prompt_index] = resolved
            del state.resolved_choices[prompt_index + 1 :]
        else:
            state.resolved_choices.append(resolved)
        return None
    if event is MenuNavigationEvent.CANCEL:
        raise _SetupCancelled()
    if event is MenuNavigationEvent.BACK:
        raise _SetupGoBack(state.active_prompt_index)
    return None


_BRACKETED_PASTE_PATTERN = re.compile(r"\x1b\[\s*200~|\x1b\[\s*201~")


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
    return curses_radiolist(
        question,
        choices,
        selected=default,
        cancel_returns=-1,
        description=description,
    )


def prompt_choice(question: str, choices: list, default: int = 0, description: str | None = None) -> int:
    """Prompt for a choice from a list with arrow key navigation.

    Escape cancels an active setup wizard. Outside setup it keeps the current
    default. The curses component owns its own numbered fallback, so a cancel
    result must never be mistaken for a request to open another prompt.
    Ctrl+C exits the wizard.
    """
    idx = _curses_prompt_choice(question, choices, default, description=description)
    if idx >= 0:
        if idx == default:
            print_info("  Skipped (keeping current)")
            print()
            return default
        print()
        return idx

    return default


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

    # Setup owns a scoped curses navigation handler. Route binary selections
    # through the same menu surface so ESC and left-arrow work consistently,
    # while preserving the traditional line prompt for every other caller.
    if _SETUP_NAVIGATION.get() is not None:
        default_index = 0 if default else 1
        return _curses_prompt_choice(
            question,
            ["Yes", "No"],
            default_index,
        ) == 0

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


# ── Section 1: Model & Provider Configuration ──


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
        from hermes_cli.auth_error_copy import provider_setup_failure_lines
        lead, *rest = provider_setup_failure_lines(exc, retry_command="hermes model")
        print_warning(lead)
        for line in rest:
            print_info(line)

    # Re-sync from disk in place: cmd_model saved via its own load/save cycle and the wizard's
    # final save_config(config) must not clobber it with stale values. Rotation, vision and TTS
    # keep safe defaults (configure via `hermes auth add` / `hermes setup tts`).
    config.clear()
    config.update(load_config())
    save_config(config)


# =============================================================================
# Section 1b: TTS Provider Configuration
# =============================================================================


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

            # Managed uv first: $HERMES_HOME/bin is never on PATH, so a bare
            # which() misses the uv Hermes installed. Bootstrapping one is
            # welcome here — this is the interactive setup wizard, already
            # mid-install, and the alternative tier is a pip that a `uv venv`
            # venv may not even have.
            from hermes_cli.managed_uv import ensure_uv

            uv_bin = ensure_uv()
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "--python", sys.executable, "vercel"],
                    capture_output=True,
                    text=True,
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
    save_config(config)
    print_success("Applied recommended defaults:")
    _info("  Max iterations: 150", "  Tool progress: all", "  Compression threshold: 0.50",
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
def setup_agent_settings(config: dict):
    """Configure agent behavior: iterations, progress display and compression."""
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

    save_config(config)


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


def _run_portal_one_shot(config: dict) -> None:
    """One-shot Nous Portal setup — OAuth + model pick + provider + Tool Gateway.

    Wired into ``hermes setup --portal`` and ``hermes portal``. This is the
    Nous-Portal slice of the first-time quick setup, collapsed into a single
    shareable command so a brand-new user goes from zero to a fully working
    Hermes session — model selected, provider set, and web/image/tts/browser
    tools routed via their Portal sub — without being told to run
    ``hermes setup`` and hunt for the quick-setup option.

    The login + model selection + provider switch + Tool Gateway opt-in are all
    delegated to ``_model_flow_nous`` — the exact same flow quick setup uses
    (``_run_first_time_quick_setup``) and the same one ``hermes model`` runs
    when you pick Nous. Routing through it (instead of hand-rolling the auth +
    provider write here) means ``hermes portal`` always offers a model picker,
    and there is a single source of truth for the Nous onboarding steps.
    """
    from hermes_cli.config import load_config

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐",
            Colors.MAGENTA,
        )
    )
    print(color("│     ⚕ Hermes Setup — Nous Portal (one-shot)             │", Colors.MAGENTA))
    print(
        color(
            "└─────────────────────────────────────────────────────────┘",
            Colors.MAGENTA,
        )
    )
    print()
    print_info("  One subscription, 300+ models, plus the Tool Gateway:")
    print_info("    web search, image generation, TTS, browser automation")
    print_info("    — all routed through your Nous Portal sub.")
    print()
    print_info("  Sign up: https://portal.nousresearch.com/manage-subscription")
    print()

    # _model_flow_nous handles BOTH the logged-out path (device-code OAuth,
    # which selects a model internally) and the already-logged-in path (curated
    # Nous model picker), then offers the Tool Gateway opt-in and sets
    # provider=nous via the login/model save. This is the same routine quick
    # setup calls, so `hermes portal` == quick setup's Nous step.
    try:
        from hermes_cli.main import _model_flow_nous

        _model_flow_nous(config)
    except (KeyboardInterrupt, EOFError, SystemExit):
        # _login_nous raises SystemExit(130)/(1) on cancel/failure; the
        # logged-out path inside _model_flow_nous catches it, but the
        # expired-session re-login path only catches Exception, so a
        # SystemExit there would otherwise escape and kill the whole CLI.
        # Treat all of these as a graceful cancel/abort for the portal flow.
        print()
        print_info("  Setup cancelled.")
        print_info("  You can retry later with `hermes portal`.")
        return
    except Exception as exc:
        logger.debug("_model_flow_nous error during `hermes portal`: %s", exc)
        print()
        print_error(f"  Nous Portal setup encountered an error: {exc}")
        print_info("  You can retry later with `hermes portal`.")
        return

    # Re-sync the in-memory config from disk — _model_flow_nous (and the
    # underlying login/model save) write via their own load/save cycle, so any
    # later save_config(config) by a caller must not clobber those values.
    try:
        _refreshed = load_config()
        if isinstance(_refreshed, dict):
            config.clear()
            config.update(_refreshed)
    except Exception:
        pass

    print()
    print_success("Portal setup complete.")
    print_info("  Run `hermes portal info` to inspect routing.")
    print_info("  Run `hermes` to start chatting.")


@contextmanager
def _setup_navigation_scope():
    """Install and reliably restore the setup menu navigation context."""
    from hermes_cli.curses_ui import (
        reset_menu_navigation_handler,
        set_menu_navigation_handler,
    )

    token = _SETUP_NAVIGATION.set(_SetupNavigationState())
    menu_token = set_menu_navigation_handler(_handle_setup_menu_navigation)
    try:
        yield
    finally:
        reset_menu_navigation_handler(menu_token)
        _SETUP_NAVIGATION.reset(token)


def run_setup_wizard(args):
    """Run setup with navigation control scoped to this invocation."""
    with _setup_navigation_scope():
        try:
            return _run_setup_wizard_impl(args)
        except _SetupCancelled:
            print()
            print_info("Setup cancelled. Remaining sections were not changed.")
            return None


def _run_setup_steps(
    steps: list[tuple[str, Callable[[], None]]],
) -> None:
    """Run setup sections with left-arrow navigation between choices.

    Left arrow at a section's first choice returns to the previous section.
    From a later, nested choice it replays earlier selections invisibly and
    reopens only the immediately preceding prompt.
    """
    state = _SETUP_NAVIGATION.get()
    section_index = 0
    answers_by_section: dict[int, list[object]] = {}
    replay_by_section: dict[int, list[object]] = {}
    try:
        while section_index < len(steps):
            label, action = steps[section_index]
            if state is not None:
                state.section_index = section_index
                state.prompt_index = 0
                state.active_prompt_index = -1
                state.resolved_choices = []
                state.replay_choices = copy.deepcopy(
                    replay_by_section.pop(section_index, [])
                )
            try:
                action()
            except _SetupGoBack as navigation:
                if state is not None:
                    answers_by_section[section_index] = copy.deepcopy(
                        state.resolved_choices
                    )
                if navigation.prompt_index > 0:
                    previous_index = section_index
                    target_prompt = navigation.prompt_index - 1
                    replay_by_section[previous_index] = copy.deepcopy(
                        answers_by_section.get(previous_index, [])[:target_prompt]
                    )
                else:
                    previous_index = max(0, section_index - 1)
                    previous_answers = answers_by_section.get(previous_index, [])
                    target_prompt = max(0, len(previous_answers) - 1)
                    replay_by_section[previous_index] = copy.deepcopy(
                        previous_answers[:target_prompt]
                    )
                previous_label = steps[previous_index][0]
                print()
                if previous_index == section_index:
                    print_info(f"Returning to the previous choice in {label}...")
                else:
                    print_info(f"Returning to {previous_label}...")
                section_index = previous_index
                continue
            if state is not None:
                answers_by_section[section_index] = copy.deepcopy(
                    state.resolved_choices
                )
            section_index += 1
    finally:
        if state is not None:
            state.section_index = -1
            state.prompt_index = 0
            state.active_prompt_index = -1
            state.resolved_choices = []
            state.replay_choices = []


def run_setup_action_with_navigation(
    label: str,
    action: Callable[[], None],
    *,
    cancelled_message: str = "Setup cancelled.",
) -> None:
    """Run a setup-style menu flow with Escape and nested Left navigation.

    Shared commands such as ``hermes model`` use the same provider/model
    pickers as the setup wizard, but run outside ``run_setup_wizard``.  This
    installs the setup navigation context for that standalone command and
    reuses the same prompt replay state machine.
    """
    with _setup_navigation_scope():
        try:
            _run_setup_steps([(label, action)])
        except _SetupCancelled:
            print()
            print_info(cancelled_message)


def _run_setup_wizard_impl(args):
    """Run the interactive setup wizard.


def _run_setup_section(config: dict, section: str) -> None:
    """``hermes setup <section>``: run one SETUP_SECTIONS entry under the banner."""
    entry = next(((label, func) for key, label, func in SETUP_SECTIONS if key == section), None)
    if entry is None:
        print_error(f"Unknown setup section: {section}")
        print_info(f"Available sections: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")
        return
    label, func = entry
    _print_banner(f"│     ☤ Hermes Setup — {label:<34s} │")
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
    # Back up BEFORE --reset: save_config below overwrites the very file we copy (#3522, #77299).
    config_path = get_config_path()
    from hermes_cli.config_backups import backup_config
    _backup_path = backup_config(config_path, "pre-setup")
    if getattr(args, "reset", False):
        save_config(copy.deepcopy(DEFAULT_CONFIG))
        print_success("Configuration reset to defaults.")
        if _backup_path:  # --reset may exit before the end-of-wizard notice
            _info(f"Previous config backed up to: {_backup_path}")
    reconfigure_requested = bool(getattr(args, "reconfigure", False))
    quick_requested = bool(getattr(args, "quick", False))
    config = load_config()
    hermes_home = get_hermes_home()

    # Non-interactive environments (headless SSH, Docker, CI/CD)
    if getattr(args, 'non_interactive', False) or not is_interactive_stdin():
        print_noninteractive_setup_guidance("Running in a non-interactive environment (no TTY detected).")
        return
    if getattr(args, "portal", False):  # one-shot Nous Portal setup; skips the rest
        _run_portal_one_shot(config)
        return
    section = getattr(args, "section", None)
    if section:
        for key, label, func in SETUP_SECTIONS:
            if key == section:
                print()
                print(
                    color(
                        "┌─────────────────────────────────────────────────────────┐",
                        Colors.MAGENTA,
                    )
                )
                print(color(f"│     ⚕ Hermes Setup — {label:<34s} │", Colors.MAGENTA))
                print(
                    color(
                        "└─────────────────────────────────────────────────────────┘",
                        Colors.MAGENTA,
                    )
                )
                _run_setup_steps(
                    [(label, lambda setup_func=func: setup_func(config))]
                )
                save_config(config)
                print()
                print_success(f"{label} configuration complete!")
                return

        print_error(f"Unknown setup section: {section}")
        print_info(f"Available sections: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")
        return

    # Existing installation == a provider is configured
    from hermes_cli.auth import get_active_provider
    is_existing = bool(get_env_value("OPENROUTER_API_KEY") or get_env_value("OPENAI_BASE_URL")
                       or get_active_provider() is not None)
    _print_banner("│             ☤ Hermes Agent Setup Wizard                │",
                  "├─────────────────────────────────────────────────────────┤",
                  "│  Let's configure your Hermes Agent installation.       │",
                  "│  Press Ctrl+C at any time to exit.                     │")
    migration_ran = False
    if is_existing:
        # Full reconfigure wizard is the default (Enter keeps each current value); `--quick`
        # narrows it to missing items (partial OpenClaw import, cleared key). --reconfigure is a
        # backwards-compatible no-op here.
        if quick_requested:
            _run_setup_steps(
                [("Quick Setup", lambda: _run_quick_setup(config, hermes_home))]
            )
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

        setup_mode = prompt_choice(
            "How would you like to set up Hermes?",
            [
                "Quick Setup (Nous Portal) — free OAuth login, no API keys, model + tools (recommended)",
                "Full setup — configure every provider, tool & option yourself (bring your own keys)",
                "Blank Slate — everything off except the bare minimum; opt in to each capability",
            ],
            0,
        )

        if setup_mode == 0:
            _run_setup_steps(
                [
                    (
                        "Quick Setup",
                        lambda: _run_first_time_quick_setup(
                            config, hermes_home, is_existing
                        ),
                    )
                ]
            )
            return
        if setup_mode == 2:
            _run_setup_steps(
                [
                    (
                        "Blank Slate",
                        lambda: _run_blank_slate_setup(
                            config, hermes_home, is_existing
                        ),
                    )
                ]
            )
            return

    # ── Full Setup — run all sections ──
    print_header("Configuration Location")
    print_info(f"Config file:  {get_config_path()}")
    print_info(f"Secrets file: {get_env_path()}")
    print_info(f"Data folder:  {hermes_home}")
    print_info(f"Install dir:  {PROJECT_ROOT}")
    print()
    print_info("You can edit these files directly or use 'hermes config edit'")

    if migration_ran:
        print()
        print_info("Settings were imported from OpenClaw.")
        print_info("Each section below will show what was imported — press Enter to keep,")
        print_info("or choose to reconfigure if needed.")

    # Section 3: Agent Settings — no longer prompted. First installs get the
    # recommended defaults silently; existing installs keep whatever they have.
    # Tune later with `hermes setup agent`.
    if not is_existing:
        _apply_default_agent_settings(config)

    def _model_step() -> None:
        if not (
            migration_ran
            and _skip_configured_section(config, "model", "Model & Provider")
        ):
            setup_model_provider(config)

    def _terminal_step() -> None:
        if not (
            migration_ran
            and _skip_configured_section(config, "terminal", "Terminal Backend")
        ):
            setup_terminal_backend(config)

    def _gateway_step() -> None:
        if not (
            migration_ran
            and _skip_configured_section(config, "gateway", "Messaging Platforms")
        ):
            setup_gateway(config)
            return

        # A migrated gateway section can be skipped, but its service still
        # needs to exist so imported platforms and cron jobs become active.
        from hermes_cli.gateway import ensure_gateway_service

        ensure_gateway_service(context="setup")

    def _tools_step() -> None:
        if not (
            migration_ran
            and _skip_configured_section(config, "tools", "Tools")
        ):
            setup_tools(config, first_install=not is_existing)

    _run_setup_steps(
        [
            ("Model & Provider", _model_step),
            ("Terminal Backend", _terminal_step),
            ("Messaging Platforms", _gateway_step),
            ("Tools", _tools_step),
        ]
    )

    # Save and show summary
    save_config(config)
    if _backup_path and _backup_path.exists():
        _info(f"Previous config backed up to: {_backup_path}",
              "If setup changed a value you customized, restore it with:",
              f"  cp {_backup_path} {config_path}")
    _print_setup_summary(config, hermes_home)


def _run_first_time_quick_setup(config: dict, hermes_home, is_existing: bool):
    """Streamlined first-time setup via Nous Portal: OAuth, model, terminal & messaging.

    Routes straight to the Nous Portal provider — runs the device-code OAuth
    login, picks a Nous model, then configures the terminal backend and (optionally)
    a messaging platform. Applies sensible defaults for everything else (agent
    settings, tools); the user can customize later via ``hermes setup <section>``
    or switch providers with ``hermes model``.
    """
    from hermes_cli.config import load_config

    # Step 1: Nous Portal — OAuth login + model selection.
    # _model_flow_nous() handles both the logged-out path (device-code OAuth,
    # which selects a model internally) and the already-logged-in path (curated
    # Nous model picker). Provider is set to "nous" by the login/model save.
    print()
    print_header("Nous Portal")
    print_info("One subscription, 300+ models, plus the Tool Gateway:")
    print_info("  web search, image generation, TTS, browser automation.")
    print_info("Sign up: https://portal.nousresearch.com/manage-subscription")
    print()
    try:
        from hermes_cli.main import _model_flow_nous
        _model_flow_nous(config)
    except (KeyboardInterrupt, EOFError):
        print()
        print_info("Nous Portal setup cancelled.")
    except Exception as exc:
        logger.debug("_model_flow_nous error during quick setup: %s", exc)
        print_warning(f"Nous Portal setup encountered an error: {exc}")
        print_info("You can try again later with: hermes model")

    # Re-sync the wizard's config dict from disk — _model_flow_nous (and the
    # underlying login/model save) write via their own load/save cycle, and the
    # wizard's later save_config(config) must not clobber those values (#4172).
    _refreshed = load_config()
    config.clear()
    config.update(_refreshed)

    # Step 2: Terminal Backend — where commands run is a core decision
    setup_terminal_backend(config)

    # Step 3: Apply defaults for everything else
    _apply_default_agent_settings(config)

    save_config(config)

    # Step 4: Offer messaging gateway setup
    print()
    gateway_choice = prompt_choice(
        "Connect a messaging platform? (Telegram, Discord, etc.)",
        [
            "Set up messaging now (recommended)",
            "Skip — set up later with 'hermes setup gateway'",
        ],
        0,
    )

    if gateway_choice == 0:
        setup_gateway(config)
        save_config(config)
    else:
        # Messaging skipped — still install/start the gateway service so cron
        # jobs run and platforms come alive as soon as tokens are added later
        # (e.g. via `hermes import` from another machine).
        from hermes_cli.gateway import ensure_gateway_service
        ensure_gateway_service(context="setup")

    print()
    print_success("Setup complete! You're ready to go.")
    print()
    print_info("  Configure all settings:    hermes setup")
    if gateway_choice != 0:
        print_info("  Connect Telegram/Discord:  hermes setup gateway")
    _print_macos_fda_tip()
    print()

    _print_setup_summary(config, hermes_home)


def _print_macos_fda_tip() -> None:
    """One-time macOS onboarding tip: a single Full Disk Access grant kills
    every per-folder permission prompt, permanently (issue #52010 follow-up).

    Uses the same prompt-free probe as doctor's check_macos_full_disk_access
    (the TCC db dir is FDA-gated but probing it never triggers a dialog).
    Silent on non-macOS and when FDA is already granted or indeterminate.
    """
    if sys.platform != "darwin":
        return
    tcc_dir = Path.home() / "Library" / "Application Support" / "com.apple.TCC"
    try:
        os.listdir(tcc_dir)
        return  # already granted — nothing to teach
    except PermissionError:
        pass
    except OSError:
        return  # indeterminate — don't nag
    print()
    print_info("  macOS tip: silence ALL folder permission prompts with one switch —")
    print_info("  System Settings → Privacy & Security → Full Disk Access → enable")
    print_info("  your terminal (and Hermes.app if you use Desktop), or run:")
    print_info("    open \"x-apple.systempreferences:com.apple.preference"
               ".security?Privacy_AllFiles\"")
    print_info("  The grant is permanent — it survives every Hermes update.")


def _blank_slate_minimal_toolsets(config: dict):
    """Write the minimal toolset state for a Blank Slate install.

    Only ``file``, ``terminal``, ``vision``, and ``skills`` are enabled.
    Vision is part of
    the core surface: ``read_file`` cannot read images and its own description
    points at ``vision_analyze``, so an agent without it can't see screenshots
    or image files at all. Skills stay on because the essential
    ``hermes-agent`` skill (the agent's operating manual for driving,
    configuring, and troubleshooting Hermes) is always seeded — without
    ``skill_view`` it would be unloadable. Two layers enforce the selection:

    1. ``platform_toolsets["cli"] = ["file", "skills", "terminal", "vision"]``
       — an explicit list of
       configurable keys, which the resolver treats as authoritative
       (``has_explicit_config``) so default toolsets aren't re-expanded.
    2. ``agent.disabled_toolsets`` — a global hard-suppression list (applied last
       in ``_get_platform_tools``, overriding every other path including the
       non-configurable platform-toolset recovery that would otherwise re-add
       toolsets like ``kanban``). We list every known toolset except the ones we
       keep, guaranteeing a true blank slate regardless of platform/recovery
       quirks. The user re-enables any of them later via ``hermes tools`` (which
       rewrites ``platform_toolsets``) or by editing ``agent.disabled_toolsets``.
    """
    keep = {"file", "terminal", "vision", "skills"}
    config.setdefault("platform_toolsets", {})["cli"] = sorted(keep)

    try:
        from toolsets import TOOLSETS
        from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_plugin_toolset_keys

        all_keys = set()
        all_keys.update(k for k, _, _ in CONFIGURABLE_TOOLSETS)
        all_keys.update(_get_plugin_toolset_keys())
        # Plain (non-composite) TOOLSETS entries — catches recovered toolsets
        # like ``kanban`` that aren't in CONFIGURABLE_TOOLSETS but get re-added.
        for k, tdef in TOOLSETS.items():
            if k.startswith("hermes-"):
                continue  # platform composites — not user-facing toolsets
            if isinstance(tdef, dict) and tdef.get("includes"):
                continue  # composite groupings, not leaf toolsets
            if isinstance(tdef, dict) and tdef.get("posture"):
                continue  # posture toolsets (e.g. coding) are session-level
                # selections made by agent/coding_context.py — not permanent
                # user-facing disables. Adding them here causes model_tools
                # to subtract their tools (terminal, read_file, …) from the
                # minimal Blank Slate surface (#57315).
            all_keys.add(k)

        disabled = sorted(all_keys - keep)
        if disabled:
            config.setdefault("agent", {})["disabled_toolsets"] = disabled
    except Exception as exc:
        logger.debug("blank-slate disabled_toolsets computation skipped: %s", exc)


def _blank_slate_minimize_config(config: dict):
    """Turn OFF the optional config features for a Blank Slate install.

    Everything here is opt-in afterwards via ``hermes setup agent`` /
    ``hermes config set``. We keep only what's needed to run.
    """
    config.setdefault("agent", {})["max_turns"] = 90

    # Compression off — minimal footprint; user opts in if they want long sessions.
    config.setdefault("compression", {})["enabled"] = False

    # No automatic memory / user-profile capture.
    mem = config.setdefault("memory", {})
    mem["memory_enabled"] = False
    mem["user_profile_enabled"] = False

    # No filesystem checkpoints, no smart model routing, no auto session reset.
    config.setdefault("checkpoints", {})["enabled"] = False
    config.setdefault("smart_model_routing", {})["enabled"] = False
    config.setdefault("session_reset", {})["mode"] = "none"

    # Quiet, minimal display.
    config.setdefault("display", {})["tool_progress"] = "all"


def _run_blank_slate_setup(config: dict, hermes_home, is_existing: bool):
    """Blank Slate setup — start with everything off except the bare minimum.

    Forces only the essentials to run an agent (provider + model, the file and
    terminal toolsets) and turns every other tool/skill/plugin/MCP/config
    feature OFF. After applying that minimal baseline, the user chooses one of
    two paths:

      1. Start with everything disabled — finish now with the minimal agent.
      2. Walk through every configuration — opt each capability back in.

    Either way nothing is enabled that the user did not explicitly choose.
    """

    print()
    print_header("Blank Slate Setup")
    print_info("Everything starts OFF. First we force-enable only what's required")
    print_info("to run an agent, then you choose whether to stop there or walk")
    print_info("through enabling more — opting in to exactly what you want.")
    print_info("")
    print_info("Forced on: Provider & Model, File Operations, Terminal, Vision, Skills.")
    print_info("Everything else (web, browser, code exec, memory,")
    print_info("delegation, cron, plugins, MCP, …) starts disabled. The")
    print_info("essential `hermes-agent` skill is always kept so the agent")
    print_info("can help you drive and configure Hermes itself.")
    print()

    # ── Step 1: Provider & Model (REQUIRED — the agent cannot run without it) ──
    print_header("Step 1 — Provider & Model (required)")
    setup_model_provider(config)
    save_config(config)

    # ── Step 2: Terminal backend (where commands run — a core decision) ──
    print_header("Step 2 — Terminal Backend")
    setup_terminal_backend(config)

    # ── Step 3: Lock in the minimal toolset + minimized config knobs ──
    _blank_slate_minimal_toolsets(config)
    _blank_slate_minimize_config(config)
    save_config(config)
    print()
    print_success("Minimal baseline applied:")
    print_info("  Toolsets: file, terminal, vision, skills (everything else off)")
    print_info("  Compression, memory, checkpoints, smart routing: off")

    # ── The fork: stop here, or walk through enabling things ──
    print()
    print_header("How far do you want to go?")
    path = prompt_choice(
        "Your minimal agent is ready. What next?",
        [
            "Start with everything disabled — finish now (most minimal)",
            "Walk through all configurations — opt in to tools, skills, plugins, MCP",
        ],
        0,
    )

    if path == 0:
        save_config(config)
        # Blank Slate means no bundled skills; record the opt-out so future
        # `hermes update` runs don't re-inject them. Essential skills (the
        # `hermes-agent` operating manual) are still seeded by the sync.
        try:
            from tools.skills_sync import set_bundled_skills_opt_out, sync_skills
            set_bundled_skills_opt_out(True)
            sync_skills(quiet=True)
        except Exception as exc:
            logger.debug("blank-slate skill opt-out error: %s", exc)
        print()
        print_success("Blank Slate setup complete — minimal agent ready.")
        print_info("Enable anything later, on demand:")
        print_info("  Enable tools:        hermes tools")
        print_info("  Seed skills:         hermes skills opt-in --sync")
        print_info("  Add MCP servers:     hermes mcp add")
        print_info("  Enable plugins:      hermes plugins")
        print_info("  Tune agent settings: hermes setup agent")
        print()
        _print_setup_summary(config, hermes_home)
        return

    # ── Walkthrough path — opt in to each capability ──
    _blank_slate_walkthrough(config, hermes_home)


def _blank_slate_walkthrough(config: dict, hermes_home):
    """Opt-in walkthrough for Blank Slate: skills, tools, plugins, MCP, gateway."""
    from hermes_cli.config import load_config

    # ── Bundled skills — default to NONE, offer to seed all ──
    print()
    print_header("Bundled Skills")
    print_info("Blank Slate ships with NO bundled skills by default.")
    seed_skills = prompt_yes_no(
        "Seed the full bundled skill catalog? (No = start with zero skills)",
        default=False,
    )
    try:
        from tools.skills_sync import set_bundled_skills_opt_out, sync_skills
        if seed_skills:
            # Make sure no stale opt-out marker blocks the seed, then sync.
            set_bundled_skills_opt_out(False)
            result = sync_skills(quiet=True)
            copied = len(result.get("copied", [])) if isinstance(result, dict) else 0
            print_success(f"Seeded {copied} bundled skills.")
        else:
            set_bundled_skills_opt_out(True)
            # Essential skills (the `hermes-agent` operating manual) are
            # still seeded even for an opted-out profile.
            sync_skills(quiet=True)
            print_info("No skills seeded (except the essential `hermes-agent`")
            print_info("skill). A .no-bundled-skills marker keeps future")
            print_info("`hermes update` runs from re-injecting them. Opt back in any")
            print_info("time with `hermes skills opt-in --sync`.")
    except Exception as exc:
        logger.debug("blank-slate skill handling error: %s", exc)
        print_warning(f"Skill setup step encountered an error: {exc}")

    # ── Walk through enabling additional tools ──
    print()
    print_header("Tools")
    print_info("Pick exactly which additional toolsets to turn on.")
    print_info("(file and terminal are already on; leave the rest off if you want")
    print_info(" the most minimal agent.)")
    if prompt_yes_no("Open the tool selector to enable more tools?", default=False):
        try:
            from hermes_cli.tools_config import tools_command
            tools_command(first_install=False, config=config)
            # tools_command saves via its own load/save cycle — re-sync.
            _refreshed = load_config()
            config.clear()
            config.update(_refreshed)
        except Exception as exc:
            logger.debug("blank-slate tools_command error: %s", exc)
            print_warning(f"Tool selector encountered an error: {exc}")
    else:
        print_info("Keeping the minimal toolset. Add tools later with `hermes tools`.")

    # ── Built-in plugins (off unless chosen) ──
    print()
    print_header("Plugins")
    if prompt_yes_no("Review and enable built-in plugins now?", default=False):
        print_info("Manage plugins with `hermes plugins list` / `hermes plugins install`.")
    else:
        print_info("No plugins enabled. Add later with `hermes plugins`.")

    # ── MCP servers (off unless chosen) ──
    print()
    print_header("MCP Servers")
    if prompt_yes_no("Add an MCP server now?", default=False):
        print_info("Add servers with `hermes mcp add <name> --url ... | --command ...`.")
    else:
        print_info("No MCP servers configured. Add later with `hermes mcp add`.")

    # ── Optional messaging gateway ──
    print()
    if prompt_yes_no("Connect a messaging platform (Telegram, Discord, …)?", default=False):
        setup_gateway(config)

    save_config(config)

    print()
    print_success("Blank Slate setup complete — minimal agent ready.")
    print_info("  Enable more tools:   hermes tools")
    print_info("  Seed skills:         hermes skills opt-in --sync")
    print_info("  Add MCP servers:     hermes mcp add")
    print_info("  Tune agent settings: hermes setup agent")
    print()

    _print_setup_summary(config, hermes_home)


def _run_quick_setup(config: dict, hermes_home):
    """Quick setup — only configure items that are missing."""
    from hermes_cli.config import (
        get_missing_env_vars,
        get_missing_config_fields,
        check_config_version,
    )

    print()
    print_header("Quick Setup — Missing Items Only")

    # Check what's missing
    missing_required = [
        v for v in get_missing_env_vars(required_only=False) if v.get("is_required")
    ]
    missing_optional = [
        v for v in get_missing_env_vars(required_only=False) if not v.get("is_required")
    ]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()

    has_anything_missing = (
        missing_required
        or missing_optional
        or missing_config
        or current_ver < latest_ver
    )

    if not has_anything_missing:
        print_success("Everything is configured! Nothing to do.")
        print()
        print_info("Run 'hermes setup' and choose 'Full Setup' to reconfigure,")
        print_info("or pick a specific section from the menu.")
        return

    # Handle missing required env vars
    if missing_required:
        print()
        print_info(f"{len(missing_required)} required setting(s) missing:")
        for var in missing_required:
            print(f"     • {var['name']}")
        print()

        for var in missing_required:
            print()
            print(color(f"  {var['name']}", Colors.CYAN))
            print_info(f"  {var.get('description', '')}")
            if var.get("url"):
                print_info(f"  Get key at: {var['url']}")

            if var.get("password"):
                value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
            else:
                value = prompt(f"  {var.get('prompt', var['name'])}")

            if value:
                save_env_value(var["name"], value)
                print_success(f"  Saved {var['name']}")
            else:
                print_warning(f"  Skipped {var['name']}")

    # Split missing optional vars by category
    missing_tools = [v for v in missing_optional if v.get("category") == "tool"]
    missing_messaging = [
        v
        for v in missing_optional
        if v.get("category") == "messaging" and not v.get("advanced")
    ]

    # ── Tool API keys (checklist) ──
    if missing_tools:
        print()
        print_header("Tool API Keys")

        checklist_labels = []
        for var in missing_tools:
            tools = var.get("tools", [])
            tools_str = f" → {', '.join(tools[:2])}" if tools else ""
            checklist_labels.append(f"{var.get('description', var['name'])}{tools_str}")

        selected_indices = prompt_checklist(
            "Which tools would you like to configure?",
            checklist_labels,
        )

        for idx in selected_indices:
            var = missing_tools[idx]
            _prompt_api_key(var)

    # ── Messaging platforms (checklist then prompt for selected) ──
    if missing_messaging:
        print()
        print_header("Messaging Platforms")
        print_info("Connect Hermes to messaging apps to chat from anywhere.")
        print_info("You can configure these later with 'hermes setup gateway'.")

        # Group by platform (preserving order)
        platform_order = []
        platforms = {}
        for var in missing_messaging:
            name = var["name"]
            if "TELEGRAM" in name:
                plat = "Telegram"
            elif "DISCORD" in name:
                plat = "Discord"
            elif "SLACK" in name:
                plat = "Slack"
            else:
                continue
            if plat not in platforms:
                platform_order.append(plat)
            platforms.setdefault(plat, []).append(var)

        platform_labels = [
            {
                "Telegram": "📱 Telegram",
                "Discord": "💬 Discord",
                "Slack": "💼 Slack",
            }.get(p, p)
            for p in platform_order
        ]

        selected_indices = prompt_checklist(
            "Which platforms would you like to set up?",
            platform_labels,
        )

        for idx in selected_indices:
            plat = platform_order[idx]
            vars_list = platforms[plat]
            emoji = {"Telegram": "📱", "Discord": "💬", "Slack": "💼"}.get(plat, "")
            print()
            print(color(f"  ─── {emoji} {plat} ───", Colors.CYAN))
            print()
            for var in vars_list:
                print_info(f"  {var.get('description', '')}")
                if var.get("url"):
                    print_info(f"  {var['url']}")
                if var.get("password"):
                    value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
                else:
                    value = prompt(f"  {var.get('prompt', var['name'])}")
                if value:
                    save_env_value(var["name"], value)
                    print_success("  ✓ Saved")
                else:
                    print_warning("  Skipped")
                print()

    # Handle missing config fields
    if missing_config:
        print()
        print_info(
            f"Adding {len(missing_config)} new config option(s) with defaults..."
        )
        for field in missing_config:
            print_success(f"  Added {field['key']} = {field['default']}")

        # Update config version
        config["_config_version"] = latest_ver
        save_config(config)

    # Jump to summary
    _print_setup_summary(config, hermes_home)
