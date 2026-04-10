"""
Interactive setup wizard for Hermes Agent.

Modular wizard with independently-runnable sections:
  1. Model & Provider — choose your AI provider and model
  2. Terminal Backend — where your agent runs commands
  3. Agent Settings — iterations, compression, session reset
  4. Messaging Platforms — connect Telegram, Discord, etc.
  5. Tools — configure TTS, web search, image generation, etc.

Config files are stored in ~/.hermes/ for easy access.
"""

import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from hermes_cli.nous_subscription import (
    apply_nous_provider_defaults,
    get_nous_subscription_features,
)
from tools.tool_backend_helpers import managed_nous_tools_enabled
from hermes_constants import get_optional_skills_dir

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_DOCS_BASE = "https://hermes-agent.nousresearch.com/docs"


def _model_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    current_model = config.get("model")
    if isinstance(current_model, dict):
        return dict(current_model)
    if isinstance(current_model, str) and current_model.strip():
        return {"default": current_model.strip()}
    return {}


def _set_default_model(config: Dict[str, Any], model_name: str) -> None:
    if not model_name:
        return
    model_cfg = _model_config_dict(config)
    model_cfg["default"] = model_name
    config["model"] = model_cfg


def _get_credential_pool_strategies(config: Dict[str, Any]) -> Dict[str, str]:
    strategies = config.get("credential_pool_strategies")
    return dict(strategies) if isinstance(strategies, dict) else {}


def _set_credential_pool_strategy(config: Dict[str, Any], provider: str, strategy: str) -> None:
    if not provider:
        return
    strategies = _get_credential_pool_strategies(config)
    strategies[provider] = strategy
    config["credential_pool_strategies"] = strategies


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
        "claude-sonnet-4.6",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-2.5-pro",
        "grok-code-fast-1",
    ],
    "gemini": [
        "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        "gemma-4-31b-it", "gemma-4-26b-it",
    ],
    "zai": ["glm-5", "glm-4.7", "glm-4.5", "glm-4.5-flash"],
    "kimi-coding": ["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview"],
    "minimax": ["MiniMax-M1", "MiniMax-M1-40k", "MiniMax-M1-80k", "MiniMax-M1-128k", "MiniMax-M1-256k", "MiniMax-M2.5", "MiniMax-M2.7"],
    "minimax-cn": ["MiniMax-M1", "MiniMax-M1-40k", "MiniMax-M1-80k", "MiniMax-M1-128k", "MiniMax-M1-256k", "MiniMax-M2.5", "MiniMax-M2.7"],
    "ai-gateway": ["anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6", "openai/gpt-5", "google/gemini-3-flash"],
    "kilocode": ["anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "google/gemini-3-pro-preview", "google/gemini-3-flash-preview"],
    "opencode-zen": ["gpt-5.4", "gpt-5.3-codex", "claude-sonnet-4-6", "gemini-3-flash", "glm-5", "kimi-k2.5", "minimax-m2.7"],
    "opencode-go": ["glm-5", "kimi-k2.5", "mimo-v2-pro", "mimo-v2-omni", "minimax-m2.5", "minimax-m2.7"],
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


def _set_reasoning_effort(config: Dict[str, Any], effort: str) -> None:
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
        config["agent"] = agent_cfg
    agent_cfg["reasoning_effort"] = effort


def _setup_copilot_reasoning_selection(
    config: Dict[str, Any],
    model_id: str,
    prompt_choice,
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: str = "",
) -> None:
    from hermes_cli.models import github_model_reasoning_efforts, normalize_copilot_model_id

    normalized_model = normalize_copilot_model_id(
        model_id,
        catalog=catalog,
        api_key=api_key,
    ) or model_id
    efforts = github_model_reasoning_efforts(normalized_model, catalog=catalog, api_key=api_key)
    if not efforts:
        return

    current_effort = _current_reasoning_effort(config)
    choices = list(efforts) + ["Disable reasoning", f"Keep current ({current_effort or 'default'})"]

    if current_effort == "none":
        default_idx = len(efforts)
    elif current_effort in efforts:
        default_idx = efforts.index(current_effort)
    elif "medium" in efforts:
        default_idx = efforts.index("medium")
    else:
        default_idx = len(choices) - 1

    effort_idx = prompt_choice("Select reasoning effort:", choices, default_idx)
    if effort_idx < len(efforts):
        _set_reasoning_effort(config, efforts[effort_idx])
    elif effort_idx == len(efforts):
        _set_reasoning_effort(config, "none")


def _setup_provider_model_selection(config, provider_id, current_model, prompt_choice, prompt_fn):
    """Model selection for API-key providers with live /models detection.

    Tries the provider's /models endpoint first.  Falls back to a
    hardcoded default list with a warning if the endpoint is unreachable.
    Always offers a 'Custom model' escape hatch.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    from hermes_cli.config import get_env_value
    from hermes_cli.models import (
        copilot_model_api_mode,
        fetch_api_models,
        fetch_github_model_catalog,
        normalize_copilot_model_id,
        normalize_opencode_model_id,
        opencode_model_api_mode,
    )

    pconfig = PROVIDER_REGISTRY[provider_id]
    is_copilot_catalog_provider = provider_id in {"copilot", "copilot-acp"}

    # Resolve API key and base URL for the probe
    if is_copilot_catalog_provider:
        api_key = ""
        if provider_id == "copilot":
            creds = resolve_api_key_provider_credentials(provider_id)
            api_key = creds.get("api_key", "")
            base_url = creds.get("base_url", "") or pconfig.inference_base_url
        else:
            try:
                creds = resolve_api_key_provider_credentials("copilot")
                api_key = creds.get("api_key", "")
            except Exception:
                pass
            base_url = pconfig.inference_base_url
        catalog = fetch_github_model_catalog(api_key)
        current_model = normalize_copilot_model_id(
            current_model,
            catalog=catalog,
            api_key=api_key,
        ) or current_model
    else:
        api_key = ""
        for ev in pconfig.api_key_env_vars:
            api_key = get_env_value(ev) or os.getenv(ev, "")
            if api_key:
                break
        base_url_env = pconfig.base_url_env_var or ""
        base_url = (get_env_value(base_url_env) if base_url_env else "") or pconfig.inference_base_url
        catalog = None

    # Try live /models endpoint
    if is_copilot_catalog_provider and catalog:
        live_models = [item.get("id", "") for item in catalog if item.get("id")]
    else:
        live_models = fetch_api_models(api_key, base_url)

    if live_models:
        provider_models = live_models
        print_info(f"Found {len(live_models)} model(s) from {pconfig.name} API")
    else:
        fallback_provider_id = "copilot" if provider_id == "copilot-acp" else provider_id
        provider_models = _DEFAULT_PROVIDER_MODELS.get(fallback_provider_id, [])
        if provider_models:
            print_warning(
                f"Could not auto-detect models from {pconfig.name} API — showing defaults.\n"
                f"    Use \"Custom model\" if the model you expect isn't listed."
            )

    if provider_id in {"opencode-zen", "opencode-go"}:
        provider_models = [normalize_opencode_model_id(provider_id, mid) for mid in provider_models]
        current_model = normalize_opencode_model_id(provider_id, current_model)
        provider_models = list(dict.fromkeys(mid for mid in provider_models if mid))

    model_choices = list(provider_models)
    model_choices.append("Custom model")
    model_choices.append(f"Keep current ({current_model})")

    keep_idx = len(model_choices) - 1
    model_idx = prompt_choice("Select default model:", model_choices, keep_idx)

    selected_model = current_model

    if model_idx < len(provider_models):
        selected_model = provider_models[model_idx]
        if is_copilot_catalog_provider:
            selected_model = normalize_copilot_model_id(
                selected_model,
                catalog=catalog,
                api_key=api_key,
            ) or selected_model
        elif provider_id in {"opencode-zen", "opencode-go"}:
            selected_model = normalize_opencode_model_id(provider_id, selected_model)
        _set_default_model(config, selected_model)
    elif model_idx == len(provider_models):
        custom = prompt_fn("Enter model name")
        if custom:
            if is_copilot_catalog_provider:
                selected_model = normalize_copilot_model_id(
                    custom,
                    catalog=catalog,
                    api_key=api_key,
                ) or custom
            elif provider_id in {"opencode-zen", "opencode-go"}:
                selected_model = normalize_opencode_model_id(provider_id, custom)
            else:
                selected_model = custom
            _set_default_model(config, selected_model)
    else:
        # "Keep current" selected — validate it's compatible with the new
        # provider.  OpenRouter-formatted names (containing "/") won't work
        # on direct-API providers and would silently break the gateway.
        if "/" in (current_model or "") and provider_models:
            print_warning(
                f"Current model \"{current_model}\" looks like an OpenRouter model "
                f"and won't work with {pconfig.name}. "
                f"Switching to {provider_models[0]}."
            )
            selected_model = provider_models[0]
            _set_default_model(config, provider_models[0])

    if provider_id == "copilot" and selected_model:
        model_cfg = _model_config_dict(config)
        model_cfg["api_mode"] = copilot_model_api_mode(
            selected_model,
            catalog=catalog,
            api_key=api_key,
        )
        config["model"] = model_cfg
        _setup_copilot_reasoning_selection(
            config,
            selected_model,
            prompt_choice,
            catalog=catalog,
            api_key=api_key,
        )
    elif provider_id in {"opencode-zen", "opencode-go"} and selected_model:
        model_cfg = _model_config_dict(config)
        model_cfg["api_mode"] = opencode_model_api_mode(provider_id, selected_model)
        config["model"] = model_cfg


# Import config helpers
from hermes_cli.config import (
    get_hermes_home,
    get_config_path,
    get_env_path,
    load_config,
    save_config,
    save_env_value,
    get_env_value,
    ensure_hermes_home,
)
# display_hermes_home imported lazily at call sites (stale-module safety during hermes update)

from hermes_cli.colors import Colors, color


def print_header(title: str):
    """Print a section header."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def print_info(text: str):
    """Print info text."""
    print(color(f"  {text}", Colors.DIM))


def print_success(text: str):
    """Print success message."""
    print(color(f"✓ {text}", Colors.GREEN))


def print_warning(text: str):
    """Print warning message."""
    print(color(f"⚠ {text}", Colors.YELLOW))


def print_error(text: str):
    """Print error message."""
    print(color(f"✗ {text}", Colors.RED))


def is_interactive_stdin() -> bool:
    """Return True when stdin looks like a usable interactive TTY."""
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return False
    try:
        return bool(stdin.isatty())
    except Exception:
        return False


def print_noninteractive_setup_guidance(reason: str | None = None) -> None:
    """Print guidance for headless/non-interactive setup flows."""
    print()
    print(color("⚕ Hermes Setup — Non-interactive mode", Colors.CYAN, Colors.BOLD))
    print()
    if reason:
        print_info(reason)
    print_info("The interactive wizard cannot be used here.")
    print()
    print_info("Configure Hermes using environment variables or config commands:")
    print_info("  hermes config set model.provider custom")
    print_info("  hermes config set model.base_url http://localhost:8080/v1")
    print_info("  hermes config set model.default your-model-name")
    print()
    print_info("Or set OPENROUTER_API_KEY / OPENAI_API_KEY in your environment.")
    print_info("Run 'hermes setup' in an interactive terminal to use the full wizard.")
    print()


def prompt(question: str, default: str = None, password: bool = False) -> str:
    """Prompt for input with optional default."""
    if default:
        display = f"{question} [{default}]: "
    else:
        display = f"{question}: "

    try:
        if password:
            import getpass

            value = getpass.getpass(color(display, Colors.YELLOW))
        else:
            value = input(color(display, Colors.YELLOW))

        return value.strip() or default or ""
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)


def _curses_prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Single-select menu using curses to avoid simple_term_menu rendering bugs."""
    try:
        import curses
        result_holder = [default]

        def _curses_menu(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
            cursor = default
            scroll_offset = 0

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()

                # Rows available for list items: rows 2..(max_y-2) inclusive.
                visible = max(1, max_y - 3)

                # Scroll the viewport so the cursor is always visible.
                if cursor < scroll_offset:
                    scroll_offset = cursor
                elif cursor >= scroll_offset + visible:
                    scroll_offset = cursor - visible + 1
                scroll_offset = max(0, min(scroll_offset, max(0, len(choices) - visible)))

                try:
                    stdscr.addnstr(
                        0,
                        0,
                        question,
                        max_x - 1,
                        curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0),
                    )
                except curses.error:
                    pass

                for row, i in enumerate(range(scroll_offset, min(scroll_offset + visible, len(choices)))):
                    y = row + 2
                    if y >= max_y - 1:
                        break
                    arrow = "→" if i == cursor else " "
                    line = f" {arrow}  {choices[i]}"
                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(1)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass

                stdscr.refresh()
                key = stdscr.getch()
                if key in (curses.KEY_UP, ord("k")):
                    cursor = (cursor - 1) % len(choices)
                elif key in (curses.KEY_DOWN, ord("j")):
                    cursor = (cursor + 1) % len(choices)
                elif key in (curses.KEY_ENTER, 10, 13):
                    result_holder[0] = cursor
                    return
                elif key in (27, ord("q")):
                    return

        curses.wrapper(_curses_menu)
        return result_holder[0]
    except Exception:
        return -1



def prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Prompt for a choice from a list with arrow key navigation.

    Escape keeps the current default (skips the question).
    Ctrl+C exits the wizard.
    """
    idx = _curses_prompt_choice(question, choices, default)
    if idx >= 0:
        if idx == default:
            print_info("  Skipped (keeping current)")
            print()
            return default
        print()
        return idx

    print(color(question, Colors.YELLOW))
    for i, choice in enumerate(choices):
        marker = "●" if i == default else "○"
        if i == default:
            print(color(f"  {marker} {choice}", Colors.GREEN))
        else:
            print(f"  {marker} {choice}")

    print_info(f"  Enter for default ({default + 1})  Ctrl+C to exit")

    while True:
        try:
            value = input(
                color(f"  Select [1-{len(choices)}] ({default + 1}): ", Colors.DIM)
            )
            if not value:
                return default
            idx = int(value) - 1
            if 0 <= idx < len(choices):
                return idx
            print_error(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print_error("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no. Ctrl+C exits, empty input returns default."""
    default_str = "Y/n" if default else "y/N"

    while True:
        try:
            value = (
                input(color(f"{question} [{default_str}]: ", Colors.YELLOW))
                .strip()
                .lower()
            )
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)

        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print_error("Please enter 'y' or 'n'")


def prompt_checklist(title: str, items: list, pre_selected: list = None) -> list:
    """
    Display a multi-select checklist and return the indices of selected items.

    Each item in `items` is a display string. `pre_selected` is a list of
    indices that should be checked by default. A "Continue →" option is
    appended at the end — the user toggles items with Space and confirms
    with Enter on "Continue →".

    Falls back to a numbered toggle interface when simple_term_menu is
    unavailable.

    Returns:
        List of selected indices (not including the Continue option).
    """
    if pre_selected is None:
        pre_selected = []

    from hermes_cli.curses_ui import curses_checklist

    chosen = curses_checklist(
        title,
        items,
        set(pre_selected),
        cancel_returns=set(pre_selected),
    )
    return sorted(chosen)


def _prompt_api_key(var: dict):
    """Display a nicely formatted API key input screen for a single env var."""
    tools = var.get("tools", [])
    tools_str = ", ".join(tools[:3])
    if len(tools) > 3:
        tools_str += f", +{len(tools) - 3} more"

    print()
    print(color(f"  ─── {var.get('description', var['name'])} ───", Colors.CYAN))
    print()
    if tools_str:
        print_info(f"  Enables: {tools_str}")
    if var.get("url"):
        print_info(f"  Get your key at: {var['url']}")
    print()

    if var.get("password"):
        value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
    else:
        value = prompt(f"  {var.get('prompt', var['name'])}")

    if value:
        save_env_value(var["name"], value)
        print_success("  ✓ Saved")
    else:
        print_warning("  Skipped (configure later with 'hermes setup')")


def _print_setup_summary(config: dict, hermes_home):
    """Print the setup completion summary."""
    # Tool availability summary
    print()
    print_header("Tool Availability Summary")

    tool_status = []
    subscription_features = get_nous_subscription_features(config)

    # Vision — use the same runtime resolver as the actual vision tools
    try:
        from agent.auxiliary_client import get_available_vision_backends

        _vision_backends = get_available_vision_backends()
    except Exception:
        _vision_backends = []

    if _vision_backends:
        tool_status.append(("Vision (image analysis)", True, None))
    else:
        tool_status.append(("Vision (image analysis)", False, "run 'hermes setup' to configure"))

    # Mixture of Agents — requires OpenRouter specifically (calls multiple models)
    if get_env_value("OPENROUTER_API_KEY"):
        tool_status.append(("Mixture of Agents", True, None))
    else:
        tool_status.append(("Mixture of Agents", False, "OPENROUTER_API_KEY"))

    # Web tools (Exa, Parallel, Firecrawl, or Tavily)
    if subscription_features.web.managed_by_nous:
        tool_status.append(("Web Search & Extract (Nous subscription)", True, None))
    elif subscription_features.web.available:
        label = "Web Search & Extract"
        if subscription_features.web.current_provider:
            label = f"Web Search & Extract ({subscription_features.web.current_provider})"
        tool_status.append((label, True, None))
    else:
        tool_status.append(("Web Search & Extract", False, "EXA_API_KEY, PARALLEL_API_KEY, FIRECRAWL_API_KEY/FIRECRAWL_API_URL, or TAVILY_API_KEY"))

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
            missing_browser_hint = "npm install -g agent-browser"
        tool_status.append(
            ("Browser Automation", False, missing_browser_hint)
        )

    # FAL (image generation)
    if subscription_features.image_gen.managed_by_nous:
        tool_status.append(("Image Generation (Nous subscription)", True, None))
    elif subscription_features.image_gen.available:
        tool_status.append(("Image Generation", True, None))
    else:
        tool_status.append(("Image Generation", False, "FAL_KEY"))

    # TTS — show configured provider
    tts_provider = config.get("tts", {}).get("provider", "edge")
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
    elif tts_provider == "neutts":
        try:
            import importlib.util
            neutts_ok = importlib.util.find_spec("neutts") is not None
        except Exception:
            neutts_ok = False
        if neutts_ok:
            tool_status.append(("Text-to-Speech (NeuTTS local)", True, None))
        else:
            tool_status.append(("Text-to-Speech (NeuTTS — not installed)", False, "run 'hermes setup tts'"))
    else:
        tool_status.append(("Text-to-Speech (Edge TTS)", True, None))

    if subscription_features.modal.managed_by_nous:
        tool_status.append(("Modal Execution (Nous subscription)", True, None))
    elif config.get("terminal", {}).get("backend") == "modal":
        if subscription_features.modal.direct_override:
            tool_status.append(("Modal Execution (direct Modal)", True, None))
        else:
            tool_status.append(("Modal Execution", False, "run 'hermes setup terminal'"))
    elif managed_nous_tools_enabled() and subscription_features.nous_auth_present:
        tool_status.append(("Modal Execution (optional via Nous subscription)", True, None))

    # Tinker + WandB (RL training)
    if get_env_value("TINKER_API_KEY") and get_env_value("WANDB_API_KEY"):
        tool_status.append(("RL Training (Tinker)", True, None))
    elif get_env_value("TINKER_API_KEY"):
        tool_status.append(("RL Training (Tinker)", False, "WANDB_API_KEY"))
    else:
        tool_status.append(("RL Training (Tinker)", False, "TINKER_API_KEY"))

    # Home Assistant
    if get_env_value("HASS_TOKEN"):
        tool_status.append(("Smart Home (Home Assistant)", True, None))

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
    terminal["container_persistent"] = persist_str.lower() in ("yes", "true", "y", "1")

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


# Tool categories and provider config are now in tools_config.py (shared
# between `hermes tools` and `hermes setup tools`).


# =============================================================================
# Section 1: Model & Provider Configuration
# =============================================================================



def setup_model_provider(config: dict, *, quick: bool = False):
    """Configure the inference provider and default model.

    Delegates to ``cmd_model()`` (the same flow used by ``hermes model``)
    for provider selection, credential prompting, and model picking.
    This ensures a single code path for all provider setup — any new
    provider added to ``hermes model`` is automatically available here.

    When *quick* is True, skips credential rotation, vision, and TTS
    configuration — used by the streamlined first-time quick setup.
    """
    from hermes_cli.config import load_config, save_config

    print_header("Inference Provider")
    print_info("Choose how to connect to your main chat model.")
    print_info(f"   Guide: {_DOCS_BASE}/integrations/providers")
    print()

    # Delegate to the shared hermes model flow — handles provider picker,
    # credential prompting, model selection, and config persistence.
    from hermes_cli.main import select_provider_and_model
    try:
        select_provider_and_model()
    except (SystemExit, KeyboardInterrupt):
        print()
        print_info("Provider setup skipped.")
    except Exception as exc:
        logger.debug("select_provider_and_model error during setup: %s", exc)
        print_warning(f"Provider setup encountered an error: {exc}")
        print_info("You can try again later with: hermes model")

    # Re-sync the wizard's config dict from what cmd_model saved to disk.
    # This is critical: cmd_model writes to disk via its own load/save cycle,
    # and the wizard's final save_config(config) must not overwrite those
    # changes with stale values (#4172).
    _refreshed = load_config()
    config["model"] = _refreshed.get("model", config.get("model"))
    if _refreshed.get("custom_providers"):
        config["custom_providers"] = _refreshed["custom_providers"]

    # Derive the selected provider for downstream steps (vision setup).
    selected_provider = None
    _m = config.get("model")
    if isinstance(_m, dict):
        selected_provider = _m.get("provider")

    nous_subscription_selected = selected_provider == "nous"

    # ── Same-provider fallback & rotation setup (full setup only) ──
    if not quick and _supports_same_provider_pool_setup(selected_provider):
        try:
            from types import SimpleNamespace
            from agent.credential_pool import load_pool
            from hermes_cli.auth_commands import auth_add_command

            pool = load_pool(selected_provider)
            entries = pool.entries()
            entry_count = len(entries)
            manual_count = sum(1 for entry in entries if str(getattr(entry, "source", "")).startswith("manual"))
            auto_count = entry_count - manual_count
            print()
            print_header("Same-Provider Fallback & Rotation")
            print_info(
                "Hermes can keep multiple credentials for one provider and rotate between"
            )
            print_info(
                "them when a credential is exhausted or rate-limited. This preserves"
            )
            print_info(
                "your primary provider while reducing interruptions from quota issues."
            )
            print()
            if auto_count > 0:
                print_info(
                    f"Current pooled credentials for {selected_provider}: {entry_count} "
                    f"({manual_count} manual, {auto_count} auto-detected from env/shared auth)"
                )
            else:
                print_info(f"Current pooled credentials for {selected_provider}: {entry_count}")

            while prompt_yes_no("Add another credential for same-provider fallback?", False):
                auth_add_command(
                    SimpleNamespace(
                        provider=selected_provider,
                        auth_type="",
                        label=None,
                        api_key=None,
                        portal_url=None,
                        inference_url=None,
                        client_id=None,
                        scope=None,
                        no_browser=False,
                        timeout=15.0,
                        insecure=False,
                        ca_bundle=None,
                        min_key_ttl_seconds=5 * 60,
                    )
                )
                pool = load_pool(selected_provider)
                entry_count = len(pool.entries())
                print_info(f"Provider pool now has {entry_count} credential(s).")

            if entry_count > 1:
                strategy_labels = [
                    "Fill-first / sticky — keep using the first healthy credential until it is exhausted",
                    "Round robin — rotate to the next healthy credential after each selection",
                    "Random — pick a random healthy credential each time",
                ]
                current_strategy = _get_credential_pool_strategies(config).get(selected_provider, "fill_first")
                default_strategy_idx = {
                    "fill_first": 0,
                    "round_robin": 1,
                    "random": 2,
                }.get(current_strategy, 0)
                strategy_idx = prompt_choice(
                    "Select same-provider rotation strategy:",
                    strategy_labels,
                    default_strategy_idx,
                )
                strategy_value = ["fill_first", "round_robin", "random"][strategy_idx]
                _set_credential_pool_strategy(config, selected_provider, strategy_value)
                print_success(f"Saved {selected_provider} rotation strategy: {strategy_value}")
            else:
                _set_credential_pool_strategy(config, selected_provider, "fill_first")
        except Exception as exc:
            logger.debug("Could not configure same-provider fallback in setup: %s", exc)

    # ── Vision & Image Analysis Setup (full setup only) ──
    if quick:
        _vision_needs_setup = False
    else:
        try:
            from agent.auxiliary_client import get_available_vision_backends
            _vision_backends = set(get_available_vision_backends())
        except Exception:
            _vision_backends = set()

        _vision_needs_setup = not bool(_vision_backends)

        if selected_provider in _vision_backends:
            _vision_needs_setup = False

    if _vision_needs_setup:
        _prov_names = {
            "nous-api": "Nous Portal API key",
            "copilot": "GitHub Copilot",
            "copilot-acp": "GitHub Copilot ACP",
            "zai": "Z.AI / GLM",
            "kimi-coding": "Kimi / Moonshot",
            "minimax": "MiniMax",
            "minimax-cn": "MiniMax CN",
            "anthropic": "Anthropic",
            "ai-gateway": "AI Gateway",
            "custom": "your custom endpoint",
        }
        _prov_display = _prov_names.get(selected_provider, selected_provider or "your provider")

        print()
        print_header("Vision & Image Analysis (optional)")
        print_info(f"Vision uses a separate multimodal backend. {_prov_display}")
        print_info("doesn't currently provide one Hermes can auto-use for vision,")
        print_info("so choose a backend now or skip and configure later.")
        print()

        _vision_choices = [
            "OpenRouter — uses Gemini (free tier at openrouter.ai/keys)",
            "OpenAI-compatible endpoint — base URL, API key, and vision model",
            "Skip for now",
        ]
        _vision_idx = prompt_choice("Configure vision:", _vision_choices, 2)

        if _vision_idx == 0:  # OpenRouter
            _or_key = prompt("  OpenRouter API key", password=True).strip()
            if _or_key:
                save_env_value("OPENROUTER_API_KEY", _or_key)
                print_success("OpenRouter key saved — vision will use Gemini")
            else:
                print_info("Skipped — vision won't be available")
        elif _vision_idx == 1:  # OpenAI-compatible endpoint
            _base_url = prompt("  Base URL (blank for OpenAI)").strip() or "https://api.openai.com/v1"
            _api_key_label = "  API key"
            if "api.openai.com" in _base_url.lower():
                _api_key_label = "  OpenAI API key"
            _oai_key = prompt(_api_key_label, password=True).strip()
            if _oai_key:
                save_env_value("OPENAI_API_KEY", _oai_key)
                # Save vision base URL to config (not .env — only secrets go there)
                _vaux = config.setdefault("auxiliary", {}).setdefault("vision", {})
                _vaux["base_url"] = _base_url
                if "api.openai.com" in _base_url.lower():
                    _oai_vision_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
                    _vm_choices = _oai_vision_models + ["Use default (gpt-4o-mini)"]
                    _vm_idx = prompt_choice("Select vision model:", _vm_choices, 0)
                    _selected_vision_model = (
                        _oai_vision_models[_vm_idx]
                        if _vm_idx < len(_oai_vision_models)
                        else "gpt-4o-mini"
                    )
                else:
                    _selected_vision_model = prompt("  Vision model (blank = use main/custom default)").strip()
                save_env_value("AUXILIARY_VISION_MODEL", _selected_vision_model)
                print_success(
                    f"Vision configured with {_base_url}"
                    + (f" ({_selected_vision_model})" if _selected_vision_model else "")
                )
            else:
                print_info("Skipped — vision won't be available")
        else:
            print_info("Skipped — add later with 'hermes setup' or configure AUXILIARY_VISION_* settings")


    if selected_provider == "nous" and nous_subscription_selected:
        changed_defaults = apply_nous_provider_defaults(config)
        current_tts = str(config.get("tts", {}).get("provider") or "edge")
        if "tts" in changed_defaults:
            print_success("TTS provider set to: OpenAI TTS via your Nous subscription")
        else:
            print_info(f"Keeping your existing TTS provider: {current_tts}")

    save_config(config)

    if not quick and selected_provider != "nous":
        _setup_tts_provider(config)


# =============================================================================
# Section 1b: TTS Provider Configuration
# =============================================================================


def _check_espeak_ng() -> bool:
    """Check if espeak-ng is installed."""
    import shutil
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
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "neutts[all]", "--quiet"],
            check=True, timeout=300,
        )
        print_success("neutts installed successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print_error(f"Failed to install neutts: {e}")
        print_info("Try manually: python -m pip install -U neutts[all]")
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
        "minimax": "MiniMax TTS",
        "neutts": "NeuTTS",
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
            "MiniMax TTS (high quality with voice cloning, needs API key)",
            "NeuTTS (local on-device, free, ~300MB model download)",
        ]
    )
    providers.extend(["edge", "elevenlabs", "openai", "minimax", "neutts"])
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
            import importlib.util
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
    import shutil

    print_header("Terminal Backend")
    print_info("Choose where Hermes runs shell commands and code.")
    print_info("This affects tool execution, file access, and isolation.")
    print_info(f"   Guide: {_DOCS_BASE}/developer-guide/environments")
    print()

    current_backend = config.get("terminal", {}).get("backend", "local")
    is_linux = _platform.system() == "Linux"

    # Build backend choices with descriptions
    terminal_choices = [
        "Local - run directly on this machine (default)",
        "Docker - isolated container with configurable resources",
        "Modal - serverless cloud sandbox",
        "SSH - run on a remote machine",
        "Daytona - persistent cloud development environment",
    ]
    idx_to_backend = {0: "local", 1: "docker", 2: "modal", 3: "ssh", 4: "daytona"}
    backend_to_idx = {"local": 0, "docker": 1, "modal": 2, "ssh": 3, "daytona": 4}

    next_idx = 5
    if is_linux:
        terminal_choices.append("Singularity/Apptainer - HPC-friendly container")
        idx_to_backend[next_idx] = "singularity"
        backend_to_idx["singularity"] = next_idx
        next_idx += 1

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

        # CWD for messaging
        print()
        print_info("Working directory for messaging sessions:")
        print_info("  When using Hermes via Telegram/Discord, this is where")
        print_info(
            "  the agent starts. CLI mode always starts in the current directory."
        )
        current_cwd = config.get("terminal", {}).get("cwd", "")
        cwd = prompt("  Messaging working directory", current_cwd or str(Path.home()))
        if cwd:
            config["terminal"]["cwd"] = cwd

        # Sudo support
        print()
        existing_sudo = get_env_value("SUDO_PASSWORD")
        if existing_sudo:
            print_info("Sudo password: configured")
        else:
            if prompt_yes_no(
                "Enable sudo support? (stores password for apt install, etc.)", False
            ):
                sudo_pass = prompt("  Sudo password", password=True)
                if sudo_pass:
                    save_env_value("SUDO_PASSWORD", sudo_pass)
                    print_success("Sudo password saved")

    elif selected_backend == "docker":
        print_success("Terminal backend: Docker")

        # Check if Docker is available
        docker_bin = shutil.which("docker")
        if not docker_bin:
            print_warning("Docker not found in PATH!")
            print_info("Install Docker: https://docs.docker.com/get-docker/")
        else:
            print_info(f"Docker found: {docker_bin}")

        # Docker image
        current_image = config.get("terminal", {}).get(
            "docker_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )
        image = prompt("  Docker image", current_image)
        config["terminal"]["docker_image"] = image
        save_env_value("TERMINAL_DOCKER_IMAGE", image)

        _prompt_container_resources(config)

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

        current_image = config.get("terminal", {}).get(
            "singularity_image", "docker://nikolaik/python-nodejs:python3.11-nodejs20"
        )
        image = prompt("  Container image", current_image)
        config["terminal"]["singularity_image"] = image
        save_env_value("TERMINAL_SINGULARITY_IMAGE", image)

        _prompt_container_resources(config)

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
        modal_mode = normalize_modal_mode(config.get("terminal", {}).get("modal_mode"))
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
                import subprocess

                uv_bin = shutil.which("uv")
                if uv_bin:
                    result = subprocess.run(
                        [
                            uv_bin,
                            "pip",
                            "install",
                            "--python",
                            sys.executable,
                            "modal",
                        ],
                        capture_output=True,
                        text=True,
                    )
                else:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "modal"],
                        capture_output=True,
                        text=True,
                    )
                if result.returncode == 0:
                    print_success("modal SDK installed")
                else:
                    print_warning("Install failed — run manually: pip install modal")

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

        _prompt_container_resources(config)

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
            import subprocess

            uv_bin = shutil.which("uv")
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "--python", sys.executable, "daytona"],
                    capture_output=True,
                    text=True,
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "daytona"],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                print_success("daytona SDK installed")
            else:
                print_warning("Install failed — run manually: pip install daytona")
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

        # Daytona image
        current_image = config.get("terminal", {}).get(
            "daytona_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )
        image = prompt("  Sandbox image", current_image)
        config["terminal"]["daytona_image"] = image
        save_env_value("TERMINAL_DAYTONA_IMAGE", image)

        _prompt_container_resources(config)

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
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
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
    save_config(config)
    print()
    print_success(f"Terminal backend set to: {selected_backend}")


# =============================================================================
# Section 3: Agent Settings
# =============================================================================


def _apply_default_agent_settings(config: dict):
    """Apply recommended defaults for all agent settings without prompting."""
    config.setdefault("agent", {})["max_turns"] = 90
    save_env_value("HERMES_MAX_ITERATIONS", "90")

    config.setdefault("display", {})["tool_progress"] = "all"

    config.setdefault("compression", {})["enabled"] = True
    config["compression"]["threshold"] = 0.50

    config.setdefault("session_reset", {}).update({
        "mode": "both",
        "idle_minutes": 1440,
        "at_hour": 4,
    })

    save_config(config)
    print_success("Applied recommended defaults:")
    print_info("  Max iterations: 90")
    print_info("  Tool progress: all")
    print_info("  Compression threshold: 0.50")
    print_info("  Session reset: inactivity (1440 min) + daily (4:00)")
    print_info("  Run `hermes setup agent` later to customize.")


def setup_agent_settings(config: dict):
    """Configure agent behavior: iterations, progress display, compression, session reset."""

    print_header("Agent Settings")
    print_info(f"   Guide: {_DOCS_BASE}/user-guide/configuration")
    print()

    # ── Max Iterations ──
    current_max = get_env_value("HERMES_MAX_ITERATIONS") or str(
        config.get("agent", {}).get("max_turns", 90)
    )
    print_info("Maximum tool-calling iterations per conversation.")
    print_info("Higher = more complex tasks, but costs more tokens.")
    print_info("Default is 90, which works for most tasks. Use 150+ for open exploration.")

    max_iter_str = prompt("Max iterations", current_max)
    try:
        max_iter = int(max_iter_str)
        if max_iter > 0:
            save_env_value("HERMES_MAX_ITERATIONS", str(max_iter))
            config.setdefault("agent", {})["max_turns"] = max_iter
            config.pop("max_turns", None)
            print_success(f"Max iterations set to {max_iter}")
    except ValueError:
        print_warning("Invalid number, keeping current value")

    # ── Tool Progress Display ──
    print_info("")
    print_info("Tool Progress Display")
    print_info("Controls how much tool activity is shown (CLI and messaging).")
    print_info("  off     — Silent, just the final response")
    print_info("  new     — Show tool name only when it changes (less noise)")
    print_info("  all     — Show every tool call with a short preview")
    print_info("  verbose — Full args, results, and debug logs")

    current_mode = config.get("display", {}).get("tool_progress", "all")
    mode = prompt("Tool progress mode", current_mode)
    if mode.lower() in ("off", "new", "all", "verbose"):
        if "display" not in config:
            config["display"] = {}
        config["display"]["tool_progress"] = mode.lower()
        save_config(config)
        print_success(f"Tool progress set to: {mode.lower()}")
    else:
        print_warning(f"Unknown mode '{mode}', keeping '{current_mode}'")

    # ── Context Compression ──
    print_header("Context Compression")
    print_info("Automatically summarizes old messages when context gets too long.")
    print_info(
        "Higher threshold = compress later (use more context). Lower = compress sooner."
    )

    config.setdefault("compression", {})["enabled"] = True

    current_threshold = config.get("compression", {}).get("threshold", 0.50)
    threshold_str = prompt("Compression threshold (0.5-0.95)", str(current_threshold))
    try:
        threshold = float(threshold_str)
        if 0.5 <= threshold <= 0.95:
            config["compression"]["threshold"] = threshold
    except ValueError:
        pass

    print_success(
        f"Context compression threshold set to {config['compression'].get('threshold', 0.50)}"
    )

    # ── Session Reset Policy ──
    print_header("Session Reset Policy")
    print_info(
        "Messaging sessions (Telegram, Discord, etc.) accumulate context over time."
    )
    print_info(
        "Each message adds to the conversation history, which means growing API costs."
    )
    print_info("")
    print_info(
        "To manage this, sessions can automatically reset after a period of inactivity"
    )
    print_info(
        "or at a fixed time each day. When a reset happens, the agent saves important"
    )
    print_info(
        "things to its persistent memory first — but the conversation context is cleared."
    )
    print_info("")
    print_info("You can also manually reset anytime by typing /reset in chat.")
    print_info("")

    reset_choices = [
        "Inactivity + daily reset (recommended - reset whichever comes first)",
        "Inactivity only (reset after N minutes of no messages)",
        "Daily only (reset at a fixed hour each day)",
        "Never auto-reset (context lives until /reset or context compression)",
        "Keep current settings",
    ]

    current_policy = config.get("session_reset", {})
    current_mode = current_policy.get("mode", "both")
    current_idle = current_policy.get("idle_minutes", 1440)
    current_hour = current_policy.get("at_hour", 4)

    default_reset = {"both": 0, "idle": 1, "daily": 2, "none": 3}.get(current_mode, 0)

    reset_idx = prompt_choice("Session reset mode:", reset_choices, default_reset)

    config.setdefault("session_reset", {})

    if reset_idx == 0:  # Both
        config["session_reset"]["mode"] = "both"
        idle_str = prompt("  Inactivity timeout (minutes)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config["session_reset"]["idle_minutes"] = idle_val
        except ValueError:
            pass
        hour_str = prompt("  Daily reset hour (0-23, local time)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config["session_reset"]["at_hour"] = hour_val
        except ValueError:
            pass
        print_success(
            f"Sessions reset after {config['session_reset'].get('idle_minutes', 1440)} min idle or daily at {config['session_reset'].get('at_hour', 4)}:00"
        )
    elif reset_idx == 1:  # Idle only
        config["session_reset"]["mode"] = "idle"
        idle_str = prompt("  Inactivity timeout (minutes)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config["session_reset"]["idle_minutes"] = idle_val
        except ValueError:
            pass
        print_success(
            f"Sessions reset after {config['session_reset'].get('idle_minutes', 1440)} min of inactivity"
        )
    elif reset_idx == 2:  # Daily only
        config["session_reset"]["mode"] = "daily"
        hour_str = prompt("  Daily reset hour (0-23, local time)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config["session_reset"]["at_hour"] = hour_val
        except ValueError:
            pass
        print_success(
            f"Sessions reset daily at {config['session_reset'].get('at_hour', 4)}:00"
        )
    elif reset_idx == 3:  # None
        config["session_reset"]["mode"] = "none"
        print_info(
            "Sessions will never auto-reset. Context is managed only by compression."
        )
        print_warning(
            "Long conversations will grow in cost. Use /reset manually when needed."
        )
    # else: keep current (idx == 4)

    save_config(config)


# =============================================================================
# Section 4: Messaging Platforms (Gateway)
# =============================================================================


def _setup_telegram():
    """Configure Telegram bot credentials and allowlist."""
    print_header("Telegram")
    existing = get_env_value("TELEGRAM_BOT_TOKEN")
    if existing:
        print_info("Telegram: already configured")
        if not prompt_yes_no("Reconfigure Telegram?", False):
            # Check missing allowlist on existing config
            if not get_env_value("TELEGRAM_ALLOWED_USERS"):
                print_info("⚠️  Telegram has no user allowlist - anyone can use your bot!")
                if prompt_yes_no("Add allowed users now?", True):
                    print_info("   To find your Telegram user ID: message @userinfobot")
                    allowed_users = prompt("Allowed user IDs (comma-separated)")
                    if allowed_users:
                        save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users.replace(" ", ""))
                        print_success("Telegram allowlist configured")
            return

    print_info("Create a bot via @BotFather on Telegram")
    token = prompt("Telegram bot token", password=True)
    if not token:
        return
    save_env_value("TELEGRAM_BOT_TOKEN", token)
    print_success("Telegram token saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find your Telegram user ID:")
    print_info("   1. Message @userinfobot on Telegram")
    print_info("   2. It will reply with your numeric ID (e.g., 123456789)")
    print()
    allowed_users = prompt(
        "Allowed user IDs (comma-separated, leave empty for open access)"
    )
    if allowed_users:
        save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Telegram allowlist configured - only listed users can use the bot")
    else:
        print_info("⚠️  No allowlist set - anyone who finds your bot can use it!")

    print()
    print_info("📬 Home Channel: where Hermes delivers cron job results,")
    print_info("   cross-platform messages, and notifications.")
    print_info("   For Telegram DMs, this is your user ID (same as above).")

    first_user_id = allowed_users.split(",")[0].strip() if allowed_users else ""
    if first_user_id:
        if prompt_yes_no(f"Use your user ID ({first_user_id}) as the home channel?", True):
            save_env_value("TELEGRAM_HOME_CHANNEL", first_user_id)
            print_success(f"Telegram home channel set to {first_user_id}")
        else:
            home_channel = prompt("Home channel ID (or leave empty to set later with /set-home in Telegram)")
            if home_channel:
                save_env_value("TELEGRAM_HOME_CHANNEL", home_channel)
    else:
        print_info("   You can also set this later by typing /set-home in your Telegram chat.")
        home_channel = prompt("Home channel ID (leave empty to set later)")
        if home_channel:
            save_env_value("TELEGRAM_HOME_CHANNEL", home_channel)


def _setup_discord():
    """Configure Discord bot credentials and allowlist."""
    print_header("Discord")
    existing = get_env_value("DISCORD_BOT_TOKEN")
    if existing:
        print_info("Discord: already configured")
        if not prompt_yes_no("Reconfigure Discord?", False):
            if not get_env_value("DISCORD_ALLOWED_USERS"):
                print_info("⚠️  Discord has no user allowlist - anyone can use your bot!")
                if prompt_yes_no("Add allowed users now?", True):
                    print_info("   To find Discord ID: Enable Developer Mode, right-click name → Copy ID")
                    allowed_users = prompt("Allowed user IDs (comma-separated)")
                    if allowed_users:
                        cleaned_ids = _clean_discord_user_ids(allowed_users)
                        save_env_value("DISCORD_ALLOWED_USERS", ",".join(cleaned_ids))
                        print_success("Discord allowlist configured")
            return

    print_info("Create a bot at https://discord.com/developers/applications")
    token = prompt("Discord bot token", password=True)
    if not token:
        return
    save_env_value("DISCORD_BOT_TOKEN", token)
    print_success("Discord token saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find your Discord user ID:")
    print_info("   1. Enable Developer Mode in Discord settings")
    print_info("   2. Right-click your name → Copy ID")
    print()
    print_info("   You can also use Discord usernames (resolved on gateway start).")
    print()
    allowed_users = prompt(
        "Allowed user IDs or usernames (comma-separated, leave empty for open access)"
    )
    if allowed_users:
        cleaned_ids = _clean_discord_user_ids(allowed_users)
        save_env_value("DISCORD_ALLOWED_USERS", ",".join(cleaned_ids))
        print_success("Discord allowlist configured")
    else:
        print_info("⚠️  No allowlist set - anyone in servers with your bot can use it!")

    print()
    print_info("📬 Home Channel: where Hermes delivers cron job results,")
    print_info("   cross-platform messages, and notifications.")
    print_info("   To get a channel ID: right-click a channel → Copy Channel ID")
    print_info("   (requires Developer Mode in Discord settings)")
    print_info("   You can also set this later by typing /set-home in a Discord channel.")
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)")
    if home_channel:
        save_env_value("DISCORD_HOME_CHANNEL", home_channel)


def _clean_discord_user_ids(raw: str) -> list:
    """Strip common Discord mention prefixes from a comma-separated ID string."""
    cleaned = []
    for uid in raw.replace(" ", "").split(","):
        uid = uid.strip()
        if uid.startswith("<@") and uid.endswith(">"):
            uid = uid.lstrip("<@!").rstrip(">")
        if uid.lower().startswith("user:"):
            uid = uid[5:]
        if uid:
            cleaned.append(uid)
    return cleaned


def _setup_slack():
    """Configure Slack bot credentials."""
    print_header("Slack")
    existing = get_env_value("SLACK_BOT_TOKEN")
    if existing:
        print_info("Slack: already configured")
        if not prompt_yes_no("Reconfigure Slack?", False):
            return

    print_info("Steps to create a Slack app:")
    print_info("   1. Go to https://api.slack.com/apps → Create New App (from scratch)")
    print_info("   2. Enable Socket Mode: Settings → Socket Mode → Enable")
    print_info("      • Create an App-Level Token with 'connections:write' scope")
    print_info("   3. Add Bot Token Scopes: Features → OAuth & Permissions")
    print_info("      Required scopes: chat:write, app_mentions:read,")
    print_info("      channels:history, channels:read, im:history,")
    print_info("      im:read, im:write, users:read, files:write")
    print_info("      Optional for private channels: groups:history")
    print_info("   4. Subscribe to Events: Features → Event Subscriptions → Enable")
    print_info("      Required events: message.im, message.channels, app_mention")
    print_info("      Optional for private channels: message.groups")
    print_warning("   ⚠ Without message.channels the bot will ONLY work in DMs,")
    print_warning("     not public channels.")
    print_info("   5. Install to Workspace: Settings → Install App")
    print_info("   6. Reinstall the app after any scope or event changes")
    print_info("   7. After installing, invite the bot to channels: /invite @YourBot")
    print()
    print_info("   Full guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/slack/")
    print()
    bot_token = prompt("Slack Bot Token (xoxb-...)", password=True)
    if not bot_token:
        return
    save_env_value("SLACK_BOT_TOKEN", bot_token)
    app_token = prompt("Slack App Token (xapp-...)", password=True)
    if app_token:
        save_env_value("SLACK_APP_TOKEN", app_token)
    print_success("Slack tokens saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find a Member ID: click a user's name → View full profile → ⋮ → Copy member ID")
    print()
    allowed_users = prompt(
        "Allowed user IDs (comma-separated, leave empty to deny everyone except paired users)"
    )
    if allowed_users:
        save_env_value("SLACK_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Slack allowlist configured")
    else:
        print_warning("⚠️  No Slack allowlist set - unpaired users will be denied by default.")
        print_info("   Set SLACK_ALLOW_ALL_USERS=true or GATEWAY_ALLOW_ALL_USERS=true only if you intentionally want open workspace access.")


def _setup_matrix():
    """Configure Matrix credentials."""
    print_header("Matrix")
    existing = get_env_value("MATRIX_ACCESS_TOKEN") or get_env_value("MATRIX_PASSWORD")
    if existing:
        print_info("Matrix: already configured")
        if not prompt_yes_no("Reconfigure Matrix?", False):
            return

    print_info("Works with any Matrix homeserver (Synapse, Conduit, Dendrite, or matrix.org).")
    print_info("   1. Create a bot user on your homeserver, or use your own account")
    print_info("   2. Get an access token from Element, or provide user ID + password")
    print()
    homeserver = prompt("Homeserver URL (e.g. https://matrix.example.org)")
    if homeserver:
        save_env_value("MATRIX_HOMESERVER", homeserver.rstrip("/"))

    print()
    print_info("Auth: provide an access token (recommended), or user ID + password.")
    token = prompt("Access token (leave empty for password login)", password=True)
    if token:
        save_env_value("MATRIX_ACCESS_TOKEN", token)
        user_id = prompt("User ID (@bot:server — optional, will be auto-detected)")
        if user_id:
            save_env_value("MATRIX_USER_ID", user_id)
        print_success("Matrix access token saved")
    else:
        user_id = prompt("User ID (@bot:server)")
        if user_id:
            save_env_value("MATRIX_USER_ID", user_id)
        password = prompt("Password", password=True)
        if password:
            save_env_value("MATRIX_PASSWORD", password)
            print_success("Matrix credentials saved")

    if token or get_env_value("MATRIX_PASSWORD"):
        print()
        want_e2ee = prompt_yes_no("Enable end-to-end encryption (E2EE)?", False)
        if want_e2ee:
            save_env_value("MATRIX_ENCRYPTION", "true")
            print_success("E2EE enabled")

        matrix_pkg = "matrix-nio[e2e]" if want_e2ee else "matrix-nio"
        try:
            __import__("nio")
        except ImportError:
            print_info(f"Installing {matrix_pkg}...")
            import subprocess
            uv_bin = shutil.which("uv")
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "--python", sys.executable, matrix_pkg],
                    capture_output=True, text=True,
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", matrix_pkg],
                    capture_output=True, text=True,
                )
            if result.returncode == 0:
                print_success(f"{matrix_pkg} installed")
            else:
                print_warning(f"Install failed — run manually: pip install '{matrix_pkg}'")
                if result.stderr:
                    print_info(f"  Error: {result.stderr.strip().splitlines()[-1]}")

        print()
        print_info("🔒 Security: Restrict who can use your bot")
        print_info("   Matrix user IDs look like @username:server")
        print()
        allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
        if allowed_users:
            save_env_value("MATRIX_ALLOWED_USERS", allowed_users.replace(" ", ""))
            print_success("Matrix allowlist configured")
        else:
            print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")

        print()
        print_info("📬 Home Room: where Hermes delivers cron job results and notifications.")
        print_info("   Room IDs look like !abc123:server (shown in Element room settings)")
        print_info("   You can also set this later by typing /set-home in a Matrix room.")
        home_room = prompt("Home room ID (leave empty to set later with /set-home)")
        if home_room:
            save_env_value("MATRIX_HOME_ROOM", home_room)


def _setup_mattermost():
    """Configure Mattermost bot credentials."""
    print_header("Mattermost")
    existing = get_env_value("MATTERMOST_TOKEN")
    if existing:
        print_info("Mattermost: already configured")
        if not prompt_yes_no("Reconfigure Mattermost?", False):
            return

    print_info("Works with any self-hosted Mattermost instance.")
    print_info("   1. In Mattermost: Integrations → Bot Accounts → Add Bot Account")
    print_info("   2. Copy the bot token")
    print()
    mm_url = prompt("Mattermost server URL (e.g. https://mm.example.com)")
    if mm_url:
        save_env_value("MATTERMOST_URL", mm_url.rstrip("/"))
    token = prompt("Bot token", password=True)
    if not token:
        return
    save_env_value("MATTERMOST_TOKEN", token)
    print_success("Mattermost token saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find your user ID: click your avatar → Profile")
    print_info("   or use the API: GET /api/v4/users/me")
    print()
    allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
    if allowed_users:
        save_env_value("MATTERMOST_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Mattermost allowlist configured")
    else:
        print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")

    print()
    print_info("📬 Home Channel: where Hermes delivers cron job results and notifications.")
    print_info("   To get a channel ID: click channel name → View Info → copy the ID")
    print_info("   You can also set this later by typing /set-home in a Mattermost channel.")
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)")
    if home_channel:
        save_env_value("MATTERMOST_HOME_CHANNEL", home_channel)


def _setup_whatsapp():
    """Configure WhatsApp bridge."""
    print_header("WhatsApp")
    existing = get_env_value("WHATSAPP_ENABLED")
    if existing:
        print_info("WhatsApp: already enabled")
        return

    print_info("WhatsApp connects via a built-in bridge (Baileys).")
    print_info("Requires Node.js. Run 'hermes whatsapp' for guided setup.")
    print()
    if prompt_yes_no("Enable WhatsApp now?", True):
        save_env_value("WHATSAPP_ENABLED", "true")
        print_success("WhatsApp enabled")
        print_info("Run 'hermes whatsapp' to choose your mode (separate bot number")
        print_info("or personal self-chat) and pair via QR code.")


def _setup_bluebubbles():
    """Configure BlueBubbles iMessage gateway."""
    print_header("BlueBubbles (iMessage)")
    existing = get_env_value("BLUEBUBBLES_SERVER_URL")
    if existing:
        print_info("BlueBubbles: already configured")
        if not prompt_yes_no("Reconfigure BlueBubbles?", False):
            return

    print_info("Connects Hermes to iMessage via BlueBubbles — a free, open-source")
    print_info("macOS server that bridges iMessage to any device.")
    print_info("   Requires a Mac running BlueBubbles Server v1.0.0+")
    print_info("   Download: https://bluebubbles.app/")
    print()
    print_info("In BlueBubbles Server → Settings → API, note your Server URL and Password.")
    print()

    server_url = prompt("BlueBubbles server URL (e.g. http://192.168.1.10:1234)")
    if not server_url:
        print_warning("Server URL is required — skipping BlueBubbles setup")
        return
    save_env_value("BLUEBUBBLES_SERVER_URL", server_url.rstrip("/"))

    password = prompt("BlueBubbles server password", password=True)
    if not password:
        print_warning("Password is required — skipping BlueBubbles setup")
        return
    save_env_value("BLUEBUBBLES_PASSWORD", password)
    print_success("BlueBubbles credentials saved")

    print()
    print_info("🔒 Security: Restrict who can message your bot")
    print_info("   Use iMessage addresses: email (user@icloud.com) or phone (+15551234567)")
    print()
    allowed_users = prompt("Allowed iMessage addresses (comma-separated, leave empty for open access)")
    if allowed_users:
        save_env_value("BLUEBUBBLES_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("BlueBubbles allowlist configured")
    else:
        print_info("⚠️  No allowlist set — anyone who can iMessage you can use the bot!")

    print()
    print_info("📬 Home Channel: phone or email for cron job delivery and notifications.")
    print_info("   You can also set this later with /set-home in your iMessage chat.")
    home_channel = prompt("Home channel address (leave empty to set later)")
    if home_channel:
        save_env_value("BLUEBUBBLES_HOME_CHANNEL", home_channel)

    print()
    print_info("Advanced settings (defaults are fine for most setups):")
    if prompt_yes_no("Configure webhook listener settings?", False):
        webhook_port = prompt("Webhook listener port (default: 8645)")
        if webhook_port:
            try:
                save_env_value("BLUEBUBBLES_WEBHOOK_PORT", str(int(webhook_port)))
                print_success(f"Webhook port set to {webhook_port}")
            except ValueError:
                print_warning("Invalid port number, using default 8645")

    print()
    print_info("Requires the BlueBubbles Private API helper for typing indicators,")
    print_info("read receipts, and tapback reactions. Basic messaging works without it.")
    print_info("   Install: https://docs.bluebubbles.app/helper-bundle/installation")


def _setup_webhooks():
    """Configure webhook integration."""
    print_header("Webhooks")
    existing = get_env_value("WEBHOOK_ENABLED")
    if existing:
        print_info("Webhooks: already configured")
        if not prompt_yes_no("Reconfigure webhooks?", False):
            return

    print()
    print_warning("⚠  Webhook and SMS platforms require exposing gateway ports to the")
    print_warning("   internet. For security, run the gateway in a sandboxed environment")
    print_warning("   (Docker, VM, etc.) to limit blast radius from prompt injection.")
    print()
    print_info("   Full guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/")
    print()

    port = prompt("Webhook port (default 8644)")
    if port:
        try:
            save_env_value("WEBHOOK_PORT", str(int(port)))
            print_success(f"Webhook port set to {port}")
        except ValueError:
            print_warning("Invalid port number, using default 8644")

    secret = prompt("Global HMAC secret (shared across all routes)", password=True)
    if secret:
        save_env_value("WEBHOOK_SECRET", secret)
        print_success("Webhook secret saved")
    else:
        print_warning("No secret set — you must configure per-route secrets in config.yaml")

    save_env_value("WEBHOOK_ENABLED", "true")
    print()
    print_success("Webhooks enabled! Next steps:")
    from hermes_constants import display_hermes_home as _dhh
    print_info(f"   1. Define webhook routes in {_dhh()}/config.yaml")
    print_info("   2. Point your service (GitHub, GitLab, etc.) at:")
    print_info("      http://your-server:8644/webhooks/<route-name>")
    print()
    print_info("   Route configuration guide:")
    print_info("   https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/#configuring-routes")
    print()
    print_info("   Open config in your editor:  hermes config edit")


# Platform registry for the gateway checklist
_GATEWAY_PLATFORMS = [
    ("Telegram", "TELEGRAM_BOT_TOKEN", _setup_telegram),
    ("Discord", "DISCORD_BOT_TOKEN", _setup_discord),
    ("Slack", "SLACK_BOT_TOKEN", _setup_slack),
    ("Matrix", "MATRIX_ACCESS_TOKEN", _setup_matrix),
    ("Mattermost", "MATTERMOST_TOKEN", _setup_mattermost),
    ("WhatsApp", "WHATSAPP_ENABLED", _setup_whatsapp),
    ("BlueBubbles (iMessage)", "BLUEBUBBLES_SERVER_URL", _setup_bluebubbles),
    ("Webhooks (GitHub, GitLab, etc.)", "WEBHOOK_ENABLED", _setup_webhooks),
]


def setup_gateway(config: dict):
    """Configure messaging platform integrations."""
    print_header("Messaging Platforms")
    print_info("Connect to messaging platforms to chat with Hermes from anywhere.")
    print_info("Toggle with Space, confirm with Enter.")
    print()

    # Build checklist items, pre-selecting already-configured platforms
    items = []
    pre_selected = []
    for i, (name, env_var, _func) in enumerate(_GATEWAY_PLATFORMS):
        # Matrix has two possible env vars
        is_configured = bool(get_env_value(env_var))
        if name == "Matrix" and not is_configured:
            is_configured = bool(get_env_value("MATRIX_PASSWORD"))
        label = f"{name}  (configured)" if is_configured else name
        items.append(label)
        if is_configured:
            pre_selected.append(i)

    selected = prompt_checklist("Select platforms to configure:", items, pre_selected)

    if not selected:
        print_info("No platforms selected. Run 'hermes setup gateway' later to configure.")
        return

    for idx in selected:
        name, _env_var, setup_func = _GATEWAY_PLATFORMS[idx]
        setup_func()

    # ── Gateway Service Setup ──
    any_messaging = (
        get_env_value("TELEGRAM_BOT_TOKEN")
        or get_env_value("DISCORD_BOT_TOKEN")
        or get_env_value("SLACK_BOT_TOKEN")
        or get_env_value("MATTERMOST_TOKEN")
        or get_env_value("MATRIX_ACCESS_TOKEN")
        or get_env_value("MATRIX_PASSWORD")
        or get_env_value("WHATSAPP_ENABLED")
        or get_env_value("BLUEBUBBLES_SERVER_URL")
        or get_env_value("WEBHOOK_ENABLED")
    )
    if any_messaging:
        print()
        print_info("━" * 50)
        print_success("Messaging platforms configured!")

        # Check if any home channels are missing
        missing_home = []
        if get_env_value("TELEGRAM_BOT_TOKEN") and not get_env_value(
            "TELEGRAM_HOME_CHANNEL"
        ):
            missing_home.append("Telegram")
        if get_env_value("DISCORD_BOT_TOKEN") and not get_env_value(
            "DISCORD_HOME_CHANNEL"
        ):
            missing_home.append("Discord")
        if get_env_value("SLACK_BOT_TOKEN") and not get_env_value("SLACK_HOME_CHANNEL"):
            missing_home.append("Slack")
        if get_env_value("BLUEBUBBLES_SERVER_URL") and not get_env_value("BLUEBUBBLES_HOME_CHANNEL"):
            missing_home.append("BlueBubbles")

        if missing_home:
            print()
            print_warning(f"No home channel set for: {', '.join(missing_home)}")
            print_info("   Without a home channel, cron jobs and cross-platform")
            print_info("   messages can't be delivered to those platforms.")
            print_info("   Set one later with /set-home in your chat, or:")
            for plat in missing_home:
                print_info(
                    f"     hermes config set {plat.upper()}_HOME_CHANNEL <channel_id>"
                )

        # Offer to install the gateway as a system service
        import platform as _platform

        _is_linux = _platform.system() == "Linux"
        _is_macos = _platform.system() == "Darwin"

        from hermes_cli.gateway import (
            _is_service_installed,
            _is_service_running,
            has_conflicting_systemd_units,
            install_linux_gateway_from_setup,
            print_systemd_scope_conflict_warning,
            systemd_start,
            systemd_restart,
            launchd_install,
            launchd_start,
            launchd_restart,
        )

        service_installed = _is_service_installed()
        service_running = _is_service_running()

        print()
        if _is_linux and has_conflicting_systemd_units():
            print_systemd_scope_conflict_warning()
            print()

        if service_running:
            if prompt_yes_no("  Restart the gateway to pick up changes?", True):
                try:
                    if _is_linux:
                        systemd_restart()
                    elif _is_macos:
                        launchd_restart()
                except Exception as e:
                    print_error(f"  Restart failed: {e}")
        elif service_installed:
            if prompt_yes_no("  Start the gateway service?", True):
                try:
                    if _is_linux:
                        systemd_start()
                    elif _is_macos:
                        launchd_start()
                except Exception as e:
                    print_error(f"  Start failed: {e}")
        elif _is_linux or _is_macos:
            svc_name = "systemd" if _is_linux else "launchd"
            if prompt_yes_no(
                f"  Install the gateway as a {svc_name} service? (runs in background, starts on boot)",
                True,
            ):
                try:
                    installed_scope = None
                    did_install = False
                    if _is_linux:
                        installed_scope, did_install = install_linux_gateway_from_setup(force=False)
                    else:
                        launchd_install(force=False)
                        did_install = True
                    print()
                    if did_install and prompt_yes_no("  Start the service now?", True):
                        try:
                            if _is_linux:
                                systemd_start(system=installed_scope == "system")
                            elif _is_macos:
                                launchd_start()
                        except Exception as e:
                            print_error(f"  Start failed: {e}")
                except Exception as e:
                    print_error(f"  Install failed: {e}")
                    print_info("  You can try manually: hermes gateway install")
            else:
                print_info("  You can install later: hermes gateway install")
                if _is_linux:
                    print_info("  Or as a boot-time service: sudo hermes gateway install --system")
                print_info("  Or run in foreground:  hermes gateway")
        else:
            print_info("Start the gateway to bring your bots online:")
            print_info("   hermes gateway              # Run in foreground")

        print_info("━" * 50)


# =============================================================================
# Section 5: Tool Configuration (delegates to unified tools_config.py)
# =============================================================================


def setup_tools(config: dict, first_install: bool = False):
    """Configure tools — delegates to the unified tools_command() in tools_config.py.

    Both `hermes setup tools` and `hermes tools` use the same flow:
    platform selection → toolset toggles → provider/API key configuration.

    Args:
        first_install: When True, uses the simplified first-install flow
            (no platform menu, prompts for all unconfigured API keys).
    """
    from hermes_cli.tools_config import tools_command

    tools_command(first_install=first_install, config=config)


# =============================================================================
# Post-Migration Section Skip Logic
# =============================================================================


def _get_section_config_summary(config: dict, section_key: str) -> Optional[str]:
    """Return a short summary if a setup section is already configured, else None.

    Used after OpenClaw migration to detect which sections can be skipped.
    ``get_env_value`` is the module-level import from hermes_cli.config
    so that test patches on ``setup_mod.get_env_value`` take effect.
    """
    if section_key == "model":
        has_key = bool(
            get_env_value("OPENROUTER_API_KEY")
            or get_env_value("OPENAI_API_KEY")
            or get_env_value("ANTHROPIC_API_KEY")
        )
        if not has_key:
            # Check for OAuth providers
            try:
                from hermes_cli.auth import get_active_provider
                if get_active_provider():
                    has_key = True
            except Exception:
                pass
        if not has_key:
            return None
        model = config.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        if isinstance(model, dict):
            return str(model.get("default") or model.get("model") or "configured")
        return "configured"

    elif section_key == "terminal":
        backend = config.get("terminal", {}).get("backend", "local")
        return f"backend: {backend}"

    elif section_key == "agent":
        max_turns = config.get("agent", {}).get("max_turns", 90)
        return f"max turns: {max_turns}"

    elif section_key == "gateway":
        platforms = []
        if get_env_value("TELEGRAM_BOT_TOKEN"):
            platforms.append("Telegram")
        if get_env_value("DISCORD_BOT_TOKEN"):
            platforms.append("Discord")
        if get_env_value("SLACK_BOT_TOKEN"):
            platforms.append("Slack")
        if get_env_value("WHATSAPP_PHONE_NUMBER_ID"):
            platforms.append("WhatsApp")
        if get_env_value("SIGNAL_ACCOUNT"):
            platforms.append("Signal")
        if get_env_value("BLUEBUBBLES_SERVER_URL"):
            platforms.append("BlueBubbles")
        if platforms:
            return ", ".join(platforms)
        return None  # No platforms configured — section must run

    elif section_key == "tools":
        tools = []
        if get_env_value("ELEVENLABS_API_KEY"):
            tools.append("TTS/ElevenLabs")
        if get_env_value("BROWSERBASE_API_KEY"):
            tools.append("Browser")
        if get_env_value("FIRECRAWL_API_KEY"):
            tools.append("Firecrawl")
        if tools:
            return ", ".join(tools)
        return None

    return None


def _skip_configured_section(
    config: dict, section_key: str, label: str
) -> bool:
    """Show an already-configured section summary and offer to skip.

    Returns True if the user chose to skip, False if the section should run.
    """
    summary = _get_section_config_summary(config, section_key)
    if not summary:
        return False
    print()
    print_success(f"  {label}: {summary}")
    return not prompt_yes_no(f"  Reconfigure {label.lower()}?", default=False)


# =============================================================================
# OpenClaw Migration
# =============================================================================


_OPENCLAW_SCRIPT = (
    get_optional_skills_dir(PROJECT_ROOT / "optional-skills")
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)


def _load_openclaw_migration_module():
    """Load the openclaw_to_hermes migration script as a module.

    Returns the loaded module, or None if the script can't be loaded.
    """
    if not _OPENCLAW_SCRIPT.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "openclaw_to_hermes", _OPENCLAW_SCRIPT
    )
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve the module
    # (Python 3.11+ requires this for dynamically loaded modules)
    import sys as _sys
    _sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        _sys.modules.pop(spec.name, None)
        raise
    return mod


# Item kinds that represent high-impact changes warranting explicit warnings.
# Gateway tokens/channels can hijack messaging platforms from the old agent.
# Config values may have different semantics between OpenClaw and Hermes.
# Instruction/context files (.md) can contain incompatible setup procedures.
_HIGH_IMPACT_KIND_KEYWORDS = {
    "gateway": "⚠ Gateway/messaging — this will configure Hermes to use your OpenClaw messaging channels",
    "telegram": "⚠ Telegram — this will point Hermes at your OpenClaw Telegram bot",
    "slack": "⚠ Slack — this will point Hermes at your OpenClaw Slack workspace",
    "discord": "⚠ Discord — this will point Hermes at your OpenClaw Discord bot",
    "whatsapp": "⚠ WhatsApp — this will point Hermes at your OpenClaw WhatsApp connection",
    "config": "⚠ Config values — OpenClaw settings may not map 1:1 to Hermes equivalents",
    "soul": "⚠ Instruction file — may contain OpenClaw-specific setup/restart procedures",
    "memory": "⚠ Memory/context file — may reference OpenClaw-specific infrastructure",
    "context": "⚠ Context file — may contain OpenClaw-specific instructions",
}


def _print_migration_preview(report: dict):
    """Print a detailed dry-run preview of what migration would do.

    Groups items by category and adds explicit warnings for high-impact
    changes like gateway token takeover and config value differences.
    """
    items = report.get("items", [])
    if not items:
        print_info("Nothing to migrate.")
        return

    migrated_items = [i for i in items if i.get("status") == "migrated"]
    conflict_items = [i for i in items if i.get("status") == "conflict"]
    skipped_items = [i for i in items if i.get("status") == "skipped"]

    warnings_shown = set()

    if migrated_items:
        print(color("  Would import:", Colors.GREEN))
        for item in migrated_items:
            kind = item.get("kind", "unknown")
            dest = item.get("destination", "")
            if dest:
                dest_short = str(dest).replace(str(Path.home()), "~")
                print(f"      {kind:<22s} → {dest_short}")
            else:
                print(f"      {kind}")

            # Check for high-impact items and collect warnings
            kind_lower = kind.lower()
            dest_lower = str(dest).lower()
            for keyword, warning in _HIGH_IMPACT_KIND_KEYWORDS.items():
                if keyword in kind_lower or keyword in dest_lower:
                    warnings_shown.add(warning)
        print()

    if conflict_items:
        print(color("  Would overwrite (conflicts with existing Hermes config):", Colors.YELLOW))
        for item in conflict_items:
            kind = item.get("kind", "unknown")
            reason = item.get("reason", "already exists")
            print(f"      {kind:<22s}  {reason}")
        print()

    if skipped_items:
        print(color("  Would skip:", Colors.DIM))
        for item in skipped_items:
            kind = item.get("kind", "unknown")
            reason = item.get("reason", "")
            print(f"      {kind:<22s}  {reason}")
        print()

    # Print collected warnings
    if warnings_shown:
        print(color("  ── Warnings ──", Colors.YELLOW))
        for warning in sorted(warnings_shown):
            print(color(f"    {warning}", Colors.YELLOW))
        print()
        print(color("  Note: OpenClaw config values may have different semantics in Hermes.", Colors.YELLOW))
        print(color("  For example, OpenClaw's tool_call_execution: \"auto\" ≠ Hermes's yolo mode.", Colors.YELLOW))
        print(color("  Instruction files (.md) from OpenClaw may contain incompatible procedures.", Colors.YELLOW))
        print()


def _offer_openclaw_migration(hermes_home: Path) -> bool:
    """Detect ~/.openclaw and offer to migrate during first-time setup.

    Runs a dry-run first to show the user exactly what would be imported,
    overwritten, or taken over. Only executes after explicit confirmation.

    Returns True if migration ran successfully, False otherwise.
    """
    openclaw_dir = Path.home() / ".openclaw"
    if not openclaw_dir.is_dir():
        return False

    if not _OPENCLAW_SCRIPT.exists():
        return False

    print()
    print_header("OpenClaw Installation Detected")
    print_info(f"Found OpenClaw data at {openclaw_dir}")
    print_info("Hermes can preview what would be imported before making any changes.")
    print()

    if not prompt_yes_no("Would you like to see what can be imported?", default=True):
        print_info(
            "Skipping migration. You can run it later with: hermes claw migrate --dry-run"
        )
        return False

    # Ensure config.yaml exists before migration tries to read it
    config_path = get_config_path()
    if not config_path.exists():
        save_config(load_config())

    # Load the migration module
    try:
        mod = _load_openclaw_migration_module()
        if mod is None:
            print_warning("Could not load migration script.")
            return False
    except Exception as e:
        print_warning(f"Could not load migration script: {e}")
        logger.debug("OpenClaw migration module load error", exc_info=True)
        return False

    # ── Phase 1: Dry-run preview ──
    try:
        selected = mod.resolve_selected_options(None, None, preset="full")
        dry_migrator = mod.Migrator(
            source_root=openclaw_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=False,  # dry-run — no files modified
            workspace_target=None,
            overwrite=True,  # show everything including conflicts
            migrate_secrets=True,
            output_dir=None,
            selected_options=selected,
            preset_name="full",
        )
        preview_report = dry_migrator.migrate()
    except Exception as e:
        print_warning(f"Migration preview failed: {e}")
        logger.debug("OpenClaw migration preview error", exc_info=True)
        return False

    # Display the full preview
    preview_summary = preview_report.get("summary", {})
    preview_count = preview_summary.get("migrated", 0)

    if preview_count == 0:
        print()
        print_info("Nothing to import from OpenClaw.")
        return False

    print()
    print_header(f"Migration Preview — {preview_count} item(s) would be imported")
    print_info("No changes have been made yet. Review the list below:")
    print()
    _print_migration_preview(preview_report)

    # ── Phase 2: Confirm and execute ──
    if not prompt_yes_no("Proceed with migration?", default=False):
        print_info(
            "Migration cancelled. You can run it later with: hermes claw migrate"
        )
        print_info(
            "Use --dry-run to preview again, or --preset minimal for a lighter import."
        )
        return False

    # Execute the migration — overwrite=False so existing Hermes configs are
    # preserved. The user saw the preview; conflicts are skipped by default.
    try:
        migrator = mod.Migrator(
            source_root=openclaw_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=True,
            workspace_target=None,
            overwrite=False,  # preserve existing Hermes config
            migrate_secrets=True,
            output_dir=None,
            selected_options=selected,
            preset_name="full",
        )
        report = migrator.migrate()
    except Exception as e:
        print_warning(f"Migration failed: {e}")
        logger.debug("OpenClaw migration error", exc_info=True)
        return False

    # Print final summary
    summary = report.get("summary", {})
    migrated = summary.get("migrated", 0)
    skipped = summary.get("skipped", 0)
    conflicts = summary.get("conflict", 0)
    errors = summary.get("error", 0)

    print()
    if migrated:
        print_success(f"Imported {migrated} item(s) from OpenClaw.")
    if conflicts:
        print_info(f"Skipped {conflicts} item(s) that already exist in Hermes (use hermes claw migrate --overwrite to force).")
    if skipped:
        print_info(f"Skipped {skipped} item(s) (not found or unchanged).")
    if errors:
        print_warning(f"{errors} item(s) had errors — check the migration report.")

    output_dir = report.get("output_dir")
    if output_dir:
        print_info(f"Full report saved to: {output_dir}")

    print_success("Migration complete! Continuing with setup...")
    return True


# =============================================================================
# Main Wizard Orchestrator
# =============================================================================

SETUP_SECTIONS = [
    ("model", "Model & Provider", setup_model_provider),
    ("tts", "Text-to-Speech", setup_tts),
    ("terminal", "Terminal Backend", setup_terminal_backend),
    ("gateway", "Messaging Platforms (Gateway)", setup_gateway),
    ("tools", "Tools", setup_tools),
    ("agent", "Agent Settings", setup_agent_settings),
]

# The returning-user menu intentionally omits standalone TTS because model setup
# already includes TTS selection and tools setup covers the rest of the provider
# configuration. Keep this list in the same order as the visible menu entries.
RETURNING_USER_MENU_SECTION_KEYS = [
    "model",
    "terminal",
    "gateway",
    "tools",
    "agent",
]


def run_setup_wizard(args):
    """Run the interactive setup wizard.

    Supports full, quick, and section-specific setup:
      hermes setup           — full or quick (auto-detected)
      hermes setup model     — just model/provider
      hermes setup terminal  — just terminal backend
      hermes setup gateway   — just messaging platforms
      hermes setup tools     — just tool configuration
      hermes setup agent     — just agent settings
    """
    from hermes_cli.config import is_managed, managed_error
    if is_managed():
        managed_error("run setup wizard")
        return
    ensure_hermes_home()

    config = load_config()
    hermes_home = get_hermes_home()

    # Detect non-interactive environments (headless SSH, Docker, CI/CD)
    non_interactive = getattr(args, 'non_interactive', False)
    if not non_interactive and not is_interactive_stdin():
        non_interactive = True

    if non_interactive:
        print_noninteractive_setup_guidance(
            "Running in a non-interactive environment (no TTY detected)."
        )
        return

    # Check if a specific section was requested
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
                func(config)
                save_config(config)
                print()
                print_success(f"{label} configuration complete!")
                return

        print_error(f"Unknown setup section: {section}")
        print_info(f"Available sections: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")
        return

    # Check if this is an existing installation with a provider configured
    from hermes_cli.auth import get_active_provider

    active_provider = get_active_provider()
    is_existing = (
        bool(get_env_value("OPENROUTER_API_KEY"))
        or bool(get_env_value("OPENAI_BASE_URL"))
        or active_provider is not None
    )

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│             ⚕ Hermes Agent Setup Wizard                │", Colors.MAGENTA
        )
    )
    print(
        color(
            "├─────────────────────────────────────────────────────────┤",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│  Let's configure your Hermes Agent installation.       │", Colors.MAGENTA
        )
    )
    print(
        color(
            "│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘",
            Colors.MAGENTA,
        )
    )

    migration_ran = False

    if is_existing:
        # ── Returning User Menu ──
        print()
        print_header("Welcome Back!")
        print_success("You already have Hermes configured.")
        print()

        menu_choices = [
            "Quick Setup - configure missing items only",
            "Full Setup - reconfigure everything",
            "---",
            "Model & Provider",
            "Terminal Backend",
            "Messaging Platforms (Gateway)",
            "Tools",
            "Agent Settings",
            "---",
            "Exit",
        ]

        # Separator indices (not selectable, but prompt_choice doesn't filter them,
        # so we handle them below)
        choice = prompt_choice("What would you like to do?", menu_choices, 0)

        if choice == 0:
            # Quick setup
            _run_quick_setup(config, hermes_home)
            return
        elif choice == 1:
            # Full setup — fall through to run all sections
            pass
        elif choice in (2, 8):
            # Separator — treat as exit
            print_info("Exiting. Run 'hermes setup' again when ready.")
            return
        elif choice == 9:
            print_info("Exiting. Run 'hermes setup' again when ready.")
            return
        elif 3 <= choice <= 7:
            # Individual section — map by key, not by position.
            # SETUP_SECTIONS includes TTS but the returning-user menu skips it,
            # so positional indexing (choice - 3) would dispatch the wrong section.
            section_key = RETURNING_USER_MENU_SECTION_KEYS[choice - 3]
            section = next((s for s in SETUP_SECTIONS if s[0] == section_key), None)
            if section:
                _, label, func = section
                func(config)
                save_config(config)
                _print_setup_summary(config, hermes_home)
            return
    else:
        # ── First-Time Setup ──
        print()

        # Offer OpenClaw migration before configuration begins
        migration_ran = _offer_openclaw_migration(hermes_home)
        if migration_ran:
            config = load_config()

        setup_mode = prompt_choice("How would you like to set up Hermes?", [
            "Quick setup — provider, model & messaging (recommended)",
            "Full setup — configure everything",
        ], 0)

        if setup_mode == 0:
            _run_first_time_quick_setup(config, hermes_home, is_existing)
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

    # Section 1: Model & Provider
    if not (migration_ran and _skip_configured_section(config, "model", "Model & Provider")):
        setup_model_provider(config)

    # Section 2: Terminal Backend
    if not (migration_ran and _skip_configured_section(config, "terminal", "Terminal Backend")):
        setup_terminal_backend(config)

    # Section 3: Agent Settings
    if not (migration_ran and _skip_configured_section(config, "agent", "Agent Settings")):
        setup_agent_settings(config)

    # Section 4: Messaging Platforms
    if not (migration_ran and _skip_configured_section(config, "gateway", "Messaging Platforms")):
        setup_gateway(config)

    # Section 5: Tools
    if not (migration_ran and _skip_configured_section(config, "tools", "Tools")):
        setup_tools(config, first_install=not is_existing)

    # Save and show summary
    save_config(config)
    _print_setup_summary(config, hermes_home)

    _offer_launch_chat()


def _offer_launch_chat():
    """Prompt the user to jump straight into chat after setup."""
    print()
    if prompt_yes_no("Launch hermes chat now?", True):
        from hermes_cli.main import cmd_chat
        from types import SimpleNamespace
        cmd_chat(SimpleNamespace(
            query=None, resume=None, continue_last=None, model=None,
            provider=None, effort=None, skin=None, oneshot=False,
            quiet=False, verbose=False, toolsets=None, skills=None,
            yolo=False, source=None, worktree=False, checkpoints=False,
            pass_session_id=False, max_turns=None,
        ))


def _run_first_time_quick_setup(config: dict, hermes_home, is_existing: bool):
    """Streamlined first-time setup: provider + model only.

    Applies sensible defaults for TTS (Edge), terminal (local), agent
    settings, and tools — the user can customize later via
    ``hermes setup <section>``.
    """
    # Step 1: Model & Provider (essential — skips rotation/vision/TTS)
    setup_model_provider(config, quick=True)

    # Step 2: Apply defaults for everything else
    _apply_default_agent_settings(config)
    config.setdefault("terminal", {}).setdefault("backend", "local")

    save_config(config)

    # Step 3: Offer messaging gateway setup
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

    print()
    print_success("Setup complete! You're ready to go.")
    print()
    print_info("  Configure all settings:    hermes setup")
    if gateway_choice != 0:
        print_info("  Connect Telegram/Discord:  hermes setup gateway")
    print()

    _print_setup_summary(config, hermes_home)

    _offer_launch_chat()


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
