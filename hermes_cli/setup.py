"""
Interactive setup wizard for Hermes Agent.

Guides users through:
1. Installation directory confirmation
2. API key configuration
3. Model selection  
4. Terminal backend selection
5. Messaging platform setup
6. Optional features

Config files are stored in ~/.hermes/ for easy access.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Import config helpers
from hermes_cli.config import (
    get_hermes_home, get_config_path, get_env_path,
    load_config, save_config, save_env_value, get_env_value,
    ensure_hermes_home, DEFAULT_CONFIG
)

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

def prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Prompt for a choice from a list with arrow key navigation."""
    print(color(question, Colors.YELLOW))
    
    # Try to use interactive menu if available
    try:
        from simple_term_menu import TerminalMenu
        
        # Add visual indicators
        menu_choices = [f"  {choice}" for choice in choices]
        
        terminal_menu = TerminalMenu(
            menu_choices,
            cursor_index=default,
            menu_cursor="→ ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
        )
        
        idx = terminal_menu.show()
        if idx is None:  # User pressed Escape or Ctrl+C
            print()
            sys.exit(1)
        print()  # Add newline after selection
        return idx
        
    except ImportError:
        # Fallback to number-based selection
        for i, choice in enumerate(choices):
            marker = "●" if i == default else "○"
            if i == default:
                print(color(f"  {marker} {choice}", Colors.GREEN))
            else:
                print(f"  {marker} {choice}")
        
        while True:
            try:
                value = input(color(f"  Select [1-{len(choices)}] ({default + 1}): ", Colors.DIM))
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
    """Prompt for yes/no."""
    default_str = "Y/n" if default else "y/N"
    
    while True:
        value = input(color(f"{question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
        
        if not value:
            return default
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
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
    
    print(color(title, Colors.YELLOW))
    print_info("Select the tools you want, then choose Continue.")
    print()
    
    try:
        from simple_term_menu import TerminalMenu
        
        menu_items = [f"  {item}" for item in items] + ["  Continue →"]
        
        # Build preselected indices string (e.g. "0,2,4")
        preselected = [str(i) for i in pre_selected]
        
        terminal_menu = TerminalMenu(
            menu_items,
            multi_select=True,
            show_multi_select_hint=True,
            multi_select_cursor="[✓] ",
            multi_select_select_on_accept=False,
            multi_select_empty_ok=True,
            preselected_entries=preselected if preselected else None,
            menu_cursor="→ ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
        )
        
        terminal_menu.show()
        
        if terminal_menu.chosen_menu_entries is None:
            # User pressed Escape
            return []
        
        # Filter out the "Continue →" entry and return original indices
        continue_idx = len(items)
        selected = [i for i in terminal_menu.chosen_menu_indices if i != continue_idx]
        return selected
        
    except ImportError:
        # Fallback: numbered toggle interface
        selected = set(pre_selected)
        
        while True:
            for i, item in enumerate(items):
                marker = color("[✓]", Colors.GREEN) if i in selected else "[ ]"
                print(f"  {marker} {i + 1}. {item}")
            print(f"      {len(items) + 1}. {color('Continue →', Colors.GREEN)}")
            print()
            
            try:
                value = input(color("  Toggle item # (or Enter to continue): ", Colors.DIM)).strip()
                if not value:
                    break
                idx = int(value) - 1
                if idx == len(items):
                    break
                if 0 <= idx < len(items):
                    if idx in selected:
                        selected.discard(idx)
                    else:
                        selected.add(idx)
                else:
                    print_error(f"Enter a number between 1 and {len(items) + 1}")
            except ValueError:
                print_error("Enter a number")
            except (KeyboardInterrupt, EOFError):
                print()
                return []
            
            # Clear and redraw (simple approach)
            print()
        
        return sorted(selected)


def _print_setup_summary(config: dict, hermes_home):
    """Print the setup completion summary."""
    # Tool availability summary
    print()
    print_header("Tool Availability Summary")
    
    tool_status = []
    
    # OpenRouter (required for vision, moa)
    if get_env_value('OPENROUTER_API_KEY'):
        tool_status.append(("Vision (image analysis)", True, None))
        tool_status.append(("Mixture of Agents", True, None))
    else:
        tool_status.append(("Vision (image analysis)", False, "OPENROUTER_API_KEY"))
        tool_status.append(("Mixture of Agents", False, "OPENROUTER_API_KEY"))
    
    # Firecrawl (web tools)
    if get_env_value('FIRECRAWL_API_KEY'):
        tool_status.append(("Web Search & Extract", True, None))
    else:
        tool_status.append(("Web Search & Extract", False, "FIRECRAWL_API_KEY"))
    
    # Browserbase (browser tools)
    if get_env_value('BROWSERBASE_API_KEY'):
        tool_status.append(("Browser Automation", True, None))
    else:
        tool_status.append(("Browser Automation", False, "BROWSERBASE_API_KEY"))
    
    # FAL (image generation)
    if get_env_value('FAL_KEY'):
        tool_status.append(("Image Generation", True, None))
    else:
        tool_status.append(("Image Generation", False, "FAL_KEY"))
    
    # TTS (always available via Edge TTS; ElevenLabs/OpenAI are optional)
    tool_status.append(("Text-to-Speech (Edge TTS)", True, None))
    if get_env_value('ELEVENLABS_API_KEY'):
        tool_status.append(("Text-to-Speech (ElevenLabs)", True, None))
    
    # Tinker + WandB (RL training)
    if get_env_value('TINKER_API_KEY') and get_env_value('WANDB_API_KEY'):
        tool_status.append(("RL Training (Tinker)", True, None))
    elif get_env_value('TINKER_API_KEY'):
        tool_status.append(("RL Training (Tinker)", False, "WANDB_API_KEY"))
    else:
        tool_status.append(("RL Training (Tinker)", False, "TINKER_API_KEY"))
    
    # Skills Hub
    if get_env_value('GITHUB_TOKEN'):
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
            print(f"   {color('✗', Colors.RED)} {name} {color(f'(missing {missing_var})', Colors.DIM)}")
    
    print()
    
    disabled_tools = [(name, var) for name, avail, var in tool_status if not avail]
    if disabled_tools:
        print_warning("Some tools are disabled. Run 'hermes setup' again to configure them,")
        print_warning("or edit ~/.hermes/.env directly to add the missing API keys.")
        print()
    
    # Done banner
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.GREEN))
    print(color("│              ✓ Setup Complete!                          │", Colors.GREEN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.GREEN))
    print()
    
    # Show file locations prominently
    print(color("📁 All your files are in ~/.hermes/:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('Settings:', Colors.YELLOW)}  {get_config_path()}")
    print(f"   {color('API Keys:', Colors.YELLOW)}  {get_env_path()}")
    print(f"   {color('Data:', Colors.YELLOW)}      {hermes_home}/cron/, sessions/, logs/")
    print()
    
    print(color("─" * 60, Colors.DIM))
    print()
    print(color("📝 To edit your configuration:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('hermes config', Colors.GREEN)}        View current settings")
    print(f"   {color('hermes config edit', Colors.GREEN)}   Open config in your editor")
    print(f"   {color('hermes config set KEY VALUE', Colors.GREEN)}")
    print(f"                         Set a specific value")
    print()
    print(f"   Or edit the files directly:")
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


def run_setup_wizard(args):
    """Run the interactive setup wizard."""
    ensure_hermes_home()
    
    config = load_config()
    hermes_home = get_hermes_home()
    
    # Check if this is an existing installation with config (any provider or config file)
    is_existing = (
        get_env_value("OPENROUTER_API_KEY") is not None
        or get_env_value("OPENAI_BASE_URL") is not None
        or get_config_path().exists()
    )
    
    # Import migration helpers
    from hermes_cli.config import (
        get_missing_env_vars, get_missing_config_fields,
        check_config_version, migrate_config,
        REQUIRED_ENV_VARS, OPTIONAL_ENV_VARS
    )
    
    # Check what's missing
    missing_required = [v for v in get_missing_env_vars(required_only=False) if v.get("is_required")]
    missing_optional = [v for v in get_missing_env_vars(required_only=False) if not v.get("is_required")]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()
    
    has_missing = missing_required or missing_optional or missing_config or current_ver < latest_ver
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│             ⚕ Hermes Agent Setup Wizard                │", Colors.MAGENTA))
    print(color("├─────────────────────────────────────────────────────────┤", Colors.MAGENTA))
    print(color("│  Let's configure your Hermes Agent installation.       │", Colors.MAGENTA))
    print(color("│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))
    
    # If existing installation, show what's missing and offer quick mode
    quick_mode = False
    if is_existing and has_missing:
        print()
        print_header("Existing Installation Detected")
        print_success("You already have Hermes configured!")
        print()
        
        if missing_required:
            print_warning(f"  {len(missing_required)} required setting(s) missing:")
            for var in missing_required:
                print(f"     • {var['name']}")
        
        if missing_optional:
            print_info(f"  {len(missing_optional)} optional tool(s) not configured:")
            for var in missing_optional[:3]:  # Show first 3
                tools = var.get("tools", [])
                tools_str = f" → {', '.join(tools[:2])}" if tools else ""
                print(f"     • {var['name']}{tools_str}")
            if len(missing_optional) > 3:
                print(f"     • ...and {len(missing_optional) - 3} more")
        
        if missing_config:
            print_info(f"  {len(missing_config)} new config option(s) available")
        
        print()
        
        setup_choices = [
            "Quick setup - just configure missing items",
            "Full setup - reconfigure everything",
            "Skip - exit setup"
        ]
        
        choice = prompt_choice("What would you like to do?", setup_choices, 0)
        
        if choice == 0:
            quick_mode = True
        elif choice == 2:
            print()
            print_info("Exiting. Run 'hermes setup' again when ready.")
            return
        # choice == 1 continues with full setup
        
    elif is_existing and not has_missing:
        print()
        print_header("Configuration Status")
        print_success("Your configuration is complete!")
        print()
        
        if not prompt_yes_no("Would you like to reconfigure anyway?", False):
            print()
            print_info("Exiting. Your configuration is already set up.")
            print_info(f"Config: {get_config_path()}")
            print_info(f"Secrets: {get_env_path()}")
            return
    
    # Quick mode: only configure missing items
    if quick_mode:
        print()
        print_header("Quick Setup - Missing Items Only")
        
        # Handle missing required env vars
        if missing_required:
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
        
        # Handle missing optional env vars — use a checkbox to let the
        # user pick which tools to configure, then prompt for keys.
        # Filter out "advanced" vars (handled elsewhere, e.g. provider step).
        missing_tool_vars = [v for v in missing_optional if not v.get("advanced")]
        
        if missing_tool_vars:
            print()
            print_header("Optional Tools (Quick Setup)")
            
            # Build checklist labels from the missing vars
            checklist_labels = []
            for var in missing_tool_vars:
                tools = var.get("tools", [])
                tools_str = f" → {', '.join(tools[:2])}" if tools else ""
                checklist_labels.append(f"{var['name']}{tools_str}")
            
            selected_indices = prompt_checklist(
                "Which missing tools would you like to configure?",
                checklist_labels,
            )
            
            # Prompt for keys only for selected tools
            for idx in selected_indices:
                var = missing_tool_vars[idx]
                print()
                print(color(f"  {var['name']}", Colors.CYAN))
                if var.get("url"):
                    print_info(f"  Get key at: {var['url']}")
                
                if var.get("password"):
                    value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
                else:
                    value = prompt(f"  {var.get('prompt', var['name'])}")
                
                if value:
                    save_env_value(var["name"], value)
                    print_success(f"  Saved {var['name']}")
        
        # Handle missing config fields
        if missing_config:
            print()
            print_info(f"Adding {len(missing_config)} new config option(s) with defaults...")
            for field in missing_config:
                print_success(f"  Added {field['key']} = {field['default']}")
            
            # Update config version
            config["_config_version"] = latest_ver
            save_config(config)
        
        # Jump to summary
        _print_setup_summary(config, hermes_home)
        return
    
    # =========================================================================
    # Step 0: Show paths (full setup)
    # =========================================================================
    print_header("Configuration Location")
    print_info(f"Config file:  {get_config_path()}")
    print_info(f"Secrets file: {get_env_path()}")
    print_info(f"Data folder:  {hermes_home}")
    print_info(f"Install dir:  {PROJECT_ROOT}")
    print()
    print_info("You can edit these files directly or use 'hermes config edit'")
    
    # =========================================================================
    # Step 1: Inference Provider Selection
    # =========================================================================
    print_header("Inference Provider")
    print_info("Choose how to connect to your main chat model.")
    print()

    # Detect current provider state
    from hermes_cli.auth import (
        get_active_provider, get_provider_auth_state, PROVIDER_REGISTRY,
        format_auth_error, AuthError, fetch_nous_models,
        resolve_nous_runtime_credentials, _update_config_for_provider,
    )
    existing_custom = get_env_value("OPENAI_BASE_URL")
    existing_or = get_env_value("OPENROUTER_API_KEY")
    active_oauth = get_active_provider()

    # Detect if any provider is already configured
    has_any_provider = bool(active_oauth or existing_custom or existing_or)
    
    # Build "keep current" label
    if active_oauth and active_oauth in PROVIDER_REGISTRY:
        keep_label = f"Keep current ({PROVIDER_REGISTRY[active_oauth].name})"
    elif existing_custom:
        keep_label = f"Keep current (Custom: {existing_custom})"
    elif existing_or:
        keep_label = "Keep current (OpenRouter)"
    else:
        keep_label = None  # No provider configured — don't show "Keep current"

    provider_choices = [
        "Login with Nous Portal (Nous Research subscription)",
        "OpenRouter API key (100+ models, pay-per-use)",
        "Custom OpenAI-compatible endpoint (self-hosted / VLLM / etc.)",
    ]
    if keep_label:
        provider_choices.append(keep_label)
    
    # Default to "Keep current" if a provider exists, otherwise OpenRouter (most common)
    default_provider = len(provider_choices) - 1 if has_any_provider else 1
    
    if not has_any_provider:
        print_warning("An inference provider is required for Hermes to work.")
        print()
    
    provider_idx = prompt_choice("Select your inference provider:", provider_choices, default_provider)

    # Track which provider was selected for model step
    selected_provider = None  # "nous", "openrouter", "custom", or None (keep)
    nous_models = []  # populated if Nous login succeeds

    if provider_idx == 0:  # Nous Portal
        selected_provider = "nous"
        print()
        print_header("Nous Portal Login")
        print_info("This will open your browser to authenticate with Nous Portal.")
        print_info("You'll need a Nous Research account with an active subscription.")
        print()

        try:
            from hermes_cli.auth import _login_nous, ProviderConfig
            import argparse
            mock_args = argparse.Namespace(
                portal_url=None, inference_url=None, client_id=None,
                scope=None, no_browser=False, timeout=15.0,
                ca_bundle=None, insecure=False,
            )
            pconfig = PROVIDER_REGISTRY["nous"]
            _login_nous(mock_args, pconfig)

            # Fetch models for the selection step
            try:
                creds = resolve_nous_runtime_credentials(
                    min_key_ttl_seconds=5 * 60, timeout_seconds=15.0,
                )
                nous_models = fetch_nous_models(
                    inference_base_url=creds.get("base_url", ""),
                    api_key=creds.get("api_key", ""),
                )
            except Exception as e:
                logger.debug("Could not fetch Nous models after login: %s", e)

        except SystemExit:
            print_warning("Nous Portal login was cancelled or failed.")
            print_info("You can try again later with: hermes login")
            selected_provider = None
        except Exception as e:
            print_error(f"Login failed: {e}")
            print_info("You can try again later with: hermes login")
            selected_provider = None

    elif provider_idx == 1:  # OpenRouter
        selected_provider = "openrouter"
        print()
        print_header("OpenRouter API Key")
        print_info("OpenRouter provides access to 100+ models from multiple providers.")
        print_info("Get your API key at: https://openrouter.ai/keys")

        if existing_or:
            print_info(f"Current: {existing_or[:8]}... (configured)")
            if prompt_yes_no("Update OpenRouter API key?", False):
                api_key = prompt("  OpenRouter API key", password=True)
                if api_key:
                    save_env_value("OPENROUTER_API_KEY", api_key)
                    print_success("OpenRouter API key updated")
        else:
            api_key = prompt("  OpenRouter API key", password=True)
            if api_key:
                save_env_value("OPENROUTER_API_KEY", api_key)
                print_success("OpenRouter API key saved")
            else:
                print_warning("Skipped - agent won't work without an API key")

        # Clear any custom endpoint if switching to OpenRouter
        if existing_custom:
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")

    elif provider_idx == 2:  # Custom endpoint
        selected_provider = "custom"
        print()
        print_header("Custom OpenAI-Compatible Endpoint")
        print_info("Works with any API that follows OpenAI's chat completions spec")

        current_url = get_env_value("OPENAI_BASE_URL") or ""
        current_key = get_env_value("OPENAI_API_KEY")
        current_model = config.get('model', '')

        if current_url:
            print_info(f"  Current URL: {current_url}")
        if current_key:
            print_info(f"  Current key: {current_key[:8]}... (configured)")

        base_url = prompt("  API base URL (e.g., https://api.example.com/v1)", current_url)
        api_key = prompt("  API key", password=True)
        model_name = prompt("  Model name (e.g., gpt-4, claude-3-opus)", current_model)

        if base_url:
            save_env_value("OPENAI_BASE_URL", base_url)
        if api_key:
            save_env_value("OPENAI_API_KEY", api_key)
        if model_name:
            config['model'] = model_name
            save_env_value("LLM_MODEL", model_name)
        print_success("Custom endpoint configured")
    # else: provider_idx == 3 (Keep current) — only shown when a provider already exists

    # =========================================================================
    # Step 1b: OpenRouter API Key for tools (if not already set)
    # =========================================================================
    # Tools (vision, web, MoA) use OpenRouter independently of the main provider.
    # Prompt for OpenRouter key if not set and a non-OpenRouter provider was chosen.
    if selected_provider in ("nous", "custom") and not get_env_value("OPENROUTER_API_KEY"):
        print()
        print_header("OpenRouter API Key (for tools)")
        print_info("Tools like vision analysis, web search, and MoA use OpenRouter")
        print_info("independently of your main inference provider.")
        print_info("Get your API key at: https://openrouter.ai/keys")

        api_key = prompt("  OpenRouter API key (optional, press Enter to skip)", password=True)
        if api_key:
            save_env_value("OPENROUTER_API_KEY", api_key)
            print_success("OpenRouter API key saved (for tools)")
        else:
            print_info("Skipped - some tools (vision, web scraping) won't work without this")

    # =========================================================================
    # Step 2: Model Selection (adapts based on provider)
    # =========================================================================
    if selected_provider != "custom":  # Custom already prompted for model name
        print_header("Default Model")

        current_model = config.get('model', 'anthropic/claude-opus-4.6')
        print_info(f"Current: {current_model}")

        if selected_provider == "nous" and nous_models:
            # Dynamic model list from Nous Portal
            model_choices = [f"{m}" for m in nous_models]
            model_choices.append("Custom model")
            model_choices.append(f"Keep current ({current_model})")

            # Post-login validation: warn if current model might not be available
            if current_model and current_model not in nous_models:
                print_warning(f"Your current model ({current_model}) may not be available via Nous Portal.")
                print_info("Select a model from the list, or keep current to use it anyway.")
                print()

            model_idx = prompt_choice("Select default model:", model_choices, len(model_choices) - 1)

            if model_idx < len(nous_models):
                config['model'] = nous_models[model_idx]
                save_env_value("LLM_MODEL", nous_models[model_idx])
            elif model_idx == len(nous_models):  # Custom
                custom = prompt("Enter model name")
                if custom:
                    config['model'] = custom
                    save_env_value("LLM_MODEL", custom)
            # else: keep current
        else:
            # Static list for OpenRouter / fallback (from canonical list)
            from hermes_cli.models import model_ids, menu_labels

            ids = model_ids()
            model_choices = menu_labels() + [
                "Custom model",
                f"Keep current ({current_model})",
            ]

            keep_idx = len(model_choices) - 1
            model_idx = prompt_choice("Select default model:", model_choices, keep_idx)

            if model_idx < len(ids):
                config['model'] = ids[model_idx]
                save_env_value("LLM_MODEL", ids[model_idx])
            elif model_idx == len(ids):  # Custom
                custom = prompt("Enter model name (e.g., anthropic/claude-opus-4.6)")
                if custom:
                    config['model'] = custom
                    save_env_value("LLM_MODEL", custom)
            # else: Keep current
    
    # =========================================================================
    # Step 4: Terminal Backend
    # =========================================================================
    print_header("Terminal Backend")
    print_info("The terminal tool allows the agent to run commands.")
    
    current_backend = config.get('terminal', {}).get('backend', 'local')
    print_info(f"Current: {current_backend}")
    
    # Detect platform for backend availability
    import platform
    is_linux = platform.system() == "Linux"
    is_macos = platform.system() == "Darwin"
    is_windows = platform.system() == "Windows"
    
    # Build choices based on platform
    terminal_choices = [
        "Local (run commands on this machine - no isolation)",
        "Docker (isolated containers - recommended for security)",
    ]
    
    # Singularity/Apptainer is Linux-only (HPC)
    if is_linux:
        terminal_choices.append("Singularity/Apptainer (HPC clusters, shared compute)")
    
    terminal_choices.extend([
        "Modal (cloud execution, GPU access, serverless)",
        "SSH (run commands on a remote server)",
        f"Keep current ({current_backend})"
    ])
    
    # Build index map based on available choices
    if is_linux:
        backend_to_idx = {'local': 0, 'docker': 1, 'singularity': 2, 'modal': 3, 'ssh': 4}
        idx_to_backend = {0: 'local', 1: 'docker', 2: 'singularity', 3: 'modal', 4: 'ssh'}
        keep_current_idx = 5
    else:
        backend_to_idx = {'local': 0, 'docker': 1, 'modal': 2, 'ssh': 3}
        idx_to_backend = {0: 'local', 1: 'docker', 2: 'modal', 3: 'ssh'}
        keep_current_idx = 4
        if current_backend == 'singularity':
            print_warning("Singularity is only available on Linux - please select a different backend")
    
    # Default based on current
    default_terminal = backend_to_idx.get(current_backend, 0)
    
    terminal_idx = prompt_choice("Select terminal backend:", terminal_choices, keep_current_idx)
    
    # Map index to backend name (handles platform differences)
    selected_backend = idx_to_backend.get(terminal_idx)
    
    if selected_backend == 'local':
        config.setdefault('terminal', {})['backend'] = 'local'
        print_info("Local Execution Configuration:")
        print_info("Commands run directly on this machine (no isolation)")
        
        if is_windows:
            print_info("Note: On Windows, commands run via cmd.exe or PowerShell")
        
        # Messaging working directory configuration
        print_info("")
        print_info("Working Directory for Messaging (Telegram/Discord/etc):")
        print_info("  The CLI always uses the directory you run 'hermes' from")
        print_info("  But messaging bots need a static starting directory")
        
        current_cwd = get_env_value('MESSAGING_CWD') or str(Path.home())
        print_info(f"  Current: {current_cwd}")
        
        cwd_input = prompt("  Messaging working directory", current_cwd)
        # Expand ~ to full path
        if cwd_input.startswith('~'):
            cwd_expanded = str(Path.home()) + cwd_input[1:]
        else:
            cwd_expanded = cwd_input
        save_env_value("MESSAGING_CWD", cwd_expanded)
        
        if prompt_yes_no("  Enable sudo support? (allows agent to run sudo commands)", False):
            print_warning("  SECURITY WARNING: Sudo password will be stored in plaintext")
            sudo_pass = prompt("  Sudo password (leave empty to skip)", password=True)
            if sudo_pass:
                save_env_value("SUDO_PASSWORD", sudo_pass)
                print_success("  Sudo password saved")
        
        print_success("Terminal set to local")
    
    elif selected_backend == 'docker':
        config.setdefault('terminal', {})['backend'] = 'docker'
        default_docker = config.get('terminal', {}).get('docker_image', 'nikolaik/python-nodejs:python3.11-nodejs20')
        print_info("Docker Configuration:")
        if is_macos:
            print_info("Requires Docker Desktop for Mac")
        elif is_windows:
            print_info("Requires Docker Desktop for Windows")
        docker_image = prompt("  Docker image", default_docker)
        config['terminal']['docker_image'] = docker_image
        print_success("Terminal set to Docker")
    
    elif selected_backend == 'singularity':
        config.setdefault('terminal', {})['backend'] = 'singularity'
        default_singularity = config.get('terminal', {}).get('singularity_image', 'docker://nikolaik/python-nodejs:python3.11-nodejs20')
        print_info("Singularity/Apptainer Configuration:")
        print_info("Requires apptainer or singularity to be installed")
        singularity_image = prompt("  Image (docker:// prefix for Docker Hub)", default_singularity)
        config['terminal']['singularity_image'] = singularity_image
        print_success("Terminal set to Singularity/Apptainer")
    
    elif selected_backend == 'modal':
        config.setdefault('terminal', {})['backend'] = 'modal'
        default_modal = config.get('terminal', {}).get('modal_image', 'nikolaik/python-nodejs:python3.11-nodejs20')
        print_info("Modal Cloud Configuration:")
        print_info("Get credentials at: https://modal.com/settings")
        
        # Check if swe-rex[modal] is installed, install if missing
        try:
            from swerex.deployment.modal import ModalDeployment
            print_info("swe-rex[modal] package: installed ✓")
        except ImportError:
            print_info("Installing required package: swe-rex[modal]...")
            import subprocess
            import shutil
            # Prefer uv for speed, fall back to pip
            uv_bin = shutil.which("uv")
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "swe-rex[modal]>=1.4.0"],
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "swe-rex[modal]>=1.4.0"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                print_success("swe-rex[modal] installed (includes modal + boto3)")
            else:
                print_warning("Failed to install swe-rex[modal] — install manually:")
                print_info('  uv pip install "swe-rex[modal]>=1.4.0"')
        
        # Always show current status and allow reconfiguration
        current_token = get_env_value('MODAL_TOKEN_ID')
        if current_token:
            print_info(f"  Token ID: {current_token[:8]}... (configured)")
        
        modal_image = prompt("  Container image", default_modal)
        config['terminal']['modal_image'] = modal_image
        
        token_id = prompt("  Modal token ID", current_token or "")
        token_secret = prompt("  Modal token secret", password=True)
        
        if token_id:
            save_env_value("MODAL_TOKEN_ID", token_id)
        if token_secret:
            save_env_value("MODAL_TOKEN_SECRET", token_secret)
        
        print_success("Terminal set to Modal")
    
    elif selected_backend == 'ssh':
        config.setdefault('terminal', {})['backend'] = 'ssh'
        print_info("SSH Remote Execution Configuration:")
        print_info("Commands will run on a remote server over SSH")
        
        current_host = get_env_value('TERMINAL_SSH_HOST') or ''
        current_user = get_env_value('TERMINAL_SSH_USER') or os.getenv("USER", "")
        current_port = get_env_value('TERMINAL_SSH_PORT') or '22'
        current_key = get_env_value('TERMINAL_SSH_KEY') or '~/.ssh/id_rsa'
        
        if current_host:
            print_info(f"  Current host: {current_user}@{current_host}:{current_port}")
        
        ssh_host = prompt("  SSH host", current_host)
        ssh_user = prompt("  SSH user", current_user)
        ssh_port = prompt("  SSH port", current_port)
        ssh_key = prompt("  SSH key path (or leave empty for ssh-agent)", current_key)
        
        if ssh_host:
            save_env_value("TERMINAL_SSH_HOST", ssh_host)
        if ssh_user:
            save_env_value("TERMINAL_SSH_USER", ssh_user)
        if ssh_port and ssh_port != '22':
            save_env_value("TERMINAL_SSH_PORT", ssh_port)
        if ssh_key:
            save_env_value("TERMINAL_SSH_KEY", ssh_key)
        
        print_success("Terminal set to SSH")
    # else: Keep current (selected_backend is None)
    
    # =========================================================================
    # Step 5: Agent Settings
    # =========================================================================
    print_header("Agent Settings")
    
    # Max iterations
    current_max = get_env_value('HERMES_MAX_ITERATIONS') or '60'
    print_info("Maximum tool-calling iterations per conversation.")
    print_info("Higher = more complex tasks, but costs more tokens.")
    print_info("Recommended: 30-60 for most tasks, 100+ for open exploration.")
    
    max_iter_str = prompt("Max iterations", current_max)
    try:
        max_iter = int(max_iter_str)
        if max_iter > 0:
            save_env_value("HERMES_MAX_ITERATIONS", str(max_iter))
            config['max_turns'] = max_iter
            print_success(f"Max iterations set to {max_iter}")
    except ValueError:
        print_warning("Invalid number, keeping current value")
    
    # Tool progress notifications (for messaging)
    print_info("")
    print_info("Tool Progress Notifications (Messaging only)")
    print_info("Send status messages when the agent uses tools.")
    print_info("Example: '💻 ls -la...' or '🔍 web_search...'")
    
    current_progress = get_env_value('HERMES_TOOL_PROGRESS') or 'false'
    if prompt_yes_no("Enable tool progress messages?", current_progress.lower() in ('1', 'true', 'yes')):
        save_env_value("HERMES_TOOL_PROGRESS", "true")
        
        # Progress mode
        current_mode = get_env_value('HERMES_TOOL_PROGRESS_MODE') or 'new'
        print_info("  Mode options:")
        print_info("    'new' - Only when switching tools (less spam)")
        print_info("    'all' - Every tool call")
        mode = prompt("  Progress mode", current_mode)
        if mode.lower() in ('all', 'new'):
            save_env_value("HERMES_TOOL_PROGRESS_MODE", mode.lower())
        print_success("Tool progress enabled")
    else:
        save_env_value("HERMES_TOOL_PROGRESS", "false")
    
    # =========================================================================
    # Step 6: Context Compression
    # =========================================================================
    print_header("Context Compression")
    print_info("Automatically summarize old messages when context gets too long.")
    
    compression = config.get('compression', {})
    current_enabled = compression.get('enabled', True)
    
    if prompt_yes_no(f"Enable context compression?", current_enabled):
        config.setdefault('compression', {})['enabled'] = True
        
        current_threshold = compression.get('threshold', 0.85)
        threshold_str = prompt(f"Compression threshold (0.5-0.95)", str(current_threshold))
        try:
            threshold = float(threshold_str)
            if 0.5 <= threshold <= 0.95:
                config['compression']['threshold'] = threshold
        except ValueError:
            pass
        
        print_success("Context compression enabled")
    else:
        config.setdefault('compression', {})['enabled'] = False
    
    # =========================================================================
    # Step 7: Messaging Platforms (Optional)
    # =========================================================================
    print_header("Messaging Platforms (Optional)")
    print_info("Connect to messaging platforms to chat with Hermes from anywhere.")
    
    # Telegram
    existing_telegram = get_env_value('TELEGRAM_BOT_TOKEN')
    if existing_telegram:
        print_info("Telegram: already configured")
        if prompt_yes_no("Reconfigure Telegram?", False):
            existing_telegram = None
    
    if not existing_telegram and prompt_yes_no("Set up Telegram bot?", False):
        print_info("Create a bot via @BotFather on Telegram")
        token = prompt("Telegram bot token", password=True)
        if token:
            save_env_value("TELEGRAM_BOT_TOKEN", token)
            print_success("Telegram token saved")
            
            # Allowed users (security)
            print()
            print_info("🔒 Security: Restrict who can use your bot")
            print_info("   To find your Telegram user ID:")
            print_info("   1. Message @userinfobot on Telegram")
            print_info("   2. It will reply with your numeric ID (e.g., 123456789)")
            print()
            allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
            if allowed_users:
                save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users.replace(" ", ""))
                print_success("Telegram allowlist configured - only listed users can use the bot")
            else:
                print_info("⚠️  No allowlist set - anyone who finds your bot can use it!")
            
            # Home channel setup with better guidance
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
    
    # Check/update existing Telegram allowlist
    elif existing_telegram:
        existing_allowlist = get_env_value('TELEGRAM_ALLOWED_USERS')
        if not existing_allowlist:
            print_info("⚠️  Telegram has no user allowlist - anyone can use your bot!")
            if prompt_yes_no("Add allowed users now?", True):
                print_info("   To find your Telegram user ID: message @userinfobot")
                allowed_users = prompt("Allowed user IDs (comma-separated)")
                if allowed_users:
                    save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users.replace(" ", ""))
                    print_success("Telegram allowlist configured")
    
    # Discord
    existing_discord = get_env_value('DISCORD_BOT_TOKEN')
    if existing_discord:
        print_info("Discord: already configured")
        if prompt_yes_no("Reconfigure Discord?", False):
            existing_discord = None
    
    if not existing_discord and prompt_yes_no("Set up Discord bot?", False):
        print_info("Create a bot at https://discord.com/developers/applications")
        token = prompt("Discord bot token", password=True)
        if token:
            save_env_value("DISCORD_BOT_TOKEN", token)
            print_success("Discord token saved")
            
            # Allowed users (security)
            print()
            print_info("🔒 Security: Restrict who can use your bot")
            print_info("   To find your Discord user ID:")
            print_info("   1. Enable Developer Mode in Discord settings")
            print_info("   2. Right-click your name → Copy ID")
            print()
            print_info("   You can also use Discord usernames (resolved on gateway start).")
            print()
            allowed_users = prompt("Allowed user IDs or usernames (comma-separated, leave empty for open access)")
            if allowed_users:
                save_env_value("DISCORD_ALLOWED_USERS", allowed_users.replace(" ", ""))
                print_success("Discord allowlist configured")
            else:
                print_info("⚠️  No allowlist set - anyone in servers with your bot can use it!")
            
            # Home channel setup with better guidance
            print()
            print_info("📬 Home Channel: where Hermes delivers cron job results,")
            print_info("   cross-platform messages, and notifications.")
            print_info("   To get a channel ID: right-click a channel → Copy Channel ID")
            print_info("   (requires Developer Mode in Discord settings)")
            print_info("   You can also set this later by typing /set-home in a Discord channel.")
            home_channel = prompt("Home channel ID (leave empty to set later with /set-home)")
            if home_channel:
                save_env_value("DISCORD_HOME_CHANNEL", home_channel)
    
    # Check/update existing Discord allowlist
    elif existing_discord:
        existing_allowlist = get_env_value('DISCORD_ALLOWED_USERS')
        if not existing_allowlist:
            print_info("⚠️  Discord has no user allowlist - anyone can use your bot!")
            if prompt_yes_no("Add allowed users now?", True):
                print_info("   To find Discord ID: Enable Developer Mode, right-click name → Copy ID")
                allowed_users = prompt("Allowed user IDs (comma-separated)")
                if allowed_users:
                    save_env_value("DISCORD_ALLOWED_USERS", allowed_users.replace(" ", ""))
                    print_success("Discord allowlist configured")
    
    # Slack
    existing_slack = get_env_value('SLACK_BOT_TOKEN')
    if existing_slack:
        print_info("Slack: already configured")
        if prompt_yes_no("Reconfigure Slack?", False):
            existing_slack = None
    
    if not existing_slack and prompt_yes_no("Set up Slack bot?", False):
        print_info("Steps to create a Slack app:")
        print_info("   1. Go to https://api.slack.com/apps → Create New App")
        print_info("   2. Enable Socket Mode: App Settings → Socket Mode → Enable")
        print_info("   3. Bot Token: OAuth & Permissions → Install to Workspace")
        print_info("   4. App Token: Basic Information → App-Level Tokens → Generate")
        print()
        bot_token = prompt("Slack Bot Token (xoxb-...)", password=True)
        if bot_token:
            save_env_value("SLACK_BOT_TOKEN", bot_token)
            app_token = prompt("Slack App Token (xapp-...)", password=True)
            if app_token:
                save_env_value("SLACK_APP_TOKEN", app_token)
            print_success("Slack tokens saved")
            
            print()
            print_info("🔒 Security: Restrict who can use your bot")
            print_info("   Find Slack user IDs in your profile or via the Slack API")
            print()
            allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
            if allowed_users:
                save_env_value("SLACK_ALLOWED_USERS", allowed_users.replace(" ", ""))
                print_success("Slack allowlist configured")
            else:
                print_info("⚠️  No allowlist set - anyone in your workspace can use the bot!")
    
    # WhatsApp
    existing_whatsapp = get_env_value('WHATSAPP_ENABLED')
    if not existing_whatsapp and prompt_yes_no("Set up WhatsApp?", False):
        print_info("WhatsApp uses a bridge service for connectivity.")
        print_info("See docs/messaging.md for detailed WhatsApp setup instructions.")
        print()
        if prompt_yes_no("Enable WhatsApp bridge?", True):
            save_env_value("WHATSAPP_ENABLED", "true")
            print_success("WhatsApp enabled")
            print_info("Run 'hermes gateway' to complete WhatsApp pairing via QR code")
    
    # Gateway reminder
    any_messaging = (
        get_env_value('TELEGRAM_BOT_TOKEN')
        or get_env_value('DISCORD_BOT_TOKEN')
        or get_env_value('SLACK_BOT_TOKEN')
        or get_env_value('WHATSAPP_ENABLED')
    )
    if any_messaging:
        print()
        print_info("━" * 50)
        print_success("Messaging platforms configured!")
        print_info("Start the gateway after setup to bring your bots online:")
        print_info("   hermes gateway              # Run in foreground")
        print_info("   hermes gateway install      # Install as background service (Linux)")
        
        # Check if any home channels are missing
        missing_home = []
        if get_env_value('TELEGRAM_BOT_TOKEN') and not get_env_value('TELEGRAM_HOME_CHANNEL'):
            missing_home.append("Telegram")
        if get_env_value('DISCORD_BOT_TOKEN') and not get_env_value('DISCORD_HOME_CHANNEL'):
            missing_home.append("Discord")
        if get_env_value('SLACK_BOT_TOKEN') and not get_env_value('SLACK_HOME_CHANNEL'):
            missing_home.append("Slack")
        
        if missing_home:
            print()
            print_info(f"⚠️  No home channel set for: {', '.join(missing_home)}")
            print_info("   Without a home channel, cron jobs and cross-platform")
            print_info("   messages can't be delivered to those platforms.")
            print_info("   Set one later with /set-home in your chat, or:")
            for plat in missing_home:
                print_info(f"     hermes config set {plat.upper()}_HOME_CHANNEL <channel_id>")
        
        print_info("━" * 50)
    
    # =========================================================================
    # Step 8: Additional Tools (Checkbox Selection)
    # =========================================================================
    print_header("Additional Tools")
    print_info("Select which tools you'd like to configure.")
    print_info("You can always add more later with 'hermes setup'.")
    print()
    
    # Define tool categories for the checklist.
    # Each entry: (display_label, setup_function_key, check_keys)
    # check_keys = env vars that indicate this tool is already configured
    TOOL_CATEGORIES = [
        {
            "label": "🔍 Web Search & Scraping (Firecrawl)",
            "key": "firecrawl",
            "check": ["FIRECRAWL_API_KEY"],
        },
        {
            "label": "🌐 Browser Automation (Browserbase)",
            "key": "browserbase",
            "check": ["BROWSERBASE_API_KEY"],
        },
        {
            "label": "🎨 Image Generation (FAL / FLUX)",
            "key": "fal",
            "check": ["FAL_KEY"],
        },
        {
            "label": "🎤 Voice Transcription & TTS (OpenAI Whisper + TTS)",
            "key": "openai_voice",
            "check": ["HERMES_OPENAI_API_KEY"],
        },
        {
            "label": "🗣️ Premium Text-to-Speech (ElevenLabs)",
            "key": "elevenlabs",
            "check": ["ELEVENLABS_API_KEY"],
        },
        {
            "label": "🧪 RL Training (Tinker + WandB)",
            "key": "rl_training",
            "check": ["TINKER_API_KEY", "WANDB_API_KEY"],
        },
        {
            "label": "🔧 Skills Hub (GitHub token for higher rate limits)",
            "key": "github",
            "check": ["GITHUB_TOKEN"],
        },
    ]
    
    # Pre-select tools that are already configured
    pre_selected = []
    for i, cat in enumerate(TOOL_CATEGORIES):
        if all(get_env_value(k) for k in cat["check"]):
            pre_selected.append(i)
    
    checklist_labels = [cat["label"] for cat in TOOL_CATEGORIES]
    selected_indices = prompt_checklist(
        "Which tools would you like to enable?",
        checklist_labels,
        pre_selected=pre_selected,
    )
    
    selected_keys = {TOOL_CATEGORIES[i]["key"] for i in selected_indices}
    
    # Now prompt for API keys only for the tools the user selected
    
    if "firecrawl" in selected_keys:
        print()
        print(color("  ─── Web Search & Scraping (Firecrawl) ───", Colors.CYAN))
        print_info("  Get your API key at: https://firecrawl.dev/")
        existing = get_env_value('FIRECRAWL_API_KEY')
        if existing:
            print_success("  Already configured ✓")
            if prompt_yes_no("  Update API key?", False):
                api_key = prompt("    Firecrawl API key", password=True)
                if api_key:
                    save_env_value("FIRECRAWL_API_KEY", api_key)
                    print_success("    Updated")
        else:
            api_key = prompt("    Firecrawl API key", password=True)
            if api_key:
                save_env_value("FIRECRAWL_API_KEY", api_key)
                print_success("    Configured ✓")
    
    if "browserbase" in selected_keys:
        print()
        print(color("  ─── Browser Automation (Browserbase) ───", Colors.CYAN))
        print_info("  Get credentials at: https://browserbase.com/")
        existing = get_env_value('BROWSERBASE_API_KEY')
        if existing:
            print_success("  Already configured ✓")
            if prompt_yes_no("  Update credentials?", False):
                api_key = prompt("    API key", password=True)
                project_id = prompt("    Project ID")
                if api_key:
                    save_env_value("BROWSERBASE_API_KEY", api_key)
                if project_id:
                    save_env_value("BROWSERBASE_PROJECT_ID", project_id)
                print_success("    Updated")
        else:
            api_key = prompt("    Browserbase API key", password=True)
            project_id = prompt("    Browserbase Project ID")
            if api_key:
                save_env_value("BROWSERBASE_API_KEY", api_key)
            if project_id:
                save_env_value("BROWSERBASE_PROJECT_ID", project_id)
            
            # Auto-install Node.js deps if possible
            import shutil
            node_modules = PROJECT_ROOT / "node_modules" / "agent-browser"
            if not node_modules.exists() and shutil.which("npm"):
                print_info("    Installing Node.js dependencies for browser tools...")
                import subprocess
                result = subprocess.run(
                    ["npm", "install", "--silent"],
                    capture_output=True, text=True, cwd=str(PROJECT_ROOT)
                )
                if result.returncode == 0:
                    print_success("    Node.js dependencies installed")
                else:
                    print_warning("    npm install failed — run manually: cd ~/.hermes/hermes-agent && npm install")
            elif not node_modules.exists():
                print_warning("    Node.js not found — browser tools require: npm install (in the hermes-agent directory)")
            
            if api_key:
                print_success("    Configured ✓")
    
    if "fal" in selected_keys:
        print()
        print(color("  ─── Image Generation (FAL) ───", Colors.CYAN))
        print_info("  Get your API key at: https://fal.ai/")
        existing = get_env_value('FAL_KEY')
        if existing:
            print_success("  Already configured ✓")
            if prompt_yes_no("  Update API key?", False):
                api_key = prompt("    FAL API key", password=True)
                if api_key:
                    save_env_value("FAL_KEY", api_key)
                    print_success("    Updated")
        else:
            api_key = prompt("    FAL API key", password=True)
            if api_key:
                save_env_value("FAL_KEY", api_key)
                print_success("    Configured ✓")
    
    if "openai_voice" in selected_keys:
        print()
        print(color("  ─── Voice Transcription & TTS (OpenAI) ───", Colors.CYAN))
        print_info("  Used for Whisper speech-to-text and OpenAI TTS voices.")
        print_info("  Get your API key at: https://platform.openai.com/api-keys")
        existing = get_env_value('HERMES_OPENAI_API_KEY')
        if existing:
            print_success("  Already configured ✓")
            if prompt_yes_no("  Update API key?", False):
                api_key = prompt("    OpenAI API key", password=True)
                if api_key:
                    save_env_value("HERMES_OPENAI_API_KEY", api_key)
                    print_success("    Updated")
        else:
            api_key = prompt("    OpenAI API key", password=True)
            if api_key:
                save_env_value("HERMES_OPENAI_API_KEY", api_key)
                print_success("    Configured ✓")
    
    if "elevenlabs" in selected_keys:
        print()
        print(color("  ─── Premium TTS (ElevenLabs) ───", Colors.CYAN))
        print_info("  High-quality voice synthesis. Free Edge TTS works without a key.")
        print_info("  Get your API key at: https://elevenlabs.io/")
        existing = get_env_value('ELEVENLABS_API_KEY')
        if existing:
            print_success("  Already configured ✓")
            if prompt_yes_no("  Update API key?", False):
                api_key = prompt("    ElevenLabs API key", password=True)
                if api_key:
                    save_env_value("ELEVENLABS_API_KEY", api_key)
                    print_success("    Updated")
        else:
            api_key = prompt("    ElevenLabs API key", password=True)
            if api_key:
                save_env_value("ELEVENLABS_API_KEY", api_key)
                print_success("    Configured ✓")
    
    if "rl_training" in selected_keys:
        print()
        print(color("  ─── RL Training (Tinker + WandB) ───", Colors.CYAN))
        
        rl_python_ok = sys.version_info >= (3, 11)
        if not rl_python_ok:
            print_error(f"  Requires Python 3.11+ (current: {sys.version_info.major}.{sys.version_info.minor})")
            print_info("  Upgrade Python and reinstall to enable RL training tools")
        else:
            print_info("  Get Tinker key at: https://tinker-console.thinkingmachines.ai/keys")
            print_info("  Get WandB key at: https://wandb.ai/authorize")
            
            tinker_existing = get_env_value('TINKER_API_KEY')
            wandb_existing = get_env_value('WANDB_API_KEY')
            
            if tinker_existing and wandb_existing:
                print_success("  Already configured ✓")
                if prompt_yes_no("  Update credentials?", False):
                    api_key = prompt("    Tinker API key", password=True)
                    if api_key:
                        save_env_value("TINKER_API_KEY", api_key)
                    wandb_key = prompt("    WandB API key", password=True)
                    if wandb_key:
                        save_env_value("WANDB_API_KEY", wandb_key)
                    print_success("    Updated")
            else:
                api_key = prompt("    Tinker API key", password=True)
                if api_key:
                    save_env_value("TINKER_API_KEY", api_key)
                wandb_key = prompt("    WandB API key", password=True)
                if wandb_key:
                    save_env_value("WANDB_API_KEY", wandb_key)
                
                # Auto-install tinker-atropos submodule if missing
                try:
                    __import__("tinker_atropos")
                except ImportError:
                    tinker_dir = PROJECT_ROOT / "tinker-atropos"
                    if tinker_dir.exists() and (tinker_dir / "pyproject.toml").exists():
                        print_info("    Installing tinker-atropos submodule...")
                        import subprocess
                        import shutil
                        uv_bin = shutil.which("uv")
                        if uv_bin:
                            result = subprocess.run(
                                [uv_bin, "pip", "install", "-e", str(tinker_dir)],
                                capture_output=True, text=True
                            )
                        else:
                            result = subprocess.run(
                                [sys.executable, "-m", "pip", "install", "-e", str(tinker_dir)],
                                capture_output=True, text=True
                            )
                        if result.returncode == 0:
                            print_success("    tinker-atropos installed")
                        else:
                            print_warning("    tinker-atropos install failed — run manually:")
                            print_info('      uv pip install -e "./tinker-atropos"')
                    else:
                        print_warning("    tinker-atropos submodule not found — run:")
                        print_info("      git submodule update --init --recursive")
                        print_info('      uv pip install -e "./tinker-atropos"')
                
                if api_key and wandb_key:
                    print_success("    Configured ✓")
                else:
                    print_warning("    Partially configured (both keys required)")
    
    if "github" in selected_keys:
        print()
        print(color("  ─── Skills Hub (GitHub) ───", Colors.CYAN))
        print_info("  Enables higher API rate limits for skill search/install")
        print_info("  and publishing skills via GitHub PRs.")
        print_info("  Get a token at: https://github.com/settings/tokens")
        existing = get_env_value('GITHUB_TOKEN')
        if existing:
            print_success("  Already configured ✓")
            if prompt_yes_no("  Update token?", False):
                token = prompt("    GitHub Token (ghp_...)", password=True)
                if token:
                    save_env_value("GITHUB_TOKEN", token)
                    print_success("    Updated")
        else:
            token = prompt("    GitHub Token", password=True)
            if token:
                save_env_value("GITHUB_TOKEN", token)
                print_success("    Configured ✓")

    # =========================================================================
    # Save config and show summary
    # =========================================================================
    save_config(config)
    _print_setup_summary(config, hermes_home)
