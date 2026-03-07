"""
Interactive setup wizard for Hermes Agent.

Modular wizard with independently-runnable sections:
  1. Model & Provider — choose your AI provider and model
  2. Terminal Backend — where your agent runs commands
  3. Messaging Platforms — connect Telegram, Discord, etc.
  4. Tools — configure TTS, web search, image generation, etc.
  5. Agent Settings — iterations, compression, session reset

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
    """Prompt for a choice from a list with arrow key navigation.
    
    Escape keeps the current default (skips the question).
    Ctrl+C exits the wizard.
    """
    print(color(question, Colors.YELLOW))
    
    # Try to use interactive menu if available
    try:
        from simple_term_menu import TerminalMenu
        import re
        
        # Strip emoji characters — simple_term_menu miscalculates visual
        # width of emojis, causing duplicated/garbled lines on redraw.
        _emoji_re = re.compile(
            "[\U0001f300-\U0001f9ff\U00002600-\U000027bf\U0000fe00-\U0000fe0f"
            "\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\u200d]+", flags=re.UNICODE
        )
        menu_choices = [f"  {_emoji_re.sub('', choice).strip()}" for choice in choices]
        
        print_info("  ↑/↓ Navigate  Enter Select  Esc Skip  Ctrl+C Exit")
        
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
        if idx is None:  # User pressed Escape — keep current value
            print_info(f"  Skipped (keeping current)")
            print()
            return default
        print()  # Add newline after selection
        return idx
        
    except (ImportError, NotImplementedError):
        pass
    except Exception as e:
        print(f"  (Interactive menu unavailable: {e})")

    # Fallback to number-based selection (simple_term_menu doesn't support Windows)
    for i, choice in enumerate(choices):
        marker = "●" if i == default else "○"
        if i == default:
            print(color(f"  {marker} {choice}", Colors.GREEN))
        else:
            print(f"  {marker} {choice}")

    print_info(f"  Enter for default ({default + 1})  Ctrl+C to exit")

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
    """Prompt for yes/no. Ctrl+C exits, empty input returns default."""
    default_str = "Y/n" if default else "y/N"
    
    while True:
        try:
            value = input(color(f"{question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)
        
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
    print_info("  SPACE Toggle  ENTER Confirm  ESC Skip  Ctrl+C Exit")
    print()
    
    try:
        from simple_term_menu import TerminalMenu
        import re
        
        # Strip emoji characters from menu labels — simple_term_menu miscalculates
        # visual width of emojis on macOS, causing duplicated/garbled lines.
        _emoji_re = re.compile(
            "[\U0001f300-\U0001f9ff\U00002600-\U000027bf\U0000fe00-\U0000fe0f"
            "\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\u200d]+", flags=re.UNICODE
        )
        menu_items = [f"  {_emoji_re.sub('', item).strip()}" for item in items]
        
        # Map pre-selected indices to the actual menu entry strings
        preselected = [menu_items[i] for i in pre_selected if i < len(menu_items)]
        
        terminal_menu = TerminalMenu(
            menu_items,
            multi_select=True,
            show_multi_select_hint=False,
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
            print_info("  Skipped (keeping current)")
            return list(pre_selected)
        
        selected = list(terminal_menu.chosen_menu_indices or [])
        return selected
        
    except (ImportError, NotImplementedError):
        # Fallback: numbered toggle interface (simple_term_menu doesn't support Windows)
        selected = set(pre_selected)
        
        while True:
            for i, item in enumerate(items):
                marker = color("[✓]", Colors.GREEN) if i in selected else "[ ]"
                print(f"  {marker} {i + 1}. {item}")
            print()
            
            try:
                value = input(color("  Toggle # (or Enter to confirm): ", Colors.DIM)).strip()
                if not value:
                    break
                idx = int(value) - 1
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
        print_success(f"  ✓ Saved")
    else:
        print_warning(f"  Skipped (configure later with 'hermes setup')")


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
    if get_env_value('FIRECRAWL_API_KEY') or get_env_value('FIRECRAWL_API_URL'):
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
    
    # TTS — show configured provider
    tts_provider = config.get('tts', {}).get('provider', 'edge')
    if tts_provider == 'elevenlabs' and get_env_value('ELEVENLABS_API_KEY'):
        tool_status.append(("Text-to-Speech (ElevenLabs)", True, None))
    elif tts_provider == 'openai' and get_env_value('VOICE_TOOLS_OPENAI_KEY'):
        tool_status.append(("Text-to-Speech (OpenAI)", True, None))
    else:
        tool_status.append(("Text-to-Speech (Edge TTS)", True, None))
    
    # Tinker + WandB (RL training)
    if get_env_value('TINKER_API_KEY') and get_env_value('WANDB_API_KEY'):
        tool_status.append(("RL Training (Tinker)", True, None))
    elif get_env_value('TINKER_API_KEY'):
        tool_status.append(("RL Training (Tinker)", False, "WANDB_API_KEY"))
    else:
        tool_status.append(("RL Training (Tinker)", False, "TINKER_API_KEY"))
    
    # Home Assistant
    if get_env_value('HASS_TOKEN'):
        tool_status.append(("Smart Home (Home Assistant)", True, None))
    
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
        print_warning("Some tools are disabled. Run 'hermes setup tools' to configure them,")
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
    print(f"   {color('hermes setup', Colors.GREEN)}          Re-run the full wizard")
    print(f"   {color('hermes setup model', Colors.GREEN)}    Change model/provider")
    print(f"   {color('hermes setup terminal', Colors.GREEN)} Change terminal backend")
    print(f"   {color('hermes setup gateway', Colors.GREEN)}  Configure messaging")
    print(f"   {color('hermes setup tools', Colors.GREEN)}    Configure tool providers")
    print()
    print(f"   {color('hermes config', Colors.GREEN)}         View current settings")
    print(f"   {color('hermes config edit', Colors.GREEN)}    Open config in your editor")
    print(f"   {color('hermes config set KEY VALUE', Colors.GREEN)}")
    print(f"                          Set a specific value")
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


def _prompt_container_resources(config: dict):
    """Prompt for container resource settings (Docker, Singularity, Modal, Daytona)."""
    terminal = config.setdefault('terminal', {})

    print()
    print_info("Container Resource Settings:")

    # Persistence
    current_persist = terminal.get('container_persistent', True)
    persist_label = "yes" if current_persist else "no"
    print_info("  Persistent filesystem keeps files between sessions.")
    print_info("  Set to 'no' for ephemeral sandboxes that reset each time.")
    persist_str = prompt(f"  Persist filesystem across sessions? (yes/no)", persist_label)
    terminal['container_persistent'] = persist_str.lower() in ('yes', 'true', 'y', '1')

    # CPU
    current_cpu = terminal.get('container_cpu', 1)
    cpu_str = prompt(f"  CPU cores", str(current_cpu))
    try:
        terminal['container_cpu'] = float(cpu_str)
    except ValueError:
        pass

    # Memory
    current_mem = terminal.get('container_memory', 5120)
    mem_str = prompt(f"  Memory in MB (5120 = 5GB)", str(current_mem))
    try:
        terminal['container_memory'] = int(mem_str)
    except ValueError:
        pass

    # Disk
    current_disk = terminal.get('container_disk', 51200)
    disk_str = prompt(f"  Disk in MB (51200 = 50GB)", str(current_disk))
    try:
        terminal['container_disk'] = int(disk_str)
    except ValueError:
        pass


# =============================================================================
# Tool Categories — category-first UX for tool configuration
# =============================================================================
# Each category represents a tool type. Within each category, users choose
# a provider. This avoids showing "OpenAI Voice" and "ElevenLabs" as separate
# tools — instead they see "Text-to-Speech" then pick a provider.

TOOL_CATEGORIES = [
    {
        "name": "Text-to-Speech",
        "icon": "🎤",
        "description": "Convert text to voice messages",
        "providers": [
            {
                "name": "Microsoft Edge TTS",
                "tag": "Free - no API key needed",
                "env_vars": [],
                "tts_provider": "edge",
            },
            {
                "name": "OpenAI TTS",
                "tag": "Premium - high quality voices",
                "env_vars": [
                    {"key": "VOICE_TOOLS_OPENAI_KEY", "prompt": "OpenAI API key", "url": "https://platform.openai.com/api-keys"},
                ],
                "tts_provider": "openai",
            },
            {
                "name": "ElevenLabs",
                "tag": "Premium - most natural voices",
                "env_vars": [
                    {"key": "ELEVENLABS_API_KEY", "prompt": "ElevenLabs API key", "url": "https://elevenlabs.io/app/settings/api-keys"},
                ],
                "tts_provider": "elevenlabs",
            },
        ],
    },
    {
        "name": "Web Search & Extract",
        "icon": "🔍",
        "description": "Search the web and extract content from URLs",
        "providers": [
            {
                "name": "Firecrawl Cloud",
                "tag": "Recommended - hosted service",
                "env_vars": [
                    {"key": "FIRECRAWL_API_KEY", "prompt": "Firecrawl API key", "url": "https://firecrawl.dev"},
                ],
            },
            {
                "name": "Firecrawl Self-Hosted",
                "tag": "Free - run your own instance",
                "env_vars": [
                    {"key": "FIRECRAWL_API_URL", "prompt": "Your Firecrawl instance URL (e.g., http://localhost:3002)"},
                ],
            },
        ],
    },
    {
        "name": "Image Generation",
        "icon": "🎨",
        "description": "Generate images from text prompts (FLUX 2 Pro + upscaling)",
        "providers": [
            {
                "name": "FAL.ai",
                "tag": "FLUX 2 Pro with auto-upscaling",
                "env_vars": [
                    {"key": "FAL_KEY", "prompt": "FAL API key", "url": "https://fal.ai/dashboard/keys"},
                ],
            },
        ],
    },
    {
        "name": "Browser Automation",
        "icon": "🌐",
        "description": "Control a cloud browser for web interactions",
        "providers": [
            {
                "name": "Browserbase",
                "tag": "Cloud browser with stealth mode",
                "env_vars": [
                    {"key": "BROWSERBASE_API_KEY", "prompt": "Browserbase API key", "url": "https://browserbase.com"},
                    {"key": "BROWSERBASE_PROJECT_ID", "prompt": "Browserbase project ID"},
                ],
                "post_setup": "browserbase",
            },
        ],
    },
    {
        "name": "Smart Home",
        "icon": "🏠",
        "description": "Control Home Assistant lights, switches, and devices",
        "providers": [
            {
                "name": "Home Assistant",
                "tag": "REST API integration",
                "env_vars": [
                    {"key": "HASS_TOKEN", "prompt": "Home Assistant Long-Lived Access Token"},
                    {"key": "HASS_URL", "prompt": "Home Assistant URL", "default": "http://homeassistant.local:8123"},
                ],
            },
        ],
    },
    {
        "name": "RL Training",
        "icon": "🧪",
        "description": "Run reinforcement learning training jobs",
        "requires_python": (3, 11),
        "providers": [
            {
                "name": "Tinker / Atropos",
                "tag": "RL training platform",
                "env_vars": [
                    {"key": "TINKER_API_KEY", "prompt": "Tinker API key", "url": "https://tinker-console.thinkingmachines.ai/keys"},
                    {"key": "WANDB_API_KEY", "prompt": "WandB API key", "url": "https://wandb.ai/authorize"},
                ],
                "post_setup": "rl_training",
            },
        ],
    },
    {
        "name": "GitHub Integration",
        "icon": "🔧",
        "description": "Higher rate limits for Skills Hub + PR publishing",
        "providers": [
            {
                "name": "GitHub Personal Access Token",
                "tag": "For skill search, install, and publishing",
                "env_vars": [
                    {"key": "GITHUB_TOKEN", "prompt": "GitHub Token (ghp_...)", "url": "https://github.com/settings/tokens"},
                ],
            },
        ],
    },
]


def _run_post_setup(post_setup_key: str):
    """Run post-setup hooks for tools that need extra installation steps."""
    if post_setup_key == "browserbase":
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

    elif post_setup_key == "rl_training":
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


# =============================================================================
# Section 1: Model & Provider Configuration
# =============================================================================

def setup_model_provider(config: dict):
    """Configure the inference provider and default model."""
    from hermes_cli.auth import (
        get_active_provider, get_provider_auth_state, PROVIDER_REGISTRY,
        format_auth_error, AuthError, fetch_nous_models,
        resolve_nous_runtime_credentials, _update_config_for_provider,
        _login_openai_codex, get_codex_auth_status, DEFAULT_CODEX_BASE_URL,
        detect_external_credentials,
    )

    print_header("Inference Provider")
    print_info("Choose how to connect to your main chat model.")
    print()

    existing_or = get_env_value("OPENROUTER_API_KEY")
    active_oauth = get_active_provider()
    existing_custom = get_env_value("OPENAI_BASE_URL")

    # Detect credentials from other CLI tools
    detected_creds = detect_external_credentials()
    if detected_creds:
        print_info("Detected existing credentials:")
        for cred in detected_creds:
            if cred["provider"] == "openai-codex":
                print_success(f"  * {cred['label']} -- select \"OpenAI Codex\" to use it")
            else:
                print_info(f"  * {cred['label']}")
        print()

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
        "Login with OpenAI Codex",
        "OpenRouter API key (100+ models, pay-per-use)",
        "Custom OpenAI-compatible endpoint (self-hosted / VLLM / etc.)",
        "Z.AI / GLM (Zhipu AI models)",
        "Kimi / Moonshot (Kimi coding models)",
        "MiniMax (global endpoint)",
        "MiniMax China (mainland China endpoint)",
    ]
    if keep_label:
        provider_choices.append(keep_label)
    
    # Default to "Keep current" if a provider exists, otherwise OpenRouter (most common)
    default_provider = len(provider_choices) - 1 if has_any_provider else 2
    
    if not has_any_provider:
        print_warning("An inference provider is required for Hermes to work.")
        print()
    
    provider_idx = prompt_choice("Select your inference provider:", provider_choices, default_provider)

    # Track which provider was selected for model step
    selected_provider = None  # "nous", "openai-codex", "openrouter", "custom", or None (keep)
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
            print_info("You can try again later with: hermes model")
            selected_provider = None
        except Exception as e:
            print_error(f"Login failed: {e}")
            print_info("You can try again later with: hermes model")
            selected_provider = None

    elif provider_idx == 1:  # OpenAI Codex
        selected_provider = "openai-codex"
        print()
        print_header("OpenAI Codex Login")
        print()

        try:
            import argparse
            mock_args = argparse.Namespace()
            _login_openai_codex(mock_args, PROVIDER_REGISTRY["openai-codex"])
            # Clear custom endpoint vars that would override provider routing.
            if existing_custom:
                save_env_value("OPENAI_BASE_URL", "")
                save_env_value("OPENAI_API_KEY", "")
            _update_config_for_provider("openai-codex", DEFAULT_CODEX_BASE_URL)
        except SystemExit:
            print_warning("OpenAI Codex login was cancelled or failed.")
            print_info("You can try again later with: hermes model")
            selected_provider = None
        except Exception as e:
            print_error(f"Login failed: {e}")
            print_info("You can try again later with: hermes model")
            selected_provider = None

    elif provider_idx == 2:  # OpenRouter
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

    elif provider_idx == 3:  # Custom endpoint
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

    elif provider_idx == 4:  # Z.AI / GLM
        selected_provider = "zai"
        print()
        print_header("Z.AI / GLM API Key")
        pconfig = PROVIDER_REGISTRY["zai"]
        print_info(f"Provider: {pconfig.name}")
        print_info(f"Base URL: {pconfig.inference_base_url}")
        print_info("Get your API key at: https://open.bigmodel.cn/")
        print()

        existing_key = get_env_value("GLM_API_KEY") or get_env_value("ZAI_API_KEY")
        if existing_key:
            print_info(f"Current: {existing_key[:8]}... (configured)")
            if prompt_yes_no("Update API key?", False):
                api_key = prompt("  GLM API key", password=True)
                if api_key:
                    save_env_value("GLM_API_KEY", api_key)
                    print_success("GLM API key updated")
        else:
            api_key = prompt("  GLM API key", password=True)
            if api_key:
                save_env_value("GLM_API_KEY", api_key)
                print_success("GLM API key saved")
            else:
                print_warning("Skipped - agent won't work without an API key")

        # Clear custom endpoint vars if switching
        if existing_custom:
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
        _update_config_for_provider("zai", pconfig.inference_base_url)

    elif provider_idx == 5:  # Kimi / Moonshot
        selected_provider = "kimi-coding"
        print()
        print_header("Kimi / Moonshot API Key")
        pconfig = PROVIDER_REGISTRY["kimi-coding"]
        print_info(f"Provider: {pconfig.name}")
        print_info(f"Base URL: {pconfig.inference_base_url}")
        print_info("Get your API key at: https://platform.moonshot.cn/")
        print()

        existing_key = get_env_value("KIMI_API_KEY")
        if existing_key:
            print_info(f"Current: {existing_key[:8]}... (configured)")
            if prompt_yes_no("Update API key?", False):
                api_key = prompt("  Kimi API key", password=True)
                if api_key:
                    save_env_value("KIMI_API_KEY", api_key)
                    print_success("Kimi API key updated")
        else:
            api_key = prompt("  Kimi API key", password=True)
            if api_key:
                save_env_value("KIMI_API_KEY", api_key)
                print_success("Kimi API key saved")
            else:
                print_warning("Skipped - agent won't work without an API key")

        # Clear custom endpoint vars if switching
        if existing_custom:
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
        _update_config_for_provider("kimi-coding", pconfig.inference_base_url)

    elif provider_idx == 6:  # MiniMax
        selected_provider = "minimax"
        print()
        print_header("MiniMax API Key")
        pconfig = PROVIDER_REGISTRY["minimax"]
        print_info(f"Provider: {pconfig.name}")
        print_info(f"Base URL: {pconfig.inference_base_url}")
        print_info("Get your API key at: https://platform.minimaxi.com/")
        print()

        existing_key = get_env_value("MINIMAX_API_KEY")
        if existing_key:
            print_info(f"Current: {existing_key[:8]}... (configured)")
            if prompt_yes_no("Update API key?", False):
                api_key = prompt("  MiniMax API key", password=True)
                if api_key:
                    save_env_value("MINIMAX_API_KEY", api_key)
                    print_success("MiniMax API key updated")
        else:
            api_key = prompt("  MiniMax API key", password=True)
            if api_key:
                save_env_value("MINIMAX_API_KEY", api_key)
                print_success("MiniMax API key saved")
            else:
                print_warning("Skipped - agent won't work without an API key")

        # Clear custom endpoint vars if switching
        if existing_custom:
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
        _update_config_for_provider("minimax", pconfig.inference_base_url)

    elif provider_idx == 7:  # MiniMax China
        selected_provider = "minimax-cn"
        print()
        print_header("MiniMax China API Key")
        pconfig = PROVIDER_REGISTRY["minimax-cn"]
        print_info(f"Provider: {pconfig.name}")
        print_info(f"Base URL: {pconfig.inference_base_url}")
        print_info("Get your API key at: https://platform.minimaxi.com/")
        print()

        existing_key = get_env_value("MINIMAX_CN_API_KEY")
        if existing_key:
            print_info(f"Current: {existing_key[:8]}... (configured)")
            if prompt_yes_no("Update API key?", False):
                api_key = prompt("  MiniMax CN API key", password=True)
                if api_key:
                    save_env_value("MINIMAX_CN_API_KEY", api_key)
                    print_success("MiniMax CN API key updated")
        else:
            api_key = prompt("  MiniMax CN API key", password=True)
            if api_key:
                save_env_value("MINIMAX_CN_API_KEY", api_key)
                print_success("MiniMax CN API key saved")
            else:
                print_warning("Skipped - agent won't work without an API key")

        # Clear custom endpoint vars if switching
        if existing_custom:
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
        _update_config_for_provider("minimax-cn", pconfig.inference_base_url)

    # else: provider_idx == 8 (Keep current) — only shown when a provider already exists

    # ── OpenRouter API Key for tools (if not already set) ──
    # Tools (vision, web, MoA) use OpenRouter independently of the main provider.
    # Prompt for OpenRouter key if not set and a non-OpenRouter provider was chosen.
    if selected_provider in ("nous", "openai-codex", "custom", "zai", "kimi-coding", "minimax", "minimax-cn") and not get_env_value("OPENROUTER_API_KEY"):
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

    # ── Model Selection (adapts based on provider) ──
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
            elif model_idx == len(model_choices) - 2:  # Custom
                model_name = prompt("  Model name")
                if model_name:
                    config['model'] = model_name
            # else: keep current

        elif selected_provider == "openai-codex":
            from hermes_cli.codex_models import get_codex_models
            codex_models = get_codex_models()
            model_choices = codex_models + [f"Keep current ({current_model})"]
            default_codex = 0
            if current_model in codex_models:
                default_codex = codex_models.index(current_model)
            elif current_model:
                default_codex = len(model_choices) - 1

            model_idx = prompt_choice("Select default model:", model_choices, default_codex)
            if model_idx < len(codex_models):
                config['model'] = codex_models[model_idx]
                save_env_value("LLM_MODEL", codex_models[model_idx])
            elif model_idx == len(codex_models):
                custom = prompt("Enter model name")
                if custom:
                    config['model'] = custom
                    save_env_value("LLM_MODEL", custom)
            _update_config_for_provider("openai-codex", DEFAULT_CODEX_BASE_URL)
        elif selected_provider == "zai":
            zai_models = ["glm-5", "glm-4.7", "glm-4.5", "glm-4.5-flash"]
            model_choices = list(zai_models)
            model_choices.append("Custom model")
            model_choices.append(f"Keep current ({current_model})")

            keep_idx = len(model_choices) - 1
            model_idx = prompt_choice("Select default model:", model_choices, keep_idx)

            if model_idx < len(zai_models):
                config['model'] = zai_models[model_idx]
                save_env_value("LLM_MODEL", zai_models[model_idx])
            elif model_idx == len(zai_models):
                custom = prompt("Enter model name")
                if custom:
                    config['model'] = custom
                    save_env_value("LLM_MODEL", custom)
            # else: keep current
        elif selected_provider == "kimi-coding":
            kimi_models = ["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview"]
            model_choices = list(kimi_models)
            model_choices.append("Custom model")
            model_choices.append(f"Keep current ({current_model})")

            keep_idx = len(model_choices) - 1
            model_idx = prompt_choice("Select default model:", model_choices, keep_idx)

            if model_idx < len(kimi_models):
                config['model'] = kimi_models[model_idx]
                save_env_value("LLM_MODEL", kimi_models[model_idx])
            elif model_idx == len(kimi_models):
                custom = prompt("Enter model name")
                if custom:
                    config['model'] = custom
                    save_env_value("LLM_MODEL", custom)
            # else: keep current
        elif selected_provider in ("minimax", "minimax-cn"):
            minimax_models = ["MiniMax-M2.5", "MiniMax-M2.5-highspeed", "MiniMax-M2.1"]
            model_choices = list(minimax_models)
            model_choices.append("Custom model")
            model_choices.append(f"Keep current ({current_model})")

            keep_idx = len(model_choices) - 1
            model_idx = prompt_choice("Select default model:", model_choices, keep_idx)

            if model_idx < len(minimax_models):
                config['model'] = minimax_models[model_idx]
                save_env_value("LLM_MODEL", minimax_models[model_idx])
            elif model_idx == len(minimax_models):
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

        if config.get('model'):
            print_success(f"Model set to: {config['model']}")

    save_config(config)


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
    print()

    current_backend = config.get('terminal', {}).get('backend', 'local')
    is_linux = _platform.system() == "Linux"

    # Build backend choices with descriptions
    terminal_choices = [
        "Local - run directly on this machine (default)",
        "Docker - isolated container with configurable resources",
    ]
    idx_to_backend = {0: "local", 1: "docker"}
    backend_to_idx = {"local": 0, "docker": 1}

    next_idx = 2
    if is_linux:
        terminal_choices.append("Singularity/Apptainer - HPC-friendly container")
        idx_to_backend[next_idx] = "singularity"
        backend_to_idx["singularity"] = next_idx
        next_idx += 1

    terminal_choices.append("Modal - serverless cloud sandbox")
    idx_to_backend[next_idx] = "modal"
    backend_to_idx["modal"] = next_idx
    next_idx += 1

    terminal_choices.append("Daytona - persistent cloud development environment")
    idx_to_backend[next_idx] = "daytona"
    backend_to_idx["daytona"] = next_idx
    next_idx += 1

    terminal_choices.append("SSH - run on a remote machine")
    idx_to_backend[next_idx] = "ssh"
    backend_to_idx["ssh"] = next_idx
    next_idx += 1

    # Add keep current option
    keep_current_idx = next_idx
    terminal_choices.append(f"Keep current ({current_backend})")
    idx_to_backend[keep_current_idx] = current_backend

    default_terminal = backend_to_idx.get(current_backend, 0)

    terminal_idx = prompt_choice("Select terminal backend:", terminal_choices, keep_current_idx)

    selected_backend = idx_to_backend.get(terminal_idx)

    if terminal_idx == keep_current_idx:
        print_info(f"Keeping current backend: {current_backend}")
        return

    config.setdefault('terminal', {})['backend'] = selected_backend

    if selected_backend == "local":
        print_success("Terminal backend: Local")
        print_info("Commands run directly on this machine.")
        
        # CWD for messaging
        print()
        print_info("Working directory for messaging sessions:")
        print_info("  When using Hermes via Telegram/Discord, this is where")
        print_info("  the agent starts. CLI mode always starts in the current directory.")
        current_cwd = config.get('terminal', {}).get('cwd', '')
        cwd = prompt("  Messaging working directory", current_cwd or str(Path.home()))
        if cwd:
            config['terminal']['cwd'] = cwd

        # Sudo support
        print()
        existing_sudo = get_env_value("SUDO_PASSWORD")
        if existing_sudo:
            print_info("Sudo password: configured")
        else:
            if prompt_yes_no("Enable sudo support? (stores password for apt install, etc.)", False):
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
        current_image = config.get('terminal', {}).get('docker_image', 'python:3.11-slim')
        image = prompt("  Docker image", current_image)
        config['terminal']['docker_image'] = image
        save_env_value("TERMINAL_DOCKER_IMAGE", image)

        _prompt_container_resources(config)

    elif selected_backend == "singularity":
        print_success("Terminal backend: Singularity/Apptainer")

        # Check if singularity/apptainer is available
        sing_bin = shutil.which("apptainer") or shutil.which("singularity")
        if not sing_bin:
            print_warning("Singularity/Apptainer not found in PATH!")
            print_info("Install: https://apptainer.org/docs/admin/main/installation.html")
        else:
            print_info(f"Found: {sing_bin}")

        current_image = config.get('terminal', {}).get('singularity_image', 'docker://python:3.11-slim')
        image = prompt("  Container image", current_image)
        config['terminal']['singularity_image'] = image
        save_env_value("TERMINAL_SINGULARITY_IMAGE", image)

        _prompt_container_resources(config)

    elif selected_backend == "modal":
        print_success("Terminal backend: Modal")
        print_info("Serverless cloud sandboxes. Each session gets its own container.")
        print_info("Requires a Modal account: https://modal.com")

        # Check if swe-rex[modal] is installed
        try:
            __import__("swe_rex")
        except ImportError:
            print_info("Installing swe-rex[modal]...")
            import subprocess
            uv_bin = shutil.which("uv")
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "swe-rex[modal]"],
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "swe-rex[modal]"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                print_success("swe-rex[modal] installed")
            else:
                print_warning("Install failed — run manually: pip install 'swe-rex[modal]'")

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
                    [uv_bin, "pip", "install", "daytona"],
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "daytona"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                print_success("daytona SDK installed")
            else:
                print_warning("Install failed — run manually: pip install daytona")

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
        current_image = config.get('terminal', {}).get('daytona_image', 'nikolaik/python-nodejs:python3.11-nodejs20')
        image = prompt("  Sandbox image", current_image)
        config['terminal']['daytona_image'] = image
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
    save_config(config)
    print()
    print_success(f"Terminal backend set to: {selected_backend}")


# =============================================================================
# Section 3: Agent Settings
# =============================================================================

def setup_agent_settings(config: dict):
    """Configure agent behavior: iterations, progress display, compression, session reset."""

    # ── Max Iterations ──
    print_header("Agent Settings")

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
    print_info("Higher threshold = compress later (use more context). Lower = compress sooner.")
    
    config.setdefault('compression', {})['enabled'] = True
    
    current_threshold = config.get('compression', {}).get('threshold', 0.85)
    threshold_str = prompt("Compression threshold (0.5-0.95)", str(current_threshold))
    try:
        threshold = float(threshold_str)
        if 0.5 <= threshold <= 0.95:
            config['compression']['threshold'] = threshold
    except ValueError:
        pass
    
    print_success(f"Context compression threshold set to {config['compression'].get('threshold', 0.85)}")

    # ── Session Reset Policy ──
    print_header("Session Reset Policy")
    print_info("Messaging sessions (Telegram, Discord, etc.) accumulate context over time.")
    print_info("Each message adds to the conversation history, which means growing API costs.")
    print_info("")
    print_info("To manage this, sessions can automatically reset after a period of inactivity")
    print_info("or at a fixed time each day. When a reset happens, the agent saves important")
    print_info("things to its persistent memory first — but the conversation context is cleared.")
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
    
    current_policy = config.get('session_reset', {})
    current_mode = current_policy.get('mode', 'both')
    current_idle = current_policy.get('idle_minutes', 1440)
    current_hour = current_policy.get('at_hour', 4)
    
    default_reset = {"both": 0, "idle": 1, "daily": 2, "none": 3}.get(current_mode, 0)
    
    reset_idx = prompt_choice("Session reset mode:", reset_choices, default_reset)
    
    config.setdefault('session_reset', {})
    
    if reset_idx == 0:  # Both
        config['session_reset']['mode'] = 'both'
        idle_str = prompt("  Inactivity timeout (minutes)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config['session_reset']['idle_minutes'] = idle_val
        except ValueError:
            pass
        hour_str = prompt("  Daily reset hour (0-23, local time)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config['session_reset']['at_hour'] = hour_val
        except ValueError:
            pass
        print_success(f"Sessions reset after {config['session_reset'].get('idle_minutes', 1440)} min idle or daily at {config['session_reset'].get('at_hour', 4)}:00")
    elif reset_idx == 1:  # Idle only
        config['session_reset']['mode'] = 'idle'
        idle_str = prompt("  Inactivity timeout (minutes)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config['session_reset']['idle_minutes'] = idle_val
        except ValueError:
            pass
        print_success(f"Sessions reset after {config['session_reset'].get('idle_minutes', 1440)} min of inactivity")
    elif reset_idx == 2:  # Daily only
        config['session_reset']['mode'] = 'daily'
        hour_str = prompt("  Daily reset hour (0-23, local time)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config['session_reset']['at_hour'] = hour_val
        except ValueError:
            pass
        print_success(f"Sessions reset daily at {config['session_reset'].get('at_hour', 4)}:00")
    elif reset_idx == 3:  # None
        config['session_reset']['mode'] = 'none'
        print_info("Sessions will never auto-reset. Context is managed only by compression.")
        print_warning("Long conversations will grow in cost. Use /reset manually when needed.")
    # else: keep current (idx == 4)
    
    save_config(config)


# =============================================================================
# Section 4: Messaging Platforms (Gateway)
# =============================================================================

def setup_gateway(config: dict):
    """Configure messaging platform integrations."""
    print_header("Messaging Platforms")
    print_info("Connect to messaging platforms to chat with Hermes from anywhere.")
    print()

    # ── Telegram ──
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
    
    # ── Discord ──
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
    
    # ── Slack ──
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
    
    # ── WhatsApp ──
    existing_whatsapp = get_env_value('WHATSAPP_ENABLED')
    if not existing_whatsapp and prompt_yes_no("Set up WhatsApp?", False):
        print_info("WhatsApp connects via a built-in bridge (Baileys).")
        print_info("Requires Node.js. Run 'hermes whatsapp' for guided setup.")
        print()
        if prompt_yes_no("Enable WhatsApp now?", True):
            save_env_value("WHATSAPP_ENABLED", "true")
            print_success("WhatsApp enabled")
            print_info("Run 'hermes whatsapp' to choose your mode (separate bot number")
            print_info("or personal self-chat) and pair via QR code.")
    
    # ── Gateway Service Setup ──
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
            print_warning(f"No home channel set for: {', '.join(missing_home)}")
            print_info("   Without a home channel, cron jobs and cross-platform")
            print_info("   messages can't be delivered to those platforms.")
            print_info("   Set one later with /set-home in your chat, or:")
            for plat in missing_home:
                print_info(f"     hermes config set {plat.upper()}_HOME_CHANNEL <channel_id>")

        # Offer to install the gateway as a system service
        import platform as _platform
        _is_linux = _platform.system() == "Linux"
        _is_macos = _platform.system() == "Darwin"

        from hermes_cli.gateway import (
            _is_service_installed, _is_service_running,
            systemd_install, systemd_start, systemd_restart,
            launchd_install, launchd_start, launchd_restart,
        )

        service_installed = _is_service_installed()
        service_running = _is_service_running()

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
            if prompt_yes_no(f"  Install the gateway as a {svc_name} service? (runs in background, starts on boot)", True):
                try:
                    if _is_linux:
                        systemd_install(force=False)
                    else:
                        launchd_install(force=False)
                    print()
                    if prompt_yes_no("  Start the service now?", True):
                        try:
                            if _is_linux:
                                systemd_start()
                            elif _is_macos:
                                launchd_start()
                        except Exception as e:
                            print_error(f"  Start failed: {e}")
                except Exception as e:
                    print_error(f"  Install failed: {e}")
                    print_info("  You can try manually: hermes gateway install")
            else:
                print_info("  You can install later: hermes gateway install")
                print_info("  Or run in foreground:  hermes gateway")
        else:
            print_info("Start the gateway to bring your bots online:")
            print_info("   hermes gateway              # Run in foreground")

        print_info("━" * 50)


# =============================================================================
# Section 5: Tool Configuration (Category-First UX)
# =============================================================================

def setup_tools(config: dict):
    """Configure tools with a category-first UX.
    
    Instead of showing flat list of API keys, this shows tool categories
    (TTS, Web Search, Image Gen, etc.) and lets users pick a provider
    within each category.
    """
    print_header("Tool Configuration")
    print_info("Select which tools you'd like to enable.")
    print_info("For tools with multiple providers, you'll choose one next.")
    print_info("You can always reconfigure later with 'hermes setup tools'.")
    print()

    # Build checklist from TOOL_CATEGORIES
    # NOTE: Do NOT use color() / ANSI codes in menu labels —
    # simple_term_menu miscalculates widths and causes garbled redraws.
    checklist_labels = []
    for cat in TOOL_CATEGORIES:
        icon = cat.get("icon", "")
        name = cat["name"]
        desc = cat.get("description", "")

        # Check if already configured — plain text only (no ANSI codes)
        configured = _is_tool_configured(cat)
        status = " [configured]" if configured else ""

        checklist_labels.append(f"{icon} {name} - {desc}{status}")

    # Pre-select tools that are already configured
    pre_selected = [i for i, cat in enumerate(TOOL_CATEGORIES) if _is_tool_configured(cat)]

    selected_indices = prompt_checklist(
        "Which tools would you like to enable?",
        checklist_labels,
        pre_selected=pre_selected,
    )

    # For each selected tool, configure its provider
    for idx in selected_indices:
        cat = TOOL_CATEGORIES[idx]
        _configure_tool_category(cat, config)

    save_config(config)
    print()
    print_success("Tool configuration complete!")


def _is_tool_configured(cat: dict) -> bool:
    """Check if a tool category has at least one provider configured."""
    for provider in cat["providers"]:
        env_vars = provider.get("env_vars", [])
        if not env_vars:
            # No env vars needed (e.g., Edge TTS) — check if it's the active provider
            if provider.get("tts_provider"):
                from hermes_cli.config import load_config as _lc
                cfg = _lc()
                if cfg.get("tts", {}).get("provider") == provider["tts_provider"]:
                    return True
            else:
                return True
        elif all(get_env_value(v["key"]) for v in env_vars):
            return True
    return False


def _configure_tool_category(cat: dict, config: dict):
    """Configure a single tool category — pick provider and enter API keys."""
    icon = cat.get("icon", "")
    name = cat["name"]
    providers = cat["providers"]

    # Check Python version requirement
    if cat.get("requires_python"):
        req = cat["requires_python"]
        if sys.version_info < req:
            print()
            print(color(f"  ─── {icon} {name} ───", Colors.CYAN))
            print_error(f"  Requires Python {req[0]}.{req[1]}+ (current: {sys.version_info.major}.{sys.version_info.minor})")
            print_info("  Upgrade Python and reinstall to enable this tool.")
            return

    if len(providers) == 1:
        # Single provider — just configure it directly
        provider = providers[0]
        print()
        print(color(f"  ─── {icon} {name} ({provider['name']}) ───", Colors.CYAN))
        if provider.get("tag"):
            print_info(f"  {provider['tag']}")
        _configure_provider(provider, config, cat)
    else:
        # Multiple providers — let user choose
        print()
        print(color(f"  ─── {icon} {name} — Choose a provider ───", Colors.CYAN))
        print()

        # NOTE: Do NOT use color() / ANSI codes in menu labels —
        # simple_term_menu miscalculates widths and causes garbled redraws.
        provider_choices = []
        for p in providers:
            tag = f" ({p['tag']})" if p.get("tag") else ""
            configured = ""
            env_vars = p.get("env_vars", [])
            if not env_vars or all(get_env_value(v["key"]) for v in env_vars):
                # Check TTS provider match for edge
                if p.get("tts_provider"):
                    if config.get("tts", {}).get("provider") == p["tts_provider"]:
                        configured = " [active]"
                elif not env_vars:
                    configured = " [active]"
                else:
                    configured = " [configured]"
            provider_choices.append(f"{p['name']}{tag}{configured}")

        # Detect current provider as default
        default_provider_idx = 0
        for i, p in enumerate(providers):
            if p.get("tts_provider") and config.get("tts", {}).get("provider") == p["tts_provider"]:
                default_provider_idx = i
                break
            env_vars = p.get("env_vars", [])
            if env_vars and all(get_env_value(v["key"]) for v in env_vars):
                default_provider_idx = i
                break

        provider_idx = prompt_choice("Select provider:", provider_choices, default_provider_idx)
        provider = providers[provider_idx]

        _configure_provider(provider, config, cat)


def _configure_provider(provider: dict, config: dict, cat: dict):
    """Configure a single provider — prompt for API keys and set config values."""
    env_vars = provider.get("env_vars", [])

    # Set TTS provider in config if applicable
    if provider.get("tts_provider"):
        config.setdefault("tts", {})["provider"] = provider["tts_provider"]

    if not env_vars:
        # No API keys needed (e.g., Edge TTS)
        print_success(f"  {provider['name']} — no configuration needed!")
        return

    # Prompt for each required env var
    all_configured = True
    for var in env_vars:
        existing = get_env_value(var["key"])
        if existing:
            print_success(f"  {var['key']}: already configured")
            if prompt_yes_no(f"  Update {var.get('prompt', var['key'])}?", False):
                value = prompt(f"    {var.get('prompt', var['key'])}", password=True)
                if value:
                    save_env_value(var["key"], value)
                    print_success("    Updated")
        else:
            url = var.get("url", "")
            if url:
                print_info(f"  Get yours at: {url}")

            default_val = var.get("default", "")
            if default_val:
                value = prompt(f"    {var.get('prompt', var['key'])}", default_val)
            else:
                value = prompt(f"    {var.get('prompt', var['key'])}", password=True)

            if value:
                save_env_value(var["key"], value)
                print_success(f"    ✓ Saved")
            else:
                print_warning(f"    Skipped")
                all_configured = False

    # Run post-setup hooks if needed
    if provider.get("post_setup") and all_configured:
        _run_post_setup(provider["post_setup"])

    if all_configured:
        print_success(f"  {provider['name']} configured!")


# =============================================================================
# Main Wizard Orchestrator
# =============================================================================

SETUP_SECTIONS = [
    ("model", "Model & Provider", setup_model_provider),
    ("terminal", "Terminal Backend", setup_terminal_backend),
    ("gateway", "Messaging Platforms (Gateway)", setup_gateway),
    ("tools", "Tools", setup_tools),
    ("agent", "Agent Settings", setup_agent_settings),
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
    ensure_hermes_home()
    
    config = load_config()
    hermes_home = get_hermes_home()
    
    # Check if a specific section was requested
    section = getattr(args, 'section', None)
    if section:
        for key, label, func in SETUP_SECTIONS:
            if key == section:
                print()
                print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
                print(color(f"│     ⚕ Hermes Setup — {label:<34s} │", Colors.MAGENTA))
                print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))
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
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│             ⚕ Hermes Agent Setup Wizard                │", Colors.MAGENTA))
    print(color("├─────────────────────────────────────────────────────────┤", Colors.MAGENTA))
    print(color("│  Let's configure your Hermes Agent installation.       │", Colors.MAGENTA))
    print(color("│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))
    
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
            # Individual section
            section_idx = choice - 3
            _, label, func = SETUP_SECTIONS[section_idx]
            func(config)
            save_config(config)
            _print_setup_summary(config, hermes_home)
            return
    else:
        # ── First-Time Setup ──
        print()
        print_info("We'll walk you through:")
        print_info("  1. Model & Provider — choose your AI provider and model")
        print_info("  2. Terminal Backend — where your agent runs commands")
        print_info("  3. Messaging Platforms — connect Telegram, Discord, etc.")
        print_info("  4. Tools — configure TTS, web search, image generation, etc.")
        print_info("  5. Agent Settings — iterations, compression, session reset")
        print()
        print_info("Press Enter to begin, or Ctrl+C to exit.")
        try:
            input(color("  Press Enter to start... ", Colors.YELLOW))
        except (KeyboardInterrupt, EOFError):
            print()
            return

    # ── Full Setup — run all sections ──
    print_header("Configuration Location")
    print_info(f"Config file:  {get_config_path()}")
    print_info(f"Secrets file: {get_env_path()}")
    print_info(f"Data folder:  {hermes_home}")
    print_info(f"Install dir:  {PROJECT_ROOT}")
    print()
    print_info("You can edit these files directly or use 'hermes config edit'")

    # Section 1: Model & Provider
    setup_model_provider(config)

    # Section 2: Terminal Backend
    setup_terminal_backend(config)

    # Section 3: Agent Settings
    setup_agent_settings(config)

    # Section 4: Messaging Platforms
    setup_gateway(config)

    # Section 5: Tools
    setup_tools(config)

    # Save and show summary
    save_config(config)
    _print_setup_summary(config, hermes_home)


def _run_quick_setup(config: dict, hermes_home):
    """Quick setup — only configure items that are missing."""
    from hermes_cli.config import (
        get_missing_env_vars, get_missing_config_fields,
        check_config_version, migrate_config,
    )

    print()
    print_header("Quick Setup — Missing Items Only")

    # Check what's missing
    missing_required = [v for v in get_missing_env_vars(required_only=False) if v.get("is_required")]
    missing_optional = [v for v in get_missing_env_vars(required_only=False) if not v.get("is_required")]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()

    has_anything_missing = missing_required or missing_optional or missing_config or current_ver < latest_ver

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
    missing_messaging = [v for v in missing_optional if v.get("category") == "messaging" and not v.get("advanced")]

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
            {"Telegram": "📱 Telegram", "Discord": "💬 Discord", "Slack": "💼 Slack"}.get(p, p)
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
                    print_success(f"  ✓ Saved")
                else:
                    print_warning(f"  Skipped")
                print()

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
