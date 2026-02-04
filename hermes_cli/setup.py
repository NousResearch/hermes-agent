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

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Import config helpers
from hermes_cli.config import (
    get_hermes_home, get_config_path, get_env_path,
    load_config, save_config, save_env_value, get_env_value,
    ensure_hermes_home, DEFAULT_CONFIG
)

# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

def color(text: str, *codes) -> str:
    """Apply color codes to text."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + Colors.RESET

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
    
    # Tinker + WandB (RL training)
    if get_env_value('TINKER_API_KEY') and get_env_value('WANDB_API_KEY'):
        tool_status.append(("RL Training (Tinker)", True, None))
    elif get_env_value('TINKER_API_KEY'):
        tool_status.append(("RL Training (Tinker)", False, "WANDB_API_KEY"))
    else:
        tool_status.append(("RL Training (Tinker)", False, "TINKER_API_KEY"))
    
    # Terminal (always available if system deps met)
    tool_status.append(("Terminal/Commands", True, None))
    
    # Skills (always available if skills dir exists)
    tool_status.append(("Skills Knowledge Base", True, None))
    
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
    
    # Check if this is an existing installation with config
    is_existing = get_env_value("OPENROUTER_API_KEY") is not None or get_config_path().exists()
    
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
    print(color("│             🦋 Hermes Agent Setup Wizard                │", Colors.MAGENTA))
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
        
        # Handle missing optional env vars
        if missing_optional:
            print()
            print_header("Optional Tools (Quick Setup)")
            
            for var in missing_optional:
                tools = var.get("tools", [])
                tools_str = f" (enables: {', '.join(tools[:2])})" if tools else ""
                
                if prompt_yes_no(f"Configure {var['name']}{tools_str}?", False):
                    if var.get("url"):
                        print_info(f"  Get key at: {var['url']}")
                    
                    if var.get("password"):
                        value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
                    else:
                        value = prompt(f"  {var.get('prompt', var['name'])}")
                    
                    if value:
                        save_env_value(var["name"], value)
                        print_success(f"  Saved")
        
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
    # Step 1: OpenRouter API Key (Required for tools)
    # =========================================================================
    print_header("OpenRouter API Key (Required)")
    print_info("OpenRouter is used for vision, web scraping, and tool operations")
    print_info("even if you use a custom endpoint for your main agent.")
    print_info("Get your API key at: https://openrouter.ai/keys")
    
    existing_or = get_env_value("OPENROUTER_API_KEY")
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
            print_warning("Skipped - some tools (vision, web scraping) won't work without this")
    
    # =========================================================================
    # Step 2: Main Agent Provider
    # =========================================================================
    print_header("Main Agent Provider")
    print_info("Choose how to connect to your main chat model.")
    
    existing_custom = get_env_value("OPENAI_BASE_URL")
    
    provider_choices = [
        "OpenRouter (use same key for agent - recommended)",
        "Custom OpenAI-compatible endpoint (separate from OpenRouter)",
        f"Keep current" + (f" ({existing_custom})" if existing_custom else " (OpenRouter)")
    ]
    
    provider_idx = prompt_choice("Select your main agent provider:", provider_choices, 2)
    
    if provider_idx == 0:  # OpenRouter for agent too
        # Clear any custom endpoint - will use OpenRouter
        if existing_custom:
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
        print_success("Agent will use OpenRouter")
    
    elif provider_idx == 1:  # Custom endpoint
        print_info("Custom OpenAI-Compatible Endpoint Configuration:")
        print_info("Works with any API that follows OpenAI's chat completions spec")
        
        # Show current values if set
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
        print_success("Custom endpoint configured")
    # else: Keep current (provider_idx == 2)
    
    # =========================================================================
    # Step 3: Model Selection
    # =========================================================================
    print_header("Default Model")
    
    current_model = config.get('model', 'anthropic/claude-sonnet-4')
    print_info(f"Current: {current_model}")
    
    model_choices = [
        "anthropic/claude-sonnet-4.5 (recommended)",
        "anthropic/claude-opus-4.5",
        "openai/gpt-5.2",
        "openai/gpt-5.2-codex",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
        "z-ai/glm-4.7",
        "moonshotai/kimi-k2.5",
        "minimax/minimax-m2.1",
        "Custom model",
        f"Keep current ({current_model})"
    ]
    
    model_idx = prompt_choice("Select default model:", model_choices, 10)  # Default: keep current
    
    model_map = {
        0: "anthropic/claude-sonnet-4.5",
        1: "anthropic/claude-opus-4.5",
        2: "openai/gpt-5.2",
        3: "openai/gpt-5.2-codex",
        4: "google/gemini-3-pro-preview",
        5: "google/gemini-3-flash-preview",
        6: "z-ai/glm-4.7",
        7: "moonshotai/kimi-k2.5",
        8: "minimax/minimax-m2.1",
    }
    
    if model_idx in model_map:
        config['model'] = model_map[model_idx]
    elif model_idx == 9:  # Custom
        custom = prompt("Enter model name (e.g., anthropic/claude-sonnet-4.5)")
        if custom:
            config['model'] = custom
    # else: Keep current (model_idx == 10)
    
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
            
            home_channel = prompt("Home channel ID (optional, for cron delivery)")
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
            allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
            if allowed_users:
                save_env_value("DISCORD_ALLOWED_USERS", allowed_users.replace(" ", ""))
                print_success("Discord allowlist configured")
            else:
                print_info("⚠️  No allowlist set - anyone in servers with your bot can use it!")
            
            home_channel = prompt("Home channel ID (optional, for cron delivery)")
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
    
    # =========================================================================
    # Step 8: Additional Tools (Optional)
    # =========================================================================
    print_header("Additional Tools (Optional)")
    print_info("These tools extend the agent's capabilities.")
    print_info("Without their API keys, the corresponding features will be disabled.")
    print()
    
    # Firecrawl - Web scraping
    print_info("─" * 50)
    print(color("  Web Search & Scraping (Firecrawl)", Colors.CYAN))
    print_info("  Enables: web_search, web_extract tools")
    print_info("  Use case: Search the web, read webpage content")
    if get_env_value('FIRECRAWL_API_KEY'):
        print_success("  Status: Configured ✓")
        if prompt_yes_no("  Update Firecrawl API key?", False):
            api_key = prompt("    API key", password=True)
            if api_key:
                save_env_value("FIRECRAWL_API_KEY", api_key)
                print_success("    Updated")
    else:
        print_warning("  Status: Not configured (tools will be disabled)")
        if prompt_yes_no("  Set up Firecrawl?", False):
            print_info("    Get your API key at: https://firecrawl.dev/")
            api_key = prompt("    API key", password=True)
            if api_key:
                save_env_value("FIRECRAWL_API_KEY", api_key)
                print_success("    Configured ✓")
    print()
    
    # Browserbase - Browser automation
    print_info("─" * 50)
    print(color("  Browser Automation (Browserbase)", Colors.CYAN))
    print_info("  Enables: browser_navigate, browser_click, etc.")
    print_info("  Use case: Interact with web pages, fill forms, screenshots")
    if get_env_value('BROWSERBASE_API_KEY'):
        print_success("  Status: Configured ✓")
        if prompt_yes_no("  Update Browserbase credentials?", False):
            api_key = prompt("    API key", password=True)
            project_id = prompt("    Project ID")
            if api_key:
                save_env_value("BROWSERBASE_API_KEY", api_key)
            if project_id:
                save_env_value("BROWSERBASE_PROJECT_ID", project_id)
            print_success("    Updated")
    else:
        print_warning("  Status: Not configured (tools will be disabled)")
        if prompt_yes_no("  Set up Browserbase?", False):
            print_info("    Get credentials at: https://browserbase.com/")
            api_key = prompt("    API key", password=True)
            project_id = prompt("    Project ID")
            if api_key:
                save_env_value("BROWSERBASE_API_KEY", api_key)
            if project_id:
                save_env_value("BROWSERBASE_PROJECT_ID", project_id)
            print_success("    Configured ✓")
    print()
    
    # FAL - Image generation
    print_info("─" * 50)
    print(color("  Image Generation (FAL)", Colors.CYAN))
    print_info("  Enables: image_generate tool")
    print_info("  Use case: Generate images from text prompts (FLUX)")
    if get_env_value('FAL_KEY'):
        print_success("  Status: Configured ✓")
        if prompt_yes_no("  Update FAL API key?", False):
            api_key = prompt("    API key", password=True)
            if api_key:
                save_env_value("FAL_KEY", api_key)
                print_success("    Updated")
    else:
        print_warning("  Status: Not configured (tool will be disabled)")
        if prompt_yes_no("  Set up FAL?", False):
            print_info("    Get your API key at: https://fal.ai/")
            api_key = prompt("    API key", password=True)
            if api_key:
                save_env_value("FAL_KEY", api_key)
                print_success("    Configured ✓")
    print()
    
    # Tinker + WandB - RL Training
    print_info("─" * 50)
    print(color("  RL Training (Tinker + WandB)", Colors.CYAN))
    print_info("  Enables: rl_start_training, rl_check_status, rl_get_results tools")
    print_info("  Use case: Run reinforcement learning training via Tinker API")
    tinker_configured = get_env_value('TINKER_API_KEY')
    wandb_configured = get_env_value('WANDB_API_KEY')
    
    if tinker_configured and wandb_configured:
        print_success("  Status: Configured ✓")
        if prompt_yes_no("  Update RL training credentials?", False):
            api_key = prompt("    Tinker API key", password=True)
            if api_key:
                save_env_value("TINKER_API_KEY", api_key)
            wandb_key = prompt("    WandB API key", password=True)
            if wandb_key:
                save_env_value("WANDB_API_KEY", wandb_key)
            print_success("    Updated")
    else:
        if tinker_configured:
            print_warning("  Status: Tinker configured, WandB missing")
        elif wandb_configured:
            print_warning("  Status: WandB configured, Tinker missing")
        else:
            print_warning("  Status: Not configured (tools will be disabled)")
        
        if prompt_yes_no("  Set up RL Training?", False):
            print_info("    Get Tinker key at: https://tinker-console.thinkingmachines.ai/keys")
            print_info("    Get WandB key at: https://wandb.ai/authorize")
            api_key = prompt("    Tinker API key", password=True)
            if api_key:
                save_env_value("TINKER_API_KEY", api_key)
            wandb_key = prompt("    WandB API key", password=True)
            if wandb_key:
                save_env_value("WANDB_API_KEY", wandb_key)
            if api_key and wandb_key:
                print_success("    Configured ✓")
            else:
                print_warning("    Partially configured (both keys required)")
    
    # =========================================================================
    # Save config and show summary
    # =========================================================================
    save_config(config)
    _print_setup_summary(config, hermes_home)
