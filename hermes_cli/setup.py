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
    """Prompt for a choice from a list."""
    print(color(question, Colors.YELLOW))
    
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


def run_setup_wizard(args):
    """Run the interactive setup wizard."""
    ensure_hermes_home()
    
    config = load_config()
    hermes_home = get_hermes_home()
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│             🦋 Hermes Agent Setup Wizard                │", Colors.MAGENTA))
    print(color("├─────────────────────────────────────────────────────────┤", Colors.MAGENTA))
    print(color("│  Let's configure your Hermes Agent installation.       │", Colors.MAGENTA))
    print(color("│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))
    
    # =========================================================================
    # Step 0: Show paths
    # =========================================================================
    print_header("Configuration Location")
    print_info(f"Config file:  {get_config_path()}")
    print_info(f"Secrets file: {get_env_path()}")
    print_info(f"Data folder:  {hermes_home}")
    print_info(f"Install dir:  {PROJECT_ROOT}")
    print()
    print_info("You can edit these files directly or use 'hermes config edit'")
    
    # =========================================================================
    # Step 1: API Keys
    # =========================================================================
    print_header("Model/Auth Provider")
    
    # Check if already configured
    existing_or = get_env_value("OPENROUTER_API_KEY")
    existing_ant = get_env_value("ANTHROPIC_API_KEY")
    
    if existing_or or existing_ant:
        configured = "OpenRouter" if existing_or else "Anthropic"
        print_info(f"Currently configured: {configured}")
        if not prompt_yes_no("Reconfigure API provider?", False):
            print_info("Keeping existing configuration")
        else:
            existing_or = None  # Force reconfigure
    
    if not existing_or and not existing_ant:
        provider_choices = [
            "OpenRouter (recommended - access to all models)",
            "Anthropic API (direct Claude access)",
            "OpenAI API",
            "Skip for now"
        ]
        
        provider_idx = prompt_choice("Select your primary model provider:", provider_choices, 0)
        
        if provider_idx == 0:  # OpenRouter
            print_info("Get your API key at: https://openrouter.ai/keys")
            api_key = prompt("OpenRouter API key", password=True)
            if api_key:
                save_env_value("OPENROUTER_API_KEY", api_key)
                print_success("OpenRouter API key saved")
        
        elif provider_idx == 1:  # Anthropic
            print_info("Get your API key at: https://console.anthropic.com/")
            api_key = prompt("Anthropic API key", password=True)
            if api_key:
                save_env_value("ANTHROPIC_API_KEY", api_key)
                print_success("Anthropic API key saved")
        
        elif provider_idx == 2:  # OpenAI
            print_info("Get your API key at: https://platform.openai.com/api-keys")
            api_key = prompt("OpenAI API key", password=True)
            if api_key:
                save_env_value("OPENAI_API_KEY", api_key)
                print_success("OpenAI API key saved")
    
    # =========================================================================
    # Step 2: Model Selection
    # =========================================================================
    print_header("Default Model")
    
    current_model = config.get('model', 'anthropic/claude-sonnet-4')
    print_info(f"Current: {current_model}")
    
    model_choices = [
        "anthropic/claude-sonnet-4 (recommended)",
        "anthropic/claude-opus-4",
        "openai/gpt-4o",
        "google/gemini-2.0-flash",
        "Enter custom model",
        "Keep current"
    ]
    
    model_idx = prompt_choice("Select default model:", model_choices, 5)  # Default: keep current
    
    if model_idx == 0:
        config['model'] = "anthropic/claude-sonnet-4"
    elif model_idx == 1:
        config['model'] = "anthropic/claude-opus-4"
    elif model_idx == 2:
        config['model'] = "openai/gpt-4o"
    elif model_idx == 3:
        config['model'] = "google/gemini-2.0-flash"
    elif model_idx == 4:
        custom = prompt("Enter model name (e.g., anthropic/claude-sonnet-4)")
        if custom:
            config['model'] = custom
    
    # =========================================================================
    # Step 3: Terminal Backend
    # =========================================================================
    print_header("Terminal Backend")
    print_info("The terminal tool allows the agent to run commands.")
    
    current_backend = config.get('terminal', {}).get('backend', 'local')
    print_info(f"Current: {current_backend}")
    
    terminal_choices = [
        "Local (run commands on this machine - no isolation)",
        "Docker (isolated containers - recommended for security)",
        "SSH (run commands on a remote server)",
        "Keep current"
    ]
    
    # Default based on current
    default_terminal = {'local': 0, 'docker': 1, 'ssh': 2}.get(current_backend, 0)
    
    terminal_idx = prompt_choice("Select terminal backend:", terminal_choices, 3)  # Default: keep
    
    if terminal_idx == 0:  # Local
        config.setdefault('terminal', {})['backend'] = 'local'
        print_success("Terminal set to local")
        
        if prompt_yes_no("Enable sudo support? (allows agent to run sudo commands)", False):
            print_warning("SECURITY WARNING: Sudo password will be stored in plaintext")
            sudo_pass = prompt("Sudo password (leave empty to skip)", password=True)
            if sudo_pass:
                save_env_value("SUDO_PASSWORD", sudo_pass)
                print_success("Sudo password saved")
    
    elif terminal_idx == 1:  # Docker
        config.setdefault('terminal', {})['backend'] = 'docker'
        docker_image = prompt("Docker image", config.get('terminal', {}).get('docker_image', 'python:3.11-slim'))
        config['terminal']['docker_image'] = docker_image
        print_success("Terminal set to Docker")
    
    elif terminal_idx == 2:  # SSH
        config.setdefault('terminal', {})['backend'] = 'ssh'
        
        current_host = get_env_value('TERMINAL_SSH_HOST') or ''
        current_user = get_env_value('TERMINAL_SSH_USER') or os.getenv("USER", "")
        
        ssh_host = prompt("SSH host", current_host)
        ssh_user = prompt("SSH user", current_user)
        ssh_key = prompt("SSH key path", "~/.ssh/id_rsa")
        
        if ssh_host:
            save_env_value("TERMINAL_SSH_HOST", ssh_host)
        if ssh_user:
            save_env_value("TERMINAL_SSH_USER", ssh_user)
        if ssh_key:
            save_env_value("TERMINAL_SSH_KEY", ssh_key)
        
        print_success("Terminal set to SSH")
    
    # =========================================================================
    # Step 4: Context Compression
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
    # Step 5: Messaging Platforms (Optional)
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
            
            home_channel = prompt("Home channel ID (optional, for cron delivery)")
            if home_channel:
                save_env_value("TELEGRAM_HOME_CHANNEL", home_channel)
    
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
            
            home_channel = prompt("Home channel ID (optional, for cron delivery)")
            if home_channel:
                save_env_value("DISCORD_HOME_CHANNEL", home_channel)
    
    # =========================================================================
    # Step 6: Additional Tools (Optional)
    # =========================================================================
    print_header("Additional Tools (Optional)")
    
    # Firecrawl
    if not get_env_value('FIRECRAWL_API_KEY'):
        if prompt_yes_no("Set up web scraping (Firecrawl)?", False):
            print_info("Get your API key at: https://firecrawl.dev/")
            api_key = prompt("Firecrawl API key", password=True)
            if api_key:
                save_env_value("FIRECRAWL_API_KEY", api_key)
                print_success("Firecrawl API key saved")
    else:
        print_info("Firecrawl: already configured")
    
    # Browserbase
    if not get_env_value('BROWSERBASE_API_KEY'):
        if prompt_yes_no("Set up browser automation (Browserbase)?", False):
            print_info("Get your API key at: https://browserbase.com/")
            api_key = prompt("Browserbase API key", password=True)
            project_id = prompt("Browserbase project ID")
            if api_key:
                save_env_value("BROWSERBASE_API_KEY", api_key)
            if project_id:
                save_env_value("BROWSERBASE_PROJECT_ID", project_id)
            print_success("Browserbase configured")
    else:
        print_info("Browserbase: already configured")
    
    # FAL
    if not get_env_value('FAL_KEY'):
        if prompt_yes_no("Set up image generation (FAL)?", False):
            print_info("Get your API key at: https://fal.ai/")
            api_key = prompt("FAL API key", password=True)
            if api_key:
                save_env_value("FAL_KEY", api_key)
                print_success("FAL API key saved")
    else:
        print_info("FAL: already configured")
    
    # =========================================================================
    # Save config
    # =========================================================================
    save_config(config)
    
    # =========================================================================
    # Done!
    # =========================================================================
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.GREEN))
    print(color("│              ✓ Setup Complete!                          │", Colors.GREEN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.GREEN))
    print()
    
    # Show file locations prominently
    print(color("📁 Your configuration files:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('Settings:', Colors.YELLOW)}  {get_config_path()}")
    print(f"              Model, terminal backend, compression, etc.")
    print()
    print(f"   {color('API Keys:', Colors.YELLOW)}  {get_env_path()}")
    print(f"              OpenRouter, Anthropic, Firecrawl, etc.")
    print()
    print(f"   {color('Data:', Colors.YELLOW)}      {hermes_home}/")
    print(f"              Cron jobs, sessions, logs")
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
