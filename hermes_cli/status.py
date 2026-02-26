"""
Status command for hermes CLI.

Shows the status of all Hermes Agent components.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from hermes_cli.colors import Colors, color
from hermes_cli.model_profiles import get_active_profile, normalize_model_config
from hermes_cli.provider_registry import (
    get_provider,
    list_provider_ids,
    resolve_provider_api_key,
    resolve_provider_base_url,
)

def check_mark(ok: bool) -> str:
    if ok:
        return color("✓", Colors.GREEN)
    return color("✗", Colors.RED)

def redact_key(key: str) -> str:
    """Redact an API key for display."""
    if not key:
        return "(not set)"
    if len(key) < 12:
        return "***"
    return key[:4] + "..." + key[-4:]


def _format_iso_timestamp(value) -> str:
    """Format ISO timestamps for status output, converting to local timezone."""
    if not value or not isinstance(value, str):
        return "(unknown)"
    from datetime import datetime, timezone
    text = value.strip()
    if not text:
        return "(unknown)"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def show_status(args):
    """Show status of all Hermes Agent components."""
    show_all = getattr(args, 'all', False)
    deep = getattr(args, 'deep', False)
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 ⚕ Hermes Agent Status                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    
    # =========================================================================
    # Environment
    # =========================================================================
    print()
    print(color("◆ Environment", Colors.CYAN, Colors.BOLD))
    print(f"  Project:      {PROJECT_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")
    
    env_path = PROJECT_ROOT / '.env'
    print(f"  .env file:    {check_mark(env_path.exists())} {'exists' if env_path.exists() else 'not found'}")
    
    # =========================================================================
    # Inference Providers
    # =========================================================================
    print()
    print(color("◆ Inference Providers", Colors.CYAN, Colors.BOLD))

    try:
        from hermes_cli.config import load_config

        model_cfg = normalize_model_config(load_config().get("model"))
        active_profile = get_active_profile(model_cfg)
        print(
            f"  Active:       {active_profile.get('name')} "
            f"[{active_profile.get('provider')}/{active_profile.get('model')}]"
        )
    except Exception:
        pass

    api_provider_ids = list_provider_ids(include_oauth=False, include_api_key=True, include_custom=True)
    for provider_id in api_provider_ids:
        meta = get_provider(provider_id)
        if not meta:
            continue

        api_key = resolve_provider_api_key(provider_id) or ""
        base_url = resolve_provider_base_url(provider_id) or ""
        if provider_id == "custom":
            configured = bool(base_url)
        else:
            configured = bool(api_key)

        key_display = api_key if show_all else redact_key(api_key)
        print(f"  {meta.label:<12}  {check_mark(configured)} {provider_id}")
        print(f"    API key:   {key_display}")
        if meta.api_key_env_vars:
            print(f"    Key vars:  {', '.join(meta.api_key_env_vars)}")
        if base_url:
            base_label = meta.base_url_env_var or "(provider default)"
            print(f"    Base URL:  {base_url} ({base_label})")
        else:
            print("    Base URL:  (not set)")

    # =========================================================================
    # Tool API Keys
    # =========================================================================
    print()
    print(color("◆ Tool API Keys", Colors.CYAN, Colors.BOLD))

    keys = {
        "Anthropic": "ANTHROPIC_API_KEY",
        "Firecrawl": "FIRECRAWL_API_KEY",
        "Browserbase": "BROWSERBASE_API_KEY",
        "FAL": "FAL_KEY",
        "Tinker": "TINKER_API_KEY",
        "WandB": "WANDB_API_KEY",
        "ElevenLabs": "ELEVENLABS_API_KEY",
        "GitHub": "GITHUB_TOKEN",
    }

    for name, env_var in keys.items():
        value = os.getenv(env_var, "")
        has_key = bool(value)
        display = value if show_all else redact_key(value)
        print(f"  {name:<12}  {check_mark(has_key)} {display}")

    # =========================================================================
    # Auth Providers (OAuth)
    # =========================================================================
    print()
    print(color("◆ Auth Providers", Colors.CYAN, Colors.BOLD))

    try:
        from hermes_cli.auth import get_nous_auth_status
        nous_status = get_nous_auth_status()
    except Exception:
        nous_status = {}

    nous_logged_in = bool(nous_status.get("logged_in"))
    print(
        f"  {'Nous Portal':<12}  {check_mark(nous_logged_in)} "
        f"{'logged in' if nous_logged_in else 'not logged in (run: hermes login)'}"
    )
    if nous_logged_in:
        portal_url = nous_status.get("portal_base_url") or "(unknown)"
        access_exp = _format_iso_timestamp(nous_status.get("access_expires_at"))
        key_exp = _format_iso_timestamp(nous_status.get("agent_key_expires_at"))
        refresh_label = "yes" if nous_status.get("has_refresh_token") else "no"
        print(f"    Portal URL: {portal_url}")
        print(f"    Access exp: {access_exp}")
        print(f"    Key exp:    {key_exp}")
        print(f"    Refresh:    {refresh_label}")

    # =========================================================================
    # Terminal Configuration
    # =========================================================================
    print()
    print(color("◆ Terminal Backend", Colors.CYAN, Colors.BOLD))
    
    terminal_env = os.getenv("TERMINAL_ENV", "")
    if not terminal_env:
        # Fall back to config file value when env var isn't set
        # (hermes status doesn't go through cli.py's config loading)
        try:
            from hermes_cli.config import load_config
            _cfg = load_config()
            terminal_env = _cfg.get("terminal", {}).get("backend", "local")
        except Exception:
            terminal_env = "local"
    print(f"  Backend:      {terminal_env}")
    
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST", "")
        ssh_user = os.getenv("TERMINAL_SSH_USER", "")
        print(f"  SSH Host:     {ssh_host or '(not set)'}")
        print(f"  SSH User:     {ssh_user or '(not set)'}")
    elif terminal_env == "docker":
        docker_image = os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11-slim")
        print(f"  Docker Image: {docker_image}")
    
    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  Sudo:         {check_mark(bool(sudo_password))} {'enabled' if sudo_password else 'disabled'}")
    
    # =========================================================================
    # Messaging Platforms
    # =========================================================================
    print()
    print(color("◆ Messaging Platforms", Colors.CYAN, Colors.BOLD))
    
    platforms = {
        "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
        "Discord": ("DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"),
        "WhatsApp": ("WHATSAPP_ENABLED", None),
    }
    
    for name, (token_var, home_var) in platforms.items():
        token = os.getenv(token_var, "")
        has_token = bool(token)
        
        home_channel = ""
        if home_var:
            home_channel = os.getenv(home_var, "")
        
        status = "configured" if has_token else "not configured"
        if home_channel:
            status += f" (home: {home_channel})"
        
        print(f"  {name:<12}  {check_mark(has_token)} {status}")
    
    # =========================================================================
    # Gateway Status
    # =========================================================================
    print()
    print(color("◆ Gateway Service", Colors.CYAN, Colors.BOLD))
    
    if sys.platform.startswith('linux'):
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "hermes-gateway"],
            capture_output=True,
            text=True
        )
        is_active = result.stdout.strip() == "active"
        print(f"  Status:       {check_mark(is_active)} {'running' if is_active else 'stopped'}")
        print(f"  Manager:      systemd (user)")
        
    elif sys.platform == 'darwin':
        result = subprocess.run(
            ["launchctl", "list", "ai.hermes.gateway"],
            capture_output=True,
            text=True
        )
        is_loaded = result.returncode == 0
        print(f"  Status:       {check_mark(is_loaded)} {'loaded' if is_loaded else 'not loaded'}")
        print(f"  Manager:      launchd")
    else:
        print(f"  Status:       {color('N/A', Colors.DIM)}")
        print(f"  Manager:      (not supported on this platform)")
    
    # =========================================================================
    # Cron Jobs
    # =========================================================================
    print()
    print(color("◆ Scheduled Jobs", Colors.CYAN, Colors.BOLD))
    
    jobs_file = Path.home() / ".hermes" / "cron" / "jobs.json"
    if jobs_file.exists():
        import json
        try:
            with open(jobs_file) as f:
                data = json.load(f)
                jobs = data.get("jobs", [])
                enabled_jobs = [j for j in jobs if j.get("enabled", True)]
                print(f"  Jobs:         {len(enabled_jobs)} active, {len(jobs)} total")
        except Exception:
            print(f"  Jobs:         (error reading jobs file)")
    else:
        print(f"  Jobs:         0")
    
    # =========================================================================
    # Sessions
    # =========================================================================
    print()
    print(color("◆ Sessions", Colors.CYAN, Colors.BOLD))
    
    sessions_file = Path.home() / ".hermes" / "sessions" / "sessions.json"
    if sessions_file.exists():
        import json
        try:
            with open(sessions_file) as f:
                data = json.load(f)
                print(f"  Active:       {len(data)} session(s)")
        except Exception:
            print(f"  Active:       (error reading sessions file)")
    else:
        print(f"  Active:       0")
    
    # =========================================================================
    # Deep checks
    # =========================================================================
    if deep:
        print()
        print(color("◆ Deep Checks", Colors.CYAN, Colors.BOLD))

        try:
            import httpx
            provider_ids = list_provider_ids(include_oauth=False, include_api_key=True, include_custom=True)
            for provider_id in provider_ids:
                meta = get_provider(provider_id)
                if not meta:
                    continue

                api_key = resolve_provider_api_key(provider_id) or ""
                base_url = resolve_provider_base_url(provider_id) or ""
                if provider_id != "custom" and not api_key:
                    continue
                if not base_url:
                    continue

                endpoint = f"{base_url.rstrip('/')}/chat/completions"
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": "hermes-healthcheck",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                }

                try:
                    response = httpx.post(endpoint, headers=headers, json=payload, timeout=10)
                    status_code = response.status_code
                    ok = status_code in (200, 400, 401, 403, 422, 429)
                    detail = "chat endpoint reachable" if ok else f"HTTP {status_code}"
                    print(f"  {meta.label:<12}  {check_mark(ok)} {detail}")
                except Exception as exc:
                    print(f"  {meta.label:<12}  {check_mark(False)} error: {exc}")
        except Exception as exc:
            print(f"  Provider API checks unavailable: {exc}")
        
        # Check gateway port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 18789))
            sock.close()
            # Port in use = gateway likely running
            port_in_use = result == 0
            # This is informational, not necessarily bad
            print(f"  Port 18789:   {'in use' if port_in_use else 'available'}")
        except OSError:
            pass
    
    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  Run 'hermes doctor' for detailed diagnostics", Colors.DIM))
    print(color("  Run 'hermes setup' to configure", Colors.DIM))
    print()
