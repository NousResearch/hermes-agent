"""
Doctor command for hermes CLI.

Diagnoses issues with Hermes Agent setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

from hermes_cli.config import get_project_root, get_hermes_home, get_env_path

PROJECT_ROOT = get_project_root()
HERMES_HOME = get_hermes_home()

# Load environment variables from ~/.hermes/.env so API key checks work
from dotenv import load_dotenv
_env_path = get_env_path()
if _env_path.exists():
    load_dotenv(_env_path)
# Also try project .env as fallback
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"

def color(text: str, *codes) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + Colors.RESET

def check_ok(text: str, detail: str = ""):
    print(f"  {color('✓', Colors.GREEN)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_warn(text: str, detail: str = ""):
    print(f"  {color('⚠', Colors.YELLOW)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_fail(text: str, detail: str = ""):
    print(f"  {color('✗', Colors.RED)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_info(text: str):
    print(f"    {color('→', Colors.CYAN)} {text}")


def run_doctor(args):
    """Run diagnostic checks."""
    should_fix = getattr(args, 'fix', False)
    
    issues = []
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 🩺 Hermes Doctor                        │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    
    # =========================================================================
    # Check: Python version
    # =========================================================================
    print()
    print(color("◆ Python Environment", Colors.CYAN, Colors.BOLD))
    
    py_version = sys.version_info
    if py_version >= (3, 11):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    elif py_version >= (3, 10):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        check_warn("Python 3.11+ recommended for RL Training tools (tinker requires >= 3.11)")
    elif py_version >= (3, 8):
        check_warn(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}", "(3.10+ recommended)")
    else:
        check_fail(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}", "(3.10+ required)")
        issues.append("Upgrade Python to 3.10+")
    
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        check_ok("Virtual environment active")
    else:
        check_warn("Not in virtual environment", "(recommended)")
    
    # =========================================================================
    # Check: Required packages
    # =========================================================================
    print()
    print(color("◆ Required Packages", Colors.CYAN, Colors.BOLD))
    
    required_packages = [
        ("openai", "OpenAI SDK"),
        ("rich", "Rich (terminal UI)"),
        ("dotenv", "python-dotenv"),
        ("yaml", "PyYAML"),
        ("httpx", "HTTPX"),
    ]
    
    optional_packages = [
        ("croniter", "Croniter (cron expressions)"),
        ("telegram", "python-telegram-bot"),
        ("discord", "discord.py"),
    ]
    
    for module, name in required_packages:
        try:
            __import__(module)
            check_ok(name)
        except ImportError:
            check_fail(name, "(missing)")
            issues.append(f"Install {name}: uv pip install {module}")
    
    for module, name in optional_packages:
        try:
            __import__(module)
            check_ok(name, "(optional)")
        except ImportError:
            check_warn(name, "(optional, not installed)")
    
    # =========================================================================
    # Check: Configuration files
    # =========================================================================
    print()
    print(color("◆ Configuration Files", Colors.CYAN, Colors.BOLD))
    
    # Check ~/.hermes/.env (primary location for user config)
    env_path = HERMES_HOME / '.env'
    if env_path.exists():
        check_ok("~/.hermes/.env file exists")
        
        # Check for common issues
        content = env_path.read_text()
        if "OPENROUTER_API_KEY" in content or "ANTHROPIC_API_KEY" in content:
            check_ok("API key configured")
        else:
            check_warn("No API key found in ~/.hermes/.env")
            issues.append("Run 'hermes setup' to configure API keys")
    else:
        # Also check project root as fallback
        fallback_env = PROJECT_ROOT / '.env'
        if fallback_env.exists():
            check_ok(".env file exists (in project directory)")
        else:
            check_fail("~/.hermes/.env file missing")
            check_info("Run 'hermes setup' to create one")
            issues.append("Run 'hermes setup' to create .env")
    
    # Check ~/.hermes/config.yaml (primary) or project cli-config.yaml (fallback)
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        check_ok("~/.hermes/config.yaml exists")
    else:
        fallback_config = PROJECT_ROOT / 'cli-config.yaml'
        if fallback_config.exists():
            check_ok("cli-config.yaml exists (in project directory)")
        else:
            check_warn("config.yaml not found", "(using defaults)")
    
    # =========================================================================
    # Check: Directory structure
    # =========================================================================
    print()
    print(color("◆ Directory Structure", Colors.CYAN, Colors.BOLD))
    
    hermes_home = Path.home() / ".hermes"
    if hermes_home.exists():
        check_ok("~/.hermes directory exists")
    else:
        check_warn("~/.hermes not found", "(will be created on first use)")
    
    # Check for SOUL.md persona file
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8").strip()
        # Check if it's just the template comments (no real content)
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith(("<!--", "-->", "#"))]
        if lines:
            check_ok("~/.hermes/SOUL.md exists (persona configured)")
        else:
            check_info("~/.hermes/SOUL.md exists but is empty — edit it to customize personality")
    else:
        check_warn("~/.hermes/SOUL.md not found", "(create it to give Hermes a custom personality)")
        if should_fix:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text("# Hermes Agent Persona\n\n<!-- Edit this file to customize how Hermes communicates. -->\n", encoding="utf-8")
            check_ok("Created ~/.hermes/SOUL.md")
    
    logs_dir = PROJECT_ROOT / "logs"
    if logs_dir.exists():
        check_ok("logs/ directory exists")
    else:
        check_warn("logs/ not found", "(will be created on first use)")
    
    # =========================================================================
    # Check: External tools
    # =========================================================================
    print()
    print(color("◆ External Tools", Colors.CYAN, Colors.BOLD))
    
    # Git
    if shutil.which("git"):
        check_ok("git")
    else:
        check_warn("git not found", "(optional)")
    
    # ripgrep (optional, for faster file search)
    if shutil.which("rg"):
        check_ok("ripgrep (rg)", "(faster file search)")
    else:
        check_warn("ripgrep (rg) not found", "(file search uses grep fallback)")
        check_info("Install for faster search: sudo apt install ripgrep")
    
    # Docker (optional)
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    if terminal_env == "docker":
        if shutil.which("docker"):
            # Check if docker daemon is running
            result = subprocess.run(["docker", "info"], capture_output=True)
            if result.returncode == 0:
                check_ok("docker", "(daemon running)")
            else:
                check_fail("docker daemon not running")
                issues.append("Start Docker daemon")
        else:
            check_fail("docker not found", "(required for TERMINAL_ENV=docker)")
            issues.append("Install Docker or change TERMINAL_ENV")
    else:
        if shutil.which("docker"):
            check_ok("docker", "(optional)")
        else:
            check_warn("docker not found", "(optional)")
    
    # SSH (if using ssh backend)
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST")
        if ssh_host:
            # Try to connect
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_host, "echo ok"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                check_ok(f"SSH connection to {ssh_host}")
            else:
                check_fail(f"SSH connection to {ssh_host}")
                issues.append(f"Check SSH configuration for {ssh_host}")
        else:
            check_fail("TERMINAL_SSH_HOST not set", "(required for TERMINAL_ENV=ssh)")
            issues.append("Set TERMINAL_SSH_HOST in .env")
    
    # Node.js + agent-browser (for browser automation tools)
    if shutil.which("node"):
        check_ok("Node.js")
        # Check if agent-browser is installed
        agent_browser_path = PROJECT_ROOT / "node_modules" / "agent-browser"
        if agent_browser_path.exists():
            check_ok("agent-browser (Node.js)", "(browser automation)")
        else:
            check_warn("agent-browser not installed", "(run: npm install)")
    else:
        check_warn("Node.js not found", "(optional, needed for browser tools)")
    
    # =========================================================================
    # Check: API connectivity
    # =========================================================================
    print()
    print(color("◆ API Connectivity", Colors.CYAN, Colors.BOLD))
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            import httpx
            response = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {openrouter_key}"},
                timeout=10
            )
            if response.status_code == 200:
                check_ok("OpenRouter API")
            elif response.status_code == 401:
                check_fail("OpenRouter API", "(invalid API key)")
                issues.append("Check OPENROUTER_API_KEY in .env")
            else:
                check_fail("OpenRouter API", f"(HTTP {response.status_code})")
        except Exception as e:
            check_fail("OpenRouter API", f"({e})")
            issues.append("Check network connectivity")
    else:
        check_warn("OpenRouter API", "(not configured)")
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import httpx
            response = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01"
                },
                timeout=10
            )
            if response.status_code == 200:
                check_ok("Anthropic API")
            elif response.status_code == 401:
                check_fail("Anthropic API", "(invalid API key)")
            else:
                # Note: Anthropic may not have /models endpoint
                check_warn("Anthropic API", "(couldn't verify)")
        except Exception as e:
            check_warn("Anthropic API", f"({e})")
    
    # =========================================================================
    # Check: Submodules
    # =========================================================================
    print()
    print(color("◆ Submodules", Colors.CYAN, Colors.BOLD))
    
    # mini-swe-agent (terminal tool backend)
    mini_swe_dir = PROJECT_ROOT / "mini-swe-agent"
    if mini_swe_dir.exists() and (mini_swe_dir / "pyproject.toml").exists():
        try:
            __import__("minisweagent")
            check_ok("mini-swe-agent", "(terminal backend)")
        except ImportError:
            check_warn("mini-swe-agent found but not installed", "(run: uv pip install -e ./mini-swe-agent)")
            issues.append("Install mini-swe-agent: uv pip install -e ./mini-swe-agent")
    else:
        check_warn("mini-swe-agent not found", "(run: git submodule update --init --recursive)")
    
    # tinker-atropos (RL training backend)
    tinker_dir = PROJECT_ROOT / "tinker-atropos"
    if tinker_dir.exists() and (tinker_dir / "pyproject.toml").exists():
        if py_version >= (3, 11):
            try:
                __import__("tinker_atropos")
                check_ok("tinker-atropos", "(RL training backend)")
            except ImportError:
                check_warn("tinker-atropos found but not installed", "(run: uv pip install -e ./tinker-atropos)")
                issues.append("Install tinker-atropos: uv pip install -e ./tinker-atropos")
        else:
            check_warn("tinker-atropos requires Python 3.11+", f"(current: {py_version.major}.{py_version.minor})")
    else:
        check_warn("tinker-atropos not found", "(run: git submodule update --init --recursive)")
    
    # =========================================================================
    # Check: Tool Availability
    # =========================================================================
    print()
    print(color("◆ Tool Availability", Colors.CYAN, Colors.BOLD))
    
    try:
        # Add project root to path for imports
        sys.path.insert(0, str(PROJECT_ROOT))
        from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
        
        available, unavailable = check_tool_availability()
        
        for tid in available:
            info = TOOLSET_REQUIREMENTS.get(tid, {})
            check_ok(info.get("name", tid))
        
        for item in unavailable:
            if item["missing_vars"]:
                vars_str = ", ".join(item["missing_vars"])
                check_warn(item["name"], f"(missing {vars_str})")
            else:
                check_warn(item["name"], "(system dependency not met)")
        
        # Count disabled tools with API key requirements
        api_disabled = [u for u in unavailable if u["missing_vars"]]
        if api_disabled:
            issues.append("Run 'hermes setup' to configure missing API keys for full tool access")
    except Exception as e:
        check_warn("Could not check tool availability", f"({e})")
    
    # =========================================================================
    # Check: Skills Hub
    # =========================================================================
    print()
    print(color("◆ Skills Hub", Colors.CYAN, Colors.BOLD))

    hub_dir = PROJECT_ROOT / "skills" / ".hub"
    if hub_dir.exists():
        check_ok("Skills Hub directory exists")
        lock_file = hub_dir / "lock.json"
        if lock_file.exists():
            try:
                import json
                lock_data = json.loads(lock_file.read_text())
                count = len(lock_data.get("installed", {}))
                check_ok(f"Lock file OK ({count} hub-installed skill(s))")
            except Exception:
                check_warn("Lock file", "(corrupted or unreadable)")
        quarantine = hub_dir / "quarantine"
        q_count = sum(1 for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else 0
        if q_count > 0:
            check_warn(f"{q_count} skill(s) in quarantine", "(pending review)")
    else:
        check_warn("Skills Hub directory not initialized", "(run: hermes skills list)")

    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if github_token:
        check_ok("GitHub token configured (authenticated API access)")
    else:
        check_warn("No GITHUB_TOKEN", "(60 req/hr rate limit — set in ~/.hermes/.env for better rates)")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    if issues:
        print(color("─" * 60, Colors.YELLOW))
        print(color(f"  Found {len(issues)} issue(s) to address:", Colors.YELLOW, Colors.BOLD))
        print()
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        
        if should_fix:
            print(color("  Attempting auto-fix is not yet implemented.", Colors.DIM))
            print(color("  Please resolve issues manually.", Colors.DIM))
    else:
        print(color("─" * 60, Colors.GREEN))
        print(color("  All checks passed! 🎉", Colors.GREEN, Colors.BOLD))
    
    print()
