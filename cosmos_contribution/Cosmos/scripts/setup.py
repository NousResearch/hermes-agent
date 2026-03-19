#!/usr/bin/env python3
"""
Farnsworth Installation Script

Handles:
- Dependency installation
- Data directory creation
- Model downloading
- Claude Code configuration
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def print_error(text):
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")


def check_python_version():
    """Check Python version meets requirements."""
    print_header("Checking Python Version")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False

    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_system():
    """Check system compatibility."""
    print_header("System Information")

    system = platform.system()
    machine = platform.machine()

    print_info(f"Operating System: {system}")
    print_info(f"Architecture: {machine}")

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_success(f"CUDA GPU: {gpu_name} ({vram:.1f}GB VRAM)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success("Apple Silicon GPU (MPS) available")
        else:
            print_warning("No GPU detected - will use CPU")
    except ImportError:
        print_info("PyTorch not installed yet")

    return True


def install_dependencies():
    """Install Python dependencies."""
    print_header("Installing Dependencies")

    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"

    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False

    # Core dependencies
    print_info("Installing core dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print_error("Dependency installation failed")
        print(result.stderr)
        return False

    print_success("Dependencies installed")

    # Optional: Install GPU-specific packages
    try:
        import torch
        if torch.cuda.is_available():
            print_info("Installing CUDA-optimized packages...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "faiss-gpu"],
                capture_output=True,
            )
    except ImportError:
        pass

    return True


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directories")

    project_root = Path(__file__).parent.parent

    directories = [
        project_root / "data",
        project_root / "data" / "memories",
        project_root / "data" / "models",
        project_root / "data" / "embeddings",
        project_root / "data" / "evolution",
        project_root / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {directory.relative_to(project_root)}")

    return True


def configure_claude_code():
    """Generate Claude Code MCP configuration."""
    print_header("Claude Code Configuration")

    project_root = Path(__file__).parent.parent

    mcp_config = {
        "mcpServers": {
            "farnsworth": {
                "command": "python",
                "args": ["-m", "farnsworth.mcp_server"],
                "cwd": str(project_root)
            }
        }
    }

    # Detect Claude Code config location
    if platform.system() == "Windows":
        config_locations = [
            Path.home() / "AppData" / "Roaming" / "claude-code" / "mcp.json",
            Path.home() / ".claude-code" / "mcp.json",
        ]
    elif platform.system() == "Darwin":  # macOS
        config_locations = [
            Path.home() / "Library" / "Application Support" / "claude-code" / "mcp.json",
            Path.home() / ".config" / "claude-code" / "mcp.json",
        ]
    else:  # Linux
        config_locations = [
            Path.home() / ".config" / "claude-code" / "mcp.json",
        ]

    print_info("To integrate with Claude Code, add this to your MCP settings:")
    print()
    print(json.dumps(mcp_config, indent=2))
    print()

    # Save config example
    config_file = project_root / "claude_code_config.json"
    with open(config_file, "w") as f:
        json.dump(mcp_config, f, indent=2)

    print_success(f"Configuration saved to: {config_file}")
    return True


def verify_installation():
    """Verify Farnsworth installation."""
    print_header("Verifying Installation")

    checks_passed = True

    # Check imports
    try:
        from farnsworth.memory.memory_system import MemorySystem
        print_success("Memory system module OK")
    except ImportError as e:
        print_error(f"Memory system import failed: {e}")
        checks_passed = False

    try:
        from farnsworth.agents.swarm_orchestrator import SwarmOrchestrator
        print_success("Agent swarm module OK")
    except ImportError as e:
        print_error(f"Agent swarm import failed: {e}")
        checks_passed = False

    try:
        from farnsworth.evolution.fitness_tracker import FitnessTracker
        print_success("Evolution module OK")
    except ImportError as e:
        print_error(f"Evolution import failed: {e}")
        checks_passed = False

    try:
        from farnsworth.mcp_server.server import FarnsworthMCPServer
        print_success("MCP server module OK")
    except ImportError as e:
        print_warning(f"MCP server import warning: {e}")
        print_info("MCP library may need installation: pip install mcp")

    return checks_passed


def print_next_steps():
    """Print next steps for the user."""
    print_header("Next Steps")

    print("""
1. Download a local LLM (recommended):
   - Install Ollama: https://ollama.ai
   - Pull a model: ollama pull deepseek-r1:1.5b

2. Start Farnsworth:
   python main.py --setup    # First-time GRANULAR configuration (Privacy, Swarm, Engines)
   python main.py            # Start all services

3. Configure Claude Code:
   - Open Claude Code settings
   - Add the MCP configuration (see claude_code_config.json)
   - Restart Claude Code

4. Test the integration:
   - Open Claude Code
   - Try: "Remember that I prefer Python"
   - Later: "What programming language do I prefer?"

For more information, see README.md
    """)


def main():
    """Run the setup process."""
    print(f"""
{Colors.BOLD}╔═══════════════════════════════════════════╗
║     Farnsworth Installation Script        ║
║     Your Claude Companion AI              ║
╚═══════════════════════════════════════════╝{Colors.END}
    """)

    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("System Check", check_system),
        ("Directories", create_directories),
        ("Dependencies", install_dependencies),
        ("Claude Code Config", configure_claude_code),
        ("Verification", verify_installation),
    ]

    all_passed = True
    for name, func in steps:
        try:
            if not func():
                all_passed = False
                print_warning(f"{name} step had issues")
        except Exception as e:
            print_error(f"{name} failed: {e}")
            all_passed = False

    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}Installation Complete!{Colors.END}")
        print_next_steps()
    else:
        print(f"\n{Colors.YELLOW}Installation completed with warnings.{Colors.END}")
        print("Some features may not work until issues are resolved.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
