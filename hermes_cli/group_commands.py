"""
Hermes Group Commands - Multi-agent group chat functionality.

Usage:
    hermes group agents                 - List all registered agents
    hermes group register <name>        - Register an agent from profile
    hermes group unregister <name>      - Unregister an agent
    hermes group create <name> <agents> - Create a group
    hermes group list                  - List all groups
    hermes group delete <name>         - Delete a group
    hermes group info <name>           - Show group details
    hermes group add <group> <agent>   - Add agent to group
    hermes group remove <group> <agent> - Remove agent from group
    hermes group chat <name>           - Start group chat
    hermes group continue <file>       - Continue from memory file
    hermes group memory                - List memory files
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Global flag for interrupt handling
_interrupted = False
_original_sigint_handler = None

# Terminal utilities
try:
    from hermes_cli.curses_ui import flush_stdin
except ImportError:
    import termios
    def flush_stdin():
        """Flush any stray bytes from stdin (only when needed)."""
        try:
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


def _sigint_handler(signum, frame):
    """Custom SIGINT handler that sets a flag instead of terminating."""
    global _interrupted
    _interrupted = True


def _setup_readline() -> None:
    """Setup readline for proper line editing."""
    try:
        import readline

        # Set up basic editing bindings for cross-platform compatibility
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind(r'"\e[3~": delete-char')
        readline.parse_and_bind(r'"\e[2~": quoted-insert')
        readline.parse_and_bind(r'"\e[H": beginning-of-line')
        readline.parse_and_bind(r'"\e[F": end-of-line')
        readline.parse_and_bind(r'"\e[1~": beginning-of-line')
        readline.parse_and_bind(r'"\e[4~": end-of-line')

        # Enable vi/emacs mode (emacs is more common and intuitive for most)
        readline.parse_and_bind("set editing-mode emacs")

        # History settings
        readline.set_history_length(100)
    except ImportError:
        pass
    except Exception:
        pass


def _get_input(prompt: str = "You: ") -> str:
    """Get user input with proper line editing support."""
    _setup_readline()
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        raise

# Lazy imports for optional packages - loaded inside functions to avoid hard deps

# =============================================================================
# Constants
# =============================================================================

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

# Profiles are always in ~/.hermes/profiles/ (root level), not in HERMES_HOME
HERMES_ROOT = Path.home() / ".hermes"
PROFILES_DIR = HERMES_ROOT / "profiles"

# Group data is stored in the root hermes directory (shared across profiles)
GROUPS_DIR = HERMES_ROOT / "groups"
AGENTS_REGISTRY = HERMES_ROOT / "agents" / "registry.yaml"
MEMORY_DIR = HERMES_ROOT / "group_memory"

# Ensure directories exist
GROUPS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
AGENTS_REGISTRY.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Agent Registry
# =============================================================================

def load_agent_registry() -> dict[str, dict[str, Any]]:
    """Load agent registry from YAML file."""
    if not AGENTS_REGISTRY.exists():
        return {}
    try:
        with open(AGENTS_REGISTRY, "r") as f:
            data = yaml.safe_load(f)
            return data.get("agents", {}) if data else {}
    except Exception:
        return {}


def save_agent_registry(registry: dict[str, dict[str, Any]]) -> None:
    """Save agent registry to YAML file."""
    with open(AGENTS_REGISTRY, "w") as f:
        yaml.dump({"agents": registry}, f, default_flow_style=False)


def list_agents() -> list[dict[str, Any]]:
    """List all registered agents."""
    registry = load_agent_registry()
    return [
        {"name": name, **info}
        for name, info in registry.items()
    ]


def register_agent(name: str, profile: str | None = None) -> bool:
    """Register an agent from a profile."""
    registry = load_agent_registry()

    # Check if profile exists (profiles are in ~/.hermes/profiles/)
    profile_path = PROFILES_DIR / (profile or name)
    if not profile_path.exists():
        print(f"Error: Profile '{profile or name}' does not exist")
        print(f"Available profiles:")
        if PROFILES_DIR.exists():
            for p in sorted(PROFILES_DIR.iterdir()):
                if p.is_dir():
                    print(f"  - {p.name}")
        return False

    # Get soul.md content as system prompt
    soul_path = profile_path / "SOUL.md"
    system_message = ""
    if soul_path.exists():
        system_message = soul_path.read_text()

    registry[name] = {
        "profile": profile or name,
        "registered_at": datetime.now().isoformat(),
        "system_message": system_message if system_message else f"You are {name}.",
    }
    save_agent_registry(registry)
    return True


def unregister_agent(name: str) -> bool:
    """Unregister an agent."""
    registry = load_agent_registry()
    if name not in registry:
        print(f"Error: Agent '{name}' is not registered")
        return False
    del registry[name]
    save_agent_registry(registry)
    return True


# =============================================================================
# Group Management
# =============================================================================

def load_groups_config() -> dict[str, dict[str, Any]]:
    """Load groups configuration."""
    config_file = GROUPS_DIR / "config.yaml"
    if not config_file.exists():
        return {"groups": {}}
    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {"groups": {}}
    except Exception:
        return {"groups": {}}


def save_groups_config(config: dict[str, dict[str, Any]]) -> None:
    """Save groups configuration."""
    config_file = GROUPS_DIR / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_group(name: str, agents: list[str]) -> bool:
    """Create a new group with specified agents."""
    if len(agents) < 2:
        print("Error: A group must have at least 2 agents")
        return False

    registry = load_agent_registry()

    # Verify all agents are registered
    for agent in agents:
        if agent not in registry:
            print(f"Error: Agent '{agent}' is not registered")
            print("Run 'hermes group agents' to see registered agents")
            return False

    config = load_groups_config()

    if name in config.get("groups", {}):
        print(f"Error: Group '{name}' already exists")
        return False

    if "groups" not in config:
        config["groups"] = {}

    config["groups"][name] = {
        "agents": agents,
        "created_at": datetime.now().isoformat(),
        "last_chat": None,
    }
    save_groups_config(config)

    print(f"Group '{name}' created with agents: {', '.join(agents)}")
    return True


def list_groups() -> list[dict[str, Any]]:
    """List all groups."""
    config = load_groups_config()
    groups = config.get("groups", {})
    return [
        {
            "name": name,
            "agents": info.get("agents", []),
            "created_at": info.get("created_at", ""),
            "last_chat": info.get("last_chat"),
        }
        for name, info in groups.items()
    ]


def delete_group(name: str) -> bool:
    """Delete a group."""
    config = load_groups_config()
    if name not in config.get("groups", {}):
        print(f"Error: Group '{name}' does not exist")
        return False
    del config["groups"][name]
    save_groups_config(config)
    print(f"Group '{name}' deleted")
    return True


def show_group_info(name: str) -> dict[str, Any] | None:
    """Show detailed information about a group."""
    config = load_groups_config()
    group = config.get("groups", {}).get(name)
    if not group:
        print(f"Error: Group '{name}' does not exist")
        return None
    return {"name": name, **group}


def add_agent_to_group(group_name: str, agent_name: str) -> bool:
    """Add an agent to an existing group."""
    registry = load_agent_registry()
    if agent_name not in registry:
        print(f"Error: Agent '{agent_name}' is not registered")
        return False

    config = load_groups_config()
    if group_name not in config.get("groups", {}):
        print(f"Error: Group '{group_name}' does not exist")
        return False

    agents = config["groups"][group_name].get("agents", [])
    if agent_name in agents:
        print(f"Agent '{agent_name}' is already in group '{group_name}'")
        return False

    agents.append(agent_name)
    config["groups"][group_name]["agents"] = agents
    save_groups_config(config)
    print(f"Agent '{agent_name}' added to group '{group_name}'")
    return True


def remove_agent_from_group(group_name: str, agent_name: str) -> bool:
    """Remove an agent from a group."""
    config = load_groups_config()
    if group_name not in config.get("groups", {}):
        print(f"Error: Group '{group_name}' does not exist")
        return False

    agents = config["groups"][group_name].get("agents", [])
    if agent_name not in agents:
        print(f"Error: Agent '{agent_name}' is not in group '{group_name}'")
        return False

    if len(agents) <= 2:
        print("Error: A group must have at least 2 agents")
        return False

    agents.remove(agent_name)
    config["groups"][group_name]["agents"] = agents
    save_groups_config(config)
    print(f"Agent '{agent_name}' removed from group '{group_name}'")
    return True


# =============================================================================
# Memory System
# =============================================================================

def get_memory_filename(group_name: str) -> Path:
    """Generate a memory filename for a group."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return MEMORY_DIR / f"{timestamp}_{group_name}.jsonl"


def save_message_to_memory(filename: Path, message: dict[str, Any]) -> None:
    """Save a message to the memory file."""
    with open(filename, "a") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def load_memory_file(filename: Path) -> list[dict[str, Any]]:
    """Load messages from a memory file."""
    if not filename.exists():
        return []
    messages = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return messages


def list_memory_files() -> list[dict[str, Any]]:
    """List all memory files."""
    if not MEMORY_DIR.exists():
        return []
    files = []
    for f in sorted(MEMORY_DIR.iterdir()):
        if f.suffix == ".jsonl":
            stat = f.stat()
            files.append({
                "name": f.name,
                "path": str(f),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return files


# =============================================================================
# Group Chat (using autogen-agentchat 0.7.x)
# =============================================================================

def get_llm_config() -> dict[str, str]:
    """Get LLM configuration from config.yaml."""
    config_file = HERMES_ROOT / "config.yaml"
    if not config_file.exists():
        return {}

    try:
        import yaml
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        return {
            "model": model_config.get("default", "deepseek-v3"),
            "api_key": model_config.get("api_key", ""),
            "base_url": model_config.get("base_url", "https://api.newcoin.top/v1"),
        }
    except Exception:
        return {}


def run_group_chat(group_name: str, continue_file: Path | None = None) -> None:
    """Run an interactive group chat."""
    # Reset terminal state before any input() - crucial for Chinese IME
    import os
    os.system("stty sane 2>/dev/null")

    # Reset interrupted flag
    global _interrupted
    _interrupted = False

    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_core.models import ModelInfo
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError as e:
        print("Error: autogen packages are required for group chat")
        print(f"Import error: {e}")
        print("Install with: pip install autogen-agentchat autogen-ext --break-system-packages")
        return

    config = load_groups_config()
    group = config.get("groups", {}).get(group_name)
    if not group:
        print(f"Error: Group '{group_name}' does not exist")
        return

    agents_list = group.get("agents", [])
    registry = load_agent_registry()

    # Get LLM config from config.yaml
    llm_config = get_llm_config()

    # Fallback to environment variables
    model = llm_config.get("model") or os.environ.get("HERMES_GROUP_MODEL", "deepseek-v3")
    api_key = llm_config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = llm_config.get("base_url") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.newcoin.top/v1")

    if not api_key:
        print("Error: No API key configured")
        print("Please set up your model in Hermes first: hermes model")
        return

    # Create model client
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        model_info=ModelInfo(
            family="deepseek",
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=False,
            multiple_system_messages=True,
        ),
    )

    # Create agents (without user proxy - we'll handle input manually)
    participants = []
    for agent_name in agents_list:
        agent_info = registry.get(agent_name, {})
        system_msg = agent_info.get(
            "system_message",
            f"You are {agent_name}."
        )

        agent = AssistantAgent(
            name=agent_name,
            model_client=model_client,
            system_message=system_msg,
        )
        participants.append(agent)

    # Create group chat WITHOUT user - we handle input manually
    chat = RoundRobinGroupChat(
        participants=participants,
        max_turns=50,
    )

    # Memory file for this session
    memory_file = get_memory_filename(group_name)

    print("\n" + "=" * 60)
    print(f"GROUP CHAT: {group_name}")
    print(f"Agents: {', '.join(agents_list)}")
    print("Type 'exit' to end | 'stop' to interrupt | 'save' to save")
    print("=" * 60 + "\n")

    # Save session start
    save_message_to_memory(memory_file, {
        "type": "session_start",
        "group": group_name,
        "agents": agents_list,
        "timestamp": datetime.now().isoformat(),
    })

    # Interactive chat loop - manually send messages to group chat
    has_messages = False
    current_task = None

    def _handle_interrupt(signum, frame):
        """Handle Ctrl+C: cancel current task and break loop."""
        global _interrupted, current_task
        _interrupted = True
        if current_task and not current_task.done():
            current_task.cancel()
        raise KeyboardInterrupt

    # Set up signal handler that cancels asyncio task
    _original_sigint_handler = signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        while True:
            # Check if we were interrupted by Ctrl+C
            if _interrupted:
                print("\n\n[Ctrl+C detected]")
                break

            try:
                user_input = _get_input("You: ")
            except KeyboardInterrupt:
                print("\n\n[Ctrl+C detected]")
                break
            except EOFError:
                print("\nInput ended.")
                break

            if not user_input:
                continue

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "stop":
                if current_task and not current_task.done():
                    current_task.cancel()
                print("\n[任务已中断]")
                continue
            elif user_input.lower() == "save":
                # Save without prompting
                if memory_file.exists():
                    print(f"\n对话已保存到: {memory_file}")
                else:
                    print("\n暂无对话可保存")
                continue

            print()  # New line after input

            # Run the group chat with the user's message
            try:
                # Create task and store reference so signal handler can cancel it
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    current_task = loop.create_task(chat.run(task=user_input))
                    result = loop.run_until_complete(current_task)
                finally:
                    loop.close()

                # Reset interrupted flag in case task was cancelled but handled
                _interrupted = False

                # Print agent responses
                for msg in result.messages:
                    if hasattr(msg, 'source') and msg.source not in ("user", "system"):
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        print(f"\n[{msg.source}]:\n{content}\n")

                        # Save to memory
                        save_message_to_memory(memory_file, {
                            "type": "message",
                            "sender": msg.source,
                            "content": content,
                            "timestamp": datetime.now().isoformat(),
                        })
                        has_messages = True

            except asyncio.CancelledError:
                print("\n[任务已取消]")
                _interrupted = False
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

    except EOFError:
        print("\n\nInput ended.")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, _original_sigint_handler)

    # Always ask if user wants to save (after Ctrl+C or normal exit)
    # Default to NO to prevent automatic saves
    save = "n"
    try:
        save = _get_input("\n保存对话? [Y/n]: ").lower()
    except (KeyboardInterrupt, EOFError):
        print()  # newline after Ctrl+C
        save = "y"

    # Normalize input
    if save in ("n", "no"):
        try:
            memory_file.unlink(missing_ok=True)
            print("对话已丢弃")
        except Exception as e:
            print(f"无法删除文件: {e}")
    else:
        print(f"\n对话已保存到: {memory_file}")

    # Update last_chat
    config["groups"][group_name]["last_chat"] = datetime.now().isoformat()
    save_groups_config(config)


def save_chat_messages_sync(chat, memory_file: Path) -> None:
    """Save chat messages to memory file (sync version)."""
    try:
        messages = []
        if hasattr(chat, 'messages'):
            messages = chat.messages or []

        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                sender = getattr(msg, 'name', 'unknown')
            elif isinstance(msg, dict):
                content = msg.get('content', '')
                sender = msg.get('name', msg.get('role', 'unknown'))
            else:
                continue

            save_message_to_memory(memory_file, {
                "type": "message",
                "sender": sender,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            })
    except Exception as e:
        print(f"Warning: Could not save messages to memory: {e}")


# =============================================================================
# CLI Command Handler
# =============================================================================

def group_command(args: Any) -> None:
    """Main entry point for group commands."""
    action = getattr(args, "group_action", None)

    if action is None:
        # No subcommand, show help
        print("hermes group - Multi-agent group chat management")
        print("\nCommands:")
        print("  agents                 List all registered agents")
        print("  register <name>        Register an agent from profile")
        print("  unregister <name>      Unregister an agent")
        print("  create <name> <agents> Create a group (space-separated agent names)")
        print("  list                   List all groups")
        print("  delete <name>          Delete a group")
        print("  info <name>            Show group details")
        print("  add <group> <agent>    Add agent to group")
        print("  remove <group> <agent> Remove agent from group")
        print("  chat <name>             Start group chat")
        print("  continue <file>        Continue from memory file")
        print("  memory                 List memory files")
        return

    # Agent commands
    if action == "agents":
        if getattr(args, "register", None):
            # hermes group agents register <name>
            name = args.register
            profile = getattr(args, "profile", None)
            if register_agent(name, profile):
                print(f"Agent '{name}' registered from profile '{profile or name}'")
        elif getattr(args, "unregister", None):
            # hermes group agents unregister <name>
            name = args.unregister
            if unregister_agent(name):
                print(f"Agent '{name}' unregistered")
        else:
            # List agents
            agents = list_agents()
            if not agents:
                print("No agents registered.")
                print("Run 'hermes group agents register <name>' to register an agent")
                return
            print(f"\nRegistered Agents ({len(agents)}):")
            print("-" * 40)
            for agent in agents:
                print(f"  {agent['name']}")
                print(f"    Profile: {agent.get('profile', 'N/A')}")
                print(f"    Registered: {agent.get('registered_at', 'N/A')}")
                print()

    elif action == "register":
        name = getattr(args, "agent_name", None)
        if not name:
            print("Error: Agent name required")
            print("Usage: hermes group register <name> [--profile <profile>]")
            return
        profile = getattr(args, "profile", None)
        register_agent(name, profile)

    elif action == "unregister":
        name = getattr(args, "agent_name", None)
        if not name:
            print("Error: Agent name required")
            return
        unregister_agent(name)

    # Group commands
    elif action == "create":
        name = getattr(args, "group_name", None)
        agents = getattr(args, "agents", [])
        if not name:
            print("Error: Group name required")
            print("Usage: hermes group create <name> <agent1> <agent2> ...")
            return
        if len(agents) < 2:
            print("Error: A group must have at least 2 agents")
            print("Usage: hermes group create <name> <agent1> <agent2> ...")
            return
        create_group(name, agents)

    elif action == "list":
        groups = list_groups()
        if not groups:
            print("No groups created.")
            print("Run 'hermes group create <name> <agents...>' to create a group")
            return
        print(f"\nGroups ({len(groups)}):")
        print("-" * 50)
        for g in groups:
            print(f"  {g['name']}")
            print(f"    Agents: {', '.join(g['agents'])}")
            print(f"    Created: {g['created_at']}")
            if g.get('last_chat'):
                print(f"    Last chat: {g['last_chat']}")
            print()

    elif action == "delete":
        name = getattr(args, "group_name", None)
        if not name:
            print("Error: Group name required")
            return
        delete_group(name)

    elif action == "info":
        name = getattr(args, "group_name", None)
        if not name:
            print("Error: Group name required")
            return
        info = show_group_info(name)
        if info:
            print(f"\nGroup: {info['name']}")
            print("-" * 40)
            print(f"Agents: {', '.join(info.get('agents', []))}")
            print(f"Created: {info.get('created_at', 'N/A')}")
            print(f"Last chat: {info.get('last_chat', 'Never')}")

    elif action == "add":
        group_name = getattr(args, "group_name", None)
        agent_name = getattr(args, "agent_name", None)
        if not group_name or not agent_name:
            print("Error: Group name and agent name required")
            return
        add_agent_to_group(group_name, agent_name)

    elif action == "remove":
        group_name = getattr(args, "group_name", None)
        agent_name = getattr(args, "agent_name", None)
        if not group_name or not agent_name:
            print("Error: Group name and agent name required")
            return
        remove_agent_from_group(group_name, agent_name)

    elif action == "chat":
        name = getattr(args, "group_name", None)
        if not name:
            print("Error: Group name required")
            print("Usage: hermes group chat <name>")
            return
        run_group_chat(name)

    elif action == "continue":
        filename = getattr(args, "memory_file", None)
        if not filename:
            print("Error: Memory file required")
            print("Usage: hermes group continue <memory_file>")
            return
        path = Path(filename)
        if not path.exists():
            # Try to find in memory directory
            path = MEMORY_DIR / filename
            if not path.exists():
                print(f"Error: Memory file '{filename}' not found")
                return

        # Extract group name from filename
        # Format: YYYY-MM-DD_HH-MM-SS_groupname.jsonl
        group_name = path.stem.split("_", 2)[-1] if "_" in path.stem else path.stem
        run_group_chat(group_name, path)

    elif action == "memory":
        files = list_memory_files()
        if not files:
            print("No memory files found.")
            print("Memory files are created when you run 'hermes group chat'")
            return
        print(f"\nMemory Files ({len(files)}):")
        print("-" * 60)
        for f in files:
            print(f"  {f['name']}")
            print(f"    Size: {f['size']} bytes")
            print(f"    Modified: {f['modified']}")
            print()
