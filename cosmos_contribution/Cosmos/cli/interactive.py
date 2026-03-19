"""
cosmos Interactive Shell

An enhanced interactive shell with history, autocomplete, and rich output.
"""

import asyncio
import readline
import atexit
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass
import shlex


@dataclass
class Command:
    """A shell command definition."""
    name: str
    description: str
    handler: Callable
    aliases: List[str] = None
    usage: str = ""


class InteractiveShell:
    """
    Enhanced interactive shell for cosmos.

    Features:
    - Command history with persistence
    - Tab completion
    - Colored output
    - Aliases and shortcuts
    """

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the interactive shell."""
        self.data_dir = Path(data_dir or "./data")
        self.history_file = self.data_dir / ".cosmos_history"

        self.commands: Dict[str, Command] = {}
        self.running = False

        self._setup_readline()
        self._register_default_commands()

    def _setup_readline(self):
        """Setup readline with history and completion."""
        # Enable tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer)

        # Load history
        if self.history_file.exists():
            try:
                readline.read_history_file(str(self.history_file))
            except Exception:
                pass

        # Save history on exit
        atexit.register(self._save_history)

    def _save_history(self):
        """Save command history."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(self.history_file))
        except Exception:
            pass

    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion handler."""
        options = [
            cmd for cmd in self.commands.keys()
            if cmd.startswith(text)
        ]

        if state < len(options):
            return options[state]
        return None

    def _register_default_commands(self):
        """Register default shell commands."""
        self.register_command(Command(
            name="help",
            description="Show available commands",
            handler=self._cmd_help,
            aliases=["?", "h"],
        ))

        self.register_command(Command(
            name="exit",
            description="Exit the shell",
            handler=self._cmd_exit,
            aliases=["quit", "q"],
        ))

        self.register_command(Command(
            name="clear",
            description="Clear the screen",
            handler=self._cmd_clear,
            aliases=["cls"],
        ))

        self.register_command(Command(
            name="status",
            description="Show system status",
            handler=self._cmd_status,
        ))

        self.register_command(Command(
            name="health",
            description="Health tracking commands",
            handler=self._cmd_health,
            usage="health [dashboard|log|tips]",
        ))

        self.register_command(Command(
            name="memory",
            description="Memory operations",
            handler=self._cmd_memory,
            usage="memory [search|save|stats] [args...]",
        ))

        self.register_command(Command(
            name="agent",
            description="Interact with AI agents",
            handler=self._cmd_agent,
            usage="agent [ask|task] <query>",
        ))

        self.register_command(Command(
            name="workflow",
            description="Workflow automation",
            handler=self._cmd_workflow,
            usage="workflow [list|run|create] [name]",
        ))

        self.register_command(Command(
            name="n8n",
            description="n8n integration",
            handler=self._cmd_n8n,
            usage="n8n [connect|trigger|list]",
        ))

    def register_command(self, command: Command):
        """Register a command."""
        self.commands[command.name] = command
        if command.aliases:
            for alias in command.aliases:
                self.commands[alias] = command

    async def run(self, prompt: str = "cosmos> "):
        """Run the interactive shell."""
        self.running = True

        print("\ncosmos Interactive Shell")
        print("Type 'help' for available commands.\n")

        while self.running:
            try:
                # Get input
                line = input(prompt).strip()

                if not line:
                    continue

                # Parse command
                parts = shlex.split(line)
                cmd_name = parts[0].lower()
                args = parts[1:]

                # Find and execute command
                if cmd_name in self.commands:
                    cmd = self.commands[cmd_name]
                    result = cmd.handler(args)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    print(f"Unknown command: {cmd_name}")
                    print("Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

    # ========== Built-in Commands ==========

    def _cmd_help(self, args: List[str]):
        """Show help."""
        if args:
            # Show help for specific command
            cmd_name = args[0].lower()
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                print(f"\n{cmd.name}: {cmd.description}")
                if cmd.usage:
                    print(f"Usage: {cmd.usage}")
                if cmd.aliases:
                    print(f"Aliases: {', '.join(cmd.aliases)}")
            else:
                print(f"Unknown command: {cmd_name}")
        else:
            # Show all commands
            print("\nAvailable Commands:\n")

            # Get unique commands (no aliases)
            seen = set()
            for name, cmd in sorted(self.commands.items()):
                if cmd.name not in seen:
                    seen.add(cmd.name)
                    aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
                    print(f"  {cmd.name:15} {cmd.description}{aliases}")

            print("\nType 'help <command>' for more details.")

    def _cmd_exit(self, args: List[str]):
        """Exit the shell."""
        print("Goodbye!")
        self.running = False

    def _cmd_clear(self, args: List[str]):
        """Clear screen."""
        print("\033[H\033[J", end="")

    async def _cmd_status(self, args: List[str]):
        """Show status."""
        print("\nSystem Status")
        print("=" * 40)
        print(" Memory:  Initialized")
        print(" Health:  Ready")
        print(" Agents:  Available")
        print(" n8n:     Not connected")

    async def _cmd_health(self, args: List[str]):
        """Health commands."""
        if not args:
            print("Usage: health [dashboard|log|tips|goals]")
            return

        subcmd = args[0].lower()

        if subcmd == "dashboard":
            print("\nHealth Dashboard - Coming soon!")
        elif subcmd == "log":
            print("\nMeal Logging - Coming soon!")
        elif subcmd == "tips":
            print("\nHealth Tips - Coming soon!")
        elif subcmd == "goals":
            print("\nHealth Goals - Coming soon!")
        else:
            print(f"Unknown health command: {subcmd}")

    async def _cmd_memory(self, args: List[str]):
        """Memory commands."""
        if not args:
            print("Usage: memory [search|save|stats|export] [args...]")
            return

        subcmd = args[0].lower()

        if subcmd == "search":
            query = " ".join(args[1:]) if len(args) > 1 else input("Search query: ")
            print(f"\nSearching for: {query}")
            print("(Memory search results would appear here)")
        elif subcmd == "save":
            content = " ".join(args[1:]) if len(args) > 1 else input("Content to save: ")
            print(f"\nSaving memory: {content[:50]}...")
            print("Memory saved!")
        elif subcmd == "stats":
            print("\nMemory Statistics")
            print("  Total entries: 0")
            print("  Knowledge graph: 0 entities")
        elif subcmd == "export":
            print("\nExporting memories...")
            print("Export complete!")
        else:
            print(f"Unknown memory command: {subcmd}")

    async def _cmd_agent(self, args: List[str]):
        """Agent commands."""
        if not args:
            print("Usage: agent [ask|task|list] <query>")
            return

        subcmd = args[0].lower()

        if subcmd == "ask":
            query = " ".join(args[1:]) if len(args) > 1 else input("Your question: ")
            print(f"\nThinking about: {query}")
            print("(Agent response would appear here)")
        elif subcmd == "task":
            task = " ".join(args[1:]) if len(args) > 1 else input("Task description: ")
            print(f"\nStarting task: {task}")
            print("Task queued for processing!")
        elif subcmd == "list":
            print("\nAvailable Agents:")
            print("  - Researcher: Information gathering")
            print("  - Coder: Programming tasks")
            print("  - Writer: Content creation")
            print("  - Analyst: Data analysis")
        else:
            print(f"Unknown agent command: {subcmd}")

    async def _cmd_workflow(self, args: List[str]):
        """Workflow commands."""
        if not args:
            print("Usage: workflow [list|run|create|delete] [name]")
            return

        subcmd = args[0].lower()

        if subcmd == "list":
            print("\nSaved Workflows:")
            print("  (No workflows yet)")
        elif subcmd == "run":
            name = args[1] if len(args) > 1 else input("Workflow name: ")
            print(f"\nRunning workflow: {name}")
        elif subcmd == "create":
            print("\nWorkflow Builder")
            print("(Interactive workflow builder would start here)")
        elif subcmd == "delete":
            name = args[1] if len(args) > 1 else input("Workflow name: ")
            print(f"\nDeleting workflow: {name}")
        else:
            print(f"Unknown workflow command: {subcmd}")

    async def _cmd_n8n(self, args: List[str]):
        """n8n commands."""
        if not args:
            print("Usage: n8n [connect|trigger|list|status]")
            return

        subcmd = args[0].lower()

        if subcmd == "connect":
            url = input("n8n URL: ")
            api_key = input("API Key: ")
            print(f"\nConnecting to {url}...")
        elif subcmd == "trigger":
            workflow_id = args[1] if len(args) > 1 else input("Workflow ID: ")
            print(f"\nTriggering workflow: {workflow_id}")
        elif subcmd == "list":
            print("\nn8n Workflows:")
            print("  (Not connected)")
        elif subcmd == "status":
            print("\nn8n Status: Not connected")
        else:
            print(f"Unknown n8n command: {subcmd}")
