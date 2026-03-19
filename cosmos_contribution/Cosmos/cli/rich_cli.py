"""
Cosmos Rich CLI - Terminal UI with Swarm Integration.

AGI v1.8.4 Feature: Enhanced terminal interface using Rich/Textual
for real-time swarm interaction with visual agent status and signal monitoring.

Features:
- Split-pane TUI with agent status and signal feed
- Real-time chat with deliberation display
- Matplotlib canvas output support
- Command palette with autocomplete
- Live agent health indicators

Layout:
    ┌─────────────────────────────────────────────────────┐
    │ 🧠 Cosmos Swarm CLI                 [agents: 8] │
    ├─────────────────────────────────────────────────────┤
    │ ┌───────────────────┐ ┌───────────────────────────┐ │
    │ │ Agent Status      │ │ Nexus Signals             │ │
    │ │ ✓ grok (idle)     │ │ DIALOGUE_PROPOSE          │ │
    │ └───────────────────┘ └───────────────────────────┘ │
    ├─────────────────────────────────────────────────────┤
    │ Chat                                                │
    │ You: What's the best approach for caching?          │
    │ [grok] PROPOSE: Redis with consistent hashing...    │
    │ [CONSENSUS]: Redis + local L1 with TTL eviction     │
    ├─────────────────────────────────────────────────────┤
    │ > _                                                 │
    └─────────────────────────────────────────────────────┘
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

from loguru import logger

# Rich imports with graceful fallback
try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.style import Style
    from rich.box import ROUNDED, HEAVY
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.warning("Rich not installed. Run: pip install rich")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PanelType(Enum):
    """Types of panels in the TUI."""
    AGENTS = "agents"
    SIGNALS = "signals"
    CHAT = "chat"
    CANVAS = "canvas"


@dataclass
class AgentDisplay:
    """Display state for an agent."""
    agent_id: str
    status: str = "offline"
    last_activity: str = ""
    response_count: int = 0
    is_responding: bool = False

    def get_icon(self) -> str:
        if self.is_responding:
            return "⟳"
        elif self.status == "online":
            return "✓"
        elif self.status == "busy":
            return "●"
        else:
            return "○"

    def get_color(self) -> str:
        if self.is_responding:
            return "yellow"
        elif self.status == "online":
            return "green"
        elif self.status == "busy":
            return "orange1"
        else:
            return "dim"


@dataclass
class SignalDisplay:
    """Display state for a Nexus signal."""
    signal_type: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    urgency: float = 0.5
    payload_preview: str = ""

    def get_color(self) -> str:
        if self.urgency > 0.7:
            return "red"
        elif self.urgency > 0.4:
            return "yellow"
        else:
            return "dim"


@dataclass
class ChatMessage:
    """A chat message in the display."""
    role: str  # "user", "agent", "system", "consensus"
    content: str
    agent_id: Optional[str] = None
    phase: Optional[str] = None  # "propose", "critique", "refine", "vote"
    timestamp: datetime = field(default_factory=datetime.now)

    def get_prefix(self) -> str:
        if self.role == "user":
            return "You"
        elif self.role == "consensus":
            return "[CONSENSUS]"
        elif self.role == "system":
            return "[system]"
        elif self.agent_id:
            phase_str = f" {self.phase.upper()}" if self.phase else ""
            return f"[{self.agent_id}]{phase_str}"
        return ""

    def get_color(self) -> str:
        if self.role == "user":
            return "cyan"
        elif self.role == "consensus":
            return "green bold"
        elif self.role == "system":
            return "dim"
        else:
            return "white"


# =============================================================================
# RICH CLI
# =============================================================================

class RichCLI:
    """
    Rich terminal UI for Cosmos swarm interaction.

    Provides a visual interface with:
    - Real-time agent status panel
    - Nexus signal feed
    - Chat with deliberation phases
    - Canvas output viewing
    """

    # Agent colors for visual distinction
    AGENT_COLORS = {
        "grok": "blue",
        "claude": "magenta",
        "gemini": "yellow",
        "deepseek": "cyan",
        "kimi": "green",
        "phi": "red",
        "huggingface": "orange1",
        "cosmos": "bright_white",
    }

    def __init__(self, data_dir: str = "./data"):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required. Install with: pip install rich")

        self.data_dir = Path(data_dir)
        self.console = Console()

        # State
        self._running = False
        self._agents: Dict[str, AgentDisplay] = {}
        self._signals: List[SignalDisplay] = []
        self._chat_messages: List[ChatMessage] = []
        self._canvas_figures: List[Any] = []

        # Configuration
        self._max_signals = 20
        self._max_chat_messages = 100
        self._show_agents_panel = True
        self._show_signals_panel = True
        self._show_canvas_panel = False

        # Session integration
        self._swarm_session = None
        self._nexus = None

        # Command handlers
        self._commands: Dict[str, Callable] = {}
        self._register_default_commands()

    def _register_default_commands(self) -> None:
        """Register default CLI commands."""
        self._commands = {
            "help": self._cmd_help,
            "chat": self._cmd_chat,
            "deliberate": self._cmd_deliberate,
            "agents": self._cmd_toggle_agents,
            "signals": self._cmd_toggle_signals,
            "canvas": self._cmd_toggle_canvas,
            "session": self._cmd_session,
            "status": self._cmd_status,
            "clear": self._cmd_clear,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
        }

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def run(self) -> None:
        """Run the Rich CLI."""
        self._running = True

        # Print header
        self._print_header()

        # Initialize swarm session
        await self._init_session()

        # Main command loop
        while self._running:
            try:
                # Get user input
                command_line = Prompt.ask(
                    "[bold cyan]cosmos>[/]",
                    console=self.console,
                )

                if not command_line.strip():
                    continue

                # Parse and execute command
                await self._execute_command(command_line)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use 'exit' or 'quit' to leave.[/]")
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/]")
                logger.error(f"CLI error: {e}")

        # Cleanup
        await self._cleanup()

    async def _init_session(self) -> None:
        """Initialize swarm session."""
        try:
            from Cosmos.cli.swarm_session import create_swarm_session

            self._swarm_session = create_swarm_session()
            await self._swarm_session.connect()

            # Initialize agent displays
            for agent_id in self._swarm_session._active_agents:
                self._agents[agent_id] = AgentDisplay(
                    agent_id=agent_id,
                    status="online",
                )

            self.console.print("[green]✓ Connected to swarm[/]")

        except Exception as e:
            self.console.print(f"[yellow]⚠ Running in offline mode: {e}[/]")
            # Add default agents for display
            for agent_id in ["grok", "claude", "gemini", "deepseek"]:
                self._agents[agent_id] = AgentDisplay(
                    agent_id=agent_id,
                    status="offline",
                )

    async def _cleanup(self) -> None:
        """Cleanup on exit."""
        if self._swarm_session:
            await self._swarm_session.disconnect()
        self.console.print("[dim]Goodbye![/]")

    # =========================================================================
    # COMMAND EXECUTION
    # =========================================================================

    async def _execute_command(self, command_line: str) -> None:
        """Parse and execute a command."""
        parts = command_line.strip().split(maxsplit=1)
        command = parts[0].lower().lstrip("/")
        args = parts[1] if len(parts) > 1 else ""

        # Check for direct chat (no command prefix)
        if command not in self._commands and not command_line.startswith("/"):
            # Treat as chat message
            await self._cmd_chat(command_line)
            return

        # Execute command
        if command in self._commands:
            handler = self._commands[command]
            if asyncio.iscoroutinefunction(handler):
                await handler(args)
            else:
                handler(args)
        else:
            self.console.print(f"[red]Unknown command: {command}[/]")
            self.console.print("[dim]Type 'help' for available commands.[/]")

    # =========================================================================
    # COMMANDS
    # =========================================================================

    def _cmd_help(self, args: str) -> None:
        """Show help."""
        help_text = """
[bold]Cosmos Rich CLI Commands[/]

[cyan]Chat & Deliberation:[/]
  /chat <message>        Send message to swarm (or just type directly)
  /deliberate <prompt>   Start full deliberation on a prompt

[cyan]Display:[/]
  /agents                Toggle agent status panel
  /signals               Toggle Nexus signal feed
  /canvas                Show last canvas output

[cyan]Session:[/]
  /session start|end     Start or end swarm session
  /status                Show session status

[cyan]General:[/]
  /clear                 Clear chat history
  /help                  Show this help
  /exit, /quit           Exit the CLI

[dim]Tip: You can chat without the /chat prefix - just type your message.[/]
"""
        self.console.print(Panel(help_text, title="Help", border_style="blue"))

    async def _cmd_chat(self, message: str) -> None:
        """Chat with the swarm."""
        if not message.strip():
            self.console.print("[dim]Usage: /chat <message> or just type your message[/]")
            return

        # Add user message to display
        self._add_chat_message(ChatMessage(role="user", content=message))

        # Show thinking indicator
        self.console.print("[dim]Thinking...[/]")

        if self._swarm_session and self._swarm_session.is_connected():
            # Stream responses from swarm
            async for response in self._swarm_session.chat(message, stream=True):
                # Update agent display
                if response.agent_id in self._agents:
                    self._agents[response.agent_id].is_responding = False
                    self._agents[response.agent_id].response_count += 1

                # Add to chat
                self._add_chat_message(ChatMessage(
                    role="agent",
                    content=response.content,
                    agent_id=response.agent_id,
                    phase=response.response_type if response.response_type != "final" else None,
                ))

                # Display response
                self._display_agent_response(response.agent_id, response.content)
        else:
            self.console.print("[yellow]Not connected to swarm. Showing offline response.[/]")
            self._add_chat_message(ChatMessage(
                role="system",
                content="Swarm session not connected. Start a session with '/session start'",
            ))

    async def _cmd_deliberate(self, prompt: str) -> None:
        """Start a deliberation."""
        if not prompt.strip():
            self.console.print("[dim]Usage: /deliberate <prompt>[/]")
            return

        self._add_chat_message(ChatMessage(
            role="user",
            content=f"[DELIBERATION] {prompt}",
        ))

        self.console.print(Panel(
            f"[bold]Starting deliberation...[/]\n{prompt}",
            title="Deliberation",
            border_style="yellow",
        ))

        if self._swarm_session and self._swarm_session.is_connected():
            phase_count = {"propose": 0, "critique": 0, "refine": 0, "vote": 0}

            async for response in self._swarm_session.start_deliberation(prompt):
                phase = response.response_type
                if phase in phase_count:
                    phase_count[phase] += 1

                # Display by phase
                if response.is_final:
                    self.console.print(Panel(
                        response.content,
                        title=f"[green]CONSENSUS from {response.agent_id}[/]",
                        border_style="green",
                    ))
                    self._add_chat_message(ChatMessage(
                        role="consensus",
                        content=response.content,
                        agent_id=response.agent_id,
                    ))
                else:
                    self._display_deliberation_phase(
                        response.agent_id,
                        phase,
                        response.content,
                    )
                    self._add_chat_message(ChatMessage(
                        role="agent",
                        content=response.content,
                        agent_id=response.agent_id,
                        phase=phase,
                    ))

            self.console.print(f"\n[dim]Deliberation complete: {phase_count}[/]")
        else:
            self.console.print("[yellow]Not connected to swarm.[/]")

    def _cmd_toggle_agents(self, args: str) -> None:
        """Toggle agent status panel."""
        self._show_agents_panel = not self._show_agents_panel
        status = "shown" if self._show_agents_panel else "hidden"
        self.console.print(f"[dim]Agent panel {status}[/]")

        if self._show_agents_panel:
            self._display_agent_panel()

    def _cmd_toggle_signals(self, args: str) -> None:
        """Toggle signal feed panel."""
        self._show_signals_panel = not self._show_signals_panel
        status = "shown" if self._show_signals_panel else "hidden"
        self.console.print(f"[dim]Signal panel {status}[/]")

        if self._show_signals_panel:
            self._display_signal_panel()

    def _cmd_toggle_canvas(self, args: str) -> None:
        """Toggle/show canvas output."""
        if self._canvas_figures:
            self.console.print(Panel(
                f"Last canvas output ({len(self._canvas_figures)} figures available)",
                title="Canvas",
                border_style="blue",
            ))
            # Note: Actual matplotlib rendering would require additional setup
        else:
            self.console.print("[dim]No canvas output available[/]")

    async def _cmd_session(self, args: str) -> None:
        """Manage swarm session."""
        action = args.strip().lower()

        if action == "start":
            if self._swarm_session and self._swarm_session.is_connected():
                self.console.print("[yellow]Session already active[/]")
            else:
                await self._init_session()

        elif action == "end":
            if self._swarm_session:
                await self._swarm_session.disconnect()
                self.console.print("[green]Session ended[/]")
            else:
                self.console.print("[dim]No active session[/]")

        else:
            self.console.print("[dim]Usage: /session start|end[/]")

    def _cmd_status(self, args: str) -> None:
        """Show session status."""
        if self._swarm_session:
            stats = self._swarm_session.get_stats()
            self._display_status(stats)
        else:
            self.console.print("[dim]No active session[/]")

        # Also show agent panel
        self._display_agent_panel()

    def _cmd_clear(self, args: str) -> None:
        """Clear chat history."""
        self._chat_messages.clear()
        self.console.clear()
        self._print_header()
        self.console.print("[dim]Chat cleared[/]")

    def _cmd_exit(self, args: str) -> None:
        """Exit the CLI."""
        self._running = False

    # =========================================================================
    # DISPLAY HELPERS
    # =========================================================================

    def _print_header(self) -> None:
        """Print the CLI header."""
        agent_count = len(self._agents)
        header = Text()
        header.append("🧠 ", style="bold")
        header.append("Cosmos Swarm CLI", style="bold bright_white")
        header.append(f"  [agents: {agent_count}]", style="dim")

        self.console.print(Panel(
            header,
            border_style="bright_blue",
            box=HEAVY,
        ))

    def _display_agent_panel(self) -> None:
        """Display the agent status panel."""
        table = Table(
            title="Agent Status",
            show_header=True,
            header_style="bold",
            box=ROUNDED,
        )
        table.add_column("Agent", style="cyan")
        table.add_column("Status")
        table.add_column("Responses", justify="right")

        for agent_id, agent in sorted(self._agents.items()):
            color = self.AGENT_COLORS.get(agent_id, "white")
            icon = agent.get_icon()
            status_color = agent.get_color()

            table.add_row(
                f"[{color}]{agent_id}[/]",
                f"[{status_color}]{icon} {agent.status}[/]",
                str(agent.response_count),
            )

        self.console.print(table)

    def _display_signal_panel(self) -> None:
        """Display the Nexus signal feed."""
        if not self._signals:
            self.console.print(Panel(
                "[dim]No signals yet[/]",
                title="Nexus Signals",
                border_style="dim",
            ))
            return

        table = Table(
            title="Nexus Signals",
            show_header=True,
            header_style="bold",
            box=ROUNDED,
        )
        table.add_column("Signal", style="cyan")
        table.add_column("Source")
        table.add_column("Time", style="dim")

        for signal in self._signals[-10:]:
            color = signal.get_color()
            time_str = signal.timestamp.strftime("%H:%M:%S")

            table.add_row(
                f"[{color}]{signal.signal_type}[/]",
                signal.source,
                time_str,
            )

        self.console.print(table)

    def _display_agent_response(
        self,
        agent_id: str,
        content: str,
    ) -> None:
        """Display a response from an agent."""
        color = self.AGENT_COLORS.get(agent_id, "white")

        # Truncate long responses
        display_content = content
        if len(content) > 500:
            display_content = content[:500] + "..."

        self.console.print(Panel(
            display_content,
            title=f"[{color}]{agent_id}[/]",
            border_style=color,
        ))

    def _display_deliberation_phase(
        self,
        agent_id: str,
        phase: str,
        content: str,
    ) -> None:
        """Display a deliberation phase response."""
        color = self.AGENT_COLORS.get(agent_id, "white")

        phase_colors = {
            "propose": "blue",
            "critique": "yellow",
            "refine": "magenta",
            "vote": "green",
        }
        phase_color = phase_colors.get(phase, "white")

        # Truncate for display
        display_content = content[:300] + "..." if len(content) > 300 else content

        self.console.print(Panel(
            display_content,
            title=f"[{color}]{agent_id}[/] [{phase_color}]{phase.upper()}[/]",
            border_style=phase_color,
        ))

    def _display_status(self, stats: Dict[str, Any]) -> None:
        """Display session status."""
        table = Table(
            title="Session Status",
            show_header=False,
            box=ROUNDED,
        )
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        for key, value in stats.items():
            if key == "current_deliberation" and value is None:
                continue
            table.add_row(key, str(value))

        self.console.print(table)

    def _add_chat_message(self, message: ChatMessage) -> None:
        """Add a message to chat history."""
        self._chat_messages.append(message)
        if len(self._chat_messages) > self._max_chat_messages:
            self._chat_messages = self._chat_messages[-self._max_chat_messages:]

    def add_signal(self, signal: SignalDisplay) -> None:
        """Add a signal to the display."""
        self._signals.append(signal)
        if len(self._signals) > self._max_signals:
            self._signals = self._signals[-self._max_signals:]

    def add_canvas_figure(self, figure: Any) -> None:
        """Add a matplotlib figure to canvas."""
        self._canvas_figures.append(figure)

    # =========================================================================
    # NEXUS INTEGRATION
    # =========================================================================

    async def connect_nexus(self, nexus) -> None:
        """Connect to Nexus for signal monitoring."""
        self._nexus = nexus

        # Subscribe to signals for display
        async def on_signal(signal):
            self.add_signal(SignalDisplay(
                signal_type=signal.type.value,
                source=signal.source_id,
                urgency=signal.urgency,
                timestamp=signal.timestamp,
            ))

        # Subscribe to common signals
        from Cosmos.core.nexus import SignalType

        for signal_type in [
            SignalType.DIALOGUE_PROPOSE,
            SignalType.DIALOGUE_CRITIQUE,
            SignalType.DIALOGUE_REFINE,
            SignalType.DIALOGUE_VOTE,
            SignalType.DIALOGUE_CONSENSUS,
            SignalType.CLI_COMMAND,
        ]:
            nexus.subscribe(signal_type, on_signal)


# =============================================================================
# ENTRY POINT
# =============================================================================

async def run_rich_cli(data_dir: str = "./data") -> None:
    """Entry point for running the Rich CLI."""
    if not RICH_AVAILABLE:
        print("Error: Rich library not installed.")
        print("Install with: pip install rich")
        sys.exit(1)

    cli = RichCLI(data_dir=data_dir)
    await cli.run()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cosmos Rich CLI")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    args = parser.parse_args()

    asyncio.run(run_rich_cli(data_dir=args.data_dir))


if __name__ == "__main__":
    main()
