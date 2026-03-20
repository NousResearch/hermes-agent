"""Memory backend management CLI.

Provides `hermes memory` command for switching between Honcho and QMD
memory backends. This is a strict OR relationship — only one backend
can be active at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hermes_cli.config import load_config, save_config_value

console = Console()


def memory_command(args):
    """Dispatch to the appropriate memory subcommand."""
    cmd = getattr(args, "memory_command", None)

    if cmd == "status":
        return cmd_status(args)
    elif cmd == "mode":
        return cmd_mode(args)
    elif cmd == "setup":
        return cmd_setup(args)
    elif cmd == "config":
        return cmd_config(args)
    else:
        # No subcommand -- show status by default
        return cmd_status(args)


def _load_hermes_config() -> dict:
    """Load Hermes config from ~/.hermes/config.yaml."""
    try:
        return load_config()
    except Exception as e:
        console.print(f"[red]Failed to load config: {e}[/red]")
        return {}


def _get_memory_backend(cfg: dict) -> str:
    """Determine which memory backend is currently active."""
    qmd = cfg.get("qmd", {})
    if qmd and qmd.get("enabled"):
        return "qmd"

    # Check if Honcho is enabled
    try:
        from honcho_integration.client import HonchoClientConfig
        hcfg = HonchoClientConfig.from_global_config()
        if hcfg.enabled and hcfg.api_key:
            return "honcho"
    except Exception:
        pass

    return "off"


def _get_backend_details(cfg: dict) -> dict:
    """Get detailed configuration for the active backend."""
    backend = _get_memory_backend(cfg)
    details = {"backend": backend}

    if backend == "qmd":
        qmd = cfg.get("qmd", {})
        details.update({
            "server": f"{qmd.get('host', '127.0.0.1')}:{qmd.get('port', 8181)}",
            "embedding_model": qmd.get("embedding_model", "qwen3.5b:0.8b"),
            "lite_mode": qmd.get("lite_mode", False),
            "index_name": qmd.get("index_name", "qmd_memory"),
            "write_frequency": qmd.get("write_frequency", "async"),
            "anticipatory": qmd.get("anticipatory_enabled", True),
        })
    elif backend == "honcho":
        try:
            from honcho_integration.client import HonchoClientConfig
            hcfg = HonchoClientConfig.from_global_config()
            details.update({
                "workspace": hcfg.workspace_id,
                "peer_name": hcfg.peer_name or "(from config)",
                "ai_peer": hcfg.ai_peer,
                "memory_mode": hcfg.memory_mode,
                "write_frequency": hcfg.write_frequency,
            })
        except Exception as e:
            details["error"] = str(e)

    return details


def cmd_status(args):
    """Show current memory backend and configuration."""
    cfg = _load_hermes_config()
    backend = _get_memory_backend(cfg)
    details = _get_backend_details(cfg)

    console.print(f"\n[bold]Memory Backend:[/bold] {backend.upper()}")

    if backend == "qmd":
        table = Table(title="QMD Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Server", details.get("server", "localhost:8181"))
        table.add_row("Embedding Model", details.get("embedding_model", "qwen3.5b:0.8b"))
        table.add_row("Lite Mode", "Yes" if details.get("lite_mode") else "No")
        table.add_row("Index", details.get("index_name", "qmd_memory"))
        table.add_row("Write Frequency", details.get("write_frequency", "async"))
        table.add_row("Anticipatory Context", "Enabled" if details.get("anticipatory") else "Disabled")
        console.print(table)

        # Check if QMD server is running
        try:
            import httpx
            server = details.get("server", "localhost:8181")
            if not server.startswith("http"):
                server = f"http://{server}"
            resp = httpx.get(f"{server}/status", timeout=5)
            if resp.status_code == 200:
                status = resp.json()
                console.print(f"[green]✓ QMD server is running[/green]")
                console.print(f"  Model: {status.get('model', 'unknown')}")
                console.print(f"  Memories: {status.get('memory_count', 0)}")
            else:
                console.print(f"[yellow]⚠ QMD server returned status {resp.status_code}[/yellow]")
        except Exception:
            console.print("[red]✗ QMD server is not running[/red]")
            console.print("  Start with: [cyan]qmd server[/cyan] or [cyan]python ~/.hermes/qmd_server/server.py[/cyan]")

    elif backend == "honcho":
        table = Table(title="Honcho Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Workspace", details.get("workspace", "unknown"))
        table.add_row("Peer Name", details.get("peer_name", "unknown"))
        table.add_row("AI Peer", details.get("ai_peer", "unknown"))
        table.add_row("Memory Mode", details.get("memory_mode", "unknown"))
        table.add_row("Write Frequency", details.get("write_frequency", "async"))
        console.print(table)

    else:
        console.print("\n[dim]No external memory backend active.[/dim]")
        console.print("  Using local MEMORY.md for session memory only.")
        console.print("\nTo enable a memory backend:")
        console.print("  [cyan]hermes memory mode qmd[/cyan]  - Enable QMD (local, no cloud)")
        console.print("  [cyan]hermes memory mode honcho[/cyan] - Enable Honcho (cloud-backed)")
        console.print("  [cyan]hermes memory setup[/cyan]       - Interactive setup")

    console.print()


def cmd_mode(args):
    """Show or set the memory backend."""
    cfg = _load_hermes_config()
    backend = getattr(args, "backend", None)

    if not backend:
        # Show current mode
        current = _get_memory_backend(cfg)
        console.print(f"Current memory backend: [bold]{current}[/bold]")
        console.print("\nAvailable backends:")
        console.print("  [cyan]honcho[/cyan] - Cloud-backed cross-session memory (honcho.dev)")
        console.print("  [cyan]qmd[/cyan]    - Local vector-based memory with FlowState")
        console.print("  [cyan]off[/cyan]    - Local MEMORY.md only (no cross-session)")
        return

    # Set the backend
    if backend == "qmd":
        _enable_qmd(cfg)
    elif backend == "honcho":
        _enable_honcho(cfg)
    elif backend == "off":
        _disable_memory(cfg)


def _enable_qmd(cfg: dict):
    """Enable QMD as the memory backend."""
    console.print("\n[bold]Enabling QMD memory backend...[/bold]")

    # Check if qmd section exists
    if "qmd" not in cfg:
        cfg["qmd"] = {}

    # Enable QMD
    cfg["qmd"]["enabled"] = True

    # Ensure required fields
    if "host" not in cfg["qmd"]:
        cfg["qmd"]["host"] = "127.0.0.1"
    if "port" not in cfg["qmd"]:
        cfg["qmd"]["port"] = 8181
    if "embedding_model" not in cfg["qmd"]:
        cfg["qmd"]["embedding_model"] = "qwen3.5b:0.8b"

    try:
        save_config_value("qmd.enabled", True)
        save_config_value("qmd.host", cfg["qmd"]["host"])
        save_config_value("qmd.port", cfg["qmd"]["port"])
        save_config_value("qmd.embedding_model", cfg["qmd"]["embedding_model"])
        console.print("[green]✓ QMD enabled in config.yaml[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to save config: {e}[/red]")
        return

    console.print("\nQMD is now enabled with settings:")
    console.print(f"  Server: http://{cfg['qmd']['host']}:{cfg['qmd']['port']}")
    console.print(f"  Embedding: {cfg['qmd']['embedding_model']}")
    console.print("\nTo start the QMD server:")
    console.print("  [cyan]qmd server[/cyan]")
    console.print("\nNote: Honcho has been disabled.")


def _enable_honcho(cfg: dict):
    """Enable Honcho as the memory backend."""
    console.print("\n[bold]Enabling Honcho memory backend...[/bold]")

    # Disable QMD in config
    if "qmd" in cfg:
        try:
            save_config_value("qmd.enabled", False)
            console.print("[dim]QMD disabled in config.yaml[/dim]")
        except Exception:
            pass

    # Check if Honcho is configured
    try:
        from honcho_integration.client import HonchoClientConfig
        hcfg = HonchoClientConfig.from_global_config()

        if not hcfg.api_key:
            console.print("[yellow]Honcho API key not found.[/yellow]")
            console.print("\nTo set up Honcho:")
            console.print("  [cyan]hermes honcho setup[/cyan]")
            return

        console.print(f"[green]✓ Honcho is configured[/green]")
        console.print(f"  Workspace: {hcfg.workspace_id}")
        console.print(f"  Peer: {hcfg.peer_name or 'hermes'}")
        console.print("\nHoncho will be active on next Hermes restart.")

    except ImportError:
        console.print("[red]honcho-ai package not installed[/red]")
        console.print("  Install with: [cyan]pip install honcho-ai[/cyan]")
    except Exception as e:
        console.print(f"[red]Honcho configuration error: {e}[/red]")


def _disable_memory(cfg: dict):
    """Disable both Honcho and QMD."""
    console.print("\n[bold]Disabling external memory backends...[/bold]")

    # Disable QMD
    if "qmd" in cfg:
        try:
            save_config_value("qmd.enabled", False)
            console.print("[dim]QMD disabled[/dim]")
        except Exception:
            pass

    console.print("[green]✓ External memory backends disabled[/green]")
    console.print("\nHermes will use local MEMORY.md for session memory only.")
    console.print("Cross-session memory features will not be available.")


def cmd_setup(args):
    """Interactive setup for memory backends."""
    backend = getattr(args, "backend", None)

    if not backend:
        console.print("\n[bold]Memory Backend Setup[/bold]\n")
        console.print("Select a memory backend to set up:\n")
        console.print("  [1] [cyan]QMD[/cyan] - Local vector memory with FlowState anticipatory context")
        console.print("      No cloud required, fast local retrieval")
        console.print("      [dim]Default embedding: qwen3.5b:0.8b[/dim]\n")
        console.print("  [2] [cyan]Honcho[/cyan] - Cloud-backed cross-session user modeling")
        console.print("      Requires honcho.dev account and API key\n")
        console.print("  [3] [cyan]Cancel[/cyan]\n")

        try:
            choice = input("Enter choice [1-3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Setup cancelled.[/yellow]")
            return

        if choice == "1":
            backend = "qmd"
        elif choice == "2":
            backend = "honcho"
        else:
            console.print("[yellow]Setup cancelled.[/yellow]")
            return

    if backend == "qmd":
        _setup_qmd_interactive()
    elif backend == "honcho":
        _setup_honcho_interactive()


def _setup_qmd_interactive():
    """Interactive QMD setup."""
    console.print("\n[bold]QMD Setup[/bold]\n")
    console.print("QMD is a local vector-based memory system.\n")

    # Get server URL
    default_host = "127.0.0.1"
    default_port = "8181"

    try:
        host = input(f"QMD server host [{default_host}]: ").strip() or default_host
        port = input(f"QMD server port [{default_port}]: ").strip() or default_port
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled.[/yellow]")
        return

    # Get embedding model
    console.print("\nEmbedding models:")
    console.print("  [1] qwen3.5b:0.8b (default, fast local model)")
    console.print("  [2] nomic-embed-text (cross-lingual, larger)")
    console.print("  [3] BAAI/bge-m3 (sentence-transformers, requires API)")

    try:
        choice = input("Select embedding model [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled.[/yellow]")
        return

    embed_model = {
        "1": "qwen3.5b:0.8b",
        "2": "nomic-embed-text",
        "3": "BAAI/bge-m3",
    }.get(choice, "qwen3.5b:0.8b")

    # Save config
    try:
        save_config_value("qmd.enabled", True)
        save_config_value("qmd.host", host)
        save_config_value("qmd.port", int(port))
        save_config_value("qmd.embedding_model", embed_model)
        console.print("\n[green]✓ QMD configured successfully![/green]")
    except Exception as e:
        console.print(f"\n[red]✗ Failed to save config: {e}[/red]")
        return

    console.print(f"\nQMD Server: http://{host}:{port}")
    console.print(f"Embedding Model: {embed_model}")
    console.print("\nTo start the QMD server:")
    console.print("  [cyan]qmd server[/cyan]")
    console.print("  or")
    console.print("  [cyan]python ~/.hermes/qmd_server/server.py[/cyan]")


def _setup_honcho_interactive():
    """Interactive Honcho setup."""
    console.print("\n[bold]Honcho Setup[/bold]\n")
    console.print("Honcho requires an API key from honcho.dev\n")

    try:
        api_key = input("Enter Honcho API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled.[/yellow]")
        return

    if not api_key:
        console.print("[yellow]No API key entered. Setup cancelled.[/yellow]")
        return

    # Save to .env
    from pathlib import Path
    env_path = Path.home() / ".hermes" / ".env"
    env_path.parent.mkdir(exist_ok=True)

    # Read existing .env
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                existing[key.strip()] = val.strip()

    existing["HONCHO_API_KEY"] = api_key
    env_path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")

    console.print("\n[green]✓ Honcho API key saved to ~/.hermes/.env[/green]")
    console.print("\nTo complete Honcho setup, configure your workspace:")
    console.print("  [cyan]hermes honcho setup[/cyan]")


def cmd_config(args):
    """Show or update memory configuration."""
    cfg = _load_hermes_config()
    show = getattr(args, "show", False)
    embed_model = getattr(args, "embedding_model", None)
    server = getattr(args, "server", None)
    lite_mode = getattr(args, "lite_mode", False)

    # If no args, show current config
    if not any([show, embed_model, server, lite_mode]):
        return cmd_status(args)

    backend = _get_memory_backend(cfg)

    if backend != "qmd":
        console.print(f"[yellow]Configuration options only apply to QMD backend (current: {backend})[/yellow]")
        return

    if "qmd" not in cfg:
        cfg["qmd"] = {"enabled": True}

    if embed_model:
        save_config_value("qmd.embedding_model", embed_model)
        console.print(f"[green]✓ Embedding model set to {embed_model}[/green]")

    if server:
        # Parse URL
        if server.startswith("http"):
            from urllib.parse import urlparse
            parsed = urlparse(server)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8181
        else:
            host = server
            port = 8181

        save_config_value("qmd.host", host)
        save_config_value("qmd.port", port)
        console.print(f"[green]✓ Server URL set to http://{host}:{port}[/green]")

    if lite_mode:
        save_config_value("qmd.lite_mode", True)
        console.print("[green]✓ Lite mode enabled[/green]")

    console.print("\nRestart Hermes for changes to take effect.")
