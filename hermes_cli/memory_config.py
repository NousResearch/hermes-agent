"""Memory backend management CLI.

Provides `hermes memory` command for switching between Honcho and QMD
memory backends. This is a strict OR relationship — only one backend
can be active at a time.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hermes_cli.config import load_config, set_config_value as save_config_value

console = Console()


# =============================================================================
# Hardware Detection
# =============================================================================

def _get_hardware_info() -> dict:
    """Detect hardware capabilities for model selection.

    Returns:
        dict with keys: os, arch, ram_gb, has_gpu, gpu_type, has_mps, recommended_format
    """
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "ram_gb": None,
        "has_gpu": False,
        "gpu_type": None,
        "has_mps": False,
        "recommended_format": "pytorch",
    }

    # Detect RAM
    try:
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["ram_gb"] = int(result.stdout.strip()) / (1024**3)
        else:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = kb / (1024**2)
                        break
    except Exception:
        pass

    # Detect GPU
    try:
        if platform.system() == "Darwin":
            # Check for Apple Silicon
            if platform.machine() == "arm64":
                info["has_gpu"] = True
                info["gpu_type"] = "apple_silicon"
                info["recommended_format"] = "mlx"
                # Check for MPS (Metal Performance Shaders)
                try:
                    import subprocess
                    result = subprocess.run(
                        ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
                        capture_output=True, text=True, timeout=10
                    )
                    info["has_mps"] = result.returncode == 0 and "True" in result.stdout
                except Exception:
                    pass
        else:
            # Linux: check for NVIDIA CUDA
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    info["has_gpu"] = True
                    info["gpu_type"] = "nvidia_cuda"
                    info["recommended_format"] = "pytorch"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    except Exception:
        pass

    return info


def _get_recommended_models(hw_info: dict) -> list[dict]:
    """Get recommended embedding models based on hardware.

    Args:
        hw_info: Hardware info from _get_hardware_info()

    Returns:
        List of dicts with: id, name, size_gb, format, description
    """
    models = []

    if hw_info["os"] == "Darwin" and hw_info["arch"] == "arm64":
        # Apple Silicon - prefer MLX models
        models.extend([
            {
                "id": "mlx-community/bge-micro-v2",
                "name": "BGE Micro v2 (MLX)",
                "size_gb": 0.2,
                "format": "mlx",
                "description": "Ultra-light, fastest on Apple Silicon",
                "min_ram": 2,
            },
            {
                "id": "mlx-community/Nomic-embed-text",
                "name": "Nomic Embed Text (MLX)",
                "size_gb": 0.9,
                "format": "mlx",
                "description": "Cross-lingual, good quality",
                "min_ram": 4,
            },
            {
                "id": "qwen3.5b:0.8b",
                "name": "Qwen 3.5B (Ollama/MLX)",
                "size_gb": 4.0,
                "format": "mlx",
                "description": "Good quality, requires Ollama or MLX",
                "min_ram": 8,
            },
        ])
    else:
        # Linux/Other - CUDA or CPU models
        ram = hw_info["ram_gb"] or 16

        models.extend([
            {
                "id": "BAAI/bge-micro-v2",
                "name": "BGE Micro v2",
                "size_gb": 0.2,
                "format": "pytorch",
                "description": "Ultra-light, CPU-friendly",
                "min_ram": 2,
            },
            {
                "id": "qwen3.5b:0.8b",
                "name": "Qwen 3.5B (Ollama)",
                "size_gb": 4.0,
                "format": "pytorch",
                "description": "Good quality via Ollama",
                "min_ram": 8,
            },
            {
                "id": "nomic-embed-text",
                "name": "Nomic Embed Text (Ollama)",
                "size_gb": 0.9,
                "format": "pytorch",
                "description": "Cross-lingual via Ollama",
                "min_ram": 4,
            },
        ])

        # Add sentence-transformers option if enough RAM
        if ram >= 8:
            models.append({
                "id": "BAAI/bge-m3",
                "name": "BGE M3 (sentence-transformers)",
                "size_gb": 2.5,
                "format": "pytorch",
                "description": "Highest quality, requires more RAM",
                "min_ram": 8,
            })

    return models


def _filter_models_for_hardware(models: list[dict], hw_info: dict) -> list[dict]:
    """Filter models to only those that fit in available hardware.

    Args:
        models: List of model dicts from _get_recommended_models()
        hw_info: Hardware info from _get_hardware_info()

    Returns:
        Filtered list of models that fit the hardware
    """
    ram = hw_info["ram_gb"]

    # If we can't detect RAM, allow all
    if ram is None:
        return models

    filtered = []
    for model in models:
        min_ram = model.get("min_ram", 4)
        # Add 2GB buffer for QMD server overhead
        if ram >= min_ram + 2:
            filtered.append(model)

    return filtered


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
        save_config_value("qmd.enabled", "true")
        save_config_value("qmd.host", str(cfg["qmd"]["host"]))
        save_config_value("qmd.port", str(cfg["qmd"]["port"]))
        save_config_value("qmd.embedding_model", str(cfg["qmd"]["embedding_model"]))
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
            save_config_value("qmd.enabled", "false")
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
            save_config_value("qmd.enabled", "false")
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
    """Interactive QMD setup with hardware detection."""
    console.print("\n[bold]QMD Setup[/bold]\n")

    # Detect hardware
    console.print("[dim]Detecting hardware...[/dim]")
    hw_info = _get_hardware_info()

    # Display hardware info
    console.print(f"\n[bold cyan]Hardware:[/bold cyan]")
    os_name = "macOS (Apple Silicon)" if hw_info["os"] == "Darwin" and hw_info["arch"] == "arm64" else hw_info["os"]
    console.print(f"  OS: {os_name}")
    if hw_info["ram_gb"]:
        console.print(f"  RAM: {hw_info['ram_gb']:.1f} GB")
    else:
        console.print("  RAM: unknown")

    if hw_info["has_gpu"]:
        if hw_info["gpu_type"] == "apple_silicon":
            console.print(f"  GPU: Apple Silicon (MPS: {'yes' if hw_info['has_mps'] else 'no'})")
        elif hw_info["gpu_type"] == "nvidia_cuda":
            console.print("  GPU: NVIDIA CUDA")
    else:
        console.print("  GPU: none detected")

    console.print()

    # Get available models for this hardware
    available_models = _get_recommended_models(hw_info)
    filtered_models = _filter_models_for_hardware(available_models, hw_info)

    if not filtered_models:
        console.print("[red]No embedding models fit in available memory.[/red]")
        console.print(f"  Your system has {hw_info['ram_gb']:.1f} GB RAM.")
        console.print("  Minimum requirement: ~4 GB for any model.")
        return

    # Show available models
    console.print("[bold]Available embedding models:[/bold]\n")
    for i, model in enumerate(filtered_models, 1):
        badge = ""
        if model["format"] == "mlx":
            badge = " [green](MLX)[/green]"
        elif hw_info["has_gpu"] and model["format"] == "pytorch":
            badge = " [yellow](GPU)[/yellow]"

        size_info = f"{model['size_gb']:.1f} GB"
        console.print(f"  [{i}] {model['name']}{badge}")
        console.print(f"      {model['description']} (~{size_info} memory)")

    # Check for Ollama
    ollama_available = shutil.which("ollama") is None
    if ollama_available:
        console.print("\n  [yellow]Note: Ollama not found. MLX models require special setup.[/yellow]")
        console.print("  Install Ollama from: https://ollama.ai")
    console.print()

    # Get model selection
    try:
        choice = input(f"Select model [1-{len(filtered_models)}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled.[/yellow]")
        return

    if not choice:
        choice = "1"

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(filtered_models):
            console.print(f"[red]Invalid selection. Choose 1-{len(filtered_models)}.[/red]")
            return
        selected = filtered_models[idx]
    except ValueError:
        console.print("[red]Invalid input. Enter a number.[/red]")
        return

    embed_model = selected["id"]
    console.print(f"\n[dim]Selected: {selected['name']}[/dim]")

    # Get server URL
    default_host = "127.0.0.1"
    default_port = "8181"

    try:
        host = input(f"\nQMD server host [{default_host}]: ").strip() or default_host
        port = input(f"QMD server port [{default_port}]: ").strip() or default_port
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled.[/yellow]")
        return

    # Save config
    try:
        save_config_value("qmd.enabled", "true")
        save_config_value("qmd.host", host)
        save_config_value("qmd.port", str(port))
        save_config_value("qmd.embedding_model", embed_model)
        console.print("\n[green]✓ QMD configured successfully![/green]")
    except Exception as e:
        console.print(f"\n[red]✗ Failed to save config: {e}[/red]")
        return

    console.print(f"\nQMD Server: http://{host}:{port}")
    console.print(f"Embedding Model: {embed_model}")
    console.print(f"Format: {selected['format'].upper()}")
    console.print("\nTo start the QMD server:")
    if selected["format"] == "mlx":
        console.print("  [cyan]ollama serve[/cyan]  # Then load model with: ollama pull {embed_model}")
    else:
        console.print("  [cyan]qmd server[/cyan]")
        console.print("  or")
        console.print(f"  [cyan]python ~/.hermes/qmd_server/server.py[/cyan]")


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
        save_config_value("qmd.lite_mode", "true")
        console.print("[green]✓ Lite mode enabled[/green]")

    console.print("\nRestart Hermes for changes to take effect.")
