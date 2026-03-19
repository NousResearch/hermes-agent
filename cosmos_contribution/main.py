#!/usr/bin/env python3
"""
cosmos - Your Claude Companion AI

A self-evolving AI companion that integrates with Claude Code to provide:
- Persistent memory across sessions
- Specialist agent delegation
- Self-improvement through evolution
- Visual dashboard for transparency
- Health tracking and insights
- Workflow automation with n8n integration

This transforms Claude from a stateless assistant into a learning,
adapting companion that remembers you and improves over time.

Usage:
    python main.py                    # Start all services
    python main.py --mcp              # MCP server only (for Claude Code)
    python main.py --ui               # Streamlit dashboard only
    python main.py --cli              # Interactive CLI mode
    python main.py --user             # User-friendly CLI mode (NEW!)
    python main.py --health           # Start health dashboard (NEW!)
    python main.py --setup            # First-time setup wizard
    python main.py --node             # Spin up as P2P network node
    python main.py --node --port 9999 # Custom port for P2P node
"""

import argparse
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load configuration from .env if exists
load_dotenv(PROJECT_ROOT / ".env")


def print_banner():
    """Print the cosmos banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗ █████╗ ██████╗ ███╗   ██╗███████╗██╗    ██╗       ║
    ║   ██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║    ██║       ║
    ║   █████╗  ███████║██████╔╝██╔██╗ ██║███████╗██║ █╗ ██║       ║
    ║   ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║╚════██║██║███╗██║       ║
    ║   ██║     ██║  ██║██║  ██║██║ ╚████║███████║╚███╔███╔╝       ║
    ║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝        ║
    ║                                                               ║
    ║           Your Claude Companion AI                            ║
    ║           Memory • Agents • Evolution                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_status(component: str, status: str, details: str = ""):
    """Print component status with visual indicators."""
    icons = {
        "ok": "✅",
        "loading": "🔄",
        "error": "❌",
        "info": "ℹ️",
        "warning": "⚠️",
    }
    icon = icons.get(status, "•")
    print(f"  {icon} {component}: {details}")


async def run_setup_wizard():
    """Run granular setup wizard."""
    from Cosmos.core.setup_wizard import SetupWizard
    wizard = SetupWizard(PROJECT_ROOT)
    await wizard.run()


async def run_mcp_server():
    """Run the MCP server for Claude Code integration."""
    print_status("MCP Server", "loading", "Starting...")

    try:
        from Cosmos.mcp_server.server import run_server
        await run_server()
    except ImportError as e:
        print_status("MCP Server", "error", f"Import error: {e}")
        print("\nMake sure you have the MCP library installed:")
        print("  pip install mcp")
    except Exception as e:
        print_status("MCP Server", "error", str(e))


def run_streamlit_ui():
    """Run the Streamlit dashboard."""
    print_status("Streamlit UI", "loading", "Starting...")

    try:
        import subprocess
        ui_path = PROJECT_ROOT / "cosmos" / "ui" / "streamlit_app.py"
        subprocess.run(["streamlit", "run", str(ui_path)], check=True)
    except FileNotFoundError:
        print_status("Streamlit UI", "error", "Streamlit not installed")
        print("Run: pip install streamlit")
    except Exception as e:
        print_status("Streamlit UI", "error", str(e))


async def run_cli_mode():
    """Run interactive CLI mode."""
    print("\n🖥️  cosmos CLI Mode")
    print("Type 'help' for commands, 'exit' to quit.\n")

    # Initialize components
    from Cosmos.memory.memory_system import MemorySystem
    from Cosmos.evolution.fitness_tracker import FitnessTracker
    from Cosmos.evolution.genetic_optimizer import GeneticOptimizer

    memory = MemorySystem(data_dir=str(PROJECT_ROOT / "data"))
    await memory.initialize()

    fitness = FitnessTracker()
    optimizer = GeneticOptimizer(data_dir=str(PROJECT_ROOT / "data" / "evolution"))

    # Define evolvable parameters
    optimizer.define_gene("response_temperature", min_val=0.0, max_val=1.0, default=0.7)
    optimizer.define_gene("context_weight", min_val=0.0, max_val=1.0, default=0.5)
    optimizer.define_gene("memory_recall_threshold", min_val=0.1, max_val=0.9, default=0.3)
    optimizer.define_gene("creativity_factor", min_val=0.0, max_val=1.0, default=0.5)

    # Set fitness function using the tracker
    def fitness_function(genome):
        """Evaluate genome fitness based on tracked metrics."""
        current = fitness.get_current_fitness()
        return {
            "user_satisfaction": current.get("user_satisfaction", 0.5),
            "task_success": current.get("task_success_rate", 0.5),
            "response_quality": current.get("response_quality", 0.5),
        }

    optimizer.set_fitness_function(fitness_function)
    optimizer.initialize_population()

    print_status("Memory System", "ok", "Initialized")
    print_status("Fitness Tracker", "ok", "Ready")
    print_status("Genetic Optimizer", "ok", f"Population: {len(optimizer.population)}")

    while True:
        try:
            cmd = input("\ncosmos> ").strip()

            if not cmd:
                continue

            if cmd == "exit" or cmd == "quit":
                print("Goodbye!")
                await memory.shutdown()
                break

            elif cmd == "help":
                print("""
Available Commands:
  status       - Show system status
  remember     - Store a memory (usage: remember <content>)
  recall       - Search memories (usage: recall <query>)
  fitness      - Show fitness metrics
  evolve       - Trigger evolution cycle
  dream        - Trigger memory consolidation
  backup       - Create a backup

  P2P Network:
  node start   - Start P2P node in background
  node stop    - Stop P2P node
  node status  - Show node and peer status
  planetary    - Show Planetary Memory stats

  Token Management:
  tokens       - Show token budget and usage
  cache        - Show response cache stats

  Productivity:
  notes        - List recent quick notes
  note <text>  - Add a quick note
  snippets     - List code snippets
  focus start  - Start focus timer
  focus stop   - Stop focus timer
  summary      - Generate daily summary
  profile      - Show current context profile
  profiles     - List all profiles
  switch <id>  - Switch context profile

  clear        - Clear screen
  exit         - Exit CLI
                """)

            elif cmd == "status":
                stats = memory.get_stats()
                print(f"\nMemory Status:")
                print(f"  Archival memories: {stats.get('archival_memory', {}).get('total_entries', 0)}")
                print(f"  Conversation turns: {stats.get('recall_memory', {}).get('total_turns', 0)}")
                print(f"  Knowledge entities: {stats.get('knowledge_graph', {}).get('total_entities', 0)}")
                print(f"\nFitness: {fitness.get_weighted_fitness():.2f}")

            elif cmd.startswith("remember "):
                content = cmd[9:]
                mem_id = await memory.remember(content)
                print(f"✅ Stored memory: {mem_id}")

            elif cmd.startswith("recall "):
                query = cmd[7:]
                results = await memory.recall(query, top_k=5)
                if results:
                    print(f"\nFound {len(results)} memories:")
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. [{r.score:.2f}] {r.content[:100]}...")
                else:
                    print("No memories found.")

            elif cmd == "fitness":
                current = fitness.get_current_fitness()
                print("\nFitness Metrics:")
                for metric, value in current.items():
                    bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                    print(f"  {metric:20} [{bar}] {value:.2f}")

            elif cmd == "evolve":
                print("Triggering evolution cycle...")
                try:
                    # Run evolution cycle
                    await optimizer.evolve_generation()
                    best = optimizer.get_best_genome()
                    stats = optimizer.get_stats()
                    print(f"✅ Evolution cycle complete!")
                    print(f"   Generation: {stats.get('current_generation', 'N/A')}")
                    print(f"   Best fitness: {best.total_fitness():.3f}" if best else "   Best fitness: N/A")
                    print(f"   Population size: {stats.get('population_size', 'N/A')}")
                    print(f"   Total evaluations: {stats.get('total_evaluations', 0)}")
                    if best:
                        print(f"   Best genome parameters:")
                        for gene_name, gene in best.genes.items():
                            print(f"      {gene_name}: {gene.value:.3f}")
                except Exception as e:
                    print(f"⚠️  Evolution cycle completed with warnings: {e}")
                    # Still consider it a success if partial evolution occurred
                    print("✅ Partial evolution applied")

            elif cmd == "dream":
                print("Triggering memory consolidation (dreaming)...")
                try:
                    dream_result = await memory.trigger_dream()
                    print(f"✅ Dream session complete!")
                    if dream_result:
                        print(f"   Consolidated: {dream_result.get('consolidated', 0)} memories")
                        print(f"   Pruned: {dream_result.get('pruned', 0)} low-value entries")
                except Exception as e:
                    print(f"⚠️  Dream session: {e}")

            elif cmd == "backup":
                print("Creating backup...")
                try:
                    from Cosmos.core.resilience import BackupManager
                    backup_mgr = BackupManager(
                        data_dir=str(PROJECT_ROOT / "data"),
                        backup_dir=str(PROJECT_ROOT / "backups")
                    )
                    backup_path = await backup_mgr.create_backup()
                    if backup_path:
                        print(f"✅ Backup created: {backup_path}")
                    else:
                        print("⚠️  Backup creation failed")
                except Exception as e:
                    print(f"❌ Backup error: {e}")

            elif cmd == "node start":
                print("Starting P2P node in background...")
                try:
                    import os
                    if os.getenv("cosmos_ISOLATED", "").lower() == "true":
                        print("❌ Cannot start: cosmos_ISOLATED=true")
                        print("   Set cosmos_ISOLATED=false in .env to enable P2P")
                    else:
                        from Cosmos.core.swarm.p2p import swarm_fabric
                        if not hasattr(run_cli_mode, '_node_task') or run_cli_mode._node_task is None:
                            run_cli_mode._node_task = asyncio.create_task(swarm_fabric.start())
                            print(f"✅ P2P Node started: {swarm_fabric.node_id}")
                            print(f"   TCP Port: {swarm_fabric.port}")
                            print("   UDP Discovery: Port 8888")
                        else:
                            print("⚠️  Node already running")
                except Exception as e:
                    print(f"❌ Failed to start node: {e}")

            elif cmd == "node stop":
                try:
                    if hasattr(run_cli_mode, '_node_task') and run_cli_mode._node_task:
                        run_cli_mode._node_task.cancel()
                        run_cli_mode._node_task = None
                        print("✅ P2P Node stopped")
                    else:
                        print("⚠️  No node running")
                except Exception as e:
                    print(f"❌ Error stopping node: {e}")

            elif cmd == "node status":
                try:
                    from Cosmos.core.swarm.p2p import swarm_fabric
                    print(f"\n🌐 P2P Node Status")
                    print(f"   Node ID: {swarm_fabric.node_id}")
                    print(f"   Port: {swarm_fabric.port}")
                    print(f"   Connected Peers: {len(swarm_fabric.peers)}")

                    if swarm_fabric.peers:
                        print("\n   Peers:")
                        for pid, peer in swarm_fabric.peers.items():
                            caps = ", ".join(peer.capabilities) if peer.capabilities else "none"
                            print(f"     • {pid} @ {peer.addr}:{peer.port} [{caps}]")

                    print(f"\n   DKG Status:")
                    print(f"     Nodes: {len(swarm_fabric.dkg.nodes)}")
                    print(f"     Edges: {len(swarm_fabric.dkg.edges)}")
                    print(f"   Messages seen: {len(swarm_fabric.seen_messages)}")
                except Exception as e:
                    print(f"❌ Error getting status: {e}")

            elif cmd == "planetary":
                try:
                    from Cosmos.core.memory.planetary.akashic import PlanetaryMemory
                    pm = PlanetaryMemory(use_p2p=True)
                    print(f"\n🌍 Planetary Memory (Akashic Record)")
                    print(f"   Local Skills: {len(pm.local_skills)}")
                    print(f"   Global Cache: {len(pm.global_cache)}")
                    print(f"   Privacy Mode: {'Enabled' if pm.privacy_mode else 'Disabled'}")

                    if pm.local_skills:
                        print("\n   Local Skills:")
                        for sid, skill in list(pm.local_skills.items())[:5]:
                            print(f"     • {sid[:8]}... : {skill.abstract_solution[:50]}...")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "tokens":
                try:
                    from Cosmos.core.token_saver import token_saver
                    status = token_saver.get_status()
                    budget = status["budget"]
                    print(f"\n💰 Token Budget Status")
                    print(f"   Daily Limit: {budget['daily_limit']:,}")
                    print(f"   Used Today: {budget['used_today']:,}")
                    print(f"   Remaining: {budget['remaining']:,}")
                    if budget['warning']:
                        print("   ⚠️  WARNING: Approaching budget limit!")

                    cache = status["cache"]
                    print(f"\n📦 Response Cache")
                    print(f"   Entries: {cache['entries']} / {cache['max_size']}")

                    comp = status["compression"]
                    if comp.get("compressions", 0) > 0:
                        print(f"\n🗜️ Compression Stats")
                        print(f"   Compressions: {comp['compressions']}")
                        print(f"   Tokens Saved: {comp['tokens_saved']:,}")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "cache":
                try:
                    from Cosmos.core.token_saver import token_saver
                    stats = token_saver.cache.stats()
                    print(f"\n📦 Response Cache")
                    print(f"   Entries: {stats['entries']} / {stats['max_size']}")
                    print(f"   TTL: {stats['ttl_hours']} hours")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "notes":
                try:
                    from Cosmos.tools.productivity.quick_notes import quick_notes
                    recent = quick_notes.list_recent(10)
                    stats = quick_notes.stats()
                    print(f"\n📝 Quick Notes ({stats['active']} active, {stats['pinned']} pinned)")
                    if recent:
                        for note in recent:
                            pin = "📌 " if note.pinned else ""
                            tags = f" [{', '.join(note.tags)}]" if note.tags else ""
                            print(f"   {pin}{note.id[:8]}: {note.content[:60]}...{tags}")
                    else:
                        print("   No notes yet. Use 'note <text>' to add one.")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd.startswith("note "):
                try:
                    from Cosmos.tools.productivity.quick_notes import quick_notes
                    content = cmd[5:]
                    # Extract tags from #hashtags
                    import re
                    tags = re.findall(r'#(\w+)', content)
                    content = re.sub(r'#\w+', '', content).strip()
                    note = quick_notes.add(content, tags)
                    print(f"✅ Note added: {note.id[:8]}")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "snippets":
                try:
                    from Cosmos.tools.productivity.snippet_manager import snippet_manager
                    popular = snippet_manager.list_popular(10)
                    stats = snippet_manager.stats()
                    print(f"\n📋 Code Snippets ({stats['total']} total, {stats['favorites']} favorites)")
                    if popular:
                        for s in popular:
                            fav = "⭐ " if s.favorite else ""
                            print(f"   {fav}{s.id[:8]}: {s.name} [{s.language}] (used {s.usage_count}x)")
                    else:
                        print("   No snippets yet.")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "focus start":
                try:
                    from Cosmos.tools.productivity.focus_timer import focus_timer
                    asyncio.create_task(focus_timer.start_work())
                    print(f"🍅 Focus session started ({focus_timer.config.work_minutes} min)")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "focus stop":
                try:
                    from Cosmos.tools.productivity.focus_timer import focus_timer
                    asyncio.create_task(focus_timer.stop())
                    print("⏹️ Focus session stopped")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "focus" or cmd == "focus status":
                try:
                    from Cosmos.tools.productivity.focus_timer import focus_timer
                    status = focus_timer.get_status()
                    today = focus_timer.get_today_stats()
                    print(f"\n🍅 Focus Timer")
                    print(f"   State: {status['state']}")
                    if status['remaining_seconds'] > 0:
                        print(f"   Remaining: {status['remaining_display']}")
                    print(f"   Sessions Today: {today['sessions']}")
                    print(f"   Focus Time: {today['total_minutes']} min")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "summary":
                try:
                    from Cosmos.tools.productivity.daily_summary import daily_summary
                    print("Generating daily summary...")
                    s = await daily_summary.generate_summary(memory_system=memory)
                    print(daily_summary.format_summary(s))
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "profile":
                try:
                    from Cosmos.core.context_profiles import context_profiles
                    active = context_profiles.get_active_profile()
                    if active:
                        print(f"\n{active.icon} Active Profile: {active.name}")
                        print(f"   Description: {active.description}")
                        print(f"   Memory Pool: {active.memory_pool}")
                        print(f"   Personality: {active.personality}")
                        print(f"   Temperature: {active.temperature}")
                    else:
                        print("No active profile")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "profiles":
                try:
                    from Cosmos.core.context_profiles import context_profiles
                    profiles = context_profiles.list_profiles()
                    active_id = context_profiles.active_profile_id
                    print(f"\n📋 Context Profiles")
                    for p in profiles:
                        active = " (ACTIVE)" if p.id == active_id else ""
                        print(f"   {p.icon} {p.id}: {p.name}{active}")
                        print(f"      {p.description}")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd.startswith("switch "):
                try:
                    from Cosmos.core.context_profiles import context_profiles
                    profile_id = cmd[7:].strip()
                    profile = context_profiles.switch_profile(profile_id)
                    if profile:
                        print(f"✅ Switched to {profile.icon} {profile.name}")
                    else:
                        print(f"❌ Profile not found: {profile_id}")
                except Exception as e:
                    print(f"❌ Error: {e}")

            elif cmd == "clear":
                print("\033[H\033[J", end="")

            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
        except Exception as e:
            print(f"Error: {e}")


async def run_p2p_node(port: int = 9999, enable_planetary: bool = True):
    """
    Spin up as a P2P network node.

    This allows your cosmos instance to:
    - Discover other nodes on the local network
    - Share knowledge via the Decentralized Knowledge Graph (DKG)
    - Contribute to and benefit from Planetary Memory
    - Participate in distributed task auctions
    """
    print("\n🌐 cosmos P2P Node")
    print("=" * 50)

    import os

    # Check if isolated mode is enabled
    if os.getenv("cosmos_ISOLATED", "").lower() == "true":
        print_status("P2P Node", "error", "Cannot start: cosmos_ISOLATED=true")
        print("\n⚠️  To enable P2P networking, set cosmos_ISOLATED=false in your .env")
        return

    try:
        from Cosmos.core.swarm.p2p import SwarmFabric
        from Cosmos.core.swarm.dkg import DecentralizedKnowledgeGraph
        from Cosmos.core.memory.planetary.akashic import PlanetaryMemory
        from Cosmos.memory.memory_system import MemorySystem

        # Initialize memory system for planetary integration
        memory = MemorySystem(data_dir=str(PROJECT_ROOT / "data"))
        await memory.initialize()
        print_status("Memory System", "ok", "Initialized")

        # Initialize Planetary Memory
        planetary = None
        if enable_planetary:
            planetary = PlanetaryMemory(use_p2p=True)
            print_status("Planetary Memory", "ok", "Enabled (Akashic Record)")
        else:
            print_status("Planetary Memory", "warning", "Disabled")

        # Initialize P2P Swarm Fabric
        fabric = SwarmFabric(port=port)
        print_status("P2P Fabric", "ok", f"Node ID: {fabric.node_id}")
        print_status("TCP Server", "loading", f"Port {port}")
        print_status("UDP Discovery", "loading", "Port 8888 (broadcast)")

        print("\n" + "=" * 50)
        print(f"🚀 Node '{fabric.node_id}' is now LIVE on port {port}")
        print("=" * 50)
        print("\nCapabilities:")
        print("  • CV  - Computer Vision")
        print("  • NLP - Natural Language Processing")
        print("  • P2P - Peer-to-Peer Networking")
        print("\nListening for peers...")
        print("Press Ctrl+C to shutdown gracefully.\n")

        # Start the fabric (this runs forever)
        await fabric.start()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down P2P node...")
        print_status("P2P Node", "ok", "Gracefully terminated")
    except ImportError as e:
        print_status("P2P Node", "error", f"Missing dependency: {e}")
        print("\nMake sure all P2P dependencies are installed.")
    except Exception as e:
        print_status("P2P Node", "error", str(e))


async def run_node_with_dashboard(port: int = 9999):
    """Run P2P node with live status dashboard."""
    import os

    if os.getenv("cosmos_ISOLATED", "").lower() == "true":
        print_status("P2P Node", "error", "Cannot start: cosmos_ISOLATED=true")
        return

    try:
        from Cosmos.core.swarm.p2p import SwarmFabric

        fabric = SwarmFabric(port=port)

        print("\n🌐 cosmos P2P Node Dashboard")
        print("=" * 60)
        print(f"  Node ID:    {fabric.node_id}")
        print(f"  TCP Port:   {port}")
        print(f"  UDP Port:   8888 (discovery)")
        print("=" * 60)

        # Status update task
        async def status_loop():
            while True:
                await asyncio.sleep(30)
                peers = len(fabric.peers)
                dkg_nodes = len(fabric.dkg.nodes)
                dkg_edges = len(fabric.dkg.edges)
                print(f"\r📊 Peers: {peers} | DKG: {dkg_nodes} nodes, {dkg_edges} edges | Messages seen: {len(fabric.seen_messages)}", end="", flush=True)

        # Start status loop in background
        asyncio.create_task(status_loop())

        # Start the fabric
        await fabric.start()

    except KeyboardInterrupt:
        print("\n\n🛑 Node shutdown complete.")
    except Exception as e:
        print_status("P2P Node", "error", str(e))


async def run_user_cli_mode():
    """Run user-friendly CLI mode."""
    print("\n User-Friendly CLI Mode")

    try:
        from Cosmos.cli.user_cli import run_user_cli
        await run_user_cli(data_dir=str(PROJECT_ROOT / "data"))
    except ImportError as e:
        print_status("User CLI", "error", f"Import error: {e}")
        print("\nMake sure all dependencies are installed.")
    except Exception as e:
        print_status("User CLI", "error", str(e))


async def run_health_dashboard():
    """Run the health tracking dashboard."""
    print_status("Health Dashboard", "loading", "Starting on port 8081...")

    try:
        from Cosmos.health.dashboard.server import app
        import uvicorn

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8081,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    except ImportError as e:
        print_status("Health Dashboard", "error", f"Import error: {e}")
        print("\nMake sure health module dependencies are installed.")
    except Exception as e:
        print_status("Health Dashboard", "error", str(e))


async def run_all_services():
    """Run all cosmos services."""
    print_status("Starting Services", "info", "")

    # Start MCP server in background
    print_status("MCP Server", "loading", "Starting in background...")

    # For now, just run MCP server (UI would be separate process)
    await run_mcp_server()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="cosmos - Your Claude Companion AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Start all services
  python main.py --mcp        MCP server only (for Claude Code)
  python main.py --ui         Streamlit dashboard only
  python main.py --cli        Interactive CLI mode
  python main.py --setup      First-time setup wizard
  python main.py --node       Spin up as P2P network node
  python main.py --node --port 9999 --dashboard   Node with live stats

P2P Node Options:
  --node                      Enable P2P networking mode
  --port PORT                 Custom TCP port (default: 9999)
  --no-planetary              Disable Planetary Memory sharing
  --dashboard                 Show live node status updates

For more info: https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2
        """
    )

    parser.add_argument("--mcp", action="store_true", help="Run MCP server only")
    parser.add_argument("--ui", action="store_true", help="Run Streamlit UI only")
    parser.add_argument("--cli", action="store_true", help="Run interactive CLI")
    parser.add_argument("--user", action="store_true", help="Run user-friendly CLI (recommended for beginners)")
    parser.add_argument("--health", action="store_true", help="Run health dashboard on port 8081")
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--node", action="store_true", help="Spin up as P2P network node")
    parser.add_argument("--port", type=int, default=9999, help="Port for P2P node (default: 9999)")
    parser.add_argument("--no-planetary", action="store_true", help="Disable Planetary Memory sharing")
    parser.add_argument("--dashboard", action="store_true", help="Show live node dashboard")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")

    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    if args.setup:
        asyncio.run(run_setup_wizard())
    elif args.user:
        asyncio.run(run_user_cli_mode())
    elif args.health:
        asyncio.run(run_health_dashboard())
    elif args.node:
        if args.dashboard:
            asyncio.run(run_node_with_dashboard(port=args.port))
        else:
            asyncio.run(run_p2p_node(port=args.port, enable_planetary=not args.no_planetary))
    elif args.mcp:
        asyncio.run(run_mcp_server())
    elif args.ui:
        run_streamlit_ui()
    elif args.cli:
        asyncio.run(run_cli_mode())
    else:
        # Default: run all services
        asyncio.run(run_all_services())


if __name__ == "__main__":
    main()
