"""
Cosmos Web Routes - Modular API route definitions.

Each module defines an APIRouter that gets included into the main FastAPI app.
Shared state (managers, globals, helpers) lives in server.py and is imported
by each route module as needed via the _get_shared() lazy import pattern.

Route Modules:
- chat.py         - Main chat, memory, notes, snippets, focus, profiles, health, tools, crypto, oracle, solana
- claude_teams.py - Claude Teams AGI v1.9 integration (delegate, team, plan, switches, stats)
- swarm.py        - Swarm Chat WebSocket, status, history, learning, memory bridge, dedup, limits
- quantum.py      - Quantum computing (IBM), quantum proof, organism, orchestrator, evolution engine
- websocket.py    - WebSocket live feed, sessions, health check
- media.py        - TTS voice cloning, multi-voice, code analysis, AirLLM
- admin.py        - Workers, staging, evolution loop, cognition, heartbeat
- polymarket.py   - Polymarket prediction engine
- autogram.py     - AutoGram social network for AI agents
- bot_tracker.py  - Bot Tracker token ID registration & verification
- x_engagement.py  - X Engagement mega threads, trending topics
- skills.py        - Skill Registry:  search, register skills across the swarm
- cli_bridge_api.py - CLI Bridge: OpenAI-compatible /v1/chat/completions backed by CLI tools
- forge.py        - FORGE: Swarm-powered development orchestration (plan, deliberate, execute, verify)
- assimilate.py   - Assimilate: Federation landing page, installer downloads, agent registration
- hackathon.py    - Hackathon: Colosseum dashboard, live status, deliberation feed, manual task trigger
- gateway.py      - External Gateway: sandboxed external agent communication, rate-limited, secret-scrubbed
- orchestrator.py - Token Orchestrator: dynamic budget dashboard, tandem sessions, efficiency metrics
"""
