"""CLI commands for Honcho integration management.

Handles: hermes honcho setup | status | sessions | map | peer
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GLOBAL_CONFIG_PATH = Path.home() / ".honcho" / "config.json"
HOST = "hermes"


def _read_config() -> dict:
    if GLOBAL_CONFIG_PATH.exists():
        try:
            return json.loads(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_config(cfg: dict) -> None:
    GLOBAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    if secret:
        if sys.stdin.isatty():
            import getpass
            val = getpass.getpass(prompt="")
        else:
            # Non-TTY (piped input, test runners) — read plaintext
            val = sys.stdin.readline().strip()
    else:
        val = sys.stdin.readline().strip()
    return val or (default or "")


def _ensure_sdk_installed() -> bool:
    """Check honcho-ai is importable; offer to install if not. Returns True if ready."""
    try:
        import honcho  # noqa: F401
        return True
    except ImportError:
        pass

    print("  honcho-ai is not installed.")
    answer = _prompt("Install it now? (honcho-ai>=2.0.1)", default="y")
    if answer.lower() not in ("y", "yes"):
        print("  Skipping install. Run: pip install 'honcho-ai>=2.0.1'\n")
        return False

    import subprocess
    print("  Installing honcho-ai...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "honcho-ai>=2.0.1"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  Installed.\n")
        return True
    else:
        print(f"  Install failed:\n{result.stderr.strip()}")
        print("  Run manually: pip install 'honcho-ai>=2.0.1'\n")
        return False


def cmd_setup(args) -> None:
    """Interactive Honcho setup wizard."""
    cfg = _read_config()

    print("\nHoncho memory setup\n" + "─" * 40)
    print("  Honcho gives Hermes persistent cross-session memory.")
    print("  Config is shared with other hosts at ~/.honcho/config.json\n")

    if not _ensure_sdk_installed():
        return

    # API key
    current_key = cfg.get("apiKey", "")
    masked = f"...{current_key[-8:]}" if len(current_key) > 8 else ("set" if current_key else "not set")
    print(f"  Current API key: {masked}")
    new_key = _prompt("Honcho API key (leave blank to keep current)", secret=True)
    if new_key:
        cfg["apiKey"] = new_key

    if not cfg.get("apiKey"):
        print("\n  No API key configured. Get your API key at https://app.honcho.dev")
        print("  Run 'hermes honcho setup' again once you have a key.\n")
        return

    # Peer name
    current_peer = cfg.get("peerName", "")
    new_peer = _prompt("Your name (user peer)", default=current_peer or os.getenv("USER", "user"))
    if new_peer:
        cfg["peerName"] = new_peer

    # Host block
    hosts = cfg.setdefault("hosts", {})
    hermes_host = hosts.setdefault(HOST, {})

    current_workspace = hermes_host.get("workspace") or cfg.get("workspace", "hermes")
    new_workspace = _prompt("Workspace ID", default=current_workspace)
    if new_workspace:
        hermes_host["workspace"] = new_workspace
        # Also update flat workspace if it was the primary one
        if cfg.get("workspace") == current_workspace:
            cfg["workspace"] = new_workspace

    hermes_host.setdefault("aiPeer", HOST)

    # Memory mode
    current_mode = cfg.get("memoryMode", "hybrid")
    print(f"\n  Memory mode options:")
    print("    hybrid  — write to both Honcho and local MEMORY.md (default)")
    print("    honcho  — Honcho only, skip MEMORY.md writes")
    print("    local   — MEMORY.md only, Honcho disabled")
    new_mode = _prompt("Memory mode", default=current_mode)
    if new_mode in ("hybrid", "honcho", "local"):
        cfg["memoryMode"] = new_mode
    else:
        cfg["memoryMode"] = "hybrid"

    # Write frequency
    current_wf = str(cfg.get("writeFrequency", "async"))
    print(f"\n  Write frequency options:")
    print("    async   — background thread, no token cost (recommended)")
    print("    turn    — sync write after every turn")
    print("    session — batch write at session end only")
    print("    N       — write every N turns (e.g. 5)")
    new_wf = _prompt("Write frequency", default=current_wf)
    try:
        cfg["writeFrequency"] = int(new_wf)
    except (ValueError, TypeError):
        cfg["writeFrequency"] = new_wf if new_wf in ("async", "turn", "session") else "async"

    # Recall mode
    current_recall = cfg.get("recallMode", "hybrid")
    print(f"\n  Recall mode options:")
    print("    hybrid  — pre-warmed context + memory tools available (default)")
    print("    context — pre-warmed context only, memory tools suppressed")
    print("    tools   — no pre-loaded context, rely on tool calls only")
    new_recall = _prompt("Recall mode", default=current_recall)
    if new_recall in ("hybrid", "context", "tools"):
        cfg["recallMode"] = new_recall

    # Session strategy
    current_strat = cfg.get("sessionStrategy", "per-session")
    print(f"\n  Session strategy options:")
    print("    per-session   — new Honcho session each run, named by Hermes session ID (default)")
    print("    per-repo      — one session per git repository (uses repo root name)")
    print("    per-directory — one session per working directory")
    print("    global        — single session across all directories")
    new_strat = _prompt("Session strategy", default=current_strat)
    if new_strat in ("per-session", "per-repo", "per-directory", "global"):
        cfg["sessionStrategy"] = new_strat

    cfg.setdefault("enabled", True)
    cfg.setdefault("saveMessages", True)

    _write_config(cfg)
    print(f"\n  Config written to {GLOBAL_CONFIG_PATH}")

    # Test connection
    print("  Testing connection... ", end="", flush=True)
    try:
        from honcho_integration.client import HonchoClientConfig, get_honcho_client, reset_honcho_client
        reset_honcho_client()
        hcfg = HonchoClientConfig.from_global_config()
        get_honcho_client(hcfg)
        print("OK")
    except Exception as e:
        print(f"FAILED\n  Error: {e}")
        return

    print(f"\n  Honcho is ready.")
    print(f"  Session:   {hcfg.resolve_session_name()}")
    print(f"  Workspace: {hcfg.workspace_id}")
    print(f"  Peer:      {hcfg.peer_name}")
    _mode_str = hcfg.memory_mode
    if hcfg.peer_memory_modes:
        overrides = ", ".join(f"{k}={v}" for k, v in hcfg.peer_memory_modes.items())
        _mode_str = f"{hcfg.memory_mode}  (peers: {overrides})"
    print(f"  Mode:      {_mode_str}")
    print(f"  Frequency: {hcfg.write_frequency}")
    print(f"\n  Tools available in chat:")
    print(f"    query_user_context  — ask Honcho a question about you (LLM-synthesized)")
    print(f"    honcho_search       — semantic search over your history (no LLM)")
    print(f"    honcho_profile      — your peer card, key facts (no LLM)")
    print(f"\n  Other commands:")
    print(f"    hermes honcho status     — show full config")
    print(f"    hermes honcho mode       — show or change memory mode")
    print(f"    hermes honcho tokens     — show or set token budgets")
    print(f"    hermes honcho identity   — seed or show AI peer identity")
    print(f"    hermes honcho map <name> — map this directory to a session name\n")


def cmd_status(args) -> None:
    """Show current Honcho config and connection status."""
    try:
        import honcho  # noqa: F401
    except ImportError:
        print("  honcho-ai is not installed. Run: hermes honcho setup\n")
        return

    cfg = _read_config()

    if not cfg:
        print("  No Honcho config found at ~/.honcho/config.json")
        print("  Run 'hermes honcho setup' to configure.\n")
        return

    try:
        from honcho_integration.client import HonchoClientConfig, get_honcho_client
        hcfg = HonchoClientConfig.from_global_config()
    except Exception as e:
        print(f"  Config error: {e}\n")
        return

    api_key = hcfg.api_key or ""
    masked = f"...{api_key[-8:]}" if len(api_key) > 8 else ("set" if api_key else "not set")

    print(f"\nHoncho status\n" + "─" * 40)
    print(f"  Enabled:        {hcfg.enabled}")
    print(f"  API key:        {masked}")
    print(f"  Workspace:      {hcfg.workspace_id}")
    print(f"  Host:           {hcfg.host}")
    print(f"  Config path:    {GLOBAL_CONFIG_PATH}")
    print(f"  AI peer:        {hcfg.ai_peer}")
    print(f"  User peer:      {hcfg.peer_name or 'not set'}")
    print(f"  Session key:    {hcfg.resolve_session_name()}")
    print(f"  Recall mode:    {hcfg.recall_mode}")
    print(f"  Memory mode:    {hcfg.memory_mode}")
    if hcfg.peer_memory_modes:
        print(f"  Per-peer modes:")
        for peer, mode in hcfg.peer_memory_modes.items():
            print(f"    {peer}: {mode}")
    print(f"  Write freq:     {hcfg.write_frequency}")

    if hcfg.enabled and hcfg.api_key:
        print("\n  Connection... ", end="", flush=True)
        try:
            get_honcho_client(hcfg)
            print("OK\n")
        except Exception as e:
            print(f"FAILED ({e})\n")
    else:
        reason = "disabled" if not hcfg.enabled else "no API key"
        print(f"\n  Not connected ({reason})\n")


def cmd_sessions(args) -> None:
    """List known directory → session name mappings."""
    cfg = _read_config()
    sessions = cfg.get("sessions", {})

    if not sessions:
        print("  No session mappings configured.\n")
        print("  Add one with: hermes honcho map <session-name>")
        print("  Or edit ~/.honcho/config.json directly.\n")
        return

    cwd = os.getcwd()
    print(f"\nHoncho session mappings ({len(sessions)})\n" + "─" * 40)
    for path, name in sorted(sessions.items()):
        marker = " ←" if path == cwd else ""
        print(f"  {name:<30} {path}{marker}")
    print()


def cmd_map(args) -> None:
    """Map current directory to a Honcho session name."""
    if not args.session_name:
        cmd_sessions(args)
        return

    cwd = os.getcwd()
    session_name = args.session_name.strip()

    if not session_name:
        print("  Session name cannot be empty.\n")
        return

    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', session_name).strip('-')
    if sanitized != session_name:
        print(f"  Session name sanitized to: {sanitized}")
        session_name = sanitized

    cfg = _read_config()
    cfg.setdefault("sessions", {})[cwd] = session_name
    _write_config(cfg)
    print(f"  Mapped {cwd}\n     → {session_name}\n")


def cmd_peer(args) -> None:
    """Show or update peer names and dialectic reasoning level."""
    cfg = _read_config()
    changed = False

    user_name = getattr(args, "user", None)
    ai_name = getattr(args, "ai", None)
    reasoning = getattr(args, "reasoning", None)

    REASONING_LEVELS = ("minimal", "low", "medium", "high", "max")

    if user_name is None and ai_name is None and reasoning is None:
        # Show current values
        hosts = cfg.get("hosts", {})
        hermes = hosts.get(HOST, {})
        user = cfg.get('peerName') or '(not set)'
        ai = hermes.get('aiPeer') or cfg.get('aiPeer') or HOST
        lvl = hermes.get("dialecticReasoningLevel") or cfg.get("dialecticReasoningLevel") or "low"
        max_chars = hermes.get("dialecticMaxChars") or cfg.get("dialecticMaxChars") or 600
        print(f"\nHoncho peers\n" + "─" * 40)
        print(f"  User peer:   {user}")
        print(f"    Your identity in Honcho. Messages you send build this peer's card.")
        print(f"  AI peer:     {ai}")
        print(f"    Hermes' identity in Honcho. Seed with 'hermes honcho identity <file>'.")
        print(f"    Dialectic calls ask this peer questions to warm session context.")
        print()
        print(f"  Dialectic reasoning:  {lvl}  ({', '.join(REASONING_LEVELS)})")
        print(f"  Dialectic cap:        {max_chars} chars\n")
        return

    if user_name is not None:
        cfg["peerName"] = user_name.strip()
        changed = True
        print(f"  User peer → {cfg['peerName']}")

    if ai_name is not None:
        cfg.setdefault("hosts", {}).setdefault(HOST, {})["aiPeer"] = ai_name.strip()
        changed = True
        print(f"  AI peer   → {ai_name.strip()}")

    if reasoning is not None:
        if reasoning not in REASONING_LEVELS:
            print(f"  Invalid reasoning level '{reasoning}'. Options: {', '.join(REASONING_LEVELS)}")
            return
        cfg.setdefault("hosts", {}).setdefault(HOST, {})["dialecticReasoningLevel"] = reasoning
        changed = True
        print(f"  Dialectic reasoning level → {reasoning}")

    if changed:
        _write_config(cfg)
        print(f"  Saved to {GLOBAL_CONFIG_PATH}\n")


def cmd_mode(args) -> None:
    """Show or set the memory mode."""
    MODES = {
        "hybrid": "write to both Honcho and local MEMORY.md (default)",
        "honcho": "Honcho only — MEMORY.md writes disabled",
        "local":  "MEMORY.md only — Honcho disabled",
    }
    cfg = _read_config()
    mode_arg = getattr(args, "mode", None)

    if mode_arg is None:
        current = (
            (cfg.get("hosts") or {}).get(HOST, {}).get("memoryMode")
            or cfg.get("memoryMode")
            or "hybrid"
        )
        print(f"\nHoncho memory mode\n" + "─" * 40)
        for m, desc in MODES.items():
            marker = " ←" if m == current else ""
            print(f"  {m:<8}  {desc}{marker}")
        print(f"\n  Set with: hermes honcho mode [hybrid|honcho|local]\n")
        return

    if mode_arg not in MODES:
        print(f"  Invalid mode '{mode_arg}'. Options: {', '.join(MODES)}\n")
        return

    cfg.setdefault("hosts", {}).setdefault(HOST, {})["memoryMode"] = mode_arg
    _write_config(cfg)
    print(f"  Memory mode → {mode_arg}  ({MODES[mode_arg]})\n")


def cmd_tokens(args) -> None:
    """Show or set token budget settings."""
    cfg = _read_config()
    hosts = cfg.get("hosts", {})
    hermes = hosts.get(HOST, {})

    context = getattr(args, "context", None)
    dialectic = getattr(args, "dialectic", None)

    if context is None and dialectic is None:
        ctx_tokens = hermes.get("contextTokens") or cfg.get("contextTokens") or "(Honcho default)"
        d_chars = hermes.get("dialecticMaxChars") or cfg.get("dialecticMaxChars") or 600
        d_level = hermes.get("dialecticReasoningLevel") or cfg.get("dialecticReasoningLevel") or "low"
        print(f"\nHoncho budgets\n" + "─" * 40)
        print()
        print(f"  Context     {ctx_tokens} tokens")
        print(f"    Raw memory retrieval. Honcho returns stored facts/history about")
        print(f"    the user and session, injected directly into the system prompt.")
        print()
        print(f"  Dialectic   {d_chars} chars, reasoning: {d_level}")
        print(f"    AI-to-AI inference. Hermes asks Honcho's AI peer a question")
        print(f"    (e.g. \"what were we working on?\") and Honcho runs its own model")
        print(f"    to synthesize an answer. Used for first-turn session continuity.")
        print(f"    Level controls how much reasoning Honcho spends on the answer.")
        print(f"\n  Set with: hermes honcho tokens [--context N] [--dialectic N]\n")
        return

    changed = False
    if context is not None:
        cfg.setdefault("hosts", {}).setdefault(HOST, {})["contextTokens"] = context
        print(f"  context tokens → {context}")
        changed = True
    if dialectic is not None:
        cfg.setdefault("hosts", {}).setdefault(HOST, {})["dialecticMaxChars"] = dialectic
        print(f"  dialectic cap  → {dialectic} chars")
        changed = True

    if changed:
        _write_config(cfg)
        print(f"  Saved to {GLOBAL_CONFIG_PATH}\n")


def cmd_identity(args) -> None:
    """Seed AI peer identity or show both peer representations."""
    cfg = _read_config()
    if not cfg.get("apiKey"):
        print("  No API key configured. Run 'hermes honcho setup' first.\n")
        return

    file_path = getattr(args, "file", None)
    show = getattr(args, "show", False)

    try:
        from honcho_integration.client import HonchoClientConfig, get_honcho_client
        from honcho_integration.session import HonchoSessionManager
        hcfg = HonchoClientConfig.from_global_config()
        client = get_honcho_client(hcfg)
        mgr = HonchoSessionManager(honcho=client, config=hcfg)
        session_key = hcfg.resolve_session_name()
        mgr.get_or_create(session_key)
    except Exception as e:
        print(f"  Honcho connection failed: {e}\n")
        return

    if show:
        # ── User peer ────────────────────────────────────────────────────────
        user_card = mgr.get_peer_card(session_key)
        print(f"\nUser peer ({hcfg.peer_name or 'not set'})\n" + "─" * 40)
        if user_card:
            for fact in user_card:
                print(f"  {fact}")
        else:
            print("  No user peer card yet. Send a few messages to build one.")

        # ── AI peer ──────────────────────────────────────────────────────────
        ai_rep = mgr.get_ai_representation(session_key)
        print(f"\nAI peer ({hcfg.ai_peer})\n" + "─" * 40)
        if ai_rep.get("representation"):
            print(ai_rep["representation"])
        elif ai_rep.get("card"):
            print(ai_rep["card"])
        else:
            print("  No representation built yet.")
            print("  Run 'hermes honcho identity <file>' to seed one.")
        print()
        return

    if not file_path:
        print("\nHoncho identity management\n" + "─" * 40)
        print(f"  User peer: {hcfg.peer_name or 'not set'}")
        print(f"  AI peer:   {hcfg.ai_peer}")
        print()
        print("    hermes honcho identity --show        — show both peer representations")
        print("    hermes honcho identity <file>        — seed AI peer from SOUL.md or any .md/.txt\n")
        return

    from pathlib import Path
    p = Path(file_path).expanduser()
    if not p.exists():
        print(f"  File not found: {p}\n")
        return

    content = p.read_text(encoding="utf-8").strip()
    if not content:
        print(f"  File is empty: {p}\n")
        return

    source = p.name
    ok = mgr.seed_ai_identity(session_key, content, source=source)
    if ok:
        print(f"  Seeded AI peer identity from {p.name} into session '{session_key}'")
        print(f"  Honcho will incorporate this into {hcfg.ai_peer}'s representation over time.\n")
    else:
        print(f"  Failed to seed identity. Check logs for details.\n")


def cmd_migrate(args) -> None:
    """Step-by-step migration guide: OpenClaw native memory → Hermes + Honcho."""
    from pathlib import Path

    # ── Detect OpenClaw native memory files ──────────────────────────────────
    cwd = Path(os.getcwd())
    openclaw_home = Path.home() / ".openclaw"

    # User peer: facts about the user
    user_file_names = ["USER.md", "MEMORY.md"]
    # AI peer: agent identity / configuration
    agent_file_names = ["SOUL.md", "IDENTITY.md", "AGENTS.md", "TOOLS.md", "BOOTSTRAP.md"]

    user_files: list[Path] = []
    agent_files: list[Path] = []
    for name in user_file_names:
        for d in [cwd, openclaw_home]:
            p = d / name
            if p.exists() and p not in user_files:
                user_files.append(p)
    for name in agent_file_names:
        for d in [cwd, openclaw_home]:
            p = d / name
            if p.exists() and p not in agent_files:
                agent_files.append(p)

    cfg = _read_config()
    has_key = bool(cfg.get("apiKey", ""))

    print("\nHoncho migration: OpenClaw native memory → Hermes\n" + "─" * 50)
    print()
    print("  OpenClaw's native memory stores context in local markdown files")
    print("  (USER.md, MEMORY.md, SOUL.md, ...) and injects them via QMD search.")
    print("  Honcho replaces that with a cloud-backed, LLM-observable memory layer:")
    print("  context is retrieved semantically, injected automatically each turn,")
    print("  and enriched by a dialectic reasoning layer that builds over time.")
    print()

    # ── Step 1: Honcho account ────────────────────────────────────────────────
    print("Step 1  Create a Honcho account")
    print()
    if has_key:
        masked = f"...{cfg['apiKey'][-8:]}" if len(cfg["apiKey"]) > 8 else "set"
        print(f"  Honcho API key already configured: {masked}")
        print("  Skip to Step 2.")
    else:
        print("  Honcho is a cloud memory service that gives Hermes persistent memory")
        print("  across sessions. You need an API key to use it.")
        print()
        print("  1. Get your API key at https://app.honcho.dev")
        print("  2. Run:  hermes honcho setup")
        print("     Paste the key when prompted.")
        print()
        answer = _prompt("  Run 'hermes honcho setup' now?", default="y")
        if answer.lower() in ("y", "yes"):
            cmd_setup(args)
            cfg = _read_config()
            has_key = bool(cfg.get("apiKey", ""))
        else:
            print()
            print("  Run 'hermes honcho setup' when ready, then re-run this walkthrough.")

    # ── Step 2: Detected files ────────────────────────────────────────────────
    print()
    print("Step 2  Detected OpenClaw memory files")
    print()
    if user_files or agent_files:
        if user_files:
            print(f"  User memory ({len(user_files)} file(s)) — will go to Honcho user peer:")
            for f in user_files:
                print(f"    {f}")
        if agent_files:
            print(f"  Agent identity ({len(agent_files)} file(s)) — will go to Honcho AI peer:")
            for f in agent_files:
                print(f"    {f}")
    else:
        print("  No OpenClaw native memory files found in cwd or ~/.openclaw/.")
        print("  If your files are elsewhere, copy them here before continuing,")
        print("  or seed them manually:  hermes honcho identity <path/to/file>")

    # ── Step 3: Migrate user memory ───────────────────────────────────────────
    print()
    print("Step 3  Migrate user memory files → Honcho user peer")
    print()
    print("  USER.md and MEMORY.md contain facts about you that the agent should")
    print("  remember across sessions. Honcho will store these under your user peer")
    print("  and inject relevant excerpts into the system prompt automatically.")
    print()
    if user_files:
        print(f"  Found: {', '.join(f.name for f in user_files)}")
        print()
        print("  These are picked up automatically the first time you run 'hermes'")
        print("  with Honcho configured and no prior session history.")
        print("  (Hermes calls migrate_memory_files() on first session init.)")
        print()
        print("  If you want to migrate them now without starting a session:")
        for f in user_files:
            print(f"    hermes honcho migrate  — this step handles it interactively")
        if has_key:
            answer = _prompt("  Upload user memory files to Honcho now?", default="y")
            if answer.lower() in ("y", "yes"):
                try:
                    from honcho_integration.client import (
                        HonchoClientConfig,
                        get_honcho_client,
                        reset_honcho_client,
                    )
                    from honcho_integration.session import HonchoSessionManager

                    reset_honcho_client()
                    hcfg = HonchoClientConfig.from_global_config()
                    client = get_honcho_client(hcfg)
                    mgr = HonchoSessionManager(honcho=client, config=hcfg)
                    session_key = hcfg.resolve_session_name()
                    mgr.get_or_create(session_key)
                    # Upload from each directory that had user files
                    dirs_with_files = set(str(f.parent) for f in user_files)
                    any_uploaded = False
                    for d in dirs_with_files:
                        if mgr.migrate_memory_files(session_key, d):
                            any_uploaded = True
                    if any_uploaded:
                        print(f"  Uploaded user memory files from: {', '.join(dirs_with_files)}")
                    else:
                        print("  Nothing uploaded (files may already be migrated or empty).")
                except Exception as e:
                    print(f"  Failed: {e}")
        else:
            print("  Run 'hermes honcho setup' first, then re-run this step.")
    else:
        print("  No user memory files detected. Nothing to migrate here.")

    # ── Step 4: Seed AI identity ──────────────────────────────────────────────
    print()
    print("Step 4  Seed AI identity files → Honcho AI peer")
    print()
    print("  SOUL.md, IDENTITY.md, AGENTS.md, TOOLS.md, BOOTSTRAP.md define the")
    print("  agent's character, capabilities, and behavioral rules. In OpenClaw")
    print("  these are injected via file search at prompt-build time.")
    print()
    print("  In Hermes, they are seeded once into Honcho's AI peer through the")
    print("  observation pipeline. Honcho builds a representation from them and")
    print("  from every subsequent assistant message (observe_me=True). Over time")
    print("  the representation reflects actual behavior, not just declaration.")
    print()
    if agent_files:
        print(f"  Found: {', '.join(f.name for f in agent_files)}")
        print()
        if has_key:
            answer = _prompt("  Seed AI identity from all detected files now?", default="y")
            if answer.lower() in ("y", "yes"):
                try:
                    from honcho_integration.client import (
                        HonchoClientConfig,
                        get_honcho_client,
                        reset_honcho_client,
                    )
                    from honcho_integration.session import HonchoSessionManager

                    reset_honcho_client()
                    hcfg = HonchoClientConfig.from_global_config()
                    client = get_honcho_client(hcfg)
                    mgr = HonchoSessionManager(honcho=client, config=hcfg)
                    session_key = hcfg.resolve_session_name()
                    mgr.get_or_create(session_key)
                    for f in agent_files:
                        content = f.read_text(encoding="utf-8").strip()
                        if content:
                            ok = mgr.seed_ai_identity(session_key, content, source=f.name)
                            status = "seeded" if ok else "failed"
                            print(f"    {f.name}: {status}")
                except Exception as e:
                    print(f"  Failed: {e}")
        else:
            print("  Run 'hermes honcho setup' first, then seed manually:")
            for f in agent_files:
                print(f"    hermes honcho identity {f}")
    else:
        print("  No agent identity files detected.")
        print("  To seed manually:  hermes honcho identity <path/to/SOUL.md>")

    # ── Step 5: What changes ──────────────────────────────────────────────────
    print()
    print("Step 5  What changes vs. OpenClaw native memory")
    print()
    print("  Storage")
    print("    OpenClaw: markdown files on disk, searched via QMD at prompt-build time.")
    print("    Hermes:   cloud-backed Honcho peers. Files can stay on disk as source")
    print("              of truth; Honcho holds the live representation.")
    print()
    print("  Context injection")
    print("    OpenClaw: file excerpts injected synchronously before each LLM call.")
    print("    Hermes:   Honcho context prefetched async at turn end, injected next turn.")
    print("              First turn has no Honcho context; subsequent turns are loaded.")
    print()
    print("  Memory growth")
    print("    OpenClaw: you edit files manually to update memory.")
    print("    Hermes:   Honcho observes every message and updates representations")
    print("              automatically. Files become the seed, not the live store.")
    print()
    print("  Tool surface (available to the agent during conversation)")
    print("    query_user_context   — ask Honcho a question, get a synthesized answer (LLM)")
    print("    honcho_search        — semantic search over stored context (no LLM)")
    print("    honcho_profile       — fast peer card snapshot (no LLM)")
    print()
    print("  Session naming")
    print("    OpenClaw: no persistent session concept — files are global.")
    print("    Hermes:   per-session by default — each run gets a new Honcho session")
    print("              Map a custom name:  hermes honcho map <session-name>")

    # ── Step 6: Next steps ────────────────────────────────────────────────────
    print()
    print("Step 6  Next steps")
    print()
    if not has_key:
        print("  1. hermes honcho setup              — configure API key (required)")
        print("  2. hermes honcho migrate            — re-run this walkthrough")
    else:
        print("  1. hermes honcho status             — verify Honcho connection")
        print("  2. hermes                           — start a session")
        print("     (user memory files auto-uploaded on first turn if not done above)")
        print("  3. hermes honcho identity --show    — verify AI peer representation")
        print("  4. hermes honcho tokens             — tune context and dialectic budgets")
        print("  5. hermes honcho mode               — view or change memory mode")
    print()


def honcho_command(args) -> None:
    """Route honcho subcommands."""
    sub = getattr(args, "honcho_command", None)
    if sub == "setup" or sub is None:
        cmd_setup(args)
    elif sub == "status":
        cmd_status(args)
    elif sub == "sessions":
        cmd_sessions(args)
    elif sub == "map":
        cmd_map(args)
    elif sub == "peer":
        cmd_peer(args)
    elif sub == "mode":
        cmd_mode(args)
    elif sub == "tokens":
        cmd_tokens(args)
    elif sub == "identity":
        cmd_identity(args)
    elif sub == "migrate":
        cmd_migrate(args)
    else:
        print(f"  Unknown honcho command: {sub}")
        print("  Available: setup, status, sessions, map, peer, mode, tokens, identity, migrate\n")
