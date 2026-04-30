"""CLI commands for Honcho integration management.

Handles: hermes honcho setup | status | sessions | map | peer
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from hermes_constants import get_hermes_home
from plugins.memory.honcho.client import resolve_active_host, resolve_config_path, HOST
from hermes_cli.config import cfg_get


def clone_honcho_for_profile(profile_name: str) -> bool:
    """Auto-clone Honcho config for a new profile from the default host block.

    Called during profile creation. If Honcho is configured on the default
    host, creates a new host block for the profile with inherited settings
    and auto-derived workspace/aiPeer.

    Returns True if a host block was created, False if Honcho isn't configured.
    """
    cfg = _read_config()
    if not cfg:
        return False

    hosts = cfg.get("hosts", {})
    default_block = hosts.get(HOST, {})

    # No default host block and no root-level API key = Honcho not configured
    has_key = bool(cfg.get("apiKey") or os.environ.get("HONCHO_API_KEY"))
    if not default_block and not has_key:
        return False

    new_host = f"{HOST}.{profile_name}"
    if new_host in hosts:
        return False  # already exists

    # Clone settings from default block, override identity fields
    new_block = {}
    for key in ("recallMode", "writeFrequency", "sessionStrategy",
                "sessionPeerPrefix", "contextTokens", "dialecticReasoningLevel",
                "dialecticDynamic", "dialecticMaxChars", "messageMaxChars",
                "dialecticMaxInputChars", "saveMessages", "observation"):
        val = default_block.get(key)
        if val is not None:
            new_block[key] = val

    # Inherit peer name from default
    peer_name = default_block.get("peerName") or cfg.get("peerName")
    if peer_name:
        new_block["peerName"] = peer_name

    # AI peer is profile-specific; workspace is shared so all profiles
    # see the same user context, sessions, and project history.
    # Use the bare profile name as the peer identity (not the host key)
    # because Honcho's peer ID pattern is ^[a-zA-Z0-9_-]+$ (no dots).
    new_block["aiPeer"] = profile_name
    new_block["workspace"] = default_block.get("workspace") or cfg.get("workspace") or HOST
    new_block["enabled"] = default_block.get("enabled", True)

    cfg.setdefault("hosts", {})[new_host] = new_block
    _write_config(cfg)

    # Eagerly create the peer in Honcho so it exists before first message
    _ensure_peer_exists(new_host)
    return True


def _ensure_peer_exists(host_key: str | None = None) -> bool:
    """Create the AI peer in Honcho if it doesn't already exist.

    Idempotent -- safe to call multiple times. Returns True if the peer
    was created or already exists, False on failure.
    """
    try:
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
        hcfg = HonchoClientConfig.from_global_config(host=host_key)
        if not hcfg.enabled or not (hcfg.api_key or hcfg.base_url):
            return False
        client = get_honcho_client(hcfg)
        # peer() is idempotent -- creates if missing, returns if exists
        client.peer(hcfg.ai_peer)
        if hcfg.peer_name:
            client.peer(hcfg.peer_name)
        return True
    except Exception:
        return False


def cmd_enable(args) -> None:
    """Enable Honcho for the active profile."""
    cfg = _read_config()
    host = _host_key()
    label = f"[{host}] " if host != "hermes" else ""
    block = cfg.setdefault("hosts", {}).setdefault(host, {})

    if block.get("enabled") is True:
        print(f"  {label}Honcho is already enabled.\n")
        return

    block["enabled"] = True

    # If this is a new profile host block with no settings, clone from default
    if not block.get("aiPeer"):
        default_block = cfg_get(cfg, "hosts", HOST, default={})
        for key in ("recallMode", "writeFrequency", "sessionStrategy",
                    "contextTokens", "dialecticReasoningLevel", "dialecticDynamic",
                    "dialecticMaxChars", "messageMaxChars", "dialecticMaxInputChars",
                    "saveMessages", "observation"):
            val = default_block.get(key)
            if val is not None and key not in block:
                block[key] = val
        peer_name = default_block.get("peerName") or cfg.get("peerName")
        if peer_name and "peerName" not in block:
            block["peerName"] = peer_name
        # Use bare profile name as AI peer, not the host key
        ai_peer = host.split(".", 1)[1] if "." in host else host
        block.setdefault("aiPeer", ai_peer)
        block.setdefault("workspace", default_block.get("workspace") or cfg.get("workspace") or HOST)

    _write_config(cfg)
    print(f"  {label}Honcho enabled.")

    # Create peer eagerly
    if _ensure_peer_exists(host):
        print(f"  {label}Peer '{block.get('aiPeer', host)}' ready.")
    else:
        print(f"  {label}Peer creation deferred (no connection).")

    print(f"  Saved to {_config_path()}\n")


def cmd_disable(args) -> None:
    """Disable Honcho for the active profile."""
    cfg = _read_config()
    host = _host_key()
    label = f"[{host}] " if host != "hermes" else ""
    block = cfg_get(cfg, "hosts", host, default={})

    if not block or block.get("enabled") is False:
        print(f"  {label}Honcho is already disabled.\n")
        return

    block["enabled"] = False
    _write_config(cfg)
    print(f"  {label}Honcho disabled.")
    print(f"  Saved to {_config_path()}\n")


def cmd_sync(args) -> None:
    """Sync Honcho config to all existing profiles.

    Scans all Hermes profiles and creates host blocks for any that don't
    have one yet. Inherits settings from the default host block.
    """
    try:
        from hermes_cli.profiles import list_profiles
        profiles = list_profiles()
    except Exception as e:
        print(f"  Could not list profiles: {e}\n")
        return

    cfg = _read_config()
    if not cfg:
        print("  No Honcho config found. Run 'hermes honcho setup' first.\n")
        return

    hosts = cfg.get("hosts", {})
    default_block = hosts.get(HOST, {})
    has_key = bool(cfg.get("apiKey") or os.environ.get("HONCHO_API_KEY"))

    if not default_block and not has_key:
        print("  Honcho not configured on default profile. Run 'hermes honcho setup' first.\n")
        return

    created = 0
    skipped = 0
    for p in profiles:
        if p.name == "default":
            continue
        if clone_honcho_for_profile(p.name):
            print(f"  + {p.name} -> hermes.{p.name}")
            created += 1
        else:
            skipped += 1

    if created:
        print(f"\n  {created} profile(s) synced.")
    else:
        print("  All profiles already have Honcho config.")
    if skipped:
        print(f"  {skipped} profile(s) already configured (skipped).")
    print()


def sync_honcho_profiles_quiet() -> int:
    """Sync Honcho host blocks for all profiles. Returns count of newly created blocks.

    Called from `hermes update` -- no output, no exceptions.
    """
    try:
        from hermes_cli.profiles import list_profiles
        profiles = list_profiles()
    except Exception:
        return 0

    cfg = _read_config()
    if not cfg:
        return 0

    default_block = cfg_get(cfg, "hosts", HOST, default={})
    has_key = bool(cfg.get("apiKey") or os.environ.get("HONCHO_API_KEY"))
    if not default_block and not has_key:
        return 0

    created = 0
    for p in profiles:
        if p.name == "default":
            continue
        if clone_honcho_for_profile(p.name):
            created += 1
    return created


_profile_override: str | None = None


def _host_key() -> str:
    """Return the active Honcho host key, derived from the current Hermes profile."""
    if _profile_override:
        if _profile_override in ("default", "custom"):
            return HOST
        return f"{HOST}.{_profile_override}"
    return resolve_active_host()


def _config_path() -> Path:
    """Return the active Honcho config path for reading (instance-local or global)."""
    return resolve_config_path()


def _local_config_path() -> Path:
    """Return the instance-local Honcho config path for writing.

    Always returns $HERMES_HOME/honcho.json so each profile/instance gets
    its own config file.  The global ~/.honcho/config.json is only used as
    a read fallback (via resolve_config_path) for cross-app interop.
    """
    return get_hermes_home() / "honcho.json"


def _read_config() -> dict:
    path = _config_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_config(cfg: dict, path: Path | None = None) -> None:
    path = path or _local_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_api_key(cfg: dict) -> str:
    """Resolve API key with host -> root -> env fallback.

    For self-hosted instances configured with ``baseUrl`` instead of an API
    key, returns ``"local"`` so that credential guards throughout the CLI
    don't reject a valid configuration.  The ``baseUrl`` is scheme-validated
    (http/https only) so that a typo like ``baseUrl: true`` can't silently
    pass the guard.  Schemeless strings that look like host:port (legacy
    config shapes, e.g. ``localhost:8000``) still pass — the Honcho SDK
    will reject them itself with a clearer error than ours.
    """
    host_key = ((cfg.get("hosts") or {}).get(_host_key()) or {}).get("apiKey")
    key = host_key or cfg.get("apiKey", "") or os.environ.get("HONCHO_API_KEY", "")
    if not key:
        base_url = cfg.get("baseUrl") or cfg.get("base_url") or os.environ.get("HONCHO_BASE_URL", "")
        base_url = (base_url or "").strip()
        if base_url:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(base_url)
            except (TypeError, ValueError):
                parsed = None
            if parsed and parsed.scheme in ("http", "https") and parsed.netloc:
                return "local"
            # Schemeless but looks like a host (contains '.' or ':' and isn't
            # a boolean literal): let it through so legacy configs don't
            # regress into "no API key configured" when they previously worked.
            lowered = base_url.lower()
            if lowered not in ("true", "false", "none", "null") and any(
                c in base_url for c in ".:"
            ) and not base_url.isdigit():
                return "local"
    return key


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

    write_path = _local_config_path()
    read_path = _config_path()
    print("\nHoncho memory setup\n" + "─" * 40)
    print("  Honcho gives Hermes persistent cross-session memory.")
    print(f"  Config: {write_path}")
    if read_path != write_path and read_path.exists():
        print(f"  (seeding from existing config at {read_path})")
    print()

    if not _ensure_sdk_installed():
        return

    hosts = cfg.setdefault("hosts", {})
    hermes_host = hosts.setdefault(_host_key(), {})

    # --- 1. Cloud or local? ---
    print("  Deployment:")
    print("    cloud -- Honcho cloud (api.honcho.dev)")
    print("    local -- self-hosted Honcho server")
    current_deploy = "local" if any(
        h in (cfg.get("baseUrl") or cfg.get("base_url") or "")
        for h in ("localhost", "127.0.0.1", "::1")
    ) else "cloud"
    deploy = _prompt("Cloud or local?", default=current_deploy)
    is_local = deploy.lower() in ("local", "l")

    # Clean up legacy snake_case key
    cfg.pop("base_url", None)

    if is_local:
        # --- Local: ask for base URL, skip or clear API key ---
        current_url = cfg.get("baseUrl") or ""
        new_url = _prompt("Base URL", default=current_url or "http://localhost:8000")
        if new_url:
            cfg["baseUrl"] = new_url

        # For local no-auth, the SDK must not send an API key.
        # We keep the key in config (for cloud switching later) but
        # the client should skip auth when baseUrl is local.
        current_key = cfg.get("apiKey", "")
        if current_key:
            print(f"\n  API key present in config (kept for cloud/hybrid use).")
            print("  Local connections will skip auth automatically.")
        else:
            print("\n  No API key set. Local no-auth ready.")
    else:
        # --- Cloud: set default base URL, require API key ---
        cfg.pop("baseUrl", None)  # cloud uses SDK default

        current_key = cfg.get("apiKey", "")
        masked = f"...{current_key[-8:]}" if len(current_key) > 8 else ("set" if current_key else "not set")
        print(f"\n  Current API key: {masked}")
        new_key = _prompt("Honcho API key (leave blank to keep current)", secret=True)
        if new_key:
            cfg["apiKey"] = new_key

        if not cfg.get("apiKey"):
            print("\n  No API key configured. Get yours at https://app.honcho.dev")
            print("  Run 'hermes honcho setup' again once you have a key.\n")
            return

    # --- 3. Identity ---
    current_peer = hermes_host.get("peerName") or cfg.get("peerName", "")
    new_peer = _prompt("Your name (user peer)", default=current_peer or os.getenv("USER", "user"))
    if new_peer:
        hermes_host["peerName"] = new_peer

    current_ai = hermes_host.get("aiPeer") or cfg.get("aiPeer", "hermes")
    new_ai = _prompt("AI peer name", default=current_ai)
    if new_ai:
        hermes_host["aiPeer"] = new_ai

    current_workspace = hermes_host.get("workspace") or cfg.get("workspace", "hermes")
    new_workspace = _prompt("Workspace ID", default=current_workspace)
    if new_workspace:
        hermes_host["workspace"] = new_workspace

    # --- 4. Observation mode ---
    current_obs = hermes_host.get("observationMode") or cfg.get("observationMode", "directional")
    print("\n  Observation mode:")
    print("    directional  -- all observations on, each AI peer builds its own view (default)")
    print("    unified      -- shared pool, user observes self, AI observes others only")
    new_obs = _prompt("Observation mode", default=current_obs)
    if new_obs in ("unified", "directional"):
        hermes_host["observationMode"] = new_obs
    else:
        hermes_host["observationMode"] = "directional"

    # --- 5. Write frequency ---
    current_wf = str(hermes_host.get("writeFrequency") or cfg.get("writeFrequency", "async"))
    print("\n  Write frequency:")
    print("    async   -- background thread, no token cost (recommended)")
    print("    turn    -- sync write after every turn")
    print("    session -- batch write at session end only")
    print("    N       -- write every N turns (e.g. 5)")
    new_wf = _prompt("Write frequency", default=current_wf)
    try:
        hermes_host["writeFrequency"] = int(new_wf)
    except (ValueError, TypeError):
        hermes_host["writeFrequency"] = new_wf if new_wf in ("async", "turn", "session") else "async"

    # --- 6. Recall mode ---
    _raw_recall = hermes_host.get("recallMode") or cfg.get("recallMode", "hybrid")
    current_recall = "hybrid" if _raw_recall not in ("hybrid", "context", "tools") else _raw_recall
    print("\n  Recall mode:")
    print("    hybrid  -- auto-injected context + Honcho tools available (default)")
    print("    context -- auto-injected context only, Honcho tools hidden")
    print("    tools   -- Honcho tools only, no auto-injected context")
    new_recall = _prompt("Recall mode", default=current_recall)
    if new_recall in ("hybrid", "context", "tools"):
        hermes_host["recallMode"] = new_recall

    # --- 7. Context token budget ---
    current_ctx_tokens = hermes_host.get("contextTokens") or cfg.get("contextTokens")
    current_display = str(current_ctx_tokens) if current_ctx_tokens else "uncapped"
    print("\n  Context injection per turn (hybrid/context recall modes only):")
    print("    uncapped -- no limit (default)")
    print("    N        -- token limit per turn (e.g. 1200)")
    new_ctx_tokens = _prompt("Context tokens", default=current_display)
    if new_ctx_tokens.strip().lower() in ("none", "uncapped", "no limit"):
        hermes_host.pop("contextTokens", None)
    elif new_ctx_tokens.strip() == "":
        pass  # keep current
    else:
        try:
            val = int(new_ctx_tokens)
            if val >= 0:
                hermes_host["contextTokens"] = val
        except (ValueError, TypeError):
            pass  # keep current

    # --- 7b. Dialectic cadence ---
    current_dialectic = str(hermes_host.get("dialecticCadence") or cfg.get("dialecticCadence") or "2")
    print("\n  Dialectic cadence:")
    print("    How often Honcho rebuilds its user model (LLM call on Honcho backend).")
    print("    1 = every turn, 2 = every other turn, 3+ = sparser.")
    print("    Recommended: 1-5.")
    new_dialectic = _prompt("Dialectic cadence", default=current_dialectic)
    try:
        val = int(new_dialectic)
        if val >= 1:
            hermes_host["dialecticCadence"] = val
    except (ValueError, TypeError):
        hermes_host["dialecticCadence"] = 2

    # --- 7c. Dialectic reasoning level ---
    current_reasoning = (
        hermes_host.get("dialecticReasoningLevel")
        or cfg.get("dialecticReasoningLevel")
        or "low"
    )
    print("\n  Dialectic reasoning level:")
    print("    Depth Honcho uses when synthesizing user context on auto-injected calls.")
    print("    minimal  -- quick factual lookups")
    print("    low      -- straightforward questions (default)")
    print("    medium   -- multi-aspect synthesis")
    print("    high     -- complex behavioral patterns")
    print("    max      -- thorough audit-level analysis")
    new_reasoning = _prompt("Reasoning level", default=current_reasoning)
    if new_reasoning in ("minimal", "low", "medium", "high", "max"):
        hermes_host["dialecticReasoningLevel"] = new_reasoning
    else:
        hermes_host["dialecticReasoningLevel"] = "low"

    # --- 8. Session strategy ---
    current_strat = hermes_host.get("sessionStrategy") or cfg.get("sessionStrategy", "per-session")
    print("\n  Session strategy:")
    print("    per-session   -- each run starts clean, Honcho injects context automatically")
    print("    per-directory -- reuses session per dir, prior context auto-injected each run")
    print("    per-repo      -- one session per git repository")
    print("    global        -- single session across all directories")
    new_strat = _prompt("Session strategy", default=current_strat)
    if new_strat in ("per-session", "per-repo", "per-directory", "global"):
        hermes_host["sessionStrategy"] = new_strat

    hermes_host["enabled"] = True
    hermes_host.setdefault("saveMessages", True)

    _write_config(cfg)
    print(f"\n  Config written to {write_path}")

    # --- Auto-enable Honcho as memory provider in config.yaml ---
    try:
        from hermes_cli.config import load_config, save_config
        hermes_config = load_config()
        hermes_config.setdefault("memory", {})["provider"] = "honcho"
        save_config(hermes_config)
        print("  Memory provider set to 'honcho' in config.yaml")
    except Exception as e:
        print(f"  Could not auto-enable in config.yaml: {e}")
        print("  Run: hermes config set memory.provider honcho")

    # --- Test connection ---
    print("  Testing connection... ", end="", flush=True)
    try:
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client, reset_honcho_client
        reset_honcho_client()
        hcfg = HonchoClientConfig.from_global_config(host=_host_key())
        get_honcho_client(hcfg)
        print("OK")
    except Exception as e:
        print(f"FAILED\n  Error: {e}")
        return

    print("\n  Honcho is ready.")
    print(f"  Session:   {hcfg.resolve_session_name()}")
    print(f"  Workspace: {hcfg.workspace_id}")
    print(f"  User:      {hcfg.peer_name}")
    print(f"  AI peer:   {hcfg.ai_peer}")
    print(f"  Observe:   {hcfg.observation_mode}")
    print(f"  Frequency: {hcfg.write_frequency}")
    print(f"  Recall:    {hcfg.recall_mode}")
    print(f"  Sessions:  {hcfg.session_strategy}")
    print("\n  Honcho tools available in chat:")
    print("    honcho_context   -- session context: summary, representation, card, messages")
    print("    honcho_search    -- semantic search over history")
    print("    honcho_profile   -- peer card, key facts")
    print("    honcho_reasoning -- ask Honcho a question, synthesized answer")
    print("    honcho_conclude  -- persist a user fact to memory")
    print("\n  Other commands:")
    print("    hermes honcho status     -- show full config")
    print("    hermes honcho mode       -- change recall/observation mode")
    print("    hermes honcho tokens     -- tune context and dialectic budgets")
    print("    hermes honcho peer       -- update peer names")
    print("    hermes honcho map <name> -- map this directory to a session name\n")


def _active_profile_name() -> str:
    """Return the active Hermes profile name (respects --target-profile override)."""
    if _profile_override:
        return _profile_override
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name()
    except Exception:
        return "default"


def _all_profile_host_configs() -> list[tuple[str, str, dict]]:
    """Return (profile_name, host_key, host_block) for every known profile.

    Reads honcho.json once and maps each profile to its host block.
    """
    try:
        from hermes_cli.profiles import list_profiles
        profiles = list_profiles()
    except Exception:
        return [(_active_profile_name(), _host_key(), {})]

    cfg = _read_config()
    hosts = cfg.get("hosts", {})
    results = []

    # Default profile
    default_block = hosts.get(HOST, {})
    results.append(("default", HOST, default_block))

    for p in profiles:
        if p.name == "default":
            continue
        h = f"{HOST}.{p.name}"
        results.append((p.name, h, hosts.get(h, {})))

    return results


def cmd_status(args) -> None:
    """Show current Honcho config and connection status."""
    show_all = getattr(args, "all", False)

    if show_all:
        _cmd_status_all()
        return

    try:
        import honcho  # noqa: F401
    except ImportError:
        print("  honcho-ai is not installed. Run: hermes honcho setup\n")
        return

    cfg = _read_config()

    active_path = _config_path()
    write_path = _local_config_path()

    if not cfg:
        print(f"  No Honcho config found at {active_path}")
        print("  Run 'hermes honcho setup' to configure.\n")
        return

    try:
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
        hcfg = HonchoClientConfig.from_global_config(host=_host_key())
    except Exception as e:
        print(f"  Config error: {e}\n")
        return

    api_key = hcfg.api_key or ""
    masked = f"...{api_key[-8:]}" if len(api_key) > 8 else ("set" if api_key else "not set")

    profile = _active_profile_name()
    profile_label = f" [{hcfg.host}]" if profile != "default" else ""

    print(f"\nHoncho status{profile_label}\n" + "─" * 40)
    if profile != "default":
        print(f"  Profile:        {profile}")
    print(f"  Host:           {hcfg.host}")
    print(f"  Enabled:        {hcfg.enabled}")
    print(f"  API key:        {masked}")
    print(f"  Workspace:      {hcfg.workspace_id}")

    # Config paths — show where config was read from and where writes go
    global_path = Path.home() / ".honcho" / "config.json"
    print(f"  Config:         {active_path}")
    if write_path != active_path:
        print(f"  Write to:       {write_path}  (profile-local)")
    if active_path == global_path:
        print(f"  Fallback:       (none — using global ~/.honcho/config.json)")
    elif global_path.exists():
        print(f"  Fallback:       {global_path}  (exists, cross-app interop)")

    print(f"  AI peer:        {hcfg.ai_peer}")
    print(f"  User peer:      {hcfg.peer_name or 'not set'}")
    print(f"  Session key:    {hcfg.resolve_session_name()}")
    print(f"  Session strat:  {hcfg.session_strategy}")
    print(f"  Recall mode:    {hcfg.recall_mode}")
    print(f"  Context budget: {hcfg.context_tokens or '(uncapped)'} tokens")
    raw = getattr(hcfg, "raw", None) or {}
    dialectic_cadence = raw.get("dialecticCadence") or 1
    print(f"  Dialectic cad:  every {dialectic_cadence} turn{'s' if dialectic_cadence != 1 else ''}")
    reasoning_cap = raw.get("reasoningLevelCap") or hcfg.reasoning_level_cap
    heuristic_on = "on" if hcfg.reasoning_heuristic else "off"
    print(f"  Reasoning:      base={hcfg.dialectic_reasoning_level}, cap={reasoning_cap}, heuristic={heuristic_on}")
    print(f"  Observation:    user(me={hcfg.user_observe_me},others={hcfg.user_observe_others}) ai(me={hcfg.ai_observe_me},others={hcfg.ai_observe_others})")
    print(f"  Write freq:     {hcfg.write_frequency}")

    if hcfg.enabled and (hcfg.api_key or hcfg.base_url):
        print("\n  Connection... ", end="", flush=True)
        try:
            client = get_honcho_client(hcfg)
            _show_peer_cards(hcfg, client)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})\n")
    else:
        reason = "disabled" if not hcfg.enabled else "no API key or base URL"
        print(f"\n  Not connected ({reason})\n")


def _show_peer_cards(hcfg, client) -> None:
    """Fetch and display peer cards for the active profile.

    Uses get_or_create to ensure the session exists with peers configured.
    This is idempotent -- if the session already exists on the server it's
    just retrieved, not duplicated.
    """
    try:
        from plugins.memory.honcho.session import HonchoSessionManager
        mgr = HonchoSessionManager(honcho=client, config=hcfg)
        session_key = hcfg.resolve_session_name()
        mgr.get_or_create(session_key)

        # User peer card
        card = mgr.get_peer_card(session_key)
        if card:
            print(f"\n  User peer card ({len(card)} facts):")
            for fact in card[:10]:
                print(f"    - {fact}")
            if len(card) > 10:
                print(f"    ... and {len(card) - 10} more")

        # AI peer representation
        ai_rep = mgr.get_ai_representation(session_key)
        ai_text = ai_rep.get("representation", "")
        if ai_text:
            # Truncate to first 200 chars
            display = ai_text[:200] + ("..." if len(ai_text) > 200 else "")
            print(f"\n  AI peer representation:")
            print(f"    {display}")

        if not card and not ai_text:
            print("\n  No peer data yet (accumulates after first conversation)")

        print()
    except Exception as e:
        print(f"\n  Peer data unavailable: {e}\n")


def _cmd_status_all() -> None:
    """Show Honcho config overview across all profiles."""
    rows = _all_profile_host_configs()
    cfg = _read_config()
    active = _active_profile_name()

    print(f"\nHoncho profiles ({len(rows)})\n" + "─" * 55)
    print(f"  {'Profile':<14} {'Host':<22} {'Enabled':<9} {'Recall':<9} {'Write'}")
    print(f"  {'─' * 14} {'─' * 22} {'─' * 9} {'─' * 9} {'─' * 9}")

    for name, host, block in rows:
        enabled = block.get("enabled", cfg.get("enabled"))
        if enabled is None:
            has_creds = bool(cfg.get("apiKey") or os.environ.get("HONCHO_API_KEY"))
            enabled = has_creds if block else False
        enabled_str = "yes" if enabled else "no"

        recall = block.get("recallMode") or cfg.get("recallMode", "hybrid")
        write = block.get("writeFrequency") or cfg.get("writeFrequency", "async")

        marker = " *" if name == active else ""
        print(f"  {name + marker:<14} {host:<22} {enabled_str:<9} {recall:<9} {write}")

    print(f"\n  * active profile\n")


def cmd_peers(args) -> None:
    """Show peer identities across all profiles."""
    rows = _all_profile_host_configs()
    cfg = _read_config()

    print(f"\nHoncho peer identities ({len(rows)} profiles)\n" + "─" * 50)
    print(f"  {'Profile':<14} {'User peer':<16} {'AI peer'}")
    print(f"  {'─' * 14} {'─' * 16} {'─' * 18}")

    for name, host, block in rows:
        user = block.get("peerName") or cfg.get("peerName") or "(not set)"
        ai = block.get("aiPeer") or cfg.get("aiPeer") or host
        print(f"  {name:<14} {user:<16} {ai}")

    print()


def cmd_sessions(args) -> None:
    """List known directory → session name mappings."""
    cfg = _read_config()
    sessions = cfg.get("sessions", {})

    if not sessions:
        print("  No session mappings configured.\n")
        print("  Add one with: hermes honcho map <session-name>")
        print(f"  Or edit {_config_path()} directly.\n")
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
        hermes = hosts.get(_host_key(), {})
        user = hermes.get('peerName') or cfg.get('peerName') or '(not set)'
        ai = hermes.get('aiPeer') or cfg.get('aiPeer') or _host_key()
        lvl = hermes.get("dialecticReasoningLevel") or cfg.get("dialecticReasoningLevel") or "low"
        max_chars = hermes.get("dialecticMaxChars") or cfg.get("dialecticMaxChars") or 600
        print("\nHoncho peers\n" + "─" * 40)
        print(f"  User peer:   {user}")
        print("    Your identity in Honcho. Messages you send build this peer's card.")
        print(f"  AI peer:     {ai}")
        print("    Hermes' identity in Honcho. Seed with 'hermes honcho identity <file>'.")
        print("    Dialectic calls ask this peer questions to warm session context.")
        print()
        print(f"  Dialectic reasoning:  {lvl}  ({', '.join(REASONING_LEVELS)})")
        print(f"  Dialectic cap:        {max_chars} chars\n")
        return

    host = _host_key()
    label = f"[{host}] " if host != "hermes" else ""

    if user_name is not None:
        cfg.setdefault("hosts", {}).setdefault(host, {})["peerName"] = user_name.strip()
        changed = True
        print(f"  {label}User peer -> {user_name.strip()}")

    if ai_name is not None:
        cfg.setdefault("hosts", {}).setdefault(host, {})["aiPeer"] = ai_name.strip()
        changed = True
        print(f"  {label}AI peer   -> {ai_name.strip()}")

    if reasoning is not None:
        if reasoning not in REASONING_LEVELS:
            print(f"  Invalid reasoning level '{reasoning}'. Options: {', '.join(REASONING_LEVELS)}")
            return
        cfg.setdefault("hosts", {}).setdefault(host, {})["dialecticReasoningLevel"] = reasoning
        changed = True
        print(f"  {label}Dialectic reasoning level -> {reasoning}")

    if changed:
        _write_config(cfg)
        print(f"  Saved to {_config_path()}\n")


def cmd_mode(args) -> None:
    """Show or set the recall mode."""
    MODES = {
        "hybrid": "auto-injected context + Honcho tools available (default)",
        "context": "auto-injected context only, Honcho tools hidden",
        "tools": "Honcho tools only, no auto-injected context",
    }
    cfg = _read_config()
    mode_arg = getattr(args, "mode", None)

    if mode_arg is None:
        current = (
            (cfg.get("hosts") or {}).get(_host_key(), {}).get("recallMode")
            or cfg.get("recallMode")
            or "hybrid"
        )
        print("\nHoncho recall mode\n" + "─" * 40)
        for m, desc in MODES.items():
            marker = " <-" if m == current else ""
            print(f"  {m:<10}  {desc}{marker}")
        print(f"\n  Set with: hermes honcho mode [hybrid|context|tools]\n")
        return

    if mode_arg not in MODES:
        print(f"  Invalid mode '{mode_arg}'. Options: {', '.join(MODES)}\n")
        return

    host = _host_key()
    label = f"[{host}] " if host != "hermes" else ""
    cfg.setdefault("hosts", {}).setdefault(host, {})["recallMode"] = mode_arg
    _write_config(cfg)
    print(f"  {label}Recall mode -> {mode_arg}  ({MODES[mode_arg]})\n")


def cmd_strategy(args) -> None:
    """Show or set the session strategy."""
    STRATEGIES = {
        "per-session": "each run starts clean, Honcho injects context automatically",
        "per-directory": "reuses session per dir, prior context auto-injected each run",
        "per-repo": "one session per git repository",
        "global": "single session across all directories",
    }
    cfg = _read_config()
    strat_arg = getattr(args, "strategy", None)

    if strat_arg is None:
        current = (
            (cfg.get("hosts") or {}).get(_host_key(), {}).get("sessionStrategy")
            or cfg.get("sessionStrategy")
            or "per-session"
        )
        print("\nHoncho session strategy\n" + "─" * 40)
        for s, desc in STRATEGIES.items():
            marker = " <-" if s == current else ""
            print(f"  {s:<15}  {desc}{marker}")
        print(f"\n  Set with: hermes honcho strategy [per-session|per-directory|per-repo|global]\n")
        return

    if strat_arg not in STRATEGIES:
        print(f"  Invalid strategy '{strat_arg}'. Options: {', '.join(STRATEGIES)}\n")
        return

    host = _host_key()
    label = f"[{host}] " if host != "hermes" else ""
    cfg.setdefault("hosts", {}).setdefault(host, {})["sessionStrategy"] = strat_arg
    _write_config(cfg)
    print(f"  {label}Session strategy -> {strat_arg}  ({STRATEGIES[strat_arg]})\n")


def cmd_tokens(args) -> None:
    """Show or set token budget settings."""
    cfg = _read_config()
    hosts = cfg.get("hosts", {})
    hermes = hosts.get(_host_key(), {})

    context = getattr(args, "context", None)
    dialectic = getattr(args, "dialectic", None)

    if context is None and dialectic is None:
        ctx_tokens = hermes.get("contextTokens") or cfg.get("contextTokens") or "(Honcho default)"
        d_chars = hermes.get("dialecticMaxChars") or cfg.get("dialecticMaxChars") or 600
        d_level = hermes.get("dialecticReasoningLevel") or cfg.get("dialecticReasoningLevel") or "low"
        print("\nHoncho budgets\n" + "─" * 40)
        print()
        print(f"  Context     {ctx_tokens} tokens")
        print("    Raw memory retrieval. Honcho returns stored facts/history about")
        print("    the user and session, injected directly into the system prompt.")
        print()
        print(f"  Dialectic   {d_chars} chars, reasoning: {d_level}")
        print("    AI-to-AI inference. Hermes asks Honcho's AI peer a question")
        print("    (e.g. \"what were we working on?\") and Honcho runs its own model")
        print("    to synthesize an answer. Used for first-turn session continuity.")
        print("    Level controls how much reasoning Honcho spends on the answer.")
        print("\n  Set with: hermes honcho tokens [--context N] [--dialectic N]\n")
        return

    host = _host_key()
    label = f"[{host}] " if host != "hermes" else ""
    changed = False
    if context is not None:
        cfg.setdefault("hosts", {}).setdefault(host, {})["contextTokens"] = context
        print(f"  {label}context tokens -> {context}")
        changed = True
    if dialectic is not None:
        cfg.setdefault("hosts", {}).setdefault(host, {})["dialecticMaxChars"] = dialectic
        print(f"  {label}dialectic cap  -> {dialectic} chars")
        changed = True

    if changed:
        _write_config(cfg)
        print(f"  Saved to {_config_path()}\n")


def cmd_identity(args) -> None:
    """Seed AI peer identity or show both peer representations."""
    cfg = _read_config()
    if not _resolve_api_key(cfg):
        print("  No API key configured. Run 'hermes honcho setup' first.\n")
        return

    file_path = getattr(args, "file", None)
    show = getattr(args, "show", False)

    try:
        from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
        from plugins.memory.honcho.session import HonchoSessionManager
        hcfg = HonchoClientConfig.from_global_config(host=_host_key())
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
        print("  Failed to seed identity. Check logs for details.\n")


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
    has_key = bool(_resolve_api_key(cfg))

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
            print("    hermes honcho migrate  — this step handles it interactively")
        if has_key:
            answer = _prompt("  Upload user memory files to Honcho now?", default="y")
            if answer.lower() in ("y", "yes"):
                try:
                    from plugins.memory.honcho.client import (
                        HonchoClientConfig,
                        get_honcho_client,
                        reset_honcho_client,
                    )
                    from plugins.memory.honcho.session import HonchoSessionManager

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
                    from plugins.memory.honcho.client import (
                        HonchoClientConfig,
                        get_honcho_client,
                        reset_honcho_client,
                    )
                    from plugins.memory.honcho.session import HonchoSessionManager

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
    print("    Hermes:   Honcho context fetched async at turn end, injected next turn.")
    print("              First turn has no Honcho context; subsequent turns are loaded.")
    print()
    print("  Memory growth")
    print("    OpenClaw: you edit files manually to update memory.")
    print("    Hermes:   Honcho observes every message and updates representations")
    print("              automatically. Files become the seed, not the live store.")
    print()
    print("  Honcho tools (available to the agent during conversation)")
    print("    honcho_context   — session context: summary, representation, card, messages")
    print("    honcho_search        — semantic search over stored context")
    print("    honcho_profile       — fast peer card snapshot")
    print("    honcho_reasoning     — ask Honcho a question, synthesized answer")
    print("    honcho_conclude      — write a conclusion/fact back to memory")
    print()
    print("  Session naming")
    print("    OpenClaw: no persistent session concept — files are global.")
    print("    Hermes:   per-session by default — each run gets its own session")
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
    global _profile_override
    _profile_override = getattr(args, "target_profile", None)

    sub = getattr(args, "honcho_command", None)
    if sub == "setup":
        # Redirect to memory setup — honcho setup goes through the unified path
        print("\n  Honcho is configured via the memory provider system.")
        print("  Running 'hermes memory setup'...\n")
        from hermes_cli.memory_setup import cmd_setup_provider
        cmd_setup_provider("honcho")
        return
    elif sub is None:
        cmd_status(args)
    elif sub == "status":
        cmd_status(args)
    elif sub == "peers":
        cmd_peers(args)
    elif sub == "sessions":
        cmd_sessions(args)
    elif sub == "map":
        cmd_map(args)
    elif sub == "peer":
        cmd_peer(args)
    elif sub == "mode":
        cmd_mode(args)
    elif sub == "strategy":
        cmd_strategy(args)
    elif sub == "tokens":
        cmd_tokens(args)
    elif sub == "identity":
        cmd_identity(args)
    elif sub == "migrate":
        cmd_migrate(args)
    elif sub == "enable":
        cmd_enable(args)
    elif sub == "disable":
        cmd_disable(args)
    elif sub == "sync":
        cmd_sync(args)
    else:
        print(f"  Unknown honcho command: {sub}")
        print("  Available: status, sessions, map, peer, mode, strategy, tokens, identity, migrate, enable, disable, sync\n")


def register_cli(subparser) -> None:
    """Build the ``hermes honcho`` argparse subcommand tree.

    Called by the plugin CLI registration system during argparse setup.
    The *subparser* is the parser for ``hermes honcho``.
    """

    subparser.add_argument(
        "--target-profile", metavar="NAME", dest="target_profile",
        help="Target a specific profile's Honcho config without switching",
    )
    subs = subparser.add_subparsers(dest="honcho_command")

    subs.add_parser(
        "setup",
        help="Initial Honcho setup (redirects to hermes memory setup)",
    )

    status_parser = subs.add_parser(
        "status", help="Show current Honcho config and connection status",
    )
    status_parser.add_argument(
        "--all", action="store_true", help="Show config overview across all profiles",
    )

    subs.add_parser("peers", help="Show peer identities across all profiles")
    subs.add_parser("sessions", help="List known Honcho session mappings")

    map_parser = subs.add_parser(
        "map", help="Map current directory to a Honcho session name (no arg = list mappings)",
    )
    map_parser.add_argument(
        "session_name", nargs="?", default=None,
        help="Session name to associate with this directory. Omit to list current mappings.",
    )

    peer_parser = subs.add_parser(
        "peer", help="Show or update peer names and dialectic reasoning level",
    )
    peer_parser.add_argument("--user", metavar="NAME", help="Set user peer name")
    peer_parser.add_argument("--ai", metavar="NAME", help="Set AI peer name")
    peer_parser.add_argument(
        "--reasoning", metavar="LEVEL",
        choices=("minimal", "low", "medium", "high", "max"),
        help="Set default dialectic reasoning level (minimal/low/medium/high/max)",
    )

    mode_parser = subs.add_parser(
        "mode", help="Show or set recall mode (hybrid/context/tools)",
    )
    mode_parser.add_argument(
        "mode", nargs="?", metavar="MODE",
        choices=("hybrid", "context", "tools"),
        help="Recall mode to set (hybrid/context/tools). Omit to show current.",
    )

    strategy_parser = subs.add_parser(
        "strategy", help="Show or set session strategy (per-session/per-directory/per-repo/global)",
    )
    strategy_parser.add_argument(
        "strategy", nargs="?", metavar="STRATEGY",
        choices=("per-session", "per-directory", "per-repo", "global"),
        help="Session strategy to set. Omit to show current.",
    )

    tokens_parser = subs.add_parser(
        "tokens", help="Show or set token budget for context and dialectic",
    )
    tokens_parser.add_argument(
        "--context", type=int, metavar="N",
        help="Max tokens Honcho returns from session.context() per turn",
    )
    tokens_parser.add_argument(
        "--dialectic", type=int, metavar="N",
        help="Max chars of dialectic result to inject into system prompt",
    )

    identity_parser = subs.add_parser(
        "identity", help="Seed or show the AI peer's Honcho identity representation",
    )
    identity_parser.add_argument(
        "file", nargs="?", default=None,
        help="Path to file to seed from (e.g. SOUL.md). Omit to show usage.",
    )
    identity_parser.add_argument(
        "--show", action="store_true",
        help="Show current AI peer representation from Honcho",
    )

    subs.add_parser(
        "migrate",
        help="Step-by-step migration guide from openclaw-honcho to Hermes Honcho",
    )
    subs.add_parser("enable", help="Enable Honcho for the active profile")
    subs.add_parser("disable", help="Disable Honcho for the active profile")
    subs.add_parser("sync", help="Sync Honcho config to all existing profiles")

    subparser.set_defaults(func=honcho_command)
