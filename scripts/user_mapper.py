#!/usr/bin/env python3
"""
User Identity Mapper — Cross-channel memory unification for Hermes Agent.

Solves: Same user on different channels (CLI, Telegram, Discord) gets
separate memory banks. This script maps multiple chat_ids to a single
user identity via symlinks, so memories are shared across channels.

Usage:
  # Auto-detect owner from gateway config and set up mapping
  python3 user_mapper.py auto-setup

  # Map a chat_id to a user
  python3 user_mapper.py map --chat-id 7359770766 --user nitrogen

  # List all mappings
  python3 user_mapper.py list

  # Show which user a chat_id belongs to
  python3 user_mapper.py resolve --chat-id 7359770766

  # Remove a mapping (reverts to isolated chat_id directory)
  python3 user_mapper.py unmap --chat-id 7359770766

  # Migrate existing chat_id data into user directory
  python3 user_mapper.py migrate --chat-id 7359770766 --user nitrogen

How it works:
  1. Creates ~/.hermes/memories/user_{username}/ as the real data directory
  2. Symlinks ~/.hermes/memories/{chat_id}/ → user_{username}/
  3. For CLI sessions (no user_id), symlinks global MEMORY.md/USER.md
  4. Hermes reads memories/{chat_id}/ and follows the symlink transparently
  5. No code changes to Hermes needed — pure filesystem solution

Companion to PR #17989 (per-user memory isolation):
  - PR #17989: isolates different users (A can't see B's data)
  - This script: unifies same user across channels (Telegram + CLI share data)

Companion to PR #9308 (Honcho owner identity):
  - #9308: auto-detects owner at gateway layer (Honcho only)
  - This script: manual mapping for ANY user + auto-setup for owner
"""

import json
import os
import sys
import shutil
from pathlib import Path

MEMORIES_DIR = Path.home() / ".hermes" / "memories"
MAPPING_FILE = Path.home() / ".hermes" / "user_mapping.json"
CONFIG_FILE = Path.home() / ".hermes" / "config.yaml"


def load_mapping() -> dict:
    """Load chat_id → user_id mapping."""
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE) as f:
            return json.load(f)
    return {}


def save_mapping(mapping: dict):
    """Save mapping to disk."""
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def user_dir(user_id: str) -> Path:
    """Get the real directory for a user."""
    return MEMORIES_DIR / f"user_{user_id}"


def chat_dir(chat_id: str) -> Path:
    """Get the directory Hermes expects for a chat_id."""
    return MEMORIES_DIR / chat_id


def detect_owner_chat_id() -> tuple:
    """Detect the bot owner's chat_id from gateway config.
    
    Returns (chat_id, platform, source) or (None, None, None) if not found.
    
    Detection strategy (conservative, same as #9308):
    1. Read config.yaml for telegram home channel
    2. Check existing memory directories for the one with most data
    3. Fall back to the largest memory directory
    """
    # Strategy 1: Read config.yaml for telegram chat_id
    if CONFIG_FILE.exists():
        try:
            import yaml
            with open(CONFIG_FILE) as f:
                config = yaml.safe_load(f)
            
            # Check telegram config
            tg = config.get('telegram', {})
            home = tg.get('home_channel', '')
            if home:
                chat_id = str(home).strip()
                if chat_id.isdigit():
                    return (chat_id, 'telegram', 'config.yaml home_channel')
        except Exception:
            pass
    
    # Strategy 2: Find the largest memory directory (most data = likely owner)
    if MEMORIES_DIR.exists():
        best_dir = None
        best_size = 0
        for item in MEMORIES_DIR.iterdir():
            if item.is_dir() and not item.name.startswith('user_') and not item.name.startswith('.'):
                # Count total size of memory files
                total = 0
                for f in item.rglob('*'):
                    if f.is_file():
                        total += f.stat().st_size
                if total > best_size:
                    best_size = total
                    best_dir = item
        
        if best_dir and best_size > 0:
            return (best_dir.name, 'detected', f'largest memory dir ({best_size} bytes)')
    
    return (None, None, None)


def cmd_auto_setup():
    """Auto-detect owner and set up cross-channel memory unification."""
    print("=== Auto-setup: Cross-channel Memory Unification ===\n")
    
    mapping = load_mapping()
    
    # Check if already set up
    if any(not k.startswith('_') for k in mapping.keys()):
        print("[INFO] Mappings already exist:")
        cmd_list()
        resp = input("\nRe-run auto-setup? This will update existing mappings. (y/N): ")
        if resp.lower() != 'y':
            print("Aborted.")
            return
    
    # Detect owner
    chat_id, platform, source = detect_owner_chat_id()
    
    if not chat_id:
        print("[ERROR] Could not detect owner's chat_id.")
        print("Please run manually: python3 user_mapper.py map --chat-id <ID> --user <name>")
        return
    
    print(f"[DETECTED] Owner chat_id: {chat_id}")
    print(f"  Platform: {platform}")
    print(f"  Source: {source}")
    
    # Get username
    import getpass
    default_user = os.environ.get('USER', 'owner')
    user = input(f"\nUsername for this identity [{default_user}]: ").strip() or default_user
    
    # Show existing memory data
    cdir = chat_dir(chat_id)
    if cdir.exists():
        files = list(cdir.iterdir())
        print(f"\n[DATA] Found {len(files)} files in memories/{chat_id}/:")
        for f in files:
            size = f.stat().st_size if f.is_file() else 0
            print(f"  {f.name} ({size} bytes)")
    
    # Confirm
    print(f"\nThis will:")
    print(f"  1. Create memories/user_{user}/ as the main directory")
    print(f"  2. Move data from memories/{chat_id}/ into it")
    print(f"  3. Symlink memories/{chat_id}/ → user_{user}/")
    print(f"  4. Symlink global MEMORY.md/USER.md → user_{user}/ (for CLI)")
    
    resp = input("\nProceed? (Y/n): ").strip().lower()
    if resp == 'n':
        print("Aborted.")
        return
    
    # Execute migration
    cmd_migrate(chat_id, user)
    
    # Also symlink global memory files for CLI
    udir = user_dir(user)
    global_mem = MEMORIES_DIR / "MEMORY.md"
    global_user = MEMORIES_DIR / "USER.md"
    
    if global_mem.exists() and not global_mem.is_symlink():
        # Backup original
        backup = MEMORIES_DIR / "MEMORY.md.backup"
        if not backup.exists():
            shutil.copy2(global_mem, backup)
            print(f"[BACKUP] {backup}")
    
    if global_user.exists() and not global_user.is_symlink():
        backup = MEMORIES_DIR / "USER.md.backup"
        if not backup.exists():
            shutil.copy2(global_user, backup)
            print(f"[BACKUP] {backup}")
    
    # Create symlinks
    if global_mem.exists() or global_mem.is_symlink():
        global_mem.unlink()
    global_mem.symlink_to(f"user_{user}/MEMORY.md")
    print(f"[LINK] {global_mem} → user_{user}/MEMORY.md")
    
    if global_user.exists() or global_user.is_symlink():
        global_user.unlink()
    global_user.symlink_to(f"user_{user}/USER.md")
    print(f"[LINK] {global_user} → user_{user}/USER.md")
    
    print(f"\n✅ Auto-setup complete!")
    print(f"   Owner '{user}' now has unified memory across all channels.")
    print(f"   CLI and Telegram (chat_id {chat_id}) share the same data.")
    print(f"\n   To add more channels later:")
    print(f"   python3 user_mapper.py map --chat-id <NEW_ID> --user {user}")


def cmd_map(chat_id: str, user_id: str):
    """Map a chat_id to a user via symlink."""
    mapping = load_mapping()
    udir = user_dir(user_id)
    cdir = chat_dir(chat_id)

    # Create user directory if it doesn't exist
    udir.mkdir(parents=True, exist_ok=True)

    # Ensure user has at least empty files
    for fname in ["MEMORY.md", "USER.md"]:
        fpath = udir / fname
        if not fpath.exists():
            fpath.write_text(f"# {fname} for {user_id}\n\n")

    # If chat_id directory exists and is NOT a symlink, migrate data first
    if cdir.exists() and not cdir.is_symlink():
        print(f"[MIGRATE] {cdir} exists and is not a symlink. Migrating data...")
        for item in cdir.iterdir():
            dest = udir / item.name
            if not dest.exists():
                if item.is_file():
                    shutil.copy2(item, dest)
                    print(f"  Copied: {item.name}")
                elif item.is_dir():
                    shutil.copytree(item, dest)
                    print(f"  Copied dir: {item.name}")
        shutil.rmtree(cdir)
        print(f"  Removed: {cdir}")

    # Create symlink
    if cdir.exists() or cdir.is_symlink():
        cdir.unlink()
    cdir.symlink_to(udir)
    print(f"[LINK] {cdir} → {udir}")

    # Update mapping
    mapping[chat_id] = user_id
    save_mapping(mapping)

    # Also create reverse mapping
    reverse_key = f"_user_{user_id}_chat_ids"
    if reverse_key not in mapping:
        mapping[reverse_key] = []
    if chat_id not in mapping[reverse_key]:
        mapping[reverse_key].append(chat_id)
    save_mapping(mapping)

    print(f"[OK] chat_id {chat_id} → user '{user_id}'")


def cmd_unmap(chat_id: str):
    """Remove a mapping, revert to isolated directory."""
    mapping = load_mapping()
    cdir = chat_dir(chat_id)

    if chat_id not in mapping:
        print(f"[ERROR] No mapping found for chat_id {chat_id}")
        return

    user_id = mapping[chat_id]
    udir = user_dir(user_id)

    # Remove symlink
    if cdir.is_symlink():
        cdir.unlink()
        print(f"[UNLINK] Removed symlink: {cdir}")

    # Copy data back to chat_id directory
    if udir.exists():
        shutil.copytree(udir, cdir)
        print(f"[COPY] Copied user data back to {cdir}")

    # Update mapping
    del mapping[chat_id]
    reverse_key = f"_user_{user_id}_chat_ids"
    if reverse_key in mapping and chat_id in mapping[reverse_key]:
        mapping[reverse_key].remove(chat_id)
    save_mapping(mapping)

    print(f"[OK] chat_id {chat_id} unmapped from user '{user_id}'")


def cmd_list():
    """List all mappings."""
    mapping = load_mapping()

    if not mapping:
        print("No mappings found.")
        print(f"Mapping file: {MAPPING_FILE}")
        print(f"\nRun: python3 user_mapper.py auto-setup")
        return

    print("=== User Mappings ===\n")
    users = {}
    for chat_id, user_id in mapping.items():
        if chat_id.startswith("_"):
            continue
        if user_id not in users:
            users[user_id] = []
        users[user_id].append(chat_id)

    for user_id, chat_ids in users.items():
        udir = user_dir(user_id)
        exists = "✅" if udir.exists() else "❌"
        print(f"  {exists} User: {user_id}")
        for cid in chat_ids:
            cdir = chat_dir(cid)
            is_link = "→" if cdir.is_symlink() else "  "
            print(f"    {is_link} chat_id: {cid}")
        
        # Check global symlinks
        global_mem = MEMORIES_DIR / "MEMORY.md"
        if global_mem.is_symlink():
            target = global_mem.resolve()
            if target.parent == udir:
                print(f"    → global MEMORY.md (CLI)")
        print()


def cmd_resolve(chat_id: str):
    """Show which user a chat_id belongs to."""
    mapping = load_mapping()

    if chat_id in mapping:
        user_id = mapping[chat_id]
        udir = user_dir(user_id)
        print(f"chat_id {chat_id} → user '{user_id}'")
        print(f"  Directory: {udir}")
        print(f"  Exists: {udir.exists()}")
        if udir.exists():
            files = list(udir.iterdir())
            print(f"  Files: {len(files)}")
            for f in files:
                print(f"    {f.name}")
    else:
        print(f"chat_id {chat_id} → no mapping (isolated)")
        cdir = chat_dir(chat_id)
        if cdir.exists():
            print(f"  Directory: {cdir}")
            files = list(cdir.iterdir())
            print(f"  Files: {len(files)}")


def cmd_migrate(chat_id: str, user_id: str):
    """Migrate existing chat_id data into user directory, then link."""
    cdir = chat_dir(chat_id)
    udir = user_dir(user_id)

    if not cdir.exists():
        print(f"[ERROR] chat_id directory doesn't exist: {cdir}")
        return

    if cdir.is_symlink():
        print(f"[WARN] {cdir} is already a symlink. Re-linking...")
        cmd_map(chat_id, user_id)
        return

    # Create user dir
    udir.mkdir(parents=True, exist_ok=True)

    # Copy all files from chat_id to user dir
    migrated = 0
    for item in cdir.iterdir():
        dest = udir / item.name
        if dest.exists():
            print(f"  SKIP (exists): {item.name}")
            continue
        if item.is_file():
            shutil.copy2(item, dest)
            print(f"  Copied: {item.name}")
            migrated += 1
        elif item.is_dir():
            shutil.copytree(item, dest)
            print(f"  Copied dir: {item.name}")
            migrated += 1

    # Remove old dir and create symlink
    shutil.rmtree(cdir)
    cdir.symlink_to(udir)

    # Update mapping
    mapping = load_mapping()
    mapping[chat_id] = user_id
    save_mapping(mapping)

    print(f"\n[OK] Migrated {migrated} items from chat_id {chat_id} to user '{user_id}'")
    print(f"     {cdir} → {udir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "auto-setup":
        cmd_auto_setup()

    elif cmd == "map":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--chat-id", required=True)
        p.add_argument("--user", required=True)
        args = p.parse_args(sys.argv[2:])
        cmd_map(args.chat_id, args.user)

    elif cmd == "unmap":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--chat-id", required=True)
        args = p.parse_args(sys.argv[2:])
        cmd_unmap(args.chat_id)

    elif cmd == "list":
        cmd_list()

    elif cmd == "resolve":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--chat-id", required=True)
        args = p.parse_args(sys.argv[2:])
        cmd_resolve(args.chat_id)

    elif cmd == "migrate":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--chat-id", required=True)
        p.add_argument("--user", required=True)
        args = p.parse_args(sys.argv[2:])
        cmd_migrate(args.chat_id, args.user)

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: auto-setup, map, unmap, list, resolve, migrate")
        sys.exit(1)
