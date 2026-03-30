"""CLI command for reviewing and managing persistent memory stores.

Exposes the existing MemoryStore (MEMORY.md / USER.md) to the command line
so users can inspect and revoke learned knowledge without starting an agent
session.  Partial implementation of #1156.
"""

import json
import sys


def memory_command(args):
    """Entry point for ``hermes memory`` subcommands."""
    from tools.memory_tool import MemoryStore

    store = MemoryStore()
    store.load_from_disk()

    action = getattr(args, "memory_action", None)

    if action is None or action == "list":
        _list_memories(store)
    elif action == "show":
        _show_memories(store, args.target)
    elif action == "delete":
        _delete_memory(store, args.target, args.substring, yes=args.yes)
    else:
        print("Unknown action. Run 'hermes memory --help' for usage.")
        sys.exit(1)


def _list_memories(store):
    """Show a summary of both memory stores."""
    for target, label in [
        ("memory", "MEMORY (agent notes)"),
        ("user", "USER (user profile)"),
    ]:
        entries = store._entries_for(target)
        current = store._char_count(target)
        limit = store._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        print(f"\n  {label}  [{pct}% \u2014 {current:,}/{limit:,} chars, {len(entries)} entries]")
        print(f"  {'─' * 50}")
        if not entries:
            print("  (empty)")
        else:
            for i, entry in enumerate(entries, 1):
                preview = entry.replace("\n", " ")[:80]
                if len(entry) > 80:
                    preview += "..."
                print(f"  {i:3d}. {preview}")
    print()


def _show_memories(store, target):
    """Display full entries from one store."""
    entries = store._entries_for(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
    label = "MEMORY (agent notes)" if target == "memory" else "USER (user profile)"

    print(f"\n  {label}  [{pct}% \u2014 {current:,}/{limit:,} chars]")
    print(f"  {'─' * 50}")
    if not entries:
        print("  (empty)")
    else:
        for i, entry in enumerate(entries, 1):
            print(f"\n  [{i}]")
            for line in entry.splitlines():
                print(f"  {line}")
    print()


def _delete_memory(store, target, substring, *, yes=False):
    """Remove an entry identified by a unique substring."""
    entries = store._entries_for(target)
    matches = [e for e in entries if substring in e]

    if not matches:
        print(f"No entry in '{target}' matched '{substring}'.")
        sys.exit(1)

    if len(matches) > 1:
        unique = set(matches)
        if len(unique) > 1:
            print(f"Multiple entries matched '{substring}'. Be more specific:")
            for m in matches:
                preview = m.replace("\n", " ")[:80]
                print(f"  \u2022 {preview}...")
            sys.exit(1)

    print(f"Will delete from {target}:")
    preview = matches[0].replace("\n", " ")[:120]
    print(f"  {preview}")

    if not yes:
        try:
            confirm = input("\nConfirm? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return
        if confirm != "y":
            print("Cancelled.")
            return

    result = store.remove(target, substring)
    if isinstance(result, str):
        result = json.loads(result)

    if result.get("success"):
        print(f"Deleted. {result.get('usage', '')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)
