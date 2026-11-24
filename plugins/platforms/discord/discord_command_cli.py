#!/usr/bin/env python3
"""Export/import Discord slash command manifest for Git backup.

This script allows persisting the stable command manifest to a JSON file
that can be version-controlled in Git. It also supports importing to
verify or restore command state.
"""

import json
import sys
from pathlib import Path

# Add the hermes-agent to path
sys.path.insert(0, str(Path(__file__).parent))

from plugins.platforms.discord.discord_command_manifest import build_native_command_manifest


def export_manifest(output_path: Path) -> None:
    """Export the native command manifest to a JSON file."""
    manifest = build_native_command_manifest()
    data = {
        "version": 1,
        "source": "discord_command_manifest.py",
        "commands": manifest,
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Exported {len(manifest)} commands to {output_path}")


def import_manifest(input_path: Path) -> None:
    """Import and validate a command manifest from JSON."""
    data = json.loads(input_path.read_text())
    commands = data.get("commands", [])
    print(f"Imported {len(commands)} commands from {input_path}")
    
    # Validate structure
    for cmd in commands:
        required = ["name", "description"]
        for field in required:
            if field not in cmd:
                print(f"WARNING: Command missing '{field}': {cmd}")
    
    # Compare with current manifest
    current = build_native_command_manifest()
    current_names = {c["name"] for c in current}
    imported_names = {c["name"] for c in commands}
    
    if current_names == imported_names:
        print("✓ Command sets match exactly")
    else:
        added = imported_names - current_names
        removed = current_names - imported_names
        if added:
            print(f"  Added commands: {sorted(added)}")
        if removed:
            print(f"  Removed commands: {sorted(removed)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python discord_command_cli.py <export|import> [path]")
        print("  export [path]  - Export manifest to JSON (default: discord_commands.json)")
        print("  import [path]  - Import and validate manifest from JSON")
        sys.exit(1)
    
    action = sys.argv[1]
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("discord_commands.json")
    
    if action == "export":
        export_manifest(path)
    elif action == "import":
        import_manifest(path)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()