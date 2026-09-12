#!/usr/bin/env python3
"""
Push Ruflo skill content to TencentDB via Hermes memory_tencentdb_v2 provider.

Usage:
    python3 push_ruflo_to_tencentdb.py              # Push all skill content
    python3 push_ruflo_to_tencentdb.py --search     # Verify by searching
    python3 push_ruflo_to_tencentdb.py --list       # List stored entries
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure we use Hermes' Python environment
HERMES_VENV = Path.home() / ".hermes" / "hermes-agent" / "venv"
if HERMES_VENV.exists():
    os.environ["PATH"] = f"{HERMES_VENV}/bin:{os.environ.get('PATH', '')}"

# Set HERMES_HOME for the script
os.environ["HERMES_HOME"] = str(Path.home() / ".hermes")

# Add hermes-agent to path
sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))

from tools.registry import registry


def call_tool(tool_name: str, args: dict) -> str:
    """Call a Hermes tool by name with arguments."""
    try:
        result = registry.dispatch(tool_name, args, task_id="push-skill-to-tdb")
        return result
    except Exception as e:
        return f"Error calling {tool_name}: {e}"


def push_skill_content():
    """Push the Ruflo skill content to TencentDB via memory tool."""
    
    # Read the skill content
    skill_dir = Path.home() / ".hermes" / "skills" / "ruflo-workflows"
    if not skill_dir.exists():
        print(f"Skill not found at {skill_dir}")
        return False
    
    skill_md = skill_dir / "SKILL.md"
    api_ref = skill_dir / "references" / "ruflo-api.md"
    swarm_script = skill_dir / "scripts" / "ruflo-swarm.sh"
    
    content_parts = []
    
    if skill_md.exists():
        content_parts.append(f"=== SKILL.md ===\n{skill_md.read_text()}")
    
    if api_ref.exists():
        content_parts.append(f"\n=== references/ruflo-api.md ===\n{api_ref.read_text()}")
    
    if swarm_script.exists():
        content_parts.append(f"\n=== scripts/ruflo-swarm.sh ===\n{swarm_script.read_text()}")
    
    full_content = "\n".join(content_parts)
    
    print(f"Pushing {len(full_content)} chars to TencentDB...")
    
    # Use the memory tool to store - this gets mirrored to TencentDB via on_memory_write
    result = call_tool("memory", {
        "action": "add",
        "target": "skill",
        "content": f"Ruflo Workflows Skill\n\n{full_content}"
    })
    
    print(f"Memory tool result: {result}")
    
    # Also use the TencentDB provider's explicit search tool if available
    # This validates the provider is working
    search_result = call_tool("tdai_memory_search", {
        "query": "ruflo swarm",
        "limit": 3
    })
    print(f"TencentDB search test: {search_result}")
    
    return True


def verify_stored():
    """Verify the skill content was stored in TencentDB."""
    queries = [
        "ruflo workflows",
        "ruflo swarm",
        "ruflo federation",
        "ruflo memory",
        "ruflo goal",
    ]
    
    print("\n=== Verifying stored content ===")
    for query in queries:
        result = call_tool("tdai_memory_search", {
            "query": query,
            "limit": 3
        })
        print(f"\nQuery: '{query}'")
        print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")


def list_all_memories():
    """List all memories in TencentDB."""
    result = call_tool("tdai_memory_search", {
        "query": "",
        "limit": 20
    })
    print(f"\nAll memories (limit 20):\n{result}")


def push_via_delegate():
    """Alternative: Use delegate_task to have a subagent push the content."""
    
    result = call_tool("delegate_task", {
        "goal": (
            "Read the Ruflo skill at ~/.hermes/skills/ruflo-workflows/ and "
            "store its complete content (SKILL.md, references/ruflo-api.md, scripts/ruflo-swarm.sh) "
            "into TencentDB using the memory tool. Then verify by searching with "
            "tdai_memory_search for 'ruflo swarm'."
        ),
        "context": (
            "Available tools: memory (action=add, target=skill), tdai_memory_search, "
            "tdai_conversation_search, read_file. "
            "The memory_tencentdb_v2 provider mirrors memory tool writes to TencentDB L0 automatically."
        )
    })
    
    print(f"Delegate task result: {result}")
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Push Ruflo skill to TencentDB")
    parser.add_argument("--search", action="store_true", help="Verify by searching")
    parser.add_argument("--list", action="store_true", help="List all memories")
    parser.add_argument("--delegate", action="store_true", help="Use delegate_task subagent")
    args = parser.parse_args()
    
    if args.search:
        verify_stored()
        return
    
    if args.list:
        list_all_memories()
        return
    
    if args.delegate:
        push_via_delegate()
        return
    
    # Default: push content directly
    success = push_skill_content()
    if success:
        print("\nWaiting for async write to propagate...")
        time.sleep(2)
        verify_stored()


if __name__ == "__main__":
    main()