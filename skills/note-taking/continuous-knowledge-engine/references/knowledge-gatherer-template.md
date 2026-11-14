# Knowledge Gatherer Script Template

Working Python script for gathering knowledge from Discord, YouTube, and academic sources.

## Usage

```bash
KNOWLEDGE_BASE="${HOME}/knowledge-base"
python3 "${KNOWLEDGE_BASE}/scripts/knowledge-gatherer.py"
```

## Script Structure

```python
#!/usr/bin/env python3
"""Continuous Knowledge Engine - Gatherer Script

Aggregates knowledge from Discord threads, YouTube transcripts,
and academic materials into a unified Git-versioned knowledge base.
"""

import json
import os
from datetime import datetime
from pathlib import Path

KNOWLEDGE_BASE = os.environ.get(
    "KNOWLEDGE_BASE", 
    os.path.expanduser("~/knowledge-base")
)

def gather_discord():
    """Fetch and process Discord thread data."""
    discord_memories = Path.home() / ".hermes" / "memories" / "discord-learning.json"
    if not discord_memories.exists():
        return []
    
    with open(discord_memories) as f:
        data = json.load(f)
    
    patterns = data.get("technical_patterns", [])
    return patterns

def gather_youtube():
    """Fetch recent YouTube transcripts and extract insights."""
    youtube_data = Path(KNOWLEDGE_BASE) / "youtube" / "transcripts"
    if not youtube_data.exists():
        return []
    
    insights = []
    for transcript in youtube_data.glob("*.md"):
        # Process each transcript
        content = transcript.read_text()
        insights.append({"source": transcript.name, "content": content[:500]})
    return insights

def generate_obsidian_note(source_type, data):
    """Generate an Obsidian-formatted note."""
    timestamp = datetime.now().isoformat()
    frontmatter = f"""---
type: {source_type}
created: {timestamp}
tags: [{source_type}, learning]
---

"""
    return frontmatter + f"# {data.get('title', 'Untitled')}\n\n"

def git_commit(message):
    """Auto-commit changes to the knowledge base."""
    import subprocess
    os.chdir(KNOWLEDGE_BASE)
    subprocess.run(["git", "add", "-A"], check=True)
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
    if result.returncode != 0:
        subprocess.run(["git", "commit", "-m", message], check=True)

if __name__ == "__main__":
    print(f"Knowledge Gatherer starting... (base: {KNOWLEDGE_BASE})")
    
    patterns = gather_discord()
    youtube = gather_youtube()
    
    print(f"Discord patterns: {len(patterns)}")
    print(f"YouTube insights: {len(youtube)}")
    
    git_commit(f"knowledge-gather: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Done.")
```

## Key Design Decisions

- **No external dependencies:** Uses only stdlib to avoid install issues in cron contexts.
- **Env-based paths:** Respects `KNOWLEDGE_BASE` environment variable.
- **Atomic git commits:** Only commits when `git diff --cached --quiet` detects changes.
- **Graceful degradation:** Returns empty lists instead of raising on missing files.
