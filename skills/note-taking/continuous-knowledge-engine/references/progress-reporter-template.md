# Progress Reporter Script Template

Script for generating daily progress reports from the knowledge base.

## Usage

```bash
KNOWLEDGE_BASE="${HOME}/knowledge-base"
python3 "${KNOWLEDGE_BASE}/scripts/progress-reporter.py" [morning|daily|night]
```

## Script Structure

```python
#!/usr/bin/env python3
"""Continuous Knowledge Engine - Progress Reporter

Generates morning briefings, daily summaries, and night reports
from knowledge base activity logs and Obsidian notes.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

KNOWLEDGE_BASE = os.environ.get(
    "KNOWLEDGE_BASE",
    os.path.expanduser("~/knowledge-base")
)

def count_files(directory, glob_pattern="*.md"):
    """Count files matching a pattern in a directory."""
    d = Path(KNOWLEDGE_BASE) / directory
    return len(list(d.glob(glob_pattern))) if d.exists() else 0

def count_git_commits():
    """Count git commits in the knowledge base."""
    import subprocess
    os.chdir(KNOWLEDGE_BASE)
    result = subprocess.run(
        ["git", "log", "--oneline"],
        capture_output=True, text=True
    )
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

def generate_report(mode="daily"):
    """Generate a progress report."""
    now = datetime.now()
    
    discord_threads = count_files("discord/threads", "*.json")
    youtube_notes = count_files("youtube/transcripts")
    obsidian_notes = count_files("obsidian", "*.md")
    commits = count_git_commits()
    
    if mode == "morning":
        title = f"🌅 Morning Report — {now.strftime('%Y-%m-%d')}"
    elif mode == "night":
        title = f"🌙 Night Report — {now.strftime('%Y-%m-%d')}"
    else:
        title = f"📊 Daily Report — {now.strftime('%Y-%m-%d %H:%M')}"
    
    report = f"""{title}

## Knowledge Base Status
- Discord threads: {discord_threads}
- YouTube transcripts: {youtube_notes}
- Obsidian notes: {obsidian_notes}
- Git commits: {commits}

## Recent Activity
"""
    return report

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    print(generate_report(mode))
```

## Integration with Hermes

```python
cronjob(
    action='create',
    name='knowledge-morning-report',
    schedule='0 6 * * *',
    deliver='local',
    prompt='Run the progress reporter and output the morning briefing.'
)
```
