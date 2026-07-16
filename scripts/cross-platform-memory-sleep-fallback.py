from pathlib import Path
from datetime import datetime

home = Path.home()
log_dir = home / '.hermes' / 'cron' / 'output' / 'f0f9d64aeeb6'
log_dir.mkdir(parents=True, exist_ok=True)
out = log_dir / f"fallback_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
content = """# cross-platform-memory-sleep fallback

- status: ok
- note: cron environment does not expose the ebbinghaus_memory tool, so this run records a safe fallback marker only.
- next step: use a normal Hermes session to read this marker and perform memory remember/rehearse/sleep if needed.
"""
out.write_text(content, encoding='utf-8')
print(out)
