---
name: continue
description: Cross-channel conversation continuation. User types /continue [platform] to pull context from another channel/session and resume work.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [continuation, sessions, cross-channel, gateway, tui]
triggers:
  - "/continue"
  - "/continue telegram"
  - "/continue discord"
  - "/continue tui"
  - "/continue local"
---

# /continue Command

Use this skill when the user wants to resume work from another Hermes channel or session.

## Core flow

When the user types `/continue` or `/continue <platform>`:

1. Detect the command.
2. If no platform is provided, scan recent sessions and show a channel picker.
3. If a platform is provided, go straight to that platform's latest session.
4. Summarize the last 5-6 user messages in 2-3 sentences.
5. Ask `Pick up there? (yes/no)`.
6. On `yes`, continue from that context.
7. On `no`, start fresh.

## Channel picker

If the user types `/continue` with no platform argument, scan `~/.hermes/sessions` and show the latest session for each expected platform.

- Treat a session file modified within the last 20 minutes as live.
- Label missing or `null` platform values as `local`.
- Support both platform names and numeric picks from the list.
- If only one platform has recent sessions, skip the picker and summarize directly.

### Scan script

Run this with `terminal`:

```python
import json, os, glob
from datetime import datetime, timezone
from collections import defaultdict

SESSION_DIR = os.path.expanduser('~/.hermes/sessions')
LIVE_THRESHOLD_MIN = 20
KNOWN_PLATFORMS = ['telegram', 'discord', 'tui', 'local']

def get_platform_jsonl(filepath):
    try:
        with open(filepath, 'r') as f:
            first = json.loads(f.readline())
        if first.get('role') == 'session_meta':
            return first.get('platform') or 'local'
        if 'mirror_source' in first:
            return first.get('mirror_source') or 'local'
    except Exception:
        pass
    return None

def get_platform_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get('platform') or 'local'
    except Exception:
        pass
    return None

sessions = []

for filepath in glob.glob(os.path.join(SESSION_DIR, '*.jsonl')):
    stat = os.stat(filepath)
    platform = get_platform_jsonl(filepath)
    if platform:
        sessions.append({
            'file': filepath,
            'platform': platform,
            'mtime': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            'format': 'jsonl',
        })

for filepath in glob.glob(os.path.join(SESSION_DIR, 'session_*.json')):
    stat = os.stat(filepath)
    platform = get_platform_json(filepath)
    if platform:
        sessions.append({
            'file': filepath,
            'platform': platform,
            'mtime': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            'format': 'json',
        })

platforms = defaultdict(list)
for s in sessions:
    platforms[s['platform']].append(s)
for p in platforms:
    platforms[p].sort(key=lambda x: x['mtime'], reverse=True)

now = datetime.now(timezone.utc)
for idx, p in enumerate(KNOWN_PLATFORMS, 1):
    sess_list = platforms.get(p, [])
    if not sess_list:
        print(f"{idx}. {p:<10} — No recent sessions")
        continue

    latest = sess_list[0]
    age_min = int((now - latest['mtime']).total_seconds() / 60)
    live = age_min < LIVE_THRESHOLD_MIN
    if live:
        print(f"{idx}. {p:<10} — Live now ({age_min}m ago) [live]")
    elif age_min < 60:
        print(f"{idx}. {p:<10} — Last: {age_min}m ago")
    else:
        print(f"{idx}. {p:<10} — Last: {age_min//60}h {age_min%60}m ago")
```

### Display format

```text
📋 Available channels:

1. telegram   — Live now (2m ago) [live]
2. discord    — No recent sessions
3. tui        — Last: 15m ago
4. local      — No recent sessions

Pick one (name or number):
```

## Summary generation

Handle both Hermes session formats:

- `*.jsonl` — gateway sessions such as Telegram and Discord. Read line-delimited JSON and inspect the first `session_meta` line for platform.
- `session_*.json` — TUI/local sessions. Read the top-level `platform` key and extract messages from the `messages` array.

### Summary extraction script

```python
import json, os, glob

SESSION_DIR = os.path.expanduser('~/.hermes/sessions')

def find_latest_session(platform):
    candidates = []

    for filepath in glob.glob(os.path.join(SESSION_DIR, '*.jsonl')):
        try:
            with open(filepath, 'r') as f:
                meta = json.loads(f.readline())
            detected = None
            if meta.get('role') == 'session_meta':
                detected = meta.get('platform') or 'local'
            elif 'mirror_source' in meta:
                detected = meta.get('mirror_source') or 'local'
            if detected == platform:
                candidates.append((filepath, os.stat(filepath).st_mtime, 'jsonl'))
        except Exception:
            pass

    for filepath in glob.glob(os.path.join(SESSION_DIR, 'session_*.json')):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if (data.get('platform') or 'local') == platform:
                candidates.append((filepath, os.stat(filepath).st_mtime, 'json'))
        except Exception:
            pass

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0], candidates[0][2]

def extract_recent_user_messages(filepath, file_format, limit=6):
    user_msgs = []

    if file_format == 'jsonl':
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    if msg.get('role') == 'user' and msg.get('content', '').strip():
                        content = msg['content'].strip()
                        if not content.startswith('[') and not content.startswith('SYSTEM'):
                            user_msgs.append(content)
                except Exception:
                    pass
    else:
        with open(filepath, 'r') as f:
            data = json.load(f)
        for msg in data.get('messages', []):
            if msg.get('role') == 'user' and msg.get('content', '').strip():
                content = msg['content'].strip()
                if not content.startswith('[') and not content.startswith('SYSTEM'):
                    user_msgs.append(content)

    return user_msgs[-limit:]
```

### Summary format

```text
📋 Last tui session (15m ago):

We were debugging the /continue workflow. We found that TUI sessions are stored as .json files while Telegram sessions are stored as .jsonl files, so the session scan needed to handle both formats.

Pick up there? (yes/no)
```

## Confirmation behavior

### If the user says yes

- Continue from that context.
- Inspect the session more deeply only if needed.
- Do not dump the full transcript unless the user asks for it.

### If the user says no

Reply:

```text
Got it. What would you like to work on instead?
```

## Edge cases

| Scenario | Handling |
|---|---|
| No sessions found | Say `No sessions found. Start fresh?` |
| Only one platform has sessions | Skip picker and summarize directly |
| Platform requested but no matching session file exists | Fall back to `session_search` |
| `session_search` also fails | Say `Can't find that session. Start fresh?` |
| User picks current platform | Summarize from current context instead of searching files |
| User replies with a number instead of a platform name | Map the number from the displayed picker to the selected platform |
| Old sessions with `platform: null` | Treat as `local` |
| TUI/local sessions stored as `.json` | Scan `session_*.json` and read the `messages` array |

## Implementation checklist

- Parse the optional platform argument.
- If no platform is provided, scan sessions and show a channel picker.
- Include expected platforms even when some have no recent sessions.
- Support both platform names and numeric picks.
- If a platform is provided, locate the latest session for that platform.
- Support both `.jsonl` and `.json` session formats.
- Read only the last 5-6 user messages.
- Summarize in 2-3 sentences.
- Ask `Pick up there? (yes/no)`.
- Continue on `yes`; start fresh on `no`.

## Notes

- Keep the summary short.
- Skip tool output, code blocks, and system/invoke messages.
- Prefer the most recent live session when multiple sessions exist for one platform.
- If the user typed `/continue <platform>` directly, do not show the picker first.
