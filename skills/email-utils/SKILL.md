---
name: email-utils
description: 'Shared utilities for email-processing cron precheck scripts: people
  resolution, episodic memory, cross-cron dedup, action quality logging, and TF-IDF
  topic clustering.'
version: 1.0.0
author: Hermes Agent
license: MIT
platforms:
- linux
- macos
metadata:
  hermes:
    tags:
    - email
    - cron
    - memory
    - episodic
    - relationship-graph
    - productivity
    category: productivity
    created_by: agent
    config:
    - key: email-utils.enabled
      description: Enable email-utils skill behavior
      default: true
      prompt: Enable email-utils skill?
---
---

# Email Utils — Shared Utilities for Email Processing

Provides five reusable components for Hermes Agent email-processing cron scripts.
All components are pure Python stdlib — no external dependencies. All data is
local-only (privacy class B). Never written to durable Hermes memory.

## When to use

- Building or improving email-processing cron precheck scripts
- Adding people/relationship resolution to email workflows
- Tracking email interaction episodes over time
- Preventing multiple crons from double-surfacing the same thread
- Logging draft outcomes for quality feedback loops
- Grouping email subjects into topic clusters without external ML libraries

## Components

### 1. PeopleResolver

Resolves email addresses and names against a `people.yaml` file (JSON-format
YAML with stable `person_id`, aliases, and circle memberships).

Returns: `person_id`, `display_name`, `circle_ids`, `sensitivity`, `is_self`,
`is_known`, `priority_hint`, `style_hint`.

Falls back gracefully — unknown senders get `person_id=None` and `is_known=False`.

```python
from email_utils import PeopleResolver

resolver = PeopleResolver()
ctx = resolver.resolve(email="someone@example.com")
# → {"person_id": "person_ed_johnson", "circle_ids": ["circle_fortified_strength"], ...}

# Or enrich a message dict in-place:
msg = {"from": "Ed Johnson <coach@your-org.org>", "subject": "Test"}
resolver.enrich_message(msg)
# msg now has person_id, person_name, circle_ids, sender_is_known, priority_hint, style_hint
```

**people.yaml format** (JSON stored in .yaml file):
```json
{
  "schema_version": 1,
  "people": [
    {
      "person_id": "person_ed_johnson",
      "display_name": "Ed Johnson",
      "aliases": {
        "names": ["Ed Johnson"],
        "emails": ["coach@your-org.org", "coach.alt@example.com"]
      },
      "circle_ids": ["circle_fortified_strength"]
    }
  ]
}
```

**circles.yaml format**:
```json
{
  "circles": [
    {
      "circle_id": "circle_fortified_strength",
      "default_priority": "high",
      "default_response_style": "concise_professional"
    }
  ]
}
```

### 2. EpisodeStore

Lightweight episodic memory for email threads. Each thread becomes an episode
linked to person_ids, with a status lifecycle and temporal decay.

```python
from email_utils import EpisodeStore

store = EpisodeStore()

# Create/update an episode
store.upsert("personal", "thread123",
    person_id="person_ed_johnson",
    subject="Spring fundraiser venue",
    status="needs_action",
    message_count=3)

# Get context for a thread (returns None if no episode)
ctx = store.get_episode_context("personal", "thread123")
# → {"episode_id": "ep_thread123", "status": "needs_action", "action_summary": "", ...}

# Find episodes awaiting reply for 3+ days
awaiting = store.query_awaiting_reply(older_than_days=3)

# Get recent episodes for a person
ed_eps = store.query_by_person("person_ed_johnson", limit=10)
```

**Episode lifecycle**: `active` → `needs_action` → `awaiting_reply` → `resolved` → `stale` → `archived`

**Retention**: 365 days, 500-episode cap (oldest resolved archived first).

**Privacy**: Class B — local-only JSON, no raw email bodies, metadata-level summaries only.

### 3. RecentlySurfaced

Cross-cron dedup state. Prevents multiple crons (e.g. triage every 30m + sweep
daily) from double-surfacing the same email thread within a 12-hour window.

```python
from email_utils import RecentlySurfaced

rs = RecentlySurfaced()

# Check if thread was recently surfaced by another cron
if rs.check("personal", "thread123"):
    print("Already surfaced — skip")

# Mark as surfaced
rs.mark("personal", "thread123", "inbox_triage")
```

### 4. ActionQualityLog

Tracks draft outcomes (sent/discarded/edited) for email-processing feedback.
90-day retention. Provides summary stats.

```python
from email_utils import ActionQualityLog

log = ActionQualityLog()

# Record an outcome
log.record(account="personal", thread_id="t123",
    action="draft_created", outcome="pending",
    cron_source="inbox_triage")

# Get stats
stats = log.stats()
# → {"total": 42, "by_outcome": {"sent_by_jim": 15, "discarded": 8, ...}}
```

### 5. TopicClusterer

Pure-Python TF-IDF topic clustering for email subjects. No external dependencies
(no numpy, no sklearn, no sentence-transformers). Groups similar subject lines
by cosine similarity on TF-IDF vectors.

```python
from email_utils import TopicClusterer

clusterer = TopicClusterer(min_similarity=0.30)
clusterer.fit([
    "Tax filing deadline extension",
    "IRS tax extension request",
    "Spring fundraiser venue selection",
    "Fundraiser budget approval needed",
    "Kayaking trip invitation",
])
clusters = clusterer.cluster()
# → {0: [0, 1], 1: [2, 3], 2: [4]}  (tax threads grouped, fundraiser threads grouped)

for cid, indices in clusters.items():
    if len(indices) >= 2:
        label = clusterer.label_for_cluster(indices)
        print(f"Cluster: '{label}' — {len(indices)} subjects")
```

## Installation

### For your own Hermes deployment

Copy `scripts/email_utils.py` to your `$HERMES_HOME/scripts/` directory:

```bash
cp scripts/email_utils.py $HERMES_HOME/scripts/
chmod 644 $HERMES_HOME/scripts/email_utils.py
```

Then import from your precheck scripts:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__.environ.get("HERMES_HOME", "/opt/data")) / "scripts"))
from email_utils import PeopleResolver, EpisodeStore, RecentlySurfaced, ActionQualityLog, TopicClusterer
```

### Creating a people.yaml

If you don't have one yet, create a minimal `people.yaml` (JSON format):

```json
{
  "schema_version": 1,
  "people": [
    {
      "person_id": "self",
      "display_name": "Your Name",
      "aliases": {
        "names": ["Your Name"],
        "emails": ["you@example.com"]
      },
      "circle_ids": []
    }
  ]
}
```

The resolver will work without circles.yaml — it just won't provide
priority/style hints.

## Integration with precheck scripts

The typical integration pattern for a cron precheck script:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__.environ.get("HERMES_HOME", "/opt/data")) / "scripts"))
from email_utils import PeopleResolver, RecentlySurfaced, EpisodeStore

_people_resolver = PeopleResolver()
_recently_surfaced = RecentlySurfaced()
_episode_store = EpisodeStore()

def minimal(account, msg, drafted_threads=None):
    """Enrich a message with person context, thread state, and episode context."""
    _people_resolver.enrich_message(msg)
    ep_ctx = _episode_store.get_episode_context(account, msg.get('threadId', ''))
    return {
        'account': account,
        'id': msg.get('id'),
        'threadId': msg.get('threadId'),
        'from': msg.get('from'),
        'subject': msg.get('subject'),
        'person_id': msg.get('person_id'),
        'person_name': msg.get('person_name'),
        'circle_ids': msg.get('circle_ids', []),
        'sender_is_known': msg.get('sender_is_known', False),
        'priority_hint': msg.get('priority_hint'),
        'episode_context': ep_ctx,
    }

# After surfacing threads, mark them:
for msg in surfaced_messages:
    _recently_surfaced.mark(msg['account'], msg['threadId'], 'my_cron_name')
    _episode_store.upsert(msg['account'], msg['threadId'],
        person_id=msg.get('person_id'),
        subject=msg.get('subject', ''),
        status='active')
```

## Privacy

All components store data locally in `$HERMES_HOME/cron/state/` with 0600
permissions and atomic writes. No data ever leaves the machine. No raw email
bodies are stored — only metadata-level summaries. Nothing is written to
durable Hermes memory.

| Component | State file | Retention | Privacy class |
|-----------|-----------|-----------|---------------|
| PeopleResolver | Reads people.yaml/circles.yaml | N/A (read-only) | B |
| EpisodeStore | `cron/state/email_episodes.json` | 365 days, 500 cap | B |
| RecentlySurfaced | `cron/state/recently_surfaced.json` | 12 hours | B |
| ActionQualityLog | `cron/state/action_quality_log.json` | 90 days | B |
| TopicClusterer | No state (in-memory) | N/A | N/A |

## Requirements

- Python 3.10+ (uses `|` type union syntax)
- No external packages — pure stdlib
- `people.yaml` and `circles.yaml` for PeopleResolver (optional — resolver
  works without them, just returns `is_known=False` for everyone)

## References

- `scripts/email_utils.py` — the actual module (copy this to your deployment)
- See the `personal-context-integration` skill for the full relationship graph
  privacy model, approval gates, and schema design
- See the `script-first-cron-design` skill for how to build zero-LLM precheck
  scripts that use these utilities