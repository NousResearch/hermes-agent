---
name: podcast-intel
description: Turn your Overcast listening history into actionable intelligence. Syncs episodes, transcripts, and chapters to SQLite, then uses LLM analysis to surface insights from what you've listened to and connect them to your current projects and interests. Depth of analysis is caller-configured.
version: 1.0.0
author: hbmartin
license: Apache-2.0
metadata:
  hermes:
    tags: [Podcasts, Overcast, SQLite, Transcripts, Intelligence, RSS]
    homepage: https://github.com/hbmartin/overcast-to-sqlite
prerequisites:
  commands: [uvx, uv]
---

# podcast-intel

Turns your Overcast listening history into a structured knowledge base, then surfaces insights
from recent episodes and connects them to your current work and interests.

Built on three tools by Harold Martin:
- overcast-to-sqlite  — https://github.com/hbmartin/overcast-to-sqlite
- podcast-transcript-convert — https://github.com/hbmartin/podcast-transcript-convert
- podcast-chapter-tools — https://github.com/hbmartin/podcast-chapter-tools

---

## One-Time Setup

### 1. Install the tools

```bash
uv tool install overcast-to-sqlite
uv tool install podcast-transcript-convert
```

### 2. Authenticate with Overcast

```bash
overcast-to-sqlite auth
# Logs into Overcast and saves an auth cookie to ./auth.json
# Your password is NOT saved — only the session cookie
```

Store auth.json somewhere stable, e.g. ~/.overcast/auth.json:
```bash
mkdir -p ~/.overcast
mv auth.json ~/.overcast/auth.json
```

### 3. Run the first full sync (takes a while the first time)

```bash
overcast-to-sqlite all -a ~/.overcast/auth.json ~/.overcast/overcast.db -v
```

This runs save → extend → transcripts → chapters sequentially.
First run downloads XML for every subscribed feed — may take several minutes.
Transcripts are saved to ~/.overcast/archive/transcripts/ by default.

---

## Daily Sync

Run this to pull in the latest listening activity:

```bash
overcast-to-sqlite all -a ~/.overcast/auth.json ~/.overcast/overcast.db
```

Or for a faster update (skips feed XML re-download):
```bash
overcast-to-sqlite save -a ~/.overcast/auth.json ~/.overcast/overcast.db
overcast-to-sqlite transcripts -a ~/.overcast/auth.json ~/.overcast/overcast.db
```

To fetch transcripts for starred episodes only:
```bash
overcast-to-sqlite transcripts -s -a ~/.overcast/auth.json ~/.overcast/overcast.db
```

> Suggested cron schedule: run `overcast-to-sqlite all` once daily, e.g. at 4am before
> any morning digest jobs that depend on it. Use launchd on macOS or cron on Linux.

---

## Querying Recent Listening

Use these SQL queries against ~/.overcast/overcast.db:

### Episodes played or significantly progressed in the last 24 hours
```sql
SELECT
  e.title,
  f.title AS podcast,
  e.overcastUrl,
  e.userRecommendedDate,
  e.transcriptDownloadPath,
  e.progress,
  e.played
FROM episodes e
JOIN feeds f ON e.feedId = f.overcastId
WHERE (e.played = 1 OR e.progress > 300)
  AND e.userUpdatedDate >= datetime('now', '-1 day')
ORDER BY
  e.userRecommendedDate DESC,
  e.userUpdatedDate DESC;
```

### Starred episodes with transcripts available
```sql
SELECT
  e.title,
  f.title AS podcast,
  e.overcastUrl,
  e.userRecommendedDate,
  e.transcriptDownloadPath
FROM episodes_starred e
JOIN feeds f ON e.feedId = f.overcastId
WHERE e.transcriptDownloadPath IS NOT NULL
ORDER BY e.userRecommendedDate DESC
LIMIT 20;
```

### Full-text search across chapter content
```sql
SELECT c.content, e.title, f.title AS podcast, c.time
FROM chapters_fts
JOIN chapters c ON chapters_fts.rowid = c.rowid
JOIN episodes e ON c.enclosureUrl = e.enclosureUrl
JOIN feeds f ON e.feedId = f.overcastId
WHERE chapters_fts MATCH 'your search term'
ORDER BY rank;
```

---

## Processing Transcripts

Transcripts are stored in mixed formats (SRT, WebVTT, HTML, JSON).
Use podcast-transcript-convert to normalize them to PodcastIndex JSON:

```bash
transcript2json ~/.overcast/archive/transcripts/ ~/.overcast/archive/transcripts-json/
```

To read a transcript as plain text for LLM analysis, parse the JSON:

```bash
python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
for seg in data.get('segments', []):
    print(seg.get('speaker', ''), seg.get('body', ''))
" ~/.overcast/archive/transcripts-json/episode.json
```

---

## LLM Analysis Workflow

When asked to analyze recent listening, follow this process:

### Step 1 — Query the DB for recent episodes
Run the last-24h query above using the terminal tool against ~/.overcast/overcast.db.

### Step 2 — Separate starred from non-starred
Episodes with a non-null userRecommendedDate are starred. Give these deeper treatment.

### Step 3 — Load and analyze transcripts
For each episode with a transcriptDownloadPath:
1. Read the transcript file (convert if needed using transcript2json)
2. Extract key concepts, claims, techniques, and names mentioned
3. Note timestamps/chapters where important ideas appear

For episodes without transcripts, use the episode description from episodes_extended.description.

### Step 4 — Cross-reference with user interests
Ask the user what they are currently working on and interested in, or read from context.
For each episode, identify:
- Direct connections to current projects or problems the user is solving
- Techniques or frameworks mentioned that could be applied
- People, papers, or tools referenced worth following up on
- Contrarian or surprising takes worth sitting with

### Step 5 — Format the output
Depth is determined by the user when invoking the skill. Default structure:

**[Starred] Episode Title — Podcast Name**
Summary: 2-3 sentence overview of what was covered
Key insight: The most actionable or interesting idea
Connections: How this relates to what the user is working on
Follow-up: Papers, people, tools, or questions worth pursuing

**[Played] Episode Title — Podcast Name**
One-line summary + any standout idea worth surfacing

---

## Listening Stats

```bash
overcast-to-sqlite stats ~/.overcast/overcast.db
```

Shows: total episodes played, total listening time, starred count, top podcasts by time.

---

## Searching Your History

```bash
overcast-to-sqlite search "reinforcement learning" ~/.overcast/overcast.db
overcast-to-sqlite search "agentic" ~/.overcast/overcast.db -l 5
```

Searches across episode titles, feed descriptions, and chapter content (FTS5).

---

## Database Location

Default: ~/.overcast/overcast.db
Default transcript archive: ~/.overcast/archive/transcripts/
Default auth: ~/.overcast/auth.json

Override any path via CLI flags. See overcast-to-sqlite --help for full options.

---

## Notes

- Transcripts are only available for episodes where the podcast publisher provides them
  via the podcast:transcript RSS tag. Not all episodes have transcripts.
- The extend command adds ~2MB per feed to the DB — expect a large file with many subscriptions
- auth.json contains only a session cookie, not your password. Rotate it via overcast-to-sqlite auth
- For starred-only transcript downloads use the -s flag on the transcripts command
- Chapter FTS5 search is a powerful way to find where a specific topic was discussed across
  your entire listening history without reading full transcripts
