# Kanban Ticket Template for KB URL Batches

Use this template when creating tickets on the `default` board for the researcher
to process URLs into the knowledge base.

## Ticket creation command

```bash
# For video content (Reels, YouTube, Threads) — uses transcription pipeline
hermes kanban --board default create \
  --assignee researcher-videos \
  --skill knowledge-base \
  --max-runtime 3600 \
  --parent <previous_ticket_id> \
  --body "..." \
  "KB: <description> (lot X/N)"

# For text/image content (carousels, threads, posts) — no transcription needed
hermes kanban --board default create \
  --assignee researcher \
  --skill knowledge-base \
  --max-runtime 3600 \
  --parent <previous_ticket_id> \
  --body "..." \
  "KB: <description> (lot X/N)"
```

## Ticket body template

For Instagram-only batches:

```
Lot X/N — N reels Instagram. Rate-limit strictly: sleep 8-15s between Reels, max 2MB/s.

1. URL — Auteur/Topic (optional annotation)
2. URL (carousel — extraire tous les slides, HARD CAP 2 slides)
3. URL
...

For Instagram: use cookies at /root/.hermes/cookies/ig_cookies.txt. For carousel posts: extract all slides (HARD CAP 2). **Language: content in source language, labels in English.** Save to Knowledge base/. Push after each note.
```

For mixed batches (Instagram + Threads):

```
Batch X — N URLs. Rate-limit strictly: sleep 8-15s between Reels, max 2MB/s.

N. URL — Auteur/Topic (optional annotation)
...

For Instagram: use cookies at /root/.hermes/cookies/ig_cookies.txt. For carousel posts: extract all slides. For Threads: try web_extract or browser. **Language: content in source language, labels in English.** Save to Knowledge base/. Push after each note.
```

## Assignee selection

| Content type | Assignee | Why |
|---|---|---|
| Image posts, text threads, carousels | `researcher` | No transcription — metadata extraction, web_extract, or browser |
| Video (Instagram Reels, YouTube, Threads) | `researcher-videos` | Requires download → diarization → transcription pipeline. `max_spawn=1` for RAM. |

## Key elements

| Element | Purpose |
|---------|---------|
| `Rate-limit strictly: sleep 8-15s` | Avoids Instagram rate-limiting / shadow-ban |
| `max 2MB/s` | Throttle yt-dlp download speed |
| `cookies at /root/.hermes/cookies/ig_cookies.txt` | Required for Instagram authentication |
| `HARD CAP 2 slides` | Carousel anti-bot limitation — slides 3+ are blocked |
| `Keep original language` | Content in source language, labels in English |
| `Knowledge base/` | Target folder in Obsidian vault |
| `Push after each note` | Git push so Obsidian syncs |
| `--parent <id>` | Chain tickets so they process sequentially |
| `--max-runtime 3600` | 1h safety net per ticket |

## Convention

- 5 URLs per ticket max
- Chain with `--parent` so each batch waits for the previous one
- Skill: `knowledge-base`

## Parent/child delegation

When moving a URL (Reel, YouTube, Mega, etc.) to a child ticket:

1. **Create** the child ticket for that URL only
2. **Update the parent ticket body** — remove the delegated URL from the numbered list, or mark it `DELEGATED → child <child_id>`. The body drives execution; a comment alone is insufficient.
3. **Comment** on the parent: `"URL N delegated to child <child_id>"`
4. **handoff.md** — list each URL as `DONE`, `DELEGATED (child <id>)`, or `TO-DO`
5. **On resume** — read comments and handoff.md before processing the next URL

Duplicate transcription detection (`ps aux | grep transcribe`): `video-pipeline-global.md` (Pre-flight).

## Recently completed batch titles ("titre des done")

When the user asks for titles of recently processed content from completed batches:

```bash
cd "$OBSIDIAN_VAULT_PATH" && git log --oneline -20
# Then for range: git diff --name-only HEAD~N..HEAD | sort
```

Present note slugs as a bullet list. Don't re-read every note — quick inventory only.
