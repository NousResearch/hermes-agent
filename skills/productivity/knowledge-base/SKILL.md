---
name: knowledge-base
description: "Manage a personal knowledge base in the Obsidian vault: capture and structure information"
version: 1.7.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [knowledge-management, research, notes]
    related_skills: [obsidian]
---

# Knowledge Base

Personal knowledge base in the Obsidian vault at `Knowledge base/`.
Categorize with Obsidian `tags` in frontmatter.
Uses the `obsidian` skill for file operations. Git push after each note.

## Content sources

### YouTube

Pipeline: download → diarize → transcribe → summarize → archive with faster-whisper `large-v3`.

- Phase A: `references/pipeline-youtube.md`
- Phase B: `references/resume-prompt.md` + `references/youtube-note-template.md`
- Kanban pattern: same two-phase flow as Mega (`references/pipeline-mega.md`)

### Instagram

Route by URL path:

| Path | Pipeline |
|------|----------|
| `/reel/` | `references/pipeline-instagram.md` |
| `/p/` | `scripts/ig-carousel-extract.py` (first 2 slides; manual screenshots for slides 3+) |

When the user's description disagrees with the URL path, confirm the type first.

- Cookies: validate `sessionid` before Reel downloads (see `references/edge-cases.md`)
- Metadata: `references/instagram-extraction.md`
- Vision: `references/vision-pitfalls.md`

### Video (all platforms)

`references/video-pipeline-global.md` — background execution, mandatory diarization (pyannote), `large-v3` transcription, canonical scripts, rate limits, transcription persistence.

### Books (ePub/PDF)

`references/book-extraction.md` — extract text, read chapter by chapter, synthesize. Template: `templates/book-note-template.md`.

### Web search

`references/web-providers.md`

## Kanban batch processing

When the user drops URLs, create tickets on the `default` board.
See `references/kanban-ticket-template.md`.

- 5 URLs per ticket; chain with `--parent`
- 2 video transcriptions per worker session (`video-pipeline-global.md`)
- `--max-runtime 3600`
- Assignee: `researcher` (text/image) or `researcher-videos` (video)
- Worker setup: `references/researcher-profile-setup.md`

## When to add a note

User shares content → extract, verify when possible, save.

If `web_extract` is unavailable: `curl` + Googlebot UA (`references/instagram-extraction.md`) or yt-dlp for Reel transcripts (`references/pipeline-instagram.md`).

## Adding a note — workflow

1. **Load obsidian skill** (`skill_view(name='obsidian')`)
2. **Extract** core claim, fact, or insight
3. **Language:** content in source language; section labels per template (English)
4. **Verify** when possible — search, cross-reference
5. **Tag** via frontmatter (`tags`)
6. **Create** `Knowledge base/<slug>.md` via `OBSIDIAN_VAULT_PATH`
7. **Upload** source to MinIO when applicable — `references/minio-upload.md`
8. **Push:** `cd "$OBSIDIAN_VAULT_PATH" && git add -A && git commit -m "add: <slug>" && git push`
9. **Confirm** what was saved

Edge cases (cookies, music-only Reels, delegation): `references/edge-cases.md`

## Template

```markdown
---
date: YYYY-MM-DD
source: <platform/author, date>
source_url: <optional URL>
confidence: verified | plausible | emerging | debunked | untested
tags: [tag1, tag2, tag3]
---

# Title

## Summary
2-3 sentences. The essential up top.

## The Claim
What is claimed. Quote if relevant.

## Context / Analysis
Development, evidence, counter-arguments.

## Nuances
What the source omits, exaggerates, or distorts. Limitations.

## Reliability
✅ verified | ⚠️ plausible | 🔬 emerging | ❌ debunked | ❓ untested

## Sources
- Original source
- Consulted sources

## See Also
- [[Related note 1]]
- [[Related note 2]]
```

## Confidence levels

| Level | Meaning |
|-------|---------|
| ✅ verified | Confirmed by multiple solid sources |
| ⚠️ plausible | Logical, some sources, no consensus |
| 🔬 emerging | Preliminary, promising but limited |
| ❌ debunked | Contradicted by available evidence |
| ❓ untested | No source found |

## Retrieving information

When the user asks "qu'est-ce qu'on a sur X ?":
1. Load obsidian skill
2. `search_files(target='content', pattern='<keyword>', path='<vault>/Knowledge base/')`
3. Present findings grouped by confidence
4. Offer to research and add if nothing matches

Batch inventory ("titre des done"): `references/kanban-ticket-template.md`

## Working principles

- Complete every pipeline step — the user will wait for quality
- Document discoveries in skill files (skills over memory)
- Show non-trivial commands before running them

## Reference index

| Topic | File |
|-------|------|
| Edge cases | `references/edge-cases.md` |
| Video global rules | `references/video-pipeline-global.md` |
| YouTube pipeline | `references/pipeline-youtube.md` |
| Instagram pipeline | `references/pipeline-instagram.md` |
| Mega / external video | `references/pipeline-mega.md` |
| Video summarization | `references/resume-prompt.md` |
| YouTube note template | `references/youtube-note-template.md` |
| Books | `references/book-extraction.md` |
| Kanban tickets | `references/kanban-ticket-template.md` |
| Worker profiles | `references/researcher-profile-setup.md` |
| MinIO | `references/minio-storage.md`, `references/minio-upload.md` |
| Dependencies / fresh install | `references/dependencies.md`, `references/fresh-install-checklist.md` |
| Fact-checking | `references/fact-check-workflow.md` |
| Web search | `references/web-providers.md` |

Vault path: `OBSIDIAN_VAULT_PATH` (from `~/.hermes/.env`). Git sync: obsidian skill `references/git-sync.md`.
Persist transcripts and source files to MinIO
