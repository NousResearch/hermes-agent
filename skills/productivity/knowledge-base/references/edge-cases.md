# Edge Cases

Load when the standard workflow needs a branch. Each item states what to do.

## Instagram URL routing

| Path | Action |
|------|--------|
| `/reel/` | Video pipeline — `pipeline-instagram.md` |
| `/p/` | Carousel — `scripts/ig-carousel-extract.py` |

When the user's label disagrees with the URL path, confirm before proceeding.

## Cookie validation

Before Reel downloads:

```bash
grep -c sessionid /root/.hermes/cookies/ig_cookies.txt   # must be ≥ 1
```

A file with only the Netscape header has no cookies — re-export from Chrome using a Reel URL.

Use cookies for every Reel in a batch (Instagram may allow the first 1–2 without auth, then require login).

## Music-only Reels

When transcription segments total < 50 characters:

1. Use caption/metadata via Googlebot UA as primary content
2. Annotate the note: `⚠️ Music-only Reel — analysis based on caption text`
3. Proceed to note creation with caption content

## Kanban parent/child delegation

When a URL moves to a child ticket:

1. Child ticket owns that URL only
2. Parent ticket **body** updated — delegated URL removed or marked `DELEGATED → child <id>`
3. Parent comment + handoff.md track `DONE` / `DELEGATED` / `TO-DO`
4. Resuming parent: read comments and handoff before the next URL

Full template: `kanban-ticket-template.md` (Parent/child delegation).

## Section labels

Content follows source language. Section headers follow the template — English (`Summary`, `The Claim`, `Context / Analysis`, `See Also`).

## Note location

Save at `Knowledge base/<slug>.md`. Categorize with `tags` in frontmatter.

## marker-pdf install

After installing `marker-pdf`, re-pin hermes-agent packages per `dependencies.md`.

## Transcription persistence

Short Reels (≤60s): embed in note. Long videos (>2min): upload to MinIO.
Details: `video-pipeline-global.md`.
