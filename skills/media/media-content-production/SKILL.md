---
name: media-content-production
description: "Media umbrella: YouTube transcript extraction, GIF search/download, audio feature visualization, music generation, and song prompt workflows."
---

# Media Content Production

Use this skill for retrieving, transforming, analyzing, or generating media content.

## Universal workflow

1. Identify the media source, rights/usage constraints, target format, and quality requirements.
2. Prefer official APIs or deterministic CLI tools over browser scraping.
3. Download or generate a small sample first for long media jobs.
4. Verify the resulting file exists, opens, and has expected duration/format/metadata.

## YouTube content

- Fetch transcripts with metadata and timestamps when possible.
- Summaries should preserve source context, speaker/topic shifts, and timestamps for claims.
- If transcript extraction fails, report that directly and try alternate language/fallback methods.

## GIF search

- Use Tenor/API search with explicit query, rating/safety, and result size.
- Prefer preview/tinygif for chat reactions and full GIF for deliverables.
- Download and verify file type/size before attaching.

## Audio analysis

- Use spectrograms, mel/chroma/MFCC, and time slices to inspect audio structure.
- Report parameters (window, hop, sample rate, slice) with the visualization.

## Music generation and songwriting

- For lyrics: structure, rhyme/meter, emotional arc, and genre conventions matter more than long prompts.
- For generation tools: check hardware/API availability, generate a short sample first, and deliver prompt + audio artifact when possible.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
