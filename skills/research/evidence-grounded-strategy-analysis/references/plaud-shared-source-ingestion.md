# Plaud Shared-Source Ingestion

Use this when a user supplies a public `web.plaud.ai/s/...` recording link for evidence-grounded analysis.

## Why this matters

The visible **Highlights** are generated summaries, not primary evidence. Preserve the complete timestamped transcript before analysis so quotations, context, omissions, and contradictory statements remain auditable.

## Procedure

1. Open the supplied URL directly. Record the displayed title, recording date, and duration.
2. The page normally embeds a same-origin `/nshare/...` iframe. Select the **Transcript** tab and verify that the body expands substantially; do not mistake the initially visible Highlights pane for the transcript.
3. Identify the public access request made by the iframe:
   `https://api.plaud.ai/share/access/<public-share-id>`
   Fetch it without adding private account credentials. Treat the share ID as sensitive and never repeat it in reports or logs.
4. Preserve the complete JSON response as the immutable source artifact. In `data_file`:
   - `trans_result` is the raw timestamped transcript.
   - `transaction_polish` is Plaud's polished transcript variant.
   - `outline_result` supplies generated topic labels.
   - `notes_list` usually contains generated highlights; use these only as orientation, never as quotation evidence.
5. Render raw and polished transcript files with each segment's `start_time` and `end_time` converted from milliseconds to `HH:MM:SS`. Preserve segment boundaries exactly.
6. Write a manifest containing title, date, duration, segment count, source JSON hash, and hashes/byte counts for every rendered transcript.
7. Verify the displayed duration against API duration, confirm raw and polished segment counts, and inspect the first and last segments for truncation.
8. If an analysis system has a capture ceiling, split only at segment boundaries. Record each part's hash and verify concatenation reconstructs the canonical rendered transcript before analysis.

## Security and provenance

- Do not publish the public-share token, signed object-storage URLs, cookies, or temporary query signatures.
- Signed download links expire and are not durable provenance; the preserved source JSON hash is.
- Keep both raw and polished variants. Use polished text for readability only after checking important quotations against raw text or audio.
- Do not infer speaker names when diarization is missing or unreliable.
- If the shared page title conflicts with prior conversational context, name the actual source explicitly and treat the direct source as current evidence rather than silently relabeling it.

## Minimum verification record

```text
title: ...
date: YYYY-MM-DD
duration_ms: ...
segments_raw: ...
segments_polished: ...
source_sha256: ...
raw_transcript_sha256: ...
polished_transcript_sha256: ...
```
