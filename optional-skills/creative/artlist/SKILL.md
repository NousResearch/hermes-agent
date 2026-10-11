---
name: artlist
description: "Search and download royalty-free music from Artlist via API."
version: 0.1.0
author: Limestudio90, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [artlist, music, royalty-free, audio, api, download]
---

# Artlist Skill

Search and download royalty-free music from Artlist (artlist.io) using its internal API for search and a browser step for the WAV download. It does not generate or edit music. Credential-free by design: the user signs in to artlist.io once in a persistent Playwright Chromium profile and the script reuses that session cookie — no token or personal data is stored in the skill.

## When to Use

- Sourcing royalty-free music tracks for a video or editing project
- Searching Artlist by mood, genre, or free-text, then bulk-downloading full-quality WAV files
- Don't use for: SFX, footage, or templates (different asset types and endpoints)

## Prerequisites

- Python with `requests` and `playwright`: `pip install requests playwright && playwright install chromium`
- An Artlist account, signed in once inside the persistent browser profile

## How to Run

- `python scripts/artlist.py search --profile <dir> --term "elegant" --category 62`
- `python scripts/artlist.py download --profile <dir> --url <song-url> --out <dir>`

## Quick Reference

- `search` — `--profile <dir> [--term TEXT] [--category ID] [--vocal INSTRUMENTAL]`
- `download` — `--profile <dir> --url <song-url> --out <dir>`

## Procedure

1. Log in to artlist.io in a persistent Playwright Chromium profile (one time). The `session-token` cookie lands in the profile's `state.json`.
2. Run `search` to list songs; confirm it returns `songId`, `songName`, and `artistName`.
3. Run `download` for each chosen song URL; the script clicks the `direct download` button and the `WAV` option, then saves the file.
4. Verify each `.wav` exists with a size in the tens of MB.

## Reference

### Auth

`GET https://artlist.io/api/auth/session` with the session cookie returns a short-lived `accessToken` (~30 min). Call the API with `Authorization: Bearer <accessToken>`; re-fetch the token each run.

### Search API

`POST https://search-api.artlist.io/v2/graphql` (Apollo server; introspection disabled). Query `SongList` with `page`, `take`, `songSortType`, `vocalType`, `categoryIds`, and `searchTerm`. Confirmed enums: `songSortType = "NEWEST"`, `vocalType = "INSTRUMENTAL"` (also `"VOCAL"`). Other sort values such as `RELEVANCE`, `POPULAR`, and `TOP` are rejected by the schema.

Common category IDs — mood: Uplifting=5, Happy=7, Love=9, Peaceful=10, Serious=12, Dramatic=13, Sexy=93, Dark=92, Mysterious=320; genre: Ambient=57, Cinematic=62, Classical=72, Electronic=64, Funk=89, Jazz=61, Lounge=97, Pop=69.

### Download

Song URL format: `https://artlist.io/royalty-free-music/song/<nameForURL>/<songId>`. Click the visible `[aria-label="direct download"]` button, then the element whose exact text is `WAV` in the popover. Capture with `page.expect_download()` and `save_as()`.

## Pitfalls

- Access token expires after ~30 min — re-fetch from `/api/auth/session` each run, never cache it.
- Cloudflare can challenge a fresh browser launch — keep `headless=False`; the user may need to click a challenge once.
- GraphQL introspection is disabled — use the queries shipped in `scripts/artlist.py`, do not introspect.
- Enum values are case-sensitive and a small fixed set (`NEWEST`, `INSTRUMENTAL`, `VOCAL`).
- `getUserAssetDownloadDetails` returns `{}` for a never-downloaded asset — that is not an error.

## Verification

- `search` returns a non-empty `songList.songs` with `songId` and `songName`.
- `download` produces a `.wav` file whose size is in the tens of MB.
