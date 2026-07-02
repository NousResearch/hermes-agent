---
name: quarry
description: >-
  Public resource discovery engine. Find download routes for movies, TV, anime,
  music, software, and books across 28 sources (cloud drives, torrents, ebooks).
  Quality-aware ranking, link viability probing, structured JSON output.
  Use when the user asks to find, download, or compare releases for media content.
version: "1.1.0"
author: taffy-owo
license: MIT-0
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [media, resource-discovery, torrent, pan-links, ebook, download]
    requires_tools: [terminal]
---

# Quarry

Multi-source resource discovery engine. Searches 28 public sources across cloud drives (Aliyun, Quark, Baidu), torrents (YTS, Nyaa, EZTV, 1337x), and books (Anna's Archive, LibGen). Returns quality-ranked, deduplicated results with link viability verification.

## When to Use
- User wants to find a movie, TV show, anime, music, software, or book download
- User asks for pan links (阿里云盘, 夸克, 百度), magnets, or torrent results
- User wants to compare releases for quality (4K, 1080p, BluRay, REMUX, FLAC)
- Another tool or script needs structured search results

## Setup

```bash
git clone https://github.com/taffy-owo/quarry.git
cd quarry
# Optional performance extras
pip install httpx pycryptodome curl-cffi
```

## Quick Reference

```bash
QUARRY="path/to/quarry/scripts"

# Search (zero dependencies for basic use)
python3 "$QUARRY/hunt.py" search "Oppenheimer 2023" --4k --json
python3 "$QUARRY/hunt.py" search "The Boys S05E03" --tv
python3 "$QUARRY/hunt.py" search "Kamiina Botan" --anime
python3 "$QUARRY/hunt.py" search "Clean Code epub" --book
python3 "$QUARRY/hunt.py" search "Adobe Photoshop 2024" --software --channel pan

# Fast mode (<3s, top 6 sources)
python3 "$QUARRY/hunt.py" search "Interstellar" --fast --json

# Diagnostics
python3 "$QUARRY/hunt.py" doctor --json
python3 "$QUARRY/hunt.py" sources --probe --json
```

## Procedure

1. **Translate the query to English.** The engine only matches English release titles. CJK titles fail silently.
   - `黑袍纠察队` → `The Boys`
   - `三体` → `3 Body Problem`
   - `上伊那ぼたん` → `Kamiina Botan`

2. **Format the query using release naming conventions:**
   - TV: `{Title} S{XX}E{XX}` → `The Boys S05E03`
   - Movie: `{Title} {year}` → `Oppenheimer 2023`
   - Anime: `{Romanized Title}` → `Kamiina Botan`
   - Book: `{Title} {format}` → `Clean Code epub`

3. **Run the search** with the appropriate category flag:
   ```bash
   python3 "$QUARRY/hunt.py" search "The Boys S05E03" --tv --json
   ```

4. **Interpret results:**
   - `tier`: `top` = high confidence, `related` = decent, `risky` = unreliable
   - `source_health.link_alive`: `true` = verified, `false` = dead (skip), `null` = unknown
   - `confidence`: 0.0–1.0 match score

5. **Handle no results:** try alternative English titles or shortened names before reporting failure.

## Source Routing

| Category | Primary → Fallback |
|:---------|:-------------------|
| Movie | Pan → YTS/TorrentGalaxy/TPB → 1337x |
| TV | EZTV/TorrentGalaxy/TPB → Pan |
| Anime | Nyaa/DMHY/Bangumi Moe → Pan |
| Book | Anna's Archive → Libgen → Pan |
| Music | Pan → DMHY/Nyaa |
| Software | Pan → FitGirl/TorrentMac |

## Pitfalls
- **Never search with untranslated CJK titles** — the engine does NOT translate internally
- **Always check `link_alive` before presenting pan links** — skip `false` results
- **Do not use for private/login-only resources** — this skill handles public routes only
- **Book results are detail page URLs** — user visits to choose download format

## Verification
1. At least one result with `tier: "top"` exists
2. The `canonical_identity` matches the user's intent
3. For pan results: `source_health.link_alive` is `true`
