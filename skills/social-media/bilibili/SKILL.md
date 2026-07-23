---
name: bilibili
description: "Use when the user wants to search, browse, or get details about Bilibili (B站) videos — Chinese video platform. Primarily read-only via bili-cli (no login needed for search and metadata). Subtitle support via OpenCLI."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [bilibili, b站, chinese, video, search, trending, bili-cli]
    related_skills: [youtube-content, v2ex]
prerequisites:
  commands: [bili]
---

# Bilibili (B站) Skill

Bilibili is the dominant Chinese video/streaming platform. The `bili-cli` tool
provides read-only access to search, video metadata, trending, and rankings
with no login required for most operations.

## When to Use

- User asks to search Bilibili for videos by keyword
- User shares a Bilibili URL and wants video details (title, uploader, stats)
- User wants trending/hot videos or category rankings
- User wants audio from a Bilibili video for transcription

Don't use for: subtitles without OpenCLI (bili-cli can't extract them), or
writing comments/posting (requires login).

## Prerequisites

```bash
uv tool install bilibili-cli
# or: pipx install bilibili-cli
```

Verify: `bili hot -n 3`

## Quick Reference

```bash
# Search videos
bili search "keyword" --type video -n 5

# Video detail (supports BV ID or full URL)
bili video BVxxxxxxxxxx

# Hot / trending
bili hot -n 10

# Rankings
bili rank -n 10

# Download audio for transcription
bili audio BVxxxxxxxxxx

# Optional: login for dynamics/favorites
bili login
```

## Video Detail Output

`bili video BVxxx` returns: title, description, uploader name, duration,
publish date, play count, danmaku count, likes, coins, favorites, shares,
and subtitle availability flag.

## Curl Fallback (zero-dependency)

When bili-cli is unavailable, the Bilibili search API is accessible directly:

```bash
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
curl -s -c /tmp/bili_ck.txt -o /dev/null -A "$UA" "https://www.bilibili.com/"
curl -s -b /tmp/bili_ck.txt -A "$UA" -e "https://www.bilibili.com/" \
  "https://api.bilibili.com/x/web-interface/search/all/v2?keyword=QUERY&page=1"
```

Response is JSON with `data.result[]` — each result has `type` and `data` with
title, author, play count, danmaku, duration.

## Subtitles (OpenCLI, desktop only)

When OpenCLI is installed (Chrome extension, reuses browser login):

```bash
opencli bilibili subtitle BVxxxxxxxxxx
```

## Procedure

1. Determine intent: search, video detail, trending, or audio for transcription.
2. For search: `bili search "query" --type video -n 5`
3. For video detail: extract BV ID from URL (e.g. `BV1xx411c7mD`) and run `bili video BVxxx`
4. For trending: `bili hot -n 10`
5. If bili-cli is not installed, fall back to the curl search API.
6. If subtitles are needed and OpenCLI is available, use `opencli bilibili subtitle`.

## Common Pitfalls

1. **Never use yt-dlp for Bilibili.** B站 risk control fully blocks yt-dlp
   (412 errors in all configurations). Use bili-cli or the curl API fallback.
2. **Subtitles require OpenCLI.** bili-cli cannot extract subtitles on its own.
3. **bili-cli upstream may be stale.** Last updated 2026-03. If commands fail,
   try the curl fallback.
4. **Network latency.** Bilibili may be slow outside China. Consider a CN proxy
   if consistently slow.
5. **No comment reading.** bili-cli only reads video metadata, not comments.

## Verification Checklist

- [ ] `bili hot -n 3` returns trending videos
- [ ] `bili search "test" -n 2` returns search results
- [ ] The curl fallback API also works for the current network
