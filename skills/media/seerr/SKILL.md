---
name: seerr
description: Request movies and TV shows for your media server.
version: 3.0.0
author: nyakojiru (https://github.com/nyakojiru), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [seerr, movies, tv, media-requests, plex, jellyfin, emby, radarr, sonarr]
    category: media
    config:
      - key: seerr.url
        description: Base URL of the Seerr server, e.g. http://localhost:5055
        prompt: Seerr base URL
        default: http://localhost:5055
required_environment_variables:
  - name: SEERR_URL
    prompt: Seerr base URL (e.g. http://localhost:5055)
    help: The URL you open Seerr on. Also settable as skills.config.seerr.url in config.yaml.
    required_for: Every Seerr call
  - name: SEERR_API_KEY
    prompt: Seerr API key
    help: Seerr -> Settings -> General -> API Key
    required_for: Every Seerr call
---

# Seerr Skill

Conversational media-request assistant for [Seerr](https://github.com/seerr-team/seerr),
the request manager that Overseerr and Jellyseerr merged into. Search a title,
pick a library, submit the request, and report download progress back.

Seerr proxies Radarr and Sonarr, so root folders, quality profiles and download
progress all come from Seerr — no Radarr/Sonarr credentials are needed for the
core flow. This skill submits requests; it does not approve them or manage users.

## When to Use

- Someone wants to watch, download, get, or request a movie or TV show.
- Someone asks whether something they requested has finished downloading.
- Someone asks how much disk space is left.

## When NOT to Use

- Approving/denying requests or changing permissions — use the Seerr web UI.
- Radarr/Sonarr administration (indexers, custom formats, quality definitions).

## Prerequisites

- A reachable Seerr server with Radarr and/or Sonarr already connected to it.
- `SEERR_URL` and `SEERR_API_KEY` in the environment. Copy the block from
  `.env.example` into your `.env` and fill it in — that file is dotenv
  (`KEY=value`), not YAML.
- The non-secret URL may instead live in `~/.hermes/config.yaml` as real YAML:

  ```yaml
  skills:
    config:
      seerr.url: http://localhost:5055
  ```

  Keep `SEERR_API_KEY` out of `config.yaml`; secrets belong in `.env`.
- **Optional** — `RADARR_URL` / `RADARR_API_KEY` and `SONARR_URL` /
  `SONARR_API_KEY`. These are needed *only* for `diskspace` and `queue`, the two
  things Seerr cannot report. Everything else works without them.
- Python 3.9+ (stdlib only).

If Hermes has not exported the variables into the shell, `scripts/seerr.py`
falls back to sourcing `~/.hermes/.env` on its own, so it also works when run
by hand.

## How to Run

Use the `terminal` tool. `scripts/seerr.py` is the **only** API surface — never
construct a curl command inline; the script handles percent-encoding, headers,
the correct request fields, and response parsing.

```
python3 scripts/seerr.py <command> [options]
```

Add `--json` to any command for the raw payload. The script exits non-zero and
prints a one-line `error:` on failure.

## Quick Reference

| Command | Purpose |
|---|---|
| `search "<title>" [--type movie\|tv]` | Numbered results with TMDB ids |
| `servers radarr\|sonarr` | Servers, **root folders**, quality profiles |
| `seasons <tmdb_id>` | Season list (Specials excluded) |
| `request --type movie --tmdb-id <id> [--root-folder <path>]` | Request a movie |
| `request --type tv --tmdb-id <id> --seasons 1,2\|all [--root-folder <path>]` | Request a show |
| `status [--take N]` | Recent requests with download progress |
| `diskspace [--service radarr\|sonarr]` | Free space (needs Radarr/Sonarr creds) |
| `queue [--service radarr\|sonarr]` | Per-item queue detail (needs Radarr/Sonarr creds) |

## Root Folders — always fetch, never hardcode

Root folders differ per install, and are often **per-person libraries** mapping
1:1 to a media-server library. Never keep a table of paths in this file; they go
stale. Ask Seerr:

```
python3 scripts/seerr.py servers radarr
```

Present what it returns as a numbered list, let the user pick, then pass their
choice through with `--root-folder`.

Seerr routes to a default root folder on its own, and may apply per-user
override rules. Only ask about root folders when the destination is ambiguous or
the user wants a specific library.

## Procedure

1. **Search.** `search "the bear"` — results are numbered and carry a TMDB id.
   Anything flagged `[already in library]` is already there; say so instead of
   re-requesting.
2. **Disambiguate.** Several plausible matches → show the numbered list and ask.
   Exactly one → confirm it in a single line.
3. **Pick the library** with `servers`, when it matters.
4. **TV only — pick seasons.** `seasons <tmdb_id>`, then pass `--seasons 1,2` or
   `--seasons all`.
5. **Submit,** then confirm the outcome in one short message.

## Personality

Users usually reach this skill from a messaging platform (WhatsApp, Telegram,
Discord), so write for a chat window:

- Emojis for structure (🎬 movies, 📺 TV, ✅ done, ❌ error, 🔍 searching)
- `*bold*` for titles and key info
- Short messages — no markdown headers, no bullet dashes, no code blocks
- Multiple results → numbered list, ask the user to pick
- Confirm before submitting if there is any ambiguity
- Always state the outcome clearly

## What NEVER to send to the user

- Never show terminal commands, script output, raw JSON, or tool-call results.
- Never show debugging detail, HTTP status codes, or internal logic.
- Never send a message while a tool call is running — one final message per action.
- Every message costs the user attention. Only send meaningful replies.

## Pitfalls

- **Never hardcode a root-folder path.** `servers` is the only supported source.
- **The request field is `rootFolder`, not `rootFolderPath`.** Seerr silently
  ignores an unknown key and files the request under the default folder — it
  looks exactly like success while landing in the wrong library.
- **Seerr has per-user override rules.** With no `--root-folder`, a request lands
  in the requesting user's override folder, not necessarily the global default.
- **Do not build query strings by hand.** The script percent-encodes them; a
  title such as `Tom & Jerry` would otherwise break the query on the bare `&`.
- **Season 0 is Specials** and is excluded from `seasons` output.
- **A request is not a download.** A pending request may await approval in Seerr.
- `mediaId` in a request is the **tmdbId** from search — not a Radarr/Sonarr id.
- Requesting a title that already exists reports "already requested" (HTTP 409);
  that is not an error to retry.

## Verification

```
python3 scripts/seerr.py servers radarr
```

Root folders listed → the URL and API key are good. `SEERR_URL is not set` or a
401 means the environment is not loaded.

Tests: `scripts/run_tests.sh tests/skills/test_seerr_skill.py -q`
