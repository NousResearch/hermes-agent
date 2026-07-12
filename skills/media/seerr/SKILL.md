---
name: seerr
description: Request movies and TV shows for your media server.
version: 2.0.0
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
  - name: SEERR_API_KEY
    prompt: Seerr API key
    help: Seerr -> Settings -> General -> API Key
    required_for: Every Seerr call
  - name: SEERR_URL
    prompt: Seerr base URL (e.g. http://localhost:5055)
    help: The URL you open Seerr on. Also settable as skills.config.seerr.url in config.yaml.
    required_for: Every Seerr call
---

# Seerr Skill

Search a media catalogue and submit movie/TV requests to [Seerr](https://github.com/seerr-team/seerr),
the request manager that Overseerr and Jellyseerr merged into. Seerr is the only
service this skill talks to — it proxies Radarr and Sonarr, so root folders,
quality profiles and download progress all come from Seerr and no Radarr/Sonarr
credentials are needed.

This skill submits requests. It does not approve them, manage users, or move
files — do that in the Seerr web UI.

## When to Use

- Someone asks to watch, download, get, or request a movie or TV show.
- Someone asks whether a previously requested title has finished downloading.
- Someone asks what has been requested recently.

## When NOT to Use

- Approving or denying requests, or changing permissions — use the Seerr UI.
- Direct Radarr/Sonarr administration (indexers, custom formats, disk space).

## Prerequisites

- A reachable Seerr server (v1+), with Radarr and/or Sonarr already connected to it.
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
- Python 3.9+ (stdlib only).

## How to Run

Run every call with the `terminal` tool, from the skill directory:

```
python3 scripts/seerr.py <command> [options]
```

Add `--json` to any command for the raw API payload. The script exits non-zero
and prints a one-line `error:` on failure.

## Quick Reference

| Command | Purpose |
|---|---|
| `search "<title>" [--type movie\|tv]` | Numbered search results with TMDB ids |
| `servers radarr\|sonarr` | Servers, **root folders**, quality profiles |
| `seasons <tmdb_id>` | Season list for a show (Specials excluded) |
| `request --type movie --tmdb-id <id> [--root-folder <path>]` | Request a movie |
| `request --type tv --tmdb-id <id> --seasons 1,2\|all [--root-folder <path>]` | Request a show |
| `status [--take N]` | Recent requests with download progress |

## Procedure

### 1. Search

```
python3 scripts/seerr.py search "the bear"
```

Results are numbered and carry a TMDB id. Titles already in the library are
flagged `[already in library]` — say so rather than re-requesting.

If several results plausibly match, show the numbered list and ask which one.
If exactly one matches, confirm it in one line before requesting.

### 2. Choose a root folder (only when the user needs a specific library)

Seerr routes a request to its default root folder on its own. Only ask about
root folders when the user wants a specific library, or when the server exposes
more than one and the destination is ambiguous.

```
python3 scripts/seerr.py servers radarr
```

This prints the real root-folder paths straight from Radarr/Sonarr. Never
invent a path or hard-code one in this file: present what the command returns
and let the user pick, then pass it through with `--root-folder`.

### 3. For TV, get the seasons

```
python3 scripts/seerr.py seasons 136315
```

Ask which seasons they want, then pass `--seasons 1,2` or `--seasons all`.

### 4. Submit

```
python3 scripts/seerr.py request --type tv --tmdb-id 136315 --seasons all
python3 scripts/seerr.py request --type movie --tmdb-id 27205 --root-folder /movies
```

### 5. Report back

Confirm what was requested in one short line. For progress:

```
python3 scripts/seerr.py status --take 5
```

## Pitfalls

- **Never hard-code root-folder paths.** They differ per install. `servers` is
  the only supported source.
- **The request field is `rootFolder`, not `rootFolderPath`.** The script sends
  the right one; Seerr silently ignores an unknown key and files the request
  under the default folder, which looks like success.
- **Do not build query strings by hand.** The script percent-encodes them; a
  title such as `Tom & Jerry` would otherwise break the query on the bare `&`.
- **Season 0 is Specials** and is excluded from `seasons` output.
- **A request is not a download.** A `pending` request may be awaiting approval
  in Seerr; say so rather than reporting it as downloading.
- Requesting a title that already exists returns HTTP 409 from Seerr; treat it
  as "already requested", not as an error to retry.

## Verification

```
python3 scripts/seerr.py servers radarr
```

A list of servers with root folders means the URL and API key are good. A
`SEERR_URL is not set` or HTTP 401 error means the environment is not loaded.

Tests: `scripts/run_tests.sh tests/skills/test_seerr_skill.py -q`
