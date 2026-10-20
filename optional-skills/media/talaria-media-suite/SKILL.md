---
name: talaria-media-suite
description: Control a self-hosted *arr media stack from the terminal.
version: 1.0.0
author: Nhut Duong (nhutdm), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [ArrApiManagement, Dashboards, Subtitles, Indexers, LibraryAnalytics]
    blueprint:
      schedule: "0 8 * * *"
      deliver: origin
      prompt: "Talaria stack health: run status + diagnose on the media stack, report only services with issues (warnings, down, key problems) — silent when all seven are healthy."
required_environment_variables:
  - name: TALARIA_HOST
    prompt: Host IP/hostname of the media stack
    help: The single LAN host running all seven services
    required_for: every command
  - name: SONARR_KEY
    prompt: Sonarr API key
    help: Settings -> General -> API Key in Sonarr
    required_for: sonarr commands
  - name: RADARR_KEY
    prompt: Radarr API key
    help: Settings -> General -> API Key in Radarr
    required_for: radarr commands
  - name: LIDARR_KEY
    prompt: Lidarr API key
    help: Settings -> General -> API Key in Lidarr
    required_for: lidarr commands
  - name: PROWLARR_KEY
    prompt: Prowlarr API key
    help: Settings -> General -> API Key in Prowlarr
    required_for: prowlarr commands
  - name: BAZARR_KEY
    prompt: Bazarr API key
    help: Settings -> API Key in Bazarr
    required_for: bazarr commands
  - name: OVERSEERR_KEY
    prompt: Overseerr API key
    help: Settings -> General -> API Key in Overseerr
    required_for: overseerr commands
  - name: JELLYFIN_KEY
    prompt: Jellyfin API key
    help: Dashboard -> API Keys in Jellyfin
    required_for: jellyfin commands
---

# Talaria Media Suite

A Python package driving a self-hosted *arr + Jellyfin media stack from
the terminal: Sonarr, Radarr, Lidarr, Prowlarr, Bazarr, Overseerr,
Jellyfin at one LAN host. Built for Hermes Agent (2026).

## When to Use

- Stack health checks and automated troubleshooting (`status`,
  `diagnose`, `discover`)
- Library analytics and exploration (`library …`, `jellyfin counts …`)
- Jellyfin live viewing: sessions, continue-watching, next-up, activity
  (`jellyfin sessions|resume|nextup|activity`)
- Overseerr request approval (`overseerr list|approve|deny`)
- Dashboard generation as a file artifact (`dashboard out.html`)
- Indexer management (`prowlarr indexers|test|stats|search`)
- Subtitles (`bazarr wanted|search|languages`)
- Content add/remove/rename/refresh (`sonarr|radarr add|remove|rename|
  refresh`, `jellyfin refresh`)

Don't use for: services outside the seven above (Readarr, Plex,
Tautulli, SABnzbd, Recyclarr, Kometa, Unpackerr, Notifiarr,
Maintainerr and the Trakt/Simkl/Letterboxd trackers are out of scope);
download-client control (pause/resume went with SABnzbd); session-grain
play history (Jellyfin core has none — the approx analytics subs are honest
approximations from item play-state).

## Prerequisites

Binaries (audited against the shipped code, not guessed):

| Bin | Status | Why |
|-----|--------|-----|
| python3 ≥ 3.8 | required | the whole skill is `python3 -m talaria …`; stdlib only — no pip, no venv, no third-party imports |
| bash | optional | only `tests/run_tests.sh` (the skill itself never shells out) |

Environment — all vars degrade gracefully per service; "required" in
the frontmatter means Hermes surfaces setup guidance, not hard runtime
dependence. Every var gates exactly its own service (exit 1 naming the
var, zero requests sent):

| Var | Purpose |
|-----|---------|
| `TALARIA_HOST` | host IP/hostname for every call |
| `SONARR_KEY` | Sonarr API key — TV ops |
| `RADARR_KEY` | Radarr API key — movie ops |
| `LIDARR_KEY` | Lidarr API key — music ops |
| `PROWLARR_KEY` | Prowlarr API key — indexer mgmt |
| `BAZARR_KEY` | Bazarr API key — subtitles |
| `OVERSEERR_KEY` | Overseerr API key — requests |
| `JELLYFIN_KEY` | Jellyfin API key (created in Dashboard → API Keys) — sessions/libraries/analytics |

Optional per-service port overrides (defaults shown):
`SONARR_PORT` 8989 · `RADARR_PORT` 7878 · `LIDARR_PORT` 8686 ·
`PROWLARR_PORT` 9696 · `BAZARR_PORT` 6767 · `OVERSEERR_PORT` 5055 ·
`JELLYFIN_PORT` 8096.

Network egress: **LAN-only — zero external hosts.** The skill makes no
internet calls at all; every request targets `TALARIA_HOST` on one of
the seven ports above. No config files, no token files, no OAuth flows.

## How to Run

Frame every invocation through the `terminal` tool with `${HERMES_SKILL_DIR}`
as workdir — Hermes substitutes this token with the installed skill
directory's absolute path at load (the `talaria` package sits beside this
SKILL.md), so no path math is ever needed:

    terminal(command="python3 -m talaria status", workdir=${HERMES_SKILL_DIR}, timeout=60)

Global flags: `--json` (machine payload passed through VERBATIM, not
redacted — documented carve-out, see Pitfalls; approximation subs carry
`"approx": true`), `--timeout N` (per-request seconds, default 10), `--host H` (one-shot
host override). Exit codes: 0 ok · 1 any talaria error (one stderr line
prefixed `talaria:`) · 2 usage. File outputs (dashboards) are delivered
as absolute paths; do not `open` them. Per-sub docs: `python3 -m
talaria <group> <sub> --help` (purpose + args + endpoint, never
env-gated).

## Quick Reference

68 subs across 10 groups plus promoted top-level forms (`status setup
discover diagnose queue dashboard version`).

| Group | Subs |
|-------|------|
| sonarr (9) | `lookup <q>` · `add <q> *` · `remove <id> *` · `wanted` · `calendar` · `history` · `rename <id> *` · `refresh [<id>] *` · `queue` |
| radarr (9) | same shape as sonarr (movies) |
| lidarr (2) | `lookup <q>` · `queue` |
| prowlarr (9) | `indexers` · `test [<id>]` · `stats` · `search <q>` · `apps` · `add-app --type *` · `sync *` · `status` · `logs` |
| bazarr (4) | `wanted` · `history` · `search *` · `languages` |
| overseerr (5) | `list [filter]` · `approve <id> *` · `deny <id> *` · `info <id>` · `stats` |
| jellyfin (14) | `sessions` · `libraries` · `counts` · `latest` · `resume` · `nextup` · `users` · `activity` · `refresh *` · `history ~` · `most-watched ~` · `user-stats ~` · `play-totals ~` · `popular-genres ~` |
| library (10) | `stats` · `quality` · `missing` · `unmonitored` · `recent` · `genres` · `years` · `studios` · `nofiles` · `disk` |
| ops (5) | `status` · `setup <host>` · `discover <host>` · `diagnose` · `queue` |
| dashboard (1) | `dashboard <out.html>` |

`*` marks a write op (explicit-invocation rule below); `~` marks an
approximation sub (approx banner + `"approx": true` in --json)
+ `"approx": true` (honest-output rule).

## Procedure

1. **Env readiness.** `python3 -m talaria status` — done when the badge
   table renders; missing keys degrade to notes, down services to red.
2. **First run on a new host.** `python3 -m talaria setup <host>` —
   arr keys auto-recovered via the public `/initialize.json` probe and
   echoed as an export block; create the Jellyfin key in Dashboard →
   API Keys and export `JELLYFIN_KEY` (setup verifies it from the env
   when present). Done when the export block is captured outside the
   transcript.
3. **Health pass.** `status`, then `diagnose` — done when the report
   prints without unexplained API errors.
4. **Read ops.** `library stats|missing|disk`, `jellyfin
   sessions|counts|resume`, `queue` — done when the data is in the
   transcript.
5. **Artifacts.** `dashboard <out.html>` — done when the HTML exists at
   an absolute path; deliver that path and end the reply with the
   `[[as_document]]` directive so the dashboard is delivered as a
   downloadable file attachment, not an inline preview.
6. **Write ops — explicit invocation only** (the sub IS the confirmation,
   no --yes flag):
   `add remove rename refresh` (sonarr/radarr) · `search` (bazarr) ·
   `add-app sync` (prowlarr) · `approve deny` (overseerr) · `refresh`
   (jellyfin). Done when the API confirmation is echoed.

## Pitfalls

- **Jellyfin 12.x auth traps.** `X-Emby-Token`, `?api_key=`, and the
  `/emby/…` route prefix are legacy-gated or removed; talaria never
  emits them and only uses `Authorization: MediaBrowser Token="…"`.
  Copy old curl snippets at your peril.
- **Approximation (~) outputs are estimates** derived from item play-state, not session
  history — quote watch-time as approximate, always.
- **Ports are overridable per service** via `<SERVICE>_PORT`; the
  seven defaults match the stock containers.
- `setup` prints recovered keys in its export block by design (capture
  to a 0600 file); no other sub prints key material, and errors are
  redacted at the core layer — but `--json` payloads pass through
  verbatim (documented carve-out, security review A-2): on live
  services some payloads embed key material (e.g. `prowarr apps`
  fields), so redirect to a file or jq consciously, never into a
  transcript.
- Platforms: linux + macos only. python3 runs on Windows too, but the
  test harness needs bash and nothing was verified there — not widened
  (audit, don't guess).

## Verification

1. **Offline self-test** (mock-backed, dead-proxy egress trap, never a
   live API):
   `terminal(command="tests/run_tests.sh", workdir=${HERMES_SKILL_DIR}, timeout=600)`
   — TP0 structural, TP1 CLI sweep, TP2/TP2w/TP2j read/write/jellyfin
   E2E. Done when every tier reports OK.
2. **Live smoke on the user's own stack**: export `TALARIA_HOST` plus
   one service key, run `python3 -m talaria status`; done when that
   service's badge renders. The skill itself never calls live APIs to
   verify anything.
