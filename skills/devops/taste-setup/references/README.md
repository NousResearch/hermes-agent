# Taste Setup Skill

Complete setup for the Taste skill — initialize data structures, configure Spotify MCP integration, create scanning scripts, and register recurring cron jobs for consumption signal collection from email, calendar, and Spotify.

## Quick Start

```bash
# Load the skill
skill_view taste-setup

# Run setup (manual steps documented in SKILL.md)
# 1. Initialize data directory
# 2. Create config.json
# 3. Create JSONL files
# 4. Set up Python venv
# 5. Configure Spotify MCP
# 6. Create scripts (use scripts/ from this skill)
# 7. Register cron job
```

## What This Skill Does

- Initializes Taste data directory structure
- Creates full config.json with all default fields
- Sets up Python virtual environment with dependencies
- Configures Spotify MCP integration in Hermes config
- Provides scanning scripts for email, calendar, and Spotify
- Registers daily cron job for automatic scanning

## Scripts Included

- `scripts/taste_scan.py` — Main entry point
- `scripts/email_scan.py` — Gmail and Calendar scanner
- `scripts/spotify_sync_mcp.py` — Spotify sync via MCP

## Prerequisites

- Google Workspace OAuth configured at `~/.hermes/google_token.json`
- Spotify MCP server installed (`@darrenjaws/spotify-mcp`)
- Spotify Client ID and Secret (set as environment variables)

## Cron Job

Daily scan at 6am UTC:
```bash
hermes cron create --name taste:scan --skill ocas-taste "0 6 * * *" "cd /root/.hermes/commons/data/ocas-taste && source venv/bin/activate && python3 scripts/taste_scan.py"
```

## Data Sources

- Gmail: DoorDash, Instacart, Good Eggs, Tock, OpenTable, Yelp, Amazon, hotels
- Google Calendar: Restaurant reservations, hotel bookings
- Spotify: Recently played tracks, top tracks

## Verification

Check cron jobs:
```bash
hermes cron list | grep taste
```

Run manual scan:
```bash
cd /root/.hermes/commons/data/ocas-taste
source venv/bin/activate
python3 scripts/taste_scan.py
```

## Common Issues

See SKILL.md for troubleshooting:
- Missing python3-venv
- Spotify credentials not set
- MCP server not found

## Related Skills

- `ocas-taste` — The Taste skill itself
- `google-workspace-setup` — Google OAuth setup
- `native-mcp` — MCP server management