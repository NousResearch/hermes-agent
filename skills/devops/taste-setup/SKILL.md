---
name: taste-setup
description: >
  Complete setup for the Taste skill — initialize data structures, configure
  Spotify MCP integration, create scanning scripts, and register recurring
  cron jobs for email/calendar/Spotify consumption signal collection.
metadata:
  author: Indigo Karasu
  email: mx.indigo.karasu@gmail.com
  version: "1.1.0"
  hermes:
    tags: [taste, setup, mcp, cron]
    category: devops
---

# Taste Setup

Complete setup for the Taste skill — initialize data structures, configure Spotify MCP integration, create scanning scripts, and register recurring cron jobs for consumption signal collection from email, calendar, and Spotify.

## When to use

- Setting up Taste skill for the first time
- Reconfiguring Taste after system changes
- Adding new data sources to Taste
- Troubleshooting Taste cron jobs or MCP integration

## Prerequisites

### Google Workspace OAuth
Already configured at `~/.hermes/google_token.json` with scopes for Gmail and Calendar.

### Spotify MCP Server
The Spotify MCP server (`@darrenjaws/spotify-mcp`) is typically already installed at `/root/.hermes/node/lib/node_modules/@darrenjaws/spotify-mcp/`.

If not installed:
```bash
npm install -g @darrenjaws/spotify-mcp
```

### Spotify Credentials
Set environment variables:
```bash
export SPOTIFY_CLIENT_ID='your_client_id'
export SPOTIFY_CLIENT_SECRET='your_client_secret'
```

Get credentials from https://developer.spotify.com/dashboard (redirect URI: `http://localhost:8888/callback`)

## Setup Steps

### 1. Initialize Taste Data Directory

Create the data directory structure:
```bash
mkdir -p /root/.hermes/commons/data/ocas-taste/reports
mkdir -p /root/.hermes/commons/data/ocas-taste/music
```

### 2. Create Config File

Write full config.json with all default fields:
```json
{
  "skill_id": "ocas-taste",
  "skill_version": "3.0.0",
  "config_version": "2",
  "created_at": "2026-04-10T01:15:26+00:00",
  "updated_at": "2026-04-11T05:53:00+00:00",
  "domains": {
    "enabled": ["music", "restaurant", "book", "movie", "product", "travel", "event"]
  },
  "decay": {
    "halflife_days": 180
  },
  "retention": {
    "days": 0,
    "max_records": 10000
  },
  "email_scan": {
    "enabled": true,
    "last_scan_timestamp": null,
    "extraction_confidence_threshold": 0.6,
    "auto_promote_threshold": 0.8
  },
  "email_sources": {
    "doordash": { "sender_patterns": ["no-reply@doordash.com", "orders@doordash.com"], "domain": "restaurant", "source_type": "purchase" },
    "instacart": { "sender_patterns": ["no-reply@instacart.com"], "domain": "product", "source_type": "purchase" },
    "good_eggs": { "sender_patterns": ["*@goodeggs.com"], "domain": "product", "source_type": "purchase" },
    "tock": { "sender_patterns": ["*@exploretock.com"], "domain": "restaurant", "source_type": "visit" },
    "opentable": { "sender_patterns": ["*@opentable.com"], "domain": "restaurant", "source_type": "visit" },
    "yelp": { "sender_patterns": ["no-reply@yelp.com"], "domain": "restaurant", "source_type": "visit" },
    "amazon": { "sender_patterns": ["auto-confirm@amazon.com", "ship-confirm@amazon.com"], "domain": "product", "source_type": "purchase" },
    "hotels": { "sender_patterns": ["*@booking.com", "*@hotels.com", "*@marriott.com", "*@hilton.com", "*@hyatt.com", "*@ihg.com", "*@airbnb.com"], "domain": "travel", "source_type": "stay" }
  },
  "strength": {
    "base_purchase": 0.80,
    "base_visit": 0.70,
    "base_stay": 0.75,
    "base_play": 0.60,
    "base_watch": 0.60,
    "base_manual": 0.60,
    "frequency_bonus_per_visit": 0.05,
    "frequency_bonus_cap": 0.15,
    "recency_bonus_days": 30,
    "recency_bonus_value": 0.05
  },
  "user_preferences": {
    "dietary_restrictions": [],
    "dietary_preferences": [],
    "cuisine_dislikes": [],
    "notes": ""
  }
}
```

### 3. Create JSONL Data Files

Create empty JSONL files:
```bash
touch /root/.hermes/commons/data/ocas-taste/signals.jsonl
touch /root/.hermes/commons/data/ocas-taste/items.jsonl
touch /root/.hermes/commons/data/ocas-taste/links.jsonl
touch /root/.hermes/commons/data/ocas-taste/decisions.jsonl
touch /root/.hermes/commons/data/ocas-taste/extractions.jsonl
```

### 4. Set Up Python Virtual Environment

Install python3-venv if needed:
```bash
apt update && apt install -y python3.13-venv
```

Create venv and install dependencies:
```bash
cd /root/.hermes/commons/data/ocas-taste
python3 -m venv venv
source venv/bin/activate
pip install spotipy google-api-python-client
```

### 5. Configure Spotify MCP in Hermes Config

Add to `/root/.hermes/config.yaml` under `mcp_servers`:
```yaml
mcp_servers:
  spotify:
    command: node
    args:
      - /root/.hermes/node/lib/node_modules/@darrenjaws/spotify-mcp/build/bin.js
    env:
      SPOTIFY_CLIENT_ID: ${SPOTIFY_CLIENT_ID}
      SPOTIFY_CLIENT_SECRET: ${SPOTIFY_CLIENT_SECRET}
      SPOTIFY_REDIRECT_URI: http://localhost:8888/callback
```

### 6. Create Scanning Scripts

Create `scripts/` directory and add:
- `taste_scan.py` — Main entry point
- `email_scan.py` — Gmail and Calendar scanner
- `spotify_sync_mcp.py` — Spotify sync via MCP (BROKEN — uses `hermes mcp call` which doesn't exist)
- `spotify_auth_helper.py` — Interactive OAuth authorization helper (use for first-time setup)
- `README.md` — Documentation

See the full script implementations in the references directory.

### 7. Register Cron Jobs

Create daily scan job:
```bash
hermes cron create --name taste:scan --skill ocas-taste "0 6 * * *" "cd /root/.hermes/commons/data/ocas-taste && source venv/bin/activate && python3 scripts/taste_scan.py"
```

## Verification

Check cron jobs:
```bash
hermes cron list | grep taste
```

Should show `taste:scan` scheduled for daily 6am UTC.

Run manual scan:
```bash
cd /root/.hermes/commons/data/ocas-taste
source venv/bin/activate
python3 scripts/taste_scan.py
```

## Data Files

- `signals.jsonl` — Consumption signals
- `items.jsonl` — Item records (restaurants, tracks, etc.)
- `extractions.jsonl` — Raw email/calendar extractions
- `music/spotify_sync_checkpoint.json` — Spotify sync state

## Spotify MCP Tools Used

- `spotify_get_recently_played` — Get recently played tracks (last 50)
- `spotify_get_top_items` — Get top artists or tracks (short term, 20 items)

## Calling Spotify API — Use Spotipy Directly, NOT `hermes mcp call`

**IMPORTANT**: The `hermes mcp call` command does NOT exist. The Hermes MCP CLI only supports
`add`, `remove`, `list`, `serve`, `test`, and `configure` — there is no `call` subcommand.
The `spotify_sync_mcp.py` script uses `hermes mcp call spotify <tool_name>` which will always fail.

Instead, use the Spotipy Python library directly with an OAuth access token:

```python
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth=access_token)
results = sp.current_user_recently_played(limit=50)
top_tracks = sp.current_user_top_tracks(limit=20, time_range='short_term')
```

The Spotify MCP server is a stdio-based server for agent integration, not a CLI tool. It cannot
be invoked programmatically from cron scripts.

## OAuth Authorization — Required for User Data

Spotify's user-data endpoints (recently played, top tracks) require the **Authorization Code flow**
with browser-based user consent. The Client Credentials flow only supports public endpoints.

### One-Time OAuth Setup (Interactive — Requires Browser)

Run the auth helper script in a browser-accessible environment:
```bash
cd /root/.hermes/commons/data/ocas-taste
source venv/bin/activate
python3 scripts/spotify_auth_helper.py
```

This will:
1. Start a callback server on port 8888
2. Open the Spotify authorization URL in a browser
3. Capture the authorization code
4. Exchange it for access + refresh tokens
5. Save tokens to `music/spotify_token.json` and `~/.cache-spotipy-taste`
6. Optionally run the full sync immediately

### Manual Code Entry (Headless Environments)

If no browser is available on the machine, authorize on another device:
1. Visit the auth URL (printed by the script) in any browser
2. After consent, copy the `code` parameter from the redirect URL
3. Run: `python3 scripts/spotify_auth_helper.py --code <authorization_code>`

### Token Caching

Once authorized, the OAuth token is saved to:
- `/root/.hermes/commons/data/ocas-taste/music/spotify_token.json` — Taste-specific token
- `~/.cache-spotipy-taste` — Spotipy cache for refresh
- `~/.spotify-mcp/tokens.json` — Spotify MCP server's native token location

The refresh token allows subsequent syncs to work without browser interaction.
If the access token expires, Spotipy will automatically refresh it using the refresh token.

### Cron Job Sync (After OAuth Setup)

After the initial OAuth authorization, cron jobs can use the cached refresh token:

```python
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=str(CACHE_FILE)
)
token_info = sp_oauth.get_cached_token()  # Uses refresh token if access token expired
sp = spotipy.Spotify(auth=token_info['access_token'])
```

If no cached token exists (e.g., first run, token revoked), the cron job MUST fail and
report that interactive OAuth authorization is needed.

## Common Issues

### Missing python3-venv
Error: `The virtual environment was not created successfully because ensurepip is not available`

Fix: `apt install python3.13-venv`

### Spotify credentials not set
Warning: `SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables not set`

Fix: Set environment variables before running scan. Credentials are stored in
`~/.hermes/.env` and must be exported or passed via the config.yaml `mcp_servers.spotify.env` section.

### Spotify OAuth not completed
Error: `NEEDS AUTHORIZATION: Browser-based OAuth required` or `No cached token found`

This happens on first run or after token expiration/revocation. Run the auth helper script
interively (see "One-Time OAuth Setup" above). The cron job cannot complete OAuth without
a browser — it will write a journal entry and update the checkpoint with `last_sync_status: failed`.

### `hermes mcp call` fails
Error: `hermes mcp: error: argument mcp_action: invalid choice: 'call'`

This is expected — `hermes mcp call` does not exist. Use Spotipy directly with an OAuth
token instead of trying to call MCP tools from Python scripts.

### MCP server not found
Error: `MCP tool call failed`

Fix: Verify Spotify MCP is installed and configured in config.yaml. But note that
MCP tools cannot be invoked from cron scripts — use Spotipy directly.

## Cron Job Management

List cron jobs:
```bash
hermes cron list
```

Remove cron job:
```bash
hermes cron remove <job_id>
```

Update cron job:
```bash
hermes cron remove <job_id>
hermes cron create --name <name> --skill <skill> <schedule> <prompt>
```