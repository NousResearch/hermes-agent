---
name: whoop-api
description: Pull Whoop fitness data via OAuth2 API.
version: 1.0.0
author: Nirbhay Shah (nirbhayshah)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [whoop, fitness, health, api, oauth, wearable]
    related_skills: [fitness, fitness-coaching]
    requires_toolsets: [terminal]
---

# Whoop API Skill

Pull recovery, strain, sleep, and workout data from the Whoop API using OAuth2.
Read-only — does not push data or modify Whoop settings.

## When to Use

- User asks for Whoop data (recovery score, strain, sleep, HRV)
- Fitness or health coaching needs real wearable metrics
- Cron job needs to sync Whoop data for dashboards or analysis
- Don't use for: modifying Whoop settings, push notifications, or real-time streaming

## Prerequisites

- Whoop Developer App registered at [developer.whoop.com](https://developer.whoop.com)
- Python 3.10+
- `requests` library: `pip install requests`
- macOS: tokens stored in Keychain (service: `whoop-api`)
- Linux/Windows: tokens stored in JSON file at `~/.config/whoop-api/tokens.json`

### App Registration

1. Go to [developer.whoop.com](https://developer.whoop.com) and create an application
2. Set redirect URI to `http://localhost:8647/callback`
3. Select scopes: `read:recovery`, `read:cycles`, `read:sleep`, `read:workout`, `read:body_measurement`, `read:profile`, `offline`
4. Copy Client ID and Client Secret — you'll need these during setup

### Privacy Policy (Required by Whoop)

Whoop requires a publicly accessible privacy policy URL for app registration. A sample is included at `references/privacy-policy.html`.

**GitHub Pages (recommended):** Create a repo, add the HTML file, enable Pages. Update the contact URL in the HTML to your own repo.

**GitHub Gist:** `gist references/privacy-policy.html` — use the raw URL as your privacy policy URL.

**Any static host:** Upload the file, update the contact link.

After hosting, update `PRIVACY_POLICY_URL` in `scripts/whoop_sync.py` to match your URL.

## How to Run

```bash
# First-time setup (run in your CLI — opens browser for OAuth)
python scripts/whoop_sync.py setup

# After setup, the agent can run these via the terminal tool:
python scripts/whoop_sync.py status
python scripts/whoop_sync.py pull
python scripts/whoop_sync.py refresh
```

All commands run from the skill directory. Use absolute paths when calling from outside.

## Quick Reference

### CLI Commands

| Command | Description |
|---|---|
| `setup` | Interactive OAuth flow — prompts for credentials, opens browser, stores tokens, creates cron jobs |
| `status` | Show credential state, token state, cron status |
| `pull` | Fetch all endpoints, save JSON to `whoop_data/{date}/` |
| `refresh` | Refresh access token if expired |

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--data-dir PATH` | `./whoop_data` | Output directory for JSON files (use absolute paths for cron) |

### API Endpoints

| Endpoint | Path | Key Data |
|---|---|---|
| Cycle | `/v2/cycle` | Strain, heart rate, kilojoules |
| Recovery | `/v2/recovery` | Recovery %, HRV, resting HR |
| Sleep | `/v2/activity/sleep` | Sleep stages, efficiency, debt |
| Workout | `/v2/activity/workout` | Strain by activity, duration |
| Body | `/v2/user/measurement/body` | Weight, height |
| Profile | `/v2/user/profile/basic` | User info |

### Data Output

```
whoop_data/
└── 2026-05-19/
    ├── cycle.json
    ├── recovery.json
    ├── sleep.json
    ├── workout.json
    ├── body.json
    └── profile.json
```

## Procedure

### 1. First-Time Setup (User Action)

The `setup` command opens a browser for OAuth authorization. This must be run by the user in their own terminal — the agent cannot interact with the browser.

Tell the user:

> "You need to run the setup command yourself in your terminal. It will open a browser for Whoop authorization. Run:
> ```
> python /path/to/skills/health/whoop-api/scripts/whoop_sync.py setup
> ```
> You'll need your Whoop Client ID and Client Secret from [developer.whoop.com](https://developer.whoop.com)."

After the user completes setup, confirm with:

```bash
python scripts/whoop_sync.py status
```

### 2. Pull Data (Agent Action)

```bash
python scripts/whoop_sync.py pull --data-dir /absolute/path/to/data
```

Fetches all endpoints for the current day. Access tokens auto-refresh if expired.

### 3. Schedule Regular Pulls

The `setup` command automatically creates two Hermes cron jobs:
- **Token refresh** every 55 minutes (keeps access tokens alive)
- **Daily data pull** at 7:00 AM local time

If you're not using Hermes cron, or want to use a different scheduler, set up manually:

**System crontab (Linux/macOS):**
```bash
0 7 * * * cd /path/to/whoop-api && python scripts/whoop_sync.py pull --data-dir /path/to/data
*/55 * * * * cd /path/to/whoop-api && python scripts/whoop_sync.py refresh
```

**Windows Task Scheduler:** Create a scheduled task pointing to the `pull` command.

See `templates/cron-entry.yaml` for Hermes cron examples.

## Pitfalls

- **Rate limit: 100 requests/minute, 10,000/day.** Each pull uses 6 requests (one per endpoint). Minimum safe interval: 5 minutes.
- **Token expiry: 1 hour.** Access tokens expire every 3600s. `pull` auto-refreshes. If you get 401s, run `refresh` manually.
- **Keychain discovery.** On macOS, the storage module locates the login keychain dynamically. If Keychain is unavailable (headless, SSH), tokens fall back to `~/.config/whoop-api/tokens.json`.
- **OAuth callback port 8647.** Must be available during `setup`. If port is in use, the script exits with a clear error.
- **No real-time data.** Whoop API is not streaming. Data appears 5-15 minutes after activity ends.
- **`setup` is interactive.** It prompts for credentials and opens a browser — the agent cannot complete this step on behalf of the user. Direct them to run it in their own terminal.

## Verification

```bash
python scripts/whoop_sync.py status && python scripts/whoop_sync.py pull && ls whoop_data/*/cycle.json
```

If `status` shows valid credentials and `pull` produces a `cycle.json` with valid JSON, the skill is working.