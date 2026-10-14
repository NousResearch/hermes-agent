# DS4 Dwarfstar Dashboard — Setup Guide

## Step 1: Clone the Repository

```bash
git clone git@github.com:shagghiesuperstar/ds4-dashboard.git ~/ds4-dashboard
cd ~/ds4-dashboard
```

## Step 2: Create Python Virtual Environment

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt** installs: fastapi, uvicorn, httpx, pydantic, psutil, starlette, websockets.

## Step 3: Ensure DS4 Server is Running

DS4 must be running on port 8001 for the dashboard to collect telemetry.

```bash
# Verify DS4 status
curl -s http://127.0.0.1:8001/telem | jq .
```

If DS4 is not running, start it:

```bash
# Via launchd (if configured)
launchctl kickstart gui/$(id -u)/com.ds4.server

# Or directly
~/ds4/ds4-server --model ~/ds4/ds4flash.gguf --port 8001 --ctx-size 131072
```

## Step 4: Start the Dashboard

### Direct (foreground)
```bash
cd ~/ds4-dashboard
source .venv/bin/activate
python dashboard.py
# → http://127.0.0.1:8765
```

### Via launchd (background, auto-restart)
```bash
cp com.ds4.dashboard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ds4.dashboard.plist
```

Or use the `scripts/start.sh` wrapper:
```bash
cd ~/ds4-dashboard
chmod +x scripts/start.sh
scripts/start.sh
```

## Step 5: Verify Everything

```bash
# Check dashboard status
curl -s http://127.0.0.1:8765/api/status | jq .

# Expected: state=stopped or running, config populated, system metrics present
```

Full verification script: `scripts/verify.sh`

```bash
cd ~/ds4-dashboard
chmod +x scripts/verify.sh
scripts/verify.sh
```

## Step 6: Open the Frontend

Open `http://127.0.0.1:8765` in a browser. The Dwarfstar dashboard shows:
- Live status bar (DS4 state, uptime, model, port)
- KV cache usage bar
- Telemetry metrics (tokens/s, context utilization)
- System metrics (GPU, CPU, temps, memory pressure)
- Config browser (all DS4 CLI options with current values)
- Benchmark runner (select suite, run, compare results)
- Update checker (GitHub release status)

## Quick Config Change Example

```bash
# Set context window to 64K
curl -X PATCH http://127.0.0.1:8765/api/config \
  -H 'Content-Type: application/json' \
  -d '{"key": "context_window", "value": 65536}'

# Verify the change
curl -s http://127.0.0.1:8765/api/config | jq '.context_window'

# Restart DS4 for the change to take effect
scripts/restart-ds4.sh
```

## Troubleshooting Setup

- **`.venv` not found**: run `python3.9 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- **Dashboard starts but status is empty**: DS4 is not running — start DS4 first
- **Port 8765 already in use**: kill existing process: `kill $(lsof -ti:8765)`
- **Benchmarks fail**: verify DS4 accepts completions: `curl -s http://127.0.0.1:8001/v1/chat/completions`
