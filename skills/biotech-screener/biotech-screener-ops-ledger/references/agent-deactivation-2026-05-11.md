# Agent Deactivation Procedure — May 11, 2026

## Context
User requested to deactivate agents to conserve Grok token usage after checking token balance.

## Steps Performed

### 1. Stop Running Agent Processes
```bash
# Kill OpenClaw gateway
pkill -f "openclaw/dist/index.js gateway"

# Kill Hermes agent gateway
pkill -f "hermes_cli.main gateway"
```

### 2. Disable Systemd Services
```bash
# Stop services
systemctl --user stop hermes-gateway.service openclaw-gateway.service

# Disable auto-restart on boot
systemctl --user disable hermes-gateway.service openclaw-gateway.service
```

### 3. Comment Out Cron Jobs
```bash
# Get current crontab
crontab -l > /tmp/crontab_backup.txt

# Comment out agent-related cron jobs
crontab -l | sed '/biotech_screener.*run_agent_direct.py/{s/^/#/}' | crontab -
crontab -l | sed '/grok_biotech_watch/{s/^/#/}' | crontab -
crontab -l | sed '/fetch_conference_abstracts_grok/{s/^/#/}' | crontab -
```

### 4. Verification
```bash
# Check no agent processes running
ps aux | grep -E "openclaw|hermes" | grep -v grep

# Check systemd status
systemctl --user status hermes-gateway.service openclaw-gateway.service

# View crontab
crontab -l
```

## Results
- ✅ All agent processes stopped
- ✅ Systemd services disabled
- ✅ Cron jobs commented out
- ✅ Grok API calls disabled
- ✅ Token usage conserved

## Notes
- The `agent_heartbeat_checks.py` cron job remains active (30 17 * * 1-5) as it's lightweight and only invokes an LLM when anomalies are detected.
- This procedure can be reversed when agents need to be reactivated (re-enable services, uncomment cron jobs).