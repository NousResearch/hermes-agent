# ARGUS - Agent Resource Guardian & Unified Supervisor

## The Hundred-Eyed Watchman

ARGUS is a background daemon that monitors all agent sessions (cron jobs, delegate_task sessions, manual sessions) and takes corrective actions based on the prime directive (ML data collection directives).

**Named after Argus Panoptes, the hundred-eyed giant of Greek mythology who served as Hera's faithful watchman.**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Agent Watcher Daemon (Mac Mini, launchd)                │
├─────────────────────────────────────────────────────────┤
│ 1. Session Discovery (cron + delegate_task + manual)    │
│ 2. Metrics Collector (tool calls, files, quality)       │
│ 3. Entropy Detector (repeat commands, stuck loops)      │
│ 4. Decision Engine (restart vs kill vs alert)           │
│ 5. Action Executor (inject, restart, kill)              │
│ 6. Notifier (Telegram on restarts/quality drops)        │
├─────────────────────────────────────────────────────────┤
│ State: SQLite (watcher.db) + JSON logs                  │
│ Scope: All agents, fully credentialed                   │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Session Discovery
- **Cron jobs**: Monitors all enabled cron jobs via `cronjob list`
- **Delegate tasks**: Detects subprocess sessions spawned by main agent
- **Manual sessions**: Monitors main hermes agent processes

### 2. Metrics Collection
- **Tool calls**: Tracks all tool invocations, arguments, success/failure
- **File changes**: Detects actual file modifications (hash comparison)
- **Quality metrics**: Reads from holographic_memory.db
- **Terminal commands**: Tracks shell command execution

### 3. Entropy Detection
- **Repeat tool calls**: Same tool + args called 3+ times in 10 minutes
- **Repeat commands**: Same terminal command executed 3+ times
- **Stuck loops**: Same sequence of tool calls repeating
- **No file changes**: Write operations that don't modify files

### 4. Decision Engine
- **Restart triggers**:
  - Critical entropy (except repeat tool calls)
  - Prime directive violations
  - Quality below threshold (< 0.92)
  
- **Kill triggers**:
  - High entropy: repeat tool calls detected 3+ times
  - Max restarts reached (default: 3)

### 5. Action Executor
- **Restart**: Re-aligns to prime directive, tighter constraints
- **Kill**: Terminates high-entropy sessions
- **Inject prompt**: Adds corrective instructions to running session

### 6. Notifier
- **Telegram alerts**: On restarts, quality drops, kills
- **Message format**: Session ID, type, task, action, reason, timestamp

## Configuration

Edit `agent_watcher.py` CONFIG dictionary:

```python
CONFIG = {
    'db_path': '~/hermes/data/watcher/watcher.db',
    'log_dir': '~/hermes/logs/watcher',
    'poll_interval': 30,  # seconds between checks
    'entropy_threshold': 3,  # repeat detections before action
    'quality_threshold': 0.92,  # minimum quality score
    'max_restart_count': 3,  # max restarts before kill
    'session_timeout_minutes': 60,  # session timeout
}
```

## Installation

### Quick Install
```bash
watcher-control install
```

### Manual Install
```bash
# 1. Create directories
mkdir -p ~/hermes/logs/watcher
mkdir -p ~/hermes/data/watcher

# 2. Initialize database
sqlite3 ~/hermes/data/watcher/watcher.db < ~/hermes/scripts/watcher/watcher_schema.sql

# 3. Load launchd agent
launchctl load ~/Library/LaunchAgents/com.hermes.agent-watcher.plist
```

## Usage

### Control Commands
```bash
# Start watcher
watcher-control start

# Stop watcher
watcher-control stop

# Restart watcher
watcher-control restart

# Check status
watcher-control status

# View logs
watcher-control logs

# View database
watcher-control database

# Uninstall
watcher-control uninstall
```

### Manual Run (for testing)
```bash
python3 ~/hermes/scripts/watcher/agent_watcher.py
```

## Database Schema

### Tables
- **sessions**: Active sessions being monitored
- **tool_calls**: All tool invocations
- **file_changes**: Detected file modifications
- **terminal_commands**: Shell command execution
- **quality_metrics**: Quality scores over time
- **entropy_detections**: Detected entropy patterns
- **watcher_actions**: Actions taken by watcher
- **notifications**: Telegram alerts sent
- **directive_checks**: Prime directive compliance

### Views
- **active_sessions**: Currently monitored sessions
- **high_entropy_sessions**: Sessions with critical entropy
- **low_quality_sessions**: Sessions below quality threshold
- **recent_actions**: Actions taken in last 24 hours

## Entropy Detection Patterns

### 1. Repeat Tool Calls
```sql
SELECT tool_name, tool_args, COUNT(*) as count
FROM tool_calls
WHERE session_id = ? AND timestamp > datetime('now', '-10 minutes')
GROUP BY tool_name, tool_args
HAVING count >= 3
```

### 2. Repeat Commands
```sql
SELECT command, COUNT(*) as count
FROM terminal_commands
WHERE session_id = ? AND timestamp > datetime('now', '-10 minutes')
GROUP BY command
HAVING count >= 3
```

### 3. Stuck Loops
Analyzes last 10 tool calls for repeating patterns (2-3 call sequences).

### 4. No File Changes
Detects `write_file`/`patch` calls where file hash didn't change.

## Prime Directive Checks

### Pipeline Compliance
Verifies session produces all 4 required outputs:
1. Target output (analysis file)
2. holographic_memory.db facts (quality ≥ 0.92)
3. holographic_memory.db trajectories (quality ≥ 0.93)
4. susy-works KB enrichment

### Quality Gates
Checks quality metrics against thresholds:
- Trajectory quality ≥ 0.93
- Fact quality ≥ 0.92
- Overall ML value ≥ 7

### Trajectory Generation
Ensures session generates 2+ trajectories per analysis.

### Fact Extraction
Ensures session extracts 1+ facts per analysis.

## Notification Format

```
🤖 Agent Watcher Alert

Session: cron_ec1a5e9f4c12
Type: cron
Task: SuSy Groovy ML Analysis Swarm
Action: RESTART
Reason: Prime directive violation: pipeline_compliance
Time: 2026-04-08T18:30:00-04:00
```

## Logs

### Location
- Main log: `~/hermes/logs/watcher/watcher.log`
- stdout: `~/hermes/logs/watcher/stdout.log`
- stderr: `~/hermes/logs/watcher/stderr.log`

### Log Levels
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: Failed operations
- DEBUG: Detailed debugging (disabled by default)

## Troubleshooting

### Watcher won't start
```bash
# Check if already running
launchctl list | grep agent-watcher

# Check logs
tail -f ~/hermes/logs/watcher/stderr.log

# Verify plist
plutil -lint ~/Library/LaunchAgents/com.hermes.agent-watcher.plist
```

### Database issues
```bash
# Check database integrity
sqlite3 ~/hermes/data/watcher/watcher.db "PRAGMA integrity_check;"

# Reinitialize database
sqlite3 ~/hermes/data/watcher/watcher.db < ~/hermes/scripts/watcher/watcher_schema.sql
```

### No notifications
```bash
# Check Telegram credentials
ls -la ~/hermes/credentials/telegram.env

# Test notification manually
python3 -c "from agent_watcher import AgentWatcher; w = AgentWatcher(); w._send_notification('test', 'test', 'Test message')"
```

## Development

### Adding new entropy detection
1. Add method to `AgentWatcher` class
2. Call from `detect_entropy()` method
3. Return detection dict with `entropy_type`, `severity`, `details`

### Adding new prime directive check
1. Add method to `AgentWatcher` class
2. Call from `check_prime_directive()` method
3. Return check dict with `check_type`, `passed`, `details`

### Adding new action type
1. Add method to `AgentWatcher` class
2. Call from `execute_action()` method
3. Record in `watcher_actions` table

## Security

- Fully credentialed access to all hermes systems
- Can modify cron jobs, spawn agents, access credentials
- Runs as background daemon with user permissions
- All actions logged in database for audit

## Future Enhancements

- [ ] Telegram bot integration for interactive control
- [ ] Web dashboard for monitoring
- [ ] Machine learning for entropy prediction
- [ ] Automatic quality threshold tuning
- [ ] Cross-node monitoring (Pi 5)
- [ ] Real-time metrics dashboard
- [ ] Integration with Prometheus/Grafana

## License

Internal use only - Hermes Agent Ecosystem