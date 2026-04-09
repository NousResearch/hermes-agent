# ARGUS → Hermes Integration Analysis

**Date:** 2026-04-08
**Goal:** Identify ARGUS code that reimplements existing Hermes internals and port it.

---

## Summary

| Category | ARGUS Current | Hermes Replacement | Savings |
|----------|---------------|-------------------|---------|
| Cron management | 4 methods, subprocess calls | `cronjob_tools` Python API | ~60 lines, no PATH issues |
| Session discovery | `ps aux` + own SQLite table | `SessionDB.list_sessions_rich()` | ~80 lines, single source of truth |
| Configuration | Own CONFIG dict + YAML loader | `hermes_cli.config.load_config()` | ~20 lines, single config |
| Credentials | Manual env file parsing | `env_loader.load_hermes_dotenv()` | ~15 lines, sops-aware |
| Telegram | Raw `urllib.request` Bot API | `TelegramAdapter.send_message()` | ~30 lines, retries + formatting |
| Env setup | `_get_cron_env()` manual PATH | `env_loader` + fallback | ~20 lines, consistent |

**Total removable:** ~225 lines of reimplemented functionality.

---

## 1. Cron Management → `cronjob_tools` Python API

### Current (subprocess)
```python
# _restart_cron_session, _kill_cron_session, _inject_cron_prompt
result = self._safe_subprocess(['hermes', 'cron', 'pause', str(job_id)])
result = self._safe_subprocess(['hermes', 'cron', 'resume', str(job_id)])
result = self._safe_subprocess(['hermes', 'cron', 'run', str(job_id)])
result = self._safe_subprocess(['hermes', 'cron', 'list', '--all'])
```

### Replacement (direct Python)
```python
from tools.cronjob_tools import pause_job, resume_job, trigger_job, list_jobs, get_job, update_job

# Pause
pause_job(job_id=job_id)

# Resume
resume_job(job_id=job_id)

# Force run
trigger_job(job_id=job_id)

# List all jobs
jobs = list_jobs()

# Get specific job
job = get_job(job_id=job_id)

# Update job prompt
update_job(job_id=job_id, prompt=new_prompt)
```

### Impact
- **Methods affected:** `_restart_cron_session`, `_kill_cron_session`, `_inject_cron_prompt`, `_discover_cron_sessions`, `_update_cron_prompt`
- **Saves:** ~60 lines, eliminates subprocess + PATH + timeout handling
- **Bonus:** Direct return values instead of parsing CLI stdout

### Caveats
- `cronjob_tools` functions may expect to be called from within the hermes agent context
- Need to verify they work outside of the tool-dispatch loop
- May need to set up `hermes_tools` context (DB path, etc.)

---

## 2. Session Discovery → `hermes_state.SessionDB`

### Current (`ps aux` + own table)
```python
# _discover_cron_sessions — subprocess hermes cron list
# _discover_delegate_sessions — ps aux, grep for delegate_task
# _discover_manual_sessions — ps aux, grep for hermes/python
# register_session — INSERT into own sessions table
```

### Replacement (`SessionDB`)
```python
from hermes_state import SessionDB, DEFAULT_DB_PATH

db = SessionDB(DEFAULT_DB_PATH)

# List all active sessions
sessions = db.list_sessions_rich(limit=100)

# Filter by source/type
cron_sessions = db.search_sessions(source='cron', limit=50)

# Get specific session
session = db.get_session(session_id)

# Session dict includes: session_id, title, source, started_at, 
# last_active_at, message_count, token counts, metadata
```

### Impact
- **Methods affected:** `discover_sessions`, `_discover_cron_sessions`, `_discover_delegate_sessions`, `_discover_manual_sessions`, `register_session`
- **Saves:** ~80 lines, eliminates `ps aux` parsing + own sessions table
- **Bonus:** Real-time session state from the actual session store

### Caveats
- ARGUS tracks `restart_count`, `kill_count`, `entropy_score` — these stay in argus.db
- SessionDB doesn't track process PIDs — need to keep PID-based kill logic
- Hybrid approach: use SessionDB for discovery, argus.db for ARGUS-specific metadata

---

## 3. Configuration → `hermes_cli.config`

### Current (own CONFIG + YAML)
```python
CONFIG = {
    'db_path': os.path.expanduser('~/hermes/data/watcher/argus.db'),
    'poll_interval': 30,
    'quality_threshold': 0.92,
    ...
}

def load_config(config_path):
    with open(config_path) as f:
        user_config = yaml.safe_load(f)
    return {**defaults, **user_config}
```

### Replacement (merge into config.yaml)
```python
from hermes_cli.config import load_config as hermes_load_config

# Load global config
config = hermes_load_config()

# ARGUS settings under 'argus' key
argus_config = config.get('argus', {})
poll_interval = argus_config.get('poll_interval', 30)
quality_threshold = argus_config.get('quality_threshold', 0.92)
```

Config.yaml addition:
```yaml
argus:
  poll_interval: 30
  entropy_threshold: 3
  quality_threshold: 0.92
  max_restart_count: 3
  session_timeout_minutes: 60
```

### Impact
- **Saves:** ~20 lines, single config source
- **Bonus:** `hermes config set argus.poll_interval 60` works

---

## 4. Credentials → `env_loader.load_hermes_dotenv`

### Current (manual parsing)
```python
def _load_telegram_credentials(credential_path: str) -> Dict[str, str]:
    creds = {}
    with open(credential_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, _, value = line.partition('=')
                creds[key.strip()] = value.strip().strip('"').strip("'")
    return creds
```

### Replacement
```python
from hermes_cli.env_loader import load_hermes_dotenv
import os

# Load all .env files from hermes home
load_hermes_dotenv()

# Access credentials directly
bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
chat_id = os.environ.get('TELEGRAM_CHAT_ID')
```

### Impact
- **Saves:** ~15 lines
- **Bonus:** Supports sops-encrypted files, direnv, hermes-credentials

---

## 5. Telegram Notifications → `TelegramAdapter`

### Current (raw urllib)
```python
url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
payload = json.dumps({'chat_id': chat_id, 'text': full_message, 'parse_mode': 'HTML'}).encode()
req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=10) as resp:
    result = json.loads(resp.read())
```

### Replacement
```python
from gateway.platforms.telegram import TelegramAdapter

# Create adapter instance
adapter = TelegramAdapter(config)
await adapter.send_message(chat_id=chat_id, text=full_message)
```

### Impact
- **Saves:** ~30 lines
- **Bonus:** Automatic retries, rate limiting, error handling, markdown/HTML parsing

### Caveats
- `TelegramAdapter` is async — ARGUS is sync
- Options: (a) use `asyncio.run()` wrapper, (b) keep raw urllib for simplicity, (c) create a sync wrapper
- **Recommendation:** Keep raw urllib for now. The TelegramAdapter adds complexity (async, ptb library) that's not worth it for simple notifications.

---

## 6. Environment Setup → `env_loader`

### Current (manual PATH construction)
```python
def _get_cron_env(self) -> Dict[str, str]:
    env = os.environ.copy()
    paths = ['/opt/homebrew/bin', '/usr/local/bin', ...]
    env['PATH'] = ':'.join(paths)
    env['HOME'] = os.path.expanduser('~')
    return env
```

### Replacement
```python
from hermes_cli.env_loader import load_hermes_dotenv

def _get_cron_env(self) -> Dict[str, str]:
    env = os.environ.copy()
    # Load hermes env (adds PATH, credentials, etc.)
    load_hermes_dotenv(env=env)
    # Fallback PATH if env_loader didn't set it
    if 'hermes' not in env.get('PATH', '').lower():
        paths = ['/opt/homebrew/bin', '/usr/local/bin', os.path.expanduser('~/hermes/bin'), ...]
        env['PATH'] = ':'.join(paths)
    return env
```

---

## Recommended Porting Order

### Phase 1: Low Risk, High Impact
1. **Config → `load_config()`** — merge argus settings into config.yaml
2. **Credentials → `load_hermes_dotenv()`** — replace manual env parsing

### Phase 2: Medium Risk, High Impact
3. **Cron → `cronjob_tools`** — replace subprocess calls with Python API
   - Test: `pause_job`, `resume_job`, `trigger_job`, `list_jobs`
   - Keep `_safe_subprocess` as fallback

### Phase 3: Medium Risk, Medium Impact
4. **Sessions → `SessionDB`** — use for discovery, keep argus.db for metadata
   - Hybrid: SessionDB for listing, argus.db for entropy/restart tracking

### Phase 4: Skip (Not Worth It)
5. **Telegram → `TelegramAdapter`** — async complexity > benefit for simple alerts
6. **Env setup → `env_loader`** — _get_cron_env() works fine, env_loader adds dependency

---

## Files to Modify

| File | Change |
|------|--------|
| `argus.py` | Import `cronjob_tools`, `SessionDB`, `load_config`, `load_hermes_dotenv` |
| `config.yaml` | Add `argus:` section |
| `test_argus.py` | Update mocks for new API calls |

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| Config | Low | Fallback to defaults if config.yaml missing argus key |
| Credentials | Low | Fallback to manual parsing if env_loader fails |
| Cron | Medium | Keep `_safe_subprocess` as fallback, test each function |
| Sessions | Medium | Keep argus.db schema, use SessionDB for discovery only |
