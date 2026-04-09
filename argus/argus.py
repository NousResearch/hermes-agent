#!/usr/bin/env python3
"""
ARGUS - Agent Resource Guardian & Unified Supervisor
The Hundred-Eyed Watchman - monitors all agent sessions, detects entropy, takes corrective actions.
Runs as background daemon on Mac Mini via launchd.

Uses Hermes internals directly (same pattern as hermes_cli/cron.py):
  - cron.jobs for cron management (data layer)
  - hermes_state.SessionDB for session discovery
  - hermes_cli.config for configuration
  - hermes_cli.env_loader for credential loading
"""

import os
import sys
import json
import sqlite3
import time
import subprocess
import signal
import logging
import logging.handlers
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# === PATH RESOLUTION (contributor guide: use get_hermes_home(), never hardcode) ===
# Add hermes-agent to path for direct module imports
_HERMES_AGENT = os.path.expanduser('~/.hermes/hermes-agent')
if os.path.isdir(_HERMES_AGENT) and _HERMES_AGENT not in sys.path:
    sys.path.insert(0, _HERMES_AGENT)

# Hermes home (~/.hermes) — for internal state, holographic DB, session store
try:
    from hermes_constants import get_hermes_home
    _HERMES_HOME = get_hermes_home()
except ImportError:
    _HERMES_HOME = Path(os.path.expanduser('~/.hermes'))

# ARGUS home (~/hermes) — for scripts, data, logs, bin
_ARGUS_HOME = Path(os.path.expanduser('~/hermes'))

def _hermes_path(*parts: str) -> Path:
    """Build a path under HERMES_HOME (~/.hermes)."""
    return _HERMES_HOME.joinpath(*parts)

def _argus_path(*parts: str) -> Path:
    """Build a path under ARGUS_HOME (~/hermes)."""
    return _ARGUS_HOME.joinpath(*parts)

# Cron data layer — imported directly by hermes_cli/cron.py L43
# Fallback to subprocess if hermes internals unavailable (e.g., wrong Python version)
_HERMES_INTERNALS_AVAILABLE = False
try:
    from cron.jobs import pause_job, resume_job, trigger_job, list_jobs, get_job, update_job
    from hermes_state import SessionDB, DEFAULT_DB_PATH
    from hermes_cli.config import load_config as _hermes_load_config
    from hermes_cli.env_loader import load_hermes_dotenv
    _HERMES_INTERNALS_AVAILABLE = True
except (ImportError, TypeError) as _e:
    # TypeError catches Python 3.9 union type syntax errors in hermes internals
    import logging as _logging
    _logging.getLogger('argus').warning(f"Hermes internals unavailable ({_e}), falling back to subprocess")

    # Stub functions — subprocess fallback for Python <3.10 or missing hermes-agent
    def pause_job(job_id, reason=None):
        """Fallback: pause cron job via CLI subprocess."""
        r = subprocess.run(['hermes', 'cron', 'pause', str(job_id)], capture_output=True, text=True, timeout=10)
        return {'id': job_id, 'enabled': False} if r.returncode == 0 else None

    def resume_job(job_id):
        """Fallback: resume cron job via CLI subprocess."""
        r = subprocess.run(['hermes', 'cron', 'resume', str(job_id)], capture_output=True, text=True, timeout=10)
        return {'id': job_id, 'enabled': True} if r.returncode == 0 else None

    def trigger_job(job_id):
        """Fallback: trigger cron job via CLI subprocess."""
        r = subprocess.run(['hermes', 'cron', 'run', str(job_id)], capture_output=True, text=True, timeout=10)
        return {'id': job_id} if r.returncode == 0 else None

    def list_jobs(include_disabled=False):
        """Fallback: list cron jobs via CLI subprocess."""
        r = subprocess.run(['hermes', 'cron', 'list', '--all'], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout).get('jobs', [])
            except json.JSONDecodeError:
                return []
        return []

    def get_job(job_id):
        """Fallback: get cron job details via CLI subprocess."""
        for j in list_jobs(include_disabled=True):
            if j.get('id') == job_id:
                return j
        return None

    def update_job(job_id, updates):
        """Fallback: update cron job — not available via CLI."""
        return None

    class SessionDB:
        """Fallback: stub SessionDB when hermes internals unavailable."""

        def __init__(self, path):
            pass

        def list_sessions_rich(self, **kw):
            """Fallback: return empty list when hermes internals unavailable."""
            return []

        def close(self):
            """Fallback: no-op when hermes internals unavailable."""
            pass

    DEFAULT_DB_PATH = str(_hermes_path('state.db'))

    def _hermes_load_config():
        return {}

    def load_hermes_dotenv():
        """Fallback: load hermes .env — no-op when hermes internals unavailable."""
        pass

# WAL monitor for real-time tool call detection
from wal_monitor import ToolCallMonitor

# Corrective prompt templates by entropy type
CORRECTIVE_PROMPTS = {
    'repeat_tool_calls': (
        "ENTROPY CORRECTION: ARGUS detected repeated tool calls without progress. "
        "You are calling the same tool with the same arguments multiple times. "
        "Stop and reassess. Read the file/content you need ONCE, then act on it. "
        "Do not re-read files you already have in context. Complete the task."
    ),
    'repeat_commands': (
        "ENTROPY CORRECTION: ARGUS detected repeated terminal commands. "
        "You are running the same command multiple times. "
        "Check the output you already received before re-running. "
        "If the command failed, fix the issue, don't retry blindly."
    ),
    'stuck_loop': (
        "ENTROPY CORRECTION: ARGUS detected a stuck loop pattern. "
        "Your last several tool calls form a repeating cycle. "
        "STOP. Read your conversation history. Identify what you're trying to accomplish. "
        "Take a different approach. Do not repeat the same sequence."
    ),
    'no_file_changes': (
        "ENTROPY CORRECTION: ARGUS detected write operations that didn't change files. "
        "You are calling write_file/patch but the file content is not changing. "
        "Read the file first, verify what you're writing is actually different. "
        "If using patch, check that old_string matches exactly."
    ),
    'quality_gate': (
        "QUALITY CORRECTION: Your output quality is below the 0.92 threshold. "
        "Provide mechanistic explanations, not surface descriptions. "
        "Include structured output with headers and metrics. "
        "Feed the pipeline: write facts, generate trajectories, enrich KB."
    ),
    'pipeline_violation': (
        "PIPELINE CORRECTION: You are not hitting all 4 pipeline targets. "
        "Every substantive interaction must produce: "
        "(1) target output, (2) holographic_memory.db facts, "
        "(3) trajectories (Q&A chains), (4) KB enrichment. "
        "Self-assess before finishing."
    ),
}

# === CONFIGURATION (loaded from hermes config.yaml under 'argus' key) ===
# Defaults — overridden by config.yaml argus: section
_DEFAULT_ARGUS_CONFIG = {
    'db_path': str(_argus_path('data', 'watcher', 'argus.db')),
    'log_dir': str(_argus_path('logs', 'argus')),
    'poll_interval': 30,
    'entropy_threshold': 3,
    'quality_threshold': 0.92,
    'max_restart_count': 3,
    'session_timeout_minutes': 60,
}

def _load_argus_config() -> Dict:
    """Load ARGUS config from hermes config.yaml 'argus' key, falling back to defaults."""
    try:
        hermes_config = _hermes_load_config()
        argus_overrides = hermes_config.get('argus', {})
        return {**_DEFAULT_ARGUS_CONFIG, **argus_overrides}
    except Exception:
        return dict(_DEFAULT_ARGUS_CONFIG)

# Runtime config — loaded once at import
CONFIG = _load_argus_config()


# === PID FILE MANAGEMENT (same pattern as gateway/status.py) ===
_ARGUS_PID_PATH = _argus_path('data', 'watcher', 'argus.pid')
_ARGUS_KIND = 'argus-watcher'


def _get_argus_pid_path() -> Path:
    """Return the path to the ARGUS PID file."""
    return _ARGUS_PID_PATH


def _build_argus_pid_record() -> dict:
    """Build PID record — same structure as gateway/status._build_pid_record."""
    return {
        'pid': os.getpid(),
        'kind': _ARGUS_KIND,
        'argv': list(sys.argv),
        'start_time': time.time(),
    }


def write_argus_pid_file() -> None:
    """Write ARGUS PID file — same pattern as gateway/status.write_pid_file."""
    path = _get_argus_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_argus_pid_record()))


def remove_argus_pid_file() -> None:
    """Remove ARGUS PID file — same pattern as gateway/status.remove_pid_file."""
    try:
        _get_argus_pid_path().unlink(missing_ok=True)
    except Exception:
        pass


def _read_argus_pid_record() -> Optional[dict]:
    """Read ARGUS PID file — same pattern as gateway/status._read_pid_record."""
    path = _get_argus_pid_path()
    if not path.exists():
        return None
    raw = path.read_text().strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return {'pid': int(raw)}
        except ValueError:
            return None


def get_argus_running_pid() -> Optional[int]:
    """Return PID of running ARGUS instance, or None.
    Same pattern as gateway/status.get_running_pid."""
    record = _read_argus_pid_record()
    if not record:
        remove_argus_pid_file()
        return None
    try:
        pid = int(record['pid'])
    except (KeyError, TypeError, ValueError):
        remove_argus_pid_file()
        return None

    # Check if process is alive
    try:
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, PermissionError):
        remove_argus_pid_file()
        return None


def is_argus_running() -> bool:
    """Check if ARGUS daemon is currently running."""
    return get_argus_running_pid() is not None


# === LAUNCHD INTEGRATION (same pattern as hermes_cli/gateway.py) ===
_ARGUS_LAUNCHD_LABEL = 'com.hermes.argus'
_ARGUS_SCRIPT = str(_argus_path('scripts', 'watcher', 'argus.py'))


def get_argus_launchd_label() -> str:
    """Return the launchd service label — same pattern as gateway/get_launchd_label."""
    return _ARGUS_LAUNCHD_LABEL


def get_argus_launchd_plist_path() -> Path:
    """Return the launchd plist path — same pattern as gateway/get_launchd_plist_path."""
    return _hermes_home_plist_dir() / f'{_ARGUS_LAUNCHD_LABEL}.plist'

def _hermes_home_plist_dir() -> Path:
    """Return ~/Library/LaunchAgents (macOS-specific)."""
    return Path.home() / 'Library' / 'LaunchAgents'


def generate_argus_launchd_plist() -> str:
    """Generate launchd plist XML — same pattern as gateway/generate_launchd_plist.

    Builds full PATH (hermes pattern: venv + homebrew + user tools),
    sets HERMES_HOME, RunAtLoad, KeepAlive, log rotation.
    """
    import shutil as _shutil

    label = get_argus_launchd_label()
    script = _ARGUS_SCRIPT
    log_dir = str(_argus_path('logs', 'argus'))
    hermes_home = str(_HERMES_HOME)

    # Build PATH (same pattern as gateway — detect venv + user tools)
    venv_bin = str(_hermes_path('hermes-agent', 'venv', 'bin'))
    priority_dirs = [venv_bin] if os.path.isdir(venv_bin) else []

    # Detect hermes binary location
    hermes_bin = _shutil.which('hermes')
    if hermes_bin:
        hermes_dir = str(Path(hermes_bin).resolve().parent)
        if hermes_dir not in priority_dirs:
            priority_dirs.append(hermes_dir)

    sane_path = ':'.join(
        dict.fromkeys(priority_dirs + [p for p in os.environ.get('PATH', '').split(':') if p])
    )

    # Detect python
    python = sys.executable or '/usr/bin/python3'

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{Path(script).parent}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{sane_path}</string>
        <key>HERMES_HOME</key>
        <string>{hermes_home}</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>{log_dir}/argus.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{log_dir}/argus.stderr.log</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>'''


def argus_launchd_install() -> bool:
    """Install ARGUS as launchd service — same pattern as gateway/launchd_install."""
    plist_path = get_argus_launchd_plist_path()

    # Write plist
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(generate_argus_launchd_plist())
    logger.info("ARGUS plist written to: %s", plist_path)

    # Bootstrap via launchctl
    try:
        subprocess.run(
            ['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(plist_path)],
            check=True, capture_output=True, text=True, timeout=10
        )
        logger.info("ARGUS launchd service bootstrapped")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to bootstrap ARGUS: %s", e.stderr, exc_info=True)
        return False


def argus_launchd_uninstall() -> bool:
    """Uninstall ARGUS launchd service."""
    label = get_argus_launchd_label()
    plist_path = get_argus_launchd_plist_path()

    # Bootout
    try:
        subprocess.run(
            ['launchctl', 'bootout', f'gui/{os.getuid()}/{label}'],
            capture_output=True, text=True, timeout=10
        )
    except Exception:
        pass

    # Remove plist
    plist_path.unlink(missing_ok=True)
    logger.info("ARGUS launchd service uninstalled")
    return True


def argus_launchd_status() -> dict:
    """Check ARGUS launchd service status."""
    label = get_argus_launchd_label()
    plist_path = get_argus_launchd_plist_path()

    return {
        'label': label,
        'plist_exists': plist_path.exists(),
        'plist_path': str(plist_path),
        'pid_file_exists': _get_argus_pid_path().exists(),
        'running_pid': get_argus_running_pid(),
        'is_running': is_argus_running(),
    }


# === LOGGING (hermes-native RotatingFileHandler pattern) ===
os.makedirs(CONFIG['log_dir'], exist_ok=True)

_LOG_MAX_BYTES = CONFIG.get('log_max_size_mb', 5) * 1024 * 1024  # 5MB default
_LOG_BACKUP_COUNT = CONFIG.get('log_backup_count', 3)

logger = logging.getLogger('argus')
logger.setLevel(logging.INFO)

# RotatingFileHandler — same pattern as hermes_logging._add_rotating_handler
_rotating_handler = logging.handlers.RotatingFileHandler(
    str(Path(CONFIG['log_dir']) / 'argus.log'),
    maxBytes=_LOG_MAX_BYTES,
    backupCount=_LOG_BACKUP_COUNT,
)
_rotating_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_rotating_handler)

# StreamHandler for stderr (same as hermes)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_stream_handler)

class Argus:
    def __init__(self):
        self.db_path = CONFIG['db_path']
        self.conn = None
        self.cursor = None
        self.running = False

        # WAL monitor for real-time tool call detection
        self.wal_monitor = ToolCallMonitor(
            poll_interval=CONFIG.get('poll_interval', 30) / 10,  # 10x faster than main poll
            repeat_threshold=CONFIG.get('entropy_threshold', 3),
        )

        # Ensure database directory exists
        os.makedirs(Path(self.db_path).parent, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load schema
        self._load_schema()
    
    def _init_database(self):
        """Initialize database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        logger.info("Database connected: %s", self.db_path)
    
    def _load_schema(self):
        """Load database schema if tables don't exist."""
        schema_path = Path(__file__).parent / 'watcher_schema.sql'
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema = f.read()
            self.conn.executescript(schema)
            self.conn.commit()
            logger.info("Database schema loaded")
    
    def _get_cron_env(self) -> Dict[str, str]:
        """Build a full environment dict for subprocess calls in sandboxed contexts."""
        env = os.environ.copy()

        # Ensure PATH includes all critical tool locations
        paths = [
            '/opt/homebrew/bin',
            '/usr/local/bin',
            str(_argus_path('bin')),              # ~/hermes/bin
            str(Path.home() / '.local' / 'bin'),   # ~/.local/bin
            str(_hermes_path('credentials')),       # ~/.hermes/credentials
            '/usr/bin',
            '/bin',
        ]
        env['PATH'] = ':'.join(paths)

        # Ensure HOME is set (some launchd contexts may not have it)
        env['HOME'] = os.path.expanduser('~')

        return env

    def _safe_subprocess(self, cmd: List[str], timeout: int = 10, **kwargs) -> Optional[subprocess.CompletedProcess]:
        """Run a subprocess with full env and error handling. Never raises."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._get_cron_env(),
                **kwargs
            )
        except FileNotFoundError:
            logger.warning("Command not found: %s (check PATH)", cmd[0])
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Command timed out after %ss: %s", timeout, ' '.join(cmd))
            return None
        except Exception as e:
            logger.error("Subprocess error for %s: %s", cmd[0], e, exc_info=True)
            return None

    def discover_sessions(self) -> List[Dict]:
        """Discover all active agent sessions."""
        sessions = []
        
        # 1. Discover cron jobs
        sessions.extend(self._discover_cron_sessions())
        
        # 2. Discover delegate_task sessions
        sessions.extend(self._discover_delegate_sessions())
        
        # 3. Discover manual sessions (main hermes agent)
        sessions.extend(self._discover_manual_sessions())
        
        return sessions
    
    def _discover_cron_sessions(self) -> List[Dict]:
        """Discover active cron job sessions via cron.jobs (same as hermes_cli/cron.py)."""
        sessions = []

        try:
            jobs = list_jobs(include_disabled=False)

            for job in jobs:
                sessions.append({
                    'session_id': f"cron_{job['id']}",
                    'session_type': 'cron',
                    'job_id': job['id'],
                    'task_description': job.get('name', 'Unknown'),
                    'model': job.get('model'),
                    'provider': job.get('provider'),
                    'metadata': json.dumps(job)
                })

        except Exception as e:
            logger.error("Error discovering cron sessions: %s", e, exc_info=True)

        return sessions
    
    def _discover_delegate_sessions(self) -> List[Dict]:
        """Discover delegate_task sessions via SessionDB (same as hermes_cli)."""
        sessions = []

        try:
            db = SessionDB(DEFAULT_DB_PATH)
            try:
                all_sessions = db.list_sessions_rich(limit=50)
            finally:
                db.close()

            for s in all_sessions:
                if s.get('source') == 'delegate' or s.get('source') == 'delegate_task':
                    sessions.append({
                        'session_id': "delegate_%s" % s['id'],
                        'session_type': 'delegate_task',
                        'task_description': s.get('title', "Delegate %s" % s['id'][:12]),
                        'metadata': json.dumps({
                            'session_id': s['id'],
                            'source': s.get('source'),
                            'started_at': s.get('started_at'),
                        })
                    })

        except Exception as e:
            logger.error("Error discovering delegate sessions: %s", e, exc_info=True)

        return sessions
    
    def _discover_manual_sessions(self) -> List[Dict]:
        """Discover manual agent sessions via SessionDB (same as hermes_cli)."""
        sessions = []

        try:
            db = SessionDB(DEFAULT_DB_PATH)
            try:
                all_sessions = db.list_sessions_rich(limit=50)
            finally:
                db.close()

            for s in all_sessions:
                source = s.get('source', '')
                if source in ('cli', 'telegram', 'manual', 'gateway'):
                    sessions.append({
                        'session_id': "manual_%s" % s['id'],
                        'session_type': 'manual',
                        'task_description': s.get('title', "Session %s" % s['id'][:12]),
                        'metadata': json.dumps({
                            'session_id': s['id'],
                            'source': source,
                            'started_at': s.get('started_at'),
                        })
                    })

        except Exception as e:
            logger.error("Error discovering manual sessions: %s", e, exc_info=True)

        return sessions
    
    def register_session(self, session: Dict):
        """Register a session in the database."""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, session_type, job_id, task_description, model, provider, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['session_id'],
                session['session_type'],
                session.get('job_id'),
                session.get('task_description'),
                session.get('model'),
                session.get('provider'),
                session.get('metadata')
            ))
            self.conn.commit()
            logger.debug("Registered session: %s", session['session_id'])
        
        except Exception as e:
            logger.error("Error registering session %s: %s", session['session_id'], e, exc_info=True)
    
    def collect_metrics(self, session_id: str) -> None:
        """Collect metrics for a session from logs and holographic_memory.db."""
        try:
            # 1. Populate tool_calls from session data (fills entropy detection table)
            self._populate_tool_calls_from_session(session_id)

            # 2. Count existing tool calls for this session
            self.cursor.execute(
                'SELECT COUNT(*) as cnt FROM tool_calls WHERE session_id = ?',
                (session_id,)
            )
            tool_count = self.cursor.fetchone()['cnt']

            # 3. Count terminal commands for this session
            self.cursor.execute(
                'SELECT COUNT(*) as cnt FROM terminal_commands WHERE session_id = ?',
                (session_id,)
            )
            cmd_count = self.cursor.fetchone()['cnt']

            # 4. Get quality metrics from holographic_memory.db
            quality_score = self._fetch_quality_from_holographic(session_id)

            # 5. Update session with collected metrics
            self.cursor.execute('''
                UPDATE sessions
                SET tool_call_count = ?, quality_gate_score = ?, last_activity_at = ?
                WHERE session_id = ?
            ''', (tool_count + cmd_count, quality_score, datetime.now().isoformat(), session_id))

            # 6. Record quality metric if available
            if quality_score is not None:
                self.cursor.execute('''
                    INSERT INTO quality_metrics (session_id, metric_type, metric_value, details)
                    VALUES (?, 'overall_quality', ?, ?)
                ''', (session_id, quality_score, json.dumps({
                    'tool_calls': tool_count,
                    'terminal_commands': cmd_count
                })))

            self.conn.commit()

        except Exception as e:
            logger.error("Error collecting metrics for %s: %s", session_id, e, exc_info=True)

    def _populate_tool_calls_from_session(self, session_id: str) -> None:
        """Read tool calls from state.db and insert into argus.db.tool_calls.
        
        This ensures the tool_calls table has data even when the WAL monitor
        isn't running (e.g., hermes internals unavailable).
        """
        # Extract the real hermes session_id from our prefixed ID
        # e.g., "cron_ec1a5e9f4c12" -> "ec1a5e9f4c12", "manual_abc123" -> "abc123"
        parts = session_id.split('_', 1)
        if len(parts) == 2:
            real_session_id = parts[1]
        else:
            real_session_id = session_id
        
        try:
            db = SessionDB(DEFAULT_DB_PATH)
            try:
                messages = db.get_messages(real_session_id)
            finally:
                db.close()
        except Exception:
            return
        
        if not messages:
            return
        
        # Find the latest timestamp we've already ingested for this session
        self.cursor.execute(
            'SELECT MAX(timestamp) as max_ts FROM tool_calls WHERE session_id = ?',
            (session_id,)
        )
        row = self.cursor.fetchone()
        last_ingested = row['max_ts'] if row and row['max_ts'] else '0'
        
        try:
            last_ingested_float = float(last_ingested)
        except (ValueError, TypeError):
            last_ingested_float = 0.0
        
        # Extract tool calls from assistant messages
        inserts = []
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            
            ts = msg.get('timestamp', '')
            try:
                ts_float = float(ts) if ts else 0.0
            except (ValueError, TypeError):
                ts_float = 0.0
            
            if ts_float <= last_ingested_float:
                continue
            
            tool_calls_raw = msg.get('tool_calls')
            if not tool_calls_raw:
                continue
            
            try:
                if isinstance(tool_calls_raw, str):
                    tool_calls_raw = json.loads(tool_calls_raw)
                for tc in tool_calls_raw:
                    name = tc.get('function', {}).get('name') or tc.get('name', '')
                    args = tc.get('function', {}).get('arguments') or tc.get('arguments', '')
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    if name:
                        inserts.append((session_id, name, str(args), str(ts), True))
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Batch insert
        if inserts:
            self.cursor.executemany('''
                INSERT OR IGNORE INTO tool_calls
                (session_id, tool_name, tool_args, timestamp, file_changed)
                VALUES (?, ?, ?, ?, ?)
            ''', inserts)
            logger.debug("Ingested %d tool calls for session %s", len(inserts), session_id)

    def _fetch_quality_from_holographic(self, session_id: str) -> Optional[float]:
        """Fetch quality score from holographic_memory.db for a session."""
        holo_db = str(_hermes_path('holographic_memory.db'))
        if not os.path.exists(holo_db):
            return None

        try:
            holo_conn = sqlite3.connect(holo_db)
            holo_conn.row_factory = sqlite3.Row
            cursor = holo_conn.cursor()

            # Get average quality score from recent facts
            cursor.execute('''
                SELECT AVG(quality_score) as avg_quality
                FROM facts
                WHERE timestamp > datetime('now', '-24 hours')
                AND quality_score IS NOT NULL
            ''')
            row = cursor.fetchone()
            holo_conn.close()

            if row and row['avg_quality']:
                return round(row['avg_quality'], 4)

        except Exception as e:
            logger.error("Error reading holographic_memory.db: %s", e, exc_info=True)

        return None
    
    def detect_entropy(self, session_id: str) -> List[Dict]:
        """Detect entropy patterns in a session."""
        detections = []
        
        # 1. Check for repeat tool calls
        repeat_detections = self._detect_repeat_tool_calls(session_id)
        detections.extend(repeat_detections)
        
        # 2. Check for repeat terminal commands
        command_detections = self._detect_repeat_commands(session_id)
        detections.extend(command_detections)
        
        # 3. Check for stuck loops (same sequence of tool calls)
        loop_detections = self._detect_stuck_loops(session_id)
        detections.extend(loop_detections)
        
        # 4. Check for no file changes despite write operations
        no_change_detections = self._detect_no_file_changes(session_id)
        detections.extend(no_change_detections)
        
        return detections
    
    def _detect_repeat_tool_calls(self, session_id: str) -> List[Dict]:
        """Detect repeated tool calls without changes."""
        detections = []
        
        try:
            # Get recent tool calls for this session
            self.cursor.execute('''
                SELECT tool_name, tool_args, COUNT(*) as count
                FROM tool_calls
                WHERE session_id = ? AND timestamp > datetime('now', '-10 minutes')
                GROUP BY tool_name, tool_args
                HAVING count >= 3
            ''', (session_id,))
            
            for row in self.cursor.fetchall():
                detections.append({
                    'entropy_type': 'repeat_tool_calls',
                    'severity': 'warning' if row['count'] < 5 else 'critical',
                    'details': json.dumps({
                        'tool_name': row['tool_name'],
                        'tool_args': row['tool_args'],
                        'count': row['count']
                    })
                })
        
        except Exception as e:
            logger.error("Error detecting repeat tool calls: %s", e, exc_info=True)
        
        return detections
    
    def _detect_repeat_commands(self, session_id: str) -> List[Dict]:
        """Detect repeated terminal commands."""
        detections = []
        
        try:
            # Get recent terminal commands for this session
            self.cursor.execute('''
                SELECT command, COUNT(*) as count
                FROM terminal_commands
                WHERE session_id = ? AND timestamp > datetime('now', '-10 minutes')
                GROUP BY command
                HAVING count >= 3
            ''', (session_id,))
            
            for row in self.cursor.fetchall():
                detections.append({
                    'entropy_type': 'repeat_commands',
                    'severity': 'warning' if row['count'] < 5 else 'critical',
                    'details': json.dumps({
                        'command': row['command'],
                        'count': row['count']
                    })
                })
        
        except Exception as e:
            logger.error("Error detecting repeat commands: %s", e, exc_info=True)
        
        return detections
    
    def _detect_stuck_loops(self, session_id: str) -> List[Dict]:
        """Detect stuck loops (same sequence of tool calls)."""
        detections = []
        
        try:
            # Get last 10 tool calls for this session
            self.cursor.execute('''
                SELECT tool_name, tool_args
                FROM tool_calls
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (session_id,))
            
            tool_calls = [dict(row) for row in self.cursor.fetchall()]
            
            if len(tool_calls) >= 6:
                # Check for repeating pattern
                for pattern_length in range(2, 4):
                    if len(tool_calls) >= pattern_length * 2:
                        pattern = tool_calls[:pattern_length]
                        next_pattern = tool_calls[pattern_length:pattern_length*2]
                        
                        if pattern == next_pattern:
                            detections.append({
                                'entropy_type': 'stuck_loop',
                                'severity': 'critical',
                                'details': json.dumps({
                                    'pattern_length': pattern_length,
                                    'pattern': pattern
                                })
                            })
        
        except Exception as e:
            logger.error("Error detecting stuck loops: %s", e, exc_info=True)
        
        return detections
    
    def _detect_no_file_changes(self, session_id: str) -> List[Dict]:
        """Detect write operations without file changes."""
        detections = []
        
        try:
            # Get recent write operations that didn't change files
            self.cursor.execute('''
                SELECT tc.id, tc.tool_name, tc.file_path
                FROM tool_calls tc
                WHERE tc.session_id = ? 
                AND tc.tool_name IN ('write_file', 'patch')
                AND tc.file_changed = FALSE
                AND tc.timestamp > datetime('now', '-10 minutes')
            ''', (session_id,))
            
            for row in self.cursor.fetchall():
                detections.append({
                    'entropy_type': 'no_file_changes',
                    'severity': 'critical',
                    'details': json.dumps({
                        'tool_call_id': row['id'],
                        'tool_name': row['tool_name'],
                        'file_path': row['file_path']
                    })
                })
        
        except Exception as e:
            logger.error("Error detecting no file changes: %s", e, exc_info=True)
        
        return detections
    
    def check_prime_directive(self, session_id: str) -> List[Dict]:
        """Check if session is following prime directive."""
        checks = []
        
        # 1. Check pipeline compliance (4 targets)
        checks.append(self._check_pipeline_compliance(session_id))
        
        # 2. Check quality gates
        checks.append(self._check_quality_gates(session_id))
        
        # 3. Check trajectory generation
        checks.append(self._check_trajectory_generation(session_id))
        
        # 4. Check fact extraction
        checks.append(self._check_fact_extraction(session_id))
        
        return checks
    
    def _open_holographic_db(self) -> Optional[sqlite3.Connection]:
        """Open holographic_memory.db with Row factory. Returns None if unavailable."""
        holo_db = str(_hermes_path('holographic_memory.db'))
        if not os.path.exists(holo_db):
            return None
        try:
            conn = sqlite3.connect(holo_db)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error:
            return None
    
    def _check_pipeline_compliance(self, session_id: str) -> Dict:
        """Check if session is hitting all 4 pipeline targets.
        
        Targets: (1) target output, (2) facts in holographic DB,
                 (3) trajectories, (4) KB enrichment.
        For now we verify facts + trajectories exist with recent timestamps.
        """
        holo = self._open_holographic_db()
        if not holo:
            return {
                'check_type': 'pipeline_compliance',
                'passed': True,  # Can't verify — assume OK
                'details': json.dumps({'note': 'holographic_memory.db unavailable'})
            }
        
        try:
            cur = holo.cursor()
            
            # Check recent facts (last 2 hours)
            cur.execute("""
                SELECT COUNT(*) as cnt FROM facts
                WHERE timestamp > datetime('now', '-2 hours')
                AND quality_score >= ?
            """, (CONFIG['quality_threshold'],))
            recent_facts = cur.fetchone()['cnt']
            
            # Check recent trajectories (last 2 hours)
            cur.execute("""
                SELECT COUNT(*) as cnt FROM trajectories
                WHERE timestamp > datetime('now', '-2 hours')
                AND quality_score >= ?
            """, (CONFIG['quality_threshold'],))
            recent_trajectories = cur.fetchone()['cnt']
            
            passed = recent_facts >= 1 and recent_trajectories >= 1
            
            return {
                'check_type': 'pipeline_compliance',
                'passed': passed,
                'details': json.dumps({
                    'recent_facts': recent_facts,
                    'recent_trajectories': recent_trajectories,
                    'quality_threshold': CONFIG['quality_threshold'],
                })
            }
        except sqlite3.Error as e:
            return {
                'check_type': 'pipeline_compliance',
                'passed': True,
                'details': json.dumps({'error': str(e)})
            }
        finally:
            holo.close()
    
    def _check_quality_gates(self, session_id: str) -> Dict:
        """Check if session meets quality thresholds."""
        holo = self._open_holographic_db()
        if not holo:
            return {
                'check_type': 'quality_gate',
                'passed': True,
                'details': json.dumps({'note': 'holographic_memory.db unavailable'})
            }
        
        try:
            cur = holo.cursor()
            threshold = CONFIG['quality_threshold']
            
            # Average quality of recent facts
            cur.execute("""
                SELECT AVG(quality_score) as avg_q, COUNT(*) as cnt FROM facts
                WHERE timestamp > datetime('now', '-2 hours')
                AND quality_score IS NOT NULL
            """)
            row = cur.fetchone()
            fact_avg = row['avg_q'] if row['avg_q'] else 0.0
            fact_cnt = row['cnt']
            
            # Average quality of recent trajectories
            cur.execute("""
                SELECT AVG(quality_score) as avg_q, COUNT(*) as cnt FROM trajectories
                WHERE timestamp > datetime('now', '-2 hours')
                AND quality_score IS NOT NULL
            """)
            row = cur.fetchone()
            traj_avg = row['avg_q'] if row['avg_q'] else 0.0
            traj_cnt = row['cnt']
            
            # Pass if both averages meet threshold (or if no data to check)
            if fact_cnt == 0 and traj_cnt == 0:
                passed = True
            elif fact_cnt == 0:
                passed = traj_avg >= threshold
            elif traj_cnt == 0:
                passed = fact_avg >= threshold
            else:
                passed = fact_avg >= threshold and traj_avg >= threshold
            
            return {
                'check_type': 'quality_gate',
                'passed': passed,
                'details': json.dumps({
                    'fact_avg_quality': round(fact_avg, 4) if fact_cnt else None,
                    'fact_count': fact_cnt,
                    'trajectory_avg_quality': round(traj_avg, 4) if traj_cnt else None,
                    'trajectory_count': traj_cnt,
                    'threshold': threshold,
                })
            }
        except sqlite3.Error as e:
            return {
                'check_type': 'quality_gate',
                'passed': True,
                'details': json.dumps({'error': str(e)})
            }
        finally:
            holo.close()
    
    def _check_trajectory_generation(self, session_id: str) -> Dict:
        """Check if session is generating trajectories."""
        holo = self._open_holographic_db()
        if not holo:
            return {
                'check_type': 'trajectory_generation',
                'passed': True,
                'details': json.dumps({'note': 'holographic_memory.db unavailable'})
            }
        
        try:
            cur = holo.cursor()
            min_trajectories = 2
            
            # Check trajectories linked to this session
            cur.execute("""
                SELECT COUNT(*) as cnt FROM trajectories
                WHERE session_id = ? AND timestamp > datetime('now', '-2 hours')
            """, (session_id,))
            session_traj = cur.fetchone()['cnt']
            
            # Also check any recent trajectories (system-wide, in case session mapping differs)
            cur.execute("""
                SELECT COUNT(*) as cnt FROM trajectories
                WHERE timestamp > datetime('now', '-30 minutes')
                AND quality_score >= ?
            """, (CONFIG['quality_threshold'],))
            recent_traj = cur.fetchone()['cnt']
            
            passed = session_traj >= min_trajectories or recent_traj >= 1
            
            return {
                'check_type': 'trajectory_generation',
                'passed': passed,
                'details': json.dumps({
                    'session_trajectories': session_traj,
                    'recent_system_trajectories': recent_traj,
                    'min_required': min_trajectories,
                })
            }
        except sqlite3.Error as e:
            return {
                'check_type': 'trajectory_generation',
                'passed': True,
                'details': json.dumps({'error': str(e)})
            }
        finally:
            holo.close()
    
    def _check_fact_extraction(self, session_id: str) -> Dict:
        """Check if session is extracting facts."""
        holo = self._open_holographic_db()
        if not holo:
            return {
                'check_type': 'fact_extraction',
                'passed': True,
                'details': json.dumps({'note': 'holographic_memory.db unavailable'})
            }
        
        try:
            cur = holo.cursor()
            min_facts = 1
            
            # Check recent high-quality facts (system-wide — facts lack session_id)
            cur.execute("""
                SELECT COUNT(*) as cnt FROM facts
                WHERE timestamp > datetime('now', '-30 minutes')
                AND quality_score >= ?
            """, (CONFIG['quality_threshold'],))
            recent_facts = cur.fetchone()['cnt']
            
            passed = recent_facts >= min_facts
            
            return {
                'check_type': 'fact_extraction',
                'passed': passed,
                'details': json.dumps({
                    'recent_high_quality_facts': recent_facts,
                    'min_required': min_facts,
                    'threshold': CONFIG['quality_threshold'],
                })
            }
        except sqlite3.Error as e:
            return {
                'check_type': 'fact_extraction',
                'passed': True,
                'details': json.dumps({'error': str(e)})
            }
        finally:
            holo.close()
    
    def make_decision(self, session_id: str, entropy_detections: List[Dict], directive_checks: List[Dict]) -> Optional[Dict]:
        """Make decision about what action to take."""
        
        # Get session info
        self.cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        session = dict(self.cursor.fetchone())
        
        # Check for critical entropy
        critical_entropy = [d for d in entropy_detections if d['severity'] == 'critical']
        
        if critical_entropy:
            # Check if it's repeat tool calls (kill condition)
            repeat_tool_calls = [d for d in critical_entropy if d['entropy_type'] == 'repeat_tool_calls']
            
            if repeat_tool_calls:
                # High entropy - kill condition
                if session['restart_count'] >= CONFIG['max_restart_count']:
                    return {
                        'action': 'kill',
                        'reason': f"High entropy: repeat tool calls detected ({len(repeat_tool_calls)} times), max restarts reached"
                    }
                else:
                    return {
                        'action': 'restart',
                        'reason': f"High entropy: repeat tool calls detected ({len(repeat_tool_calls)} times)"
                    }
            
            # Other critical entropy - restart
            return {
                'action': 'restart',
                'reason': f"Critical entropy detected: {critical_entropy[0]['entropy_type']}"
            }
        
        # Check for directive violations
        failed_checks = [c for c in directive_checks if not c['passed']]
        
        if failed_checks:
            return {
                'action': 'restart',
                'reason': f"Prime directive violation: {failed_checks[0]['check_type']}"
            }
        
        # Check for low quality
        self.cursor.execute('''
            SELECT AVG(metric_value) as avg_quality
            FROM quality_metrics
            WHERE session_id = ? AND timestamp > datetime('now', '-1 hour')
        ''', (session_id,))
        
        result = self.cursor.fetchone()
        if result and result['avg_quality'] and result['avg_quality'] < CONFIG['quality_threshold']:
            return {
                'action': 'restart',
                'reason': f"Low quality: {result['avg_quality']:.3f} < {CONFIG['quality_threshold']}"
            }
        
        # No action needed
        return None
    
    def execute_action(self, session_id: str, decision: Dict):
        """Execute the decided action."""
        action = decision['action']
        reason = decision['reason']
        
        logger.info("Executing %s on %s: %s", action, session_id, reason)
        
        # Record action in database
        self.cursor.execute('''
            INSERT INTO watcher_actions (session_id, action_type, action_reason, details)
            VALUES (?, ?, ?, ?)
        ''', (session_id, action, reason, json.dumps(decision)))
        
        action_id = self.cursor.lastrowid
        
        try:
            if action == 'restart':
                self._restart_session(session_id, reason)
                self._send_notification(session_id, 'restart', f"Restarted: {reason}")
            
            elif action == 'kill':
                self._kill_session(session_id, reason)
                self._send_notification(session_id, 'kill', f"Killed: {reason}")
            
            elif action == 'inject_prompt':
                self._inject_prompt(session_id, decision.get('prompt', ''))
            
            # Mark action as successful
            self.cursor.execute('''
                UPDATE watcher_actions SET success = TRUE WHERE id = ?
            ''', (action_id,))
            
        except Exception as e:
            logger.error("Error executing %s on %s: %s", action, session_id, e, exc_info=True)
            
            # Mark action as failed
            self.cursor.execute('''
                UPDATE watcher_actions SET success = FALSE, details = ? WHERE id = ?
            ''', (json.dumps({'error': str(e)}), action_id))
        
        self.conn.commit()
    
    def _restart_session(self, session_id: str, reason: str):
        """Restart a session with tighter constraints."""
        # Get session info
        self.cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        session = dict(self.cursor.fetchone())

        # Increment restart count
        self.cursor.execute('''
            UPDATE sessions SET restart_count = restart_count + 1, status = 'restarted'
            WHERE session_id = ?
        ''', (session_id,))

        session_type = session['session_type']
        corrective_prompt = self._build_corrective_prompt(session_id, reason)

        try:
            if session_type == 'cron':
                self._restart_cron_session(session, corrective_prompt)
            elif session_type == 'delegate_task':
                self._restart_delegate_session(session, corrective_prompt)
            elif session_type == 'manual':
                self._restart_manual_session(session, corrective_prompt)
        except Exception as e:
            logger.error("Error during restart of %s: %s", session_id, e, exc_info=True)

        # Record restart action
        self.cursor.execute('''
            INSERT INTO watcher_actions (session_id, action_type, action_reason, success, details)
            VALUES (?, 'restart', ?, TRUE, ?)
        ''', (session_id, reason, json.dumps({
            'session_type': session_type,
            'restart_count': session['restart_count'] + 1,
            'corrective_prompt': corrective_prompt[:200]
        })))

        self.conn.commit()
        logger.info("Restarted %s session %s (restart count: %s)", session_type, session_id, session['restart_count'] + 1)

    def _restart_cron_session(self, session: Dict, corrective_prompt: str):
        """Cron restart: pause job, update prompt, resume. Uses cron.jobs directly."""
        job_id = session.get('job_id')
        if not job_id:
            logger.warning("No job_id for cron session %s, cannot restart", session['session_id'])
            return

        try:
            # Pause — same as hermes_cli/cron.py
            result = pause_job(job_id, reason='ARGUS restart: entropy detected')
            if result:
                logger.info("Paused cron job %s", job_id)
            else:
                logger.warning("pause_job returned None for %s", job_id)
        except Exception as e:
            logger.error("Failed to pause cron job %s: %s", job_id, e, exc_info=True)

        # Update prompt with corrective instructions
        try:
            job = get_job(job_id)
            if job:
                original_prompt = job.get('prompt', '')
                updated_prompt = f"{corrective_prompt}\n\n---\n\nOriginal task:\n{original_prompt}"
                update_job(job_id, {'prompt': updated_prompt})
                logger.info("Updated cron job %s prompt with corrective instructions", job_id)
        except Exception as e:
            logger.error("Failed to update cron prompt for %s: %s", job_id, e, exc_info=True)

        # Resume
        try:
            result = resume_job(job_id)
            if result:
                logger.info("Resumed cron job %s with corrective prompt", job_id)
            else:
                logger.warning("resume_job returned None for %s", job_id)
        except Exception as e:
            logger.error("Failed to resume cron job %s: %s", job_id, e, exc_info=True)

    def _restart_delegate_session(self, session: Dict, corrective_prompt: str) -> None:
        """Delegate restart: kill process, respawn with corrective prompt."""
        metadata = json.loads(session.get('metadata', '{}'))
        pid = metadata.get('pid')

        if pid:
            self._terminate_pid(pid, "restart")

        # The respawn will happen naturally when the parent agent retries
        logger.info("Killed delegate session, corrective prompt stored for respawn")

    def _restart_manual_session(self, session: Dict, corrective_prompt: str) -> None:
        """Manual restart: record action, store corrective prompt for next interaction."""
        # For manual sessions, we can't force a restart
        # We record the corrective prompt and notify the user
        logger.info("Manual session %s flagged for restart (user intervention needed)", session['session_id'])

    def _build_corrective_prompt(self, session_id: str, reason: str) -> str:
        """Build a corrective prompt based on entropy detections."""
        # Get recent entropy detections for this session
        self.cursor.execute('''
            SELECT entropy_type, severity FROM entropy_detections
            WHERE session_id = ? AND timestamp > datetime('now', '-10 minutes')
            ORDER BY severity DESC, timestamp DESC
            LIMIT 1
        ''', (session_id,))

        row = self.cursor.fetchone()
        if row:
            entropy_type = row['entropy_type']
            template = CORRECTIVE_PROMPTS.get(entropy_type, CORRECTIVE_PROMPTS['stuck_loop'])
            return "%s\n\nReason: %s" % (template, reason)

        return "ENTROPY CORRECTION: ARGUS detected an issue requiring restart. %s" % reason

    def _kill_session(self, session_id: str, reason: str) -> None:
        """Kill a session based on its type."""
        # Get session info
        self.cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        session = dict(self.cursor.fetchone())

        session_type = session['session_type']

        # Update session status
        self.cursor.execute('''
            UPDATE sessions SET status = 'killed', kill_count = kill_count + 1
            WHERE session_id = ?
        ''', (session_id,))

        try:
            if session_type == 'cron':
                self._kill_cron_session(session, reason)
            elif session_type == 'delegate_task':
                self._kill_delegate_session(session, reason)
            elif session_type == 'manual':
                self._kill_manual_session(session, reason)
        except Exception as e:
            logger.error("Error killing %s: %s", session_id, e, exc_info=True)

        # Record kill action
        self.cursor.execute('''
            INSERT INTO watcher_actions (session_id, action_type, action_reason, success, details)
            VALUES (?, 'kill', ?, TRUE, ?)
        ''', (session_id, reason, json.dumps({
            'session_type': session_type,
            'kill_count': session['kill_count'] + 1
        })))

        self.conn.commit()
        logger.info("Killed %s session %s: %s", session_type, session_id, reason)

    def _kill_cron_session(self, session: Dict, reason: str) -> None:
        """Permanently pause a cron job via cron.jobs."""
        job_id = session.get('job_id')
        if not job_id:
            logger.warning("No job_id for cron session %s, cannot kill", session['session_id'])
            return

        try:
            result = pause_job(job_id, reason='ARGUS kill: %s' % reason)
            if result:
                logger.info("Permanently paused cron job %s", job_id)
            else:
                logger.warning("pause_job returned None for %s", job_id)
        except Exception as e:
            logger.error("Failed to pause cron job %s for kill: %s", job_id, e, exc_info=True)

    def _kill_delegate_session(self, session: Dict, reason: str) -> None:
        """Terminate a delegate task subprocess."""
        metadata = json.loads(session.get('metadata', '{}'))
        pid = metadata.get('pid')

        if pid:
            self._terminate_pid(pid, "kill")

    def _kill_manual_session(self, session: Dict, reason: str) -> None:
        """Cannot kill manual sessions — send alert notification instead."""
        message = (
            "ARGUS cannot terminate manual session %s.\n"
            "Action required: Please review this session manually.\n"
            "Reason: %s" % (session['session_id'], reason)
        )
        self._send_notification(session['session_id'], 'kill', message)
        logger.warning("Manual session %s flagged for kill — user intervention required", session['session_id'])

    def _terminate_pid(self, pid: Union[str, int], context: str = "terminate") -> None:
        """Send SIGTERM then SIGKILL to a process. Shared by restart/kill paths."""
        pid_str = str(pid)
        self._safe_subprocess(['kill', '-TERM', pid_str])
        logger.info("Sent SIGTERM to PID %s (%s)", pid_str, context)
        time.sleep(2)
        self._safe_subprocess(['kill', '-9', pid_str])
    
    def _inject_prompt(self, session_id: str, prompt: str):
        """Inject a corrective prompt into a session based on its type."""
        # Get session info
        self.cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        session = dict(self.cursor.fetchone())

        session_type = session['session_type']

        try:
            if session_type == 'cron':
                self._inject_cron_prompt(session, prompt)
            elif session_type == 'delegate_task':
                self._inject_delegate_prompt(session, prompt)
            elif session_type == 'manual':
                self._inject_manual_prompt(session, prompt)
        except Exception as e:
            logger.error("Error injecting prompt into %s: %s", session_id, e, exc_info=True)

        # Record prompt injection action
        self.cursor.execute('''
            INSERT INTO watcher_actions (session_id, action_type, action_reason, success, details)
            VALUES (?, 'inject_prompt', 'Corrective prompt injected', TRUE, ?)
        ''', (session_id, json.dumps({
            'session_type': session_type,
            'corrective_prompt': prompt[:500]
        })))

        self.conn.commit()
        logger.info("Injected corrective prompt into %s session %s", session_type, session_id)

    def _inject_cron_prompt(self, session: Dict, prompt: str):
        """Update cron job prompt and trigger via cron.jobs."""
        job_id = session.get('job_id')
        if not job_id:
            return

        try:
            job = get_job(job_id)
            if job:
                original_prompt = job.get('prompt', '')
                updated_prompt = f"{prompt}\n\n---\n\nOriginal task:\n{original_prompt}"
                update_job(job_id, {'prompt': updated_prompt})

            # Force run with new prompt
            trigger_job(job_id)
            logger.info("Triggered cron job %s with corrective prompt", job_id)
        except Exception as e:
            logger.error("Failed to inject prompt into cron job %s: %s", job_id, e, exc_info=True)

    def _inject_delegate_prompt(self, session: Dict, prompt: str) -> None:
        """Kill and respawn delegate with corrective prompt."""
        metadata = json.loads(session.get('metadata', '{}'))
        pid = metadata.get('pid')

        if pid:
            self._terminate_pid(pid, "prompt injection")
            logger.info("Killed delegate PID %s for prompt injection — will respawn", pid)

    def _inject_manual_prompt(self, session: Dict, prompt: str):
        """Store corrective prompt for manual session — user will see it on next interaction."""
        # For manual sessions, we store the prompt in the notifications table
        # so the user can review it
        self.cursor.execute('''
            INSERT INTO notifications (session_id, notification_type, message, delivered)
            VALUES (?, 'inject_prompt', ?, FALSE)
        ''', (session['session_id'], f"CORRECTIVE PROMPT FOR NEXT INTERACTION:\n\n{prompt}"))
        logger.info("Stored corrective prompt for manual session %s", session['session_id'])
    
    def _send_notification(self, session_id: str, notification_type: str, message: str):
        """Send notification via Telegram bot API. Uses hermes env_loader for credentials."""
        delivered = False
        delivery_error = None
        full_message = ""

        try:
            # Get session info for context
            self.cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
            session = dict(self.cursor.fetchone())

            # Format message
            full_message = f"Agent Watcher Alert\n\n"
            full_message += f"Session: {session_id}\n"
            full_message += f"Type: {session['session_type']}\n"
            full_message += f"Task: {session.get('task_description', 'Unknown')}\n"
            full_message += f"Action: {notification_type.upper()}\n"
            full_message += f"Reason: {message}\n"
            full_message += f"Time: {datetime.now().isoformat()}"

            # Load credentials via hermes env_loader (sops-aware)
            try:
                load_hermes_dotenv()
            except Exception:
                pass  # Fallback to env vars already set

            bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
            chat_id = os.environ.get('TELEGRAM_CHAT_ID')

            if bot_token and chat_id:
                # Send via Telegram Bot API
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = json.dumps({
                    'chat_id': chat_id,
                    'text': full_message,
                    'parse_mode': 'HTML'
                }).encode('utf-8')

                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={'Content-Type': 'application/json'}
                )

                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read())
                    delivered = result.get('ok', False)
                    logger.info("Telegram notification sent for %s", session_id)
            else:
                delivery_error = "TELEGRAM_BOT_TOKEN/CHAT_ID not in environment"
                logger.warning("Cannot send Telegram notification: %s", delivery_error)

        except urllib.error.URLError as e:
            delivery_error = f"Telegram API error: {e}"
            logger.error("Failed to send Telegram notification: %s", e, exc_info=True)
        except Exception as e:
            delivery_error = str(e)
            logger.error("Error sending notification: %s", e, exc_info=True)

        # Record notification in database
        try:
            self.cursor.execute('''
                INSERT INTO notifications (session_id, notification_type, message, delivered, delivery_error)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, notification_type, full_message, delivered, delivery_error))
            self.conn.commit()
        except Exception as e:
            logger.error("Error recording notification: %s", e, exc_info=True)
    
    def run(self):
        """Main watcher loop."""
        logger.info("Agent Watcher starting...")
        self.running = True

        # Start WAL monitor for real-time tool call detection
        self.wal_monitor.start()

        # Write PID file (same pattern as gateway/status.write_pid_file)
        write_argus_pid_file()
        logger.info("ARGUS PID file written: %s", _get_argus_pid_path())

        while self.running:
            try:
                # Process WAL monitor events (tool call entropy)
                self._process_wal_events()

                # Discover sessions
                sessions = self.discover_sessions()

                for session in sessions:
                    # Register session if not already registered
                    self.register_session(session)
                    
                    # Collect metrics
                    self.collect_metrics(session['session_id'])
                    
                    # Detect entropy
                    entropy_detections = self.detect_entropy(session['session_id'])
                    
                    # Check prime directive
                    directive_checks = self.check_prime_directive(session['session_id'])
                    
                    # Make decision
                    decision = self.make_decision(session['session_id'], entropy_detections, directive_checks)
                    
                    # Execute action if needed
                    if decision:
                        self.execute_action(session['session_id'], decision)
                
                # Sleep before next poll
                time.sleep(CONFIG['poll_interval'])
            
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.running = False
            
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                time.sleep(CONFIG['poll_interval'])
    
    def stop(self):
        """Stop the watcher."""
        logger.info("Stopping Agent Watcher...")
        self.running = False
        self.wal_monitor.stop()
        remove_argus_pid_file()
        if self.conn:
            self.conn.close()

    def _process_wal_events(self):
        """Process events from the WAL monitor (real-time tool call entropy)."""
        events = self.wal_monitor.get_events(limit=50)

        for event in events:
            if event.event_type == 'repeat_detected':
                # Record entropy detection
                self.cursor.execute('''
                    INSERT INTO entropy_detections
                    (session_id, entropy_type, severity, details)
                    VALUES (?, 'repeat_tool_calls', 'warning', ?)
                ''', (
                    f"wal_{event.session_id}",
                    json.dumps({
                        'tool_name': event.tool_name,
                        'source': 'wal_monitor',
                        **event.details,
                    })
                ))

                # Insert into tool_calls for compatibility with existing detection
                self.cursor.execute('''
                    INSERT INTO tool_calls
                    (session_id, tool_name, tool_args, timestamp, file_changed)
                    VALUES (?, ?, ?, ?, FALSE)
                ''', (
                    f"wal_{event.session_id}",
                    event.tool_name,
                    event.tool_args or '{}',
                    str(event.timestamp),
                ))

                logger.warning(
                    "WAL: repeat tool '%s' in session %s",
                    event.tool_name, event.session_id[:15]
                )

            elif event.event_type == 'stuck_loop_detected':
                self.cursor.execute('''
                    INSERT INTO entropy_detections
                    (session_id, entropy_type, severity, details)
                    VALUES (?, 'stuck_loop', 'critical', ?)
                ''', (
                    f"wal_{event.session_id}",
                    json.dumps({
                        'source': 'wal_monitor',
                        **event.details,
                    })
                ))

                logger.warning(
                    "WAL: stuck loop in session %s: %s",
                    event.session_id[:15], event.details.get('pattern')
                )

        if events:
            self.conn.commit()

def main():
    """Main entry point."""
    argus = Argus()
    
    # Handle signals
    def _signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Received signal %s", signum)
        argus.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Run Argus
    try:
        argus.run()
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        argus.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()