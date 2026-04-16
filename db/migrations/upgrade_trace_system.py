"""
Migration 006: Upgrade Trace System to V2

This migration:
1. Creates new trace_v2.db with optimized schema
2. Migrates existing data from traces.jsonl to new database
3. Maintains backward compatibility with original trace system
"""

import json
import logging
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Paths
TRACE_DIR = Path.home() / ".hermes" / "trace"
TRACE_JSONL_PATH = TRACE_DIR / "traces.jsonl"
TRACE_DB_PATH = TRACE_DIR / "trace_v2.db"
SCHEMA_PATH = Path(__file__).parent.parent / "trace_schema_v2.sql"


def upgrade(dry_run: bool = False) -> Dict[str, Any]:
    """
    Upgrade trace system to V2.
    
    Args:
        dry_run: If True, only simulate the migration
        
    Returns:
        Migration result summary
    """
    result = {
        "status": "success",
        "started_at": datetime.now().isoformat(),
        "steps": [],
        "errors": [],
        "stats": {
            "sessions_created": 0,
            "traces_created": 0,
            "events_migrated": 0,
            "skipped": 0
        }
    }
    
    try:
        # Step 1: Check prerequisites
        result["steps"].append("Checking prerequisites")
        if not TRACE_JSONL_PATH.exists():
            result["status"] = "skipped"
            result["message"] = "No existing traces.jsonl file found"
            return result
        
        # Step 2: Create backup
        result["steps"].append("Creating backup")
        if not dry_run:
            backup_path = TRACE_JSONL_PATH.with_suffix('.jsonl.backup')
            shutil.copy2(TRACE_JSONL_PATH, backup_path)
            result["backup_path"] = str(backup_path)
        
        # Step 3: Initialize new database
        result["steps"].append("Initializing V2 database")
        if not dry_run:
            if TRACE_DB_PATH.exists():
                # Backup existing V2 database
                backup_db_path = TRACE_DB_PATH.with_suffix('.db.backup')
                shutil.copy2(TRACE_DB_PATH, backup_db_path)
            
            # Create new database with schema
            conn = sqlite3.connect(str(TRACE_DB_PATH))
            conn.row_factory = sqlite3.Row
            
            # Read and execute schema
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            conn.executescript(schema_sql)
            
            result["steps"].append("Database schema created")
        
        # Step 4: Read existing traces
        result["steps"].append("Reading existing traces")
        events_by_session: Dict[str, List[Dict[str, Any]]] = {}
        events_by_trace: Dict[str, List[Dict[str, Any]]] = {}
        
        with open(TRACE_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event = json.loads(line.strip())
                    trace_id = event.get("trace_id")
                    
                    if not trace_id:
                        result["stats"]["skipped"] += 1
                        continue
                    
                    # Group by trace_id (treating each trace_id as a session)
                    if trace_id not in events_by_trace:
                        events_by_trace[trace_id] = []
                    events_by_trace[trace_id].append(event)
                    
                    result["stats"]["events_migrated"] += 1
                    
                except json.JSONDecodeError as e:
                    result["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                    result["stats"]["skipped"] += 1
                except Exception as e:
                    result["errors"].append(f"Line {line_num}: {e}")
                    result["stats"]["skipped"] += 1
        
        result["steps"].append(f"Found {len(events_by_trace)} traces with {result['stats']['events_migrated']} events")
        
        # Step 5: Migrate to new database
        if not dry_run:
            result["steps"].append("Migrating data to V2 database")
            
            for trace_id, events in events_by_trace.items():
                try:
                    # Create session (using trace_id as session_id for backward compatibility)
                    session_id = trace_id
                    
                    # Get timestamps
                    timestamps = [e.get("timestamp", "") for e in events if e.get("timestamp")]
                    started_at = min(timestamps) if timestamps else datetime.now().isoformat()
                    ended_at = max(timestamps) if timestamps else None
                    
                    # Determine status
                    has_errors = any(e.get("error") for e in events)
                    status = "error" if has_errors else "completed"
                    
                    # Count tool calls
                    tool_calls = set()
                    for event in events:
                        if event.get("event_type") in ("tool_start", "tool_complete"):
                            tool_calls.add(event.get("tool_name", "unknown"))
                    
                    # Insert session
                    conn.execute("""
                        INSERT OR IGNORE INTO trace_sessions 
                        (session_id, started_at, ended_at, status, event_count, trace_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        started_at,
                        ended_at,
                        status,
                        len(events),
                        1  # Each trace_id becomes one session
                    ))
                    
                    # Insert trace
                    conn.execute("""
                        INSERT OR IGNORE INTO trace_traces 
                        (trace_id, session_id, started_at, ended_at, status, event_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        trace_id,
                        session_id,
                        started_at,
                        ended_at,
                        status,
                        len(events)
                    ))
                    
                    # Insert events
                    for event in events:
                        event_type = event.get("event_type", "custom")
                        timestamp = event.get("timestamp", datetime.now().isoformat())
                        
                        # Extract tool_call_id if present (generate from event data)
                        tool_call_id = None
                        if event_type in ("tool_start", "tool_complete"):
                            tool_name = event.get("tool_name", "")
                            tool_args_hash = hash(str(event.get("tool_args", "")))
                            tool_call_id = f"{tool_name}_{tool_args_hash}"[:32]
                        
                        conn.execute("""
                            INSERT INTO trace_events 
                            (session_id, trace_id, tool_call_id, event_type, timestamp,
                             duration_ms, tool_name, tool_args, tool_result, error,
                             model, response_preview, extra)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            session_id,
                            trace_id,
                            tool_call_id,
                            event_type,
                            timestamp,
                            event.get("duration_ms"),
                            event.get("tool_name"),
                            event.get("tool_args"),
                            event.get("tool_result"),
                            event.get("error"),
                            event.get("model"),
                            event.get("response_preview"),
                            json.dumps({k: v for k, v in event.items() 
                                       if k not in ("trace_id", "event_type", "timestamp", 
                                                   "duration_ms", "tool_name", "tool_args",
                                                   "tool_result", "error", "model", 
                                                   "response_preview", "trace_id_short")})
                        ))
                    
                    result["stats"]["sessions_created"] += 1
                    result["stats"]["traces_created"] += 1
                    
                except Exception as e:
                    result["errors"].append(f"Failed to migrate trace {trace_id}: {e}")
            
            conn.commit()
            conn.close()
        
        result["steps"].append("Migration completed")
        result["completed_at"] = datetime.now().isoformat()
        
        # Step 6: Update configuration to enable V2
        if not dry_run:
            result["steps"].append("Updating configuration")
            _update_config_for_v2()
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        logger.error(f"Migration failed: {e}", exc_info=True)
    
    return result


def downgrade() -> Dict[str, Any]:
    """
    Downgrade from V2 to V1 trace system.
    
    Returns:
        Downgrade result summary
    """
    result = {
        "status": "success",
        "started_at": datetime.now().isoformat(),
        "steps": [],
        "errors": []
    }
    
    try:
        # Step 1: Check if V2 database exists
        if not TRACE_DB_PATH.exists():
            result["status"] = "skipped"
            result["message"] = "No V2 database found"
            return result
        
        # Step 2: Create backup
        result["steps"].append("Creating backup")
        backup_path = TRACE_DB_PATH.with_suffix('.db.downgrade_backup')
        shutil.copy2(TRACE_DB_PATH, backup_path)
        result["backup_path"] = str(backup_path)
        
        # Step 3: Remove V2 database
        result["steps"].append("Removing V2 database")
        TRACE_DB_PATH.unlink()
        
        # Step 4: Update configuration to disable V2
        result["steps"].append("Updating configuration")
        _update_config_for_v1()
        
        result["steps"].append("Downgrade completed")
        result["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        logger.error(f"Downgrade failed: {e}", exc_info=True)
    
    return result


def _update_config_for_v2():
    """Update configuration to use V2 trace system."""
    config_path = Path.home() / ".hermes" / "config.yaml"
    
    if not config_path.exists():
        return
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Ensure trace section exists
        if "trace" not in config:
            config["trace"] = {}
        
        # Set V2 flags
        config["trace"]["v2_enabled"] = True
        config["trace"]["v2_db_path"] = str(TRACE_DB_PATH)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Configuration updated for V2 trace system")
        
    except Exception as e:
        logger.warning(f"Failed to update configuration: {e}")


def _update_config_for_v1():
    """Update configuration to use V1 trace system."""
    config_path = Path.home() / ".hermes" / "config.yaml"
    
    if not config_path.exists():
        return
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Ensure trace section exists
        if "trace" not in config:
            config["trace"] = {}
        
        # Remove V2 flags
        config["trace"]["v2_enabled"] = False
        config["trace"].pop("v2_db_path", None)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Configuration updated for V1 trace system")
        
    except Exception as e:
        logger.warning(f"Failed to update configuration: {e}")


def check_migration_status() -> Dict[str, Any]:
    """
    Check the current migration status.
    
    Returns:
        Status information
    """
    status = {
        "v1_exists": TRACE_JSONL_PATH.exists(),
        "v2_exists": TRACE_DB_PATH.exists(),
        "v1_size": TRACE_JSONL_PATH.stat().st_size if TRACE_JSONL_PATH.exists() else 0,
        "v2_size": TRACE_DB_PATH.stat().st_size if TRACE_DB_PATH.exists() else 0,
        "v1_event_count": 0,
        "v2_session_count": 0,
        "v2_event_count": 0
    }
    
    # Count V1 events
    if status["v1_exists"]:
        try:
            with open(TRACE_JSONL_PATH, 'r', encoding='utf-8') as f:
                status["v1_event_count"] = sum(1 for _ in f)
        except Exception:
            pass
    
    # Count V2 sessions and events
    if status["v2_exists"]:
        try:
            conn = sqlite3.connect(str(TRACE_DB_PATH))
            cursor = conn.execute("SELECT COUNT(*) FROM trace_sessions")
            status["v2_session_count"] = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM trace_events")
            status["v2_event_count"] = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass
    
    return status


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trace System V2 Migration")
    parser.add_argument("action", choices=["upgrade", "downgrade", "status"],
                       help="Migration action")
    parser.add_argument("--dry-run", action="store_true",
                       help="Simulate migration without making changes")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.action == "upgrade":
        result = upgrade(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    elif args.action == "downgrade":
        result = downgrade()
        print(json.dumps(result, indent=2))
    elif args.action == "status":
        status = check_migration_status()
        print(json.dumps(status, indent=2))