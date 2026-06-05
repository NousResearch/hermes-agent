#!/usr/bin/env python3
"""Self-Model — 理性层：成功率追踪、反模式库、策略自动调整

使用方法：
    python3 self_model.py init
    python3 self_model.py record --task <任务> --action <动作> --tool <工具> --success <0/1>
    python3 self_model.py status
    python3 self_model.py advise --task <任务>
    python3 self_model.py anti-pattern --pattern <错误> --correct <正确>
    python3 self_model.py analyze
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def get_evolution_dir() -> Path:
    """Get the evolution data directory."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(hermes_home) / "evolution"


def get_db_path() -> Path:
    """Get the self-model database path."""
    return get_evolution_dir() / "self-model.db"


def init_db():
    """Initialize the self-model database."""
    evolution_dir = get_evolution_dir()
    evolution_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create outcomes table with all columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            task TEXT NOT NULL,
            action TEXT NOT NULL,
            tool TEXT NOT NULL,
            success INTEGER NOT NULL,
            duration REAL DEFAULT 0,
            domain TEXT DEFAULT '通用',
            error_type TEXT DEFAULT '',
            error_msg TEXT DEFAULT '',
            details TEXT
        )
    ''')
    
    # Create anti_patterns table with last_seen
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anti_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pattern TEXT NOT NULL,
            correct TEXT NOT NULL,
            domain TEXT DEFAULT '通用',
            frequency INTEGER DEFAULT 1,
            last_seen TEXT
        )
    ''')
    
    # Create self_model table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS self_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            details TEXT
        )
    ''')
    
    # Migrate existing tables if needed (add missing columns)
    try:
        cursor.execute("ALTER TABLE outcomes ADD COLUMN duration REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE outcomes ADD COLUMN domain TEXT DEFAULT '通用'")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE outcomes ADD COLUMN error_type TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE outcomes ADD COLUMN error_msg TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE anti_patterns ADD COLUMN last_seen TEXT")
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()
    
    print(f"✅ Self-model database initialized: {db_path}")


def record_outcome(task: str, action: str, tool: str, success: int, details: str = ""):
    """Record an execution outcome."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 self_model.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO outcomes (timestamp, task, action, tool, success, details)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, task, action, tool, success, details))
    
    conn.commit()
    conn.close()
    
    status = "✓" if success else "✗"
    print(f"{status} Recorded: {task} / {action} / {tool}")


def show_status():
    """Show current self-model status."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 self_model.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Total outcomes
    cursor.execute("SELECT COUNT(*) FROM outcomes")
    total = cursor.fetchone()[0]
    
    # Success rate
    cursor.execute("SELECT COUNT(*) FROM outcomes WHERE success = 1")
    success_count = cursor.fetchone()[0]
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    # Anti-patterns count
    cursor.execute("SELECT COUNT(*) FROM anti_patterns")
    anti_patterns_count = cursor.fetchone()[0]
    
    # Recent outcomes
    cursor.execute('''
        SELECT task, action, tool, success, timestamp
        FROM outcomes
        ORDER BY timestamp DESC
        LIMIT 5
    ''')
    recent = cursor.fetchall()
    
    conn.close()
    
    print("📊 Self-Model Status")
    print("=" * 40)
    print(f"Total outcomes: {total}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Anti-patterns: {anti_patterns_count}")
    print()
    
    if recent:
        print("Recent outcomes:")
        for task, action, tool, success, timestamp in recent:
            status = "✓" if success else "✗"
            print(f"  {status} {task} / {action} / {tool}")


def advise_for_task(task: str):
    """Get strategy advice for a task."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 self_model.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get success rate for this task
    cursor.execute('''
        SELECT COUNT(*) as total, SUM(success) as successes
        FROM outcomes
        WHERE task = ?
    ''', (task,))
    
    row = cursor.fetchone()
    total = row[0]
    successes = row[1] or 0
    
    if total == 0:
        print(f"ℹ️ No history for task: {task}")
        print("   Recommendation: Proceed with caution, record outcomes.")
    else:
        success_rate = (successes / total * 100)
        print(f"📊 Task: {task}")
        print(f"   History: {total} attempts, {success_rate:.1f}% success rate")
        
        if success_rate >= 80:
            print("   Recommendation: High confidence. Proceed normally.")
        elif success_rate >= 50:
            print("   Recommendation: Medium confidence. Verify results.")
        else:
            print("   Recommendation: Low confidence. Consider alternative approach.")
    
    # Get related anti-patterns
    cursor.execute('''
        SELECT pattern, correct
        FROM anti_patterns
        WHERE domain = ? OR domain = '通用'
    ''', (task,))
    
    anti_patterns = cursor.fetchall()
    if anti_patterns:
        print(f"\n⚠️ Related anti-patterns ({len(anti_patterns)}):")
        for pattern, correct in anti_patterns:
            print(f"   ✗ {pattern}")
            print(f"     → {correct}")
    
    conn.close()


def add_anti_pattern(pattern: str, correct: str, domain: str = "通用"):
    """Add an anti-pattern."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 self_model.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    # Check if pattern already exists
    cursor.execute("SELECT id, frequency FROM anti_patterns WHERE pattern = ?", (pattern,))
    existing = cursor.fetchone()
    
    if existing:
        # Increment frequency
        cursor.execute('''
            UPDATE anti_patterns
            SET frequency = frequency + 1, timestamp = ?
            WHERE id = ?
        ''', (timestamp, existing[0]))
        print(f"⚠️ Anti-pattern frequency incremented: {pattern}")
    else:
        # Insert new pattern
        cursor.execute('''
            INSERT INTO anti_patterns (timestamp, pattern, correct, domain)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, pattern, correct, domain))
        print(f"✓ Anti-pattern added: {pattern}")
    
    conn.commit()
    conn.close()


def analyze_history():
    """Analyze historical patterns."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 self_model.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Task success rates
    cursor.execute('''
        SELECT task, COUNT(*) as total, SUM(success) as successes
        FROM outcomes
        GROUP BY task
        ORDER BY total DESC
    ''')
    
    tasks = cursor.fetchall()
    
    print("📊 Analysis")
    print("=" * 40)
    
    if tasks:
        print("\nTask success rates:")
        for task, total, successes in tasks:
            success_rate = (successes / total * 100) if total > 0 else 0
            print(f"  {task}: {total} attempts, {success_rate:.1f}% success")
    
    # Tool success rates
    cursor.execute('''
        SELECT tool, COUNT(*) as total, SUM(success) as successes
        FROM outcomes
        GROUP BY tool
        ORDER BY total DESC
    ''')
    
    tools = cursor.fetchall()
    
    if tools:
        print("\nTool success rates:")
        for tool, total, successes in tools:
            success_rate = (successes / total * 100) if total > 0 else 0
            print(f"  {tool}: {total} attempts, {success_rate:.1f}% success")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Self-Model — 理性层")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # init
    subparsers.add_parser("init", help="初始化数据库")
    
    # record
    record_parser = subparsers.add_parser("record", help="记录执行结果")
    record_parser.add_argument("--task", required=True, help="任务类型")
    record_parser.add_argument("--action", required=True, help="执行动作")
    record_parser.add_argument("--tool", required=True, help="使用的工具")
    record_parser.add_argument("--success", type=int, required=True, help="是否成功（0或1）")
    record_parser.add_argument("--details", default="", help="详细信息")
    
    # status
    subparsers.add_parser("status", help="查看状态")
    
    # advise
    advise_parser = subparsers.add_parser("advise", help="获取策略建议")
    advise_parser.add_argument("--task", required=True, help="任务类型")
    
    # anti-pattern
    anti_pattern_parser = subparsers.add_parser("anti-pattern", help="记录反模式")
    anti_pattern_parser.add_argument("--pattern", required=True, help="错误模式")
    anti_pattern_parser.add_argument("--correct", required=True, help="正确做法")
    anti_pattern_parser.add_argument("--domain", default="通用", help="领域")
    
    # analyze
    subparsers.add_parser("analyze", help="分析历史")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_db()
    elif args.command == "record":
        record_outcome(args.task, args.action, args.tool, args.success, args.details)
    elif args.command == "status":
        show_status()
    elif args.command == "advise":
        advise_for_task(args.task)
    elif args.command == "anti-pattern":
        add_anti_pattern(args.pattern, args.correct, args.domain)
    elif args.command == "analyze":
        analyze_history()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
