#!/usr/bin/env python3
"""Fusion Engine — 感性层：情绪状态、融合决策、进化报告

使用方法：
    python3 fusion_engine.py init
    python3 fusion_engine.py report [--days <天数>]
    python3 fusion_engine.py status
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path


def get_evolution_dir() -> Path:
    """Get the evolution data directory."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(hermes_home) / "evolution"


def get_db_path() -> Path:
    """Get the fusion-state database path."""
    return get_evolution_dir() / "fusion-state.db"


def init_db():
    """Initialize the fusion-state database."""
    evolution_dir = get_evolution_dir()
    evolution_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create emotional_state table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotional_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            emotion TEXT NOT NULL,
            intensity REAL NOT NULL,
            trigger TEXT,
            context TEXT
        )
    ''')
    
    # Create fusion_decisions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fusion_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            situation TEXT NOT NULL,
            rational_score REAL,
            emotional_score REAL,
            final_decision TEXT,
            outcome TEXT
        )
    ''')
    
    # Create evolution_metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evolution_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            details TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"✅ Fusion-state database initialized: {db_path}")


def record_emotion(emotion: str, intensity: float, trigger: str = "", context: str = ""):
    """Record an emotional state."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 fusion_engine.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO emotional_state (timestamp, emotion, intensity, trigger, context)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, emotion, intensity, trigger, context))
    
    conn.commit()
    conn.close()
    
    print(f"🎭 Recorded emotion: {emotion} (intensity: {intensity})")


def show_status():
    """Show current fusion-state status."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 fusion_engine.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Total emotional states
    cursor.execute("SELECT COUNT(*) FROM emotional_state")
    total_emotions = cursor.fetchone()[0]
    
    # Recent emotions
    cursor.execute('''
        SELECT emotion, intensity, trigger, timestamp
        FROM emotional_state
        ORDER BY timestamp DESC
        LIMIT 5
    ''')
    recent_emotions = cursor.fetchall()
    
    # Total decisions
    cursor.execute("SELECT COUNT(*) FROM fusion_decisions")
    total_decisions = cursor.fetchone()[0]
    
    conn.close()
    
    print("🎭 Fusion-State Status")
    print("=" * 40)
    print(f"Total emotional states: {total_emotions}")
    print(f"Total fusion decisions: {total_decisions}")
    print()
    
    if recent_emotions:
        print("Recent emotions:")
        for emotion, intensity, trigger, timestamp in recent_emotions:
            print(f"  {emotion} (intensity: {intensity}) - {trigger}")


def generate_report(days: int = 7):
    """Generate an evolution report."""
    db_path = get_db_path()
    if not db_path.exists():
        print("❌ Database not initialized. Run: python3 fusion_engine.py init")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Emotional states in period
    cursor.execute('''
        SELECT emotion, COUNT(*) as count, AVG(intensity) as avg_intensity
        FROM emotional_state
        WHERE timestamp >= ?
        GROUP BY emotion
        ORDER BY count DESC
    ''', (start_date.isoformat(),))
    
    emotions = cursor.fetchall()
    
    # Fusion decisions in period
    cursor.execute('''
        SELECT COUNT(*) as total, AVG(rational_score) as avg_rational, AVG(emotional_score) as avg_emotional
        FROM fusion_decisions
        WHERE timestamp >= ?
    ''', (start_date.isoformat(),))
    
    decision_stats = cursor.fetchone()
    
    conn.close()
    
    print(f"📊 Evolution Report ({days} days)")
    print("=" * 40)
    
    if emotions:
        print("\nEmotional states:")
        for emotion, count, avg_intensity in emotions:
            print(f"  {emotion}: {count} times, avg intensity: {avg_intensity:.2f}")
    else:
        print("\nNo emotional states recorded in this period.")
    
    if decision_stats and decision_stats[0] > 0:
        total, avg_rational, avg_emotional = decision_stats
        print(f"\nFusion decisions: {total}")
        print(f"  Avg rational score: {avg_rational:.2f}")
        print(f"  Avg emotional score: {avg_emotional:.2f}")
    else:
        print("\nNo fusion decisions recorded in this period.")
    
    print("\n" + "=" * 40)
    print("💡 Recommendations:")
    
    if emotions:
        dominant_emotion = emotions[0][0]
        if dominant_emotion in ["焦虑", "挫败", "谨慎"]:
            print("  - Consider slowing down and verifying more carefully")
        elif dominant_emotion in ["信心", "满意"]:
            print("  - Good momentum, but stay vigilant for errors")
        elif dominant_emotion in ["好奇"]:
            print("  - Explore new approaches, but stay focused")
    
    print("  - Record more outcomes to improve strategy recommendations")
    print("  - Review anti-patterns regularly to avoid repeating mistakes")


def main():
    parser = argparse.ArgumentParser(description="Fusion Engine — 感性层")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # init
    subparsers.add_parser("init", help="初始化数据库")
    
    # record-emotion
    emotion_parser = subparsers.add_parser("record-emotion", help="记录情绪状态")
    emotion_parser.add_argument("--emotion", required=True, help="情绪类型")
    emotion_parser.add_argument("--intensity", type=float, required=True, help="强度（0-1）")
    emotion_parser.add_argument("--trigger", default="", help="触发原因")
    emotion_parser.add_argument("--context", default="", help="上下文")
    
    # status
    subparsers.add_parser("status", help="查看状态")
    
    # report
    report_parser = subparsers.add_parser("report", help="生成进化报告")
    report_parser.add_argument("--days", type=int, default=7, help="报告天数")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_db()
    elif args.command == "record-emotion":
        record_emotion(args.emotion, args.intensity, args.trigger, args.context)
    elif args.command == "status":
        show_status()
    elif args.command == "report":
        generate_report(args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
