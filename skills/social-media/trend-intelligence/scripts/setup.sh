#!/bin/bash
# Trend Intelligence Engine — Setup
# By ENERGENAI LLC (https://tiamat.live)

set -e

echo "=== Trend Intelligence Engine Setup ==="

# Check Python
python3 --version || { echo "ERROR: Python 3 required"; exit 1; }

# Install dependencies
pip install aiohttp beautifulsoup4 2>/dev/null || pip3 install aiohttp beautifulsoup4

# Optional: Bluesky firehose client
pip install atproto 2>/dev/null || echo "NOTE: atproto not installed (optional — for Bluesky firehose)"

# Initialize database
python3 -c "
import sqlite3, os
db_path = os.path.expanduser('~/.hermes/trend-history.db')
os.makedirs(os.path.dirname(db_path), exist_ok=True)
conn = sqlite3.connect(db_path)
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('''CREATE TABLE IF NOT EXISTS trend_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    source TEXT NOT NULL,
    volume REAL DEFAULT 0,
    velocity_24h REAL DEFAULT 0,
    velocity_3d REAL DEFAULT 0,
    velocity_7d REAL DEFAULT 0,
    phase TEXT DEFAULT \"dormant\",
    confidence REAL DEFAULT 0,
    raw_data TEXT,
    scanned_at TEXT NOT NULL DEFAULT (strftime(\"%Y-%m-%dT%H:%M:%SZ\", \"now\"))
)''')
conn.execute('CREATE INDEX IF NOT EXISTS idx_trend_topic_time ON trend_scans(topic, scanned_at)')
conn.commit()
conn.close()
print('Database initialized at', db_path)
"

echo ""
echo "Setup complete! Run: python3 scripts/trend_engine.py scan"
echo ""
