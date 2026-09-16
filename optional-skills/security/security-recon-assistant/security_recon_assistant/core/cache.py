import sqlite3, json, hashlib, time, os
from pathlib import Path

class Cache:
    def __init__(self, db_path="cache.db"):
        self.conn = sqlite3.connect(db_path)
        self._init()
    def _init(self):
        self.conn.execute("CREATE TABLE IF NOT EXISTS results (target TEXT, scanner TEXT, result TEXT, timestamp REAL)")
        self.conn.commit()
    def get(self, target, scanner):
        cur = self.conn.execute("SELECT result FROM results WHERE target=? AND scanner=?", (target, scanner))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None
    def set(self, target, scanner, result):
        self.conn.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?)", (target, scanner, json.dumps(result), time.time()))
        self.conn.commit()
    def audit_log(self, message: str, logfile="audit.log"):
        with open(logfile, "a") as f:
            f.write(f"{time.ctime()} | {message}\n")
