#!/usr/bin/env python3
"""
mssql_store.py - MSSQL storage backend for Neural Memory
Uses pyodbc with credentials from env vars or .env file.

Credentials resolution order:
  1. Environment variables: MSSQL_SERVER, MSSQL_DATABASE, MSSQL_USERNAME, MSSQL_PASSWORD
  2. .env file (~/.hermes/.env, CWD, plugin dir)
  3. Defaults (localhost, NeuralMemory, SA)
"""
import os
import struct
from pathlib import Path
from typing import Optional


def _load_dotenv(paths: list[str]) -> dict:
    env = {}
    for p in paths:
        path = Path(p).expanduser()
        if path.is_file():
            try:
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip("\"'")
                        if key and val:
                            env.setdefault(key, val)
            except Exception:
                pass
    return env


_dotenv = _load_dotenv([
    ".env",
    str(Path.home() / ".hermes" / ".env"),
    str(Path(__file__).parent / ".env"),
])


def _env(key: str, fallback: str = "") -> str:
    return os.environ.get(key) or _dotenv.get(key, fallback)

SCHEMA_SQL = """
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'NeuralMemory')
    CREATE DATABASE NeuralMemory;
GO

USE NeuralMemory;
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'memories')
CREATE TABLE memories (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    label NVARCHAR(256),
    content NVARCHAR(MAX),
    embedding VARBINARY(8000),
    vector_dim INT NOT NULL,
    salience FLOAT DEFAULT 1.0,
    created_at DATETIME2(7) DEFAULT SYSUTCDATETIME(),
    last_accessed DATETIME2(7) DEFAULT SYSUTCDATETIME(),
    access_count INT DEFAULT 0
);

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'connections')
CREATE TABLE connections (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    source_id BIGINT,
    target_id BIGINT,
    weight FLOAT DEFAULT 0.5,
    edge_type NVARCHAR(50) DEFAULT 'similar',
    created_at DATETIME2(7) DEFAULT SYSUTCDATETIME(),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_source')
CREATE INDEX idx_conn_source ON connections(source_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_target')
CREATE INDEX idx_conn_target ON connections(target_id);
"""


class MSSQLStore:
    """MSSQL-backed memory store.

    Credentials: env vars > .env > defaults.
    """

    def __init__(self, server='', database='', username='', password='',
                 driver=''):
        import pyodbc

        server = server or _env('MSSQL_SERVER', '127.0.0.1')
        if server == 'localhost':
            server = '127.0.0.1'  # MSSQL IPv4 only
        database = database or _env('MSSQL_DATABASE', 'NeuralMemory')
        username = username or _env('MSSQL_USERNAME', 'SA')
        password = password or _env('MSSQL_PASSWORD', '')
        driver = driver or _env('MSSQL_DRIVER', '{ODBC Driver 18 for SQL Server}')

        if not password:
            import logging
            logging.getLogger(__name__).warning(
                "MSSQL_PASSWORD not set — add it to ~/.hermes/.env"
            )

        self.conn_str = (
            f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
            f'TrustServerCertificate=yes;'
        )
        self.conn = pyodbc.connect(self.conn_str, autocommit=True)
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        # Check if memories table exists
        try:
            cursor.execute("SELECT COUNT(*) FROM memories")
        except Exception:
            # Table doesn't exist, create it
            for stmt in SCHEMA_SQL.split(';'):
                stmt = stmt.strip()
                if stmt and 'GO' not in stmt and 'CREATE DATABASE' not in stmt:
                    try:
                        cursor.execute(stmt)
                    except Exception as e:
                        pass  # Ignore if already exists
            self.conn.commit()
    
    def store(self, label: str, content: str, embedding: list[float]) -> int:
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memories (label, content, embedding, vector_dim) OUTPUT INSERTED.id VALUES (?, ?, ?, ?)",
            label, content, blob, len(embedding)
        )
        row = cursor.fetchone()
        self.conn.commit()
        return row[0] if row else 0
    
    def get_all(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, label, content, embedding, vector_dim, salience, access_count FROM memories ORDER BY id")
        results = []
        for row in cursor.fetchall():
            id_, label, content, blob, dim, salience, access = row
            embedding = list(struct.unpack(f'{dim}f', blob)) if blob else []
            results.append({
                'id': id_, 'label': label, 'content': content,
                'embedding': embedding, 'salience': salience, 'access_count': access
            })
        return results
    
    def get(self, id_: int) -> Optional[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, label, content, embedding, vector_dim, salience, access_count FROM memories WHERE id = ?", id_)
        row = cursor.fetchone()
        if not row:
            return None
        id_, label, content, blob, dim, salience, access = row
        embedding = list(struct.unpack(f'{dim}f', blob)) if blob else []
        return {'id': id_, 'label': label, 'content': content, 'embedding': embedding,
                'salience': salience, 'access_count': access}
    
    def touch(self, id_: int):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET last_accessed = SYSUTCDATETIME(), access_count = access_count + 1 WHERE id = ?",
            id_
        )
        self.conn.commit()
    
    def add_connection(self, source: int, target: int, weight: float, edge_type: str = "similar"):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO connections (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
            source, target, weight, edge_type
        )
        self.conn.commit()
    
    def get_connections(self, node_id: int) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT source_id, target_id, weight, edge_type FROM connections WHERE source_id = ? OR target_id = ? ORDER BY weight DESC",
            node_id, node_id
        )
        return [{'source': r[0], 'target': r[1], 'weight': r[2], 'type': r[3]} for r in cursor.fetchall()]
    
    def stats(self) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        mc = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM connections")
        cc = cursor.fetchone()[0]
        return {'memories': mc, 'connections': cc}
    
    def close(self):
        self.conn.close()


# Quick test
if __name__ == "__main__":
    try:
        store = MSSQLStore()
        mid = store.store("test", "Hello MSSQL", [0.1] * 384)
        print(f"Stored: {mid}")
        m = store.get(mid)
        print(f"Retrieved: {m['label']}")
        s = store.stats()
        print(f"Stats: {s}")
        store.close()
        print("MSSQL: OK")
    except Exception as e:
        print(f"MSSQL error: {e}")
