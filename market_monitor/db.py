from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Sequence

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  homepage_url TEXT,
  source_level TEXT,
  update_frequency TEXT,
  access_mode TEXT,
  active INTEGER DEFAULT 1,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  category TEXT NOT NULL,
  metric_scope TEXT NOT NULL,
  entity_granularity TEXT NOT NULL,
  time_granularity TEXT NOT NULL,
  notes TEXT,
  FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS raw_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  dataset_id TEXT,
  fetch_time TEXT NOT NULL,
  source_url TEXT,
  period_hint TEXT,
  content_type TEXT,
  content_hash TEXT NOT NULL,
  local_path TEXT NOT NULL,
  parse_status TEXT DEFAULT 'pending',
  parser_version TEXT,
  notes TEXT,
  FOREIGN KEY (source_id) REFERENCES sources(source_id),
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_raw_snapshots_source_hash ON raw_snapshots(source_id, content_hash);

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY,
  entity_type TEXT NOT NULL,
  name_raw TEXT,
  name_norm TEXT NOT NULL,
  parent_entity_id TEXT,
  notes TEXT,
  FOREIGN KEY (parent_entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE IF NOT EXISTS alias_mappings (
  alias_id TEXT PRIMARY KEY,
  entity_type TEXT NOT NULL,
  alias_raw TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  source_id TEXT,
  confidence REAL DEFAULT 1.0,
  active INTEGER DEFAULT 1,
  FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
  FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS observations (
  obs_id TEXT PRIMARY KEY,
  observation_key TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  snapshot_id TEXT,
  source_id TEXT NOT NULL,
  period_label TEXT NOT NULL,
  period_type TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_scope TEXT NOT NULL,
  metric_type TEXT NOT NULL,
  energy_type TEXT,
  value_numeric REAL,
  value_text TEXT,
  unit TEXT,
  ranking INTEGER,
  published_at TEXT,
  source_url TEXT,
  is_latest INTEGER DEFAULT 1,
  revision_no INTEGER DEFAULT 1,
  notes TEXT,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
  FOREIGN KEY (snapshot_id) REFERENCES raw_snapshots(snapshot_id),
  FOREIGN KEY (source_id) REFERENCES sources(source_id)
);
CREATE INDEX IF NOT EXISTS idx_observations_key_latest ON observations(observation_key, is_latest, revision_no);

CREATE TABLE IF NOT EXISTS observation_entities (
  obs_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  entity_role TEXT NOT NULL,
  PRIMARY KEY (obs_id, entity_id, entity_role),
  FOREIGN KEY (obs_id) REFERENCES observations(obs_id) ON DELETE CASCADE,
  FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE IF NOT EXISTS parse_warnings (
  warning_id TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  dataset_id TEXT,
  warning_text TEXT NOT NULL,
  FOREIGN KEY (snapshot_id) REFERENCES raw_snapshots(snapshot_id),
  FOREIGN KEY (source_id) REFERENCES sources(source_id),
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE TABLE IF NOT EXISTS ingestion_job_log (
  job_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  dataset_id TEXT,
  run_at TEXT NOT NULL,
  status TEXT NOT NULL,
  rows_extracted INTEGER DEFAULT 0,
  snapshots_created INTEGER DEFAULT 0,
  error_message TEXT,
  duration_seconds REAL,
  notes TEXT,
  FOREIGN KEY (source_id) REFERENCES sources(source_id),
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);
"""

DEFAULT_SOURCES = [
    ("cpca", "乘联会"),
    ("caam", "中汽协"),
    ("cada", "中国汽车流通协会"),
    ("dongchedi", "懂车帝"),
    ("evcipa", "中国充电联盟"),
]

DEFAULT_DATASETS = [
    ("cpca_monthly_market", "cpca", "CPCA 月度市场总览", "sales", "retail", "market_total", "monthly"),
    ("cpca_brand_rank", "cpca", "CPCA 品牌排名", "ranking", "wholesale", "brand", "monthly"),
    ("caam_nev_prod_sales", "caam", "CAAM 新能源汽车产销", "sales", "production", "market_total", "monthly"),
    ("cada_nev_report_meta", "cada", "CADA 新能源报告元信息", "meta", "retail", "market_total", "monthly"),
    ("dongchedi_model_rank", "dongchedi", "懂车帝车型销量榜", "ranking", "retail", "model", "monthly"),
    ("evcipa_monthly_infra", "evcipa", "EVCIPA 月度充电基础设施", "charging", "charging_infrastructure", "market_total", "monthly"),
]


def initialize_database(db_path: Path | str) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            "INSERT OR IGNORE INTO sources (source_id, source_name) VALUES (?, ?)",
            DEFAULT_SOURCES,
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO datasets (
                dataset_id, source_id, dataset_name, category, metric_scope, entity_granularity, time_granularity
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            DEFAULT_DATASETS,
        )
        conn.commit()
    finally:
        conn.close()


class Database:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self.transaction() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, params_list: Iterable[Sequence[Any]]) -> None:
        with self.transaction() as conn:
            conn.executemany(sql, list(params_list))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        conn = self.connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        conn = self.connect()
        try:
            row = conn.execute(sql, params).fetchone()
            return None if row is None else row[0]
        finally:
            conn.close()

    def ensure_source_dataset(self, source_id: str, dataset_id: str | None, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO sources (source_id, source_name) VALUES (?, ?)",
            (source_id, source_id),
        )
        if dataset_id:
            conn.execute(
                """
                INSERT OR IGNORE INTO datasets (
                    dataset_id, source_id, dataset_name, category, metric_scope, entity_granularity, time_granularity
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (dataset_id, source_id, dataset_id, "meta", "retail", "market_total", "monthly"),
            )

    def get_existing_snapshot(self, source_id: str, content_hash: str) -> sqlite3.Row | None:
        rows = self.query(
            "SELECT snapshot_id, local_path FROM raw_snapshots WHERE source_id = ? AND content_hash = ? LIMIT 1",
            (source_id, content_hash),
        )
        return rows[0] if rows else None

    def insert_job_log(
        self,
        *,
        job_id: str,
        source_id: str,
        dataset_id: str | None,
        run_at: str,
        status: str,
        rows_extracted: int,
        snapshots_created: int,
        error_message: str | None = None,
        duration_seconds: float | None = None,
        notes: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            self.ensure_source_dataset(source_id, dataset_id, conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_job_log (
                    job_id, source_id, dataset_id, run_at, status,
                    rows_extracted, snapshots_created, error_message, duration_seconds, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, source_id, dataset_id, run_at, status, rows_extracted, snapshots_created, error_message, duration_seconds, notes),
            )

    def persist_ingestion(
        self,
        *,
        snapshot,
        entities,
        observations,
        observation_entities,
        warnings: list[str],
        snapshots_created: int,
        job_id: str,
        run_at: str,
    ) -> dict[str, int | str]:
        with self.transaction() as conn:
            self.ensure_source_dataset(snapshot.source_id, snapshot.dataset_id, conn)
            if snapshots_created:
                conn.execute(
                    """
                    INSERT INTO raw_snapshots (
                        snapshot_id, source_id, dataset_id, fetch_time, source_url, period_hint,
                        content_type, content_hash, local_path, parse_status, parser_version, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.source_id,
                        snapshot.dataset_id,
                        snapshot.fetch_time,
                        snapshot.source_url,
                        snapshot.period_hint,
                        snapshot.content_type,
                        snapshot.content_hash,
                        snapshot.local_path,
                        snapshot.parse_status,
                        snapshot.parser_version,
                        snapshot.notes,
                    ),
                )
            for entity in entities:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entities (entity_id, entity_type, name_raw, name_norm, parent_entity_id, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (entity.entity_id, entity.entity_type, entity.name_raw, entity.name_norm, entity.parent_entity_id, entity.notes),
                )
            persisted_obs_ids: dict[str, str] = {}
            new_revision_count = 0
            duplicate_count = 0
            for obs in observations:
                self.ensure_source_dataset(obs.source_id, obs.dataset_id, conn)
                stored_obs_id, action = self._persist_observation(conn, obs)
                persisted_obs_ids[obs.obs_id] = stored_obs_id
                persisted_obs_ids[obs.observation_key or obs.obs_id] = stored_obs_id
                if action == "inserted":
                    new_revision_count += 1
                else:
                    duplicate_count += 1
            for link in observation_entities:
                conn.execute(
                    "INSERT OR IGNORE INTO observation_entities (obs_id, entity_id, entity_role) VALUES (?, ?, ?)",
                    (persisted_obs_ids.get(link.obs_id, link.obs_id), link.entity_id, link.entity_role),
                )
            for idx, warning_text in enumerate(warnings, start=1):
                conn.execute(
                    "INSERT OR REPLACE INTO parse_warnings (warning_id, snapshot_id, source_id, dataset_id, warning_text) VALUES (?, ?, ?, ?, ?)",
                    (f"{snapshot.snapshot_id}:w{idx}", snapshot.snapshot_id, snapshot.source_id, snapshot.dataset_id, warning_text),
                )
            persisted_status = "persisted_with_warnings" if warnings else "success"
            if snapshots_created == 0 and new_revision_count == 0 and not warnings:
                persisted_status = "skipped"
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_job_log (
                    job_id, source_id, dataset_id, run_at, status,
                    rows_extracted, snapshots_created, error_message, duration_seconds, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    snapshot.source_id,
                    snapshot.dataset_id,
                    run_at,
                    persisted_status,
                    len(observations),
                    snapshots_created,
                    None,
                    None,
                    None,
                ),
            )
            return {
                "status": persisted_status,
                "snapshots_created": snapshots_created,
                "new_revision_count": new_revision_count,
                "duplicate_count": duplicate_count,
                "rows_extracted": len(observations),
            }

    def _persist_observation(self, conn: sqlite3.Connection, obs) -> tuple[str, str]:
        observation_key = obs.observation_key or obs.obs_id
        existing = conn.execute(
            "SELECT obs_id, revision_no, value_numeric, value_text, unit FROM observations WHERE observation_key = ? AND is_latest = 1 LIMIT 1",
            (observation_key,),
        ).fetchone()
        if existing is not None and existing["value_numeric"] == obs.value_numeric and existing["value_text"] == obs.value_text and existing["unit"] == obs.unit:
            return existing["obs_id"], "duplicate"
        revision_no = 1
        stored_obs_id = obs.obs_id
        if existing is not None:
            conn.execute("UPDATE observations SET is_latest = 0 WHERE obs_id = ?", (existing["obs_id"],))
            revision_no = int(existing["revision_no"]) + 1
            stored_obs_id = f"{observation_key}::r{revision_no}"
        conn.execute(
            """
            INSERT INTO observations (
                obs_id, observation_key, dataset_id, snapshot_id, source_id, period_label, period_type,
                metric_name, metric_scope, metric_type, energy_type, value_numeric, value_text,
                unit, ranking, published_at, source_url, is_latest, revision_no, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stored_obs_id,
                observation_key,
                obs.dataset_id,
                obs.snapshot_id,
                obs.source_id,
                obs.period_label,
                obs.period_type,
                obs.metric_name,
                obs.metric_scope,
                obs.metric_type,
                obs.energy_type,
                obs.value_numeric,
                obs.value_text,
                obs.unit,
                obs.ranking,
                obs.published_at,
                obs.source_url,
                1,
                revision_no,
                obs.notes,
            ),
        )
        return stored_obs_id, "inserted"
