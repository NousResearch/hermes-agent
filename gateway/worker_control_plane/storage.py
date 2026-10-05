"""Dedicated SQLite persistence for the test-only control plane."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager

from .config import WorkerControlPlaneSettings, resolve_test_database_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS workers(worker_id TEXT PRIMARY KEY, worker_name TEXT NOT NULL, allowed_capabilities TEXT NOT NULL, enabled INTEGER NOT NULL, revoked_at TEXT);
CREATE TABLE IF NOT EXISTS worker_credentials(credential_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL REFERENCES workers(worker_id), kind TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE, salt TEXT, issued_at TEXT NOT NULL, expires_at TEXT, revoked_at TEXT);
CREATE TABLE IF NOT EXISTS worker_instances(registration_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL REFERENCES workers(worker_id), instance_id TEXT NOT NULL, status TEXT NOT NULL, worker_version TEXT NOT NULL, protocol_version TEXT NOT NULL, registered_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, access_credential_id TEXT NOT NULL REFERENCES worker_credentials(credential_id), current_task_id TEXT, UNIQUE(worker_id, instance_id));
CREATE TABLE IF NOT EXISTS worker_tasks(task_id TEXT PRIMARY KEY, task_type TEXT NOT NULL CHECK(task_type='system.echo'), payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL, state TEXT NOT NULL, created_at TEXT NOT NULL, available_at TEXT NOT NULL, leased_until TEXT, attempt INTEGER NOT NULL, max_attempts INTEGER NOT NULL, creation_idempotency_key TEXT NOT NULL UNIQUE, trace_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS worker_deliveries(delivery_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES worker_tasks(task_id), worker_id TEXT NOT NULL, registration_id TEXT NOT NULL REFERENCES worker_instances(registration_id), attempt INTEGER NOT NULL, state TEXT NOT NULL, leased_at TEXT NOT NULL, ack_deadline_at TEXT NOT NULL, lease_expires_at TEXT NOT NULL, acknowledged_at TEXT, finished_at TEXT, UNIQUE(task_id, attempt));
CREATE TABLE IF NOT EXISTS worker_results(result_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES worker_tasks(task_id), delivery_id TEXT NOT NULL UNIQUE REFERENCES worker_deliveries(delivery_id), result_idempotency_key TEXT NOT NULL, result_hash TEXT NOT NULL, status TEXT NOT NULL, stdout TEXT NOT NULL, stderr TEXT NOT NULL, exit_code INTEGER, started_at TEXT NOT NULL, finished_at TEXT NOT NULL, duration_ms INTEGER NOT NULL, accepted_at TEXT NOT NULL, UNIQUE(task_id, result_idempotency_key));
CREATE TABLE IF NOT EXISTS worker_request_dedup(worker_id TEXT NOT NULL, registration_id TEXT NOT NULL, method TEXT NOT NULL, route TEXT NOT NULL, task_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_body_hash TEXT NOT NULL, response_status INTEGER NOT NULL, response_json TEXT NOT NULL, PRIMARY KEY(worker_id, idempotency_key));
CREATE TABLE IF NOT EXISTS worker_audit_log(audit_id INTEGER PRIMARY KEY, occurred_at TEXT NOT NULL, event_type TEXT NOT NULL, worker_id TEXT, instance_id TEXT, registration_id TEXT, task_id TEXT, delivery_id TEXT, trace_id TEXT, outcome TEXT NOT NULL, reason_code TEXT, details_json TEXT);
CREATE INDEX IF NOT EXISTS idx_tasks_queue ON worker_tasks(state,available_at,created_at);
CREATE INDEX IF NOT EXISTS idx_deliveries_lease ON worker_deliveries(state,lease_expires_at);
"""
class WorkerControlPlaneStore:
 def __init__(self, settings: WorkerControlPlaneSettings):
  if not isinstance(settings, WorkerControlPlaneSettings):
   raise TypeError("WorkerControlPlaneStore requires validated settings")
  if not settings.enabled or not settings.test_mode:
   raise ValueError("Worker Control Plane storage requires enabled test mode")
  root,path=resolve_test_database_path(settings.approved_test_root,settings.db_path)
  if root != settings.approved_test_root:
   raise ValueError("approved test root changed after settings validation")
  self.conn=sqlite3.connect(path, check_same_thread=False); self.conn.row_factory=sqlite3.Row
  self.conn.execute("PRAGMA foreign_keys=ON"); self.conn.execute("PRAGMA synchronous=FULL"); self.conn.execute("PRAGMA secure_delete=ON"); self.conn.execute("PRAGMA cell_size_check=ON")
  try: self.conn.execute("PRAGMA journal_mode=WAL")
  except sqlite3.DatabaseError: self.conn.execute("PRAGMA journal_mode=DELETE")
  self.conn.executescript(SCHEMA); self.conn.execute("INSERT OR IGNORE INTO schema_migrations VALUES('worker_control_plane_schema_v1',datetime('now'))"); self.conn.commit()
 @contextmanager
 def transaction(self):
  try:
   self.conn.execute("BEGIN IMMEDIATE"); yield self.conn; self.conn.commit()
  except Exception: self.conn.rollback(); raise
 def close(self): self.conn.close()
