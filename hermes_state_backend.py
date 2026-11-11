"""Resolve a profile's session database without changing process configuration."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from hermes_constants import get_hermes_home, get_process_hermes_home, profile_name_for_home


@dataclass(frozen=True)
class DatabaseSettings:
    backend: str = "sqlite"
    database_url: str | None = field(default=None, repr=False)
    schema: str | None = None


def resolve_database_settings(db_path: str | Path | None = None) -> DatabaseSettings:
    """Resolve the store owning ``db_path``; in-memory databases always use SQLite.

    Each profile owns its config and credentials. Process credentials apply only to the
    process home, never to another profile opened by the same gateway.
    """
    if str(db_path) == ":memory:":
        return DatabaseSettings()
    path = (Path(db_path) if db_path is not None else get_hermes_home() / "state.db")
    path = path.expanduser().resolve()
    home = path.parent

    from hermes_cli.config_effective import load_user_config_effective

    # Fail open: a config.yaml torn mid-edit serves the last-good user layer (this
    # process's last successful parse first, then the newest backups/config copy), so a
    # torn write can never drop a postgres selection or abort a reader surface on the
    # default backend; with no last-good layer at all it falls to the SQLite default.
    # Readable-but-invalid configs (non-mapping database section, unknown backend,
    # missing DSN, invalid schema) keep raising below.
    config = load_user_config_effective(home / "config.yaml")
    database = config.get("database", {})
    if not isinstance(database, dict):
        raise ValueError("database must be a mapping in config.yaml")
    backend = database.get("backend", "sqlite")
    if backend not in ("sqlite", "postgres"):
        raise ValueError("database.backend must be sqlite or postgres")
    if backend == "sqlite":
        return DatabaseSettings()

    from dotenv import dotenv_values

    env_path = home / ".env"
    try:
        with env_path.open(encoding="utf-8") as stream:
            database_url = dotenv_values(stream=stream, interpolate=False).get("HERMES_DATABASE_URL")
    except FileNotFoundError:
        database_url = None
    if home == get_process_hermes_home().expanduser().resolve():
        database_url = os.environ.get("HERMES_DATABASE_URL", database_url)  # health: allow HX002 -- per-profile credential (masked DSN in the profile's .env), not a behavioural knob
    if not database_url or not database_url.strip():
        raise ValueError(f"PostgreSQL requires HERMES_DATABASE_URL in {env_path}")

    schema = database.get("schema")
    if schema is None:
        profile_name = profile_name_for_home(home) or "default"
        profile_name = re.sub(r"[^a-z0-9_]", "_", profile_name.lower())[:35]
        path_hash = hashlib.sha256(os.fsencode(os.path.normcase(str(path)))).hexdigest()[:12]
        schema = f"hermes_{profile_name}_{path_hash}"
    if not isinstance(schema, str) or not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema):
        raise ValueError("database.schema must be a lowercase PostgreSQL identifier of 1–63 characters")
    if schema.startswith("pg_"):
        raise ValueError("database.schema must not use PostgreSQL's reserved pg_ prefix")
    return DatabaseSettings(backend=backend, database_url=database_url.strip(), schema=schema)
