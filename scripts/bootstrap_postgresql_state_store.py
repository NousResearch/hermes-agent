#!/usr/bin/env python3
"""User-invoked PostgreSQL state-store Compose bootstrap.

Normally invoked by choosing the local PostgreSQL option in ``hermes setup
state-store`` or ``hermes config state-store``. It intentionally does nothing
until that option is selected.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.config import get_env_value, get_hermes_home, save_env_value
from hermes_cli.postgresql_bootstrap import bootstrap_postgresql_state_store
from hermes_cli.setup_state_store import POSTGRES_DSN_ENV


def main() -> int:
    dsn = bootstrap_postgresql_state_store(
        get_hermes_home(), existing_dsn=get_env_value(POSTGRES_DSN_ENV)
    )
    save_env_value(POSTGRES_DSN_ENV, dsn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
