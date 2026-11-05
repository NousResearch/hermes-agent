"""persistence.sqlite._schema and _schema_ddl stand alone: the DDL carries the table text the updater and gateway downgrade guards look for, and importing both modules loads none of the ``hermes_state`` store or mixin modules.
"""

import subprocess
import sys
from pathlib import Path

from persistence.sqlite import _schema, _schema_ddl

DDL_SOURCE = Path(_schema_ddl.__file__).read_text(encoding="utf-8")


def test_downgrade_guard_marker_present():
    # Main's updater and gateway downgrade guards look for this text in the schema source.
    assert "CREATE TABLE IF NOT EXISTS session_admissions" in DDL_SOURCE


def test_imports_no_legacy_store_or_mixin_module():
    probe = (
        "import sys, persistence.sqlite._schema, persistence.sqlite._schema_ddl\n"
        "mixins = ('hermes_state', 'hermes_state_schema', 'hermes_state_fts', 'hermes_state_search',"
        " 'hermes_state_titles', 'hermes_state_gateway', 'hermes_state_sessions', 'hermes_state_messages')\n"
        "print(sorted(m for m in mixins if m in sys.modules))\n"
    )
    root = Path(_schema.__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", probe], cwd=root, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
