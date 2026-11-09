"""``hermes_state`` keeps re-exporting the storage helpers main published through it.

``hermes_cli/data/plugin_api.json`` lists ``hermes_state`` as a whole module, so every public name it had is plugin
API. Each helper below lives in a sibling module; ``hermes_state`` must hand out that same object.
"""

import importlib

import pytest

import hermes_state

HELPERS = {
    "hermes_constants": ("mkdir_under_hermes_home",),
    "hermes_state_dbfile": (
        "RetiredGenerationCaptureError", "capture_retired_wal_generation", "has_invalid_sqlite_header_preopen",
        "is_zeroed_state_db", "quarantine_cross_process_lock", "quarantine_invalid_state_db",
        "refuse_deleted_wal_generation",
    ),
    "hermes_state_errors": ("is_malformed_schema_error", "is_sqlite_lock_error"),
    "hermes_state_health": (
        "STORAGE_CORRUPT", "mark_storage_corrupt", "note_storage_error", "storage_corrupt_reason", "storage_state",
    ),
    "hermes_state_holders": ("read_only_db_uri",),
    "hermes_state_lockowners": ("log_write_lock_holders",),
    "hermes_state_repair": ("repair_state_db_schema",),
    "hermes_state_wal": ("apply_database_pragmas", "apply_wal_with_fallback"),
}


@pytest.mark.parametrize(
    ("module", "name"), [(module, name) for module, names in HELPERS.items() for name in names],
)
def test_hermes_state_re_exports_the_storage_helper(module, name):
    assert getattr(hermes_state, name) is getattr(importlib.import_module(module), name)
