"""persistence.sqlite._schema and _schema_ddl carry main's state.db DDL, migration ladder and FTS setup unchanged.

Every copied constant equals its legacy counterpart, and every copied function has the legacy source text
after the listed line substitutions (the one lazy ``hermes_state`` import, the store's own folder for the files a
store reads beside it, and the column reconciler that reads no store) and the conversation copies'
``receiver_form``, which puts a legacy class member at module level with the store holder as its receiver,
apart from the allow comments that waive main's own code-health findings on unchanged lines, so any drift between
the two fails here.
"""

import ast
import functools
import importlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from persistence.sqlite import _schema, _schema_ddl
import hermes_state_common as common
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conversation._copies import Catalog, core_members, receiver_form
from tests.persistence_sqlite.core._moved import strip_moved_allows

__all__ = ["legacy_store"]

SCHEMA_SOURCE = Path(_schema.__file__).read_text(encoding="utf-8")
DDL_SOURCE = Path(_schema_ddl.__file__).read_text(encoding="utf-8")


def _top_level_assigns(source: str) -> set[str]:
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _functions_of(cls) -> set[str]:
    return {name for name, value in vars(cls).items()
            if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod))}


# Module-level constants: legacy module -> names copied into _schema.
CONSTANTS = {
    "hermes_state_common": (
        "_RESET_END_REASONS", "FTS_REBUILD_DEFERRAL_KEY",
        "_BRANCH_CHILD_SQL", "_COMPRESSION_CHILD_SQL", "_RESET_END_REASONS_SQL", "_RESET_CHILD_SQL",
        "SCHEMA_VERSION", "FTS_STORAGE_VERSION", "FTS_TOOL_CONTENT_PREFIX_CHARS", "_FTS_NEW_INDEXED_CONTENT_SQL",
        "_FTS_OLD_INDEXED_CONTENT_SQL", "_FTS_TRIGGERS", "SCHEMA_SQL", "DEFERRED_INDEX_SQL", "FTS_SQL",
        "FTS_TRIGRAM_EXCLUDED_SOURCES", "FTS_TRIGRAM_SESSION_SQL", "FTS_TRIGRAM_SQL", "_FTS_CJK_TRIGGERS",
        "FTS_CJK_STALE_KEY", "FTS_STALE_KEY", "LEGACY_FTS_SQL", "LEGACY_FTS_TRIGRAM_SQL",
    ),
    "hermes_state_fts": (
        "FTS_CJK_TABLE_SQL", "FTS_CJK_TRIGGER_SQL", "_FTS_DDL_TABLES", "FTS_VTABLE_LAYOUTS", "_FTS5_SHADOW_SUFFIXES",
    ),
    "hermes_state_titles": ("_NUMBERED_TITLE_RE",),
    # Every constant the schema module declares; its probe cache is compared through its function below.
    "hermes_state_schema": (
        "_CLEAR_REBUILD_MARKERS_SQL", "_DROP_RETIRED_TOOL_HIGH_WATER_SQL", "_FTS_BASE_TRIGGERS", "_FTS_DDL",
        "_FTS_HOLDER_ESCALATE_ATTEMPTS", "_FTS_HOLDER_ESCALATE_SECONDS", "_FTS_HOLDER_FUTILE_ATTEMPTS",
        "_FTS_HOLDER_FUTILE_SECONDS", "_FTS_STALE_RETRY_MAX_SECONDS", "_FTS_STALE_RETRY_SECONDS",
        "_FTS_TRIGRAM_TRIGGERS", "_LEGACY_INLINE_CONCAT_SQL", "_MESSAGE_UID_BACKFILL_BUDGET_S",
        "_MESSAGE_UID_BACKFILL_CHUNK", "_MESSAGE_UID_BACKFILL_CURSOR", "_MESSAGE_UID_BACKFILL_DONE",
        "_SESSION_MODEL_USAGE_HEAL_DDL", "_SESSION_MODEL_USAGE_INDEX_SQL", "_SESSION_MODEL_USAGE_V20_SEED_SQL",
        "_SESSION_MODEL_USAGE_V22_DDL", "_STALE_KEY_UPSERT_SQL", "_STATE_META_UPSERT_SQL", "_TITLE_UNIQUE_INDEX_SQL",
    ),
}

FUNCTIONS = {
    "hermes_state_common": (
        "_sql_literal", "_sql_json_extract", "_legacy_reset_child_sql", "_ephemeral_child_sql",
        "_fts_indexed_content_sql", "fts_trigram_session_sql",
    ),
    "hermes_state_search": ("_meta_row", "_delete_meta", "_positive_int"),
    "hermes_state_titles": ("next_title_in_lineage",),
    "hermes_state_fts": (
        "fts5_cjk_so_path", "_cjk_fts_config_enabled", "load_fts5_cjk_extension", "_stored_form",
        "_expected_stored_sql", "_expected_vtable_sql", "_stored_vtables", "_is_older_layout",
        "base_index_recognised", "base_index_realigned_on_open", "fts_layout_drift", "fts_layout_current",
        "base_source_view_is_stale", "_drop_orphan_fts_shadow_tables",
    ),
    "hermes_state_schema": (
        "_holder_cmdline", "_legacy_inline_reinsert_sql", "_q", "apply_declared_schema", "reconcile_state_schema",
        "schema_read_probe_statements",
    ),
}

METHODS = {
    "hermes_state_schema.SessionSchemaMixin": (
        "_advance_message_uid_backfill", "_backfill_gateway_metadata_from_sessions_json",
        "_dedupe_legacy_system_prompts", "_defer_stale_fts_for_holders", "_drop_all_fts_triggers",
        "_ensure_unique_title_index", "_execute_ddl_script_transactional", "_execute_ddl_skipping_settled_triggers",
        "_fts_index_is_misaligned_source", "_fts_table_probe", "_fts_triggers_missing",
        "_fts_update_trigger_needs_narrowing", "_heal_gateway_routing_pk", "_heal_session_model_usage_pk", "_init_fts",
        "_init_schema", "_live_pk_columns", "_migrate_broad_fts_update_triggers", "_migrate_misaligned_fts_source",
        "_migrate_trigram_cron_exclusion", "_migrate_v22_session_model_usage", "_parse_schema_columns",
        "_quarantine_cjk_after_update_of_migration", "_rebuild_fts_indexes", "_rebuild_table",
        "_rebuild_table_statements", "_reconcile_columns", "_recover_stale_fts", "_recover_stale_fts_locked",
        "_run_admitted_startup_rebuild", "_run_data_migrations", "_sqlite_supports_fts5", "_sqlite_table_exists",
        "_stamp_settled_fts_layout", "_trigram_tokenizer_available", "retry_deferred_fts_recovery",
    ),
    "hermes_state_fts.SessionFtsSetupMixin": (
        "_is_fts5_unavailable_error", "_is_trigram_unavailable_error", "_db_has_legacy_inline_fts",
        "_db_has_trigram_tool_calls_projection", "_db_needs_fts_storage_upgrade", "_warn_trigram_unavailable",
        "_warn_fts5_unavailable", "_ensure_fts_cjk_schema", "_drop_fts_triggers", "_ensure_fts_schema",
        "_is_fts_write_corruption_error", "_enter_fts_fail_open", "_has_fts_trash",
    ),
    "hermes_state_search.SessionSearchMixin": (
        "_try_incremental_merge_fts", "_fts_external_index_empty_with_messages", "_fts_table_exists",
        "_present_fts_tables", "_merge_fts_incrementally",
    ),
    "hermes_state_titles.SessionTitlesMixin": ("_title_rank",),
    "hermes_state_gateway.SessionGatewayMixin": ("_reap_inactive_orphan_desktop_holders",),
}

CLASS_ATTRIBUTES = {
    "hermes_state.SessionDB": ("TITLE_SOURCE_DERIVED", "TITLE_SOURCE_LLM", "TITLE_SOURCE_USER", "_TITLE_SOURCE_RANK"),
    "hermes_state_fts.SessionFtsSetupMixin": ("_FTS_TRASH_PREFIX", "_FTS_TABLES"),
}

# The only lines that differ from main before ``receiver_form``, per copied function: the lazy
# ``from hermes_state import psutil`` (the module binds psutil itself), the store's own folder in place of
# the ambient home for the files a store reads beside it, and the column reconciler, which reads nothing from
# a store and so takes no holder. Stripped text, old -> new.
SUBSTITUTIONS = {
    # The column cache lives in the folder of the store being read, never the ambient home.
    "_parse_schema_columns": {
        "def _parse_schema_columns(schema_sql: str) -> dict[str, dict[str, str]]:":
            "def _parse_schema_columns(schema_sql: str, home: Path) -> dict[str, dict[str, str]]:",
        "PRAGMA table_info (no regex). Memoized on disk keyed by a DDL hash (~85ms per":
            "PRAGMA table_info (no regex). Memoized under ``home``, the store's folder, keyed by a DDL hash (~85ms per",
        "# Late import: resolves a test-patched hermes_constants.get_hermes_home.": None,
        "from hermes_constants import get_hermes_home as _home": None,
        'cache_path = _home() / "cache" / "schema_columns.json"': 'cache_path = home / "cache" / "schema_columns.json"',
    },
    # Passes the store's folder through to the column cache.
    "_reconcile_columns": {
        "def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:":
            "def _reconcile_columns(cursor: sqlite3.Cursor, home: Path) -> None:",
        "expected = self._parse_schema_columns(SCHEMA_SQL)": "expected = self._parse_schema_columns(SCHEMA_SQL, home)",
    },
    # A store's open caches the column parse in the store's own folder.
    "_init_schema": {
        "self._reconcile_columns(cursor)": "self._reconcile_columns(cursor, self.db_path.parent)",
    },
    # The v18 backfill reads the sessions.json beside the store it migrates.
    "_backfill_gateway_metadata_from_sessions_json": {
        'sessions_file = get_hermes_home() / "sessions" / "sessions.json"':
            'sessions_file = self.db_path.parent / "sessions" / "sessions.json"',
    },
    # The bare instance the column reconciler ran on is gone with the holder it stood in for.
    "apply_declared_schema": {
        "callers. ``_reconcile_columns`` only touches the staticmethod ``_parse_schema_columns``, so a bare":
            "callers. ``_reconcile_columns`` reads nothing from a store, so the column reconciler keeps one",
        'instance is enough and the column reconciler keeps one implementation."""': 'implementation."""',
        "shim = object.__new__(SessionSchemaMixin)": None,
        "def apply_declared_schema(cursor: sqlite3.Cursor) -> None:":
            "def apply_declared_schema(cursor: sqlite3.Cursor, home: Path) -> None:",
        "shim._reconcile_columns(cursor)": "_reconcile_columns(cursor, home)",
    },
    # The schema probes, the raw-connection reconciler and the CJK tokenizer take the store's folder.
    "schema_read_probe_statements": {
        "def schema_read_probe_statements() -> tuple:": "def schema_read_probe_statements(home: Path) -> tuple:",
        "tables = SessionSchemaMixin._parse_schema_columns(SCHEMA_SQL)":
            "tables = SessionSchemaMixin._parse_schema_columns(SCHEMA_SQL, home)",
    },
    "reconcile_state_schema": {
        "def reconcile_state_schema(conn: sqlite3.Connection) -> None:":
            "def reconcile_state_schema(conn: sqlite3.Connection, home: Path) -> None:",
        "apply_declared_schema(conn.cursor())": "apply_declared_schema(conn.cursor(), home)",
    },
    "fts5_cjk_so_path": {
        "def fts5_cjk_so_path() -> Path:": "def fts5_cjk_so_path(home: Path) -> Path:",
        '"""Location of the cjk_unicode61 loadable extension."""':
            '"""Location of the cjk_unicode61 loadable extension for the store in the folder ``home``."""',
        'return Path(env).expanduser() if env else get_hermes_home() / "lib" / "libfts5_cjk.so"':
            'return Path(env).expanduser() if env else home / "lib" / "libfts5_cjk.so"',
    },
    "load_fts5_cjk_extension": {
        "def load_fts5_cjk_extension(conn: sqlite3.Connection) -> bool:":
            "def load_fts5_cjk_extension(conn: sqlite3.Connection, home: Path) -> bool:",
        "path = fts5_cjk_so_path()": "path = fts5_cjk_so_path(home)",
    },
    "_ensure_fts_cjk_schema": {
        "fts5_cjk_so_path(),": "fts5_cjk_so_path(self.db_path.parent),",
    },
    "_reap_inactive_orphan_desktop_holders": {
        "from hermes_state import psutil": None,
        "from hermes_state_dbfile import _concrete_state_db_holder_pids, _is_inactive_orphan_desktop_holder": (
            "from hermes_state_dbfile import _concrete_state_db_holder_pids, "
            "_is_inactive_orphan_desktop_holder"),
        "from hermes_cli.dashboard_procs import _is_ephemeral_port_zero_backend":
            "from ._dashboard_procs import _is_ephemeral_port_zero_backend",
    },
}


def _source(value) -> str:
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    return textwrap.dedent(inspect.getsource(value))


def _legacy(owner: str):
    """The legacy module or class ``owner`` names (``module`` or ``module.Class``), from the pinned checkout."""
    module, _, cls = owner.partition(".")
    value = importlib.import_module(module)
    return getattr(value, cls) if cls else value


@functools.cache
def _catalog() -> Catalog:
    """The legacy store's classes, whose members the copies reach as module functions of ``_schema``."""
    return Catalog({}, core_members(), frozenset({owner.rpartition(".")[2] for owner in METHODS} | {"SessionDB"}))


def _member_kind(value, name: str) -> str:
    """A legacy member's kind for ``receiver_form``; an instance method whose copy takes no holder is static."""
    kind = "static" if isinstance(value, staticmethod) else "class" if isinstance(value, classmethod) else "instance"
    return "static" if kind == "instance" and _catalog().members[name].kind == "static" else kind


def _expected_source(legacy: str, substitutions: dict) -> str:
    lines, used = [], set()
    for line in legacy.splitlines():
        stripped = line.strip()
        if stripped in substitutions:
            used.add(stripped)
            if substitutions[stripped] is None:
                continue
            line = line[:len(line) - len(line.lstrip())] + substitutions[stripped]
        lines.append(line)
    assert used == set(substitutions), f"substitution not applied: {set(substitutions) - used}"
    return "\n".join(lines)


def _copy(name):
    """The copied value: the DDL module's when it defines the name, else the schema module's."""
    return vars(_schema_ddl)[name] if name in _defined(_schema_ddl, DDL_SOURCE) else getattr(_schema, name)


def _defined(module, source: str) -> set[str]:
    functions = {name for name, value in vars(module).items()
                 if inspect.isfunction(value) and value.__module__ == module.__name__ and not name.startswith("__")}
    return functions | _top_level_assigns(source)


def _cases(manifest):
    return [pytest.param(owner, name, id=f"{owner.rpartition('.')[2]}.{name}") for owner, names in manifest.items()
            for name in names]


@pytest.mark.parametrize(("module", "name"), _cases(CONSTANTS))
def test_constant_equals_m1(legacy_store, module, name):
    legacy, copied = getattr(_legacy(module), name), _copy(name)
    if hasattr(legacy, "pattern"):
        assert (copied.pattern, copied.flags) == (legacy.pattern, legacy.flags)
    else:
        assert copied == legacy


def test_versions_equal_m1(legacy_store):
    import hermes_state_common

    assert _schema.SCHEMA_VERSION == hermes_state_common.SCHEMA_VERSION == 32
    assert _schema.FTS_STORAGE_VERSION == hermes_state_common.FTS_STORAGE_VERSION == 3


@pytest.mark.parametrize(("table", "index"), [
    pytest.param(table, index, id=f"{table}[{index}]")
    for table, layouts in _schema_ddl.FTS_VTABLE_LAYOUTS.items() for index in range(len(layouts))
])
def test_fts_layout_entry_equals_m1(legacy_store, table, index):
    import hermes_state_fts

    assert len(_schema_ddl.FTS_VTABLE_LAYOUTS[table]) == len(hermes_state_fts.FTS_VTABLE_LAYOUTS[table])
    assert _schema_ddl.FTS_VTABLE_LAYOUTS[table][index] == hermes_state_fts.FTS_VTABLE_LAYOUTS[table][index]


def test_fts_layout_tables_equal_m1(legacy_store):
    import hermes_state_fts

    assert set(_schema_ddl.FTS_VTABLE_LAYOUTS) == set(hermes_state_fts.FTS_VTABLE_LAYOUTS)


def test_read_probe_statements_equal_m1(tmp_path, legacy_store):
    import hermes_state_schema

    assert _schema.schema_read_probe_statements(tmp_path) == hermes_state_schema.schema_read_probe_statements()


# The two argument shapes the transcript and session readers pass.
@pytest.mark.parametrize("reasons_sql", [
    _schema_ddl._RESET_END_REASONS_SQL, common._placeholders(common._RESET_END_REASONS)])
def test_legacy_reset_child_sql_text_equals_m1(legacy_store, reasons_sql):
    import hermes_state_common

    backend = _schema_ddl._legacy_reset_child_sql("child", reasons_sql).encode()
    assert backend == hermes_state_common._legacy_reset_child_sql("child", reasons_sql).encode()


@pytest.mark.parametrize(("module", "name"), _cases(FUNCTIONS))
def test_function_source_equals_m1(legacy_store, module, name):
    expected = receiver_form(_expected_source(_source(getattr(_legacy(module), name)), SUBSTITUTIONS.get(name, {})),
                             _catalog())
    assert strip_moved_allows(_source(_copy(name))).rstrip("\n") == expected.rstrip("\n")


@pytest.mark.parametrize(("cls", "name"), _cases(METHODS))
def test_method_source_equals_m1(legacy_store, cls, name):
    legacy = vars(_legacy(cls))[name]
    text = inspect.getsource(legacy.__func__ if isinstance(legacy, (staticmethod, classmethod)) else legacy)
    expected = receiver_form(_expected_source(text, SUBSTITUTIONS.get(name, {})), _catalog(),
                             _member_kind(legacy, name))
    assert strip_moved_allows(_source(getattr(_schema, name))).rstrip("\n") == expected.rstrip("\n")


@pytest.mark.parametrize(("cls", "name"), _cases(CLASS_ATTRIBUTES))
def test_class_attribute_equals_m1(legacy_store, cls, name):
    assert getattr(_schema, name) == vars(_legacy(cls))[name]


def test_nothing_copied_outside_the_manifests(legacy_store):
    """Every function, former class member and constant in _schema is compared above, and every one the legacy schema
    module declares is listed."""
    import hermes_state_schema

    methods = {name for names in METHODS.values() for name in names}
    attributes = {name for names in CLASS_ATTRIBUTES.values() for name in names}
    assert set(_catalog().members) == methods | attributes
    modules = ((_schema, SCHEMA_SOURCE), (_schema_ddl, DDL_SOURCE))
    functions = {name for module, source in modules for name in _defined(module, source) - _top_level_assigns(source)}
    assert functions == {name for names in FUNCTIONS.values() for name in names} | methods
    constants = {name for names in CONSTANTS.values() for name in names}
    assigns = _top_level_assigns(SCHEMA_SOURCE) | _top_level_assigns(DDL_SOURCE)
    assert assigns == constants | attributes | {"logger", "_READ_PROBE_STATEMENTS"}
    # One definition per name: what the schema module uses from the DDL module, it imports.
    assert not (_defined(_schema, SCHEMA_SOURCE) - {"logger"}) & _defined(_schema_ddl, DDL_SOURCE)
    legacy_source = Path(hermes_state_schema.__file__).read_text(encoding="utf-8")
    assert set(CONSTANTS["hermes_state_schema"]) == (
        _top_level_assigns(legacy_source) - {"logger", "_READ_PROBE_STATEMENTS"})
    assert set(FUNCTIONS["hermes_state_schema"]) == {
        name for name, value in vars(hermes_state_schema).items()
        if inspect.isfunction(value) and value.__module__ == hermes_state_schema.__name__ and not name.startswith("__")}
    assert set(METHODS["hermes_state_schema.SessionSchemaMixin"]) == (
        _functions_of(hermes_state_schema.SessionSchemaMixin))


def test_shares_m1_logger_and_psutil(legacy_store):
    import hermes_state
    import hermes_state_schema

    assert _schema.logger is _schema_ddl.logger is hermes_state_schema.logger
    assert _schema.psutil is hermes_state.psutil


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
