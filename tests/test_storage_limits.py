"""``storage_limits.resolved_import_limits`` reads ``sessions.import_max_*`` from the named profile's config.yaml."""

from pathlib import Path

from persistence.housekeeping.maintenance import ImportLimits
from storage_limits import resolved_import_limits


def _write_sessions_config(home: Path, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("sessions:\n" + body)


def test_unset_keys_or_no_config_resolve_to_the_defaults(tmp_path):
    assert resolved_import_limits(tmp_path / "fresh") == ImportLimits()
    _write_sessions_config(tmp_path, "  max_export_messages: 7\n")
    assert resolved_import_limits(tmp_path) == ImportLimits()


def test_each_key_sets_its_guard_and_zero_disables_it(tmp_path):
    _write_sessions_config(
        tmp_path,
        "  import_max_sessions: 0\n"
        "  import_max_messages_per_session: 20000\n"
        "  import_max_total_messages: 3\n"
        "  import_max_session_bytes: 4096\n"
        "  import_max_total_bytes: 8192\n",
    )
    assert resolved_import_limits(tmp_path) == ImportLimits(
        max_sessions=0, max_messages_per_session=20000, max_total_messages=3, max_session_bytes=4096,
        max_total_bytes=8192)


def test_negative_or_non_numeric_values_keep_the_default(tmp_path):
    _write_sessions_config(tmp_path, "  import_max_sessions: -1\n  import_max_total_bytes: lots\n")
    assert resolved_import_limits(tmp_path) == ImportLimits()


def test_booleans_keep_the_default(tmp_path):
    """``false`` must not become 0 (guard off), nor ``true`` 1 (every import refused)."""
    _write_sessions_config(tmp_path, "  import_max_sessions: false\n  import_max_session_bytes: true\n")
    assert resolved_import_limits(tmp_path) == ImportLimits()


def test_the_named_profiles_config_wins_over_the_process_home(tmp_path):
    from hermes_constants import get_hermes_home

    _write_sessions_config(get_hermes_home(), "  import_max_sessions: 1\n")
    _write_sessions_config(tmp_path / "other", "  import_max_total_messages: 9\n")
    assert resolved_import_limits(tmp_path / "other") == ImportLimits(max_total_messages=9)
