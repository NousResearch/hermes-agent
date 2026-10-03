"""Behavioral coverage for the configurable SQLite busy timeout (``database.busy_timeout_seconds``).

The delegation ledger used to hardcode its 10s busy timeout. The knob is read through the canonical
config loader at connect time; a malformed or absent value must fail safe to that historical 10s
default rather than disabling the wait (``busy_timeout=0`` surfaces ``database is locked`` at once).
"""

from __future__ import annotations

import pytest
import yaml

from hermes_cli import sqlite_util
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_DEFAULT_MS = DEFAULT_CONFIG["database"]["busy_timeout_seconds"] * 1000


def test_shipped_default_matches_resolver_fallback():
    assert _DEFAULT_MS > 0
    assert sqlite_util.resolve_busy_timeout_ms() == _DEFAULT_MS


def test_resolver_uses_each_active_profile_config_a_b_a(tmp_path):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, seconds in zip(homes, (3, 8)):
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({"database": {"busy_timeout_seconds": seconds}}), encoding="utf-8")
    for home, expected_ms in ((homes[0], 3000), (homes[1], 8000), (homes[0], 3000)):
        token = set_hermes_home_override(home)
        try:
            assert sqlite_util.resolve_busy_timeout_ms() == expected_ms
        finally:
            reset_hermes_home_override(token)


def _write_config(monkeypatch: pytest.MonkeyPatch, tmp_path, config: object) -> None:
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

@pytest.mark.parametrize("value, expected_ms", [
    (25, 25_000),
    (2.5, 2_500),
    (2_147_483, 2_147_483_000),  # just below SQLite's signed-32-bit millisecond limit
    (2_147_484, _DEFAULT_MS),  # overflow must not turn PRAGMA busy_timeout into 0
    (10**400, _DEFAULT_MS),  # float conversion raises OverflowError without a guard
    (1e300, _DEFAULT_MS),
    (0, _DEFAULT_MS),
    (-1, _DEFAULT_MS),
    ("25", _DEFAULT_MS),
    (None, _DEFAULT_MS),
    (True, _DEFAULT_MS),
    ({"bad": 1}, _DEFAULT_MS),
])
def test_resolve_busy_timeout_ms_reads_config_or_fails_safe(monkeypatch, tmp_path, value, expected_ms):
    _write_config(monkeypatch, tmp_path, {"database": {"busy_timeout_seconds": value}})
    assert sqlite_util.resolve_busy_timeout_ms() == expected_ms

@pytest.mark.parametrize("database", [None, [], "25", 42, {}])
def test_resolve_busy_timeout_ms_defaults_without_a_usable_database_section(
    monkeypatch, tmp_path, database
):
    if database is None:  # no config.yaml at all
        home = tmp_path / "hermes-home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
    else:
        _write_config(monkeypatch, tmp_path, {"database": database})
    assert sqlite_util.resolve_busy_timeout_ms() == _DEFAULT_MS
