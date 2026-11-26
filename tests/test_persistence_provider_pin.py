"""The persistence provider pin and the provider-neutral store probes of ``persistence_assembly``."""

from __future__ import annotations

from pathlib import Path

import pytest

import persistence_assembly as pa
from persistence import StoreUnavailable


def _home(tmp_path: Path, config: str | None) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if config is not None:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


def test_an_unpinned_home_without_a_section_uses_sqlite(tmp_path):
    assert pa._provider_for_home(_home(tmp_path, "model: x\n")) == "sqlite"


def test_a_pin_names_the_provider_when_the_config_names_none(tmp_path):
    home = _home(tmp_path, "model: x\n")
    pa.pin_provider(home, "sqlite")
    assert pa.pinned_provider(home) == "sqlite"
    assert pa._provider_for_home(home) == "sqlite"


def test_a_config_naming_another_provider_than_the_pin_refuses(tmp_path):
    home = _home(tmp_path, "persistence:\n  provider: sqlite\n")
    pa.pin_provider(home, "postgres")
    with pytest.raises(StoreUnavailable, match="pinned to the 'postgres' persistence provider"):
        pa._provider_for_home(home)


def test_an_unreadable_config_refuses_unless_pinned(tmp_path):
    home = _home(tmp_path, "persistence: [unclosed\n")
    with pytest.raises(StoreUnavailable, match="cannot be read"):
        pa._provider_for_home(home)
    pa.pin_provider(home, "sqlite")
    assert pa._provider_for_home(home) == "sqlite"


def test_sqlite_store_probes_follow_the_state_db_file(tmp_path):
    home = _home(tmp_path, "model: x\n")
    assert pa.session_store_exists(home) is False
    assert pa.store_fingerprint(home) is None
    (home / "state.db").write_bytes(b"v1")
    assert pa.session_store_exists(home) is True
    first = pa.store_fingerprint(home)
    (home / "state.db-wal").write_bytes(b"wal")
    assert pa.store_fingerprint(home) not in (None, first)
