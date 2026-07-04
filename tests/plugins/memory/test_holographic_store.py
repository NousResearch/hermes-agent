"""Regression tests for the holographic memory store's entity resolution.

These exercise ``MemoryStore._resolve_entity`` directly so they do not
depend on numpy (the HRR vector path is skipped when numpy is absent).
The focus is the wildcard-injection bug: entity names extracted from fact
content can legitimately contain ``%`` and ``_``, which SQLite LIKE treats
as wildcards, silently merging distinct entities.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_HOLO_DIR = (
    Path(__file__).resolve().parents[3]
    / "plugins"
    / "memory"
    / "holographic"
)
_STORE_PATH = _HOLO_DIR / "store.py"


def _load_store_module():
    # store.py uses ``from . import holographic`` with a flat-import
    # fallback; loaded standalone the relative form fails, so make the
    # plugin directory importable for the fallback path.
    if str(_HOLO_DIR) not in sys.path:
        sys.path.insert(0, str(_HOLO_DIR))
    spec = importlib.util.spec_from_file_location("holographic_store", _STORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def store(tmp_path):
    module = _load_store_module()
    db_path = tmp_path / "memory_store.db"
    inst = module.MemoryStore(db_path=str(db_path))
    yield inst
    inst._conn.close()


def test_underscore_in_name_does_not_match_unrelated_entity(store):
    """``_`` must be a literal, not a single-char LIKE wildcard."""
    first = store._resolve_entity("CxAPI")
    # "C_API" as a LIKE pattern would match "CxAPI" — it must not.
    second = store._resolve_entity("C_API")
    assert first != second


def test_percent_in_name_does_not_match_unrelated_entity(store):
    """``%`` must be a literal, not a multi-char LIKE wildcard."""
    first = store._resolve_entity("100 dollars off")
    # "100%" as a LIKE pattern would match "100 dollars off" — it must not.
    second = store._resolve_entity("100%")
    assert first != second


def test_exact_name_match_is_case_insensitive(store):
    """The wildcard-safe match must preserve case-insensitive resolution."""
    first = store._resolve_entity("Python")
    second = store._resolve_entity("python")
    assert first == second


def test_literal_name_with_wildcard_chars_is_reused(store):
    """An identical name containing ``%``/``_`` still resolves to itself."""
    first = store._resolve_entity("C_API")
    second = store._resolve_entity("C_API")
    assert first == second


def test_alias_wildcard_does_not_overmatch(store):
    """Wildcard chars in the query must not leak into the alias LIKE pattern."""
    store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES (?, ?)",
        ("Gamma", "kappa,lambda"),
    )
    store._conn.commit()
    # "k_ppa" as a LIKE pattern would match the "kappa" alias — it must not.
    resolved = store._resolve_entity("k_ppa")
    alias_owner = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = ?", ("Gamma",)
    ).fetchone()
    assert resolved != int(alias_owner["entity_id"])


def test_alias_exact_match_still_resolves(store):
    """A literal alias token must still resolve to its owning entity."""
    cur = store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES (?, ?)",
        ("Guido", "bdfl,pythonista"),
    )
    store._conn.commit()
    owner_id = int(cur.lastrowid)
    assert store._resolve_entity("bdfl") == owner_id
