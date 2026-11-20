"""Regression: ensure_matrix_deps() fresh-dependency rebind must not NameError.

sdk-bindings-review.md #43: ``_import()`` referenced ``PaginationDirection`` /
``SyncToken`` that were never imported, and ``pm.extras.ensure_and_bind`` catches
only ImportError — so a fresh install (missing packages) crashed adapter creation
instead of rebinding the type globals or returning False with the install hint.

Tests run through the real ``ensure_and_bind`` boundary with the install seam
(``pm.extras.ensure_import``) no-op'd — no network, no live Matrix.
"""
import sys
import types

import pytest
from unittest.mock import patch

import pm.extras as pm_extras
from plugins.platforms.matrix import adapter as matrix_adapter


def _fake_mautrix_types():
    """Minimal mautrix.types with the 8 names the adapter imports/binds."""
    mod = types.ModuleType("mautrix.types")

    for name in ("EventType", "UserID", "RoomID", "EventID", "ContentURI",
                 "RoomCreatePreset", "PresenceState", "TrustState"):
        setattr(mod, name, object())
    return mod


@pytest.fixture
def fresh_dependency_boundary(monkeypatch):
    """Exercise the real importer after PM admits the SDK."""
    monkeypatch.setattr(pm_extras, "ensure_import", lambda *a, **kw: None)
    monkeypatch.delenv("MATRIX_E2EE_MODE", raising=False)
    monkeypatch.delenv("MATRIX_ENCRYPTION", raising=False)
    # ensure_and_bind writes module globals; isolate even successful rebinding.
    for name in ("EventType", "UserID", "RoomID", "EventID", "ContentURI", "RoomCreatePreset", "PresenceState", "TrustState"):
        monkeypatch.setattr(matrix_adapter, name, getattr(matrix_adapter, name))
    fake_types = _fake_mautrix_types()
    mautrix = types.ModuleType("mautrix")
    mautrix.types = fake_types
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": fake_types}):
        yield fake_types


def test_fresh_install_rebinds_type_globals_without_nameerror(fresh_dependency_boundary):
    assert matrix_adapter.ensure_matrix_deps() is True
    # The rebind actually landed on the adapter module globals.
    assert matrix_adapter.EventType is fresh_dependency_boundary.EventType
    assert matrix_adapter.UserID is fresh_dependency_boundary.UserID
    assert matrix_adapter.TrustState is fresh_dependency_boundary.TrustState


def test_failed_install_returns_false_with_hint_and_never_raises(
    fresh_dependency_boundary, caplog
):
    # Same fresh state but the post-install import genuinely fails (no mautrix):
    # must return False with the install hint — and, per the bug class, must not
    # leak anything other than ImportError out of ensure_and_bind.
    with patch.dict(sys.modules, {"mautrix": None, "mautrix.types": None}):
        with caplog.at_level("WARNING", logger="plugins.platforms.matrix.adapter"):
            assert matrix_adapter.ensure_matrix_deps() is False
    assert any("required packages not installed" in r.message for r in caplog.records)


def test_interactive_setup_explicitly_syncs_matrix(tmp_path, monkeypatch):
    import pm
    from hermes_cli import cli_output, config

    answers = iter(["https://matrix.example.test", "test-token", "@bot:example.test", "@owner:example.test", "!home:example.test"])
    monkeypatch.setattr(cli_output, "prompt", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(cli_output, "prompt_yes_no", lambda *args, **kwargs: False)
    monkeypatch.setattr(config, "get_env_value", lambda key: None)
    monkeypatch.setattr(config, "save_env_value", lambda *args: None)
    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras, **kwargs: calls.append((extras, kwargs)))
    monkeypatch.setattr(pm, "ensure_import", lambda *args: pytest.fail("setup used implicit installation"))
    monkeypatch.setattr(pm_extras, "missing", lambda extra: ("asyncpg",))
    matrix_adapter.interactive_setup()
    assert calls == [(["matrix"], {"explicit": True})]
