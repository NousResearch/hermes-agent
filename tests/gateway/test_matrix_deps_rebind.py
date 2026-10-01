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


@pytest.mark.parametrize("mode,expected_installs,expected_result", [
    ("off", [], True),           # plain client: no crypto dep, nothing to install
    ("optional", ["matrix-e2ee"], True),   # best effort: warn, stay up
    ("required", ["matrix-e2ee"], False),  # fail closed when the closure is unusable
])
def test_e2ee_extra_installs_only_when_encryption_is_configured(
    fresh_dependency_boundary, monkeypatch, mode, expected_installs, expected_result
):
    """E2EE deps live in their own extra and are pulled only when E2EE is on.

    Regression: the encryption closure used to sit in the ``[matrix]`` extra, so even
    ``MATRIX_E2EE_MODE=off`` deployments resolved python-olm (and needed a C++ toolchain).
    """
    installed: list[str] = []
    monkeypatch.setattr(pm_extras, "ensure_import", lambda extra, *a, **kw: installed.append(extra))
    monkeypatch.setattr(matrix_adapter, "_check_e2ee_deps", lambda: False)
    monkeypatch.setenv("MATRIX_E2EE_MODE", mode)
    assert matrix_adapter.ensure_matrix_deps() is expected_result
    assert [extra for extra in installed if extra == "matrix-e2ee"] == expected_installs


def test_extra_names_track_the_e2ee_mode(monkeypatch):
    """The updater's configured-features pass must ask for [matrix-e2ee] only with E2EE on."""
    monkeypatch.delenv("MATRIX_E2EE_MODE", raising=False)
    monkeypatch.delenv("MATRIX_ENCRYPTION", raising=False)
    assert matrix_adapter._extra_names() == ["matrix"]
    for mode in ("optional", "required"):
        monkeypatch.setenv("MATRIX_E2EE_MODE", mode)
        assert matrix_adapter._extra_names() == ["matrix", "matrix-e2ee"]


def test_setup_enabling_e2ee_prepares_the_e2ee_extra(tmp_path, monkeypatch):
    """Saying yes to E2EE in setup must install the closure, not just the plain SDK."""
    import pm
    from hermes_cli import cli_output, config

    answers = iter(["https://matrix.example.test", "test-token", "@bot:example.test",
                    "@owner:example.test", "!home:example.test"])
    monkeypatch.setattr(cli_output, "prompt", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(cli_output, "prompt_yes_no", lambda *args, **kwargs: True)
    monkeypatch.setattr(config, "get_env_value", lambda key: None)
    monkeypatch.setattr(config, "save_env_value", lambda *args: None)
    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras, **kwargs: calls.append((extras, kwargs)))
    monkeypatch.setattr(pm_extras, "missing", lambda extra: ())
    matrix_adapter.interactive_setup()
    assert calls == [(["matrix", "matrix-e2ee"], {"explicit": True})]
