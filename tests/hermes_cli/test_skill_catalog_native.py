"""Native callers share reservation admission and never downgrade live authority."""
import time

import pytest

from hermes_cli.plugins_context import PluginContext
from hermes_cli.skill_catalog.validation import CatalogError
from tests.hermes_cli.test_skill_catalog_service import fixture


def context(entry):
    return PluginContext(entry.manifest, manager=None)


def test_native_receipt_uses_same_reservation_binding_as_generic_confirm(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager, entry, plan = fixture(tmp_path)
    try:
        api = context(entry).skill_catalog_mutations
        native = dict(plan, native_receipt_id='native_receipt_0001')
        result = api.admit_native(native, {'acknowledgements': [], 'accept_warnings': True})
        row, = entry.mutations.projection.read()['reservations'].values()
        assert row['operation_id'] == result['operation_id']
        assert row['review_id'] == plan['review_id']
        assert api.admit_native(native, {'acknowledgements': [], 'accept_warnings': True}) == result
        assert len(entry.mutations.projection.read()['reservations']) == 1
    finally:
        manager.unload()


@pytest.mark.parametrize('choices', [
    {'acknowledgements': ['unknown']}, {'acknowledgements': [False]},
    {'acknowledgements': [], 'accept_warnings': 'true'},
    {'acknowledgements': [], 'arbitrary': True},
])
def test_native_malformed_choices_never_reserve(tmp_path, monkeypatch, choices):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager, entry, plan = fixture(tmp_path)
    try:
        with pytest.raises(CatalogError):
            context(entry).skill_catalog_mutations.admit_native(plan, choices)
        assert not entry.mutations.projection.read()['reservations']
    finally:
        manager.unload()


def test_native_pending_recovery_reuses_original_binding_after_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager, entry, plan = fixture(tmp_path)
    try:
        api = context(entry).skill_catalog_mutations
        first = api.admit_native(plan, {'acknowledgements': []})
        monkeypatch.setattr(time, 'time', lambda: plan['expires_at_ms'] / 1000 + 1)
        assert api.admit_native(plan, {'acknowledgements': []}) == first
        assert len(entry.mutations.projection.read()['reservations']) == 1
    finally:
        manager.unload()


def test_other_plugin_without_live_catalog_keeps_stock_native_path(tmp_path, monkeypatch):
    from hermes_cli.plugins_manifest import PluginManifest
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager, _entry, _plan = fixture(tmp_path)
    try:
        other = PluginContext(PluginManifest(name='other', version='1'), manager=None)
        assert other.skill_catalog_mutations.admit_native({}, {}) is None
    finally:
        manager.unload()


def test_fresh_expired_native_receipt_never_reserves(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    manager, entry, plan = fixture(tmp_path)
    try:
        monkeypatch.setattr(time, 'time', lambda: plan['expires_at_ms'] / 1000 + 1)
        with pytest.raises(CatalogError) as caught:
            context(entry).skill_catalog_mutations.admit_native(plan, {'acknowledgements': []})
        assert caught.value.code == 'review_expired'
        assert not entry.mutations.projection.read()['reservations']
    finally:
        manager.unload()
