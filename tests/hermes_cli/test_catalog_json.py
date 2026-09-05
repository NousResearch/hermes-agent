"""The pulled catalog: packaged JSON is the offline truth, a GitHub fetch
swaps entries in memory only, and min_engine gates day-0 models.

Nothing here touches disk beyond the packaged file — the design constraint
is that a git checkout must never see a dirty tracked catalog.json."""

from __future__ import annotations

import dataclasses
import io
import json
import urllib.request

import pytest

import hermes_cli.local_runtime.catalog as cat


@pytest.fixture(autouse=True)
def _reset_refresh_state(monkeypatch):
    """Each test starts outside the TTL window with the packaged catalog."""
    monkeypatch.setattr(cat, "_last_refresh_attempt", 0.0)
    packaged = cat._packaged_catalog()
    monkeypatch.setattr(cat, "CATALOG", packaged)
    yield


def _doc_from(entries):
    """A fetchable catalog document built by mutating the packaged JSON."""
    from importlib.resources import files

    doc = json.loads(files("hermes_cli.local_runtime")
                     .joinpath("catalog.json").read_text(encoding="utf-8"))
    doc["models"] = entries(doc["models"])
    return doc


def _fetch_returns(monkeypatch, doc):
    body = json.dumps(doc).encode()

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: R(body))


def test_packaged_json_round_trips_the_catalog():
    """The packaged JSON must produce a complete, selection-ready catalog:
    every entry carries estimator inputs and at least one variant, and the
    known invariants (best-first ordering, Q4 floor) hold — the same
    contract the literals obeyed."""
    assert len(cat.CATALOG) >= 4
    for e in cat.CATALOG:
        assert e.variants and e.n_ctx_train > 0 and e.per_layer_f16 >= 0
        sizes = [v.size_bytes for v in e.variants]
        assert sizes == sorted(sizes, reverse=True), f"{e.id} not best-first"


def test_refresh_swaps_in_memory_only(monkeypatch, tmp_path):
    """A fetched catalog replaces CATALOG in memory; the packaged file on
    disk is untouched (checkout stays clean)."""
    from importlib.resources import files

    packaged_path = files("hermes_cli.local_runtime").joinpath("catalog.json")
    before = packaged_path.read_text(encoding="utf-8")

    def add_day0(models):
        day0 = dict(models[0])
        day0.update(id="day0-model", display_name="Day 0",
                    description="new", min_engine="b99999")
        return models + [day0]

    _fetch_returns(monkeypatch, _doc_from(add_day0))
    assert cat.refresh_catalog(force=True) is True
    assert "day0-model" in {e.id for e in cat.CATALOG}
    assert cat.catalog_by_id()["day0-model"].min_engine == "b99999"
    assert packaged_path.read_text(encoding="utf-8") == before


def test_refresh_failure_keeps_current_catalog(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    ids_before = [e.id for e in cat.CATALOG]
    assert cat.refresh_catalog(force=True) is False
    assert [e.id for e in cat.CATALOG] == ids_before


def test_refresh_rejects_wrong_schema(monkeypatch):
    doc = _doc_from(lambda m: m)
    doc["schema_version"] = 2
    _fetch_returns(monkeypatch, doc)
    ids_before = [e.id for e in cat.CATALOG]
    assert cat.refresh_catalog(force=True) is False
    assert [e.id for e in cat.CATALOG] == ids_before


def test_loader_ignores_unknown_fields():
    doc = _doc_from(lambda m: m)
    doc["models"][0]["future_field"] = {"anything": True}
    entries = cat._load_catalog(doc)
    assert entries[0].id == doc["models"][0]["id"]


def test_min_engine_gate(monkeypatch):
    from hermes_cli.web_routers.local_models import _engine_too_old

    monkeypatch.setattr("hermes_cli.local_runtime.binaries.installed_tags",
                        lambda: ["b10362"])
    assert _engine_too_old("") is False, "no requirement, no gate"
    assert _engine_too_old("b10000") is False, "installed engine suffices"
    assert _engine_too_old("b10363") is True, "newer requirement gates"
