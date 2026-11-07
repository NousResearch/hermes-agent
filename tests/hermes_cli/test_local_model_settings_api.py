"""Contract tests for the native model-settings API routes.

Real FastAPI TestClient against the real web_server app (auth included); the
routes run against a temp HERMES_HOME and synthetic GGUFs. Covers the parent
UI response contract, the identity acceptance gate (expected_fingerprint),
CAS conflicts, payload rejection, profile isolation and reset semantics.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.hermes_cli._native_gguf_factory import write_gguf as _write_gguf

SETTING_KEYS = {"temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                "presence_penalty", "frequency_penalty", "seed"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import web_server
    from hermes_cli.local_runtime import catalog, native_model_settings as nms

    monkeypatch.setattr(catalog, "refresh_catalog_soon", lambda: None)
    nms.clear_fingerprint_cache()
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def _settings_url(model_id: str) -> str:
    return "/api/local-models/settings/" + urllib.parse.quote(model_id, safe="")


def _stage(model_id: str, pad: bytes = b"") -> None:
    from hermes_cli.local_runtime import bootstrap

    _write_gguf(bootstrap.models_dir() / f"{model_id}.gguf", pad=pad)


def _stage_and_fingerprint(client, model_id: str, pad: bytes = b"") -> str:
    _stage(model_id, pad)
    r = client.get(_settings_url(model_id))
    assert r.status_code == 200, r.text
    return r.json()["key"]["model_fingerprint"]


# ── auth + identity resolution ───────────────────────────────

def test_settings_routes_require_auth(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    unauth = TestClient(web_server.app)
    assert unauth.get("/api/local-models/settings/Any-Model").status_code == 401


def test_get_unknown_model_is_404(client):
    assert client.get(_settings_url("Never-Staged")).status_code == 404


def test_get_traversal_model_id_never_serves(client):
    for bad in ("..%2F..%2F.env", "..%2F.env", "%2E%2E"):
        r = client.get("/api/local-models/settings/" + bad)
        assert r.status_code in (400, 404), (bad, r.status_code)


def test_get_unknown_query_params_are_rejected(client):
    _stage("Param-Model")
    r = client.get(_settings_url("Param-Model"), params={"root": "/tmp/evil"})
    assert r.status_code in (400, 422)


# ── GET response contract (parent UI) ────────────────────────

def test_get_full_response_contract(client):
    from hermes_cli.local_runtime import native_model_settings as nms

    fp = _stage_and_fingerprint(client, "Contract-Model",
                                pad=b"\x01" * 32)
    body = client.get(_settings_url("Contract-Model")).json()
    assert body["model_id"] == "Contract-Model"
    assert body["schema_version"] == nms.STORE_SCHEMA_VERSION
    assert body["key"] == {"model_fingerprint": fp, "engine": "llamacpp",
                           "workflow": "chat"}
    assert body["revision"] == 0
    assert body["overrides"] == {}
    # effective: values + per-key sources over the whole trusted schema.
    assert set(body["effective"]["sources"]) == SETTING_KEYS
    assert isinstance(body["effective"]["values"], dict)
    # fields: one entry per trusted setting, fully typed.
    fields = {f["key"]: f for f in body["fields"]}
    assert set(fields) == SETTING_KEYS
    for f in fields.values():
        assert f["kind"] in ("int", "float", "bool", "enum")
        assert f["apply"] == "request"          # load stays Auto policy
        assert isinstance(f["supports_auto"], bool)
        for attr in ("minimum", "maximum", "choices"):
            assert attr in f
    # runtime facts describe the engine, never a renderer guess.
    assert body["runtime"]["engine"] == "llamacpp"
    for attr in ("backend", "installed_tag", "pinned_tag"):
        assert body["runtime"][attr] is None or isinstance(body["runtime"][attr], str)
    # loading: read-only Auto until a real planner binds knobs.
    assert body["loading"] == {"policy": "auto",
                               "context_length": body["loading"]["context_length"]}
    ctx = body["loading"]["context_length"]
    assert ctx is None or (isinstance(ctx, int) and not isinstance(ctx, bool))


def test_get_reports_preset_window_as_context_length(client):
    from hermes_cli.local_runtime.binaries import runtimes_root

    _stage("Window-Model")
    ini = runtimes_root() / "presets.ini"
    ini.parent.mkdir(parents=True, exist_ok=True)
    ini.write_text(
        '# hermes-decision: {"model_id": "Window-Model", "window": 32768,'
        ' "spilled": false}\n[Window-Model]\nctx-size = 32768\n', encoding="utf-8")
    body = client.get(_settings_url("Window-Model")).json()
    assert body["loading"] == {"policy": "auto", "context_length": 32768}


# ── PUT save: identity gate + CAS + validation ───────────────

def test_put_save_round_trip_with_tagged_auto(client):
    fp = _stage_and_fingerprint(client, "Save-Model")
    r = client.put(_settings_url("Save-Model"), json={
        "expected_revision": 0, "expected_fingerprint": fp,
        "overrides": {"temperature": 0.5, "top_k": {"auto": True}}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["revision"] == 1
    assert body["overrides"] == {"temperature": 0.5, "top_k": {"auto": True}}
    assert body["effective"]["values"]["temperature"] == 0.5
    assert body["effective"]["sources"]["temperature"] == "model"
    assert body["effective"]["values"]["top_k"] == {"auto": True}
    # The save response is the same full contract as a fresh GET.
    assert body == client.get(_settings_url("Save-Model")).json()


def test_put_stale_revision_conflicts_409(client):
    fp = _stage_and_fingerprint(client, "Cas-Model")
    ok = client.put(_settings_url("Cas-Model"), json={
        "expected_revision": 0, "expected_fingerprint": fp,
        "overrides": {"temperature": 0.5}})
    assert ok.status_code == 200
    stale = client.put(_settings_url("Cas-Model"), json={
        "expected_revision": 0, "expected_fingerprint": fp,
        "overrides": {"temperature": 0.9}})
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert detail["error"] == "revision_conflict"
    assert detail["expected_revision"] == 0 and detail["current_revision"] == 1
    # The lost write did not land.
    assert client.get(_settings_url("Cas-Model")).json()["overrides"] == {"temperature": 0.5}


def test_put_old_fingerprint_after_byte_replacement_is_model_changed_409(client):
    """The identity acceptance gate: GET → same-id bytes replaced → PUT with the
    fingerprint from the old view must 409 BEFORE the CAS, even though the new
    artifact also sits at revision 0."""
    fp_old = _stage_and_fingerprint(client, "Swap-Model", pad=b"first")
    _stage("Swap-Model", pad=b"second!!")  # same id, different artifact
    stale = client.put(_settings_url("Swap-Model"), json={
        "expected_revision": 0, "expected_fingerprint": fp_old,
        "overrides": {"temperature": 1.5}})
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert detail["error"] == "model_changed"
    fresh = client.get(_settings_url("Swap-Model")).json()
    assert fresh["revision"] == 0
    assert fresh["key"]["model_fingerprint"] != fp_old
    assert detail["current_fingerprint"] == fresh["key"]["model_fingerprint"]
    assert fresh["overrides"] == {}
    # The write only proceeds against the NEW binding.
    ok = client.put(_settings_url("Swap-Model"), json={
        "expected_revision": 0,
        "expected_fingerprint": fresh["key"]["model_fingerprint"],
        "overrides": {"temperature": 1.5}})
    assert ok.status_code == 200 and ok.json()["revision"] == 1


def test_put_companion_change_also_trips_the_identity_gate(client):
    from dataclasses import replace as dreplace

    from hermes_cli.local_runtime import bootstrap, catalog

    _stage("Mmproj-Gate")
    fp = _stage_and_fingerprint(client, "Mmproj-Gate")
    entry = catalog.CatalogEntry(
        id="mmproj-gate", display_name="M", description="", repo="x/y",
        variants=(catalog.QuantVariant(
            quant="Q4", files=(catalog.AssetFile(path="Mmproj-Gate.gguf", size_bytes=4),)),),
        n_ctx_train=8192, full_layers=1, recurrent_layers=0, per_layer_f16=1)
    mmproj = catalog.AssetFile(path="p.gguf", size_bytes=4,
                               local="Mmproj-Gate-mmproj.gguf")
    with monkeypatched_catalog(dreplace(entry, mmproj=mmproj)):
        # Declared but absent companion: identity unchanged, the write proceeds.
        ok = client.put(_settings_url("Mmproj-Gate"), json={
            "expected_revision": 0, "expected_fingerprint": fp,
            "overrides": {"temperature": 1.0}})
        assert ok.status_code == 200, ok.text
        # The companion lands → the required-artifact set changed → gate trips.
        _write_gguf(bootstrap.assets_dir() / "Mmproj-Gate-mmproj.gguf")
        stale = client.put(_settings_url("Mmproj-Gate"), json={
            "expected_revision": 1, "expected_fingerprint": fp,
            "overrides": {"temperature": 1.0}})
        assert stale.status_code == 409
        assert stale.json()["detail"]["error"] == "model_changed"


class monkeypatched_catalog:
    """Context manager swapping catalog.CATALOG (import-time-safe helper)."""

    def __init__(self, *entries):
        self.entries = entries

    def __enter__(self):
        from hermes_cli.local_runtime import catalog
        self._saved = catalog.CATALOG
        catalog.CATALOG = tuple(self.entries)
        return self

    def __exit__(self, *exc):
        from hermes_cli.local_runtime import catalog
        catalog.CATALOG = self._saved


# ── payload rejection ────────────────────────────────────────

def test_put_rejects_unsupported_payloads(client):
    fp = _stage_and_fingerprint(client, "Strict-Payload")
    base = {"expected_revision": 0, "expected_fingerprint": fp}
    bad_bodies = [
        {**base, "overrides": {"n_ctx": 4096}},                    # load knob: not in schema
        {**base, "overrides": {"temperature": {"nested": {"deep": 1}}}},
        {**base, "overrides": {"temperature": [1.0]}},
        {**base, "overrides": {"temperature": 2.5}},               # outside pinned API range
        {**base, "overrides": {"temperature": "hot"}},
        {**base, "overrides": {"temperature": {"auto": True, "x": 1}}},
        {**base, "overrides": {"temperature": 1.0}, "root": "/tmp"},  # no caller-supplied root
        {**base, "overrides": {"temperature": 1.0}, "model_fingerprint": "x"},
        {"expected_revision": 0, "overrides": {"temperature": 1.0}},  # fingerprint required
    ]
    for body in bad_bodies:
        r = client.put(_settings_url("Strict-Payload"), json=body)
        assert r.status_code in (400, 422), (body, r.status_code, r.text)
    assert client.get(_settings_url("Strict-Payload")).json()["overrides"] == {}


def test_put_error_details_never_leak_private_paths(client):
    fp = _stage_and_fingerprint(client, "Quiet-Errors")
    r = client.put(_settings_url("Quiet-Errors"), json={
        "expected_revision": 0, "expected_fingerprint": fp,
        "overrides": {"n_ctx": 4096}})
    assert r.status_code in (400, 422)
    text = r.text
    home = str(Path.home())
    assert str(bootstrap_models_dir()) not in text
    assert home not in text


def bootstrap_models_dir() -> Path:
    from hermes_cli.local_runtime import bootstrap
    return bootstrap.models_dir()


# ── POST reset ───────────────────────────────────────────────

def test_post_reset_selected_keys_and_full(client):
    fp = _stage_and_fingerprint(client, "Reset-Model")
    client.put(_settings_url("Reset-Model"), json={
        "expected_revision": 0, "expected_fingerprint": fp,
        "overrides": {"temperature": 0.5, "top_p": 0.9}})
    partial = client.post(_settings_url("Reset-Model") + "/reset", json={
        "expected_revision": 1, "expected_fingerprint": fp, "keys": ["top_p"]})
    assert partial.status_code == 200, partial.text
    body = partial.json()
    assert body["overrides"] == {"temperature": 0.5}
    assert body["revision"] == 2
    full = client.post(_settings_url("Reset-Model") + "/reset", json={
        "expected_revision": 2, "expected_fingerprint": fp, "keys": None})
    assert full.status_code == 200
    assert full.json()["overrides"] == {}
    assert full.json() == client.get(_settings_url("Reset-Model")).json()


def test_post_reset_gates_and_conflicts(client):
    fp = _stage_and_fingerprint(client, "Reset-Gate")
    client.put(_settings_url("Reset-Gate"), json={
        "expected_revision": 0, "expected_fingerprint": fp,
        "overrides": {"temperature": 0.5}})
    stale_fp = client.post(_settings_url("Reset-Gate") + "/reset", json={
        "expected_revision": 1, "expected_fingerprint": "f" * 64, "keys": None})
    assert stale_fp.status_code == 409
    assert stale_fp.json()["detail"]["error"] == "model_changed"
    stale_rev = client.post(_settings_url("Reset-Gate") + "/reset", json={
        "expected_revision": 99, "expected_fingerprint": fp, "keys": None})
    assert stale_rev.status_code == 409
    assert stale_rev.json()["detail"]["error"] == "revision_conflict"
    bad_keys = client.post(_settings_url("Reset-Gate") + "/reset", json={
        "expected_revision": 1, "expected_fingerprint": fp, "keys": ["n_ctx"]})
    assert bad_keys.status_code in (400, 422)
    missing_fp = client.post(_settings_url("Reset-Gate") + "/reset", json={
        "expected_revision": 1, "keys": None})
    assert missing_fp.status_code in (400, 422)
    # None of the rejected calls changed the store.
    assert client.get(_settings_url("Reset-Gate")).json() ["overrides"] == {"temperature": 0.5}


def test_reset_unknown_model_is_404(client):
    r = client.post(_settings_url("Never-Staged") + "/reset", json={
        "expected_revision": 0, "expected_fingerprint": "a" * 64, "keys": None})
    assert r.status_code == 404


# ── profile isolation ────────────────────────────────────────

def test_profile_query_scopes_persistence_to_the_profiles_home(client, tmp_path):
    root = tmp_path / ".hermes"
    root.mkdir(parents=True, exist_ok=True)  # the fixture never creates the home dir
    (root / ".env").write_text("", encoding="utf-8")  # home marker for profile resolution
    alt = root / "profiles" / "alt"
    alt.mkdir(parents=True)
    (alt / ".env").write_text("", encoding="utf-8")   # profile identity marker
    fp = _stage_and_fingerprint(client, "Split-Profile")

    r = client.put(_settings_url("Split-Profile"), params={"profile": "alt"}, json={
        "expected_revision": 0, "expected_fingerprint": fp,
        "overrides": {"temperature": 0.25}})
    assert r.status_code == 200, r.text
    assert (alt / "local_runtime" / "model_settings.sqlite3").is_file()
    assert not (root / "local_runtime" / "model_settings.sqlite3").exists()
    # The default profile sees its own (empty) state; alt sees the write.
    assert client.get(_settings_url("Split-Profile")).json()["revision"] == 0
    scoped = client.get(_settings_url("Split-Profile"), params={"profile": "alt"}).json()
    assert scoped["revision"] == 1
    assert scoped["overrides"] == {"temperature": 0.25}
