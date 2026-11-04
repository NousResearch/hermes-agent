"""Behaviour tests for the native llama.cpp settings adapter (identity + wire).

The adapter derives the settings identity from the EXACT bytes of the staged
GGUF (all split parts) plus required catalog companions (mmproj / MTP head —
never the optional draft, never catalog display names), hashes them by
streaming with stat-change rejection, and maps between the trusted engine
schema and the OpenAI wire shape. Synthetic tiny GGUFs and temp homes only:
no downloads, no server, no model loads.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hermes_cli.local_runtime import bootstrap, catalog, native_model_settings as nms
from hermes_cli.local_runtime.model_settings import (
    AUTO,
    InvalidSettingValueError,
    ModelSettingsService,
    ModelSettingsStore,
    UnknownSettingError,
)
from tests.hermes_cli._native_gguf_factory import write_gguf as _write_gguf


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nms.clear_fingerprint_cache()
    yield root
    nms.clear_fingerprint_cache()


# ---------------------------------------------------------------------------
# Trusted schema: pinned semantics, zero preservation, no invented ranges


def test_native_schema_keys_are_the_trusted_llamacpp_chat_set():
    keys = set(nms.NATIVE_CHAT_SCHEMA.keys())
    assert keys == {"temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                    "presence_penalty", "frequency_penalty", "seed"}
    assert nms.NATIVE_CHAT_SCHEMA.engine == "llamacpp"
    assert nms.NATIVE_CHAT_SCHEMA.workflow == "chat"
    # Nothing here is a load knob: load remains Auto policy, decided elsewhere.
    assert all(spec.apply == "request" for spec in nms.NATIVE_CHAT_SCHEMA.settings)


def test_schema_zero_is_preserved_and_negatives_follow_pinned_semantics():
    v = nms.NATIVE_CHAT_SCHEMA.validate_value
    assert v("temperature", 0) == 0
    assert v("min_p", 0.0) == 0.0
    assert v("seed", 0) == 0
    assert v("presence_penalty", -2.0) == -2.0
    with pytest.raises(InvalidSettingValueError):
        v("temperature", -0.1)
    with pytest.raises(InvalidSettingValueError):
        v("presence_penalty", -2.5)   # outside the pinned API range
    with pytest.raises(InvalidSettingValueError):
        v("temperature", float("nan"))
    with pytest.raises(InvalidSettingValueError):
        v("seed", 1.5)                # typed int: a float seed is not accepted
    with pytest.raises(UnknownSettingError):
        v("n_ctx", 4096)              # load knobs are not in this schema


def test_every_native_setting_supports_auto():
    for spec in nms.NATIVE_CHAT_SCHEMA.settings:
        assert spec.supports_auto is True
        assert nms.NATIVE_CHAT_SCHEMA.validate_value(spec.key, AUTO) is AUTO


# ---------------------------------------------------------------------------
# Identity: exact installed bytes, streamed and stat-guarded


def test_unknown_model_and_path_traversal_are_rejected(home):
    with pytest.raises(nms.UnknownModelError):
        nms.resolve_staged_model("Never-Downloaded")
    for bad in ("", "../.env", "..", "a/b", "a\\b", ".hidden"):
        with pytest.raises(nms.InvalidModelIdError):
            nms.resolve_staged_model(bad)


def test_fingerprint_tracks_exact_model_bytes(home):
    _write_gguf(bootstrap.models_dir() / "Ident-Model.gguf", pad=b"a" * 16)
    fp1 = nms.artifact_fingerprint("Ident-Model")
    assert nms.artifact_fingerprint("Ident-Model") == fp1  # cached, stable
    # Same metadata, different weight-ish bytes => different artifact.
    _write_gguf(bootstrap.models_dir() / "Ident-Model.gguf", pad=b"b" * 16)
    nms.clear_fingerprint_cache()  # the cache is stat-guarded; also prove recompute
    assert nms.artifact_fingerprint("Ident-Model") != fp1


def test_identity_covers_every_split_part(home):
    _write_gguf(bootstrap.models_dir() / "Split-Model-00001-of-00002.gguf", pad=b"x" * 8)
    _write_gguf(bootstrap.models_dir() / "Split-Model-00002-of-00002.gguf", pad=b"y" * 8)
    fp = nms.artifact_fingerprint("Split-Model")
    _write_gguf(bootstrap.models_dir() / "Split-Model-00002-of-00002.gguf", pad=b"!" * 8)
    assert nms.artifact_fingerprint("Split-Model") != fp  # stat change busts the cache


def test_required_companions_are_part_of_identity_draft_is_not(home, monkeypatch):
    _write_gguf(bootstrap.models_dir() / "Vision-Model.gguf")
    fp_without = nms.artifact_fingerprint("Vision-Model")

    mmproj = catalog.AssetFile(path="proj.gguf", size_bytes=4, local="Vision-Model-mmproj.gguf")
    draft = catalog.AssetFile(path="draft.gguf", size_bytes=4, local="Vision-Model-draft.gguf")
    entry = catalog.CatalogEntry(
        id="vision", display_name="Vision", description="", repo="x/y",
        variants=(catalog.QuantVariant(
            quant="Q4", files=(catalog.AssetFile(path="Vision-Model.gguf", size_bytes=4),)),),
        n_ctx_train=8192, full_layers=1, recurrent_layers=0, per_layer_f16=1,
        mmproj=mmproj, draft=draft)
    monkeypatch.setattr(catalog, "CATALOG", (entry,))

    fp_still_without = nms.artifact_fingerprint("Vision-Model")
    assert fp_still_without == fp_without  # declared-but-missing companions are hashed when present
    bootstrap.assets_dir().mkdir(parents=True, exist_ok=True)  # real downloads create it
    (bootstrap.assets_dir() / "Vision-Model-mmproj.gguf").write_bytes(b"m" * 4)
    assert nms.artifact_fingerprint("Vision-Model") != fp_without
    (bootstrap.assets_dir() / "Vision-Model-draft.gguf").write_bytes(b"d" * 4)
    assert nms.artifact_fingerprint("Vision-Model") != fp_without  # mmproj still changes it
    before = nms.artifact_fingerprint("Vision-Model")
    (bootstrap.assets_dir() / "Vision-Model-draft.gguf").write_bytes(b"e" * 5)
    # The optional draft never joins the identity: downloading/rotating one
    # must not orphan the model's saved settings.
    assert nms.artifact_fingerprint("Vision-Model") == before


def test_mutable_catalog_names_are_not_artifact_identity(home, monkeypatch):
    _write_gguf(bootstrap.models_dir() / "Named-Model.gguf")
    entry = catalog.CatalogEntry(
        id="named", display_name="Original Name", description="", repo="x/y",
        variants=(catalog.QuantVariant(
            quant="Q4", files=(catalog.AssetFile(path="Named-Model.gguf", size_bytes=4),)),),
        n_ctx_train=8192, full_layers=1, recurrent_layers=0, per_layer_f16=1)
    monkeypatch.setattr(catalog, "CATALOG", (entry,))
    fp = nms.artifact_fingerprint("Named-Model")
    monkeypatch.setattr(catalog, "CATALOG", (replace(entry, display_name="Renamed!"),))
    assert nms.artifact_fingerprint("Named-Model") == fp


def test_hash_is_streamed_and_change_during_hash_is_rejected(home, monkeypatch):
    path = _write_gguf(bootstrap.models_dir() / "Racing-Model.gguf", pad=b"a" * 64)
    reads: list[int] = []
    real_chunks = nms._iter_file_chunks

    def counted_chunks(p):
        reads.append(1)
        yield from real_chunks(p)

    monkeypatch.setattr(nms, "_iter_file_chunks", counted_chunks)
    fp = nms.artifact_fingerprint("Racing-Model")
    assert reads == [1]
    nms.artifact_fingerprint("Racing-Model")
    assert reads == [1]  # cache hit: the bytes were not re-read

    def racing_chunks(p):
        # Simulate a concurrent writer mid-stream: the file changes after the
        # first chunk is read. The result must be an error, never a fingerprint
        # of mixed generations.
        with open(p, "rb") as f:
            chunk = f.read(8)
        path.write_bytes(path.read_bytes() + b"appended")
        yield chunk

    monkeypatch.setattr(nms, "_iter_file_chunks", racing_chunks)
    nms.clear_fingerprint_cache()
    with pytest.raises(nms.ArtifactIdentityError):
        nms.artifact_fingerprint("Racing-Model")
    # A rejected hash does not poison the cache: the clean re-hash returns the
    # CURRENT generation, not the torn one.
    monkeypatch.setattr(nms, "_iter_file_chunks", real_chunks)
    assert nms.artifact_fingerprint("Racing-Model") != fp


def test_settings_key_is_derived_from_fingerprint(home):
    _write_gguf(bootstrap.models_dir() / "Keyed-Model.gguf")
    key = nms.settings_key_for_model("Keyed-Model")
    assert key.engine == "llamacpp" and key.workflow == "chat"
    assert key.model_fingerprint == nms.artifact_fingerprint("Keyed-Model")


# ---------------------------------------------------------------------------
# Layered defaults: real GGUF metadata first, curated catalog second


def test_publisher_defaults_come_from_gguf_metadata_and_beat_catalog(home, monkeypatch):
    _write_gguf(bootstrap.models_dir() / "Meta-Model.gguf", metadata={
        "general.architecture": "qwen2",
        "general.sampling.temperature": 0.75,
        "general.sampling.top_k": 40,
    })
    entry = catalog.CatalogEntry(
        id="meta", display_name="Meta", description="", repo="x/y",
        variants=(catalog.QuantVariant(
            quant="Q4", files=(catalog.AssetFile(path="Meta-Model.gguf", size_bytes=4),)),),
        n_ctx_train=8192, full_layers=1, recurrent_layers=0, per_layer_f16=1,
        sampling={"temp": "1.0", "top-p": "0.95", "top-k": "20"})
    monkeypatch.setattr(catalog, "CATALOG", (entry,))

    publisher = nms.gguf_sampling_metadata("Meta-Model")
    assert publisher == {"temperature": pytest.approx(0.75), "top_k": 40}
    curated = nms.catalog_sampling_defaults("Meta-Model")
    assert curated == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

    response = nms.read_model_settings("Meta-Model", home)
    assert response["effective"]["sources"]["temperature"] == "publisher"
    assert response["effective"]["values"]["temperature"] == pytest.approx(0.75)
    assert response["effective"]["sources"]["top_p"] == "catalog"
    assert response["effective"]["sources"]["seed"] == "inherit"
    assert "seed" not in response["effective"]["values"]  # unknown engine default omitted


def test_unreadable_gguf_metadata_falls_through_without_inventing(home, monkeypatch):
    monkeypatch.setattr(nms, "_read_gguf_metadata", lambda model_id: {})
    _write_gguf(bootstrap.models_dir() / "Bare-Model.gguf")
    response = nms.read_model_settings("Bare-Model", home)
    assert response["effective"]["values"] == {}
    assert set(response["effective"]["sources"]) == set(nms.NATIVE_CHAT_SCHEMA.keys())
    assert set(response["effective"]["sources"].values()) == {"inherit"}


# ---------------------------------------------------------------------------
# Wire decoding: tagged AUTO only where supported, nothing else


def test_wire_decode_accepts_tagged_auto_and_scalars_only():
    decoded = nms.decode_wire_overrides(
        {"temperature": 0.5, "seed": {"auto": True}, "top_k": 0})
    assert decoded["temperature"] == 0.5 and decoded["top_k"] == 0
    assert decoded["seed"] is AUTO
    with pytest.raises(InvalidSettingValueError):
        nms.decode_wire_overrides({"temperature": {"auto": True, "extra": 1}})
    with pytest.raises(InvalidSettingValueError):
        nms.decode_wire_overrides({"temperature": {"nested": {"deep": 1}}})
    with pytest.raises(InvalidSettingValueError):
        nms.decode_wire_overrides({"temperature": [1.0]})
    with pytest.raises(UnknownSettingError):
        nms.decode_wire_overrides({"n_ctx": 4096})


# ---------------------------------------------------------------------------
# Domain save/reset/read (store under the caller's profile root)


def test_save_read_reset_round_trip_through_domain_functions(home):
    _write_gguf(bootstrap.models_dir() / "Round-Model.gguf",
                metadata={"general.sampling.temperature": 0.5})
    fp = nms.read_model_settings("Round-Model", home)["key"]["model_fingerprint"]
    saved = nms.save_model_settings("Round-Model", home, {"temperature": 1.0}, 0, fp)
    assert saved["revision"] == 1
    assert saved["overrides"] == {"temperature": 1.0}
    assert saved["effective"]["sources"]["temperature"] == "model"

    partial = nms.save_model_settings(
        "Round-Model", home, {"temperature": 0.75, "top_k": {"auto": True}}, 1, fp)
    assert partial["revision"] == 2
    assert partial["overrides"] == {"temperature": 0.75, "top_k": {"auto": True}}

    trimmed = nms.reset_model_settings("Round-Model", home, 2, fp, keys=["top_k"])
    assert trimmed["revision"] == 3
    assert trimmed["overrides"] == {"temperature": 0.75}

    cleared = nms.reset_model_settings("Round-Model", home, 3, fp, keys=None)
    assert cleared["overrides"] == {}
    # The cleared response is exactly what a fresh read reports.
    assert cleared == nms.read_model_settings("Round-Model", home)


def test_identity_race_blocks_stale_fingerprint_before_cas(home):
    """GET → same model id replaced on disk → PUT with the OLD fingerprint must
    409 model_changed BEFORE the CAS: a coincidental revision 0 on the new
    artifact must never silently adopt the old file's settings."""
    path = bootstrap.models_dir() / "Same-Id.gguf"
    _write_gguf(path, pad=b"old-generation")
    view = nms.read_model_settings("Same-Id", home)
    old_fp = view["key"]["model_fingerprint"]

    _write_gguf(path, pad=b"new-generation!")
    fresh_fp = nms.read_model_settings("Same-Id", home)["key"]["model_fingerprint"]
    assert fresh_fp != old_fp

    with pytest.raises(nms.ModelChangedError) as exc:
        nms.save_model_settings("Same-Id", home, {"temperature": 1.0}, 0, old_fp)
    assert exc.value.current_fingerprint == fresh_fp
    # Nothing was written for the old binding, on either generation's key.
    assert nms.read_model_settings("Same-Id", home)["revision"] == 0
    with pytest.raises(nms.ModelChangedError):
        nms.reset_model_settings("Same-Id", home, 0, old_fp, keys=None)

    # A write carrying the CURRENT fingerprint proceeds normally.
    assert nms.save_model_settings(
        "Same-Id", home, {"temperature": 1.0}, 0, fresh_fp)["revision"] == 1


def test_companion_change_is_an_identity_change_for_writes(home):
    _write_gguf(bootstrap.models_dir() / "Cmp-Id.gguf")
    fp = nms.artifact_fingerprint("Cmp-Id")
    nms.save_model_settings("Cmp-Id", home, {"temperature": 1.0}, 0, fp)
    _write_gguf(bootstrap.assets_dir() / "Cmp-Id-mmproj.gguf")  # NOT declared → not hashed
    assert nms.artifact_fingerprint("Cmp-Id") == fp  # undeclared stray asset: not identity


def test_missing_expected_fingerprint_is_rejected(home):
    _write_gguf(bootstrap.models_dir() / "Guarded-Model.gguf")
    with pytest.raises(nms.MissingBindingError):
        nms.save_model_settings("Guarded-Model", home, {"temperature": 1.0}, 0, None)
    with pytest.raises(nms.MissingBindingError):
        nms.save_model_settings("Guarded-Model", home, {"temperature": 1.0}, 0, "nope")
    with pytest.raises(nms.MissingBindingError):
        nms.reset_model_settings("Guarded-Model", home, 0, "", keys=None)


def test_store_is_root_scoped_profile_isolates(home, tmp_path):
    _write_gguf(bootstrap.models_dir() / "Scoped-Model.gguf")
    fp = nms.artifact_fingerprint("Scoped-Model")
    other = tmp_path / "profile-b"
    nms.save_model_settings("Scoped-Model", home, {"temperature": 1.25}, 0, fp)
    assert nms.read_model_settings("Scoped-Model", home)["revision"] == 1
    assert nms.read_model_settings("Scoped-Model", other)["revision"] == 0
    assert nms.read_model_settings("Scoped-Model", other)["overrides"] == {}


def test_reset_rejects_keys_outside_the_trusted_schema(home):
    _write_gguf(bootstrap.models_dir() / "Strict-Model.gguf")
    fp = nms.artifact_fingerprint("Strict-Model")
    nms.save_model_settings("Strict-Model", home, {"temperature": 1.0}, 0, fp)
    with pytest.raises(UnknownSettingError):
        nms.reset_model_settings("Strict-Model", home, 1, fp, keys=["n_ctx"])


# ---------------------------------------------------------------------------
# Dispatch helper: request precedence + OpenAI wire split


def _stage_with_defaults(home) -> str:
    _write_gguf(bootstrap.models_dir() / "Dispatch-Model.gguf", metadata={
        "general.sampling.temperature": 0.75,
        "general.sampling.top_k": 40,
    })
    key = nms.settings_key_for_model("Dispatch-Model")
    ModelSettingsService(ModelSettingsStore(home)).save(
        key, nms.NATIVE_CHAT_SCHEMA, {"top_p": 0.9, "seed": AUTO}, 0)
    return "Dispatch-Model"


def test_resolve_native_request_settings_precedence_and_wire_split(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    mid = _stage_with_defaults(home)

    standard, extra = nms.resolve_native_request_settings(mid)
    assert standard == {"temperature": pytest.approx(0.75), "top_p": 0.9}
    # top_k is a llama.cpp extension; AUTO on the standard field seed is omitted.
    assert extra == {"top_k": 40}

    standard, extra = nms.resolve_native_request_settings(
        mid, {"temperature": 0.3, "top_k": {"auto": True}})
    assert standard["temperature"] == 0.3          # explicit request wins over saved
    assert extra == {"top_k": "auto"}              # tagged AUTO serializes as "auto"

    # The helper only contributes setting keys: whatever extra_body the caller
    # already holds survives a per-key merge untouched.
    assert set(extra) <= set(nms.NATIVE_CHAT_SCHEMA.keys())


def test_resolve_native_request_settings_rejects_unsupported_payload(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    mid = _stage_with_defaults(home)
    with pytest.raises(UnknownSettingError):
        nms.resolve_native_request_settings(mid, {"n_ctx": 1024})
    with pytest.raises(InvalidSettingValueError):
        nms.resolve_native_request_settings(mid, {"temperature": {"value": 1}})
    with pytest.raises(nms.UnknownModelError):
        nms.resolve_native_request_settings("Not-Staged")
