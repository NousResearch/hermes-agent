"""Vision capability for managed local models answers from ground truth.

Cloud capability catalogs have never heard of a local GGUF, so without a
managed-runtime answer every local model reads as text-only: pasted images
detour to a cloud auxiliary (a screenshot leaving a local-first machine)
or fail outright. The lookup chain must consult the managed runtime
between the user's config override and the cloud catalog."""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib

    import hermes_constants

    importlib.reload(hermes_constants)
    yield home
    importlib.reload(hermes_constants)


def _stage(home_root, name):
    # Machine-scoped models dir (the shared root — tmp HERMES_HOME IS the root here).
    from hermes_cli.local_runtime.bootstrap import models_dir

    mdir = models_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{name}.gguf").write_bytes(b"GGUF" + b"\x00" * 32)


def test_not_ours_returns_none(hermes_home):
    from hermes_cli.local_runtime.capabilities import managed_model_supports_vision

    assert managed_model_supports_vision("gpt-4o") is None


def test_catalog_vision_model_with_projector_on_disk(hermes_home):
    """Staged catalog model with an mmproj present: True (server down —
    the catalog + on-disk projector answer)."""
    from hermes_cli.local_runtime.bootstrap import assets_dir
    from hermes_cli.local_runtime.capabilities import managed_model_supports_vision
    from hermes_cli.local_runtime.catalog import CATALOG

    entry = next(e for e in CATALOG if e.mmproj is not None)
    variant = entry.variants[-1]
    _stage(hermes_home, variant.model_id)
    adir = assets_dir()
    adir.mkdir(parents=True, exist_ok=True)
    (adir / entry.mmproj.local_name).write_bytes(b"GGUF mmproj")

    assert managed_model_supports_vision(variant.model_id) is True


def test_catalog_vision_model_missing_projector_is_blind(hermes_home):
    """Same model, projector NOT on disk: False — it genuinely cannot see,
    and claiming otherwise sends an image to a model that errors on it."""
    from hermes_cli.local_runtime.capabilities import managed_model_supports_vision
    from hermes_cli.local_runtime.catalog import CATALOG

    entry = next(e for e in CATALOG if e.mmproj is not None)
    variant = entry.variants[-1]
    _stage(hermes_home, variant.model_id)

    assert managed_model_supports_vision(variant.model_id) is False


def test_live_props_beats_catalog(hermes_home, monkeypatch):
    """A running child's modalities report wins over the catalog: the
    server that will receive the image is the authority."""
    import hermes_cli.local_runtime.capabilities as caps

    from hermes_cli.local_runtime.catalog import CATALOG

    entry = next(e for e in CATALOG if e.mmproj is not None)
    variant = entry.variants[-1]
    _stage(hermes_home, variant.model_id)
    # Catalog would say False (no projector staged) — live props says True.
    monkeypatch.setattr(caps, "_props_modalities", lambda mid: True)
    assert caps.managed_model_supports_vision(variant.model_id) is True


def test_lookup_chain_consults_managed_runtime(hermes_home, monkeypatch):
    """_lookup_supports_vision: user override wins, then the managed
    answer, and the cloud catalog is never reached for a managed model."""
    import agent.image_routing as ir

    monkeypatch.setattr(
        "hermes_cli.local_runtime.capabilities.managed_model_supports_vision",
        lambda mid: True)

    def catalog_must_not_run(*a, **k):
        raise AssertionError("cloud catalog consulted for a managed model")

    monkeypatch.setattr("agent.models_dev.get_model_capabilities",
                        catalog_must_not_run)

    got = ir._lookup_supports_vision("llamacpp", "Some-Local-Model", {})
    assert got is True

    # Explicit user override still outranks the managed answer.
    cfg = {"model": {"provider": "llamacpp", "name": "Some-Local-Model",
                     "supports_vision": False}}
    got = ir._lookup_supports_vision("llamacpp", "Some-Local-Model", cfg)
    assert got is False


def test_webp_transcodes_to_png_for_managed_provider(hermes_home, monkeypatch, tmp_path):
    """A .webp attachment bound for the managed server must arrive as PNG:
    llama.cpp's stb_image decoder has no WebP support and drops the part
    SILENTLY — the model confabulates a description of an image it never
    saw. Measured live: the same red square answered 'Red' as PNG and
    'Unseen' as WebP."""
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    import agent.image_routing as ir

    webp_path = tmp_path / "shot.webp"
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    img.save(webp_path, format="WEBP")

    monkeypatch.setattr("agent.auxiliary_client._runtime_main_value",
                        lambda k: {"provider": "llamacpp",
                                   "base_url": ""}.get(k, ""))

    data_url = ir._file_to_data_url(webp_path)
    assert data_url is not None
    assert data_url.startswith("data:image/png;base64,"), (
        "webp must transcode to png for the managed server")


def test_webp_passes_through_for_cloud_providers(hermes_home, monkeypatch, tmp_path):
    """Cloud providers accept WebP natively — no transcode tax for them."""
    pytest.importorskip("PIL")
    from PIL import Image

    import agent.image_routing as ir

    webp_path = tmp_path / "shot.webp"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(webp_path, format="WEBP")

    monkeypatch.setattr("agent.auxiliary_client._runtime_main_value",
                        lambda k: {"provider": "anthropic",
                                   "base_url": ""}.get(k, ""))

    data_url = ir._file_to_data_url(webp_path)
    assert data_url is not None
    assert data_url.startswith("data:image/webp;base64,")


def test_vision_analyze_normalization_narrows_for_managed(hermes_home, monkeypatch, tmp_path):
    """vision_analyze's native fast path embeds the image into conversation
    history via _normalize_to_supported_image — for the managed server a
    WebP must convert to PNG THERE too, or the tool path re-introduces the
    silent-drop confabulation the attachment path just fixed."""
    pytest.importorskip("PIL")
    from PIL import Image

    import tools.vision_tools as vt

    webp_path = tmp_path / "img.webp"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(webp_path, format="WEBP")

    monkeypatch.setattr("agent.auxiliary_client._runtime_main_value",
                        lambda k: {"provider": "llamacpp",
                                   "base_url": ""}.get(k, ""))
    out_path, mime, err = vt._normalize_to_supported_image(webp_path, "image/webp")
    assert err is None
    assert mime == "image/png", "managed server: webp must normalize to png"

    # Cloud providers keep webp untouched.
    monkeypatch.setattr("agent.auxiliary_client._runtime_main_value",
                        lambda k: {"provider": "anthropic",
                                   "base_url": ""}.get(k, ""))
    out_path, mime, err = vt._normalize_to_supported_image(webp_path, "image/webp")
    assert err is None
    assert mime == "image/webp"
