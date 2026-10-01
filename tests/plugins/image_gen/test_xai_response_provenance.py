"""xAI response provenance must not be replaced by requested values (#87483)."""
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("branch", ["b64_json", "url", "file_output"])
def test_image_result_preserves_response_identity(monkeypatch, tmp_path, branch):
    import plugins.image_gen.xai as xai
    import plugins.image_gen._common as common

    monkeypatch.setattr(xai, "resolve_xai_http_credentials", lambda: {"api_key": "fixture"})
    monkeypatch.setattr(xai, "_live_models", lambda: {})
    monkeypatch.setattr(xai, "load_image_gen_config", lambda *_: {})
    monkeypatch.setattr(xai, "build_xai_storage_options", lambda *a, **k: None)
    monkeypatch.setattr(xai, "maybe_mark_xai_storage_notice_seen", lambda *_: None)
    monkeypatch.setattr(xai, "read_xai_imagine_storage_config", lambda *_: {"enabled": False})
    monkeypatch.setattr(common, "save_url_image", Mock(side_effect=OSError("offline fixture")))
    first = {"revised_prompt": "A red apple on yellow."}
    if branch == "file_output":
        first[branch] = {"public_url": "https://fixture.invalid/image.png"}
    else:
        first[branch] = "aW1hZ2U=" if branch == "b64_json" else "https://fixture.invalid/image.png"
    response = Mock()
    response.json.return_value = {"model": "other-model", "data": [first], "usage": {"images": 1}}
    monkeypatch.setattr(xai.requests, "post", Mock(return_value=response))
    result = xai.XAIImageGenProvider().generate("An apple", model="grok-imagine-image-quality")
    assert result["success"] is True
    assert result["model"] == "grok-imagine-image-quality"
    assert result.get("requested_model") == result["model"]
    assert result["response_model"] == "other-model"
    assert result["model_mismatch"] is True
    assert result["revised_prompt"] == first["revised_prompt"]
    assert result["output_branch"] == branch
    assert result["usage"] == {"images": 1}
    assert result["warning"]
    if branch == "b64_json":
        from pathlib import Path
        assert Path(result["image"]).read_bytes() == b"image"
    else:
        assert result["image"] == "https://fixture.invalid/image.png"


@pytest.mark.parametrize("metadata", [{}, {"model": None}, {"model": {"unexpected": "shape"}}, {"model": "grok-imagine-image"}])
def test_missing_or_matching_response_model_is_not_a_mismatch(monkeypatch, metadata):
    import plugins.image_gen.xai as xai
    monkeypatch.setattr(xai, "resolve_xai_http_credentials", lambda: {"api_key": "fixture"})
    monkeypatch.setattr(xai, "_live_models", lambda: {})
    monkeypatch.setattr(xai, "load_image_gen_config", lambda *_: {})
    monkeypatch.setattr(xai, "build_xai_storage_options", lambda *a, **k: None)
    monkeypatch.setattr(xai, "maybe_mark_xai_storage_notice_seen", lambda *_: None)
    monkeypatch.setattr(xai, "read_xai_imagine_storage_config", lambda *_: {"enabled": False})
    response = Mock()
    response.json.return_value = {**metadata, "data": [{"b64_json": "aW1hZ2U=", "revised_prompt": None}]}
    monkeypatch.setattr(xai.requests, "post", Mock(return_value=response))
    result = xai.XAIImageGenProvider().generate("An apple")
    assert result["success"] is True
    assert result.get("requested_model") == result["model"]
    assert result["response_model"] == (metadata.get("model") if isinstance(metadata.get("model"), str) else None)
    assert result["model_mismatch"] is False
    assert "warning" not in result
    assert "revised_prompt" not in result
