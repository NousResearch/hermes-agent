"""Live Vertex Model Garden catalog discovery (native publishers/google/models).

Regression: the OpenAI-compatible Vertex chat endpoint has no /models route, so the picker
served only a hand-maintained static list that lagged new Gemini releases (gemini-4-carbon/neon).
These tests cover the native catalog fetcher and its wiring into the picker, with no network.
"""

from unittest.mock import MagicMock, patch

from agent import vertex_adapter as VA


def _resp(models, next_token=None, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {"publisherModels": models, "nextPageToken": next_token}
    return r


def _entry(mid, stage="GA"):
    return {"name": f"publishers/google/models/{mid}", "launchStage": stage}


def test_returns_none_without_explicit_config():
    with patch.object(VA, "has_explicit_vertex_config", return_value=False):
        assert VA.vertex_model_ids_or_none() is None


def test_returns_none_without_credentials():
    with patch.object(VA, "has_explicit_vertex_config", return_value=True), \
         patch.object(VA, "get_vertex_credentials", return_value=(None, None)):
        assert VA.vertex_model_ids_or_none() is None


def test_filters_chat_models_and_prefixes():
    models = [
        _entry("gemini-4-neon"), _entry("gemini-4-carbon"),
        _entry("gemini-4-argon"), _entry("gemini-3.8-flash"),
        _entry("gemini-embedding-2"), _entry("gemini-2.5-flash-tts"),
        _entry("gemini-3-pro-image"), _entry("gemini-nano-banana-2.1"),
        _entry("gemini-3.5-transcribe"), _entry("gemini-live-2.5-flash-native-audio"),
        _entry("gemini-robotics-er-2-preview", "PRIVATE_PREVIEW"),
        _entry("gemini-2.5-pro-exp-03-25", "EXPERIMENTAL"),
        _entry("gemini-3.1-pro-preview", "PUBLIC_PREVIEW"),
    ]
    with patch.object(VA, "has_explicit_vertex_config", return_value=True), \
         patch.object(VA, "get_vertex_credentials", return_value=("tok", "proj")), \
         patch.object(VA, "_resolve_region", return_value="global"), \
         patch("httpx.get", return_value=_resp(models)) as g:
        ids = VA.vertex_model_ids_or_none()
    assert ids is not None
    assert "google/gemini-4-neon" in ids
    assert "google/gemini-4-carbon" in ids
    assert "google/gemini-3.1-pro-preview" in ids  # PUBLIC_PREVIEW kept
    for bad in (
        "google/gemini-embedding-2", "google/gemini-2.5-flash-tts",
        "google/gemini-3-pro-image", "google/gemini-nano-banana-2.1",
        "google/gemini-3.5-transcribe", "google/gemini-live-2.5-flash-native-audio",
        "google/gemini-robotics-er-2-preview", "google/gemini-2.5-pro-exp-03-25",
    ):
        assert bad not in ids
    # global host, no project/location path prefix on the list call
    assert g.call_args[0][0] == "https://aiplatform.googleapis.com/v1beta1/publishers/google/models"


def test_pagination():
    with patch.object(VA, "has_explicit_vertex_config", return_value=True), \
         patch.object(VA, "get_vertex_credentials", return_value=("tok", "proj")), \
         patch.object(VA, "_resolve_region", return_value="global"), \
         patch("httpx.get", side_effect=[
             _resp([_entry("gemini-4-neon")], "t2"),
             _resp([_entry("gemini-4-carbon")]),
         ]) as g:
        ids = VA.vertex_model_ids_or_none()
    assert ids == ["google/gemini-4-neon", "google/gemini-4-carbon"]
    assert g.call_count == 2


def test_falls_back_to_next_host_on_error():
    with patch.object(VA, "has_explicit_vertex_config", return_value=True), \
         patch.object(VA, "get_vertex_credentials", return_value=("tok", "proj")), \
         patch.object(VA, "_resolve_region", return_value="us"), \
         patch("httpx.get", side_effect=[
             RuntimeError("dns"),
             _resp([_entry("gemini-4-neon")]),
         ]) as g:
        ids = VA.vertex_model_ids_or_none()
    assert ids == ["google/gemini-4-neon"]
    assert g.call_count == 2  # us-aiplatform failed, us rep host succeeded


def test_all_hosts_fail_returns_none():
    with patch.object(VA, "has_explicit_vertex_config", return_value=True), \
         patch.object(VA, "get_vertex_credentials", return_value=("tok", "proj")), \
         patch.object(VA, "_resolve_region", return_value="global"), \
         patch("httpx.get", side_effect=RuntimeError("down")):
        assert VA.vertex_model_ids_or_none() is None


def test_vertex_registered_in_fetchers():
    from hermes_cli import models as M
    assert M._PROVIDER_CATALOG_FETCHERS.get("vertex") is M._vertex_catalog


def test_vertex_catalog_fetcher_merges_curated_floor():
    from hermes_cli import models as M
    live = ["google/gemini-4-neon", "google/gemini-4-carbon"]
    with patch.object(VA, "vertex_model_ids_or_none", return_value=live):
        result = M._vertex_catalog("vertex", False)
    curated = list(M.static_catalog("vertex"))
    assert result[:len(curated)] == curated  # curated-first floor + silent default preserved
    assert "google/gemini-4-neon" in result
    assert "google/gemini-4-carbon" in result


def test_vertex_catalog_fetcher_none_on_miss():
    from hermes_cli import models as M
    with patch.object(VA, "vertex_model_ids_or_none", return_value=None):
        assert M._vertex_catalog("vertex", False) is None
