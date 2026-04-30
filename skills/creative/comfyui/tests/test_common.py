"""Unit tests for _common.py — pure logic only, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from _common import (
    DEFAULT_LOCAL_HOST,
    EMBEDDING_REGEX,
    FOLDER_ALIASES,
    build_cloud_aware_url,
    cloud_endpoint,
    coerce_seed,
    folder_aliases_for,
    is_api_format,
    is_cloud_host,
    is_link,
    iter_embedding_refs,
    iter_model_deps,
    iter_nodes,
    looks_like_video_workflow,
    media_type_from_filename,
    parse_model_list,
    resolve_url,
    safe_path_join,
    unwrap_workflow,
)


# =============================================================================
# Cloud detection / URL routing
# =============================================================================

class TestCloudDetection:
    def test_cloud_host_exact(self):
        assert is_cloud_host("https://cloud.comfy.org") is True
        assert is_cloud_host("https://cloud.comfy.org/foo/bar") is True

    def test_cloud_host_subdomain(self):
        assert is_cloud_host("https://staging.cloud.comfy.org") is True
        assert is_cloud_host("https://api.cloud.comfy.org") is True

    def test_local_not_cloud(self):
        assert is_cloud_host("http://127.0.0.1:8188") is False
        assert is_cloud_host("http://localhost:8188") is False
        assert is_cloud_host("http://my-server.local:8188") is False

    def test_no_scheme(self):
        # Defaults to http://
        assert is_cloud_host("cloud.comfy.org") is True
        assert is_cloud_host("127.0.0.1:8188") is False


class TestCloudEndpointRename:
    def test_history_renamed(self):
        assert cloud_endpoint("/history") == "/history_v2"
        assert cloud_endpoint("/history/abc-123") == "/history_v2/abc-123"

    def test_history_v2_preserved(self):
        assert cloud_endpoint("/history_v2") == "/history_v2"

    def test_models_renamed(self):
        assert cloud_endpoint("/models") == "/experiment/models"
        assert cloud_endpoint("/models/checkpoints") == "/experiment/models/checkpoints"
        assert cloud_endpoint("/models/loras") == "/experiment/models/loras"

    def test_other_paths_unchanged(self):
        assert cloud_endpoint("/prompt") == "/prompt"
        assert cloud_endpoint("/queue") == "/queue"


class TestResolveURL:
    def test_local_no_prefix(self):
        assert resolve_url("http://127.0.0.1:8188", "/prompt") == "http://127.0.0.1:8188/prompt"

    def test_cloud_adds_api_prefix(self):
        assert resolve_url("https://cloud.comfy.org", "/prompt") == "https://cloud.comfy.org/api/prompt"

    def test_cloud_history_renamed(self):
        assert resolve_url("https://cloud.comfy.org", "/history/abc") == "https://cloud.comfy.org/api/history_v2/abc"

    def test_cloud_models_renamed(self):
        assert resolve_url("https://cloud.comfy.org", "/models/loras") == "https://cloud.comfy.org/api/experiment/models/loras"

    def test_cloud_already_has_api(self):
        # Don't double-prefix
        assert resolve_url("https://cloud.comfy.org", "/api/prompt") == "https://cloud.comfy.org/api/prompt"

    def test_trailing_slash_stripped(self):
        assert resolve_url("http://127.0.0.1:8188/", "/prompt") == "http://127.0.0.1:8188/prompt"


# =============================================================================
# Workflow validation
# =============================================================================

class TestAPIFormatDetection:
    def test_valid_api(self, sd15_workflow):
        assert is_api_format(sd15_workflow) is True

    def test_editor_format_rejected(self):
        editor = {"nodes": [], "links": [], "version": 0.4}
        assert is_api_format(editor) is False

    def test_empty_dict(self):
        assert is_api_format({}) is False

    def test_non_dict(self):
        assert is_api_format([]) is False
        assert is_api_format(None) is False
        assert is_api_format("string") is False

    def test_node_with_class_type(self):
        wf = {"3": {"class_type": "KSampler", "inputs": {}}}
        assert is_api_format(wf) is True


class TestUnwrapWorkflow:
    def test_passthrough_api_format(self, sd15_workflow):
        result = unwrap_workflow(sd15_workflow)
        assert result is sd15_workflow

    def test_unwrap_prompt_key(self, sd15_workflow):
        wrapped = {"prompt": sd15_workflow, "client_id": "abc"}
        result = unwrap_workflow(wrapped)
        assert result is sd15_workflow

    def test_editor_format_raises(self):
        with pytest.raises(ValueError, match="editor format"):
            unwrap_workflow({"nodes": [], "links": []})

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            unwrap_workflow({"foo": "bar"})


class TestIsLink:
    def test_valid_link(self):
        assert is_link(["3", 0]) is True
        assert is_link(["10", 1]) is True

    def test_non_link(self):
        assert is_link("string") is False
        assert is_link(42) is False
        assert is_link([]) is False
        assert is_link(["3"]) is False  # missing slot
        assert is_link(["3", "0"]) is False  # slot must be int
        assert is_link([3, 0]) is False  # node_id must be string


# =============================================================================
# Workflow iterators
# =============================================================================

class TestIterators:
    def test_iter_nodes(self, sd15_workflow):
        nodes = dict(iter_nodes(sd15_workflow))
        assert "3" in nodes
        assert nodes["3"]["class_type"] == "KSampler"

    def test_iter_nodes_skips_comments(self, sd15_workflow):
        # _comment is not a node
        nodes = dict(iter_nodes(sd15_workflow))
        assert "_comment" not in nodes

    def test_iter_model_deps(self, sd15_workflow):
        deps = list(iter_model_deps(sd15_workflow))
        names = [d["value"] for d in deps]
        assert "v1-5-pruned-emaonly.safetensors" in names

    def test_iter_model_deps_flux(self, flux_workflow):
        deps = list(iter_model_deps(flux_workflow))
        names = {d["value"]: d["folder"] for d in deps}
        assert names["flux1-dev.safetensors"] == "unet"
        assert names["t5xxl_fp16.safetensors"] == "clip"
        assert names["clip_l.safetensors"] == "clip"
        assert names["ae.safetensors"] == "vae"


# =============================================================================
# Embedding extraction
# =============================================================================

class TestEmbeddingRegex:
    def test_basic_embedding(self):
        m = EMBEDDING_REGEX.search("a cat, embedding:goodvibes, more text")
        assert m is not None
        assert m.group(1) == "goodvibes"

    def test_embedding_with_strength(self):
        m = EMBEDDING_REGEX.search("embedding:bad-hands-5:1.2")
        assert m is not None
        assert m.group(1) == "bad-hands-5"

    def test_embedding_with_extension(self):
        # Strips .pt / .safetensors / .bin
        m = EMBEDDING_REGEX.search("embedding:my-emb.pt")
        assert m is not None
        assert m.group(1) == "my-emb"

    def test_embedding_in_parens(self):
        m = EMBEDDING_REGEX.search("(embedding:foo:0.8)")
        assert m is not None
        assert m.group(1) == "foo"

    def test_multiple_in_one_string(self):
        text = "a cat, embedding:foo:1.2, and embedding:bar"
        matches = [m.group(1) for m in EMBEDDING_REGEX.finditer(text)]
        assert matches == ["foo", "bar"]

    def test_no_false_positive_on_word_embedding(self):
        # "embedding " (with space, no colon) should not match
        m = EMBEDDING_REGEX.search("the embedding is great")
        assert m is None


class TestIterEmbeddingRefs:
    def test_finds_in_clip_text_encode(self):
        wf = {
            "1": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "embedding:foo, embedding:bar:0.5", "clip": ["2", 0]}},
            "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
        }
        refs = list(iter_embedding_refs(wf))
        names = [name for _, name in refs]
        assert names == ["foo", "bar"]

    def test_ignores_non_prompt_fields(self):
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "embedding:foo.safetensors"}},
        }
        refs = list(iter_embedding_refs(wf))
        # ckpt_name is not a prompt field — ignored
        assert refs == []


# =============================================================================
# Path safety
# =============================================================================

class TestSafePathJoin:
    def test_normal_join(self, tmp_path):
        p = safe_path_join(tmp_path, "subdir", "file.png")
        assert p.is_relative_to(tmp_path)

    def test_blocks_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="path traversal"):
            safe_path_join(tmp_path, "..", "..", "etc", "passwd")

    def test_blocks_absolute(self, tmp_path):
        with pytest.raises(ValueError):
            safe_path_join(tmp_path, "/etc/passwd")

    def test_subfolder_with_filename(self, tmp_path):
        p = safe_path_join(tmp_path, "outputs", "img.png")
        assert p.name == "img.png"
        assert p.parent.name == "outputs"


# =============================================================================
# Seed coercion
# =============================================================================

class TestCoerceSeed:
    def test_explicit_int(self):
        assert coerce_seed(42) == 42
        assert coerce_seed(0) == 0

    def test_minus_one_randomizes(self):
        s = coerce_seed(-1)
        assert isinstance(s, int)
        assert 0 <= s < 2**63

    def test_none_randomizes(self):
        s = coerce_seed(None)
        assert isinstance(s, int)

    def test_string_int(self):
        # str() that converts cleanly is allowed (relaxed)
        assert coerce_seed("12345") == 12345

    def test_string_minus_one_randomizes(self):
        # CLI / JSON sometimes carries seed as a string.
        s = coerce_seed("-1")
        assert isinstance(s, int)
        assert 0 <= s < 2**63
        # And whitespace tolerated
        s2 = coerce_seed(" -1 ")
        assert isinstance(s2, int)
        assert 0 <= s2 < 2**63


# =============================================================================
# Model list normalization (cloud format)
# =============================================================================

class TestParseModelList:
    def test_local_format_strings(self):
        result = parse_model_list(["a.safetensors", "b.safetensors"])
        assert result == {"a.safetensors", "b.safetensors"}

    def test_cloud_format_dicts(self):
        result = parse_model_list([
            {"name": "a.safetensors", "pathIndex": 0},
            {"name": "b.safetensors", "pathIndex": 1},
        ])
        assert result == {"a.safetensors", "b.safetensors"}

    def test_empty(self):
        assert parse_model_list([]) == set()

    def test_garbage(self):
        assert parse_model_list("not a list") == set()
        assert parse_model_list(None) == set()

    def test_mixed_format(self):
        result = parse_model_list([
            "string-form.safetensors",
            {"name": "dict-form.safetensors"},
        ])
        assert result == {"string-form.safetensors", "dict-form.safetensors"}


# =============================================================================
# Folder aliases
# =============================================================================

class TestFolderAliases:
    def test_unet_aliases_diffusion_models(self):
        aliases = folder_aliases_for("unet")
        assert "unet" in aliases
        assert "diffusion_models" in aliases

    def test_clip_aliases_text_encoders(self):
        aliases = folder_aliases_for("clip")
        assert "clip" in aliases
        assert "text_encoders" in aliases

    def test_unknown_folder_returns_self(self):
        assert folder_aliases_for("checkpoints") == ["checkpoints"]

    def test_primary_first(self):
        # Order matters: primary should be first for human-friendly fix hints
        assert folder_aliases_for("unet")[0] == "unet"
        assert folder_aliases_for("diffusion_models")[0] == "diffusion_models"


# =============================================================================
# Media-type detection
# =============================================================================

class TestMediaType:
    def test_video_extensions(self):
        assert media_type_from_filename("vid.mp4") == "video"
        assert media_type_from_filename("foo.webm") == "video"
        assert media_type_from_filename("bar.gif") == "video"

    def test_audio_extensions(self):
        assert media_type_from_filename("song.wav") == "audio"
        assert media_type_from_filename("music.mp3") == "audio"

    def test_image_default(self):
        assert media_type_from_filename("pic.png") == "image"
        assert media_type_from_filename("image.jpg") == "image"
        assert media_type_from_filename("unknown.xyz") == "image"

    def test_3d(self):
        assert media_type_from_filename("model.glb") == "3d"
        assert media_type_from_filename("scene.gltf") == "3d"


# =============================================================================
# Cross-host header stripping (security)
# =============================================================================

class TestRedirectHeaderStripping:
    """Verify X-API-Key is dropped when redirect crosses to a different host
    (e.g. cloud /api/view → S3 signed URL). Critical to prevent leaking auth
    tokens to the storage backend.
    """

    def _build_session(self):
        from _common import _StripSensitiveOnRedirectSession, HAS_REQUESTS
        if not HAS_REQUESTS:
            import pytest
            pytest.skip("requests not installed")
        return _StripSensitiveOnRedirectSession()

    def test_strips_x_api_key_cross_host(self):
        import requests
        s = self._build_session()
        prep = requests.PreparedRequest()
        prep.prepare(method="GET", url="https://other.example.com/file",
                     headers={"X-API-Key": "leak", "Authorization": "Bearer x"})
        resp = requests.Response()
        orig = requests.PreparedRequest()
        orig.prepare(method="GET", url="https://cloud.comfy.org/api/view", headers={})
        resp.request = orig
        s.rebuild_auth(prep, resp)
        assert "X-API-Key" not in prep.headers
        assert "Authorization" not in prep.headers

    def test_preserves_x_api_key_same_host(self):
        import requests
        s = self._build_session()
        prep = requests.PreparedRequest()
        prep.prepare(method="GET", url="https://cloud.comfy.org/foo",
                     headers={"X-API-Key": "keep"})
        resp = requests.Response()
        orig = requests.PreparedRequest()
        orig.prepare(method="GET", url="https://cloud.comfy.org/bar", headers={})
        resp.request = orig
        s.rebuild_auth(prep, resp)
        assert prep.headers.get("X-API-Key") == "keep"

    def test_strips_cookie_cross_host(self):
        import requests
        s = self._build_session()
        prep = requests.PreparedRequest()
        prep.prepare(method="GET", url="https://other.example.com/x",
                     headers={"Cookie": "session=secret"})
        resp = requests.Response()
        orig = requests.PreparedRequest()
        orig.prepare(method="GET", url="https://cloud.comfy.org/foo", headers={})
        resp.request = orig
        s.rebuild_auth(prep, resp)
        assert "Cookie" not in prep.headers


# =============================================================================
# Video workflow detection
# =============================================================================

class TestVideoWorkflow:
    def test_image_workflow(self, sd15_workflow):
        assert looks_like_video_workflow(sd15_workflow) is False

    def test_animatediff_workflow(self, workflows_dir):
        import json
        wf = json.loads((workflows_dir / "animatediff_video.json").read_text())
        assert looks_like_video_workflow(wf) is True

    def test_wan_workflow(self, video_workflow):
        assert looks_like_video_workflow(video_workflow) is True
