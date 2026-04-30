"""Tests for run_workflow.py — focuses on logic that doesn't require a server."""

from __future__ import annotations

import copy
import json

import pytest

from extract_schema import extract_schema
from run_workflow import (
    ComfyRunner,
    download_outputs,
    inject_params,
    parse_input_image_arg,
)


class TestParseInputImageArg:
    def test_with_name(self, tmp_path):
        f = tmp_path / "x.png"
        f.write_text("x")
        n, p = parse_input_image_arg(f"image={f}")
        assert n == "image"
        assert p == f

    def test_without_name_defaults(self, tmp_path):
        f = tmp_path / "x.png"
        f.write_text("x")
        n, p = parse_input_image_arg(str(f))
        assert n == "image"

    def test_custom_name(self, tmp_path):
        f = tmp_path / "x.png"
        f.write_text("x")
        n, p = parse_input_image_arg(f"mask_image={f}")
        assert n == "mask_image"


class TestInjectParams:
    def test_basic_injection(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        wf, warnings = inject_params(sd15_workflow, schema, {
            "prompt": "new prompt",
            "seed": 999,
            "steps": 25,
        })
        assert wf["6"]["inputs"]["text"] == "new prompt"
        assert wf["3"]["inputs"]["seed"] == 999
        assert wf["3"]["inputs"]["steps"] == 25
        assert warnings == []

    def test_unknown_param_warns(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        _, warnings = inject_params(sd15_workflow, schema, {"foobar": "x"})
        assert any("foobar" in w for w in warnings)

    def test_seed_minus_one_randomizes(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        wf, warnings = inject_params(sd15_workflow, schema, {"seed": -1})
        assert wf["3"]["inputs"]["seed"] != -1
        assert isinstance(wf["3"]["inputs"]["seed"], int)
        assert any("expanded" in w.lower() for w in warnings)

    def test_randomize_seed_when_unset(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        original = sd15_workflow["3"]["inputs"]["seed"]
        wf, warnings = inject_params(sd15_workflow, schema, {}, randomize_seed_if_unset=True)
        assert wf["3"]["inputs"]["seed"] != original
        assert isinstance(wf["3"]["inputs"]["seed"], int)

    def test_does_not_mutate_original(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        original_text = sd15_workflow["6"]["inputs"]["text"]
        inject_params(sd15_workflow, schema, {"prompt": "MUTATED"})
        assert sd15_workflow["6"]["inputs"]["text"] == original_text

    def test_refuses_to_overwrite_link(self):
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": ["3", 0], "clip": ["1", 1]}},  # text is a link!
            "3": {"class_type": "KSampler",
                  "inputs": {"seed": 1, "steps": 20, "cfg": 7.5,
                             "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                             "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 0],
                             "latent_image": ["5", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["3", 0]}},
        }
        # Manually create a schema that has prompt pointing at 6.text
        schema = {
            "parameters": {
                "prompt": {"node_id": "6", "field": "text", "type": "string", "value": ""},
            }
        }
        wf2, warnings = inject_params(wf, schema, {"prompt": "literal value"})
        # The link should NOT have been overwritten
        assert wf2["6"]["inputs"]["text"] == ["3", 0]
        assert any("link" in w.lower() for w in warnings)


# =============================================================================
# Output download walk
# =============================================================================

class TestDownloadOutputsWalk:
    """Test that download_outputs walks the structure correctly."""

    def test_handles_videos_plural(self, tmp_path, monkeypatch):
        """Local ComfyUI uses 'videos'/'gifs' (plural) keys."""
        downloads = []

        class FakeRunner:
            def download_output(self, *, filename, subfolder, file_type, output_dir, preserve_subfolder, overwrite):
                downloads.append((filename, subfolder, file_type))
                p = output_dir / filename
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                return p

        outputs = {
            "9": {"images": [{"filename": "img1.png", "subfolder": "", "type": "output"}]},
            "10": {"videos": [{"filename": "vid1.mp4", "subfolder": "", "type": "output"}]},
            "11": {"gifs": [{"filename": "anim1.gif", "subfolder": "", "type": "output"}]},
        }

        result = download_outputs(FakeRunner(), outputs, tmp_path)
        files = sorted(d["filename"] for d in result)
        assert files == ["anim1.gif", "img1.png", "vid1.mp4"]

    def test_handles_video_singular_cloud(self, tmp_path):
        """Cloud uses 'video' (singular)."""
        class FakeRunner:
            def download_output(self, *, filename, subfolder, file_type, output_dir, preserve_subfolder, overwrite):
                p = output_dir / filename
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                return p

        outputs = {
            "10": {"video": [{"filename": "cloud.mp4", "subfolder": "", "type": "output"}]},
        }
        result = download_outputs(FakeRunner(), outputs, tmp_path)
        assert len(result) == 1
        assert result[0]["filename"] == "cloud.mp4"

    def test_preserves_subfolder(self, tmp_path):
        """When preserve_subfolder=True, server subfolder becomes local subdir."""
        class FakeRunner:
            def download_output(self, *, filename, subfolder, file_type, output_dir, preserve_subfolder, overwrite):
                if preserve_subfolder and subfolder:
                    p = output_dir / subfolder / filename
                else:
                    p = output_dir / filename
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                return p

        outputs = {
            "9": {"images": [
                {"filename": "img.png", "subfolder": "myrun", "type": "output"},
                {"filename": "img.png", "subfolder": "otherrun", "type": "output"},
            ]},
        }
        result = download_outputs(FakeRunner(), outputs, tmp_path, preserve_subfolder=True)
        files = [d["file"] for d in result]
        assert any("myrun" in f for f in files)
        assert any("otherrun" in f for f in files)
        # Both must exist (no collision)
        assert len({str(f) for f in files}) == 2


# =============================================================================
# ComfyRunner construction
# =============================================================================

class TestRunnerConstruction:
    def test_local_default(self):
        r = ComfyRunner()
        assert r.is_cloud is False
        assert r.host == "http://127.0.0.1:8188"

    def test_cloud_detection(self):
        r = ComfyRunner(host="https://cloud.comfy.org", api_key="abc")
        assert r.is_cloud is True
        assert "X-API-Key" in r.headers

    def test_cloud_subdomain_detected(self):
        r = ComfyRunner(host="https://staging.cloud.comfy.org", api_key="abc")
        assert r.is_cloud is True

    def test_partner_key_does_not_pollute_extra_data(self):
        r = ComfyRunner(host="https://cloud.comfy.org", api_key="auth-key")
        # No partner-key set → no extra_data should appear in submitted prompt
        # (This is a static check; runtime check happens in submit())
        assert r.partner_key is None

    def test_url_routing_local(self):
        r = ComfyRunner()
        url = r._url("/prompt")
        assert url == "http://127.0.0.1:8188/prompt"

    def test_url_routing_cloud(self):
        r = ComfyRunner(host="https://cloud.comfy.org", api_key="x")
        url = r._url("/prompt")
        assert url == "https://cloud.comfy.org/api/prompt"

    def test_url_routing_cloud_history_renamed(self):
        r = ComfyRunner(host="https://cloud.comfy.org", api_key="x")
        url = r._url("/history/abc-123")
        assert url == "https://cloud.comfy.org/api/history_v2/abc-123"
