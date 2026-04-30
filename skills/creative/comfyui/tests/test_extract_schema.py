"""Tests for extract_schema.py."""

from __future__ import annotations

import pytest

from extract_schema import (
    extract_schema,
    find_negative_prompt_node,
    find_positive_prompt_node,
    trace_to_node,
)


# =============================================================================
# Connection tracing
# =============================================================================

class TestConnectionTracing:
    def test_direct_link(self):
        wf = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
            "2": {"class_type": "KSampler",
                  "inputs": {"positive": ["1", 0], "negative": ["1", 0]}},
        }
        assert trace_to_node(wf, ["1", 0]) == "1"

    def test_through_reroute(self):
        wf = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
            "2": {"class_type": "Reroute", "inputs": {"input": ["1", 0]}},
            "3": {"class_type": "Reroute", "inputs": {"input": ["2", 0]}},
        }
        assert trace_to_node(wf, ["3", 0]) == "1"

    def test_circular_safe(self):
        wf = {
            "1": {"class_type": "Reroute", "inputs": {"input": ["2", 0]}},
            "2": {"class_type": "Reroute", "inputs": {"input": ["1", 0]}},
        }
        # Should hit max_hops without infinite loop
        result = trace_to_node(wf, ["1", 0], max_hops=5)
        assert result in ("1", "2")  # any node, just don't hang


class TestPositiveNegativeDetection:
    def test_basic(self, sd15_workflow):
        # In sd15_workflow.json node 6 is positive, node 7 is negative
        assert find_positive_prompt_node(sd15_workflow) == "6"
        assert find_negative_prompt_node(sd15_workflow) == "7"

    def test_swapped_order(self):
        wf = {
            "3": {"class_type": "KSampler",
                  "inputs": {
                      "positive": ["7", 0], "negative": ["6", 0],
                      "model": ["4", 0], "latent_image": ["5", 0],
                      "seed": 1, "steps": 20, "cfg": 7.5,
                      "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                  }},
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "ugly", "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "beautiful", "clip": ["4", 1]}},
        }
        # Now 7 is the positive (despite higher node ID)
        assert find_positive_prompt_node(wf) == "7"
        assert find_negative_prompt_node(wf) == "6"


# =============================================================================
# Schema extraction
# =============================================================================

class TestExtractSchema:
    def test_basic_sd15(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        params = schema["parameters"]
        assert "prompt" in params
        assert "negative_prompt" in params
        assert "seed" in params
        assert "steps" in params
        assert "cfg" in params
        assert "width" in params
        assert "height" in params

    def test_prompt_value_correct(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        # The positive prompt in the example is the landscape one
        assert "landscape" in schema["parameters"]["prompt"]["value"]
        assert "ugly" in schema["parameters"]["negative_prompt"]["value"]

    def test_model_dependencies(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        deps = schema["model_dependencies"]
        ckpts = [d["value"] for d in deps if d["folder"] == "checkpoints"]
        assert "v1-5-pruned-emaonly.safetensors" in ckpts

    def test_output_nodes(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        assert "9" in schema["output_nodes"]

    def test_summary(self, sd15_workflow):
        schema = extract_schema(sd15_workflow)
        s = schema["summary"]
        assert s["has_negative_prompt"] is True
        assert s["has_seed"] is True
        assert s["is_video_workflow"] is False
        assert s["parameter_count"] > 5

    def test_flux_workflow(self, flux_workflow):
        schema = extract_schema(flux_workflow)
        # Flux uses RandomNoise for seed
        assert schema["summary"]["has_seed"] is True
        # Flux has only positive prompt (no negative encoder)
        assert schema["summary"]["has_negative_prompt"] is False

    def test_video_detected(self, video_workflow):
        schema = extract_schema(video_workflow)
        assert schema["summary"]["is_video_workflow"] is True


class TestEmbeddingDeps:
    def test_extract_from_prompt(self):
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {
                      "text": "a cat, embedding:goodvibes, embedding:art:1.2",
                      "clip": ["1", 1]
                  }},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {
                      "text": "ugly, embedding:badhands",
                      "clip": ["1", 1]
                  }},
            "3": {"class_type": "KSampler",
                  "inputs": {
                      "positive": ["6", 0], "negative": ["7", 0],
                      "model": ["1", 0], "latent_image": ["5", 0],
                      "seed": 1, "steps": 20, "cfg": 7.5,
                      "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                  }},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["3", 0]}},
        }
        schema = extract_schema(wf)
        names = [d["embedding_name"] for d in schema["embedding_dependencies"]]
        assert sorted(names) == ["art", "badhands", "goodvibes"]


class TestDuplicateDeduplication:
    def test_two_ksamplers_get_unique_names(self):
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a", "clip": ["1", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "b", "clip": ["1", 1]}},
            "3": {"class_type": "KSampler",
                  "inputs": {
                      "positive": ["6", 0], "negative": ["7", 0],
                      "model": ["1", 0], "latent_image": ["5", 0],
                      "seed": 42, "steps": 20, "cfg": 7.5,
                      "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                  }},
            "4": {"class_type": "KSampler",
                  "inputs": {
                      "positive": ["6", 0], "negative": ["7", 0],
                      "model": ["1", 0], "latent_image": ["5", 0],
                      "seed": 99, "steps": 30, "cfg": 8.0,
                      "sampler_name": "euler", "scheduler": "normal", "denoise": 0.6,
                  }},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "images": ["3", 0]}},
        }
        schema = extract_schema(wf)
        params = schema["parameters"]
        # Both seeds present with disambiguated names
        seed_keys = [k for k in params if "seed" in k]
        # Symmetric: both renamed (no bare "seed")
        assert "seed" not in params
        assert "seed_3" in params and "seed_4" in params
        assert params["seed_3"]["value"] == 42
        assert params["seed_4"]["value"] == 99
