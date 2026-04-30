#!/usr/bin/env python3
"""
extract_schema.py — Analyze a ComfyUI API-format workflow and extract controllable parameters.

Reads a workflow JSON, identifies user-facing parameters (prompts, seed, dimensions, etc.)
by scanning node types and field names, and outputs a schema mapping.

Usage:
    python3 extract_schema.py workflow_api.json
    python3 extract_schema.py workflow_api.json --output schema.json

Output format:
    {
      "parameters": {
        "prompt": {"node_id": "6", "field": "text", "type": "string", "value": "..."},
        "seed": {"node_id": "3", "field": "seed", "type": "int", "value": 42},
        ...
      },
      "output_nodes": ["9"],
      "model_dependencies": [
        {"node_id": "4", "class_type": "CheckpointLoaderSimple", "field": "ckpt_name", "value": "..."}
      ]
    }

Requires: Python 3.10+ (stdlib only)
"""

import json
import sys
import argparse
from pathlib import Path

# Known parameter patterns: (class_type, field_name) → friendly_name
PARAM_PATTERNS = [
    # Prompts
    ("CLIPTextEncode", "text", "prompt"),
    ("CLIPTextEncodeSDXL", "text_g", "prompt"),
    ("CLIPTextEncodeSDXL", "text_l", "prompt_l"),
    # Sampling
    ("KSampler", "seed", "seed"),
    ("KSampler", "steps", "steps"),
    ("KSampler", "cfg", "cfg"),
    ("KSampler", "sampler_name", "sampler_name"),
    ("KSampler", "scheduler", "scheduler"),
    ("KSampler", "denoise", "denoise"),
    ("KSamplerAdvanced", "noise_seed", "seed"),
    ("KSamplerAdvanced", "steps", "steps"),
    ("KSamplerAdvanced", "cfg", "cfg"),
    ("KSamplerAdvanced", "sampler_name", "sampler_name"),
    ("KSamplerAdvanced", "scheduler", "scheduler"),
    # Dimensions
    ("EmptyLatentImage", "width", "width"),
    ("EmptyLatentImage", "height", "height"),
    ("EmptyLatentImage", "batch_size", "batch_size"),
    # Image input
    ("LoadImage", "image", "image"),
    ("LoadImageMask", "image", "mask_image"),
    # LoRA
    ("LoraLoader", "lora_name", "lora_name"),
    ("LoraLoader", "strength_model", "lora_strength"),
    # Output
    ("SaveImage", "filename_prefix", "filename_prefix"),
]

# Node types that produce output files
OUTPUT_NODES = {"SaveImage", "PreviewImage", "VHS_VideoCombine", "SaveAudio", "SaveAnimatedWEBP", "SaveAnimatedPNG"}

# Node types that load models (for dependency checking)
MODEL_LOADERS = {
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "CheckpointLoader": ("ckpt_name", "checkpoints"),
    "LoraLoader": ("lora_name", "loras"),
    "LoraLoaderModelOnly": ("lora_name", "loras"),
    "VAELoader": ("vae_name", "vae"),
    "ControlNetLoader": ("control_net_name", "controlnet"),
    "CLIPLoader": ("clip_name", "clip"),
    "DualCLIPLoader": ("clip_name1", "clip"),
    "UNETLoader": ("unet_name", "unet"),
    "DiffusionModelLoader": ("model_name", "diffusion_models"),
    "UpscaleModelLoader": ("model_name", "upscale_models"),
    "CLIPVisionLoader": ("clip_name", "clip_vision"),
}


def validate_api_format(workflow: dict) -> bool:
    """Check if workflow is in API format (not editor format)."""
    if "nodes" in workflow and "links" in workflow:
        return False
    # API format: top-level keys are node IDs, each has class_type
    for node_id, node in workflow.items():
        if isinstance(node, dict) and "class_type" in node:
            return True
    return False


def infer_type(value) -> str:
    """Infer JSON schema type from a Python value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "link"  # connections to other nodes
    return "unknown"


def extract_schema(workflow: dict) -> dict:
    """Extract controllable parameters from a workflow."""
    parameters = {}
    output_nodes = []
    model_deps = []
    name_counts = {}  # track duplicate friendly names

    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue

        class_type = node["class_type"]
        inputs = node.get("inputs", {})
        meta_title = node.get("_meta", {}).get("title", "")

        # Check if this is an output node
        if class_type in OUTPUT_NODES:
            output_nodes.append(node_id)

        # Check if this is a model loader
        if class_type in MODEL_LOADERS:
            field, folder = MODEL_LOADERS[class_type]
            if field in inputs and isinstance(inputs[field], str):
                model_deps.append({
                    "node_id": node_id,
                    "class_type": class_type,
                    "field": field,
                    "value": inputs[field],
                    "folder": folder,
                })

        # Extract controllable parameters
        for pattern_class, pattern_field, friendly_name in PARAM_PATTERNS:
            if class_type != pattern_class:
                continue
            if pattern_field not in inputs:
                continue
            value = inputs[pattern_field]
            val_type = infer_type(value)
            if val_type == "link":
                continue  # skip linked inputs — not directly controllable

            # Disambiguate duplicate friendly names
            # Use title hint for prompt fields
            actual_name = friendly_name
            if friendly_name == "prompt" and meta_title:
                title_lower = meta_title.lower()
                if "negative" in title_lower or "neg" in title_lower:
                    actual_name = "negative_prompt"

            # Handle remaining duplicates by appending node_id
            if actual_name in name_counts:
                name_counts[actual_name] += 1
                actual_name = f"{actual_name}_{node_id}"
            else:
                name_counts[actual_name] = 1

            parameters[actual_name] = {
                "node_id": node_id,
                "field": pattern_field,
                "type": val_type,
                "value": value,
            }

    return {
        "parameters": parameters,
        "output_nodes": output_nodes,
        "model_dependencies": model_deps,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract controllable parameters from a ComfyUI workflow")
    parser.add_argument("workflow", help="Path to workflow API JSON file")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"Error: {workflow_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(workflow_path) as f:
        workflow = json.load(f)

    if not validate_api_format(workflow):
        print("Error: Workflow is in editor format, not API format.", file=sys.stderr)
        print("Re-export from ComfyUI using 'Save (API Format)' button.", file=sys.stderr)
        sys.exit(1)

    schema = extract_schema(workflow)

    output_json = json.dumps(schema, indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Schema written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
