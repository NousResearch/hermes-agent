#!/usr/bin/env python3
"""
Vision analyzer — uses Ollama vision model to analyze frames.
"""
from __future__ import annotations
import argparse
import base64
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Add parent directory to path for models import
sys.path.insert(0, str(Path(__file__).parent))
from models import Frame, VisionResult, write_json, print_json


def check_ollama_vision(ollama_url: str, model: str) -> bool:
    """Verify Ollama is reachable and model supports vision."""
    try:
        # Check Ollama reachable
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return False

        # Check model capabilities
        data = json.dumps({"name": model}).encode()
        req = urllib.request.Request(
            f"{ollama_url}/api/show",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return False
            model_info = json.loads(resp.read().decode())

        caps = model_info.get("capabilities", [])
        if isinstance(caps, list) and "vision" in caps:
            return True
        model_info_dict = model_info.get("model_info", {})
        if any("vision.block_count" in str(k).lower() for k in model_info_dict.keys()):
            return True

        return False
    except Exception:
        return False


def encode_image(image_path: Path) -> str:
    """Encode image as raw base64 (no data URL prefix)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_vision_prompt(frames: list[Frame], batch_start_idx: int) -> str:
    """Build the prompt for vision analysis."""
    frame_descriptions = []
    for i, frame in enumerate(frames):
        timestamp_str = format_timestamp(frame.timestamp)
        frame_descriptions.append(f"Frame {batch_start_idx + i} [{timestamp_str}]")

    frames_list = "\n".join(frame_descriptions)

    return f"""Analyze these {len(frames)} video frames with timestamps. For each frame, provide:
1. Visual description (2-3 sentences)
2. Any visible text (OCR) — UI text, code, slides, captions on screen
3. UI elements detected (buttons, menus, charts, code editor, terminal, etc.)
4. Scene type: talking_head | screen_demo | slides | b_roll | diagram | other
5. Speaker change detected? (yes/no)

Frames:
{frames_list}

Output as JSON array matching frame order:
[
  {{"frame_index": 0, "description": "...", "ocr_text": "...", "ui_elements": ["..."], "scene_type": "...", "speaker_change": false}},
  ...
]"""


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def call_ollama_vision(
    ollama_url: str,
    model: str,
    prompt: str,
    images: list[str],
    timeout: int = 120,
) -> list[VisionResult]:
    """Call Ollama /api/chat with vision images."""
    # Ollama expects images in a separate "images" array, not as content parts
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": images,
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
        },
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama returned status {resp.status}")
            response = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e.code} - {e.read().decode()}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from Ollama: {e}")

    # Parse response
    content = response.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Empty response from Ollama")

    # Extract JSON from response (might have markdown code fences)
    content = content.removeprefix("```json")
    content = content.removeprefix("```")
    content = content.removesuffix("```")
    content = content.strip()

    try:
        results = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse vision JSON: {e}\nContent: {content[:500]}")

    # Convert to VisionResult objects
    vision_results = []
    for r in results:
        vision_results.append(VisionResult(
            frame_index=r.get("frame_index", 0),
            timestamp=r.get("timestamp", 0.0),
            description=r.get("description", ""),
            ocr_text=r.get("ocr_text", ""),
            ui_elements=r.get("ui_elements", []),
            scene_type=r.get("scene_type", "unknown"),
            speaker_change=r.get("speaker_change", False),
        ))

    return vision_results


def main():
    parser = argparse.ArgumentParser(description="Analyze frames with Ollama vision model")
    parser.add_argument("--frames", required=True, help="JSON file with frames list")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--vision-model", default="qwen3-vl:4b", help="Vision model name")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per batch (seconds)")
    parser.add_argument("--batch-size", type=int, default=8, help="Frames per batch")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    # Load frames
    frames_data = json.loads(Path(args.frames).read_text())
    frames = [Frame.from_dict(f) for f in frames_data.get("frames", [])]

    if not frames:
        print("Error: No frames to analyze", file=sys.stderr)
        return 3

    # Verify Ollama and model
    if not check_ollama_vision(args.ollama_url, args.vision_model):
        print(f"Error: Ollama not reachable or model '{args.vision_model}' does not support vision", file=sys.stderr)
        return 3

    # Process in batches
    all_results = []
    batch_size = args.batch_size

    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start:batch_start + batch_size]

        # Encode images
        images = [encode_image(Path(f.path)) for f in batch]

        # Build prompt
        prompt = build_vision_prompt(batch, batch_start)

        # Call Ollama
        try:
            results = call_ollama_vision(
                ollama_url=args.ollama_url,
                model=args.vision_model,
                prompt=prompt,
                images=images,
                timeout=args.timeout,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Warning: Batch {batch_start//batch_size + 1} failed: {e}", file=sys.stderr)
            # Create placeholder results for failed batch
            for i, frame in enumerate(batch):
                all_results.append(VisionResult(
                    frame_index=batch_start + i,
                    timestamp=frame.timestamp,
                    description=f"[Vision analysis failed: {e}]",
                    scene_type="error",
                ))

    if args.json:
        print_json({"results": all_results})
    return 0


if __name__ == "__main__":
    sys.exit(main())