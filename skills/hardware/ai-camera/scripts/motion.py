#!/usr/bin/env python3
"""
Motion Detection — Compare two frames to detect motion.

Lightweight pre-filter to reduce AI API calls. Only triggers
analysis when significant motion is detected between frames.

Usage:
    python3 motion.py /tmp/current.jpg /tmp/previous.jpg
    python3 motion.py /tmp/current.jpg /tmp/previous.jpg --threshold 20
    python3 motion.py /tmp/current.jpg  # First frame, always returns motion=true

Output:
    JSON: {"motion": true/false, "score": 47.2, "threshold": 15.0, "note": "..."}

Exit codes:
    0 = motion detected (proceed with AI analysis)
    1 = no motion (skip AI analysis)
    2 = error (proceed with AI analysis to be safe)
"""

import sys
import json
import argparse


def detect_motion(current_path: str, previous_path: str = None, threshold: float = 15.0) -> dict:
    """
    Compare two frames and return motion detection result.

    Args:
        current_path: Path to current frame image
        previous_path: Path to previous frame (None for first frame)
        threshold: Motion sensitivity (lower = more sensitive, default: 15.0)

    Returns:
        dict with 'motion' (bool), 'score' (float), 'threshold' (float), 'note' (str)
    """
    try:
        import numpy as np
    except ImportError:
        # NumPy not available — always assume motion (safe fallback)
        return {
            "motion": True,
            "score": -1,
            "threshold": threshold,
            "note": "numpy not installed — always assuming motion (pip install numpy)"
        }

    try:
        from PIL import Image
        curr_gray = np.array(Image.open(current_path).convert("L"), dtype=np.float32)
    except ImportError:
        # PIL not available — try with ffmpeg raw output
        curr_gray = _load_frame_ffmpeg(current_path)
        if curr_gray is None:
            return {"motion": True, "score": -1, "threshold": threshold, "note": "Cannot read current frame"}

    if curr_gray is None:
        return {"motion": True, "score": -1, "threshold": threshold, "note": "Cannot read current frame"}

    if previous_path is None:
        return {"motion": True, "score": 0, "threshold": threshold, "note": "First frame — no comparison"}

    try:
        from PIL import Image
        prev_gray = np.array(Image.open(previous_path).convert("L"), dtype=np.float32)
    except ImportError:
        prev_gray = _load_frame_ffmpeg(previous_path)

    if prev_gray is None:
        return {"motion": True, "score": 0, "threshold": threshold, "note": "Cannot read previous frame"}

    # Resize if dimensions don't match
    if prev_gray.shape != curr_gray.shape:
        from PIL import Image as PILImage
        try:
            curr_pil = PILImage.fromarray(curr_gray.astype("uint8"))
            curr_pil = curr_pil.resize((prev_gray.shape[1], prev_gray.shape[0]))
            curr_gray = np.array(curr_pil, dtype=np.float32)
        except Exception:
            # Simple nearest-neighbor resize without PIL
            row_scale = prev_gray.shape[0] / curr_gray.shape[0]
            col_scale = prev_gray.shape[1] / curr_gray.shape[1]
            row_idx = (np.arange(prev_gray.shape[0]) / row_scale).astype(int).clip(0, curr_gray.shape[0] - 1)
            col_idx = (np.arange(prev_gray.shape[1]) / col_scale).astype(int).clip(0, curr_gray.shape[1] - 1)
            curr_gray = curr_gray[np.ix_(row_idx, col_idx)]

    # Compute absolute difference
    diff = np.abs(prev_gray - curr_gray)

    # Raw score (mean pixel difference)
    raw_score = float(np.mean(diff))

    # Gaussian-blurred score (reduces sensor noise, more reliable)
    try:
        from scipy.ndimage import gaussian_filter
        blur_diff = gaussian_filter(diff, sigma=5)
        blur_score = float(np.mean(blur_diff))
    except ImportError:
        blur_score = raw_score

    # Use the more conservative (lower) blur score for decision
    decision_score = blur_score if blur_score > 0 else raw_score
    detected = decision_score > threshold

    return {
        "motion": detected,
        "score": round(raw_score, 2),
        "blur_score": round(blur_score, 2),
        "threshold": threshold,
        "note": "Motion detected" if detected else "No significant motion"
    }


def _load_frame_ffmpeg(image_path: str):
    """Load a grayscale frame using ffmpeg raw output (fallback when PIL unavailable)."""
    import subprocess, os
    import numpy as np

    raw_path = image_path + ".raw"
    try:
        cmd = [
            "ffmpeg", "-i", image_path,
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", "320x240", raw_path, "-y"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        if result.returncode != 0 or not os.path.exists(raw_path):
            return None

        with open(raw_path, "rb") as f:
            data = f.read()
        os.remove(raw_path)

        if len(data) != 320 * 240:
            return None

        return np.frombuffer(data, dtype=np.uint8).reshape(240, 320).astype(np.float32)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Detect motion between camera frames")
    parser.add_argument("current", help="Path to current frame image")
    parser.add_argument("previous", nargs="?", default=None, help="Path to previous frame image (omit for first frame)")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Motion sensitivity threshold (default: 15.0, lower = more sensitive)")
    args = parser.parse_args()

    result = detect_motion(args.current, args.previous, args.threshold)
    print(json.dumps(result, indent=2))

    # Exit code: 0 = motion, 1 = no motion, 2 = error
    if result.get("score", -1) < 0:
        sys.exit(2)
    sys.exit(0 if result["motion"] else 1)


if __name__ == "__main__":
    main()
