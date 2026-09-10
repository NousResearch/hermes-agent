#!/usr/bin/env python3
"""
Camera Frame Capture — Capture frames from any camera source.

Supports USB webcams, RTSP streams, HTTP streams, video files, and screen capture.
Requires ffmpeg to be installed.

Usage:
    python3 capture.py --source 0                          # USB webcam
    python3 capture.py --source "rtsp://user:pass@IP/..."  # IP camera
    python3 capture.py --source "http://192.168.1.50:8080/video"  # Phone camera
    python3 capture.py --source "/path/to/video.mp4"       # Video file
    python3 capture.py --source 0 --best                   # Best of 5 frames
    python3 capture.py --source 0 --timelapse 3600 60      # 1 hour, every 60s
"""

import subprocess
import sys
import argparse
import os
import json
import time


def capture(source: str, output: str = "/tmp/camera_frame.jpg", timeout: int = 10) -> dict:
    """Capture a single frame from camera source."""
    cmd = _build_ffmpeg_cmd(source, output)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
            return {"success": True, "path": output, "size": os.path.getsize(output)}
        stderr = result.stderr.decode("utf-8", errors="replace")
        return {"success": False, "error": stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Capture timed out after {timeout}s"}
    except FileNotFoundError:
        return {"success": False, "error": "ffmpeg not found. Install: apt install ffmpeg / brew install ffmpeg"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def capture_best(source: str, attempts: int = 5, output: str = "/tmp/best_frame.jpg") -> dict:
    """Capture multiple frames, return the sharpest one."""
    best_score = 0.0
    best_path = None

    for i in range(attempts):
        tmp = f"/tmp/_capture_attempt_{i}.jpg"
        result = capture(source, tmp)
        if not result.get("success"):
            continue

        score = _sharpness_score(tmp)
        if score is not None and score > best_score:
            best_score = score
            best_path = tmp

    if best_path is None:
        return {"success": False, "error": "All capture attempts failed"}

    # Copy best frame to output path
    import shutil
    shutil.copy(best_path, output)

    # Clean up temp files
    for i in range(attempts):
        tmp = f"/tmp/_capture_attempt_{i}.jpg"
        if os.path.exists(tmp) and tmp != best_path:
            try:
                os.remove(tmp)
            except OSError:
                pass

    return {"success": True, "path": output, "size": os.path.getsize(output), "sharpness": round(best_score, 1)}


def capture_timelapse(source: str, duration_s: int, interval_s: int, output_dir: str = "/tmp/timelapse/") -> dict:
    """Capture frames at regular intervals for timelapse."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration_s:
        frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        capture(source, frame_path)
        frame_count += 1

        elapsed = time.time() - start_time
        remaining = duration_s - elapsed
        if remaining > interval_s:
            time.sleep(interval_s)
        elif remaining > 0:
            time.sleep(remaining)
        else:
            break

    return {"success": True, "frames": frame_count, "dir": output_dir, "duration_s": duration_s}


def _sharpness_score(image_path: str):
    """Calculate sharpness score using Laplacian variance. Returns None if unavailable."""
    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except ImportError:
        return None


def _build_ffmpeg_cmd(source: str, output: str) -> list:
    """Build ffmpeg command for the given camera source."""
    base = ["ffmpeg", "-frames:v", "1", output, "-y"]

    if source == "screen":
        if sys.platform == "darwin":
            return ["ffmpeg", "-f", "avfoundation", "-i", "1:none"] + base
        elif sys.platform.startswith("linux"):
            display = os.environ.get("DISPLAY", ":0")
            return ["ffmpeg", "-f", "x11grab", "-i", display] + base
        else:
            return ["ffmpeg", "-f", "gdigrab", "-i", "desktop"] + base

    elif source.isdigit():
        device_num = int(source)
        if sys.platform == "darwin":
            return ["ffmpeg", "-f", "avfoundation", "-i", f"{device_num}:none"] + base
        elif sys.platform.startswith("linux"):
            return ["ffmpeg", "-f", "v4l2", "-i", f"/dev/video{device_num}"] + base
        else:
            return ["ffmpeg", "-f", "dshow", "-i", "video=Integrated Camera"] + base

    elif source.startswith("rtsp://"):
        return ["ffmpeg", "-rtsp_transport", "tcp", "-i", source] + base

    elif source.startswith(("http://", "https://")):
        return ["ffmpeg", "-i", source] + base

    else:
        return ["ffmpeg", "-i", source] + base


def main():
    parser = argparse.ArgumentParser(description="Capture frame from camera source")
    parser.add_argument("-s", "--source", default="0",
                        help="Camera source: device number, RTSP/HTTP URL, 'screen', or file path (default: 0)")
    parser.add_argument("-o", "--output", default="/tmp/camera_frame.jpg",
                        help="Output file path (default: /tmp/camera_frame.jpg)")
    parser.add_argument("-t", "--timeout", type=int, default=10,
                        help="Capture timeout in seconds (default: 10)")
    parser.add_argument("--best", action="store_true",
                        help="Capture 5 frames and return the sharpest one")
    parser.add_argument("--timelapse", nargs=2, type=int, metavar=("DURATION", "INTERVAL"),
                        help="Timelapse mode: capture every INTERVAL seconds for DURATION seconds")
    parser.add_argument("--output-dir", default="/tmp/timelapse/",
                        help="Output directory for timelapse frames")
    args = parser.parse_args()

    if args.timelapse:
        result = capture_timelapse(args.source, args.timelapse[0], args.timelapse[1], args.output_dir)
    elif args.best:
        result = capture_best(args.source, output=args.output)
    else:
        result = capture(args.source, args.output, args.timeout)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
