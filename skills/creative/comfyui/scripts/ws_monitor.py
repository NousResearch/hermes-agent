#!/usr/bin/env python3
"""
ws_monitor.py — Real-time ComfyUI WebSocket monitor.

Connects to /ws and pretty-prints execution events: node start/finish, sampling
progress, cached nodes, errors. Optionally writes preview frames to disk.

Useful for:
  - Watching a long-running job in real time without parsing JSON yourself
  - Saving in-progress preview frames for video / animation workflows
  - Debugging "why is this hanging?" — see exactly which node is stuck

Usage:
    # Local — watch all jobs from this client_id
    python3 ws_monitor.py

    # Cloud — watch a specific prompt_id
    python3 ws_monitor.py --host https://cloud.comfy.org \
        --prompt-id abc-123-def

    # Save preview frames to ./previews/
    python3 ws_monitor.py --previews ./previews

Requires: websocket-client (`pip install websocket-client`).
Falls back to a clear error message when not installed.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DEFAULT_LOCAL_HOST, ENV_API_KEY, log, new_client_id, resolve_api_key, is_cloud_host,
)


# Binary frame types from ComfyUI WebSocket protocol
BINARY_PREVIEW_IMAGE = 1
BINARY_TEXT = 3
BINARY_PREVIEW_IMAGE_WITH_METADATA = 4

# Image type codes inside PREVIEW_IMAGE
IMAGE_TYPE_JPEG = 1
IMAGE_TYPE_PNG = 2

# ANSI escape codes (works on most modern terminals)
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def fmt_color(s: str, color: str, *, color_on: bool = True) -> str:
    return f"{color}{s}{RESET}" if color_on else s


def parse_binary_frame(data: bytes) -> dict | None:
    if len(data) < 8:
        return None
    type_code = struct.unpack(">I", data[0:4])[0]
    if type_code == BINARY_PREVIEW_IMAGE:
        image_type = struct.unpack(">I", data[4:8])[0]
        ext = "jpg" if image_type == IMAGE_TYPE_JPEG else "png" if image_type == IMAGE_TYPE_PNG else "bin"
        return {
            "kind": "preview",
            "image_type": image_type,
            "ext": ext,
            "image_bytes": data[8:],
        }
    if type_code == BINARY_PREVIEW_IMAGE_WITH_METADATA:
        if len(data) < 12:
            return None
        meta_len = struct.unpack(">I", data[4:8])[0]
        meta_end = 8 + meta_len
        if len(data) < meta_end:
            return None
        try:
            meta = json.loads(data[8:meta_end].decode("utf-8"))
        except Exception:
            meta = {"raw": data[8:meta_end][:200].decode("utf-8", "replace")}
        return {
            "kind": "preview_with_metadata",
            "metadata": meta,
            "image_bytes": data[meta_end:],
            "ext": "png",
        }
    if type_code == BINARY_TEXT:
        if len(data) < 8:
            return None
        nid_len = struct.unpack(">I", data[4:8])[0]
        nid_end = 8 + nid_len
        if len(data) < nid_end:
            return None
        return {
            "kind": "text",
            "node_id": data[8:nid_end].decode("utf-8", "replace"),
            "text": data[nid_end:].decode("utf-8", "replace"),
        }
    return {"kind": "unknown", "type_code": type_code, "size": len(data)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Real-time ComfyUI WebSocket monitor")
    p.add_argument("--host", default=DEFAULT_LOCAL_HOST, help="ComfyUI server URL")
    p.add_argument("--api-key", help=f"API key for cloud (or set ${ENV_API_KEY} env var)")
    p.add_argument("--client-id", default=None, help="Client ID (default: random UUID)")
    p.add_argument("--prompt-id", default=None,
                   help="Filter to a specific prompt_id (default: all jobs)")
    p.add_argument("--previews", default=None,
                   help="Directory to save in-progress preview frames")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colour")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Hard cap on monitor duration (default 600s)")
    args = p.parse_args(argv)

    try:
        import websocket  # type: ignore[import-not-found]
    except ImportError:
        print(json.dumps({
            "error": "websocket-client not installed",
            "install": "pip install websocket-client",
        }))
        return 1

    api_key = resolve_api_key(args.api_key)
    cloud = is_cloud_host(args.host)
    client_id = args.client_id or new_client_id()

    # Build WS URL preserving any base-path component (e.g. behind reverse proxy).
    parsed = urlparse(args.host if "://" in args.host else f"http://{args.host}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc
    base_path = parsed.path.rstrip("/")
    ws_url = f"{scheme}://{netloc}{base_path}/ws?clientId={client_id}"
    if cloud and api_key:
        ws_url += f"&token={api_key}"

    color_on = not args.no_color and sys.stdout.isatty()

    preview_dir = Path(args.previews).expanduser() if args.previews else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        log(f"Saving previews to {preview_dir}")

    log(f"Connecting to {ws_url} (client_id={client_id})")
    if args.prompt_id:
        log(f"Filtering messages to prompt_id={args.prompt_id}")

    ws = websocket.create_connection(ws_url, timeout=args.timeout)
    ws.settimeout(args.timeout)

    preview_counter = 0
    try:
        while True:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                log(f"Idle for {args.timeout}s — exiting")
                return 0
            if isinstance(msg, bytes):
                parsed = parse_binary_frame(msg)
                if parsed is None:
                    continue
                if parsed["kind"] in ("preview", "preview_with_metadata") and preview_dir:
                    img_bytes = parsed.get("image_bytes", b"")
                    if img_bytes:
                        ext = parsed.get("ext", "png")
                        out = preview_dir / f"preview_{preview_counter:05d}.{ext}"
                        out.write_bytes(img_bytes)
                        preview_counter += 1
                        log(f"  [preview] saved {out.name} ({len(img_bytes)} bytes)")
                continue

            try:
                payload = json.loads(msg)
            except Exception:
                continue
            mtype = payload.get("type", "")
            mdata = payload.get("data", {}) or {}
            pid = mdata.get("prompt_id")

            if args.prompt_id and pid and pid != args.prompt_id:
                continue

            if mtype == "status":
                qr = mdata.get("status", {}).get("exec_info", {}).get("queue_remaining", "?")
                print(fmt_color(f"[status] queue_remaining={qr}", DIM, color_on=color_on))
            elif mtype == "execution_start":
                print(fmt_color(f"[start] prompt_id={pid}", BOLD, color_on=color_on))
            elif mtype == "executing":
                node = mdata.get("node")
                if node:
                    print(fmt_color(f"  [executing] node={node}", CYAN, color_on=color_on))
                else:
                    print(fmt_color(f"  [executing] (workflow done) prompt_id={pid}", DIM, color_on=color_on))
            elif mtype == "progress":
                v, m = mdata.get("value", 0), mdata.get("max", 0)
                pct = (v / m * 100) if m else 0
                print(f"    [progress] {v}/{m} ({pct:5.1f}%) node={mdata.get('node')}")
            elif mtype == "progress_state":
                # Newer extended progress message
                nodes = mdata.get("nodes") or {}
                running = [k for k, v in nodes.items() if v.get("running")]
                if running:
                    print(fmt_color(f"    [progress_state] running={running}", DIM, color_on=color_on))
            elif mtype == "executed":
                node = mdata.get("node")
                out = mdata.get("output") or {}
                summary_parts = []
                for key in ("images", "video", "videos", "gifs", "audio", "files"):
                    if out.get(key):
                        summary_parts.append(f"{key}={len(out[key])}")
                summary = ", ".join(summary_parts) if summary_parts else "(no files)"
                print(fmt_color(f"  [executed] node={node} {summary}", GREEN, color_on=color_on))
            elif mtype == "execution_cached":
                cached = mdata.get("nodes") or []
                if cached:
                    print(fmt_color(f"  [cached] {len(cached)} nodes skipped", DIM, color_on=color_on))
            elif mtype == "execution_success":
                print(fmt_color(f"[success] prompt_id={pid}", GREEN + BOLD, color_on=color_on))
                if args.prompt_id:
                    return 0
            elif mtype == "execution_error":
                exc_type = mdata.get("exception_type", "?")
                exc_msg = mdata.get("exception_message", "?")
                print(fmt_color(f"[error] {exc_type}: {exc_msg}", RED + BOLD, color_on=color_on))
                tb = mdata.get("traceback")
                if tb:
                    if isinstance(tb, list):
                        for line in tb:
                            print(fmt_color(f"  {line}", RED, color_on=color_on))
                    else:
                        print(fmt_color(f"  {tb}", RED, color_on=color_on))
                if args.prompt_id:
                    return 1
            elif mtype == "execution_interrupted":
                print(fmt_color(f"[interrupted] prompt_id={pid}", YELLOW, color_on=color_on))
                if args.prompt_id:
                    return 1
            elif mtype == "notification":
                v = mdata.get("value", "")
                print(fmt_color(f"[notification] {v}", DIM, color_on=color_on))
            else:
                # Unknown / lightly-used types: print compactly
                print(fmt_color(f"[{mtype}] {json.dumps(mdata, default=str)[:200]}", DIM, color_on=color_on))

    except KeyboardInterrupt:
        log("Interrupted")
        return 130
    finally:
        try:
            ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
