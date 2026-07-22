#!/usr/bin/env python3
"""
Video Editor MCP Server
=======================

11 tools for conversational video editing via FFmpeg + faster-whisper.
Exposes all capabilities as an MCP stdio server.

Based on the original video_editor_tool.py by @rafaumeu (PR #61293),
re-scoped from core builtin tool to MCP catalog entry per AGENTS.md
Footprint Ladder (level 5: MCP server in catalog).

Requirements:
  - ffmpeg and ffprobe on PATH (>= 6.0)
  - faster-whisper installed (pip install faster-whisper)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any

# ── MCP SDK (minimal stdio server, no external package dependency) ──────────

class MCPServer:
    """Minimal MCP stdio server — handles JSON-RPC handshake, tools/list, tools/call."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self._tools: dict[str, dict] = {}

    def register(self, name: str, description: str, schema: dict, handler):
        self._tools[name] = {"description": description, "schema": schema, "handler": handler}

    def _send(self, msg: dict):
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def _recv(self) -> dict | None:
        line = sys.stdin.readline()
        if not line:
            return None
        return json.loads(line)

    def run(self):
        while True:
            msg = self._recv()
            if msg is None:
                break
            method = msg.get("method", "")
            msg_id = msg.get("id")
            params = msg.get("params", {})

            if method == "initialize":
                self._send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": self.version},
                }})
            elif method == "notifications/initialized":
                continue  # no response needed
            elif method == "tools/list":
                tools = []
                for name, spec in self._tools.items():
                    tools.append({
                        "name": name,
                        "description": spec["description"],
                        "inputSchema": spec["schema"],
                    })
                self._send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}})
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})
                if tool_name not in self._tools:
                    self._send({"jsonrpc": "2.0", "id": msg_id, "result": {
                        "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                        "isError": True,
                    }})
                    continue
                try:
                    result = self._tools[tool_name]["handler"](args)
                    self._send({"jsonrpc": "2.0", "id": msg_id, "result": {
                        "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                    }})
                except Exception as e:
                    self._send({"jsonrpc": "2.0", "id": msg_id, "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    }})


# ── FFmpeg helpers ──────────────────────────────────────────────────────────

def _run_ffmpeg(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg"] + args
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (exit {result.returncode}): {result.stderr[-500:]}")
    return result


def _run_ffprobe(args: list[str]) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {result.stderr[-500:]}")
    return json.loads(result.stdout)


def _ensure_ffmpeg() -> None:
    for cmd in ("ffmpeg", "ffprobe"):
        if not shutil.which(cmd):
            raise RuntimeError(f"{cmd} not found on PATH. Install FFmpeg >= 6.0.")


def _probe_video(input_path: str) -> dict:
    _ensure_ffmpeg()
    data = _run_ffprobe(["-show_format", "-show_streams", input_path])
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fps_str = vs.get("r_frame_rate", "0/1")
    try:
        fps = float(fps_str) if "/" not in fps_str else eval(fps_str)
    except Exception:
        fps = 0
    return {
        "duration": float(fmt.get("duration", 0)),
        "size": int(fmt.get("size", 0)),
        "bit_rate": int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else 0,
        "video": {
            "codec": vs.get("codec_name", ""),
            "width": vs.get("width", 0),
            "height": vs.get("height", 0),
            "fps": fps,
        } if vs else None,
        "audio": {
            "codec": audio.get("codec_name", ""),
            "sample_rate": int(audio.get("sample_rate", 0)),
            "channels": audio.get("channels", 0),
        } if audio else None,
    }


def _parse_timecode(tc: str | float | int) -> float:
    if isinstance(tc, (int, float)):
        return float(tc)
    import re
    tc = tc.strip()
    m = re.match(r"^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", tc)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    m = re.match(r"^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", tc)
    if m:
        return int(m[1]) * 60 + float(m[2])
    m = re.match(r"^(\d+(?:\.\d+)?)$", tc)
    if m:
        return float(m[1])
    raise ValueError(f"Invalid timecode: {tc}")


def _output_path(input_path: str, suffix: str, extension: str = ".mp4") -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}{suffix}{extension}"


# ── Tool handlers (8 FFmpeg-native + 3 self-contained, no external deps) ───

# 1. video_probe
def handle_probe(kwargs: dict) -> dict:
    return _probe_video(kwargs["input_path"])

# 2. video_transcribe
def handle_transcribe(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    from faster_whisper import WhisperModel
    model_name = kwargs.get("model", "base")
    lang = None if kwargs.get("language", "auto") == "auto" else kwargs["language"]
    word_ts = kwargs.get("word_timestamps", True)
    model_instance = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model_instance.transcribe(input_path, language=lang, word_timestamps=word_ts, vad_filter=True)
    seg_list = []
    for seg in segments:
        d = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        if word_ts and seg.words:
            d["words"] = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words]
        seg_list.append(d)
    return {"language": info.language, "language_probability": info.language_probability, "duration": info.duration, "segments": seg_list}

# 3. video_cut
def handle_cut(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    start = _parse_timecode(kwargs["start"])
    end = _parse_timecode(kwargs["end"])
    if end <= start:
        raise ValueError(f"End ({end}) must be after start ({start})")
    output = kwargs.get("output_path") or _output_path(input_path, "_cut")
    reencode = kwargs.get("reencode", False)
    if reencode:
        cmd = ["-ss", str(start), "-i", input_path, "-t", str(end - start), "-c:v", "libx264", "-c:a", "aac", "-y", output]
    else:
        cmd = ["-ss", str(start), "-i", input_path, "-t", str(end - start), "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", output]
    _run_ffmpeg(cmd)
    return {"output_path": output, "start": start, "end": end, "duration": end - start, "reencoded": reencode}

# 4. video_overlay
def handle_overlay(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    video_path = kwargs["video_path"]
    image_path = kwargs["image_path"]
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    output = kwargs.get("output_path") or _output_path(video_path, "_overlay")
    position = kwargs.get("position", "center")
    start = _parse_timecode(kwargs.get("start", "0"))
    vi = _probe_video(video_path)
    end = _parse_timecode(kwargs.get("end", str(vi["duration"]))) if kwargs.get("end") else vi["duration"]
    fade_in = kwargs.get("fade_in", 0)
    fade_out = kwargs.get("fade_out", 0)
    scale = kwargs.get("scale", "original")
    opacity = kwargs.get("opacity", 1.0)
    pos_map = {
        "top-left": "10:10", "top-right": "main_w-overlay_w-10:10",
        "bottom-left": "10:main_h-overlay_h-10", "bottom-right": "main_w-overlay_w-10:main_h-overlay_h-10",
        "center": "(main_w-overlay_w)/2:(main_h-overlay_h)/2",
    }
    pos = pos_map.get(position, position)
    filters = []
    overlay_input = "[1:v]"
    if scale != "original":
        filters.append(f"[1:v]scale={scale}[overlay_scaled]")
        overlay_input = "[overlay_scaled]"
    if opacity < 1.0:
        filters.append(f"{overlay_input}format=rgba,colorchannelmixer=aa={opacity}[overlay_opacity]")
        overlay_input = "[overlay_opacity]"
    if fade_in > 0:
        filters.append(f"{overlay_input}fade=type=in:start_time={start}:duration={fade_in}[fade_in]")
        overlay_input = "[fade_in]"
    if fade_out > 0:
        filters.append(f"{overlay_input}fade=type=out:start_time={end - fade_out}:duration={fade_out}[fade_out]")
        overlay_input = "[fade_out]"
    overlay_filter = f"[0:v]{overlay_input}overlay={pos}:enable='between(t,{start},{end})'"
    filters.append(overlay_filter)
    filtergraph = ";".join(filters)
    _run_ffmpeg(["-i", video_path, "-i", image_path, "-filter_complex", filtergraph, "-c:a", "copy", "-y", output])
    return {"output_path": output, "position": position, "start": start, "end": end}

# 5. video_captions
def handle_captions(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    video_path = kwargs["video_path"]
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    output = kwargs.get("output_path") or _output_path(video_path, "_captioned")
    segments = kwargs.get("segments")
    srt_content = kwargs.get("srt_content")
    if not segments and not srt_content:
        raise ValueError("Must provide segments or srt_content")
    if not srt_content:
        lines = []
        for i, seg in enumerate(segments, 1):
            h = int(seg["start"] // 3600)
            m = int((seg["start"] % 3600) // 60)
            s = int(seg["start"] % 60)
            ms = int((seg["start"] % 1) * 1000)
            start_srt = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            h2 = int(seg["end"] // 3600)
            m2 = int((seg["end"] % 3600) // 60)
            s2 = int(seg["end"] % 60)
            ms2 = int((seg["end"] % 1) * 1000)
            end_srt = f"{h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}"
            lines.append(f"{i}\n{start_srt} --> {end_srt}\n{seg['text'].strip()}\n")
        srt_content = "\n".join(lines)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
        f.write(srt_content)
        srt_path = f.name
    try:
        style = kwargs.get("style", "karaoke")
        if style == "karaoke":
            font_size = int(480 * kwargs.get("font_size", 0.06))
            force_style = f"FontName=DejaVu Sans,FontSize={max(12,font_size)},PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,MarginV=20"
            _run_ffmpeg(["-i", video_path, "-vf", f"subtitles={srt_path}:force_style='{force_style}'", "-c:a", "copy", "-y", output])
        else:
            _run_ffmpeg(["-i", video_path, "-vf", f"subtitles={srt_path}", "-c:a", "copy", "-y", output])
    finally:
        os.unlink(srt_path)
    return {"output_path": output, "style": style}

# 6. video_zoom
def handle_zoom(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    output = kwargs.get("output_path") or _output_path(input_path, "_zoom")
    effect = kwargs.get("effect", "zoom_in")
    zoom_start = kwargs.get("zoom_start", 1.0)
    zoom_end = kwargs.get("zoom_end", 1.5)
    vi = _probe_video(input_path)
    vw, vh = vi["video"]["width"], vi["video"]["height"]
    fps = vi["video"]["fps"]
    total_frames = int(vi["duration"] * fps)
    z = f"min({zoom_start}+(({zoom_end}-{zoom_start})*in/{total_frames}),10)"
    x = "(iw-iw/zoom)/2"
    y = "(ih-ih/zoom)/2"
    if effect == "zoom_out":
        z = f"min({zoom_end}+(({zoom_start}-{zoom_end})*in/{total_frames}),10)"
    elif effect in ("pan_left", "ken_burns"):
        x = f"max(0,(iw-iw/zoom)*(1-in/{total_frames}))"
    elif effect == "pan_right":
        x = f"max(0,(iw-iw/zoom)*(in/{total_frames}))"
    elif effect == "pan_up":
        y = f"max(0,(ih-ih/zoom)*(1-in/{total_frames}))"
    elif effect == "pan_down":
        y = f"max(0,(ih-ih/zoom)*(in/{total_frames}))"
    _run_ffmpeg(["-i", input_path, "-vf", f"zoompan=zoom='{z}':x='{x}':y='{y}':d={total_frames}:s={vw}x{vh}:fps={fps},scale={vw}:{vh}", "-c:a", "copy", "-y", output])
    return {"output_path": output, "effect": effect, "zoom_start": zoom_start, "zoom_end": zoom_end}

# 7. video_split_screen
def handle_split_screen(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    paths = kwargs["input_paths"]
    if len(paths) < 2:
        raise ValueError("Need at least 2 input videos")
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Input not found: {p}")
    output = kwargs.get("output_path") or _output_path(paths[0], "_split")
    layout = kwargs.get("layout", "side_by_side")
    cmd = []
    for p in paths:
        cmd.extend(["-i", p])
    filters = []
    for i in range(len(paths)):
        filters.append(f"[{i}:v]scale=640:480,setsar=1[v{i}]")
    if layout == "side_by_side":
        filters.append("[v0][v1]hstack=inputs=2[outv]")
        filters.append("[0:a][1:a]amix=inputs=2:duration=shortest[outa]")
    elif layout == "top_bottom":
        filters.append("[v0][v1]vstack=inputs=2[outv]")
        filters.append("[0:a][1:a]amix=inputs=2:duration=shortest[outa]")
    elif layout == "picture_in_picture":
        filters.append("[v1]scale=213:160[pip]")
        filters.append("[v0][pip]overlay=427:310:shortest=1[outv]")
        filters.append("[0:a][1:a]amix=inputs=2:duration=shortest[outa]")
    elif layout == "grid_2x2" and len(paths) == 4:
        filters.append("[v0][v1]hstack=inputs=2[top]")
        filters.append("[v2][v3]hstack=inputs=2[bottom]")
        filters.append("[top][bottom]vstack=inputs=2[outv]")
        filters.append("[0:a][1:a][2:a][3:a]amix=inputs=4:duration=shortest[outa]")
    else:
        raise ValueError(f"Unsupported layout '{layout}' for {len(paths)} inputs")
    filter_complex = ";".join(filters)
    cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]", "-y", output])
    _run_ffmpeg(cmd)
    return {"output_path": output, "layout": layout, "inputs": len(paths)}

# 8. video_render
def handle_render(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    input_path = kwargs["input_path"]
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    output = kwargs.get("output_path") or _output_path(input_path, "_final")
    resolution = kwargs.get("resolution", "original")
    video_codec = kwargs.get("video_codec", "libx264")
    audio_codec = kwargs.get("audio_codec", "aac")
    crf = kwargs.get("crf", 23)
    preset = kwargs.get("preset", "medium")
    normalize = kwargs.get("normalize_audio", True)
    if not 0 <= crf <= 51:
        raise ValueError(f"CRF must be between 0 and 51, got {crf}")
    cmd = ["-i", input_path]
    vfilters = []
    if resolution != "original":
        parts = resolution.split("x")
        if len(parts) == 2:
            vfilters.append(f"scale={parts[0]}:{parts[1]}")
    if vfilters:
        cmd.extend(["-vf", ",".join(vfilters)])
    if normalize and audio_codec != "copy":
        cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    if video_codec == "copy":
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(["-c:v", video_codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"])
    if audio_codec == "copy":
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-c:a", audio_codec, "-b:a", "192k"])
    cmd.extend(["-movflags", "+faststart", "-y", output])
    _run_ffmpeg(cmd)
    out_info = _probe_video(output)
    return {"output_path": output, "resolution": f"{out_info['video']['width']}x{out_info['video']['height']}", "video_codec": video_codec, "audio_codec": audio_codec, "crf": crf, "preset": preset}

# 9. video_transition (self-contained — no external hermes_video_mcp needed)
def handle_transition(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    paths = kwargs["input_paths"]
    if not paths or len(paths) > 2:
        raise ValueError("Provide 1 or 2 video paths")
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Input not found: {p}")
    ttype = kwargs.get("transition_type", "fade_in")
    duration = kwargs.get("duration", 1.0)
    if len(paths) == 1:
        output = _output_path(paths[0], "_transition")
        if ttype in ("fade_in", "fade_from_black"):
            vf = f"fade=type=in:start_time=0:duration={duration}"
        elif ttype in ("fade_out", "fade_to_black"):
            vi = _probe_video(paths[0])
            vf = f"fade=type=out:start_time={vi['duration']-duration}:duration={duration}"
        else:
            raise ValueError(f"Single-video transition must be fade_in/fade_out, got {ttype}")
        _run_ffmpeg(["-i", paths[0], "-vf", vf, "-c:a", "copy", "-y", output])
    else:
        output = _output_path(paths[0], "_transition")
        if ttype == "crossfade":
            vi = _probe_video(paths[0])
            offset = vi["duration"] - duration
            _run_ffmpeg(["-i", paths[0], "-i", paths[1], "-filter_complex",
                         f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset={offset}[v];"
                         f"[0:a][1:a]acrossfade=d={duration}[a]",
                         "-map", "[v]", "-map", "[a]", "-y", output])
        elif ttype == "wipe_left":
            vi = _probe_video(paths[0])
            offset = vi["duration"] - duration
            _run_ffmpeg(["-i", paths[0], "-i", paths[1], "-filter_complex",
                         f"[0:v][1:v]xfade=transition=wipeleft:duration={duration}:offset={offset}[v];"
                         f"[0:a][1:a]acrossfade=d={duration}[a]",
                         "-map", "[v]", "-map", "[a]", "-y", output])
        elif ttype == "wipe_right":
            vi = _probe_video(paths[0])
            offset = vi["duration"] - duration
            _run_ffmpeg(["-i", paths[0], "-i", paths[1], "-filter_complex",
                         f"[0:v][1:v]xfade=transition=wiperight:duration={duration}:offset={offset}[v];"
                         f"[0:a][1:a]acrossfade=d={duration}[a]",
                         "-map", "[v]", "-map", "[a]", "-y", output])
        else:
            raise ValueError(f"Two-video transition must be crossfade/wipe_left/wipe_right, got {ttype}")
    return {"output_path": output, "transition_type": ttype, "duration": duration}

# 10. video_audio_mix (self-contained)
def handle_audio_mix(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    video_path = kwargs["video_path"]
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    mode = kwargs.get("mode", "normalize")
    output = _output_path(video_path, f"_{mode}")
    if mode == "normalize":
        _run_ffmpeg(["-i", video_path, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:v", "copy", "-y", output])
    elif mode == "mute":
        _run_ffmpeg(["-i", video_path, "-an", "-c:v", "copy", "-y", output])
    elif mode == "extract":
        output = _output_path(video_path, "_audio", ".mp3")
        _run_ffmpeg(["-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", "-y", output])
    elif mode == "adjust_volume":
        vol = kwargs.get("volume", 1.0)
        _run_ffmpeg(["-i", video_path, "-af", f"volume={vol}", "-c:v", "copy", "-y", output])
    elif mode == "replace":
        audio_path = kwargs["audio_path"]
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        _run_ffmpeg(["-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "1:a", "-shortest", "-y", output])
    elif mode == "add_background":
        audio_path = kwargs["audio_path"]
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        bg_vol = kwargs.get("background_volume", 0.3)
        duck = kwargs.get("duck", False)
        if duck:
            af = f"[1:a]volume={bg_vol}[bg];[0:a][bg]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=50[out]"
            _run_ffmpeg(["-i", video_path, "-i", audio_path, "-filter_complex", af, "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-y", output])
        else:
            af = f"[1:a]volume={bg_vol}[bg];[0:a][bg]amix=inputs=2:duration=first[out]"
            _run_ffmpeg(["-i", video_path, "-i", audio_path, "-filter_complex", af, "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-y", output])
    else:
        raise ValueError(f"Unknown audio mode: {mode}")
    return {"output_path": output, "mode": mode}

# 11. video_watermark (self-contained)
def handle_watermark(kwargs: dict) -> dict:
    _ensure_ffmpeg()
    video_path = kwargs["video_path"]
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    mode = kwargs.get("mode", "text")
    output = _output_path(video_path, "_wm")
    pos = kwargs.get("position", "bottom-right")
    opacity = kwargs.get("opacity", 0.5)
    pos_map = {
        "top-left": "10:10", "top-right": "main_w-text_w-10:10",
        "bottom-left": "10:main_h-text_h-10", "bottom-right": "main_w-text_w-10:main_h-text_h-10",
        "center": "(main_w-text_w)/2:(main_h-text_h)/2",
    }
    pos_str = pos_map.get(pos, pos)
    if mode == "text":
        text = kwargs.get("text", "")
        if not text:
            raise ValueError("text required for text watermark")
        font_size = kwargs.get("font_size", 24)
        font_color = kwargs.get("font_color", "#FFFFFF")
        drawtext = f"drawtext=text='{text}':fontcolor={font_color}:fontsize={font_size}:x={pos_str}:y={pos_str}:alpha={opacity}"
        _run_ffmpeg(["-i", video_path, "-vf", drawtext, "-c:a", "copy", "-y", output])
    elif mode == "image":
        image_path = kwargs["image_path"]
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        scale = kwargs.get("scale")
        img_filter = "[1:v]"
        filters_pre = []
        if scale:
            filters_pre.append(f"[1:v]scale={scale}[scaled]")
            img_filter = "[scaled]"
        filters_pre.append(f"{img_filter}format=rgba,colorchannelmixer=aa={opacity}[wm]")
        img_filter = "[wm]"
        filters_pre.append(f"[0:v]{img_filter}overlay={pos_str}[out]")
        fg = ";".join(filters_pre)
        _run_ffmpeg(["-i", video_path, "-i", image_path, "-filter_complex", fg, "-map", "[out]", "-map", "0:a?", "-y", output])
    else:
        raise ValueError(f"Unknown watermark mode: {mode}")
    return {"output_path": output, "mode": mode}


# ── Tool schemas (same as original PR, unchanged) ───────────────────────────

_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"input_path": {"type": "string", "description": "Path to video file"}},
    "required": ["input_path"],
}

_TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to video/audio file"},
        "model": {"type": "string", "description": "Whisper model: tiny, base, small, medium, large-v3", "default": "base"},
        "language": {"type": "string", "description": "Language code or 'auto'", "default": "auto"},
        "word_timestamps": {"type": "boolean", "description": "Include word-level timestamps", "default": True},
    },
    "required": ["input_path"],
}

_CUT_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to input video file"},
        "start": {"type": "string", "description": "Start time (HH:MM:SS.mmm or seconds)"},
        "end": {"type": "string", "description": "End time (HH:MM:SS.mmm or seconds)"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "reencode": {"type": "boolean", "description": "Re-encode instead of stream copy", "default": False},
    },
    "required": ["input_path", "start", "end"],
}

_OVERLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "Path to base video file"},
        "image_path": {"type": "string", "description": "Path to overlay image (PNG with alpha recommended)"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "position": {"type": "string", "description": "Position: top-left, top-right, bottom-left, bottom-right, center, or x:y", "default": "center"},
        "start": {"type": "string", "description": "Start time for overlay", "default": "0"},
        "end": {"type": "string", "description": "End time (overlay end)"},
        "fade_in": {"type": "number", "description": "Fade in duration in seconds", "default": 0},
        "fade_out": {"type": "number", "description": "Fade out duration in seconds", "default": 0},
        "scale": {"type": "string", "description": "Scale overlay: 'width:height' or 'iw*0.5:ih*0.5'", "default": "original"},
        "opacity": {"type": "number", "description": "Overlay opacity 0.0-1.0", "default": 1.0},
    },
    "required": ["video_path", "image_path"],
}

_CAPTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string", "description": "Path to input video file"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "segments": {
            "type": "array",
            "description": "Transcript segments: [{start, end, text}, ...]",
            "items": {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "text": {"type": "string"}}, "required": ["start", "end", "text"]},
        },
        "srt_content": {"type": "string", "description": "Raw SRT content string"},
        "font_size": {"type": "number", "description": "Font size relative to video height (0.0-1.0)", "default": 0.06},
        "font_color": {"type": "string", "description": "Font color (hex or name)", "default": "#FFFFFF"},
        "style": {"type": "string", "description": "Animation style: none, fade, karaoke", "default": "karaoke"},
        "position": {"type": "string", "description": "Vertical position: top, center, bottom", "default": "bottom"},
    },
    "required": ["video_path"],
}

_ZOOM_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to input video file"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "effect": {"type": "string", "description": "Effect: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, ken_burns", "default": "zoom_in"},
        "zoom_start": {"type": "number", "description": "Starting zoom level (1.0 = no zoom)", "default": 1.0},
        "zoom_end": {"type": "number", "description": "Ending zoom level", "default": 1.5},
        "duration": {"type": "string", "description": "Duration of effect or 'full'", "default": "full"},
    },
    "required": ["input_path"],
}

_SPLIT_SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "input_paths": {
            "type": "array",
            "description": "List of input video file paths (2-4 videos)",
            "items": {"type": "string"},
        },
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "layout": {"type": "string", "description": "Layout: side_by_side, top_bottom, grid_2x2, picture_in_picture", "default": "side_by_side"},
        "duration": {"type": "string", "description": "Output duration: shortest, longest", "default": "shortest"},
    },
    "required": ["input_paths"],
}

_RENDER_SCHEMA = {
    "type": "object",
    "properties": {
        "input_path": {"type": "string", "description": "Path to input video file (edited intermediate)"},
        "output_path": {"type": "string", "description": "Output path (optional, auto-generated)"},
        "resolution": {"type": "string", "description": "Output resolution (e.g., 1920x1080, original)", "default": "original"},
        "video_codec": {"type": "string", "description": "Video codec (libx264, libx265, copy)", "default": "libx264"},
        "audio_codec": {"type": "string", "description": "Audio codec (aac, copy, mp3)", "default": "aac"},
        "crf": {"type": "number", "description": "Quality (0-51, lower=better)", "default": 23},
        "preset": {"type": "string", "description": "Encoding preset (ultrafast, fast, medium, slow, veryslow)", "default": "medium"},
        "normalize_audio": {"type": "boolean", "description": "Normalize audio levels", "default": True},
    },
    "required": ["input_path"],
}

_TRANSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "input_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1 or 2 video paths. 1=fade in/out, 2=crossfade.",
        },
        "transition_type": {
            "type": "string",
            "enum": ["fade_to_black", "fade_from_black", "fade_in", "fade_out", "crossfade", "wipe_left", "wipe_right"],
            "default": "fade_in",
        },
        "duration": {"type": "number", "default": 1.0},
    },
    "required": ["input_paths"],
}

_AUDIO_MIX_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string"},
        "mode": {
            "type": "string",
            "enum": ["replace", "adjust_volume", "normalize", "add_background", "extract", "mute"],
            "default": "normalize",
        },
        "audio_path": {"type": "string"},
        "volume": {"type": "number", "default": 1.0},
        "background_volume": {"type": "number", "default": 0.3},
        "duck": {"type": "boolean", "default": False},
    },
    "required": ["video_path", "mode"],
}

_WATERMARK_SCHEMA = {
    "type": "object",
    "properties": {
        "video_path": {"type": "string"},
        "mode": {"type": "string", "enum": ["text", "image"], "default": "text"},
        "text": {"type": "string"},
        "image_path": {"type": "string"},
        "position": {"type": "string", "default": "bottom-right"},
        "opacity": {"type": "number", "default": 0.5},
        "font_size": {"type": "number", "default": 24},
        "font_color": {"type": "string", "default": "#FFFFFF"},
        "start": {"type": "string", "default": "0"},
        "end": {"type": "string"},
        "scale": {"type": "string"},
    },
    "required": ["video_path", "mode"],
}


# ── Server bootstrap ────────────────────────────────────────────────────────

def main():
    server = MCPServer("video-editor", "1.0.0")

    server.register("video_probe", "Get video metadata via ffprobe", _PROBE_SCHEMA, handle_probe)
    server.register("video_transcribe", "Speech-to-text with word-level timestamps", _TRANSCRIBE_SCHEMA, handle_transcribe)
    server.register("video_cut", "Extract video segments", _CUT_SCHEMA, handle_cut)
    server.register("video_overlay", "Overlay images with fade transitions", _OVERLAY_SCHEMA, handle_overlay)
    server.register("video_captions", "Burn animated subtitles (SRT/ASS karaoke)", _CAPTIONS_SCHEMA, handle_captions)
    server.register("video_zoom", "Dynamic zoom/pan (Ken Burns effect)", _ZOOM_SCHEMA, handle_zoom)
    server.register("video_split_screen", "Multi-video layouts", _SPLIT_SCREEN_SCHEMA, handle_split_screen)
    server.register("video_render", "Final composition with codec/quality settings", _RENDER_SCHEMA, handle_render)
    server.register("video_transition", "Fade, crossfade, and wipe transitions", _TRANSITION_SCHEMA, handle_transition)
    server.register("video_audio_mix", "Audio mixing, normalization, and replacement", _AUDIO_MIX_SCHEMA, handle_audio_mix)
    server.register("video_watermark", "Text or image watermarks", _WATERMARK_SCHEMA, handle_watermark)

    server.run()


if __name__ == "__main__":
    main()
