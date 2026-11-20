#!/usr/bin/env python3
"""Analyze a recording and produce a replayable skill summary.

Reads the events.jsonl + screenshots from a recording directory and produces
a structured JSON summary with:
  - Logical steps (sliding-window boundary detection)
  - Step signatures (action-type sequences) for pattern/loop detection
  - Retry detection (nearby clicks within 500ms = mis-click)
  - Clipboard events
  - Window state changes
  - Auto-generated suggested skill name
  - Optional self-contained HTML timeline viewer (--html flag)

Usage:
    python3 analyze_recording.py <recording_dir> [--html] [--output FILE]

Output:
    Prints a JSON summary to stdout (or writes to --output file).
    With --html, also generates <recording_dir>/viewer.html.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ── Event loading ────────────────────────────────────────────────────────────

def load_events(events_file: Path) -> list:
    """Load events from JSONL file."""
    events = []
    with open(events_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


# ── Sliding-window step detection ────────────────────────────────────────────

PAUSE_THRESHOLD = 1.5  # seconds — natural boundary
RETRY_DISTANCE_PX = 30  # pixels — clicks within this distance = possible retry
RETRY_TIME_MS = 500  # ms — clicks within this time = possible retry


def _event_action_type(evt: dict) -> str:
    """Extract a simplified action type for step signatures."""
    t = evt.get("type", "")
    if t in ("mouse_down",):
        return "click"
    if t in ("key_down",):
        return "type"
    if t == "scroll":
        return "scroll"
    if t == "mouse_dragged":
        return "drag"
    if t == "app_switch":
        return "app_switch"
    if t == "clipboard_copy":
        return "clipboard"
    if t == "window_resized":
        return "window_resize"
    if t == "window_moved":
        return "window_move"
    if t == "window_minimized":
        return "window_minimize"
    if t == "snapshot":
        return "snapshot"
    if t == "screenshot_skipped":
        return "screenshot_skip"
    return t


def group_into_steps(events: list, pause_threshold: float = PAUSE_THRESHOLD) -> list:
    """Group events into logical steps using sliding-window boundary detection.

    A new step starts at a natural boundary:
      - Pause > pause_threshold seconds between events
      - App switch
      - Screenshot with pixel changes (not skipped)
      - Clipboard copy event
    """
    steps = []
    current_step = {
        "step_number": 1,
        "events": [],
        "start_time": 0.0,
        "end_time": 0.0,
        "screenshots": [],
        "ax_trees": [],
        "app": None,
    }
    last_time = 0.0
    first = True

    for evt in events:
        t = evt.get("timestamp", 0.0)
        evt_type = evt.get("type", "")

        # App switch always starts a new step
        if evt_type == "app_switch":
            if current_step["events"]:
                current_step["end_time"] = last_time
                steps.append(current_step)
            current_step = {
                "step_number": len(steps) + 1,
                "events": [],
                "start_time": t,
                "end_time": t,
                "screenshots": [],
                "ax_trees": [],
                "app": evt.get("to", "unknown"),
            }
            current_step["events"].append(evt)
            last_time = t
            first = False
            continue

        # Clipboard copy is a boundary
        if evt_type == "clipboard_copy":
            if current_step["events"]:
                current_step["end_time"] = last_time
                steps.append(current_step)
            current_step = {
                "step_number": len(steps) + 1,
                "events": [],
                "start_time": t,
                "end_time": t,
                "screenshots": [],
                "ax_trees": [],
                "app": current_step["app"],
            }
            current_step["events"].append(evt)
            last_time = t
            first = False
            continue

        # Screenshot with pixel change is a boundary (but not skipped ones)
        if evt_type == "snapshot" and evt.get("screenshot"):
            screenshot = evt.get("screenshot")
            if screenshot:
                current_step["screenshots"].append(screenshot)
            ax = evt.get("ax_tree")
            if ax:
                current_step["ax_trees"].append(ax)
            current_step["app"] = evt.get("frontmost_app", {}).get("name", current_step["app"])
            current_step["events"].append(evt)
            last_time = t
            # Don't start new step for snapshots — they're periodic
            continue

        # Screenshot skipped — log but don't start new step
        if evt_type == "screenshot_skipped":
            current_step["events"].append(evt)
            last_time = t
            continue

        # Pause detection
        if not first and t - last_time > pause_threshold:
            has_interaction = any(
                _event_action_type(e) in ("click", "type", "scroll", "drag", "clipboard")
                for e in current_step["events"]
            )
            if has_interaction:
                current_step["end_time"] = last_time
                steps.append(current_step)
                current_step = {
                    "step_number": len(steps) + 1,
                    "events": [],
                    "start_time": t,
                    "end_time": t,
                    "screenshots": current_step["screenshots"][-1:] if current_step["screenshots"] else [],
                    "ax_trees": current_step["ax_trees"][-1:] if current_step["ax_trees"] else [],
                    "app": current_step["app"],
                }

        if first:
            current_step["start_time"] = t
            first = False

        current_step["events"].append(evt)
        last_time = t

    if current_step["events"]:
        current_step["end_time"] = last_time
        steps.append(current_step)

    return steps


# ── Step signature & pattern detection ───────────────────────────────────────

def compute_step_signature(step: dict) -> list:
    """Compute the sequence of action types for a step."""
    sig = []
    for evt in step["events"]:
        at = _event_action_type(evt)
        if at in ("click", "type", "scroll", "drag", "clipboard",
                   "app_switch", "window_resize", "window_move", "window_minimize"):
            sig.append(at)
    return sig


def detect_patterns(steps: list) -> list:
    """Detect repeated step signatures (loops).

    Returns a list of pattern descriptors:
      {"pattern": ["click", "type"], "count": 3, "step_range": [2, 4]}
    """
    patterns = []
    if len(steps) < 2:
        return patterns

    signatures = [compute_step_signature(s) for s in steps]

    # Look for repeated consecutive signatures
    i = 0
    while i < len(signatures):
        sig = signatures[i]
        if not sig:
            i += 1
            continue
        count = 1
        j = i + 1
        while j < len(signatures) and signatures[j] == sig:
            count += 1
            j += 1
        if count >= 2:
            patterns.append({
                "pattern": sig,
                "count": count,
                "step_range": [i + 1, j],
                "description": f"Repeated {count}x: {' → '.join(sig)}",
            })
        i = j

    return patterns


# ── Retry detection ──────────────────────────────────────────────────────────

def detect_retries(step: dict) -> list:
    """Detect retry patterns — clicks near each other within 500ms.

    Returns list of retry descriptors:
      {"event_index": N, "first_click": [x,y], "second_click": [x,y],
       "distance_px": D, "time_delta_ms": T, "note": "likely mis-click"}
    """
    retries = []
    click_events = []
    for idx, evt in enumerate(step["events"]):
        if evt.get("type") == "mouse_down":
            click_events.append((idx, evt))

    for i in range(1, len(click_events)):
        idx_prev, evt_prev = click_events[i - 1]
        idx_curr, evt_curr = click_events[i]

        t_prev = evt_prev.get("timestamp", 0)
        t_curr = evt_curr.get("timestamp", 0)
        time_delta_ms = (t_curr - t_prev) * 1000

        pos_prev = evt_prev.get("position", [0, 0])
        pos_curr = evt_curr.get("position", [0, 0])
        dist = ((pos_curr[0] - pos_prev[0]) ** 2 + (pos_curr[1] - pos_prev[1]) ** 2) ** 0.5

        if time_delta_ms < RETRY_TIME_MS and dist < RETRY_DISTANCE_PX:
            retries.append({
                "event_index": idx_curr,
                "first_click": [round(pos_prev[0]), round(pos_prev[1])],
                "second_click": [round(pos_curr[0]), round(pos_curr[1])],
                "distance_px": round(dist, 1),
                "time_delta_ms": round(time_delta_ms, 1),
                "note": "likely mis-click, second click is intended target",
            })

    return retries


# ── Step summarization ───────────────────────────────────────────────────────

def summarize_step(step: dict) -> dict:
    """Produce a rich summary of a step."""
    interactions = []
    clipboard_events = []
    window_events = []

    for evt in step["events"]:
        t = evt.get("timestamp", 0)
        evt_type = evt.get("type", "")

        if evt_type == "mouse_down":
            pos = evt.get("position", [0, 0])
            button = evt.get("button", "left")
            mods = "+".join(evt.get("modifiers", []))
            mod_str = " [{}]".format(mods) if mods else ""
            interactions.append({
                "time": round(t, 2),
                "action": "click {}{}".format(button, mod_str),
                "position": [round(pos[0]), round(pos[1])],
            })
        elif evt_type == "key_down":
            key = evt.get("key", "unknown")
            char = evt.get("character", "")
            mods = "+".join(evt.get("modifiers", []))
            if mods and key in ("shift", "control", "option", "command", "fn",
                                "caps_lock", "right_shift", "right_option", "right_control"):
                continue
            mod_str = "{}+".format(mods) if mods else ""
            interactions.append({
                "time": round(t, 2),
                "action": "key: {}{}".format(mod_str, key),
                "character": char if char and char != key else None,
            })
        elif evt_type == "scroll":
            direction = evt.get("direction", "none")
            delta = evt.get("delta_y", 0)
            interactions.append({
                "time": round(t, 2),
                "action": "scroll {} (delta: {})".format(direction, delta),
                "position": [round(evt.get("position", [0, 0])[0]),
                             round(evt.get("position", [0, 0])[1])],
            })
        elif evt_type == "mouse_dragged":
            pos = evt.get("position", [0, 0])
            interactions.append({
                "time": round(t, 2),
                "action": "drag",
                "position": [round(pos[0]), round(pos[1])],
            })
        elif evt_type == "app_switch":
            interactions.append({
                "time": round(t, 2),
                "action": "app switch: {} → {}".format(evt.get("from", "?"), evt.get("to", "?")),
            })
        elif evt_type == "clipboard_copy":
            preview = evt.get("preview", "")
            clipboard_events.append({
                "time": round(t, 2),
                "preview": preview,
                "content_hash": evt.get("content_hash", ""),
            })
        elif evt_type in ("window_resized", "window_moved", "window_minimized"):
            window_events.append({
                "time": round(t, 2),
                "action": evt_type,
                "window": evt.get("window", ""),
                "details": evt.get("details", {}),
            })

    signature = compute_step_signature(step)
    retries = detect_retries(step)

    return {
        "step_number": step["step_number"],
        "app": step["app"],
        "start_time": round(step["start_time"], 2),
        "end_time": round(step["end_time"], 2),
        "duration": round(step["end_time"] - step["start_time"], 2),
        "interaction_count": len(interactions),
        "screenshots": step["screenshots"],
        "ax_trees": step["ax_trees"],
        "signature": signature,
        "interactions": interactions,
        "clipboard_events": clipboard_events,
        "window_events": window_events,
        "retries": retries,
    }


# ── Suggested skill name generation ──────────────────────────────────────────

def generate_skill_name(steps: list) -> str:
    """Auto-generate a skill name from the app + first action."""
    if not steps:
        return "recorded-workflow"

    first_step = steps[0]
    app = first_step.get("app") or "unknown"

    # Clean app name
    app_clean = re.sub(r"[^a-zA-Z0-9]+", "-", app.lower()).strip("-")

    # Try to find first meaningful action
    action = "workflow"
    for evt in first_step.get("events", []):
        evt_type = evt.get("type", "")
        if evt_type == "app_switch":
            action = "switch"
            break
        elif evt_type == "mouse_down":
            action = "click"
            break
        elif evt_type == "key_down":
            key = evt.get("key", "")
            if key and key not in ("shift", "control", "option", "command", "fn"):
                action = "type"
                break

    return "{}-{}".format(app_clean, action)


# ── HTML viewer generation ───────────────────────────────────────────────────

def generate_html_viewer(recording_dir: Path, events: list, steps: list,
                         summaries: list, patterns: list, metadata: dict) -> str:
    """Generate a self-contained HTML timeline viewer.

    All CSS/JS inline, images as base64 data URIs.
    """
    # Collect screenshot data URIs
    screenshots = []
    screenshot_dir = recording_dir / "screenshots"
    for evt in events:
        if evt.get("type") == "snapshot" and evt.get("screenshot"):
            shot_path = screenshot_dir.parent / evt["screenshot"]
            if shot_path.exists():
                try:
                    with open(shot_path, "rb") as f:
                        data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode()
                    screenshots.append({
                        "timestamp": round(evt.get("timestamp", 0), 2),
                        "data_uri": data_uri,
                        "number": evt.get("screenshot_number", 0),
                    })
                except Exception:
                    pass

    # Collect event markers for timeline
    markers = []
    for evt in events:
        t = evt.get("timestamp", 0)
        evt_type = evt.get("type", "")
        if evt_type == "mouse_down":
            markers.append({"time": round(t, 2), "type": "click", "color": "#ff4444"})
        elif evt_type == "key_down":
            markers.append({"time": round(t, 2), "type": "key", "color": "#4488ff"})
        elif evt_type == "scroll":
            markers.append({"time": round(t, 2), "type": "scroll", "color": "#44ff44"})
        elif evt_type == "clipboard_copy":
            markers.append({"time": round(t, 2), "type": "clipboard", "color": "#ffaa44"})
        elif evt_type in ("window_resized", "window_moved", "window_minimized"):
            markers.append({"time": round(t, 2), "type": "window", "color": "#ff44ff"})

    # Step boundaries
    step_boundaries = []
    for s in summaries:
        step_boundaries.append({
            "step": s["step_number"],
            "time": s["start_time"],
            "app": s.get("app", ""),
        })

    # AX trees
    ax_trees = []
    ax_dir = recording_dir / "ax_trees"
    for evt in events:
        if evt.get("type") == "snapshot" and evt.get("ax_tree"):
            ax_path = screenshot_dir.parent / evt["ax_tree"]
            if ax_path.exists():
                try:
                    content = ax_path.read_text(errors="replace")[:5000]
                    ax_trees.append({
                        "timestamp": round(evt.get("timestamp", 0), 2),
                        "content": content,
                        "number": evt.get("screenshot_number", 0),
                    })
                except Exception:
                    pass

    # Build JSON data
    viewer_data = json.dumps({
        "screenshots": screenshots,
        "markers": markers,
        "step_boundaries": step_boundaries,
        "ax_trees": ax_trees,
        "steps": [{"step": s["step_number"], "app": s["app"],
                    "duration": s["duration"], "actions": s["signature"]} for s in summaries],
        "patterns": patterns,
        "metadata": metadata,
    })

    # Duration for timeline scale
    duration = metadata.get("duration_seconds", 10.0) or 10.0

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Recording Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; overflow: hidden; height: 100vh; }
.header { background: #16213e; padding: 12px 20px; border-bottom: 1px solid #0f3460; }
.header h1 { font-size: 18px; color: #e94560; }
.header .meta { font-size: 12px; color: #888; margin-top: 4px; }
.timeline-container { background: #16213e; padding: 16px 20px; border-bottom: 1px solid #0f3460; }
.timeline { position: relative; height: 60px; background: #0f3460; border-radius: 4px; margin-top: 8px; cursor: pointer; }
.timeline-bar { position: absolute; top: 0; left: 0; height: 100%; background: linear-gradient(90deg, #e94560, #0f3460); border-radius: 4px; opacity: 0.3; }
.timeline-marker { position: absolute; width: 8px; height: 8px; border-radius: 50%; transform: translate(-50%, -50%); top: 50%; cursor: pointer; z-index: 2; }
.timeline-marker:hover { width: 12px; height: 12px; }
.step-boundary { position: absolute; width: 1px; height: 100%; background: #e94560; opacity: 0.5; top: 0; }
.step-label { position: absolute; font-size: 9px; color: #e94560; top: -12px; transform: translateX(-50%); }
.playback-controls { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
.btn { background: #e94560; color: white; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.btn:hover { background: #c73e54; }
.btn:disabled { background: #555; cursor: not-allowed; }
.speed-select { background: #0f3460; color: #e0e0e0; border: 1px solid #333; padding: 4px 8px; border-radius: 4px; }
.main-content { display: flex; height: calc(100vh - 180px); }
.screenshot-panel { flex: 2; padding: 16px; overflow: auto; display: flex; align-items: center; justify-content: center; }
.screenshot-panel img { max-width: 100%; max-height: 100%; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
.info-panel { flex: 1; background: #16213e; padding: 16px; overflow: auto; border-left: 1px solid #0f3460; }
.info-panel h3 { color: #e94560; font-size: 14px; margin-bottom: 8px; }
.info-panel .event-list { font-size: 11px; }
.info-panel .event-item { padding: 4px 0; border-bottom: 1px solid #0f3460; }
.info-panel .event-item .time { color: #888; }
.info-panel .event-item .action { color: #4488ff; }
.ax-tree { font-family: 'SF Mono', Monaco, monospace; font-size: 10px; white-space: pre-wrap; color: #aaa; max-height: 300px; overflow: auto; background: #0a0a1a; padding: 8px; border-radius: 4px; margin-top: 8px; }
.screenshot-list { display: flex; gap: 4px; overflow-x: auto; padding: 8px 20px; background: #16213e; border-bottom: 1px solid #0f3460; }
.screenshot-thumb { width: 80px; height: 50px; border-radius: 4px; cursor: pointer; border: 2px solid transparent; flex-shrink: 0; object-fit: cover; }
.screenshot-thumb.active { border-color: #e94560; }
.screenshot-thumb:hover { border-color: #4488ff; }
.patterns { margin-top: 12px; }
.pattern-item { background: #0f3460; padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; font-size: 11px; }
.empty-state { color: #555; font-size: 14px; text-align: center; }
</style>
</head>
<body>
<div class="header">
  <h1>🖥️ Recording Viewer</h1>
  <div class="meta" id="meta-info"></div>
</div>
<div class="timeline-container">
  <div class="playback-controls">
    <button class="btn" id="play-btn">▶ Play</button>
    <select class="speed-select" id="speed">
      <option value="1">1x (Real-time)</option>
      <option value="2">2x</option>
      <option value="5">5x</option>
      <option value="10">10x</option>
    </select>
    <span id="time-display" style="color:#888;font-size:12px;">0.0s / __DURATION__s</span>
  </div>
  <div class="timeline" id="timeline">
    <div class="timeline-bar" style="width: 100%;"></div>
  </div>
</div>
<div class="screenshot-list" id="screenshot-list"></div>
<div class="main-content">
  <div class="screenshot-panel" id="screenshot-panel">
    <div class="empty-state">Select a screenshot to view</div>
  </div>
  <div class="info-panel" id="info-panel">
    <h3>Event Details</h3>
    <div class="event-list" id="event-list"><div class="empty-state">No events selected</div></div>
    <div id="ax-tree-section" style="display:none;">
      <h3 style="margin-top:16px;">AX Tree</h3>
      <div class="ax-tree" id="ax-tree-content"></div>
    </div>
    <div class="patterns" id="patterns-section" style="display:none;">
      <h3>Detected Patterns</h3>
      <div id="patterns-list"></div>
    </div>
  </div>
</div>
<script>
const DATA = __VIEWER_DATA__;
const DURATION = __DURATION__;
let currentIdx = 0;
let playing = false;
let playTimer = null;

const timeline = document.getElementById('timeline');
const screenshotList = document.getElementById('screenshot-list');
const screenshotPanel = document.getElementById('screenshot-panel');
const infoPanel = document.getElementById('info-panel');
const eventList = document.getElementById('event-list');
const playBtn = document.getElementById('play-btn');
const speedSelect = document.getElementById('speed');
const timeDisplay = document.getElementById('time-display');
const metaInfo = document.getElementById('meta-info');

// Meta info
const meta = DATA.metadata || {};
metaInfo.textContent = 'Duration: ' + (meta.duration_seconds || 0).toFixed(1) + 's | Events: ' + (meta.event_count || 0) + ' | Screenshots: ' + (meta.screenshot_count || 0) + ' | Platform: ' + (meta.platform || 'unknown');

// Render step boundaries
DATA.step_boundaries.forEach(function(sb) {
  var div = document.createElement('div');
  div.className = 'step-boundary';
  div.style.left = (sb.time / DURATION * 100) + '%';
  timeline.appendChild(div);
  var label = document.createElement('div');
  label.className = 'step-label';
  label.textContent = 'S' + sb.step;
  label.style.left = (sb.time / DURATION * 100) + '%';
  timeline.appendChild(label);
});

// Render event markers
DATA.markers.forEach(function(m) {
  var dot = document.createElement('div');
  dot.className = 'timeline-marker';
  dot.style.left = (m.time / DURATION * 100) + '%';
  dot.style.background = m.color;
  dot.title = m.type + ' @ ' + m.time + 's';
  timeline.appendChild(dot);
});

// Render screenshot thumbnails
DATA.screenshots.forEach(function(shot, idx) {
  var img = document.createElement('img');
  img.className = 'screenshot-thumb';
  img.src = shot.data_uri;
  img.dataset.idx = idx;
  img.addEventListener('click', function() { showScreenshot(idx); });
  screenshotList.appendChild(img);
});

// Patterns
if (DATA.patterns && DATA.patterns.length > 0) {
  document.getElementById('patterns-section').style.display = 'block';
  var patternsList = document.getElementById('patterns-list');
  DATA.patterns.forEach(function(p) {
    var div = document.createElement('div');
    div.className = 'pattern-item';
    div.textContent = p.description;
    patternsList.appendChild(div);
  });
}

function showScreenshot(idx) {
  currentIdx = idx;
  var thumbs = document.querySelectorAll('.screenshot-thumb');
  thumbs.forEach(function(el, i) { el.classList.toggle('active', i === idx); });
  var shot = DATA.screenshots[idx];
  screenshotPanel.innerHTML = '<img src="' + shot.data_uri + '" alt="Screenshot ' + shot.number + '">';
  timeDisplay.textContent = shot.timestamp.toFixed(1) + 's / ' + DURATION.toFixed(1) + 's';

  // Show events near this screenshot (within 2s)
  var nearbyEvents = DATA.markers.filter(function(m) { return Math.abs(m.time - shot.timestamp) < 2.0; });
  eventList.innerHTML = nearbyEvents.length > 0
    ? nearbyEvents.map(function(m) { return '<div class="event-item"><span class="time">' + m.time.toFixed(2) + 's</span> <span class="action">' + m.type + '</span></div>'; }).join('')
    : '<div class="empty-state">No events near this screenshot</div>';

  // Show AX tree if available
  var axTree = DATA.ax_trees.find(function(a) { return a.number === shot.number; });
  if (axTree) {
    document.getElementById('ax-tree-section').style.display = 'block';
    document.getElementById('ax-tree-content').textContent = axTree.content;
  } else {
    document.getElementById('ax-tree-section').style.display = 'none';
  }
}

// Timeline click to seek
timeline.addEventListener('click', function(e) {
  var rect = timeline.getBoundingClientRect();
  var pct = (e.clientX - rect.left) / rect.width;
  var seekTime = pct * DURATION;
  // Find nearest screenshot
  var nearest = 0;
  var minDist = Infinity;
  DATA.screenshots.forEach(function(s, i) {
    var d = Math.abs(s.timestamp - seekTime);
    if (d < minDist) { minDist = d; nearest = i; }
  });
  showScreenshot(nearest);
});

// Play button
playBtn.addEventListener('click', function() {
  if (playing) {
    playing = false;
    playBtn.textContent = '▶ Play';
    clearTimeout(playTimer);
  } else {
    playing = true;
    playBtn.textContent = '⏸ Pause';
    playNext();
  }
});

function playNext() {
  if (!playing || currentIdx >= DATA.screenshots.length - 1) {
    playing = false;
    playBtn.textContent = '▶ Play';
    return;
  }
  showScreenshot(currentIdx + 1);
  var speed = parseFloat(speedSelect.value);
  var curr = DATA.screenshots[currentIdx];
  var next = DATA.screenshots[currentIdx + 1];
  var delay = (next.timestamp - curr.timestamp) * 1000 / speed;
  playTimer = setTimeout(playNext, Math.max(100, Math.min(delay, 5000)));
}

// Show first screenshot
if (DATA.screenshots.length > 0) {
  showScreenshot(0);
}
</script>
</body>
</html>"""

    # Replace placeholders with actual data
    html = html.replace("__VIEWER_DATA__", viewer_data)
    html = html.replace("__DURATION__", "{:.1f}".format(duration))

    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a recording and produce a replayable skill summary."
    )
    parser.add_argument("recording_dir", type=str, help="Path to recording directory")
    parser.add_argument("--html", action="store_true", help="Generate self-contained HTML viewer")
    parser.add_argument("--output", type=str, default=None, help="Write JSON to file instead of stdout")
    parser.add_argument("--pause-threshold", type=float, default=PAUSE_THRESHOLD,
                        help="Pause threshold in seconds for step boundaries (default: {})".format(PAUSE_THRESHOLD))
    args = parser.parse_args()

    recording_dir = Path(args.recording_dir)
    events_file = recording_dir / "events" / "events.jsonl"
    metadata_file = recording_dir / "metadata.json"

    if not events_file.exists():
        print("ERROR: No events file at {}".format(events_file), file=sys.stderr)
        sys.exit(1)

    # Load metadata
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Load and analyze events
    events = load_events(events_file)
    steps = group_into_steps(events, args.pause_threshold)
    summaries = [summarize_step(s) for s in steps]
    patterns = detect_patterns(steps)
    suggested_name = generate_skill_name(steps)

    # Build the final output
    output = {
        "recording_dir": str(recording_dir),
        "metadata": metadata,
        "total_events": len(events),
        "total_steps": len(summaries),
        "suggested_skill_name": suggested_name,
        "detected_patterns": patterns,
        "steps": summaries,
    }

    # Output
    output_json = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print("Analysis written to {}".format(args.output), file=sys.stderr)
    else:
        print(output_json)

    # HTML viewer
    if args.html:
        html_content = generate_html_viewer(recording_dir, events, steps, summaries, patterns, metadata)
        html_path = recording_dir / "viewer.html"
        with open(html_path, "w") as f:
            f.write(html_content)
        print("HTML viewer written to {}".format(html_path), file=sys.stderr)


if __name__ == "__main__":
    main()
