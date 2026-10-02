---
name: ai-camera
description: >-
  Monitor any camera with AI vision. Capture frames from USB webcams, IP cameras
  (RTSP), phone cameras (HTTP), or video files, then analyze with vision_analyze
  for scene description, object detection, motion alerts, people counting, zone
  monitoring, and event logging. Use when the user wants camera monitoring,
  security alerts, activity tracking, or AI-powered vision on a camera feed.
version: 2.0.0
author: vominh1919
license: MIT
metadata:
  hermes:
    tags: [Camera, Vision, AI, Security, Monitoring, IoT, Computer-Vision]
    related_skills: [homeassistant, openhue]
---

# AI Camera — Monitor Any Camera with AI Vision

Use this skill to capture frames from cameras and analyze them with AI.

## Supported Camera Sources

| Source | Capture Command |
|---|---|
| USB webcam (Linux) | `ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 /tmp/camera.jpg -y` |
| USB webcam (macOS) | `ffmpeg -f avfoundation -i "0" -frames:v 1 /tmp/camera.jpg -y` |
| RTSP (IP camera) | `ffmpeg -rtsp_transport tcp -i "rtsp://..." -frames:v 1 /tmp/camera.jpg -y` |
| HTTP (phone) | `ffmpeg -i "http://..." -frames:v 1 /tmp/camera.jpg -y` |
| Video file | `ffmpeg -i "/path/video.mp4" -frames:v 1 /tmp/camera.jpg -y` |

Or use the helper script:

```bash
python3 SKILL_DIR/scripts/capture.py --source 0
python3 SKILL_DIR/scripts/capture.py --source "rtsp://user:pass@IP/stream"
python3 SKILL_DIR/scripts/capture.py --source "http://192.168.1.50:8080/video"
python3 SKILL_DIR/scripts/capture.py --source 0 --best
```

## Workflow: Capture and Analyze

### Single Snapshot

1. Capture a frame:
   ```bash
   python3 SKILL_DIR/scripts/capture.py --source 0 -o /tmp/snapshot.jpg
   ```

2. Analyze with AI:
   ```
   vision_analyze(image_url="/tmp/snapshot.jpg", question="Describe everything you see in detail.")
   ```

3. Report findings to the user.

### Motion Detection (Reduce API Costs)

Before calling vision_analyze, check if anything changed:

```bash
# Capture current frame
python3 SKILL_DIR/scripts/capture.py --source 0 -o /tmp/current.jpg

# Compare with previous frame
python3 SKILL_DIR/scripts/motion.py /tmp/current.jpg /tmp/previous.jpg --threshold 15
```

The script outputs JSON: `{"motion": true, "score": 47.2, "threshold": 15}`

- If `motion` is false, skip AI analysis (nothing changed).
- If `motion` is true, proceed with `vision_analyze`.
- Copy current to previous: `cp /tmp/current.jpg /tmp/previous.jpg`

### Scheduled Monitoring

Set up recurring monitoring with cronjob:

```
cronjob(
    name="camera-monitor",
    schedule="*/5 * * * *",
    prompt="Capture a frame from camera source 0 using capture.py. Check for motion against /tmp/camera_prev.jpg using motion.py. If motion detected, analyze with vision_analyze and send alert via send_message to Telegram. Copy current to /tmp/camera_prev.jpg."
)
```

## Analysis Types

### People Counting

```
vision_analyze(
    image_url="/tmp/camera.jpg",
    question="How many people are visible? For each, describe position (left/center/right), clothing color, and activity."
)
```

### Object Detection

```
vision_analyze(
    image_url="/tmp/camera.jpg",
    question="List every object you can identify. For each: type, approximate location, and notable details."
)
```

### Activity Recognition

```
vision_analyze(
    image_url="/tmp/camera.jpg",
    question="What activity is happening? Describe what each person is doing and the scene context."
)
```

### Anomaly Detection

```
vision_analyze(
    image_url="/tmp/camera.jpg",
    question="Is anything unusual, out of place, or concerning? Consider: unexpected people, objects, damage, open doors."
)
```

## Alerts

After analyzing, alert if something notable:

```
send_message(message="Camera Alert: Person detected at front door", platform="telegram")
```

### Alert with Severity

Route by urgency detected in AI analysis:

```
# If analysis mentions "urgent" or "critical":
send_message(message=f"URGENT: {analysis}", platform="telegram")
# If analysis mentions "person" or "vehicle":
send_message(message=f"Alert: {analysis}", platform="telegram")
# Otherwise: just log, don't alert
```

### Alert Debouncing

Avoid repeated alerts with cooldown script:

```bash
python3 SKILL_DIR/scripts/cooldown.py check camera_alert --minutes 10
# Exit 0 = can alert, Exit 1 = still in cooldown

python3 SKILL_DIR/scripts/cooldown.py set camera_alert
```

## Event Logging

Save events to JSON log via execute_code:

```python
import json, os
from datetime import datetime

log_file = os.path.expanduser("~/camera_log/events.json")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

event = {"time": datetime.now().isoformat(), "type": "person_detected", "description": "Unknown person"}

events = []
if os.path.exists(log_file):
    with open(log_file) as f:
        events = json.load(f)
events.append(event)
with open(log_file, "w") as f:
    json.dump(events, f, indent=2)
```

## Zone Monitoring

Check specific areas with natural language in vision_analyze:

```
vision_analyze(
    image_url="/tmp/camera.jpg",
    question="Focus on the doorway area (left third of image). Is anyone entering, exiting, or loitering?"
)
```

## Multi-Camera Setup

Monitor multiple cameras with separate captures:

```bash
python3 SKILL_DIR/scripts/capture.py --source "rtsp://..." -o /tmp/cam_front.jpg
python3 SKILL_DIR/scripts/capture.py --source "rtsp://..." -o /tmp/cam_back.jpg
python3 SKILL_DIR/scripts/capture.py --source 0 -o /tmp/cam_living.jpg
```

Analyze each with vision_analyze and send combined summary.

## Timelapse

```bash
python3 SKILL_DIR/scripts/capture.py --source 0 --timelapse 28800 60 --output-dir /tmp/timelapse/
```

Then summarize key frames with vision_analyze.

## Troubleshooting

| Issue | Solution |
|---|---|
| Camera not found | Check `/dev/video*` (Linux) or camera permissions |
| RTSP timeout | Add `-rtsp_transport tcp`, check firewall |
| Black frames | Check camera power, try different resolution |
| ffmpeg not found | `apt install ffmpeg` / `brew install ffmpeg` |
| High API costs | Use motion pre-filter, increase interval |
| Too many alerts | Enable cooldown, raise motion threshold |
