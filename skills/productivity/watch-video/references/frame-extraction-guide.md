# Frame Extraction Guide

## Overview
This document describes the ffmpeg-based frame extraction strategy used in the watch-video skill.

## Adaptive Frame Rate Policy

The skill automatically calculates the optimal frame rate based on video duration to balance coverage with token/cost budgets.

### Frame Rate Rules

| Video Duration | Frame Rate | Approx. Frames (30 min) | Use Case |
|----------------|------------|------------------------|----------|
| < 5 min (300s) | 1.0 fps | 300 → capped at 100 | Dense coverage for short videos |
| 5-30 min (300-1800s) | 0.2 fps (1/5s) | 360 → capped at 100 | Standard tutorial/lecture |
| 30-60 min (1800-3600s) | 0.1 fps (1/10s) | 360 → capped at 100 | Long presentations |
| > 60 min (>3600s) | 0.05 fps (1/20s) | 360 → capped at 100 | Very long content |

### Hard Limits
- **Maximum frames**: 100 (hard cap, enforced after extraction)
- **Maximum fps**: 2.0 (prevents excessive frame extraction)
- **Minimum frames**: 1 (for very short clips)

## Focused Mode (`--start` / `--end`)

When a time window is specified:
1. Frame rate is calculated based on **window duration**, not full video duration
2. Same frame budget (max_frames) concentrated in the target window
3. Results in higher effective frame density for the region of interest

Example: 30-min video, `--start 10:00 --end 15:00` (5 min window)
- Normal: 0.2 fps → ~60 frames over 30 min
- Focused: 1.0 fps → ~300 frames in 5 min → capped at 100 frames in window

## FFmpeg Command Structure

```bash
ffmpeg -y -hide_banner -loglevel error \
  -i input.mp4 \
  -vf "trim=start=SS:end=EE,fps=F,scale=W:-1" \
  -q:v 2 \
  frames/frame_%04d.jpg
```

### Filter Chain Explanation

| Filter | Purpose |
|--------|---------|
| `trim=start=SS:end=EE` | Time range selection (focused mode) |
| `fps=F` | Frame rate control (adaptive) |
| `scale=W:-1` | Resize to target width, maintain aspect ratio |
| `-q:v 2` | JPEG quality (2 = high quality, 31 = low) |

### Quality vs. Size Tradeoff

| Quality (`-q:v`) | File Size (512px frame) | Use Case |
|------------------|------------------------|----------|
| 2 | ~50-100 KB | Default — good detail for vision |
| 5 | ~20-40 KB | Faster upload, acceptable quality |
| 10 | ~10-20 KB | Bandwidth constrained |

Default: `-q:v 2` (high quality)

## Resolution Normalization

All frames scaled to target width (default 512px) with `-1` for height to maintain aspect ratio.

### Why 512px Default?

- Matches qwen3-vl vision encoder patch size (16x16 patches)
- 512px = 32 patches wide → optimal for vision transformer
- Balances detail with token count (larger images = more patches = more tokens)
- Compatible with most vision models' training resolution

### Alternative Resolutions

| Width | Height (16:9) | Patches | Tokens/Frame | Use Case |
|-------|---------------|---------|--------------|----------|
| 256 | 144 | 16x9 | ~144 | Fast, low detail |
| 512 | 288 | 32x18 | ~576 | **Default** |
| 768 | 432 | 48x27 | ~1296 | High detail |
| 1024 | 576 | 64x36 | ~2304 | Maximum detail |

## Timestamp Accuracy

Frame timestamps calculated as: `timestamp = start_time + (frame_index / fps)`

### Precision Considerations

- ffmpeg `fps` filter outputs at exact intervals
- Timestamp precision ≈ 1/fps seconds
- For 0.2 fps: ±5 seconds accuracy
- For 1.0 fps: ±1 second accuracy
- Sufficient for transcript alignment (±5s window in merger)

## Output Format

Frames saved as JPEG: `frame_0000.jpg`, `frame_0001.jpg`, ...

### Metadata Tracking

Each frame records:
- `path`: Full path to frame file
- `timestamp`: Calculated timestamp in seconds
- `index`: Sequential frame number (0-based)
- `width`, `height`: Actual dimensions after scaling

## Cleanup

Temp workspace automatically cleaned on pipeline completion (success or failure).
Frames directory: `work_dir/frames/`

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No frames extracted | Video codec issue | Try `-c:v libx264` re-encode first |
| Wrong timestamps | Variable frame rate | Use `-vsync cfr` for constant frame rate |
| Out of memory | Too many frames | Reduce `--max-frames` or `--resolution` |
| FFmpeg not found | Missing dependency | Install ffmpeg via package manager |