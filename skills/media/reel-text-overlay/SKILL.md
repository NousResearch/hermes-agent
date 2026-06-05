---
name: reel-text-overlay
description: Render high-quality text overlays on video scenes for Instagram Reels using Pillow + FFmpeg. Use when the user wants original text-on-video content (kinetic typography, quotes, motivational phrases) for social media.
---

# Reel Text Overlay

Use Pillow for text rendering (NOT FFmpeg drawtext — doesn't handle alpha per-phrase or multi-line reliably). FFmpeg drawtext only for simple single-line overlays.

## Requirements

```bash
python3 -m venv /tmp/reel_venv
/tmp/reel_venv/bin/pip install Pillow
```

## Workflow

1. **Generate HQ scene images** with Pillow:
   - 1 image per phrase/scene (not frame-by-frame at 30fps)
   - Use `ImageDraw.textbbox()` for exact centering
   - Add shadow (+5px offset, alpha=180) + stroke (4 directions, alpha=100) + main text (alpha=255)
   - Dim overlay: `draw.rectangle([(0,0),(W,H)], fill=(0,0,0,60))`
   - Save as PNG quality=95

2. **Convert each image to MP4 clip** with FFmpeg:
   ```
   ffmpeg -y -loop 1 -i scene.png -c:v libx264 -t DURATION -pix_fmt yuv420p -crf 16 -r 30 clip.mp4
   ```

3. **Concatenate clips**:
   ```
   printf "file 'clip_0.mp4'\n..." > list.txt
   ffmpeg -y -f concat -safe 0 -i list.txt -c copy video_only.mp4
   ```

4. **Add fades + music**:
   ```
   ffmpeg -y -i video_only.mp4 -i music.mp3 \
     -filter_complex "[0:v]fade=t=in:st=0:d=0.3,fade=t=out:st=END:d=0.3[v]" \
     -map "[v]" -map "1:a:0" -c:v libx264 -c:a aac -b:a 256k -shortest final.mp4
   ```

## Styles for BeSoul Reels
- Background: cinematic still image (MiniMax AI or similar)
- Text: white bold sans-serif, center-aligned
- Timing: ~2s per phrase, ~2.5s for brand/closing
- Music: ambient/cinematic pad (generate with ffmpeg sine tones mixed with pink noise)

## Pitfalls
- DO NOT generate frame-by-frame at 30fps — Pillow is too slow for 1080x1920 RGBA compositing
- DO NOT use FFmpeg drawtext for multi-line text with per-phrase alpha
- Always show user a preview before publishing
