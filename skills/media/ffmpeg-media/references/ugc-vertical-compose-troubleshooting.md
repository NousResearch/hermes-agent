# UGC Vertical Compose Troubleshooting (Avatar + Voice + BGM + CTA)

Use this when a generated talking-head video is exported for Reels/TikTok 9:16 and users report:
- "se ve cuadrado/deforme"
- audio bed cuts before voice ends
- ending/CTA cuts or glitches

## 1) Square source avatar in vertical output

### Symptom
- Source model returns 1:1 (e.g., 960x960) and direct `scale+crop` to 1080x1920 makes framing feel wrong or deformed.

### Robust pattern
- Build blurred full-frame background from same source.
- Overlay foreground scaled with `force_original_aspect_ratio=decrease`.

Example filter pattern:
```bash
[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=18[bg];
[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];
[bg][fg]overlay=(W-w)/2:(H-h)/2,fps=30[v]
```

## 2) BGM ends too early (perceived cut)

### Symptom
- Voice continues but background music disappears before end.

### Robust pattern
- Loop background before `amix`.

```bash
[2:a]volume=0.30,aloop=loop=-1:size=2147483647[bgm];
[voice][bgm]amix=inputs=2:duration=first:dropout_transition=3[a]
```

## 3) CTA concat truncates or fails

### Symptom
- Video/audio cuts on last seconds.
- concat failure when CTA has no audio stream.

### Robust pattern
- Re-encode concat via filter graph (not `-c copy`).
- If CTA has no audio, inject silence using `anullsrc` for CTA duration.

```bash
ffmpeg -y \
  -i main_vertical.mp4 \
  -i cta_norm.mp4 \
  -f lavfi -t <cta_secs> -i anullsrc=channel_layout=mono:sample_rate=44100 \
  -filter_complex "[0:v][0:a][1:v][2:a]concat=n=2:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" -c:v libx264 -c:a aac -pix_fmt yuv420p out_final.mp4
```

## 4) Practical QA checks before delivery

- `ffprobe` final width/height/fps/duration.
- Confirm main segment + CTA duration sum is coherent.
- Confirm no abrupt audio drop in last 3s before CTA.
- Confirm face proportions look natural in first 5s and mid-shot.
