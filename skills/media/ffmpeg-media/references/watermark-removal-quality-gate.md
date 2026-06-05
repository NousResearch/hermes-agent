# Watermark removal quality gate (Instagram-style reposts)

Use this when user asks to remove translucent watermark text from reels/clips.

## Goal
Remove watermark without obvious artifact (no hard black box unless user explicitly accepts it).

## Recommended sequence
1. **Download/source video** and probe dimensions/FPS.
2. **Extract sample frames** across timeline (early/mid/late) to verify watermark position stability.
3. Try methods in this order:
   - `delogo` (fast baseline)
   - local blur-patch overlay from nearby pixels (for dark backgrounds)
   - CV inpaint (best local option; slower)
4. After each method, run **visual QA** on at least 3 sampled frames.
5. If watermark text remains legible OR patch is obvious, do **not** claim "pro clean".

## Hard verification rule before claiming success
- Text is not legible in sampled frames.
- Patch does not look like a hard rectangular block under normal viewing.

## Reporting policy (important)
If output still looks patched, stop and report status clearly:
- what was removed,
- what artifact remains,
- next option (external model-based video inpaint) for premium quality.

Do not over-claim final quality.

## Useful commands
```bash
# sample frames
ffmpeg -y -i in.mp4 -vf "select='eq(n,10)+eq(n,540)+eq(n,980)'" -vsync vfr /tmp/check/frame_%02d.png

# baseline delogo
ffmpeg -y -i in.mp4 -vf "delogo=x=250:y=900:w=220:h=100:show=0" -c:v libx264 -crf 18 -c:a copy out_delogo.mp4

# local blur patch overlay (works better on dark zones)
ffmpeg -y -i in.mp4 -filter_complex "[0:v]split=2[base][roi];[roi]crop=220:110:250:900,gblur=sigma=18[patch];[base][patch]overlay=250:900" -c:v libx264 -crf 18 -c:a copy out_patch.mp4
```

## Pitfall
A "clean" output that still shows a dark rectangle fails social quality for brand pages. Continue iterating or escalate method.
