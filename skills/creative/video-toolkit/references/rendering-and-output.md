# Rendering and Output

## Preview Stills

Always render preview frames before a full render. This saves minutes of wasted render time:

```bash
# Preview frame at 3 seconds (frame 90 at 30fps)
npx remotion still src/index.ts MainVideo --frame=90 --output=preview.png

# Preview multiple key moments
for f in 0 45 90 180 360 720; do
  npx remotion still src/index.ts MainVideo --frame=$f --output=preview_${f}.png
done
```

Preview stills render in 3-10 seconds. A full 30-second video takes 2-5 minutes. Always preview first.

## Full Render

### Basic Render

```bash
# Default (H.264 MP4)
npx remotion render src/index.ts MainVideo --output=output.mp4

# Specify composition by ID
npx remotion render src/index.ts MyCompositionId --output=output.mp4
```

### Quality Presets

```bash
# Draft — fast, low quality (for checking timing/structure)
npx remotion render src/index.ts MainVideo \
  --output=draft.mp4 \
  --scale=0.5 \
  --quality=50 \
  --concurrency=100%

# Preview — balanced (for client review)
npx remotion render src/index.ts MainVideo \
  --output=preview.mp4 \
  --quality=80

# Production — highest quality
npx remotion render src/index.ts MainVideo \
  --output=final.mp4 \
  --codec=h264 \
  --crf=18 \
  --pixel-format=yuv420p
```

### CRF (Constant Rate Factor)

Controls quality vs file size tradeoff:
- `--crf=18` — near-lossless, large file (production)
- `--crf=23` — default, good quality, reasonable size
- `--crf=28` — smaller file, visible compression (drafts)

Lower CRF = better quality = larger file.

## Output Formats

### Video Formats

```bash
# H.264 MP4 (universal compatibility)
npx remotion render src/index.ts MainVideo --codec=h264 --output=video.mp4

# H.265/HEVC (smaller files, less compatible)
npx remotion render src/index.ts MainVideo --codec=h265 --output=video.mp4

# VP8/WebM (web-optimized)
npx remotion render src/index.ts MainVideo --codec=vp8 --output=video.webm

# ProRes (editing, post-production)
npx remotion render src/index.ts MainVideo --codec=prores --output=video.mov
```

### GIF

```bash
# Animated GIF (limited to 256 colors)
npx remotion render src/index.ts MainVideo --codec=gif --output=output.gif

# Better quality GIF: render to MP4 then convert with ffmpeg
npx remotion render src/index.ts MainVideo --output=temp.mp4
ffmpeg -i temp.mp4 -vf "fps=15,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output.gif
```

### PNG Sequence

```bash
# Individual frames as PNG files
npx remotion render src/index.ts MainVideo --image-format=png --sequence --output-location=frames/
```

### Transparent Video

For overlays and compositing:

```tsx
// In composition — use transparent background
<Composition
  id="Overlay"
  component={OverlayComponent}
  width={1920} height={1080} fps={30}
  durationInFrames={150}
/>
```

```bash
# Render with alpha channel
npx remotion render src/index.ts Overlay --codec=vp8 --output=overlay.webm
# VP8/WebM supports transparency; H.264 does not
```

## Aspect Ratio Rendering

```bash
# Standard 16:9 (YouTube, general)
# Set in Composition: width={1920} height={1080}

# Vertical 9:16 (Instagram Reels, TikTok, YouTube Shorts)
# Set in Composition: width={1080} height={1920}

# Square 1:1 (Instagram feed, Twitter)
# Set in Composition: width={1080} height={1080}
```

To render multiple aspect ratios, register multiple Compositions with different dimensions but the same content (adapt layout per ratio).

## Rendering Performance

### Concurrency

```bash
# Use all CPU cores (default)
npx remotion render ... --concurrency=100%

# Limit to 4 cores (if system is constrained)
npx remotion render ... --concurrency=4
```

### Memory

Long videos or complex scenes may hit memory limits:

```bash
# Increase Node.js memory
NODE_OPTIONS="--max-old-space-size=8192" npx remotion render ...
```

### Performance Tips

- Reduce `scale` for drafts (`--scale=0.5` renders at half resolution)
- Use `jpeg` image format instead of `png` for previews (faster)
- Pre-render complex scenes as video clips and embed them
- Minimize DOM nodes per frame — React reconciliation is the bottleneck

## Post-Processing with ffmpeg

### Add Music to Rendered Video

```bash
# Replace audio track
ffmpeg -i video.mp4 -i music.mp3 -c:v copy -c:a aac -shortest output.mp4

# Mix background music with existing voiceover
ffmpeg -i video.mp4 -i music.mp3 \
  -filter_complex "[1:a]volume=0.3[bg];[0:a][bg]amix=inputs=2:duration=first" \
  -c:v copy output.mp4
```

### Trim

```bash
# Cut from 5s to 25s
ffmpeg -i input.mp4 -ss 5 -to 25 -c copy trimmed.mp4
```

### Resize

```bash
# Scale to 720p
ffmpeg -i input.mp4 -vf scale=1280:720 -c:a copy output_720p.mp4
```

### Compress for Social Media

```bash
# Twitter/X (max 512MB, recommended <15MB for fast upload)
ffmpeg -i input.mp4 -c:v libx264 -crf 26 -preset medium -c:a aac -b:a 128k twitter.mp4

# Discord (max 25MB for free users)
ffmpeg -i input.mp4 -c:v libx264 -crf 28 -preset medium -c:a aac -b:a 96k discord.mp4
```
