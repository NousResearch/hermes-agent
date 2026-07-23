# p5.js Reference

> Full p5.js production pipeline — see original at `~/.hermes/skills/.archive/creative/p5js/`

## Quick Start

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <script>p5.disableFriendlyErrors = true;</script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"></script>
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
<script>
// === Configuration ===
const CONFIG = { seed: 42 };

// === Color Palette ===
const PALETTE = { bg: '#0a0a0f', primary: '#e8d5b7' };

// === Global State ===
let particles = [];

function setup() {
  createCanvas(1920, 1080);
  randomSeed(CONFIG.seed);
  noiseSeed(CONFIG.seed);
  colorMode(HSB, 360, 100, 100, 100);
}

function draw() {
  // Render frame...
}

function keyPressed() {
  if (key === 's' || key === 'S') saveCanvas('output', 'png');
  if (key === 'g' || key === 'G') saveGif('output', 5);
  if (key === 'r' || key === 'R') { randomSeed(millis()); noiseSeed(millis()); }
}
</script>
</body>
</html>
```

## Key Patterns

- **Seeded randomness**: Always `randomSeed()` + `noiseSeed()` for reproducibility
- **HSB color mode**: `colorMode(HSB, 360, 100, 100, 100)` — never hardcode RGB
- **Performance**: `p5.disableFriendlyErrors = true` + `Math.*` in hot loops
- **Layers**: `createGraphics()` offscreen buffers for trails, masks, composition
- **WebGL origin**: center of canvas (not top-left); `translate(-width/2, -height/2)` for P2D-like coords

## Modes

| Mode | Reference |
|------|-----------|
| Generative art | `references/visual-effects.md` |
| Data visualization | `references/interaction.md` |
| Interactive experience | `references/interaction.md` |
| Animation / motion graphics | `references/animation.md` |
| 3D scene | `references/webgl-and-3d.md` |
| Image processing | `references/visual-effects.md` § Pixel Manipulation |
| Audio-reactive | `references/interaction.md` § Audio Input |

## Export

| Format | Method |
|--------|--------|
| PNG | `saveCanvas('output', 'png')` in `keyPressed()` |
| High-res PNG | Puppeteer headless: `node scripts/export-frames.js sketch.html --width 3840 --height 2160 --frames 1` |
| GIF | `saveGif('output', 5)` |
| MP4 | `saveFrames()` + `ffmpeg -i frame-%04d.png -c:v libx264 output.mp4` |

For headless MP4: use `noLoop()` + `window._p5Ready = true` in setup, then `bash scripts/render.sh sketch.html output.mp4`.

## Reference Files

See `.archive/creative/p5js/references/` for full content:
- `core-api.md` — Canvas setup, draw loop, offscreen buffers, composition
- `shapes-and-geometry.md` — Primitives, `beginShape()`, `p5.Vector`, SDFs
- `visual-effects.md` — Noise, flow fields, particles, pixel manipulation, textures
- `animation.md` — Easing, `lerp()`, spring physics, timeline sequencing
- `typography.md` — `textToPoints()`, kinetic typography
- `color-systems.md` — HSB/RGB, palettes, `blendMode()`, gradients
- `webgl-and-3d.md` — WebGL, 3D, GLSL shaders, framebuffers
- `interaction.md` — Mouse, keyboard, audio, scroll-driven
- `export-pipeline.md` — All export methods, headless, platform export
- `troubleshooting.md` — Performance, debugging, memory leaks

## Templates

- `templates/viewer.html` — Interactive viewer with seed nav, parameter sliders, PNG download
- `scripts/export-frames.js` — Puppeteer headless frame capture
- `scripts/render.sh` — Headless MP4 rendering
- `scripts/serve.sh` — Local HTTP server for assets
