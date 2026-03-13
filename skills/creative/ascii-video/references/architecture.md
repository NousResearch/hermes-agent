# Architecture Reference

## Grid System

### Multi-Density Grids

Pre-initialize multiple grid sizes. Switch per section for visual variety.

| Key | Font Size | Grid (1920x1080) | Use |
|-----|-----------|-------------------|-----|
| xs | 8 | 400x108 | Ultra-dense data fields |
| sm | 10 | 320x83 | Dense detail, rain, starfields |
| md | 16 | 192x56 | Default balanced, transitions |
| lg | 20 | 160x45 | Quote/lyric text (readable at 1080p) |
| xl | 24 | 137x37 | Short quotes, large titles |
| xxl | 40 | 80x22 | Giant text, minimal |

**Grid sizing for text-heavy content**: When displaying readable text (quotes, lyrics, testimonials), use 20px (`lg`) as the primary grid. This gives 160 columns -- plenty for lines up to ~50 chars centered. For very short quotes (< 60 chars, <= 3 lines), 24px (`xl`) makes them more impactful. Only init the grids you actually use -- each grid pre-rasterizes all characters which costs ~0.3-0.5s.

Grid dimensions: `cols = VW // cell_width`, `rows = VH // cell_height`.

### Font Selection

Don't hardcode a single font. Choose fonts to match the project's mood. Monospace fonts are required for grid alignment but vary widely in personality:

| Font | Personality | Platform |
|------|-------------|----------|
| Menlo | Clean, neutral, Apple-native | macOS |
| Monaco | Retro terminal, compact | macOS |
| Courier New | Classic typewriter, wide | Cross-platform |
| SF Mono | Modern, tight spacing | macOS |
| Consolas | Windows native, clean | Windows |
| JetBrains Mono | Developer, ligature-ready | Install |
| Fira Code | Geometric, modern | Install |
| IBM Plex Mono | Corporate, authoritative | Install |
| Source Code Pro | Adobe, balanced | Install |

**Font detection at init**: probe available fonts and fall back gracefully:

```python
import platform

def find_font(preferences):
    """Try fonts in order, return first that exists."""
    for name, path in preferences:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No monospace font found. Tried: {[p for _,p in preferences]}")

FONT_PREFS_MACOS = [
    ("Menlo", "/System/Library/Fonts/Menlo.ttc"),
    ("Monaco", "/System/Library/Fonts/Monaco.ttf"),
    ("SF Mono", "/System/Library/Fonts/SFNSMono.ttf"),
    ("Courier", "/System/Library/Fonts/Courier.ttc"),
]
FONT_PREFS_LINUX = [
    ("DejaVu Sans Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ("Liberation Mono", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
    ("Noto Sans Mono", "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"),
    ("Ubuntu Mono", "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"),
]
FONT_PREFS = FONT_PREFS_MACOS if platform.system() == "Darwin" else FONT_PREFS_LINUX
```

**Multi-font rendering**: use different fonts for different layers (e.g., monospace for background, a bolder variant for overlay text). Each GridLayer owns its own font:

```python
grid_bg = GridLayer(find_font(FONT_PREFS), 16)       # background
grid_text = GridLayer(find_font(BOLD_PREFS), 20)      # readable text
```

### Collecting All Characters

Before initializing grids, gather all characters that need bitmap pre-rasterization:

```python
all_chars = set()
for pal in [PAL_DEFAULT, PAL_DENSE, PAL_BLOCKS, PAL_RUNE, PAL_KATA,
            PAL_GREEK, PAL_MATH, PAL_DOTS, PAL_BRAILLE, PAL_STARS,
            PAL_BINARY, PAL_MUSIC, PAL_BOX, PAL_CIRCUIT, PAL_ARROWS,
            PAL_HERMES]:  # ... all palettes used in project
    all_chars.update(pal)
# Add any overlay text characters
all_chars.update("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,-:;!?/|")
all_chars.discard(" ")  # space is never rendered
```

### GridLayer Initialization

Each grid pre-computes coordinate arrays for vectorized effect math:

```python
class GridLayer:
    def __init__(self, font_path, font_size):
        self.font = ImageFont.truetype(font_path, font_size)
        asc, desc = self.font.getmetrics()
        bbox = self.font.getbbox("M")
        self.cw = bbox[2] - bbox[0]  # character cell width
        self.ch = asc + desc  # CRITICAL: not textbbox height

        self.cols = VW // self.cw
        self.rows = VH // self.ch
        self.ox = (VW - self.cols * self.cw) // 2  # centering
        self.oy = (VH - self.rows * self.ch) // 2

        # Index arrays
        self.rr = np.arange(self.rows, dtype=np.float32)[:, None]
        self.cc = np.arange(self.cols, dtype=np.float32)[None, :]

        # Polar coordinates (aspect-corrected)
        cx, cy = self.cols / 2.0, self.rows / 2.0
        asp = self.cw / self.ch
        self.dx = self.cc - cx
        self.dy = (self.rr - cy) * asp
        self.dist = np.sqrt(self.dx**2 + self.dy**2)
        self.angle = np.arctan2(self.dy, self.dx)

        # Normalized (0-1 range) -- for distance falloff
        self.dx_n = (self.cc - cx) / max(self.cols, 1)
        self.dy_n = (self.rr - cy) / max(self.rows, 1) * asp
        self.dist_n = np.sqrt(self.dx_n**2 + self.dy_n**2)

        # Pre-rasterize all characters to float32 bitmaps
        self.bm = {}
        for c in all_chars:
            img = Image.new("L", (self.cw, self.ch), 0)
            ImageDraw.Draw(img).text((0, 0), c, fill=255, font=self.font)
            self.bm[c] = np.array(img, dtype=np.float32) / 255.0
```

### Character Render Loop

The bottleneck. Composites pre-rasterized bitmaps onto pixel canvas:

```python
def render(self, chars, colors, canvas=None):
    if canvas is None:
        canvas = np.zeros((VH, VW, 3), dtype=np.uint8)
    for row in range(self.rows):
        y = self.oy + row * self.ch
        if y + self.ch > VH: break
        for col in range(self.cols):
            c = chars[row, col]
            if c == " ": continue
            x = self.ox + col * self.cw
            if x + self.cw > VW: break
            a = self.bm[c]  # float32 bitmap
            canvas[y:y+self.ch, x:x+self.cw] = np.maximum(
                canvas[y:y+self.ch, x:x+self.cw],
                (a[:, :, None] * colors[row, col]).astype(np.uint8))
    return canvas
```

Use `np.maximum` for additive blending (brighter chars overwrite dimmer ones, never darken).

### Multi-Layer Rendering

Render multiple grids onto the same canvas for depth:

```python
canvas = np.zeros((VH, VW, 3), dtype=np.uint8)
canvas = grid_lg.render(bg_chars, bg_colors, canvas)   # background layer
canvas = grid_md.render(main_chars, main_colors, canvas)  # main layer
canvas = grid_sm.render(detail_chars, detail_colors, canvas)  # detail overlay
```

---

## Character Palettes

### Design Principles

Character palettes are the primary visual texture of ASCII video. They control not just brightness mapping but the entire visual feel. Design palettes intentionally:

- **Visual weight**: characters sorted by the amount of ink/pixels they fill. Space is always index 0.
- **Coherence**: characters within a palette should belong to the same visual family.
- **Density curve**: the brightness-to-character mapping is nonlinear. Dense palettes (many chars) give smoother gradients; sparse palettes (5-8 chars) give posterized/graphic looks.
- **Rendering compatibility**: every character in the palette must exist in the font. Test at init and remove missing glyphs.

### Palette Library

Organized by visual family. Mix and match per project -- don't default to PAL_DEFAULT for everything.

#### Density / Brightness Palettes
```python
PAL_DEFAULT  = " .`'-:;!><=+*^~?/|(){}[]#&$@%"       # classic ASCII art
PAL_DENSE    = " .:;+=xX$#@\u2588"                          # simple 11-level ramp
PAL_MINIMAL  = " .:-=+#@"                               # 8-level, graphic
PAL_BINARY   = " \u2588"                                      # 2-level, extreme contrast
PAL_GRADIENT = " \u2591\u2592\u2593\u2588"                              # 4-level block gradient
```

#### Unicode Block Elements
```python
PAL_BLOCKS   = " \u2591\u2592\u2593\u2588\u2584\u2580\u2590\u258c"                 # standard blocks
PAL_BLOCKS_EXT = " \u2596\u2597\u2598\u2599\u259a\u259b\u259c\u259d\u259e\u259f\u2591\u2592\u2593\u2588"  # quadrant blocks (more detail)
PAL_SHADE    = " \u2591\u2592\u2593\u2588\u2587\u2586\u2585\u2584\u2583\u2582\u2581"          # vertical fill progression
```

#### Symbolic / Thematic
```python
PAL_MATH     = " \u00b7\u2218\u2219\u2022\u00b0\u00b1\u2213\u00d7\u00f7\u2248\u2260\u2261\u2264\u2265\u221e\u222b\u2211\u220f\u221a\u2207\u2202\u2206\u03a9"    # math symbols
PAL_BOX      = " \u2500\u2502\u250c\u2510\u2514\u2518\u251c\u2524\u252c\u2534\u253c\u2550\u2551\u2554\u2557\u255a\u255d\u2560\u2563\u2566\u2569\u256c"          # box drawing
PAL_CIRCUIT  = " .\u00b7\u2500\u2502\u250c\u2510\u2514\u2518\u253c\u25cb\u25cf\u25a1\u25a0\u2206\u2207\u2261"                 # circuit board
PAL_RUNE     = " .\u16a0\u16a2\u16a6\u16b1\u16b7\u16c1\u16c7\u16d2\u16d6\u16da\u16de\u16df"                   # elder futhark runes
PAL_ALCHEMIC = " \u2609\u263d\u2640\u2642\u2643\u2644\u2645\u2646\u2647\u2648\u2649\u264a\u264b"            # planetary/alchemical symbols
PAL_ZODIAC   = " \u2648\u2649\u264a\u264b\u264c\u264d\u264e\u264f\u2650\u2651\u2652\u2653"            # zodiac
PAL_ARROWS   = " \u2190\u2191\u2192\u2193\u2194\u2195\u2196\u2197\u2198\u2199\u21a9\u21aa\u21bb\u27a1"             # directional arrows
PAL_MUSIC    = " \u266a\u266b\u266c\u2669\u266d\u266e\u266f\u25cb\u25cf"                       # musical notation
```

#### Script / Writing System
```python
PAL_KATA     = " \u00b7\uff66\uff67\uff68\uff69\uff6a\uff6b\uff6c\uff6d\uff6e\uff6f\uff70\uff71\uff72\uff73\uff74\uff75\uff76\uff77"          # katakana halfwidth (matrix rain)
PAL_GREEK    = " \u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03b9\u03ba\u03bb\u03bc\u03bd\u03be\u03c0\u03c1\u03c3\u03c4\u03c6\u03c8\u03c9"    # Greek lowercase
PAL_CYRILLIC = " \u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448"  # Cyrillic lowercase
PAL_ARABIC   = " \u0627\u0628\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637"       # Arabic letters (isolated forms)
```

#### Dot / Point Progressions
```python
PAL_DOTS     = " \u22c5\u2218\u2219\u25cf\u25c9\u25ce\u25c6\u2726\u2605"                   # dot size progression
PAL_BRAILLE  = " \u2801\u2802\u2803\u2804\u2805\u2806\u2807\u2808\u2809\u280a\u280b\u280c\u280d\u280e\u280f\u2810\u2811\u2812\u2813\u2814\u2815\u2816\u2817\u2818\u2819\u281a\u281b\u281c\u281d\u281e\u281f\u283f"  # braille patterns
PAL_STARS    = " \u00b7\u2727\u2726\u2729\u2728\u2605\u2736\u2733\u2738"               # star progression
```

#### Project-Specific (examples -- invent new ones per project)
```python
PAL_HERMES   = " .\u00b7~=\u2248\u221e\u26a1\u263f\u2726\u2605\u2295\u25ca\u25c6\u25b2\u25bc\u25cf\u25a0"   # mythology/tech blend
PAL_OCEAN    = " ~\u2248\u2248\u2248\u223c\u2307\u2248\u224b\u224c\u2248"                       # water/wave characters
PAL_ORGANIC  = " .\u00b0\u2218\u2022\u25e6\u25c9\u2742\u273f\u2741\u2743"                 # growing/botanical
PAL_MACHINE  = " _\u2500\u2502\u250c\u2510\u253c\u2261\u25a0\u2588\u2593\u2592\u2591"             # mechanical/industrial
```

### Creating Custom Palettes

When designing for a project, build palettes from the content's theme:

1. **Choose a visual family** (dots, blocks, symbols, script)
2. **Sort by visual weight** -- render each char at target font size, count lit pixels, sort ascending
3. **Test at target grid size** -- some chars collapse to blobs at small sizes
4. **Validate in font** -- remove chars the font can't render:

```python
def validate_palette(pal, font):
    """Remove characters the font can't render."""
    valid = []
    for c in pal:
        if c == " ":
            valid.append(c)
            continue
        img = Image.new("L", (20, 20), 0)
        ImageDraw.Draw(img).text((0, 0), c, fill=255, font=font)
        if np.array(img).max() > 0:  # char actually rendered something
            valid.append(c)
    return "".join(valid)
```

### Mapping Values to Characters

```python
def val2char(v, mask, pal=PAL_DEFAULT):
    """Map float array (0-1) to character array using palette."""
    n = len(pal)
    idx = np.clip((v * n).astype(int), 0, n - 1)
    out = np.full(v.shape, " ", dtype="U1")
    for i, ch in enumerate(pal):
        out[mask & (idx == i)] = ch
    return out
```

**Nonlinear mapping** for different visual curves:

```python
def val2char_gamma(v, mask, pal, gamma=1.0):
    """Gamma-corrected palette mapping. gamma<1 = brighter, gamma>1 = darker."""
    v_adj = np.power(np.clip(v, 0, 1), gamma)
    return val2char(v_adj, mask, pal)

def val2char_step(v, mask, pal, thresholds):
    """Custom threshold mapping. thresholds = list of float breakpoints."""
    out = np.full(v.shape, pal[0], dtype="U1")
    for i, thr in enumerate(thresholds):
        out[mask & (v > thr)] = pal[min(i + 1, len(pal) - 1)]
    return out
```

---

## Color System

### HSV->RGB (Vectorized)

All color computation in HSV for intuitive control, converted at render time:

```python
def hsv2rgb(h, s, v):
    """Vectorized HSV->RGB. h,s,v are numpy arrays. Returns (R,G,B) uint8 arrays."""
    h = h % 1.0
    c = v * s; x = c * (1 - np.abs((h*6) % 2 - 1)); m = v - c
    # ... 6 sector assignment ...
    return (np.clip((r+m)*255, 0, 255).astype(np.uint8),
            np.clip((g+m)*255, 0, 255).astype(np.uint8),
            np.clip((b+m)*255, 0, 255).astype(np.uint8))
```

### Color Mapping Strategies

Don't default to a single strategy. Choose based on the visual intent:

| Strategy | Hue source | Effect | Good for |
|----------|------------|--------|----------|
| Angle-mapped | `g.angle / (2*pi)` | Rainbow around center | Radial effects, kaleidoscopes |
| Distance-mapped | `g.dist_n * 0.3` | Gradient from center | Tunnels, depth effects |
| Frequency-mapped | `f["cent"] * 0.2` | Timbral color shifting | Audio-reactive |
| Value-mapped | `val * 0.15` | Brightness-dependent hue | Fire, heat maps |
| Time-cycled | `t * rate` | Slow color rotation | Ambient, chill |
| Source-sampled | Video frame pixel colors | Preserve original color | Video-to-ASCII |
| Palette-indexed | Discrete color lookup | Flat graphic style | Retro, pixel art |
| Temperature | Blend between warm/cool | Emotional tone | Mood-driven scenes |
| Complementary | `hue` and `hue + 0.5` | High contrast | Bold, dramatic |
| Triadic | `hue`, `hue + 0.33`, `hue + 0.66` | Vibrant, balanced | Psychedelic |
| Analogous | `hue +/- 0.08` | Harmonious, subtle | Elegant, cohesive |
| Monochrome | Fixed hue, vary S and V | Restrained, focused | Noir, minimal |

### Color Palettes (Discrete RGB)

For non-HSV workflows -- direct RGB color sets for graphic/retro looks:

```python
# Named color palettes -- use for flat/graphic styles or per-character coloring
COLORS_NEON = [(255,0,102), (0,255,153), (102,0,255), (255,255,0), (0,204,255)]
COLORS_PASTEL = [(255,179,186), (255,223,186), (255,255,186), (186,255,201), (186,225,255)]
COLORS_MONO_GREEN = [(0,40,0), (0,80,0), (0,140,0), (0,200,0), (0,255,0)]
COLORS_MONO_AMBER = [(40,20,0), (80,50,0), (140,90,0), (200,140,0), (255,191,0)]
COLORS_CYBERPUNK = [(255,0,60), (0,255,200), (180,0,255), (255,200,0)]
COLORS_VAPORWAVE = [(255,113,206), (1,205,254), (185,103,255), (5,255,161)]
COLORS_EARTH = [(86,58,26), (139,90,43), (189,154,91), (222,193,136), (245,230,193)]
COLORS_ICE = [(200,230,255), (150,200,240), (100,170,230), (60,130,210), (30,80,180)]
COLORS_BLOOD = [(80,0,0), (140,10,10), (200,20,20), (255,50,30), (255,100,80)]
COLORS_FOREST = [(10,30,10), (20,60,15), (30,100,20), (50,150,30), (80,200,50)]

def rgb_palette_map(val, mask, palette):
    """Map float array (0-1) to RGB colors from a discrete palette."""
    n = len(palette)
    idx = np.clip((val * n).astype(int), 0, n - 1)
    R = np.zeros(val.shape, dtype=np.uint8)
    G = np.zeros(val.shape, dtype=np.uint8)
    B = np.zeros(val.shape, dtype=np.uint8)
    for i, (r, g, b) in enumerate(palette):
        m = mask & (idx == i)
        R[m] = r; G[m] = g; B[m] = b
    return R, G, B
```

### Compositing Helpers

```python
def mkc(R, G, B, rows, cols):
    """Pack 3 uint8 arrays into (rows, cols, 3) color array."""
    o = np.zeros((rows, cols, 3), dtype=np.uint8)
    o[:,:,0] = R; o[:,:,1] = G; o[:,:,2] = B
    return o

def layer_over(base_ch, base_co, top_ch, top_co):
    """Composite top layer onto base. Non-space chars overwrite."""
    m = top_ch != " "
    base_ch[m] = top_ch[m]; base_co[m] = top_co[m]
    return base_ch, base_co

def layer_blend(base_co, top_co, alpha):
    """Alpha-blend top color layer onto base. alpha is float array (0-1) or scalar."""
    if isinstance(alpha, (int, float)):
        alpha = np.full(base_co.shape[:2], alpha, dtype=np.float32)
    a = alpha[:,:,None]
    return np.clip(base_co * (1 - a) + top_co * a, 0, 255).astype(np.uint8)

def stamp(ch, co, text, row, col, color=(255,255,255)):
    """Write text string at position."""
    for i, c in enumerate(text):
        cc = col + i
        if 0 <= row < ch.shape[0] and 0 <= cc < ch.shape[1]:
            ch[row, cc] = c; co[row, cc] = color
```

---

## Section System

Map time ranges to effect functions + shader configs + grid sizes:

```python
SECTIONS = [
    (0.0, "void"), (3.94, "starfield"), (21.0, "matrix"),
    (46.0, "drop"), (130.0, "glitch"), (187.0, "outro"),
]

FX_DISPATCH = {"void": fx_void, "starfield": fx_starfield, ...}
SECTION_FX = {"void": {"vignette": 0.3, "bloom": 170}, ...}
SECTION_GRID = {"void": "md", "starfield": "sm", "drop": "lg", ...}
SECTION_MIRROR = {"drop": "h", "bass_rings": "quad"}

def get_section(t):
    sec = SECTIONS[0][1]
    for ts, name in SECTIONS:
        if t >= ts: sec = name
    return sec
```

---

## Parallel Encoding

Split frames across N workers. Each pipes raw RGB to its own ffmpeg subprocess:

```python
def render_batch(batch_id, frame_start, frame_end, features, seg_path):
    r = Renderer()
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-pix_fmt", "yuv420p", seg_path]

    # CRITICAL: stderr to file, not pipe
    stderr_fh = open(os.path.join(workdir, f"err_{batch_id:02d}.log"), "w")
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=stderr_fh)

    for fi in range(frame_start, frame_end):
        t = fi / FPS
        sec = get_section(t)
        f = {k: float(features[k][fi]) for k in features}
        ch, co = FX_DISPATCH[sec](r, f, t)
        canvas = r.render(ch, co)
        canvas = apply_mirror(canvas, sec, f)
        canvas = apply_shaders(canvas, sec, f, t)
        pipe.stdin.write(canvas.tobytes())

    pipe.stdin.close()
    pipe.wait()
    stderr_fh.close()
```

Concatenate segments + mux audio:

```python
# Write concat file
with open(concat_path, "w") as cf:
    for seg in segments:
        cf.write(f"file '{seg}'\n")

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
                "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", output_path])
```

## Effect Function Contract

### v2 Protocol (Current)

Every scene function: `(renderer, features_dict, time_float, state_dict) -> canvas_uint8`

```python
def fx_example(r, f, t, S):
    """Scene function returns a full pixel canvas (uint8 H,W,3).
    Scenes have full control over multi-grid rendering and pixel-level composition.
    """
    # Render multiple layers at different grid densities
    canvas_a = _render_vf(r, "md", vf_plasma, hf_angle(0.0), PAL_DENSE, f, t, S)
    canvas_b = _render_vf(r, "sm", vf_vortex, hf_time_cycle(0.1), PAL_RUNE, f, t, S)

    # Pixel-level blend
    result = blend_canvas(canvas_a, canvas_b, "screen", 0.8)
    return result
```

See `references/scenes.md` for the full scene protocol, the Renderer class, `_render_vf()` helper, and complete scene examples.

See `references/composition.md` for blend modes, tone mapping, feedback buffers, and multi-grid composition.

### v1 Protocol (Legacy)

Simple scenes that use a single grid can still return `(chars, colors)` and let the caller handle rendering, but the v2 canvas protocol is preferred for all new code.

```python
def fx_simple(r, f, t, S):
    g = r.get_grid("md")
    val = np.sin(g.dist * 0.1 - t * 3) * f.get("bass", 0.3) * 2
    val = np.clip(val, 0, 1); mask = val > 0.03
    ch = val2char(val, mask, PAL_DEFAULT)
    R, G, B = hsv2rgb(np.full_like(val, 0.6), np.full_like(val, 0.7), val)
    co = mkc(R, G, B, g.rows, g.cols)
    return g.render(ch, co)  # returns canvas directly
```

### Persistent State

Effects that need state across frames (particles, rain columns) use the `S` dict parameter (which is `r.S` — same object, but passed explicitly for clarity):

```python
def fx_with_state(r, f, t, S):
    if "particles" not in S:
        S["particles"] = initialize_particles()
    update_particles(S["particles"])
    # ...
```

State persists across frames within a single scene/clip. Each worker process (and each scene) gets its own independent state.

### Helper Functions

```python
def hsv2rgb_scalar(h, s, v):
    """Single-value HSV to RGB. Returns (R, G, B) tuple of ints 0-255."""
    h = h % 1.0
    c = v * s; x = c * (1 - abs((h * 6) % 2 - 1)); m = v - c
    if h * 6 < 1:   r, g, b = c, x, 0
    elif h * 6 < 2:  r, g, b = x, c, 0
    elif h * 6 < 3:  r, g, b = 0, c, x
    elif h * 6 < 4:  r, g, b = 0, x, c
    elif h * 6 < 5:  r, g, b = x, 0, c
    else:             r, g, b = c, 0, x
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))

def log(msg):
    """Print timestamped log message."""
    print(msg, flush=True)
```
