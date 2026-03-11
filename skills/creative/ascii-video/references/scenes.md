# Scene System Reference

Scenes are the top-level creative unit. Each scene is a time-bounded segment with its own effect function, shader chain, feedback configuration, and tone-mapping gamma.

## Scene Protocol (v2)

### Function Signature

```python
def fx_scene_name(r, f, t, S) -> canvas:
    """
    Args:
        r: Renderer instance — access multiple grids via r.get_grid("sm")
        f: dict of audio/video features, all values normalized to [0, 1]
        t: time in seconds (global, not local to scene)
        S: dict for persistent state (particles, rain columns, etc.)

    Returns:
        canvas: numpy uint8 array, shape (VH, VW, 3) — full pixel frame
    """
```

This replaces the v1 protocol where scenes returned `(chars, colors)` tuples. The v2 protocol gives scenes full control over multi-grid rendering and pixel-level composition internally.

### The Renderer Class

```python
class Renderer:
    def __init__(self):
        self.grids = {}   # lazy-initialized grid cache
        self.g = None      # "active" grid (for backward compat)
        self.S = {}        # persistent state dict

    def get_grid(self, key):
        """Get or create a GridLayer by size key."""
        if key not in self.grids:
            sizes = {"xs": 8, "sm": 10, "md": 16, "lg": 20, "xl": 24, "xxl": 40}
            self.grids[key] = GridLayer(FONT_PATH, sizes[key])
        return self.grids[key]

    def set_grid(self, key):
        """Set active grid (legacy). Prefer get_grid() for multi-grid scenes."""
        self.g = self.get_grid(key)
        return self.g
```

**Key difference from v1**: scenes call `r.get_grid("sm")`, `r.get_grid("lg")`, etc. to access multiple grids. Each grid is lazy-initialized and cached. The `set_grid()` method still works for single-grid scenes.

### Minimal Scene (Single Grid)

```python
def fx_simple_rings(r, f, t, S):
    """Single-grid scene: rings with distance-mapped hue."""
    canvas = _render_vf(r, "md",
        lambda g, f, t, S: vf_rings(g, f, t, S, n_base=8, spacing_base=3),
        hf_distance(0.3, 0.02), PAL_STARS, f, t, S, sat=0.85)
    return canvas
```

### Standard Scene (Two Grids + Blend)

```python
def fx_tunnel_ripple(r, f, t, S):
    """Two-grid scene: tunnel depth exclusion-blended with ripple."""
    canvas_a = _render_vf(r, "md",
        lambda g, f, t, S: vf_tunnel(g, f, t, S, speed=5.0, complexity=10) * 1.3,
        hf_distance(0.55, 0.02), PAL_GREEK, f, t, S, sat=0.7)

    canvas_b = _render_vf(r, "sm",
        lambda g, f, t, S: vf_ripple(g, f, t, S,
            sources=[(0.3,0.3), (0.7,0.7), (0.5,0.2)], freq=0.5, damping=0.012) * 1.4,
        hf_angle(0.1), PAL_STARS, f, t, S, sat=0.8)

    return blend_canvas(canvas_a, canvas_b, "exclusion", 0.8)
```

### Complex Scene (Three Grids + Conditional + Custom Rendering)

```python
def fx_rings_explosion(r, f, t, S):
    """Three-grid scene with particles and conditional kaleidoscope."""
    # Layer 1: rings
    canvas_a = _render_vf(r, "sm",
        lambda g, f, t, S: vf_rings(g, f, t, S, n_base=10, spacing_base=2) * 1.4,
        lambda g, f, t, S: (g.angle / (2*np.pi) + t * 0.15) % 1.0,
        PAL_STARS, f, t, S, sat=0.9)

    # Layer 2: vortex on different grid
    canvas_b = _render_vf(r, "md",
        lambda g, f, t, S: vf_vortex(g, f, t, S, twist=6.0) * 1.2,
        hf_time_cycle(0.15), PAL_BLOCKS, f, t, S, sat=0.8)

    result = blend_canvas(canvas_b, canvas_a, "screen", 0.7)

    # Layer 3: particles (custom rendering, not _render_vf)
    g = r.get_grid("sm")
    if "px" not in S:
        S["px"], S["py"], S["vx"], S["vy"], S["life"], S["pch"] = (
            [], [], [], [], [], [])
    if f.get("beat", 0) > 0.5:
        chars = list("\u2605\u2736\u2733\u2738\u2726\u2728*+")
        for _ in range(int(80 + f.get("rms", 0.3) * 120)):
            ang = random.uniform(0, 2 * math.pi)
            sp = random.uniform(1, 10) * (0.5 + f.get("sub_r", 0.3) * 2)
            S["px"].append(float(g.cols // 2))
            S["py"].append(float(g.rows // 2))
            S["vx"].append(math.cos(ang) * sp * 2.5)
            S["vy"].append(math.sin(ang) * sp)
            S["life"].append(1.0)
            S["pch"].append(random.choice(chars))

    # Update + draw particles
    ch_p = np.full((g.rows, g.cols), " ", dtype="U1")
    co_p = np.zeros((g.rows, g.cols, 3), dtype=np.uint8)
    i = 0
    while i < len(S["px"]):
        S["px"][i] += S["vx"][i]; S["py"][i] += S["vy"][i]
        S["vy"][i] += 0.03; S["life"][i] -= 0.02
        if S["life"][i] <= 0:
            for k in ("px","py","vx","vy","life","pch"): S[k].pop(i)
        else:
            pr, pc = int(S["py"][i]), int(S["px"][i])
            if 0 <= pr < g.rows and 0 <= pc < g.cols:
                ch_p[pr, pc] = S["pch"][i]
                co_p[pr, pc] = hsv2rgb_scalar(
                    0.08 + (1-S["life"][i])*0.15, 0.95, S["life"][i])
            i += 1

    canvas_p = g.render(ch_p, co_p)
    result = blend_canvas(result, canvas_p, "add", 0.8)

    # Conditional kaleidoscope on strong beats
    if f.get("bdecay", 0) > 0.4:
        result = sh_kaleidoscope(result.copy(), folds=6)

    return result
```

### Scene with Custom Character Rendering (Matrix Rain)

When you need per-cell control beyond what `_render_vf()` provides:

```python
def fx_matrix_layered(r, f, t, S):
    """Matrix rain blended with tunnel — two grids, screen blend."""
    # Layer 1: Matrix rain (custom per-column rendering)
    g = r.get_grid("md")
    rows, cols = g.rows, g.cols
    pal = PAL_KATA

    if "ry" not in S or len(S["ry"]) != cols:
        S["ry"] = np.random.uniform(-rows, rows, cols).astype(np.float32)
        S["rsp"] = np.random.uniform(0.3, 2.0, cols).astype(np.float32)
        S["rln"] = np.random.randint(8, 35, cols)
        S["rch"] = np.random.randint(1, len(pal), (rows, cols))

    speed = 0.6 + f.get("bass", 0.3) * 3
    if f.get("beat", 0) > 0.5: speed *= 2.5
    S["ry"] += S["rsp"] * speed

    ch = np.full((rows, cols), " ", dtype="U1")
    co = np.zeros((rows, cols, 3), dtype=np.uint8)
    heads = S["ry"].astype(int)
    for c in range(cols):
        head = heads[c]
        for i in range(S["rln"][c]):
            row = head - i
            if 0 <= row < rows:
                fade = 1.0 - i / S["rln"][c]
                ch[row, c] = pal[S["rch"][row, c] % len(pal)]
                if i == 0:
                    v = int(min(255, fade * 300))
                    co[row, c] = (int(v*0.9), v, int(v*0.9))
                else:
                    v = int(fade * 240)
                    co[row, c] = (int(v*0.1), v, int(v*0.4))
    canvas_a = g.render(ch, co)

    # Layer 2: Tunnel on sm grid for depth texture
    canvas_b = _render_vf(r, "sm",
        lambda g, f, t, S: vf_tunnel(g, f, t, S, speed=5.0, complexity=10),
        hf_distance(0.3, 0.02), PAL_BLOCKS, f, t, S, sat=0.6)

    return blend_canvas(canvas_a, canvas_b, "screen", 0.5)
```

---

## Scene Table

The scene table defines the timeline: which scene plays when, with what configuration.

### Structure

```python
SCENES = [
    {
        "start": 0.0,           # start time in seconds
        "end": 3.96,            # end time in seconds
        "name": "starfield",    # identifier (used for clip filenames)
        "grid": "sm",           # default grid (for render_clip setup)
        "fx": fx_starfield,     # scene function reference (must be module-level)
        "gamma": 0.75,          # tonemap gamma override (default 0.75)
        "shaders": [            # shader chain (applied after tonemap + feedback)
            ("bloom", {"thr": 120}),
            ("vignette", {"s": 0.2}),
            ("grain", {"amt": 8}),
        ],
        "feedback": None,       # feedback buffer config (None = disabled)
        # "feedback": {"decay": 0.8, "blend": "screen", "opacity": 0.3,
        #              "transform": "zoom", "transform_amt": 0.02, "hue_shift": 0.02},
    },
    {
        "start": 3.96,
        "end": 6.58,
        "name": "matrix_layered",
        "grid": "md",
        "fx": fx_matrix_layered,
        "shaders": [
            ("crt", {"strength": 0.05}),
            ("scanlines", {"intensity": 0.12}),
            ("color_grade", {"tint": (0.7, 1.2, 0.7)}),
            ("bloom", {"thr": 100}),
        ],
        "feedback": {"decay": 0.5, "blend": "add", "opacity": 0.2},
    },
    # ... more scenes ...
]
```

### Beat-Synced Scene Cutting

Derive cut points from audio analysis:

```python
# Get beat timestamps
beats = [fi / FPS for fi in range(N_FRAMES) if features["beat"][fi] > 0.5]

# Group beats into phrase boundaries (every 4-8 beats)
cuts = [0.0]
for i in range(0, len(beats), 4):  # cut every 4 beats
    cuts.append(beats[i])
cuts.append(DURATION)

# Or use the music's structure: silence gaps, energy changes
energy = features["rms"]
# Find timestamps where energy drops significantly -> natural break points
```

### `render_clip()` — The Render Loop

This function renders one scene to a clip file:

```python
def render_clip(seg, features, clip_path):
    r = Renderer()
    r.set_grid(seg["grid"])
    S = r.S
    random.seed(hash(seg["id"]) + 42)  # deterministic per scene

    # Build shader chain from config
    chain = ShaderChain()
    for shader_name, kwargs in seg.get("shaders", []):
        chain.add(shader_name, **kwargs)

    # Setup feedback buffer
    fb = None
    fb_cfg = seg.get("feedback", None)
    if fb_cfg:
        fb = FeedbackBuffer()

    fx_fn = seg["fx"]

    # Open ffmpeg pipe
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", clip_path]
    stderr_fh = open(clip_path.replace(".mp4", ".log"), "w")
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=stderr_fh)

    for fi in range(seg["frame_start"], seg["frame_end"]):
        t = fi / FPS
        feat = {k: float(features[k][fi]) for k in features}

        # 1. Scene renders canvas
        canvas = fx_fn(r, feat, t, S)

        # 2. Tonemap normalizes brightness
        canvas = tonemap(canvas, gamma=seg.get("gamma", 0.75))

        # 3. Feedback adds temporal recursion
        if fb and fb_cfg:
            canvas = fb.apply(canvas, **{k: fb_cfg[k] for k in fb_cfg})

        # 4. Shader chain adds post-processing
        canvas = chain.apply(canvas, f=feat, t=t)

        pipe.stdin.write(canvas.tobytes())

    pipe.stdin.close(); pipe.wait(); stderr_fh.close()
```

### Building Segments from Scene Table

```python
segments = []
for i, scene in enumerate(SCENES):
    segments.append({
        "id": f"s{i:02d}_{scene['name']}",
        "name": scene["name"],
        "grid": scene["grid"],
        "fx": scene["fx"],
        "shaders": scene.get("shaders", []),
        "feedback": scene.get("feedback", None),
        "gamma": scene.get("gamma", 0.75),
        "frame_start": int(scene["start"] * FPS),
        "frame_end": int(scene["end"] * FPS),
    })
```

### Parallel Rendering

Scenes are independent units dispatched to a process pool:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {
        pool.submit(render_clip, seg, features, clip_path): seg["id"]
        for seg, clip_path in zip(segments, clip_paths)
    }
    for fut in as_completed(futures):
        try:
            fut.result()
        except Exception as e:
            log(f"ERROR {futures[fut]}: {e}")
```

**Pickling constraint**: `ProcessPoolExecutor` serializes arguments via pickle. Module-level functions can be pickled; lambdas and closures cannot. All `fx_*` scene functions MUST be defined at module level, not as closures or class methods.

### Test-Frame Mode

Render a single frame at a specific timestamp to verify visuals without a full render:

```python
if args.test_frame >= 0:
    fi = min(int(args.test_frame * FPS), N_FRAMES - 1)
    t = fi / FPS
    feat = {k: float(features[k][fi]) for k in features}
    scene = next(sc for sc in reversed(SCENES) if t >= sc["start"])
    r = Renderer()
    r.set_grid(scene["grid"])
    canvas = scene["fx"](r, feat, t, r.S)
    canvas = tonemap(canvas, gamma=scene.get("gamma", 0.75))
    chain = ShaderChain()
    for sn, kw in scene.get("shaders", []):
        chain.add(sn, **kw)
    canvas = chain.apply(canvas, f=feat, t=t)
    Image.fromarray(canvas).save(f"test_{args.test_frame:.1f}s.png")
    print(f"Mean brightness: {canvas.astype(float).mean():.1f}")
```

CLI: `python reel.py --test-frame 10.0`

---

## Scene Design Checklist

For each scene:

1. **Choose 2-3 grid sizes** — different scales create interference
2. **Choose different value fields** per layer — don't use the same effect on every grid
3. **Choose different hue fields** per layer — or at minimum different hue offsets
4. **Choose different palettes** per layer — mixing PAL_RUNE with PAL_BLOCKS looks different from PAL_RUNE with PAL_DENSE
5. **Choose a blend mode** that matches the energy — screen for bright, difference for psychedelic, exclusion for subtle
6. **Add conditional effects** on beat — kaleidoscope, mirror, glitch
7. **Configure feedback** for trailing/recursive looks — or None for clean cuts
8. **Set gamma** if using destructive shaders (solarize, posterize)
9. **Test with --test-frame** at the scene's midpoint before full render
