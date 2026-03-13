# Scene Examples

**Cross-references:**
- Grid system, palettes, color (HSV + OKLAB): `architecture.md`
- Effect building blocks (value fields, noise, SDFs, particles): `effects.md`
- `_render_vf()`, blend modes, tonemap, masking: `composition.md`
- Scene protocol, render_clip, SCENES table: `scenes.md`
- Shader pipeline, feedback buffer, ShaderChain: `shaders.md`
- Input sources (audio features, video features): `inputs.md`
- Performance tuning: `optimization.md`
- Common bugs: `troubleshooting.md`

Copy-paste-ready scene functions at increasing complexity. Each is a complete, working v2 scene function that returns a pixel canvas. See `scenes.md` for the scene protocol and `composition.md` for blend modes and tonemap.

---

## Minimal — Single Grid, Single Effect

### Breathing Plasma

One grid, one value field, one hue field. The simplest possible scene.

```python
def fx_breathing_plasma(r, f, t, S):
    """Plasma field with time-cycling hue. Audio modulates brightness."""
    canvas = _render_vf(r, "md",
        lambda g, f, t, S: vf_plasma(g, f, t, S) * 1.3,
        hf_time_cycle(0.08), PAL_DENSE, f, t, S, sat=0.8)
    return canvas
```

### Reaction-Diffusion Coral

Single grid, simulation-based field. Evolves organically over time.

```python
def fx_coral(r, f, t, S):
    """Gray-Scott reaction-diffusion — coral branching pattern.
    Slow-evolving, organic. Best for ambient/chill sections."""
    canvas = _render_vf(r, "sm",
        lambda g, f, t, S: vf_reaction_diffusion(g, f, t, S,
            feed=0.037, kill=0.060, steps_per_frame=6, init_mode="center"),
        hf_distance(0.55, 0.015), PAL_DOTS, f, t, S, sat=0.7)
    return canvas
```

### SDF Geometry

Geometric shapes from SDFs. Clean, precise, graphic.

```python
def fx_sdf_rings(r, f, t, S):
    """Concentric SDF rings with smooth pulsing."""
    def val_fn(g, f, t, S):
        d1 = sdf_ring(g, radius=0.15 + f.get("bass", 0.3) * 0.05, thickness=0.015)
        d2 = sdf_ring(g, radius=0.25 + f.get("mid", 0.3) * 0.05, thickness=0.012)
        d3 = sdf_ring(g, radius=0.35 + f.get("hi", 0.3) * 0.04, thickness=0.010)
        combined = sdf_smooth_union(sdf_smooth_union(d1, d2, 0.05), d3, 0.05)
        return sdf_glow(combined, falloff=0.08) * (0.5 + f.get("rms", 0.3) * 0.8)
    canvas = _render_vf(r, "md", val_fn, hf_angle(0.0), PAL_STARS, f, t, S, sat=0.85)
    return canvas
```

---

## Standard — Two Grids + Blend

### Tunnel Through Noise

Two grids at different densities, screen blended. The fine noise texture shows through the coarser tunnel characters.

```python
def fx_tunnel_noise(r, f, t, S):
    """Tunnel depth on md grid + fBM noise on sm grid, screen blended."""
    canvas_a = _render_vf(r, "md",
        lambda g, f, t, S: vf_tunnel(g, f, t, S, speed=4.0, complexity=8) * 1.2,
        hf_distance(0.5, 0.02), PAL_BLOCKS, f, t, S, sat=0.7)

    canvas_b = _render_vf(r, "sm",
        lambda g, f, t, S: vf_fbm(g, f, t, S, octaves=4, freq=0.05, speed=0.15) * 1.3,
        hf_time_cycle(0.06), PAL_RUNE, f, t, S, sat=0.6)

    return blend_canvas(canvas_a, canvas_b, "screen", 0.7)
```

### Voronoi Cells + Spiral Overlay

Voronoi cell edges with a spiral arm pattern overlaid.

```python
def fx_voronoi_spiral(r, f, t, S):
    """Voronoi edge detection on md + logarithmic spiral on lg."""
    canvas_a = _render_vf(r, "md",
        lambda g, f, t, S: vf_voronoi(g, f, t, S,
            n_cells=15, mode="edge", edge_width=2.0, speed=0.4),
        hf_angle(0.2), PAL_CIRCUIT, f, t, S, sat=0.75)

    canvas_b = _render_vf(r, "lg",
        lambda g, f, t, S: vf_spiral(g, f, t, S, n_arms=4, tightness=3.0) * 1.2,
        hf_distance(0.1, 0.03), PAL_BLOCKS, f, t, S, sat=0.9)

    return blend_canvas(canvas_a, canvas_b, "exclusion", 0.6)
```

### Domain-Warped fBM

Two layers of the same fBM, one domain-warped, difference-blended for psychedelic organic texture.

```python
def fx_organic_warp(r, f, t, S):
    """Clean fBM vs domain-warped fBM, difference blended."""
    canvas_a = _render_vf(r, "sm",
        lambda g, f, t, S: vf_fbm(g, f, t, S, octaves=5, freq=0.04, speed=0.1),
        hf_plasma(0.2), PAL_DENSE, f, t, S, sat=0.6)

    canvas_b = _render_vf(r, "md",
        lambda g, f, t, S: vf_domain_warp(g, f, t, S,
            warp_strength=20.0, freq=0.05, speed=0.15),
        hf_time_cycle(0.05), PAL_BRAILLE, f, t, S, sat=0.7)

    return blend_canvas(canvas_a, canvas_b, "difference", 0.7)
```

---

## Complex — Three Grids + Conditional + Feedback

### Psychedelic Cathedral

Three-grid composition with beat-triggered kaleidoscope and feedback zoom tunnel. The most visually complex pattern.

```python
def fx_cathedral(r, f, t, S):
    """Three-layer cathedral: interference + rings + noise, kaleidoscope on beat,
    feedback zoom tunnel."""
    # Layer 1: interference pattern on sm grid
    canvas_a = _render_vf(r, "sm",
        lambda g, f, t, S: vf_interference(g, f, t, S, n_waves=7) * 1.3,
        hf_angle(0.0), PAL_MATH, f, t, S, sat=0.8)

    # Layer 2: pulsing rings on md grid
    canvas_b = _render_vf(r, "md",
        lambda g, f, t, S: vf_rings(g, f, t, S, n_base=10, spacing_base=3) * 1.4,
        hf_distance(0.3, 0.02), PAL_STARS, f, t, S, sat=0.9)

    # Layer 3: temporal noise on lg grid (slow morph)
    canvas_c = _render_vf(r, "lg",
        lambda g, f, t, S: vf_temporal_noise(g, f, t, S,
            freq=0.04, t_freq=0.2, octaves=3),
        hf_time_cycle(0.12), PAL_BLOCKS, f, t, S, sat=0.7)

    # Blend: A screen B, then difference with C
    result = blend_canvas(canvas_a, canvas_b, "screen", 0.8)
    result = blend_canvas(result, canvas_c, "difference", 0.5)

    # Beat-triggered kaleidoscope
    if f.get("bdecay", 0) > 0.3:
        folds = 6 if f.get("sub_r", 0.3) > 0.4 else 8
        result = sh_kaleidoscope(result.copy(), folds=folds)

    return result

# Scene table entry with feedback:
# {"start": 30.0, "end": 50.0, "name": "cathedral", "fx": fx_cathedral,
#  "gamma": 0.65, "shaders": [("bloom", {"thr": 110}), ("chromatic", {"amt": 4}),
#                              ("vignette", {"s": 0.2}), ("grain", {"amt": 8})],
#  "feedback": {"decay": 0.75, "blend": "screen", "opacity": 0.35,
#               "transform": "zoom", "transform_amt": 0.012, "hue_shift": 0.015}}
```

### Masked Reaction-Diffusion with Attractor Overlay

Reaction-diffusion visible only through an animated iris mask, with a strange attractor density field underneath.

```python
def fx_masked_life(r, f, t, S):
    """Attractor base + reaction-diffusion visible through iris mask + particles."""
    g_sm = r.get_grid("sm")
    g_md = r.get_grid("md")

    # Layer 1: strange attractor density field (background)
    canvas_bg = _render_vf(r, "sm",
        lambda g, f, t, S: vf_strange_attractor(g, f, t, S,
            attractor="clifford", n_points=30000),
        hf_time_cycle(0.04), PAL_DOTS, f, t, S, sat=0.5)

    # Layer 2: reaction-diffusion (foreground, will be masked)
    canvas_rd = _render_vf(r, "md",
        lambda g, f, t, S: vf_reaction_diffusion(g, f, t, S,
            feed=0.046, kill=0.063, steps_per_frame=4, init_mode="ring"),
        hf_angle(0.15), PAL_HALFFILL, f, t, S, sat=0.85)

    # Animated iris mask — opens over first 5 seconds of scene
    scene_start = S.get("_scene_start", t)
    if "_scene_start" not in S:
        S["_scene_start"] = t
    mask = mask_iris(g_md, t, scene_start, scene_start + 5.0,
                     max_radius=0.6)
    canvas_rd = apply_mask_canvas(canvas_rd, mask, bg_canvas=canvas_bg)

    # Layer 3: flow-field particles following the R-D gradient
    rd_field = vf_reaction_diffusion(g_sm, f, t, S,
        feed=0.046, kill=0.063, steps_per_frame=0)  # read without stepping
    ch_p, co_p = update_flow_particles(S, g_sm, f, rd_field,
        n=300, speed=0.8, char_set=list("·•◦∘°"))
    canvas_p = g_sm.render(ch_p, co_p)

    result = blend_canvas(canvas_rd, canvas_p, "add", 0.7)
    return result
```

### Morphing Field Sequence with Eased Keyframes

Demonstrates temporal coherence: smooth morphing between effects with keyframed parameters.

```python
def fx_morphing_journey(r, f, t, S):
    """Morphs through 4 value fields over 20 seconds with eased transitions.
    Parameters (twist, arm count) also keyframed."""
    # Keyframed twist parameter
    twist = keyframe(t, [(0, 1.0), (5, 5.0), (10, 2.0), (15, 8.0), (20, 1.0)],
                     ease_fn=ease_in_out_cubic, loop=True)

    # Sequence of value fields with 2s crossfade
    fields = [
        lambda g, f, t, S: vf_plasma(g, f, t, S),
        lambda g, f, t, S: vf_vortex(g, f, t, S, twist=twist),
        lambda g, f, t, S: vf_fbm(g, f, t, S, octaves=5, freq=0.04),
        lambda g, f, t, S: vf_domain_warp(g, f, t, S, warp_strength=15),
    ]
    durations = [5.0, 5.0, 5.0, 5.0]

    val_fn = lambda g, f, t, S: vf_sequence(g, f, t, S, fields, durations,
                                             crossfade=2.0)

    # Render with slowly rotating hue
    canvas = _render_vf(r, "md", val_fn, hf_time_cycle(0.06),
                        PAL_DENSE, f, t, S, sat=0.8)

    # Second layer: tiled version of same sequence at smaller grid
    tiled_fn = lambda g, f, t, S: vf_sequence(
        make_tgrid(g, *uv_tile(g, 3, 3, mirror=True)),
        f, t, S, fields, durations, crossfade=2.0)
    canvas_b = _render_vf(r, "sm", tiled_fn, hf_angle(0.1),
                          PAL_RUNE, f, t, S, sat=0.6)

    return blend_canvas(canvas, canvas_b, "screen", 0.5)
```

---

## Specialized — Unique State Patterns

### Game of Life with Ghost Trails

Cellular automaton with analog fade trails. Beat injects random cells.

```python
def fx_life(r, f, t, S):
    """Conway's Game of Life with fading ghost trails.
    Beat events inject random live cells for disruption."""
    canvas = _render_vf(r, "sm",
        lambda g, f, t, S: vf_game_of_life(g, f, t, S,
            rule="life", steps_per_frame=1, fade=0.92, density=0.25),
        hf_fixed(0.33), PAL_BLOCKS, f, t, S, sat=0.8)

    # Overlay: coral automaton on lg grid for chunky texture
    canvas_b = _render_vf(r, "lg",
        lambda g, f, t, S: vf_game_of_life(g, f, t, S,
            rule="coral", steps_per_frame=1, fade=0.85, density=0.15, seed=99),
        hf_time_cycle(0.1), PAL_HATCH, f, t, S, sat=0.6)

    return blend_canvas(canvas, canvas_b, "screen", 0.5)
```

### Boids Flock Over Voronoi

Emergent swarm movement over a cellular background.

```python
def fx_boid_swarm(r, f, t, S):
    """Flocking boids over animated voronoi cells."""
    # Background: voronoi cells
    canvas_bg = _render_vf(r, "md",
        lambda g, f, t, S: vf_voronoi(g, f, t, S,
            n_cells=20, mode="distance", speed=0.2),
        hf_distance(0.4, 0.02), PAL_CIRCUIT, f, t, S, sat=0.5)

    # Foreground: boids
    g = r.get_grid("md")
    ch_b, co_b = update_boids(S, g, f, n_boids=150, perception=6.0,
                              max_speed=1.5, char_set=list("▸▹►▻→⟶"))
    canvas_boids = g.render(ch_b, co_b)

    # Trails for the boids
    # (boid positions are stored in S["boid_x"], S["boid_y"])
    S["px"] = list(S.get("boid_x", []))
    S["py"] = list(S.get("boid_y", []))
    ch_t, co_t = draw_particle_trails(S, g, max_trail=6, fade=0.6)
    canvas_trails = g.render(ch_t, co_t)

    result = blend_canvas(canvas_bg, canvas_trails, "add", 0.3)
    result = blend_canvas(result, canvas_boids, "add", 0.9)
    return result
```

### Fire Rising Through SDF Text Stencil

Fire effect visible only through text letterforms.

```python
def fx_fire_text(r, f, t, S):
    """Fire columns visible through text stencil. Text acts as window."""
    g = r.get_grid("lg")

    # Full-screen fire (will be masked)
    canvas_fire = _render_vf(r, "sm",
        lambda g, f, t, S: np.clip(
            vf_fbm(g, f, t, S, octaves=4, freq=0.08, speed=0.8) *
            (1.0 - g.rr / g.rows) *  # fade toward top
            (0.6 + f.get("bass", 0.3) * 0.8), 0, 1),
        hf_fixed(0.05), PAL_BLOCKS, f, t, S, sat=0.9)  # fire hue

    # Background: dark domain warp
    canvas_bg = _render_vf(r, "md",
        lambda g, f, t, S: vf_domain_warp(g, f, t, S,
            warp_strength=8, freq=0.03, speed=0.05) * 0.3,
        hf_fixed(0.6), PAL_DENSE, f, t, S, sat=0.4)

    # Text stencil mask
    mask = mask_text(g, "FIRE", row_frac=0.45)
    # Expand vertically for multi-row coverage
    for offset in range(-2, 3):
        shifted = mask_text(g, "FIRE", row_frac=0.45 + offset / g.rows)
        mask = mask_union(mask, shifted)

    canvas_masked = apply_mask_canvas(canvas_fire, mask, bg_canvas=canvas_bg)
    return canvas_masked
```

### Portrait Mode: Vertical Rain + Quote

Optimized for 9:16. Uses vertical space for long rain trails and stacked text.

```python
def fx_portrait_rain_quote(r, f, t, S):
    """Portrait-optimized: matrix rain (long vertical trails) with stacked quote.
    Designed for 1080x1920 (9:16)."""
    g = r.get_grid("md")  # ~112x100 in portrait

    # Matrix rain — long trails benefit from portrait's extra rows
    ch, co, S = eff_matrix_rain(g, f, t, S,
        hue=0.33, bri=0.6, pal=PAL_KATA, speed_base=0.4, speed_beat=2.5)
    canvas_rain = g.render(ch, co)

    # Tunnel depth underneath for texture
    canvas_tunnel = _render_vf(r, "sm",
        lambda g, f, t, S: vf_tunnel(g, f, t, S, speed=3.0, complexity=6) * 0.8,
        hf_fixed(0.33), PAL_BLOCKS, f, t, S, sat=0.5)

    result = blend_canvas(canvas_tunnel, canvas_rain, "screen", 0.8)

    # Quote text — portrait layout: short lines, many of them
    g_text = r.get_grid("lg")  # ~90x80 in portrait
    quote_lines = layout_text_portrait(
        "The code is the art and the art is the code",
        max_chars_per_line=20)
    # Center vertically
    block_start = (g_text.rows - len(quote_lines)) // 2
    ch_t = np.full((g_text.rows, g_text.cols), " ", dtype="U1")
    co_t = np.zeros((g_text.rows, g_text.cols, 3), dtype=np.uint8)
    total_chars = sum(len(l) for l in quote_lines)
    progress = min(1.0, (t - S.get("_scene_start", t)) / 3.0)
    if "_scene_start" not in S: S["_scene_start"] = t
    render_typewriter(ch_t, co_t, quote_lines, block_start, g_text.cols,
                      progress, total_chars, (200, 255, 220), t)
    canvas_text = g_text.render(ch_t, co_t)

    result = blend_canvas(result, canvas_text, "add", 0.9)
    return result
```

---

## Scene Table Template

Wire scenes into a complete video:

```python
SCENES = [
    {"start": 0.0,  "end": 5.0,  "name": "coral",
     "fx": fx_coral, "grid": "sm", "gamma": 0.70,
     "shaders": [("bloom", {"thr": 110}), ("vignette", {"s": 0.2})],
     "feedback": {"decay": 0.8, "blend": "screen", "opacity": 0.3,
                  "transform": "zoom", "transform_amt": 0.01}},

    {"start": 5.0,  "end": 15.0, "name": "tunnel_noise",
     "fx": fx_tunnel_noise, "grid": "md", "gamma": 0.75,
     "shaders": [("chromatic", {"amt": 3}), ("bloom", {"thr": 120}),
                 ("scanlines", {"intensity": 0.06}), ("grain", {"amt": 8})],
     "feedback": None},

    {"start": 15.0, "end": 35.0, "name": "cathedral",
     "fx": fx_cathedral, "grid": "sm", "gamma": 0.65,
     "shaders": [("bloom", {"thr": 100}), ("chromatic", {"amt": 5}),
                 ("color_wobble", {"amt": 0.2}), ("vignette", {"s": 0.18})],
     "feedback": {"decay": 0.75, "blend": "screen", "opacity": 0.35,
                  "transform": "zoom", "transform_amt": 0.012, "hue_shift": 0.015}},

    {"start": 35.0, "end": 50.0, "name": "morphing",
     "fx": fx_morphing_journey, "grid": "md", "gamma": 0.70,
     "shaders": [("bloom", {"thr": 110}), ("grain", {"amt": 6})],
     "feedback": {"decay": 0.7, "blend": "screen", "opacity": 0.25,
                  "transform": "rotate_cw", "transform_amt": 0.003}},
]
```
