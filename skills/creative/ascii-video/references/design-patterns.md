# Scene Design Patterns

**Cross-references:**
- Scene protocol, SCENES table: `scenes.md`
- Blend modes, multi-grid composition, tonemap: `composition.md`
- Effect building blocks (value fields, noise, SDFs): `effects.md`
- Shader pipeline, feedback buffer: `shaders.md`
- Complete scene examples: `examples.md`

Higher-order patterns for composing scenes that feel intentional rather than random. These patterns use the existing building blocks (value fields, blend modes, shaders, feedback) but organize them with compositional intent.

## Layer Hierarchy

Every scene should have clear visual layers with distinct roles:

| Layer | Grid | Brightness | Purpose |
|-------|------|-----------|---------|
| **Background** | xs or sm (dense) | 0.1–0.25 | Atmosphere, texture. Never competes with content. |
| **Content** | md (balanced) | 0.4–0.8 | The main visual idea. Carries the scene's concept. |
| **Accent** | lg or sm (sparse) | 0.5–1.0 (sparse coverage) | Highlights, punctuation, sparse bright points. |

The background sets mood. The content layer is what the scene *is about*. The accent adds visual interest without overwhelming.

```python
def fx_example(r, f, t, S):
    local = t
    progress = min(local / 5.0, 1.0)

    g_bg = r.get_grid("sm")
    g_main = r.get_grid("md")
    g_accent = r.get_grid("lg")

    # --- Background: dim atmosphere ---
    bg_val = vf_smooth_noise(g_bg, f, t * 0.3, S, octaves=2, bri=0.15)
    # ... render bg to canvas

    # --- Content: the main visual idea ---
    content_val = vf_spiral(g_main, f, t, S, n_arms=n_arms, tightness=tightness)
    # ... render content on top of canvas

    # --- Accent: sparse highlights ---
    accent_val = vf_noise_static(g_accent, f, t, S, density=0.05)
    # ... render accent on top

    return canvas
```

## Directional Parameter Arcs

Parameters should *go somewhere* over the scene's duration — not oscillate aimlessly with `sin(t * N)`.

**Bad:** `twist = 3.0 + 2.0 * math.sin(t * 0.6)` — wobbles back and forth, feels aimless.

**Good:** `twist = 2.0 + progress * 5.0` — starts gentle, ends intense. The scene *builds*.

Use `progress = min(local / duration, 1.0)` (0→1 over the scene) to drive directional change:

| Pattern | Formula | Feel |
|---------|---------|------|
| Linear ramp | `progress * range` | Steady buildup |
| Ease-out | `1 - (1 - progress) ** 2` | Fast start, gentle finish |
| Ease-in | `progress ** 2` | Slow start, accelerating |
| Step reveal | `np.clip((progress - 0.5) / 0.25, 0, 1)` | Nothing until 50%, then fades in |
| Build + plateau | `min(1.0, progress * 1.5)` | Reaches full at 67%, holds |

Oscillation is fine for *secondary* parameters (saturation shimmer, hue drift). But the *defining* parameter of the scene should have a direction.

### Examples of Directional Arcs

| Scene concept | Parameter | Arc |
|--------------|-----------|-----|
| Emergence | Ring radius | 0 → max (ease-out) |
| Shatter | Voronoi cell count | 8 → 38 (linear) |
| Descent | Tunnel speed | 2.0 → 10.0 (linear) |
| Mandala | Shape complexity | ring → +polygon → +star → +rosette (step reveals) |
| Crescendo | Layer count | 1 → 7 (staggered entry) |
| Entropy | Geometry visibility | 1.0 → 0.0 (consumed) |

## Scene Concepts

Each scene should be built around a *visual idea*, not an effect name.

**Bad:** "fx_plasma_cascade" — named after the effect. No concept.
**Good:** "fx_emergence" — a point of light expands into a field. The name tells you *what happens*.

Good scene concepts have:
1. A **visual metaphor** (emergence, descent, collision, entropy)
2. A **directional arc** (things change from A to B, not oscillate)
3. **Motivated layer choices** (each layer serves the concept)
4. **Motivated feedback** (transform direction matches the metaphor)

| Concept | Metaphor | Feedback transform | Why |
|---------|----------|-------------------|-----|
| Emergence | Birth, expansion | zoom-out | Past frames expand outward |
| Descent | Falling, acceleration | zoom-in | Past frames rush toward center |
| Inferno | Rising fire | shift-up | Past frames rise with the flames |
| Entropy | Decay, dissolution | none | Clean, no persistence — things disappear |
| Crescendo | Accumulation | zoom + hue_shift | Everything compounds and shifts |

## Compositional Techniques

### Counter-Rotating Dual Systems

Two instances of the same effect rotating in opposite directions create visual interference:

```python
# Primary spiral (clockwise)
s1_val = vf_spiral(g_main, f, t * 1.5, S, n_arms=n_arms_1, tightness=tightness_1)

# Counter-rotating spiral (counter-clockwise via negative time)
s2_val = vf_spiral(g_accent, f, -t * 1.2, S, n_arms=n_arms_2, tightness=tightness_2)

# Screen blend creates bright interference at crossing points
canvas = blend_canvas(canvas_with_s1, c2, "screen", 0.7)
```

Works with spirals, vortexes, rings. The counter-rotation creates constantly shifting interference patterns.

### Wave Collision

Two wave fronts converging from opposite sides, meeting at a collision point:

```python
collision_phase = abs(progress - 0.5) * 2  # 1→0→1 (0 at collision)

# Wave A approaches from left
offset_a = (1 - progress) * g.cols * 0.4
wave_a = np.sin((g.cc + offset_a) * 0.08 + t * 2) * 0.5 + 0.5

# Wave B approaches from right
offset_b = -(1 - progress) * g.cols * 0.4
wave_b = np.sin((g.cc + offset_b) * 0.08 - t * 2) * 0.5 + 0.5

# Interference peaks at collision
combined = wave_a * 0.5 + wave_b * 0.5 + np.abs(wave_a - wave_b) * (1 - collision_phase) * 0.5
```

### Progressive Fragmentation

Voronoi with cell count increasing over time — visual shattering:

```python
n_pts = int(8 + progress * 30)  # 8 cells → 38 cells
# Pre-generate enough points, slice to n_pts
px = base_x[:n_pts] + np.sin(t * 0.3 + np.arange(n_pts) * 0.7) * (3 + progress * 3)
```

The edge glow width can also increase with progress to emphasize the cracks.

### Entropy / Consumption

A clean geometric pattern being overtaken by an organic process:

```python
# Geometry fades out
geo_val = clean_pattern * max(0.05, 1.0 - progress * 0.9)

# Organic process grows in
rd_val = vf_reaction_diffusion(g, f, t, S) * min(1.0, progress * 1.5)

# Render geometry first, organic on top — organic consumes geometry
```

### Staggered Layer Entry (Crescendo)

Layers enter one at a time, building to overwhelming density:

```python
def layer_strength(enter_t, ramp=1.5):
    """0.0 until enter_t, ramps to 1.0 over ramp seconds."""
    return max(0.0, min(1.0, (local - enter_t) / ramp))

# Layer 1: always present
s1 = layer_strength(0.0)
# Layer 2: enters at 2s
s2 = layer_strength(2.0)
# Layer 3: enters at 4s
s3 = layer_strength(4.0)
# ... etc

# Each layer uses a different effect, grid, palette, and blend mode
# Screen blend between layers so they accumulate light
```

For a 15-second crescendo, 7 layers entering every 2 seconds works well. Use different blend modes (screen for most, add for energy, colordodge for the final wash).

## Scene Ordering

For a multi-scene reel or video:
- **Vary mood between adjacent scenes** — don't put two calm scenes next to each other
- **Randomize order** rather than grouping by type — prevents "effect demo" feel
- **End on the strongest scene** — crescendo or something with a clear payoff
- **Open with energy** — grab attention in the first 2 seconds
