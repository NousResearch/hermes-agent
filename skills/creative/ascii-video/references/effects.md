# Effect Catalog

Effect building blocks that produce visual patterns. In v2, these are used **inside scene functions** that return a pixel canvas directly. The building blocks below operate on grid coordinate arrays and produce `(chars, colors)` or value/hue fields that the scene function renders to canvas via `_render_vf()`. See `composition.md` for the v2 rendering pattern and `scenes.md` for scene function examples.

## Design Philosophy

Effects are the creative core. Don't copy these verbatim for every project -- use them as **building blocks** and **combine, modify, and invent** new ones. Every project should feel distinct.

Key principles:
- **Layer multiple effects** rather than using a single monolithic function
- **Parameterize everything** -- hue, speed, density, amplitude should all be arguments
- **React to features** -- audio/video features should modulate at least 2-3 parameters per effect
- **Vary per section** -- never use the same effect config for the entire video
- **Invent project-specific effects** -- the catalog below is a starting vocabulary, not a fixed set

---

## Background Fills

Every effect should start with a background. Never leave flat black.

### Animated Sine Field (General Purpose)
```python
def bg_sinefield(g, f, t, hue=0.6, bri=0.5, pal=PAL_DEFAULT,
                 freq=(0.13, 0.17, 0.07, 0.09), speed=(0.5, -0.4, -0.3, 0.2)):
    """Layered sine field. Adjust freq/speed tuples for different textures."""
    v1 = np.sin(g.cc*freq[0] + t*speed[0]) * np.sin(g.rr*freq[1] - t*speed[1]) * 0.5 + 0.5
    v2 = np.sin(g.cc*freq[2] - t*speed[2] + g.rr*freq[3]) * 0.4 + 0.5
    v3 = np.sin(g.dist_n*5 + t*0.2) * 0.3 + 0.4
    v4 = np.cos(g.angle*3 - t*0.6) * 0.15 + 0.5
    val = np.clip((v1*0.3 + v2*0.25 + v3*0.25 + v4*0.2) * bri * (0.6 + f["rms"]*0.6), 0.06, 1)
    mask = val > 0.03
    ch = val2char(val, mask, pal)
    h = np.full_like(val, hue) + f.get("cent", 0.5)*0.1 + val*0.08
    R, G, B = hsv2rgb(h, np.clip(0.35+f.get("flat",0.4)*0.4, 0, 1) * np.ones_like(val), val)
    return ch, mkc(R, G, B, g.rows, g.cols)
```

### Video-Source Background
```python
def bg_video(g, frame_rgb, pal=PAL_DEFAULT, brightness=0.5):
    small = np.array(Image.fromarray(frame_rgb).resize((g.cols, g.rows)))
    lum = np.mean(small, axis=2) / 255.0 * brightness
    mask = lum > 0.02
    ch = val2char(lum, mask, pal)
    co = np.clip(small * np.clip(lum[:,:,None]*1.5+0.3, 0.3, 1), 0, 255).astype(np.uint8)
    return ch, co
```

### Noise / Static Field
```python
def bg_noise(g, f, t, pal=PAL_BLOCKS, density=0.3, hue_drift=0.02):
    val = np.random.random((g.rows, g.cols)).astype(np.float32) * density * (0.5 + f["rms"]*0.5)
    val = np.clip(val, 0, 1); mask = val > 0.02
    ch = val2char(val, mask, pal)
    R, G, B = hsv2rgb(np.full_like(val, t*hue_drift % 1), np.full_like(val, 0.3), val)
    return ch, mkc(R, G, B, g.rows, g.cols)
```

### Perlin-Like Smooth Noise
```python
def bg_smooth_noise(g, f, t, hue=0.5, bri=0.5, pal=PAL_DOTS, octaves=3):
    """Layered sine approximation of Perlin noise. Cheap, smooth, organic."""
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for i in range(octaves):
        freq = 0.05 * (2 ** i)
        amp = 0.5 / (i + 1)
        phase = t * (0.3 + i * 0.2)
        val += np.sin(g.cc * freq + phase) * np.cos(g.rr * freq * 0.7 - phase * 0.5) * amp
    val = np.clip(val * 0.5 + 0.5, 0, 1) * bri
    mask = val > 0.03
    ch = val2char(val, mask, pal)
    h = np.full_like(val, hue) + val * 0.1
    R, G, B = hsv2rgb(h, np.full_like(val, 0.5), val)
    return ch, mkc(R, G, B, g.rows, g.cols)
```

### Cellular / Voronoi Approximation
```python
def bg_cellular(g, f, t, n_centers=12, hue=0.5, bri=0.6, pal=PAL_BLOCKS):
    """Voronoi-like cells using distance to nearest of N moving centers."""
    rng = np.random.RandomState(42)  # deterministic centers
    cx = (rng.rand(n_centers) * g.cols).astype(np.float32)
    cy = (rng.rand(n_centers) * g.rows).astype(np.float32)
    # Animate centers
    cx_t = cx + np.sin(t * 0.5 + np.arange(n_centers) * 0.7) * 5
    cy_t = cy + np.cos(t * 0.4 + np.arange(n_centers) * 0.9) * 3
    # Min distance to any center
    min_d = np.full((g.rows, g.cols), 999.0, dtype=np.float32)
    for i in range(n_centers):
        d = np.sqrt((g.cc - cx_t[i])**2 + (g.rr - cy_t[i])**2)
        min_d = np.minimum(min_d, d)
    val = np.clip(1.0 - min_d / (g.cols * 0.3), 0, 1) * bri
    # Cell edges (where distance is near-equal between two centers)
    # ... second-nearest trick for edge highlighting
    mask = val > 0.03
    ch = val2char(val, mask, pal)
    R, G, B = hsv2rgb(np.full_like(val, hue) + min_d * 0.005, np.full_like(val, 0.5), val)
    return ch, mkc(R, G, B, g.rows, g.cols)
```

---

## Radial Effects

### Concentric Rings
Bass/sub-driven pulsing rings from center. Scale ring count and thickness with bass energy.
```python
def eff_rings(g, f, t, hue=0.5, n_base=6, pal=PAL_DEFAULT):
    n_rings = int(n_base + f["sub_r"] * 25 + f["bass"] * 10)
    spacing = 2 + f["bass_r"] * 7 + f["rms"] * 3
    ring_cv = np.zeros((g.rows, g.cols), dtype=np.float32)
    for ri in range(n_rings):
        rad = (ri+1) * spacing + f["bdecay"] * 15
        wobble = f["mid_r"]*5*np.sin(g.angle*3 + t*4) + f["hi_r"]*3*np.sin(g.angle*7 - t*6)
        rd = np.abs(g.dist - rad - wobble)
        th = 1 + f["sub"] * 3
        ring_cv = np.maximum(ring_cv, np.clip((1 - rd/th) * (0.4 + f["bass"]*0.8), 0, 1))
    # Color by angle + distance for rainbow rings
    h = g.angle/(2*np.pi) + g.dist*0.005 + f["sub_r"]*0.2
    return ring_cv, h
```

### Radial Rays
```python
def eff_rays(g, f, t, n_base=8, hue=0.5):
    n_rays = int(n_base + f["hi_r"] * 25)
    ray = np.clip(np.cos(g.angle*n_rays + t*3) * f["bdecay"]*0.6 * (1-g.dist_n), 0, 0.7)
    return ray
```

### Spiral Arms (Logarithmic)
```python
def eff_spiral(g, f, t, n_arms=3, tightness=2.5, hue=0.5):
    arm_cv = np.zeros((g.rows, g.cols), dtype=np.float32)
    for ai in range(n_arms):
        offset = ai * 2*np.pi / n_arms
        log_r = np.log(g.dist + 1) * tightness
        arm_phase = g.angle + offset - log_r + t * 0.8
        arm_val = np.clip(np.cos(arm_phase * n_arms) * 0.6 + 0.2, 0, 1)
        arm_val *= (0.4 + f["rms"]*0.6) * np.clip(1 - g.dist_n*0.5, 0.2, 1)
        arm_cv = np.maximum(arm_cv, arm_val)
    return arm_cv
```

### Center Glow / Pulse
```python
def eff_glow(g, f, t, intensity=0.6, spread=2.0):
    return np.clip(intensity * np.exp(-g.dist_n * spread) * (0.5 + f["rms"]*2 + np.sin(t*1.2)*0.2), 0, 0.9)
```

### Tunnel / Depth
```python
def eff_tunnel(g, f, t, speed=3.0, complexity=6):
    tunnel_d = 1.0 / (g.dist_n + 0.1)
    v1 = np.sin(tunnel_d*2 - t*speed) * 0.45 + 0.55
    v2 = np.sin(g.angle*complexity + tunnel_d*1.5 - t*2) * 0.35 + 0.55
    return v1 * 0.5 + v2 * 0.5
```

### Vortex (Rotating Distortion)
```python
def eff_vortex(g, f, t, twist=3.0, pulse=True):
    """Twisting radial pattern -- distance modulates angle."""
    twisted = g.angle + g.dist_n * twist * np.sin(t * 0.5)
    val = np.sin(twisted * 4 - t * 2) * 0.5 + 0.5
    if pulse:
        val *= 0.5 + f.get("bass", 0.3) * 0.8
    return np.clip(val, 0, 1)
```

---

## Wave Effects

### Multi-Band Frequency Waves
Each frequency band draws its own wave at different spatial/temporal frequencies:
```python
def eff_freq_waves(g, f, t, bands=None):
    if bands is None:
        bands = [("sub",0.06,1.2,0.0), ("bass",0.10,2.0,0.08), ("lomid",0.15,3.0,0.16),
                 ("mid",0.22,4.5,0.25), ("himid",0.32,6.5,0.4), ("hi",0.45,8.5,0.55)]
    mid = g.rows / 2.0
    composite = np.zeros((g.rows, g.cols), dtype=np.float32)
    for band_key, sf, tf, hue_base in bands:
        amp = f.get(band_key, 0.3) * g.rows * 0.4
        y_wave = mid - np.sin(g.cc*sf + t*tf) * amp
        y_wave += np.sin(g.cc*sf*2.3 + t*tf*1.7) * amp * 0.2  # harmonic
        dist = np.abs(g.rr - y_wave)
        thickness = 2 + f.get(band_key, 0.3) * 5
        intensity = np.clip((1 - dist/thickness) * f.get(band_key, 0.3) * 1.5, 0, 1)
        composite = np.maximum(composite, intensity)
    return composite
```

### Interference Pattern
6-8 overlapping sine waves creating moire-like patterns:
```python
def eff_interference(g, f, t, n_waves=5):
    """Parametric interference -- vary n_waves for complexity."""
    # Each wave has different orientation, frequency, and feature driver
    drivers = ["mid_r", "himid_r", "bass_r", "lomid_r", "hi_r"]
    vals = np.zeros((g.rows, g.cols), dtype=np.float32)
    for i in range(min(n_waves, len(drivers))):
        angle = i * np.pi / n_waves  # spread orientations
        freq = 0.06 + i * 0.03
        sp = 0.5 + i * 0.3
        proj = g.cc * np.cos(angle) + g.rr * np.sin(angle)
        vals += np.sin(proj * freq + t * sp) * f.get(drivers[i], 0.3) * 2.5
    return np.clip(vals * 0.12 + 0.45, 0.1, 1)
```

### Aurora / Horizontal Bands
```python
def eff_aurora(g, f, t, hue=0.4, n_bands=3):
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for i in range(n_bands):
        freq_r = 0.08 + i * 0.04
        freq_c = 0.012 + i * 0.008
        sp_r = 0.7 + i * 0.3
        sp_c = 0.18 + i * 0.12
        val += np.sin(g.rr*freq_r + t*sp_r) * np.sin(g.cc*freq_c + t*sp_c) * (0.6 / n_bands)
    return np.clip(val * (f.get("lomid_r", 0.3)*3 + 0.2), 0, 0.7)
```

### Ripple (Point-Source Waves)
```python
def eff_ripple(g, f, t, sources=None, freq=0.3, damping=0.02):
    """Concentric ripples from point sources. Sources = [(row_frac, col_frac), ...]"""
    if sources is None:
        sources = [(0.5, 0.5)]  # center
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for ry, rx in sources:
        dy = g.rr - g.rows * ry
        dx = g.cc - g.cols * rx
        d = np.sqrt(dy**2 + dx**2)
        val += np.sin(d * freq - t * 4) * np.exp(-d * damping) * 0.5
    return np.clip(val + 0.5, 0, 1)
```

---

## Particle Systems

### General Pattern
All particle systems use persistent state:
```python
S = state  # dict persisted across frames
if "px" not in S:
    S["px"]=[]; S["py"]=[]; S["vx"]=[]; S["vy"]=[]; S["life"]=[]; S["char"]=[]

# Emit new particles (on beat, continuously, or on trigger)
# Update: position += velocity, apply forces, decay life
# Draw: map to grid, set char/color based on life
# Cull: remove dead, cap total count
```

### Particle Character Sets

Don't hardcode particle chars. Choose per project/mood:

```python
# Energy / explosive
PART_ENERGY  = list("*+#@\u26a1\u2726\u2605\u2588\u2593")
PART_SPARK   = list("\u00b7\u2022\u25cf\u2605\u2736*+")
# Organic / natural
PART_LEAF    = list("\u2740\u2741\u2742\u2743\u273f\u2618\u2022")
PART_SNOW    = list("\u2744\u2745\u2746\u00b7\u2022*\u25cb")
PART_RAIN    = list("|\u2502\u2503\u2551/\\")
PART_BUBBLE  = list("\u25cb\u25ce\u25c9\u25cf\u2218\u2219\u00b0")
# Data / tech
PART_DATA    = list("01{}[]<>|/\\")
PART_HEX     = list("0123456789ABCDEF")
PART_BINARY  = list("01")
# Mystical
PART_RUNE    = list("\u16a0\u16a2\u16a6\u16b1\u16b7\u16c1\u16c7\u16d2\u16d6\u16da\u16de\u16df\u2726\u2605")
PART_ZODIAC  = list("\u2648\u2649\u264a\u264b\u264c\u264d\u264e\u264f\u2650\u2651\u2652\u2653")
# Minimal
PART_DOT     = list("\u00b7\u2022\u25cf")
PART_DASH    = list("-=~\u2500\u2550")
```

### Explosion (Beat-Triggered)
```python
def emit_explosion(S, f, center_r, center_c, char_set=PART_ENERGY, count_base=80):
    if f.get("beat", 0) > 0:
        for _ in range(int(count_base + f["rms"]*150)):
            ang = random.uniform(0, 2*math.pi)
            sp = random.uniform(1, 9) * (0.5 + f.get("sub_r", 0.3)*2)
            S["px"].append(float(center_c))
            S["py"].append(float(center_r))
            S["vx"].append(math.cos(ang)*sp*2.5)
            S["vy"].append(math.sin(ang)*sp)
            S["life"].append(1.0)
            S["char"].append(random.choice(char_set))
# Update: gravity on vy += 0.03, life -= 0.015
# Color: life * 255 for brightness, hue fade controlled by caller
```

### Rising Embers
```python
# Emit: sy = rows-1, vy = -random.uniform(1,5), vx = random.uniform(-1.5,1.5)
# Update: vx += random jitter * 0.3, life -= 0.01
# Cap at ~1500 particles
```

### Dissolving Cloud
```python
# Init: N=600 particles spread across screen
# Update: slow upward drift, fade life progressively
# life -= 0.002 * (1 + elapsed * 0.05)  # accelerating fade
```

### Starfield (3D Projection)
```python
# N stars with (sx, sy, sz) in normalized coords
# Move: sz -= speed (stars approach camera)
# Project: px = cx + sx/sz * cx, py = cy + sy/sz * cy
# Reset stars that pass camera (sz <= 0.01)
# Brightness = (1 - sz), draw streaks behind bright stars
```

### Orbit (Circular/Elliptical Motion)
```python
def emit_orbit(S, n=20, radius=15, speed=1.0, char_set=PART_DOT):
    """Particles orbiting a center point."""
    for i in range(n):
        angle = i * 2 * math.pi / n
        S["px"].append(0.0); S["py"].append(0.0)  # will be computed from angle
        S["vx"].append(angle)  # store angle as "vx" for orbit
        S["vy"].append(radius + random.uniform(-2, 2))  # store radius
        S["life"].append(1.0)
        S["char"].append(random.choice(char_set))
# Update: angle += speed * dt, px = cx + radius * cos(angle), py = cy + radius * sin(angle)
```

### Gravity Well
```python
# Particles attracted toward one or more gravity points
# Update: compute force vector toward each well, apply as acceleration
# Particles that reach well center respawn at edges
```

---

## Rain / Matrix Effects

### Column Rain (Vectorized)
```python
def eff_matrix_rain(g, f, t, state, hue=0.33, bri=0.6, pal=PAL_KATA,
                    speed_base=0.5, speed_beat=3.0):
    """Vectorized matrix rain. state dict persists column positions."""
    if "ry" not in state or len(state["ry"]) != g.cols:
        state["ry"] = np.random.uniform(-g.rows, g.rows, g.cols).astype(np.float32)
        state["rsp"] = np.random.uniform(0.3, 2.0, g.cols).astype(np.float32)
        state["rln"] = np.random.randint(8, 40, g.cols)
        state["rch"] = np.random.randint(0, len(pal), (g.rows, g.cols))  # pre-assign chars

    speed_mult = speed_base + f.get("bass", 0.3)*speed_beat + f.get("sub_r", 0.3)*3
    if f.get("beat", 0) > 0: speed_mult *= 2.5
    state["ry"] += state["rsp"] * speed_mult

    # Reset columns that fall past bottom
    rst = (state["ry"] - state["rln"]) > g.rows
    state["ry"][rst] = np.random.uniform(-25, -2, rst.sum())

    # Vectorized draw using fancy indexing
    ch = np.full((g.rows, g.cols), " ", dtype="U1")
    co = np.zeros((g.rows, g.cols, 3), dtype=np.uint8)
    heads = state["ry"].astype(int)
    for c in range(g.cols):
        head = heads[c]
        trail_len = state["rln"][c]
        for i in range(trail_len):
            row = head - i
            if 0 <= row < g.rows:
                fade = 1.0 - i / trail_len
                ci = state["rch"][row, c] % len(pal)
                ch[row, c] = pal[ci]
                v = fade * bri * 255
                if i == 0:  # head is bright white-ish
                    co[row, c] = (int(v*0.9), int(min(255, v*1.1)), int(v*0.9))
                else:
                    R, G, B = hsv2rgb_single(hue, 0.7, fade * bri)
                    co[row, c] = (R, G, B)
    return ch, co, state
```

---

## Glitch / Data Effects

### Horizontal Band Displacement
```python
def eff_glitch_displace(ch, co, f, intensity=1.0):
    n_bands = int(8 + f.get("flux", 0.3)*25 + f.get("bdecay", 0)*15) * intensity
    for _ in range(int(n_bands)):
        y = random.randint(0, ch.shape[0]-1)
        h = random.randint(1, int(3 + f.get("sub", 0.3)*8))
        shift = int((random.random()-0.5) * f.get("rms", 0.3)*40 + f.get("bdecay", 0)*20*(random.random()-0.5))
        if shift != 0:
            for row in range(h):
                rr = y + row
                if 0 <= rr < ch.shape[0]:
                    ch[rr] = np.roll(ch[rr], shift)
                    co[rr] = np.roll(co[rr], shift, axis=0)
    return ch, co
```

### Block Corruption
```python
def eff_block_corrupt(ch, co, f, char_pool=None, count_base=20):
    if char_pool is None:
        char_pool = list(PAL_BLOCKS[4:] + PAL_KATA[2:8])
    for _ in range(int(count_base + f.get("flux", 0.3)*60 + f.get("bdecay", 0)*40)):
        bx = random.randint(0, max(1, ch.shape[1]-6))
        by = random.randint(0, max(1, ch.shape[0]-4))
        bw, bh = random.randint(2,6), random.randint(1,4)
        block_char = random.choice(char_pool)
        # Fill rectangle with single char and random color
        for r in range(bh):
            for c in range(bw):
                rr, cc = by+r, bx+c
                if 0 <= rr < ch.shape[0] and 0 <= cc < ch.shape[1]:
                    ch[rr, cc] = block_char
                    co[rr, cc] = (random.randint(100,255), random.randint(0,100), random.randint(0,80))
    return ch, co
```

### Scan Bars (Vertical)
```python
def eff_scanbars(ch, co, f, t, n_base=4, chars="|\u2551|!1l"):
    for bi in range(int(n_base + f.get("himid_r", 0.3)*12)):
        sx = int((t*50*(1+bi*0.3) + bi*37) % ch.shape[1])
        for rr in range(ch.shape[0]):
            if random.random() < 0.7:
                ch[rr, sx] = random.choice(chars)
    return ch, co
```

### Error Messages
```python
# Parameterize the error vocabulary per project:
ERRORS_TECH = ["SEGFAULT","0xDEADBEEF","BUFFER_OVERRUN","PANIC!","NULL_PTR",
               "CORRUPT","SIGSEGV","ERR_OVERFLOW","STACK_SMASH","BAD_ALLOC"]
ERRORS_COSMIC = ["VOID_BREACH","ENTROPY_MAX","SINGULARITY","DIMENSION_FAULT",
                 "REALITY_ERR","TIME_PARADOX","DARK_MATTER_LEAK","QUANTUM_DECOHERE"]
ERRORS_ORGANIC = ["CELL_DIVISION_ERR","DNA_MISMATCH","MUTATION_OVERFLOW",
                  "NEURAL_DEADLOCK","SYNAPSE_TIMEOUT","MEMBRANE_BREACH"]
```

### Hex Data Stream
```python
hex_str = "".join(random.choice("0123456789ABCDEF") for _ in range(random.randint(8,20)))
stamp(ch, co, hex_str, rand_row, rand_col, (0, 160, 80))
```

---

## Spectrum / Visualization

### Mirrored Spectrum Bars
```python
def eff_spectrum(g, f, t, n_bars=64, pal=PAL_BLOCKS, mirror=True):
    bar_w = max(1, g.cols // n_bars); mid = g.rows // 2
    band_vals = np.array([f.get("sub",0.3), f.get("bass",0.3), f.get("lomid",0.3),
                          f.get("mid",0.3), f.get("himid",0.3), f.get("hi",0.3)])
    ch = np.full((g.rows, g.cols), " ", dtype="U1")
    co = np.zeros((g.rows, g.cols, 3), dtype=np.uint8)
    for b in range(n_bars):
        frac = b / n_bars
        fi = frac * 5; lo_i = int(fi); hi_i = min(lo_i+1, 5)
        bval = min(1, (band_vals[lo_i]*(1-fi%1) + band_vals[hi_i]*(fi%1)) * 1.8)
        height = int(bval * (g.rows//2 - 2))
        for dy in range(height):
            hue = (f.get("cent",0.5)*0.3 + frac*0.3 + dy/max(height,1)*0.15) % 1.0
            ci = pal[min(int(dy/max(height,1)*len(pal)*0.7+len(pal)*0.2), len(pal)-1)]
            for dc in range(bar_w - (1 if bar_w > 2 else 0)):
                cc = b*bar_w + dc
                if 0 <= cc < g.cols:
                    rows_to_draw = [mid - dy, mid + dy] if mirror else [g.rows - 1 - dy]
                    for row in rows_to_draw:
                        if 0 <= row < g.rows:
                            ch[row, cc] = ci
                            co[row, cc] = hsv_to_rgb_single(hue, 0.85, 0.5+dy/max(height,1)*0.5)
    return ch, co
```

### Waveform
```python
def eff_waveform(g, f, t, row_offset=-5, hue=0.1):
    ch = np.full((g.rows, g.cols), " ", dtype="U1")
    co = np.zeros((g.rows, g.cols, 3), dtype=np.uint8)
    for c in range(g.cols):
        wv = (math.sin(c*0.15+t*5)*f.get("bass",0.3)*0.5
            + math.sin(c*0.3+t*8)*f.get("mid",0.3)*0.3
            + math.sin(c*0.6+t*12)*f.get("hi",0.3)*0.15)
        wr = g.rows + row_offset + int(wv * 4)
        if 0 <= wr < g.rows:
            ch[wr, c] = "~"
            v = int(120 + f.get("rms",0.3)*135)
            co[wr, c] = [v, int(v*0.7), int(v*0.4)]
    return ch, co
```

---

## Fire / Lava

### Fire Columns
```python
def eff_fire(g, f, t, n_base=20, hue_base=0.02, hue_range=0.12, pal=PAL_BLOCKS):
    n_cols = int(n_base + f.get("bass",0.3)*30 + f.get("sub_r",0.3)*20)
    ch = np.full((g.rows, g.cols), " ", dtype="U1")
    co = np.zeros((g.rows, g.cols, 3), dtype=np.uint8)
    for fi in range(n_cols):
        fx_c = int((fi*g.cols/n_cols + np.sin(t*2+fi*0.7)*3) % g.cols)
        height = int((f.get("bass",0.3)*0.4 + f.get("sub_r",0.3)*0.3 + f.get("rms",0.3)*0.3) * g.rows * 0.7)
        for dy in range(min(height, g.rows)):
            fr = g.rows - 1 - dy
            frac = dy / max(height, 1)
            bri = max(0.1, (1 - frac*0.6) * (0.5 + f.get("rms",0.3)*0.5))
            hue = hue_base + frac * hue_range
            ci = "\u2588" if frac<0.2 else ("\u2593" if frac<0.4 else ("\u2592" if frac<0.6 else "\u2591"))
            ch[fr, fx_c] = ci
            R, G, B = hsv2rgb_single(hue, 0.9, bri)
            co[fr, fx_c] = (R, G, B)
    return ch, co
```

### Ice / Cold Fire (same structure, different hue range)
```python
# hue_base=0.55, hue_range=0.15 -- blue to cyan
# Lower intensity, slower movement
```

---

## Text Overlays

### Scrolling Ticker
```python
def eff_ticker(ch, co, t, text, row, speed=15, color=(80, 100, 140)):
    off = int(t * speed) % max(len(text), 1)
    doubled = text + "   " + text
    stamp(ch, co, doubled[off:off+ch.shape[1]], row, 0, color)
```

### Beat-Triggered Words
```python
def eff_beat_words(ch, co, f, words, row_center=None, color=(255,240,220)):
    if f.get("beat", 0) > 0:
        w = random.choice(words)
        r = (row_center or ch.shape[0]//2) + random.randint(-5,5)
        stamp(ch, co, w, r, (ch.shape[1]-len(w))//2, color)
```

### Fading Message Sequence
```python
def eff_fading_messages(ch, co, t, elapsed, messages, period=4.0, color_base=(220,220,220)):
    msg_idx = int(elapsed / period) % len(messages)
    phase = elapsed % period
    fade = max(0, min(1.0, phase) * min(1.0, period - phase))
    if fade > 0.05:
        v = fade
        msg = messages[msg_idx]
        cr, cg, cb = [int(c * v) for c in color_base]
        stamp(ch, co, msg, ch.shape[0]//2, (ch.shape[1]-len(msg))//2, (cr, cg, cb))
```

---

## Screen Shake
Shift entire char/color arrays on beat:
```python
def eff_shake(ch, co, f, x_amp=6, y_amp=3):
    shake_x = int(f.get("sub",0.3)*x_amp*(random.random()-0.5)*2 + f.get("bdecay",0)*4*(random.random()-0.5)*2)
    shake_y = int(f.get("bass",0.3)*y_amp*(random.random()-0.5)*2)
    if abs(shake_x) > 0:
        ch = np.roll(ch, shake_x, axis=1)
        co = np.roll(co, shake_x, axis=1)
    if abs(shake_y) > 0:
        ch = np.roll(ch, shake_y, axis=0)
        co = np.roll(co, shake_y, axis=0)
    return ch, co
```

---

## Composable Effect System

The real creative power comes from **composition**. There are three levels:

### Level 1: Character-Level Layering

Stack multiple effects as `(chars, colors)` layers:

```python
class LayerStack(EffectNode):
    """Render effects bottom-to-top with character-level compositing."""
    def add(self, effect, alpha=1.0):
        """alpha < 1.0 = probabilistic override (sparse overlay)."""
        self.layers.append((effect, alpha))

# Usage:
stack = LayerStack()
stack.add(bg_effect)           # base — fills screen
stack.add(main_effect)         # overlay on top (space chars = transparent)
stack.add(particle_effect)     # sparse overlay on top of that
ch, co = stack.render(g, f, t, S)
```

### Level 2: Pixel-Level Blending

After rendering to canvases, blend with Photoshop-style modes:

```python
class PixelBlendStack:
    """Stack canvases with blend modes for complex compositing."""
    def add(self, canvas, mode="normal", opacity=1.0)
    def composite(self) -> canvas

# Usage:
pbs = PixelBlendStack()
pbs.add(canvas_a)                        # base
pbs.add(canvas_b, "screen", 0.7)        # additive glow
pbs.add(canvas_c, "difference", 0.5)    # psychedelic interference
result = pbs.composite()
```

### Level 3: Temporal Feedback

Feed previous frame back into current frame for recursive effects:

```python
fb = FeedbackBuffer()
for each frame:
    canvas = render_current()
    canvas = fb.apply(canvas, decay=0.8, blend="screen",
                      transform="zoom", transform_amt=0.015, hue_shift=0.02)
```

### Effect Nodes — Uniform Interface

In the v2 protocol, effect nodes are used **inside** scene functions. The scene function itself returns a canvas. Effect nodes produce intermediate `(chars, colors)` that are rendered to canvas via the grid's `.render()` method or `_render_vf()`.

```python
class EffectNode:
    def render(self, g, f, t, S) -> (chars, colors)

# Concrete implementations:
class ValueFieldEffect(EffectNode):
    """Wraps a value field function + hue field function + palette."""
    def __init__(self, val_fn, hue_fn, pal=PAL_DEFAULT, sat=0.7)

class LambdaEffect(EffectNode):
    """Wrap any (g,f,t,S) -> (ch,co) function."""
    def __init__(self, fn)

class ConditionalEffect(EffectNode):
    """Switch effects based on audio features."""
    def __init__(self, condition, if_true, if_false=None)
```

### Value Field Generators (Atomic Building Blocks)

These produce float32 arrays `(rows, cols)` in range [0,1]. They are the raw visual patterns. All have signature `(g, f, t, S, **params) -> float32 array`.

```python
def vf_sinefield(g, f, t, S, bri=0.5,
                 freq=(0.13, 0.17, 0.07, 0.09), speed=(0.5, -0.4, -0.3, 0.2)):
    """Layered sine field. General purpose background/texture."""
    v1 = np.sin(g.cc*freq[0] + t*speed[0]) * np.sin(g.rr*freq[1] - t*speed[1]) * 0.5 + 0.5
    v2 = np.sin(g.cc*freq[2] - t*speed[2] + g.rr*freq[3]) * 0.4 + 0.5
    v3 = np.sin(g.dist_n*5 + t*0.2) * 0.3 + 0.4
    return np.clip((v1*0.35 + v2*0.35 + v3*0.3) * bri * (0.6 + f.get("rms",0.3)*0.6), 0, 1)

def vf_smooth_noise(g, f, t, S, octaves=3, bri=0.5):
    """Multi-octave sine approximation of Perlin noise."""
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for i in range(octaves):
        freq = 0.05 * (2 ** i); amp = 0.5 / (i + 1)
        phase = t * (0.3 + i * 0.2)
        val = val + np.sin(g.cc*freq + phase) * np.cos(g.rr*freq*0.7 - phase*0.5) * amp
    return np.clip(val * 0.5 + 0.5, 0, 1) * bri

def vf_rings(g, f, t, S, n_base=6, spacing_base=4):
    """Concentric rings, bass-driven count and wobble."""
    n = int(n_base + f.get("sub_r",0.3)*25 + f.get("bass",0.3)*10)
    sp = spacing_base + f.get("bass_r",0.3)*7 + f.get("rms",0.3)*3
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for ri in range(n):
        rad = (ri+1)*sp + f.get("bdecay",0)*15
        wobble = f.get("mid_r",0.3)*5*np.sin(g.angle*3+t*4)
        rd = np.abs(g.dist - rad - wobble)
        th = 1 + f.get("sub",0.3)*3
        val = np.maximum(val, np.clip((1 - rd/th) * (0.4 + f.get("bass",0.3)*0.8), 0, 1))
    return val

def vf_spiral(g, f, t, S, n_arms=3, tightness=2.5):
    """Logarithmic spiral arms."""
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for ai in range(n_arms):
        offset = ai * 2*np.pi / n_arms
        log_r = np.log(g.dist + 1) * tightness
        arm_phase = g.angle + offset - log_r + t * 0.8
        arm_val = np.clip(np.cos(arm_phase * n_arms) * 0.6 + 0.2, 0, 1)
        arm_val *= (0.4 + f.get("rms",0.3)*0.6) * np.clip(1 - g.dist_n*0.5, 0.2, 1)
        val = np.maximum(val, arm_val)
    return val

def vf_tunnel(g, f, t, S, speed=3.0, complexity=6):
    """Tunnel depth effect — infinite zoom feeling."""
    tunnel_d = 1.0 / (g.dist_n + 0.1)
    v1 = np.sin(tunnel_d*2 - t*speed) * 0.45 + 0.55
    v2 = np.sin(g.angle*complexity + tunnel_d*1.5 - t*2) * 0.35 + 0.55
    return np.clip(v1*0.5 + v2*0.5, 0, 1)

def vf_vortex(g, f, t, S, twist=3.0):
    """Twisting radial pattern — distance modulates angle."""
    twisted = g.angle + g.dist_n * twist * np.sin(t * 0.5)
    val = np.sin(twisted * 4 - t * 2) * 0.5 + 0.5
    return np.clip(val * (0.5 + f.get("bass",0.3)*0.8), 0, 1)

def vf_interference(g, f, t, S, n_waves=6):
    """Overlapping sine waves creating moire patterns."""
    drivers = ["mid_r", "himid_r", "bass_r", "lomid_r", "hi_r", "sub_r"]
    vals = np.zeros((g.rows, g.cols), dtype=np.float32)
    for i in range(min(n_waves, len(drivers))):
        angle = i * np.pi / n_waves
        freq = 0.06 + i * 0.03; sp = 0.5 + i * 0.3
        proj = g.cc * np.cos(angle) + g.rr * np.sin(angle)
        vals = vals + np.sin(proj*freq + t*sp) * f.get(drivers[i], 0.3) * 2.5
    return np.clip(vals * 0.12 + 0.45, 0.1, 1)

def vf_aurora(g, f, t, S, n_bands=3):
    """Horizontal aurora bands."""
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for i in range(n_bands):
        fr = 0.08 + i*0.04; fc = 0.012 + i*0.008
        sr = 0.7 + i*0.3; sc = 0.18 + i*0.12
        val = val + np.sin(g.rr*fr + t*sr) * np.sin(g.cc*fc + t*sc) * (0.6/n_bands)
    return np.clip(val * (f.get("lomid_r",0.3)*3 + 0.2), 0, 0.7)

def vf_ripple(g, f, t, S, sources=None, freq=0.3, damping=0.02):
    """Concentric ripples from point sources."""
    if sources is None: sources = [(0.5, 0.5)]
    val = np.zeros((g.rows, g.cols), dtype=np.float32)
    for ry, rx in sources:
        dy = g.rr - g.rows*ry; dx = g.cc - g.cols*rx
        d = np.sqrt(dy**2 + dx**2)
        val = val + np.sin(d*freq - t*4) * np.exp(-d*damping) * 0.5
    return np.clip(val + 0.5, 0, 1)

def vf_plasma(g, f, t, S):
    """Classic plasma: sum of sines at different orientations and speeds."""
    v = np.sin(g.cc * 0.03 + t * 0.7) * 0.5
    v = v + np.sin(g.rr * 0.04 - t * 0.5) * 0.4
    v = v + np.sin((g.cc * 0.02 + g.rr * 0.03) + t * 0.3) * 0.3
    v = v + np.sin(g.dist_n * 4 - t * 0.8) * 0.3
    return np.clip(v * 0.5 + 0.5, 0, 1)

def vf_diamond(g, f, t, S, freq=0.15):
    """Diamond/checkerboard pattern."""
    val = np.abs(np.sin(g.cc * freq + t * 0.5)) * np.abs(np.sin(g.rr * freq * 1.2 - t * 0.3))
    return np.clip(val * (0.6 + f.get("rms",0.3)*0.8), 0, 1)

def vf_noise_static(g, f, t, S, density=0.4):
    """Random noise — different each frame. Non-deterministic."""
    return np.random.random((g.rows, g.cols)).astype(np.float32) * density * (0.5 + f.get("rms",0.3)*0.5)
```

### Hue Field Generators (Color Mapping)

These produce float32 hue arrays [0,1]. Independently combinable with any value field. Each is a factory returning a closure with signature `(g, f, t, S) -> float32 array`. Can also be a plain float for fixed hue.

```python
def hf_fixed(hue):
    """Single hue everywhere."""
    def fn(g, f, t, S):
        return np.full((g.rows, g.cols), hue, dtype=np.float32)
    return fn

def hf_angle(offset=0.0):
    """Hue mapped to angle from center — rainbow wheel."""
    def fn(g, f, t, S):
        return (g.angle / (2 * np.pi) + offset + t * 0.05) % 1.0
    return fn

def hf_distance(base=0.5, scale=0.02):
    """Hue mapped to distance from center."""
    def fn(g, f, t, S):
        return (base + g.dist * scale + t * 0.03) % 1.0
    return fn

def hf_time_cycle(speed=0.1):
    """Hue cycles uniformly over time."""
    def fn(g, f, t, S):
        return np.full((g.rows, g.cols), (t * speed) % 1.0, dtype=np.float32)
    return fn

def hf_audio_cent():
    """Hue follows spectral centroid — timbral color shifting."""
    def fn(g, f, t, S):
        return np.full((g.rows, g.cols), f.get("cent", 0.5) * 0.3, dtype=np.float32)
    return fn

def hf_gradient_h(start=0.0, end=1.0):
    """Left-to-right hue gradient."""
    def fn(g, f, t, S):
        h = np.broadcast_to(
            start + (g.cc / g.cols) * (end - start),
            (g.rows, g.cols)
        ).copy()  # .copy() is CRITICAL — see troubleshooting.md
        return h % 1.0
    return fn

def hf_gradient_v(start=0.0, end=1.0):
    """Top-to-bottom hue gradient."""
    def fn(g, f, t, S):
        h = np.broadcast_to(
            start + (g.rr / g.rows) * (end - start),
            (g.rows, g.cols)
        ).copy()
        return h % 1.0
    return fn

def hf_plasma(speed=0.3):
    """Plasma-style hue field — organic color variation."""
    def fn(g, f, t, S):
        return (np.sin(g.cc*0.02 + t*speed)*0.5 + np.sin(g.rr*0.015 + t*speed*0.7)*0.5) % 1.0
    return fn
```

### Combining Value Fields

The combinatorial explosion comes from mixing value fields with math:

```python
# Multiplication = intersection (only shows where both have brightness)
combined = vf_plasma(g,f,t,S) * vf_vortex(g,f,t,S)

# Addition = union (shows both, clips at 1.0)
combined = np.clip(vf_rings(g,f,t,S) + vf_spiral(g,f,t,S), 0, 1)

# Interference = beat pattern (shows XOR-like patterns)
combined = np.abs(vf_plasma(g,f,t,S) - vf_tunnel(g,f,t,S))

# Modulation = one effect shapes the other
combined = vf_rings(g,f,t,S) * (0.3 + 0.7 * vf_plasma(g,f,t,S))

# Maximum = shows the brightest of two effects
combined = np.maximum(vf_spiral(g,f,t,S), vf_aurora(g,f,t,S))
```

### Full Scene Example (v2 — Canvas Return)

A v2 scene function composes effects internally and returns a pixel canvas:

```python
def scene_complex(r, f, t, S):
    """v2 scene function: returns canvas (uint8 H,W,3).
    r = Renderer, f = audio features, t = time, S = persistent state dict."""
    g = r.grids["md"]
    rows, cols = g.rows, g.cols
    
    # 1. Value field composition
    plasma = vf_plasma(g, f, t, S)
    vortex = vf_vortex(g, f, t, S, twist=4.0)
    combined = np.clip(plasma * 0.6 + vortex * 0.5 + plasma * vortex * 0.4, 0, 1)
    
    # 2. Color from hue field
    h = (hf_angle(0.3)(g,f,t,S) * 0.5 + hf_time_cycle(0.08)(g,f,t,S) * 0.5) % 1.0
    
    # 3. Render to canvas via _render_vf helper
    canvas = _render_vf(g, combined, h, sat=0.75, pal=PAL_DENSE)
    
    # 4. Optional: blend a second layer
    overlay = _render_vf(r.grids["sm"], vf_rings(r.grids["sm"],f,t,S),
                         hf_fixed(0.6)(r.grids["sm"],f,t,S), pal=PAL_BLOCK)
    canvas = blend_canvas(canvas, overlay, "screen", 0.4)
    
    return canvas
    
# In the render_clip() loop (handled by the framework):
# canvas = scene_fn(r, f, t, S)
# canvas = tonemap(canvas, gamma=scene_gamma)
# canvas = feedback.apply(canvas, ...)
# canvas = shader_chain.apply(canvas, f=f, t=t)
# pipe.stdin.write(canvas.tobytes())
```

Vary the **value field combo**, **hue field**, **palette**, **blend modes**, **feedback config**, and **shader chain** per section for maximum visual variety. With 12 value fields × 8 hue fields × 14 palettes × 20 blend modes × 7 feedback transforms × 38 shaders, the combinations are effectively infinite.
