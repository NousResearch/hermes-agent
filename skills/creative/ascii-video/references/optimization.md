# Optimization Reference

## Hardware Detection

Detect the user's hardware at script startup and adapt rendering parameters automatically. Never hardcode worker counts or resolution.

### CPU and Memory Detection

```python
import multiprocessing
import platform
import shutil
import os

def detect_hardware():
    """Detect hardware capabilities and return render config."""
    cpu_count = multiprocessing.cpu_count()
    
    # Leave 1-2 cores free for OS + ffmpeg encoding
    if cpu_count >= 16:
        workers = cpu_count - 2
    elif cpu_count >= 8:
        workers = cpu_count - 1
    elif cpu_count >= 4:
        workers = cpu_count - 1
    else:
        workers = max(1, cpu_count)
    
    # Memory detection (platform-specific)
    try:
        if platform.system() == "Darwin":
            import subprocess
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        mem_bytes = int(line.split()[1]) * 1024
                        break
        else:
            mem_bytes = 8 * 1024**3  # assume 8GB on unknown
    except Exception:
        mem_bytes = 8 * 1024**3

    mem_gb = mem_bytes / (1024**3)
    
    # Each worker uses ~50-150MB depending on grid sizes
    # Cap workers if memory is tight
    mem_per_worker_mb = 150
    max_workers_by_mem = int(mem_gb * 1024 * 0.6 / mem_per_worker_mb)  # use 60% of RAM
    workers = min(workers, max_workers_by_mem)
    
    # ffmpeg availability and codec support
    has_ffmpeg = shutil.which("ffmpeg") is not None
    
    return {
        "cpu_count": cpu_count,
        "workers": workers,
        "mem_gb": mem_gb,
        "platform": platform.system(),
        "arch": platform.machine(),
        "has_ffmpeg": has_ffmpeg,
    }
```

### Adaptive Quality Profiles

Scale resolution, FPS, CRF, and grid density based on hardware:

```python
def quality_profile(hw, target_duration_s, user_preference="auto"):
    """
    Returns render settings adapted to hardware.
    user_preference: "auto", "draft", "preview", "production", "max"
    """
    if user_preference == "draft":
        return {"vw": 960, "vh": 540, "fps": 12, "crf": 28, "workers": min(4, hw["workers"]),
                "grid_scale": 0.5, "shaders": "minimal", "particles_max": 200}
    
    if user_preference == "preview":
        return {"vw": 1280, "vh": 720, "fps": 15, "crf": 25, "workers": hw["workers"],
                "grid_scale": 0.75, "shaders": "standard", "particles_max": 500}
    
    if user_preference == "max":
        return {"vw": 3840, "vh": 2160, "fps": 30, "crf": 15, "workers": hw["workers"],
                "grid_scale": 2.0, "shaders": "full", "particles_max": 3000}
    
    # "production" or "auto"
    # Auto-detect: estimate render time, downgrade if it would take too long
    n_frames = int(target_duration_s * 24)
    est_seconds_per_frame = 0.18  # ~180ms at 1080p
    est_total_s = n_frames * est_seconds_per_frame / max(1, hw["workers"])
    
    if hw["mem_gb"] < 4 or hw["cpu_count"] <= 2:
        # Low-end: 720p, 15fps
        return {"vw": 1280, "vh": 720, "fps": 15, "crf": 23, "workers": hw["workers"],
                "grid_scale": 0.75, "shaders": "standard", "particles_max": 500}
    
    if est_total_s > 3600:  # would take over an hour
        # Downgrade to 720p to speed up
        return {"vw": 1280, "vh": 720, "fps": 24, "crf": 20, "workers": hw["workers"],
                "grid_scale": 0.75, "shaders": "standard", "particles_max": 800}
    
    # Standard production: 1080p 24fps
    return {"vw": 1920, "vh": 1080, "fps": 24, "crf": 20, "workers": hw["workers"],
            "grid_scale": 1.0, "shaders": "full", "particles_max": 1200}


def apply_quality_profile(profile):
    """Set globals from quality profile."""
    global VW, VH, FPS, N_WORKERS
    VW = profile["vw"]
    VH = profile["vh"]
    FPS = profile["fps"]
    N_WORKERS = profile["workers"]
    # Grid sizes scale with resolution
    # CRF passed to ffmpeg encoder
    # Shader set determines which post-processing is active
```

### CLI Integration

```python
parser = argparse.ArgumentParser()
parser.add_argument("--quality", choices=["draft", "preview", "production", "max", "auto"],
                    default="auto", help="Render quality preset")
parser.add_argument("--workers", type=int, default=0, help="Override worker count (0=auto)")
parser.add_argument("--resolution", type=str, default="", help="Override resolution e.g. 1280x720")
args = parser.parse_args()

hw = detect_hardware()
if args.workers > 0:
    hw["workers"] = args.workers
profile = quality_profile(hw, target_duration, args.quality)
if args.resolution:
    w, h = args.resolution.split("x")
    profile["vw"], profile["vh"] = int(w), int(h)
apply_quality_profile(profile)

log(f"Hardware: {hw['cpu_count']} cores, {hw['mem_gb']:.1f}GB RAM, {hw['platform']}")
log(f"Render:   {profile['vw']}x{profile['vh']} @{profile['fps']}fps, "
    f"CRF {profile['crf']}, {profile['workers']} workers")
```

## Performance Budget

Target: 100-200ms per frame (5-10 fps single-threaded, 40-80 fps across 8 workers).

| Component | Time | Notes |
|-----------|------|-------|
| Feature extraction | 1-5ms | Pre-computed for all frames before render |
| Effect function | 2-15ms | Vectorized numpy, avoid Python loops |
| Character render | 80-150ms | **Bottleneck** -- per-cell Python loop |
| Shader pipeline | 5-25ms | Depends on active shaders |
| ffmpeg encode | ~5ms | Amortized by pipe buffering |

## Bitmap Pre-Rasterization

Rasterize every character at init, not per-frame:

```python
# At init time -- done once
for c in all_characters:
    img = Image.new("L", (cell_w, cell_h), 0)
    ImageDraw.Draw(img).text((0, 0), c, fill=255, font=font)
    bitmaps[c] = np.array(img, dtype=np.float32) / 255.0  # float32 for fast multiply

# At render time -- fast lookup
bitmap = bitmaps[char]
canvas[y:y+ch, x:x+cw] = np.maximum(canvas[y:y+ch, x:x+cw],
                                      (bitmap[:,:,None] * color).astype(np.uint8))
```

Collect all characters from all palettes + overlay text into the init set. Lazy-init for any missed characters.

## Coordinate Array Caching

Pre-compute all grid-relative coordinate arrays at init, not per-frame:

```python
# These are O(rows*cols) and used in every effect
self.rr = np.arange(rows)[:, None]    # row indices
self.cc = np.arange(cols)[None, :]    # col indices
self.dist = np.sqrt(dx**2 + dy**2)   # distance from center
self.angle = np.arctan2(dy, dx)       # angle from center
self.dist_n = ...                      # normalized distance
```

## Vectorized Effect Patterns

### Avoid Per-Cell Python Loops in Effects

The render loop (compositing bitmaps) is unavoidably per-cell. But effect functions must be fully vectorized numpy -- never iterate over rows/cols in Python.

Bad (O(rows*cols) Python loop):
```python
for r in range(rows):
    for c in range(cols):
        val[r, c] = math.sin(c * 0.1 + t) * math.cos(r * 0.1 - t)
```

Good (vectorized):
```python
val = np.sin(g.cc * 0.1 + t) * np.cos(g.rr * 0.1 - t)
```

### Vectorized Matrix Rain

The naive per-column per-trail-pixel loop is the second biggest bottleneck after the render loop. Use numpy fancy indexing:

```python
# Instead of nested Python loops over columns and trail pixels:
# Build row index arrays for all active trail pixels at once
all_rows = []
all_cols = []
all_fades = []
for c in range(cols):
    head = int(state["ry"][c])
    trail_len = state["rln"][c]
    for i in range(trail_len):
        row = head - i
        if 0 <= row < rows:
            all_rows.append(row)
            all_cols.append(c)
            all_fades.append(1.0 - i / trail_len)

# Vectorized assignment
ar = np.array(all_rows)
ac = np.array(all_cols)
af = np.array(all_fades, dtype=np.float32)
# Assign chars and colors in bulk using fancy indexing
ch[ar, ac] = ...  # vectorized char assignment
co[ar, ac, 1] = (af * bri * 255).astype(np.uint8)  # green channel
```

### Vectorized Fire Columns

Same pattern -- accumulate index arrays, assign in bulk:

```python
fire_val = np.zeros((rows, cols), dtype=np.float32)
for fi in range(n_cols):
    fx_c = int((fi * cols / n_cols + np.sin(t * 2 + fi * 0.7) * 3) % cols)
    height = int(energy * rows * 0.7)
    dy = np.arange(min(height, rows))
    fr = rows - 1 - dy
    frac = dy / max(height, 1)
    # Width spread: base columns wider at bottom
    for dx in range(-1, 2):  # 3-wide columns
        c = fx_c + dx
        if 0 <= c < cols:
            fire_val[fr, c] = np.maximum(fire_val[fr, c],
                                          (1 - frac * 0.6) * (0.5 + rms * 0.5))
# Now map fire_val to chars and colors in one vectorized pass
```

## Bloom Optimization

**Do NOT use `scipy.ndimage.uniform_filter`** -- measured at 424ms/frame.

Use 4x downsample + manual box blur instead -- 84ms/frame (5x faster):

```python
sm = canvas[::4, ::4].astype(np.float32)  # 4x downsample
br = np.where(sm > threshold, sm, 0)
for _ in range(3):                          # 3-pass manual box blur
    p = np.pad(br, ((1,1),(1,1),(0,0)), mode='edge')
    br = (p[:-2,:-2] + p[:-2,1:-1] + p[:-2,2:] +
          p[1:-1,:-2] + p[1:-1,1:-1] + p[1:-1,2:] +
          p[2:,:-2] + p[2:,1:-1] + p[2:,2:]) / 9.0
bl = np.repeat(np.repeat(br, 4, axis=0), 4, axis=1)[:H, :W]
```

## Vignette Caching

Distance field is resolution- and strength-dependent, never changes per frame:

```python
_vig_cache = {}
def sh_vignette(canvas, strength):
    key = (canvas.shape[0], canvas.shape[1], round(strength, 2))
    if key not in _vig_cache:
        Y = np.linspace(-1, 1, H)[:, None]
        X = np.linspace(-1, 1, W)[None, :]
        _vig_cache[key] = np.clip(1.0 - np.sqrt(X**2+Y**2) * strength, 0.15, 1).astype(np.float32)
    return np.clip(canvas * _vig_cache[key][:,:,None], 0, 255).astype(np.uint8)
```

Same pattern for CRT barrel distortion (cache remap coordinates).

## Film Grain Optimization

Generate noise at half resolution, tile up:

```python
noise = np.random.randint(-amt, amt+1, (H//2, W//2, 1), dtype=np.int16)
noise = np.repeat(np.repeat(noise, 2, axis=0), 2, axis=1)[:H, :W]
```

2x blocky grain looks like film grain and costs 1/4 the random generation.

## Parallel Rendering

### Worker Architecture

```python
hw = detect_hardware()
N_WORKERS = hw["workers"]

# Batch splitting (for non-clip architectures)
batch_size = (n_frames + N_WORKERS - 1) // N_WORKERS
batches = [(i, i*batch_size, min((i+1)*batch_size, n_frames), features, seg_path) ...]

with multiprocessing.Pool(N_WORKERS) as pool:
    segments = pool.starmap(render_batch, batches)
```

### Per-Clip Parallelism (Preferred for Segmented Videos)

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(render_clip, seg, features, path): seg["id"]
               for seg, path in clip_args}
    for fut in as_completed(futures):
        clip_id = futures[fut]
        try:
            fut.result()
            log(f"  {clip_id} done")
        except Exception as e:
            log(f"  {clip_id} FAILED: {e}")
```

### Worker Isolation

Each worker:
- Creates its own `Renderer` instance (with full grid + bitmap init)
- Opens its own ffmpeg subprocess
- Has independent random seed (`random.seed(batch_id * 10000)`)
- Writes to its own segment file and stderr log

### ffmpeg Pipe Safety

**CRITICAL**: Never `stderr=subprocess.PIPE` with long-running ffmpeg. The stderr buffer fills at ~64KB and deadlocks:

```python
# WRONG -- will deadlock
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

# RIGHT -- stderr to file
stderr_fh = open(err_path, "w")
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_fh)
# ... write all frames ...
pipe.stdin.close()
pipe.wait()
stderr_fh.close()
```

### Concatenation

```python
with open(concat_file, "w") as cf:
    for seg in segments:
        cf.write(f"file '{seg}'\n")

cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file]
if audio_path:
    cmd += ["-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
else:
    cmd += ["-c:v", "copy"]
cmd.append(output_path)
subprocess.run(cmd, capture_output=True, check=True)
```

## Particle System Performance

Cap particle counts based on quality profile:

| System | Low | Standard | High |
|--------|-----|----------|------|
| Explosion | 300 | 1000 | 2500 |
| Embers | 500 | 1500 | 3000 |
| Starfield | 300 | 800 | 1500 |
| Dissolve | 200 | 600 | 1200 |

Cull by truncating lists:
```python
MAX_PARTICLES = profile.get("particles_max", 1200)
if len(S["px"]) > MAX_PARTICLES:
    for k in ("px", "py", "vx", "vy", "life", "char"):
        S[k] = S[k][-MAX_PARTICLES:]  # keep newest
```

## Memory Management

- Feature arrays: pre-computed for all frames, shared across workers via fork semantics (COW)
- Canvas: allocated once per worker, reused (`np.zeros(...)`)
- Character arrays: allocated per frame (cheap -- rows*cols U1 strings)
- Bitmap cache: ~500KB per grid size, initialized once per worker

Total memory per worker: ~50-150MB. Total: ~400-800MB for 8 workers.

For low-memory systems (< 4GB), reduce worker count and use smaller grids.

## Brightness Verification

After render, spot-check brightness at sample timestamps:

```python
for t in [2, 30, 60, 120, 180]:
    cmd = ["ffmpeg", "-ss", str(t), "-i", output_path,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True)
    arr = np.frombuffer(r.stdout, dtype=np.uint8)
    print(f"t={t}s  mean={arr.mean():.1f}  max={arr.max()}")
```

Target: mean > 5 for quiet sections, mean > 15 for active sections. If consistently below, increase brightness floor in effects and/or global boost multiplier.

## Render Time Estimates

Scale with hardware. Baseline: 1080p, 24fps, ~180ms/frame/worker.

| Duration | Frames | 4 workers | 8 workers | 16 workers |
|----------|--------|-----------|-----------|------------|
| 30s | 720 | ~3 min | ~2 min | ~1 min |
| 2 min | 2,880 | ~13 min | ~7 min | ~4 min |
| 3.5 min | 5,040 | ~23 min | ~12 min | ~6 min |
| 5 min | 7,200 | ~33 min | ~17 min | ~9 min |
| 10 min | 14,400 | ~65 min | ~33 min | ~17 min |

At 720p: multiply times by ~0.5. At 4K: multiply by ~4.

Heavier effects (many particles, dense grids, extra shader passes) add ~20-50%.
