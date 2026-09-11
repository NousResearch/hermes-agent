# Particles Reference

Particle systems in TouchDesigner — modern POPs (Particle Operators) and the legacy particleSOP path.

For instancing static geometry (without per-instance lifetime/velocity), see `geometry-comp.md`. For GLSL-driven feedback simulations (no particle abstraction), see `operator-tips.md` (Feedback TOP section).

Always call `td_get_par_info` for the op type before setting params. Param names below reflect TD 2025.32 — verify before relying on them.

---

## Two Paths: POPs vs. SOPs

| | **POP family** (modern) | **particleSOP** (legacy) |
|---|---|---|
| GPU? | Yes (compute) | No (CPU) |
| Particle count | 100k+ comfortably | ~5k before slowdown |
| API style | Source / Force / Solver / Render chain | Single op with many params |
| Use for | New projects, anything intensive | Quick demos, low counts, TD < 2023 |

**Default to POPs.** Only fall back to particleSOP if a POP variant of an op you need doesn't exist.

---

## POP Pipeline Overview

A POP system is a chain of operators inside a `geometryCOMP`:

```
popSourceTOP / popSourceSOP   ← spawn new particles
        ↓
popForceTOP (gravity, wind, etc.)
        ↓
popForceTOP (attractor, vortex, ...)
        ↓
popDeleteTOP (lifetime, bounds)
        ↓
popSolverTOP                  ← integrates velocity, updates positions
        ↓
[render via geometryCOMP / glslMAT instancing]
```

POP buffers carry standard channels: `P` (position), `v` (velocity), `life`, `id`, `Cd` (color), plus any custom channels you add.

---

## Minimal POP Setup

```python
# Create a geometry COMP to hold the POP network
geo = root.create(geometryCOMP, 'particles_geo')

# 1. Source — emit particles from a point
src = geo.create(popSourceTOP, 'src')
src.par.birthrate = 500          # per second
src.par.life = 4.0                # seconds

# 2. Gravity force
grav = geo.create(popForceTOP, 'gravity')
grav.par.forcetype = 'gravity'
grav.par.fy = -9.8

# 3. Lifetime cleanup
delp = geo.create(popDeleteTOP, 'cull')
delp.par.condition = 'lifeleq'    # delete when life <= 0
delp.par.value = 0

# 4. Solver
solv = geo.create(popSolverTOP, 'solver')
solv.par.timestep = 'frame'

# Wire: source → force → delete → solver
src.outputConnectors[0].connect(grav.inputConnectors[0])
grav.outputConnectors[0].connect(delp.inputConnectors[0])
delp.outputConnectors[0].connect(solv.inputConnectors[0])
```

The `popSolverTOP` output IS the live particle buffer. Render it via `glslMAT` instancing on a small SOP (sphere, point) as the "shape" of each particle.

---

## Common Forces

| Force type | Effect | Common params |
|---|---|---|
| `gravity` | Constant directional pull | `fx`, `fy`, `fz` |
| `wind` | Constant velocity addition | `wx`, `wy`, `wz` |
| `drag` | Velocity damping over time | `dragstrength` |
| `noise` | Curl-noise turbulence | `noiseamp`, `noisefreq`, `noiseseed` |
| `attractor` | Pull toward a point | `position`, `strength`, `falloff` |
| `vortex` | Swirl around an axis | `axis`, `strength` |
| `point` (custom) | GLSL-evaluated arbitrary force | via `popforceadvancedTOP` |

Stack multiple `popForceTOP`s in series — each modifies velocity additively.

---

## Lifecycle Patterns

### Continuous emission (e.g. smoke plume)

```python
src.par.birthrate = 800
src.par.life = 6.0       # variance via 'lifevariance'
src.par.lifevariance = 1.5
```

### Burst emission (e.g. explosion)

```python
src.par.birthrate = 0    # no continuous emission
src.par.burst.pulse()    # one burst on demand (verify param name)
src.par.burstcount = 5000
src.par.life = 1.5
```

### Beat-triggered burst

Wire a `triggerCHOP` (from audio or MIDI) to pulse the burst:

```python
op('/project1/audio_kick_trigger').outputConnectors[0].connect(...)
# Then via a chopExecuteDAT, on each kick:
def offToOn(channel, sampleIndex, val, prev):
    op('/project1/particles_geo/src').par.burst.pulse()
    return
```

---

## Rendering Particles

### Point Sprites (simplest)

```python
# Inside the geometryCOMP, render the solver output directly
# The geo's first SOP child becomes the geometry
# But for POPs, we typically render via glslMAT on a small "shape"

# Simple billboard sphere per particle:
shape = geo.create(sphereSOP, 'shape')
shape.par.rad = 0.05
shape.par.rows = 6; shape.par.cols = 6   # low-poly to keep it fast

# Material that uses POP buffer for instancing
mat = root.create(glslMAT, 'particle_mat')
# Configure mat.par.instancingTOP = solver output (verify param name)
```

The exact instancing setup varies by TD version — call `td_get_hints(topic='popInstancing')` (or `popRender` / `instancing` — try a few).

### GPU Sprites via glslcopyPOP

For dense smoke/fire-like effects, use a `glslcopyPOP` that writes per-particle color/size from a compute shader, then render as point sprites with additive blending in a `renderTOP`.

---

## Collisions

```python
# Collision detection against an SOP
coll = geo.create(popCollideTOP, 'ground_coll')
coll.par.collidewithsop = '/project1/ground_geo'  # path to colliding SOP
coll.par.bounce = 0.3
coll.par.friction = 0.1
# Insert between force and solver
```

For plane/box collisions only, use `popPlaneCollideTOP` (cheaper).

---

## Custom Per-Particle Data

Add a custom channel via `popAttribCreateTOP` (or by writing through `glslcopyPOP`):

```python
# Add a "phase" attribute initialized random per-particle, used in render shader
attr = geo.create(popAttribCreateTOP, 'add_phase')
attr.par.attribname = 'phase'
attr.par.value0 = 'rand(@id)'   # expression in TD's POP attribute language
```

Then in the render shader, `texture(sTDPOPInputs[0].phase, ...)` (or whichever sampler convention your TD version uses — verify with `td_get_docs(topic='pops')`).

---

## Legacy particleSOP (Use Sparingly)

For quick demos or low-count systems:

```python
# Inside a geo
psrc = geo.create(addSOP, 'point_src')      # source: a single point
psrc.par.points = '0 0 0'

part = geo.create(particleSOP, 'particles')
part.par.life = 3.0
part.par.birthrate = 100
part.par.gravityy = -9.8
part.par.windx = 0.5
part.inputConnectors[0].connect(psrc)
```

CPU-bound. Beyond ~5,000 active particles you'll see frame drops.

---

## Pitfalls

1. **Particles don't appear** — usually a render-side issue. Check via `td_get_screenshot` on the solver output (renders the buffer as a TOP-like view in newer TD). Then check the `geometryCOMP`'s render path.
2. **Burst won't fire** — verify the `burst` param is a pulse, not a toggle. Pulses must use `.pulse()`, not `= True`.
3. **Particles teleport on first frame** — uninitialized velocity. Set `popSourceTOP.par.initialvelocityX/Y/Z` or zero them explicitly.
4. **Gravity feels wrong** — TD's "1 unit" depends on your scene scale. Start with `fy = -1.0` and scale up rather than using real-world 9.8.
5. **High birthrate = stuttering** — birthrate is per-second, not per-frame. At 60fps, `birthrate = 6000` is 100/frame which is fine; `birthrate = 600000` will tank.
6. **POP solver order matters** — forces apply in the order they appear in the chain. Putting gravity AFTER drag dampens gravity itself; usually not what you want.
7. **Instancing param name varies** — `mat.par.instancingTOP` vs. `mat.par.instanceop` vs. `mat.par.instances` differs across TD versions. Always check `td_get_par_info(op_type='glslMAT')`.
8. **Cooking dependency loops** — POP solvers create implicit time-loops. The "cook dependency loop" warning is expected and harmless for POPs.
9. **CHOP-driven force values** — when a force param is expression-bound to a CHOP (e.g., audio-reactive gravity), make sure the CHOP cooks before the solver. If not, force lags by one frame.

---

## Performance Targets

| Particle count | Setup | Frame budget @ 60fps |
|---|---|---|
| < 1k | particleSOP fine | trivial |
| 1k - 10k | POPs, simple forces | ~2-5ms |
| 10k - 100k | POPs, GPU-only forces | ~5-15ms |
| 100k+ | `glslcopyPOP`, custom compute | ~10-25ms |
| 1M+ | Custom GPU buffer, no POP framework | depends on shader |

Use `td_get_perf` to find which op in the POP chain is the bottleneck.

---

## Quick Recipes

| Goal | Pipeline |
|---|---|
| Smoke plume | `popSourceTOP` (point) → gravity + wind + noise → `popDeleteTOP` (life) → solver → glslMAT instancing |
| Beat-triggered burst | `triggerCHOP` (audio) → chopExecuteDAT pulses `popSourceTOP.par.burst` |
| Fireworks shell | Burst at point → drag + gravity → secondary burst on lifetime threshold |
| Snow/rain | Continuous emission across XZ plane (high y), gravity + small wind, infinite life box-deleted |
| Sparks | Burst, very short life (0.3s), bright additive render, motion blur via feedback |
| Audio particles | Birthrate driven by audio envelope, color driven by frequency band |
