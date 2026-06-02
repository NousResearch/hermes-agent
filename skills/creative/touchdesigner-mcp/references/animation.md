# Animation Reference

Patterns for time-based motion — keyframes, LFOs, timers, easing, expression-driven animation.

Always call `td_get_par_info` for the op type before setting params. Param names below reflect TD 2025.32 but verify if errors fire.

---

## Time Sources

TD has three time references — pick the right one.

| Expression | Behavior | Use for |
|---|---|---|
| `absTime.seconds` | Wall-clock seconds since TD started. Never resets. | Continuous motion, GLSL `uTime`, infinite loops |
| `absTime.frame` | Wall-clock frame count. | Frame-accurate triggers |
| `me.time.frame` | Local component frame count (resets on play/stop). | Per-COMP animation timeline |
| `me.time.seconds` | Local component seconds. | Same, in seconds |

**Rule:** for shaders and continuous motion use `absTime.seconds`. For triggered/looping animations inside a COMP use `me.time.*`.

---

## LFO CHOP — Cyclic Motion

The simplest periodic driver. Fast, GPU-cheap, expression-friendly.

```python
lfo = root.create(lfoCHOP, 'rot_driver')
lfo.par.type = 'sin'        # 'sin' | 'cos' | 'ramp' | 'square' | 'triangle' | 'pulse'
lfo.par.frequency = 0.25    # cycles per second
lfo.par.amplitude = 1.0
lfo.par.offset = 0.0
lfo.par.phase = 0.0         # 0-1, useful for offsetting parallel LFOs
```

**Drive a parameter via export:**

```python
op('/project1/geo1').par.rx.mode = ParMode.EXPRESSION
op('/project1/geo1').par.rx.expr = "op('rot_driver')['chan1'] * 360"
```

**Multiple synced LFOs (X/Y/Z rotation with phase offsets):**
Create one LFO with three channels and phase-offset each, or use three LFOs and offset their `phase` params (0.0, 0.33, 0.66).

---

## Timer CHOP — Triggered Sequences

For run-once animations, beat-locked sequences, or stage-based logic.

```python
timer = root.create(timerCHOP, 'fade_timer')
timer.par.length = 4.0       # cycle length in seconds
timer.par.cycle = False      # run once vs. loop
timer.par.outputseconds = True
```

Output channels: `timer_fraction` (0→1 across the cycle), `running`, `done`, `cycles`.

**Start the timer:**
```python
timer.par.start.pulse()
```

**Drive a fade:**
```python
op('/project1/level1').par.opacity.mode = ParMode.EXPRESSION
op('/project1/level1').par.opacity.expr = "op('fade_timer')['timer_fraction']"
```

**Easing on the timer fraction** — apply in the expression itself:

```python
# Smoothstep: ease in/out
expr = "smoothstep(0, 1, op('fade_timer')['timer_fraction'])"
# Cubic ease-out: 1 - (1-t)^3
expr = "1 - pow(1 - op('fade_timer')['timer_fraction'], 3)"
```

---

## Pattern CHOP — Custom Curves

For arbitrary waveforms (saw ramps, easing curves, custom envelopes).

```python
pat = root.create(patternCHOP, 'envelope')
pat.par.type = 'gaussian'    # 'gaussian' | 'ramp' | 'square' | 'sin' | etc.
pat.par.length = 60          # samples
pat.par.cyclelength = 1.0    # seconds at TD framerate
```

Combine with `lookupCHOP` to remap a 0-1 driver through a custom curve.

---

## Animation COMP — Keyframe-Based

For multi-keyframe motion graphics. Each animationCOMP holds channels with keyframes editable in the Animation Editor.

```python
anim = root.create(animationCOMP, 'intro_anim')
# By default has channels chan1..chanN; access via:
# op('intro_anim').par.length, .par.play, .par.cue, etc.

# Drive a parameter from a channel
op('/project1/text1').par.tx.mode = ParMode.EXPRESSION
op('/project1/text1').par.tx.expr = "op('intro_anim/out1')['chan1']"
```

**Keyframes are typically edited in the UI** (Animation Editor), but can be set via `keyframes` table internally. For programmatic keyframe creation, use `td_execute_python`:

```python
# Get the channel CHOP inside an animationCOMP
ch = op('/project1/intro_anim/chans')
# Insert a key (advanced API — verify with td_get_par_info(op_type='animationCOMP'))
ch.appendKey('chan1', frame=0, value=0.0, expression=None)
ch.appendKey('chan1', frame=120, value=1.0)
```

For most use cases, drive params with LFO/Timer/Pattern CHOPs instead — simpler and scriptable.

---

## Easing in Expressions

TD's expression evaluator supports Python math. Common easing forms:

```python
# Linear
"t"

# Smoothstep (classic ease-in-out)
"smoothstep(0, 1, t)"

# Ease-out cubic
"1 - pow(1 - t, 3)"

# Ease-in cubic
"pow(t, 3)"

# Ease-in-out cubic
"3*t*t - 2*t*t*t"

# Bounce (manual, simplified)
"abs(sin(t * 6.28 * 3) * (1 - t))"
```

Where `t` is `op('fade_timer')['timer_fraction']` or any 0-1 driver.

---

## Filter CHOP — Smoothing Existing Channels

Smooth out jittery values (e.g., audio analysis, sensor data) before driving visuals.

```python
filt = root.create(filterCHOP, 'smooth')
filt.par.filter = 'gaussian'   # or 'lowpass'
filt.par.width = 0.5            # smoothing window in seconds
filt.inputConnectors[0].connect(op('raw_signal'))
```

**WARNING:** Do NOT use Filter CHOP on AudioSpectrum output in timeslice mode — it expands the sample count and averages bins to near-zero. See `audio-reactive.md`.

---

## Lag CHOP — Asymmetric Attack/Release

Different speeds for rising vs. falling values. Standard for visualizing audio envelopes.

```python
lag = root.create(lagCHOP, 'env_smooth')
lag.par.lag1 = 0.02   # attack (rise time, seconds)
lag.par.lag2 = 0.30   # release (fall time, seconds)
lag.inputConnectors[0].connect(op('raw_envelope'))
```

Fast attack, slow release = classic VU-meter feel.

---

## Per-Frame Driving via Script DAT

For complex per-frame logic that doesn't fit expressions, use a `executeDAT` (`onFrameStart` callback) or a `chopExecuteDAT`.

```python
# In an executeDAT (frameStart):
def onFrameStart(frame):
    t = absTime.seconds
    op('/project1/circle').par.tx = math.sin(t * 2.0) * 3.0
    op('/project1/circle').par.ty = math.cos(t * 2.0) * 3.0
    return
```

Heavy logic should still be in CHOPs (CPU-cheap, deterministic). Reserve scripts for one-shots or non-realtime branching.

---

## Pitfalls

1. **Frame rate dependency** — `me.time.frame` is in TD project frames (default 60). If your project rate changes, motion speed changes. Use `seconds` for rate-independent timing.
2. **Cooking budget** — every CHOP that drives a parameter cooks every frame. Consolidate drivers (one big mathCHOP > many small ones).
3. **Expression mode** — params default to `CONSTANT`. `par.X.expr = ...` is ignored unless `par.X.mode = ParMode.EXPRESSION`.
4. **Animation editor edits** — keyframes set via UI live in the animationCOMP's internal keyframe table. They survive save/reopen. Programmatic keys via `appendKey()` work but verify the API with `td_get_docs(topic='animation')` first.
5. **Looping animations** — for seamless loops, `length` must equal `cyclelength` and the start/end values must match. Otherwise expect a visible jump.

---

## Quick Recipes

| Goal | Simplest path |
|---|---|
| Continuous rotation | LFO CHOP `type='ramp'`, expr → `geo.par.rx` |
| Fade in over 2s | Timer CHOP `length=2`, smoothstep expr → `level.par.opacity` |
| Pulse on every beat | `triggerCHOP` from audio → drive scale via expression |
| 3D Lissajous orbit | Two LFOs with different freq, drive `tx`/`ty`/`tz` |
| Random jitter | `noiseCHOP` (low-freq) added to position |
| Timed scene switch | Timer CHOP → switchTOP/CHOP `index` |
