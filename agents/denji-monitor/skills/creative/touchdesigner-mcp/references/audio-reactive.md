# Audio-Reactive Reference

Patterns for driving visuals from audio — spectrum analysis, beat detection, envelope following.

## Audio Input

```python
# Live input from audio interface
audio_in = root.create(audiodeviceinCHOP, 'audio_in')
audio_in.par.rate = 44100

# OR: from audio file (for testing)
audio_file = root.create(audiofileinCHOP, 'audio_in')
audio_file.par.file = '/path/to/track.wav'
audio_file.par.play = True
audio_file.par.repeat = 'on'       # NOT par.loop
audio_file.par.playmode = 'locked'
```

---

## Audio Band Extraction (Verified TD 2025.32460)

Use `audiofilterCHOP` for band separation (NOT `selectCHOP` by channel index):

```python
# Audio input
af = root.create(audiofileinCHOP, 'audio_in')
af.par.file = path
af.par.play = True
af.par.repeat = 'on'
af.par.playmode = 'locked'

# Low band: lowpass @ 250Hz
flt_low = root.create(audiofilterCHOP, 'flt_low')
flt_low.par.filter = 'lowpass'
flt_low.par.cutofffrequency = 250
flt_low.par.rolloff = 2
flt_low.inputConnectors[0].connect(af)

# Mid band: highpass@250 → lowpass@4000
flt_mid_hp = root.create(audiofilterCHOP, 'flt_mid_hp')
flt_mid_hp.par.filter = 'highpass'
flt_mid_hp.par.cutofffrequency = 250
flt_mid_hp.par.rolloff = 2
flt_mid_hp.inputConnectors[0].connect(af)

flt_mid_lp = root.create(audiofilterCHOP, 'flt_mid_lp')
flt_mid_lp.par.filter = 'lowpass'
flt_mid_lp.par.cutofffrequency = 4000
flt_mid_lp.par.rolloff = 2
flt_mid_lp.inputConnectors[0].connect(flt_mid_hp)

# High band: highpass @ 4000Hz
flt_high = root.create(audiofilterCHOP, 'flt_high')
flt_high.par.filter = 'highpass'
flt_high.par.cutofffrequency = 4000
flt_high.par.rolloff = 2
flt_high.inputConnectors[0].connect(af)

# Per-band: RMS → lag → gain → clamp
for name, filt in [('low', flt_low), ('mid', flt_mid_lp), ('high', flt_high)]:
    rms = root.create(analyzeCHOP, f'rms_{name}')
    rms.par.function = 'rmspower'  # NOT 'rms'
    rms.inputConnectors[0].connect(filt)

    lag = root.create(lagCHOP, f'lag_{name}')
    lag.par.lag1 = 0.05   # attack (NOT par.lagin)
    lag.par.lag2 = 0.25   # release (NOT par.lagout)
    lag.inputConnectors[0].connect(rms)

    math = root.create(mathCHOP, f'scale_{name}')
    math.par.gain = 8.0
    math.inputConnectors[0].connect(lag)

    # mathCHOP has NO par.clamp — use limitCHOP
    lim = root.create(limitCHOP, f'clamp_{name}')
    lim.par.type = 'clamp'
    lim.par.min = 0.0
    lim.par.max = 1.0
    lim.inputConnectors[0].connect(math)

    null = root.create(nullCHOP, f'out_{name}')
    null.inputConnectors[0].connect(lim)
    null.viewer = True
```

**Key TD 2025 corrections:**
- `analyzeCHOP.par.function = 'rmspower'` NOT `'rms'`
- `lagCHOP.par.lag1` / `par.lag2` NOT `par.lagin` / `par.lagout`
- `mathCHOP` has NO `par.clamp` — use separate `limitCHOP`

---

## Beat / Onset Detection

### Kick Detection (slope → trigger)

```python
slope = root.create(slopeCHOP, 'kick_slope')
slope.inputConnectors[0].connect(op('out_low'))

trig = root.create(triggerCHOP, 'kick_trig')
trig.par.threshold = 0.12
trig.par.attack = 0.005    # NOT par.attacktime
trig.par.decay = 0.15       # NOT par.decaytime
trig.par.triggeron = 'increase'
trig.inputConnectors[0].connect(slope)

kick_out = root.create(nullCHOP, 'out_kick')
kick_out.inputConnectors[0].connect(trig)
```

---

## Passing Audio to GLSL

```python
glsl.par.vec0name = 'uLow'
glsl.par.vec0valuex.expr = "op('out_low')['chan1']"
glsl.par.vec0valuex.mode = ParMode.EXPRESSION

glsl.par.vec1name = 'uKick'
glsl.par.vec1valuex.expr = "op('out_kick')['chan1']"
glsl.par.vec1valuex.mode = ParMode.EXPRESSION
```

```glsl
uniform float uLow;
uniform float uKick;
float scale = 1.0 + uKick * 0.4 + uLow * 0.2;
```

---

## Standard Audio Bus Pattern

Recommended structure:

```
audiodeviceinCHOP (audio_in)
        ↓
  [null_audio_in]
        ├──→ audiofilterCHOP (lowpass@250) → analyzeCHOP → lagCHOP → mathCHOP → limitCHOP → null
        ├──→ audiofilterCHOP (bandpass@250-4k) → analyzeCHOP → lagCHOP → mathCHOP → limitCHOP → null
        ├──→ audiofilterCHOP (highpass@4k) → analyzeCHOP → lagCHOP → mathCHOP → limitCHOP → null
        │
        └──→ slopeCHOP → triggerCHOP (beat_trigger)
```

Keep this entire bus inside a `baseCOMP` (e.g., `audio_bus`) and reference via paths from visual networks.

---

## MIDI Input

```python
midi_in = root.create(midiinCHOP, 'midi_in')
midi_in.par.device = 0  # Check midiinDAT for device index
# Outputs channels named by MIDI note/CC: 'ch1n60', 'ch1c74', etc.

# Map CC to a parameter
op('bloom1').par.threshold.mode = ParMode.EXPRESSION
op('bloom1').par.threshold.expr = "op('midi_in')['ch1c74'][0]"
```

---

## CRITICAL: DO NOT use Lag CHOP for spectrum smoothing

Lag CHOP in timeslice mode expands 256-sample spectrum to 1600-2400 samples, averaging all values to near-zero (~1e-06). The shader receives no usable data. Use `mathCHOP(gain=8)` directly, or smooth in GLSL via temporal lerp with a feedback texture.

Verified:
- Without Lag CHOP: bass bins = 5.0-5.4 (strong, usable)
- With Lag CHOP: ALL bins = 0.000001 (dead)
