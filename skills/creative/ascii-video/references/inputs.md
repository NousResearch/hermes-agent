# Input Sources

## Audio Analysis

### Loading

```python
tmp = tempfile.mktemp(suffix=".wav")
subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "22050",
                "-sample_fmt", "s16", tmp], capture_output=True, check=True)
with wave.open(tmp) as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
```

### Per-Frame FFT

```python
hop = sr // fps          # samples per frame
win = hop * 2            # analysis window (2x hop for overlap)
window = np.hanning(win)
freqs = rfftfreq(win, 1.0 / sr)

bands = {
    "sub":   (freqs >= 20)  & (freqs < 80),
    "bass":  (freqs >= 80)  & (freqs < 250),
    "lomid": (freqs >= 250) & (freqs < 500),
    "mid":   (freqs >= 500) & (freqs < 2000),
    "himid": (freqs >= 2000)& (freqs < 6000),
    "hi":    (freqs >= 6000),
}
```

For each frame: extract chunk, apply window, FFT, compute band energies.

### Feature Set

| Feature | Formula | Controls |
|---------|---------|----------|
| `rms` | `sqrt(mean(chunk²))` | Overall loudness/energy |
| `sub`..`hi` | `sqrt(mean(band_magnitudes²))` | Per-band energy |
| `centroid` | `sum(freq*mag) / sum(mag)` | Brightness/timbre |
| `flatness` | `geomean(mag) / mean(mag)` | Noise vs tone |
| `flux` | `sum(max(0, mag - prev_mag))` | Transient strength |
| `sub_r`..`hi_r` | `band / sum(all_bands)` | Spectral shape (volume-independent) |
| `cent_d` | `abs(gradient(centroid))` | Timbral change rate |
| `beat` | Flux peak detection | Binary beat onset |
| `bdecay` | Exponential decay from beats | Smooth beat pulse (0→1→0) |

**Band ratios are critical** — they decouple spectral shape from volume, so a quiet bass section and a loud bass section both read as "bassy" rather than just "loud" vs "quiet".

### Smoothing

EMA prevents visual jitter:

```python
def ema(arr, alpha):
    out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

# Slow-moving features (alpha=0.12): centroid, flatness, band ratios, cent_d
# Fast-moving features (alpha=0.3): rms, flux, raw bands
```

### Beat Detection

```python
flux_smooth = np.convolve(flux, np.ones(5)/5, mode="same")
peaks, _ = signal.find_peaks(flux_smooth, height=0.15, distance=fps//5, prominence=0.05)

beat = np.zeros(n_frames)
bdecay = np.zeros(n_frames, dtype=np.float32)
for p in peaks:
    beat[p] = 1.0
    for d in range(fps // 2):
        if p + d < n_frames:
            bdecay[p + d] = max(bdecay[p + d], math.exp(-d * 2.5 / (fps // 2)))
```

`bdecay` gives smooth 0→1→0 pulse per beat, decaying over ~0.5s. Use for flash/glitch/mirror triggers.

### Normalization

After computing all frames, normalize each feature to 0-1:

```python
for k in features:
    a = features[k]
    lo, hi = a.min(), a.max()
    features[k] = (a - lo) / (hi - lo + 1e-10)
```

## Video Sampling

### Frame Extraction

```python
# Method 1: ffmpeg pipe (memory efficient)
cmd = ["ffmpeg", "-i", input_video, "-f", "rawvideo", "-pix_fmt", "rgb24",
       "-s", f"{target_w}x{target_h}", "-r", str(fps), "-"]
pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
frame_size = target_w * target_h * 3
for fi in range(n_frames):
    raw = pipe.stdout.read(frame_size)
    if len(raw) < frame_size: break
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(target_h, target_w, 3)
    # process frame...

# Method 2: OpenCV (if available)
cap = cv2.VideoCapture(input_video)
```

### Luminance-to-Character Mapping

Convert video pixels to ASCII characters based on brightness:

```python
def frame_to_ascii(frame_rgb, grid, pal=PAL_DEFAULT):
    """Convert video frame to character + color arrays."""
    rows, cols = grid.rows, grid.cols
    # Resize frame to grid dimensions
    small = np.array(Image.fromarray(frame_rgb).resize((cols, rows), Image.LANCZOS))
    # Luminance
    lum = (0.299 * small[:,:,0] + 0.587 * small[:,:,1] + 0.114 * small[:,:,2]) / 255.0
    # Map to chars
    chars = val2char(lum, lum > 0.02, pal)
    # Colors: use source pixel colors, scaled by luminance for visibility
    colors = np.clip(small * np.clip(lum[:,:,None] * 1.5 + 0.3, 0.3, 1), 0, 255).astype(np.uint8)
    return chars, colors
```

### Edge-Weighted Character Mapping

Use edge detection for more detail in contour regions:

```python
def frame_to_ascii_edges(frame_rgb, grid, pal=PAL_DEFAULT, edge_pal=PAL_BOX):
    gray = np.mean(frame_rgb, axis=2)
    small_gray = resize(gray, (grid.rows, grid.cols))
    lum = small_gray / 255.0

    # Sobel edge detection
    gx = np.abs(small_gray[:, 2:] - small_gray[:, :-2])
    gy = np.abs(small_gray[2:, :] - small_gray[:-2, :])
    edge = np.zeros_like(small_gray)
    edge[:, 1:-1] += gx; edge[1:-1, :] += gy
    edge = np.clip(edge / edge.max(), 0, 1)

    # Edge regions get box drawing chars, flat regions get brightness chars
    is_edge = edge > 0.15
    chars = val2char(lum, lum > 0.02, pal)
    edge_chars = val2char(edge, is_edge, edge_pal)
    chars[is_edge] = edge_chars[is_edge]

    return chars, colors
```

### Motion Detection

Detect pixel changes between frames for motion-reactive effects:

```python
prev_frame = None
def compute_motion(frame):
    global prev_frame
    if prev_frame is None:
        prev_frame = frame.astype(np.float32)
        return np.zeros(frame.shape[:2])
    diff = np.abs(frame.astype(np.float32) - prev_frame).mean(axis=2)
    prev_frame = frame.astype(np.float32) * 0.7 + prev_frame * 0.3  # smoothed
    return np.clip(diff / 30.0, 0, 1)  # normalized motion map
```

Use motion map to drive particle emission, glitch intensity, or character density.

### Video Feature Extraction

Per-frame features analogous to audio features, for driving effects:

```python
def analyze_video_frame(frame_rgb):
    gray = np.mean(frame_rgb, axis=2)
    return {
        "brightness": gray.mean() / 255.0,
        "contrast": gray.std() / 128.0,
        "edge_density": compute_edge_density(gray),
        "motion": compute_motion(frame_rgb).mean(),
        "dominant_hue": compute_dominant_hue(frame_rgb),
        "color_variance": compute_color_variance(frame_rgb),
    }
```

## Image Sequence

### Static Image to ASCII

Same as single video frame conversion. For animated sequences:

```python
import glob
frames = sorted(glob.glob("frames/*.png"))
for fi, path in enumerate(frames):
    img = np.array(Image.open(path).resize((VW, VH)))
    chars, colors = frame_to_ascii(img, grid, pal)
```

### Image as Texture Source

Use an image as a background texture that effects modulate:

```python
def load_texture(path, grid):
    img = np.array(Image.open(path).resize((grid.cols, grid.rows)))
    lum = np.mean(img, axis=2) / 255.0
    return lum, img  # luminance for char mapping, RGB for colors
```

## Text / Lyrics

### SRT Parsing

```python
import re
def parse_srt(path):
    """Returns [(start_sec, end_sec, text), ...]"""
    entries = []
    with open(path) as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            times = lines[1]
            m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", times)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
                end = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
                text = " ".join(lines[2:])
                entries.append((start, end, text))
    return entries
```

### Lyrics Display Modes

- **Typewriter**: characters appear left-to-right over the time window
- **Fade-in**: whole line fades from dark to bright
- **Flash**: appear instantly on beat, fade out
- **Scatter**: characters start at random positions, converge to final position
- **Wave**: text follows a sine wave path

```python
def lyrics_typewriter(ch, co, text, row, col, t, t_start, t_end, color):
    """Reveal characters progressively over time window."""
    progress = np.clip((t - t_start) / (t_end - t_start), 0, 1)
    n_visible = int(len(text) * progress)
    stamp(ch, co, text[:n_visible], row, col, color)
```

## Generative (No Input)

For pure generative ASCII art, the "features" dict is synthesized from time:

```python
def synthetic_features(t, bpm=120):
    """Generate audio-like features from time alone."""
    beat_period = 60.0 / bpm
    beat_phase = (t % beat_period) / beat_period
    return {
        "rms": 0.5 + 0.3 * math.sin(t * 0.5),
        "bass": 0.5 + 0.4 * math.sin(t * 2 * math.pi / beat_period),
        "sub": 0.3 + 0.3 * math.sin(t * 0.8),
        "mid": 0.4 + 0.3 * math.sin(t * 1.3),
        "hi": 0.3 + 0.2 * math.sin(t * 2.1),
        "cent": 0.5 + 0.2 * math.sin(t * 0.3),
        "flat": 0.4,
        "flux": 0.3 + 0.2 * math.sin(t * 3),
        "beat": 1.0 if beat_phase < 0.05 else 0.0,
        "bdecay": max(0, 1.0 - beat_phase * 4),
        # ratios
        "sub_r": 0.2, "bass_r": 0.25, "lomid_r": 0.15,
        "mid_r": 0.2, "himid_r": 0.12, "hi_r": 0.08,
        "cent_d": 0.1,
    }
```

## TTS Integration

For narrated videos (testimonials, quotes, storytelling), generate speech audio per segment and mix with background music.

### ElevenLabs Voice Generation

```python
import requests

def generate_tts(text, voice_id, api_key, output_path, model="eleven_multilingual_v2"):
    """Generate TTS audio via ElevenLabs API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    data = {"text": text, "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    resp = requests.post(url, json=data, headers=headers, timeout=30)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
```

### Voice Assignment

Use multiple voices for variety. Shuffle deterministically so re-runs are consistent:

```python
import random as _rng

def assign_voices(n_quotes, voice_pool, seed=42):
    """Assign a different voice to each quote, cycling if needed."""
    r = _rng.Random(seed)
    shuffled = list(voice_pool)
    r.shuffle(shuffled)
    return [shuffled[i % len(shuffled)] for i in range(n_quotes)]
```

### Pronunciation Control

TTS text should be separate from display text. Common fixes:
- Brand names: spell phonetically ("Nous" -> "Noose", "nginx" -> "engine-x")
- Abbreviations: expand ("API" -> "A P I", "CLI" -> "C L I")
- Technical terms: add phonetic hints

```python
QUOTES = [("Display text here", "Author")]
QUOTES_TTS = ["TTS text with phonetic spelling here"]
# Keep both arrays in sync -- same indices
```

### Audio Pipeline

1. Generate individual TTS clips (MP3/WAV per quote)
2. Get duration of each clip
3. Calculate timing: speech start/end per quote with gaps
4. Concatenate into single TTS track with silence padding
5. Mix with background music

```python
def build_tts_track(tts_clips, target_duration, gap_seconds=2.0):
    """Concatenate TTS clips with gaps, pad to target duration."""
    # Get durations
    durations = []
    for clip in tts_clips:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", clip],
            capture_output=True, text=True)
        durations.append(float(result.stdout.strip()))
    
    # Calculate timing
    total_speech = sum(durations)
    total_gaps = target_duration - total_speech
    gap = max(0.5, total_gaps / (len(tts_clips) + 1))
    
    timing = []  # (start, end, quote_index)
    t = gap  # start after initial gap
    for i, dur in enumerate(durations):
        timing.append((t, t + dur, i))
        t += dur + gap
    
    # Concatenate with ffmpeg
    # ... silence padding + concat filter
    return timing
```

### Audio Mixing

Mix TTS (center) with background music (wide stereo, low volume):

```python
def mix_audio(tts_path, bgm_path, output_path, bgm_volume=0.15):
    """Mix TTS centered with BGM panned wide stereo."""
    cmd = [
        "ffmpeg", "-y",
        "-i", tts_path,   # mono TTS
        "-i", bgm_path,   # stereo BGM
        "-filter_complex",
        f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=mono,"
        f"pan=stereo|c0=c0|c1=c0[tts];"  # TTS center
        f"[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,"
        f"volume={bgm_volume},"
        f"extrastereo=2.5[bgm];"  # BGM wide stereo
        f"[tts][bgm]amix=inputs=2:duration=longest[out]",
        "-map", "[out]", "-c:a", "pcm_s16le", output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
```

### Feature Analysis on Mixed Audio

Run the standard audio analysis (FFT, beat detection) on the final mixed track so visual effects react to both TTS and music:

```python
# Analyze mixed_final.wav (not individual tracks)
features = analyze_audio("mixed_final.wav", fps=24)
```

This means visuals will pulse with both the music beats and the speech energy -- creating natural synchronization.
