# NeuroSkill Metric Definitions & Interpretation Guide

> **⚠️ Research Use Only:** All metrics are experimental and derived from
> consumer-grade hardware (Muse 2/S). They are not FDA/CE-cleared and must not
> be used for medical diagnosis or treatment.

---

## Hardware & Signal Acquisition

NeuroSkill is validated for **Muse 2** and **Muse S** headbands (with OpenBCI
support in the desktop app), streaming at **256 Hz** (EEG) and **64 Hz** (PPG).

### Electrode Positions (International 10-20 System)
| Channel | Electrode | Position | Primary Signals |
|---------|-----------|----------|-----------------|
| CH1 | TP9 | Left Mastoid | Auditory cortex, verbal memory, jaw-clench artifact |
| CH2 | AF7 | Left Prefrontal | Executive function, approach motivation, eye blinks |
| CH3 | AF8 | Right Prefrontal | Emotional regulation, vigilance, eye blinks |
| CH4 | TP10 | Right Mastoid | Prosody, spatial hearing, non-verbal cognition |

### Preprocessing Pipeline
1. **Filtering**: High-pass (0.5 Hz), Low-pass (50/60 Hz), Notch filter
2. **Spectral Analysis**: Hann-windowed FFT (512-sample window), Welch periodogram
3. **GPU acceleration**: ~125ms latency via `gpu_fft`

---

## EEG Frequency Bands

Relative power values (sum ≈ 1.0 across all bands):

| Band | Range (Hz) | High Means | Low Means |
|------|-----------|------------|-----------|
| **Delta (δ)** | 1–4 | Deep sleep (N3), high-amplitude artifacts | Awake, alert |
| **Theta (θ)** | 4–8 | Drowsiness, REM onset, creative ideation, cognitive load | Alert, focused |
| **Alpha (α)** | 8–13 | Relaxed wakefulness, "alpha blocking" during effort | Active thinking, anxiety |
| **Beta (β)** | 13–30 | Active concentration, problem-solving, alertness | Relaxed, unfocused |
| **Gamma (γ)** | 30–50 | Higher-order processing, perceptual binding, memory | Baseline |

### JSON Field Names
```json
"bands": {
  "rel_delta": 0.28, "rel_theta": 0.18, "rel_alpha": 0.32,
  "rel_beta": 0.17, "rel_gamma": 0.05
}
```

---

## Core Composite Scores (0–1 Scale)

### Focus
- **Formula**: σ(β / (α + θ)) — beta dominance over slow waves, sigmoid-mapped
- **> 0.70**: Deep concentration, flow state, task absorption
- **0.40–0.69**: Moderate attention, some mind-wandering
- **< 0.40**: Distracted, fatigued, difficulty concentrating

### Relaxation
- **Formula**: σ(α / (β + θ)) — alpha dominance, sigmoid-mapped
- **> 0.70**: Calm, stress-free, parasympathetic dominant
- **0.40–0.69**: Mild tension present
- **< 0.30**: Stressed, anxious, sympathetic dominant

### Engagement
- **0–1 scale**: Active mental investment and motivation
- **> 0.70**: Mentally invested, motivated, active processing
- **0.40–0.69**: Passive participation
- **< 0.30**: Bored, disengaged, autopilot mode

### Meditation
- **Composite**: Combines alpha elevation, physical stillness (IMU), and HRV coherence
- **> 0.70**: Deep meditative state
- **< 0.30**: Active, non-meditative

### Mood
- **Composite**: Derived from FAA, TAR, and BAR
- **> 0.60**: Positive affect, approach motivation
- **< 0.40**: Low mood, withdrawal tendency

### Cognitive Load
- **Formula**: (P_θ_frontal / P_α_temporal) · f(FAA, TBR) — working memory usage
- **> 0.70**: Working memory near capacity, complex processing
- **0.40–0.69**: Moderate mental effort
- **< 0.40**: Task is easy or automatic
- **Interpretation**: High load + high focus = productive struggle. High load + low focus = overwhelmed.

### Drowsiness
- **Composite**: Weighted TAR + TBR + falling Spectral Centroid
- **> 0.60**: Sleep pressure building, micro-sleep risk
- **0.30–0.59**: Mild fatigue
- **< 0.30**: Alert

---

## EEG Ratios & Spectral Indices

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **FAA** | ln(P_α_AF8) − ln(P_α_AF7) | Frontal Alpha Asymmetry. Positive = approach/positive affect. Negative = withdrawal/depression. |
| **TAR** | P_θ / P_α | Theta/Alpha Ratio. > 1.5 = drowsiness or mind-wandering. |
| **BAR** | P_β / P_α | Beta/Alpha Ratio. > 1.5 = alert, engaged cognition. Can also indicate anxiety. |
| **TBR** | P_θ / P_β | Theta/Beta Ratio. ADHD biomarker. Healthy ≈ 1.0, elevated > 1.5, clinical > 3.0. |
| **APF** | argmax_f PSD(f) in [7.5, 12.5] Hz | Alpha Peak Frequency. Typical 8–12 Hz. Higher = faster cognitive processing. Slows with age/fatigue. |
| **SNR** | 10 · log₁₀(P_signal / P_noise) | Signal-to-Noise Ratio. > 10 dB = clean, 3–10 dB = usable, < 3 dB = unreliable. |
| **Coherence** | Inter-hemispheric coherence (0–1) | Cortical connectivity between hemispheres. |
| **Mu Suppression** | Motor cortex suppression index | Low values during movement or motor imagery. |

---

## Complexity & Nonlinear Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| **Permutation Entropy (PE)** | Temporal complexity. Near 1 = maximally irregular. | Consciousness marker |
| **Higuchi Fractal Dimension (HFD)** | Waveform self-similarity. | Waking: 1.3–1.8; higher = complex |
| **DFA Exponent** | Long-range correlations. | Healthy: 0.6–0.9 |
| **PSE** | Power Spectral Entropy. Near 1.0 = white noise. | Lower = organized brain state |
| **PAC θ-γ** | Phase-Amplitude Coupling, theta-gamma. | Working memory mechanism |
| **BPS** | Band-Power Slope (1/f spectral exponent). | Steeper = inhibition-dominated |

---

## Consciousness Metrics

Derived from the nonlinear metrics above:

| Metric | Scale | Interpretation |
|--------|-------|----------------|
| **LZC** | 0–100 | Lempel-Ziv Complexity proxy (PE + HFD). > 60 = wakefulness. |
| **Wakefulness** | 0–100 | Inverse drowsiness composite. |
| **Integration** | 0–100 | Cortical integration (Coherence × PAC × Spectral Entropy). |

Status thresholds: ≥ 50 Green, 25–50 Yellow, < 25 Red.

---

## Cardiac & Autonomic Metrics (from PPG)

| Metric | Description | Normal / Green Range |
|--------|-------------|---------------------|
| **HR** | Heart rate (bpm) | 55–90 (green), 45–110 (yellow), else red |
| **RMSSD** | Primary vagal tone marker (ms) | > 50 ms healthy, < 20 ms stress |
| **SDNN** | HRV time-domain variability (ms) | Higher = better |
| **pNN50** | Parasympathetic indicator (%) | Higher = more parasympathetic activity |
| **LF/HF Ratio** | Sympatho-vagal balance | > 2.0 = stress, < 0.5 = relaxation |
| **Stress Index** | Baevsky SI: AMo / (2 × MxDMn × Mo) | 0–100 composite. > 200 raw = strong stress |
| **SpO₂ Estimate** | Blood oxygen saturation (uncalibrated) | 95–100% normal (research only) |
| **Respiratory Rate** | Breaths per minute | 12–20 normal |

---

## Motion & Artifact Detection

| Metric | Description |
|--------|-------------|
| **Stillness** | 0–1 (1 = perfectly still). From IMU accelerometer/gyroscope. |
| **Blink Count** | Eye blinks detected (large spikes in AF7/AF8). Normal: 15–20/min. |
| **Jaw Clench Count** | High-frequency EMG bursts (> 30 Hz) at TP9/TP10. |
| **Nod Count** | Head nods detected via IMU. |
| **Shake Count** | Head shakes detected via IMU. |
| **Head Pitch/Roll** | Head orientation from IMU. |

---

## Signal Quality (Per Electrode)

| Electrode | Range | Interpretation |
|-----------|-------|----------------|
| **TP9** | 0–1 | ≥ 0.9 = good, ≥ 0.7 = acceptable, < 0.7 = poor |
| **AF7** | 0–1 | Same thresholds |
| **AF8** | 0–1 | Same thresholds |
| **TP10** | 0–1 | Same thresholds |

If any electrode is below 0.7, recommend the user adjust the headband fit or
moisten the electrode contacts.

---

## Sleep Staging

Based on 5-second epochs using relative band-power ratios and AASM heuristics:

| Stage | Code | EEG Signature | Function |
|-------|------|---------------|----------|
| Wake | 0 | Alpha-dominant, BAR > 0.8 | Conscious awareness |
| N1 | 1 | Alpha → Theta transition | Light sleep onset |
| N2 | 2 | Sleep spindles, K-complexes | Memory consolidation |
| N3 (Deep) | 3 | Delta > 20% of epoch, DTR > 2 | Deep restorative sleep |
| REM | 4 | Active EEG, high Theta, low Delta | Emotional processing, dreaming |

### Healthy Adult Targets (~8h Sleep)
- **N3 (Deep)**: 15–25% of total sleep
- **REM**: 20–25%
- **Sleep Efficiency**: > 85%
- **Sleep Onset Latency**: < 20 min

---

## Composite State Patterns

| Pattern | Key Metrics | Interpretation |
|---------|-------------|----------------|
| **Flow State** | Focus > 0.75, Engagement > 0.70, Cognitive Load 0.50–0.70, HR steady | Optimal performance zone — protect it |
| **Mental Fatigue** | Focus < 0.40, Drowsiness > 0.60, TBR > 1.5, Theta elevated | Rest or break needed |
| **Anxiety** | Relaxation < 0.30, HR elevated, high Beta, high BAR, stress_index high | Calming intervention helpful |
| **Peak Alert** | Focus > 0.80, Engagement > 0.70, Drowsiness < 0.20 | Best time for hard tasks |
| **Recovery** | Relaxation > 0.70, HRV (RMSSD) rising, Alpha dominant | Integration, light tasks only |
| **Creative Mode** | High Theta, high Alpha, low Beta, moderate focus | Ideation — don't force structure |
| **Withdrawal** | FAA < 0, low Mood, low Engagement | Approach motivation needed |

---

## ZUNA Embeddings

NeuroSkill uses the **ZUNA Neural Encoder** to convert 5-second EEG epochs into
**128-dimensional vectors** stored in an HNSW index:
- **Search**: Sub-millisecond approximate nearest-neighbor queries
- **UMAP**: GPU-accelerated 3D projection for visual comparison
- **Storage**: Local SQLite + HNSW index in `~/.skill/YYYYMMDD/`
