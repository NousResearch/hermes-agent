# NeuroSkill Metric Definitions & Interpretation Guide

## EEG Band Powers
| Band | Hz Range | High Means | Low Means |
|------|----------|------------|-----------|
| Delta (δ) | 0.5–4 Hz | Deep sleep, unconscious processing | Awake, alert |
| Theta (θ) | 4–8 Hz | Mental fatigue, drowsiness, creative daydreaming | Alert, focused |
| Alpha (α) | 8–13 Hz | Relaxed alertness, calm focus, eyes-closed rest | Active thinking, anxiety |
| Beta (β) | 13–30 Hz | Active concentration, problem-solving, alertness | Relaxed, unfocused |
| Gamma (γ) | 30–100 Hz | High-level cognition, binding, peak performance | Baseline |

## Derived Cognitive Metrics (0–100 scale unless noted)

### Focus
- **High (70–100):** Deep concentration, flow state, task absorption
- **Medium (40–69):** Moderate attention, some mind-wandering
- **Low (0–39):** Distracted, fatigued, difficulty concentrating
- **Key ratio:** Beta/Theta — higher = more focused

### Relaxation
- **High (70–100):** Calm, stress-free, parasympathetic dominant
- **Medium (40–69):** Mild tension present
- **Low (0–39):** Stressed, anxious, sympathetic dominant
- **Key ratio:** Alpha/Beta — higher = more relaxed

### Engagement
- **High (70–100):** Mentally invested, motivated, active processing
- **Medium (40–69):** Passive participation
- **Low (0–39):** Bored, disengaged, autopilot mode
- **Key ratio:** Beta/(Alpha+Theta)

### Cognitive Load
- **High (70–100):** Working memory near capacity, complex processing
- **Medium (40–69):** Moderate mental effort
- **Low (0–39):** Task is easy or automatic
- **Interpretation:** High load + high focus = productive struggle. High load + low focus = overwhelmed.

### Drowsiness
- **High (70–100):** Sleep pressure building, microsleep risk
- **Medium (40–69):** Mild fatigue
- **Low (0–39):** Alert
- **Key ratio:** Theta+Alpha/Beta — rising drowsiness means this ratio climbs

## Cardiac Metrics
| Metric | Normal Range | Interpretation |
|--------|-------------|----------------|
| Heart Rate (HR) | 60–100 bpm | Elevated = stress/exertion; Low = calm/fitness |
| HRV (RMSSD) | 20–100 ms | Higher = better stress resilience, recovery |
| HRV Trend | Rising/Falling | Rising = recovering; Falling = accumulating stress |

## Sleep Staging
| Stage | EEG Signature | Function |
|-------|--------------|----------|
| Wake | Beta dominant | Conscious awareness |
| N1 | Alpha→Theta transition | Light sleep onset |
| N2 | Sleep spindles, K-complexes | Memory consolidation |
| N3 | Delta dominant | Deep restorative sleep |
| REM | Mixed, low amplitude | Emotional processing, dreaming |

## Composite State Patterns
| Pattern | Metrics | Interpretation |
|---------|---------|----------------|
| Flow State | Focus >75, Cognitive Load 50–70, HR steady | Optimal performance zone |
| Mental Fatigue | Focus <40, Drowsiness >60, Theta elevated | Rest or break needed |
| Anxiety | Relaxation <30, HR elevated, Beta >Alpha | Calming intervention helpful |
| Peak Alert | Focus >80, Engagement >70, Drowsiness <20 | Best time for hard tasks |
| Recovery | Relaxation >70, HRV rising, Alpha dominant | Integration, light tasks only |
