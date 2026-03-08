---
name: neuroskill-bci
description: Connect to a running NeuroSkill instance and incorporate the user's real-time cognitive and emotional state (focus, relaxation, stress, cognitive load, drowsiness, heart rate, HRV, sleep staging) into responses. Requires a BCI wearable (Muse, OpenBCI, AttentivU) and NeuroSkill desktop app running locally.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [BCI, Neurofeedback, Health, Focus, EEG, Cognitive-State, Biometrics]
    related_skills: []
---

# NeuroSkill BCI Integration

Connect Hermes to a running NeuroSkill instance to read real-time brain and body metrics from a BCI wearable. Use this to give cognitively-aware responses, suggest interventions, and track mental performance over time.

See `references/metrics.md` for metric definitions and `references/protocols.md` for intervention protocols.

## Prerequisites

- Node.js 20+ installed
- NeuroSkill desktop app running with a connected BCI device
- `npx neuroskill status` returns data without errors

### Verify Setup
```bash
# Check Node.js version
node --version  # Must be 20+

# Check NeuroSkill is running and device is connected
npx neuroskill status
```

If `npx neuroskill status` returns an error, tell the user:
- Make sure the NeuroSkill desktop app is open
- Ensure BCI device is powered on and connected via Bluetooth
- Check that electrode contact is good (green indicators in NeuroSkill)

---

## 1. Checking Current State

### Get Live Metrics
```bash
npx neuroskill status
```

**Example output:**
```json
{
  "device": "Muse 2",
  "connected": true,
  "session_duration": "00:23:14",
  "metrics": {
    "focus": 72,
    "relaxation": 58,
    "engagement": 65,
    "cognitive_load": 48,
    "drowsiness": 18,
    "heart_rate": 68,
    "hrv_rmssd": 42,
    "sleep_stage": null
  },
  "band_powers": {
    "delta": 0.12,
    "theta": 0.18,
    "alpha": 0.31,
    "beta": 0.28,
    "gamma": 0.11
  }
}
```

### Interpreting the Output

Parse the JSON and translate metrics into natural language. Never report raw numbers alone — always give them meaning:

**DO:**
> "Your focus is solid right now at 72 — you've been in a good flow state. Heart rate is steady at 68 bpm and HRV looks healthy. Good time to tackle something complex."

**DON'T:**
> "Focus: 72, Relaxation: 58, HR: 68"

Use `references/metrics.md` to interpret values. Key thresholds:
- Focus > 70 → flow state territory, protect it
- Focus < 40 → suggest a break or protocol
- Drowsiness > 60 → fatigue warning
- Relaxation < 30 → stress intervention
- Cognitive Load > 80 sustained → mind dump or break

---

## 2. Historical Search

### Find Past Brain States
```bash
npx neuroskill search --label "focus"
npx neuroskill search --label "stress"
npx neuroskill search --label "flow"
npx neuroskill search --query "high beta low theta morning"
```

Use this when the user asks:
- "When was I last in a flow state?"
- "Find my best focus sessions"
- "When do I usually crash in the afternoon?"

---

## 3. Session Comparison
```bash
npx neuroskill compare
npx neuroskill compare --sessions "today,yesterday"
npx neuroskill compare --sessions "this-week,last-week"
```

Use this when the user asks:
- "How does today compare to yesterday?"
- "Is my focus improving over time?"
- "Compare my morning vs afternoon sessions"

Interpret comparisons with context — mention trends, not just deltas:
> "Yesterday you had two strong focus blocks at 10am and 2pm. Today you've had one starting around 11am that's still going. Your overall engagement is higher today but there have been more stress spikes — likely related to whatever was stressful this morning."

---

## 4. Sleep Data
```bash
npx neuroskill sleep
npx neuroskill sleep --date "2026-03-07"
```

Returns hypnogram with Wake/N1/N2/N3/REM staging and sleep quality score.

Use this when the user mentions sleep, tiredness, or asks about recovery. See `references/metrics.md` for sleep stage interpretations.

---

## 5. Session List
```bash
npx neuroskill sessions
npx neuroskill sessions --limit 10
```

Use this to find available sessions for comparison or trend analysis.

---

## 6. Labeling Moments
```bash
npx neuroskill label "breakthrough"
npx neuroskill label "studying algorithms"
npx neuroskill label "post-meditation"
```

Auto-label moments when:
- User reports a breakthrough or insight
- User starts a new task type
- User completes a significant protocol
- User asks you to mark the current moment

---

## 7. Proactive State Awareness

### Session Start Check
At the beginning of a session, optionally run a status check if the user mentions they're wearing their device or asks about their state:
```bash
npx neuroskill status
```

Inject a brief state summary into context:
> "Quick check-in: your focus is building (62), relaxation is good. Looks like a solid start."

### When to Proactively Mention State

Mention cognitive state **only** when:
- User explicitly asks ("How am I doing?", "Check my focus")
- User reports difficulty concentrating, stress, or fatigue
- A critical threshold is crossed (Drowsiness > 70, Focus < 30 for > 10 min)
- User is about to do something cognitively demanding and asks for readiness

**Do NOT** interrupt flow state to report metrics. If focus > 75, protect the session.

---

## 8. Suggesting Protocols

When metrics indicate a need, suggest a protocol from `references/protocols.md`. Always ask before starting:

> "Your focus has been declining for the past 15 minutes and theta is rising — signs of mental fatigue. Want me to guide you through a 2-minute box breathing reset? It usually helps."

Key triggers:
- Focus < 40 → Box Breathing or Pomodoro Break
- Relaxation < 30 → 4-7-8 Breathing or Progressive Relaxation
- Cognitive Load > 80 sustained → Mind Dump
- Drowsiness > 70 → NASA Nap Protocol
- Flow State (Focus > 75, Engagement > 70) → Do NOT interrupt

---

## 9. WebSocket API (Advanced)

NeuroSkill also exposes a local WebSocket on `localhost` (discoverable via `lsof -i -n -P | grep neuroskill`).
```bash
# Discover port
NEURO_PORT=$(lsof -i -n -P | grep neuroskill | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
echo "NeuroSkill WebSocket on port: $NEURO_PORT"
```

For real-time streaming, the WebSocket accepts the same commands as the CLI (`status`, `search`, `compare`, `sleep`, `sessions`, `label`). Use CLI for one-off queries; WebSocket for continuous monitoring.

---

## Error Handling

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `npx neuroskill status` hangs | NeuroSkill app not running | Open NeuroSkill desktop app |
| `device: null` or `connected: false` | BCI device not connected | Check Bluetooth, device battery, electrode contact |
| All metrics return 0 | Poor electrode contact | Reposition headband, ensure electrodes are moist |
| `command not found: npx` | Node.js not installed | Install Node.js 20+ |
| Metrics seem wrong | Movement artifacts | Minimize head movement during readings |

---

## Example Interactions

**"How am I doing right now?"**
```bash
npx neuroskill status
```
→ Interpret and respond naturally based on metrics.

**"I can't concentrate"**
```bash
npx neuroskill status
```
→ Check if metrics confirm it (high theta, low beta, high drowsiness).
→ If confirmed, suggest appropriate protocol from `references/protocols.md`.

**"Compare my focus today vs yesterday"**
```bash
npx neuroskill compare --sessions "today,yesterday"
```
→ Interpret trends, not just numbers.

**"When was I last in a flow state?"**
```bash
npx neuroskill search --label "flow"
```
→ Report session timestamps and what was happening.

**"Mark this moment — I just had a breakthrough"**
```bash
npx neuroskill label "breakthrough"
```
→ Confirm label saved.
