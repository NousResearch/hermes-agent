---
name: neuroskill-bci
description: >
platforms: [linux, macos, windows]
  Connect to a running NeuroSkill instance and incorporate the user's real-time
  cognitive and emotional state (focus, relaxation, mood, cognitive load, drowsiness,
  heart rate, HRV, sleep staging, and 40+ derived EXG scores) into responses.
  Requires a BCI wearable (Muse 2/S or OpenBCI) and the NeuroSkill desktop app
  running locally.
version: 1.0.0
author: Hermes Agent + Nous Research
license: MIT
metadata:
  hermes:
    tags: [BCI, neurofeedback, health, focus, EEG, cognitive-state, biometrics, neuroskill]
    category: health
    related_skills: []
---

# NeuroSkill BCI Integration

Connect Hermes to a running [NeuroSkill](https://neuroskill.com/) instance to read
real-time brain and body metrics from a BCI wearable. Use this to give
cognitively-aware responses, suggest interventions, and track mental performance
over time.

> **⚠️ Research Use Only** — NeuroSkill is an open-source research tool. It is
> NOT a medical device and has NOT been cleared by the FDA, CE, or any regulatory
> body. Never use these metrics for clinical diagnosis or treatment.

See `references/metrics.md` for the full metric reference, `references/protocols.md`
for intervention protocols, and `references/api.md` for the WebSocket/HTTP API.

---

## Prerequisites

- **Node.js 20+** installed (`node --version`)
- **NeuroSkill desktop app** running with a connected BCI device
- **BCI hardware**: Muse 2, Muse S, or OpenBCI (4-channel EEG + PPG + IMU via BLE)
- `npx neuroskill status` returns data without errors

### Verify Setup
```bash
node --version                    # Must be 20+
npx neuroskill status             # Full system snapshot
npx neuroskill status --json      # Machine-parseable JSON
```

If `npx neuroskill status` returns an error, tell the user:
- Make sure the NeuroSkill desktop app is open
- Ensure the BCI device is powered on and connected via Bluetooth
- Check signal quality — green indicators in NeuroSkill (≥0.7 per electrode)
- If `command not found`, install Node.js 20+

---

## CLI Reference: `npx neuroskill <command>`

All commands support `--json` (raw JSON, pipe-safe) and `--full` (human summary + JSON).

| Command | Description |
|---------|-------------|
| `status` | Full system snapshot: device, scores, bands, ratios, sleep, history |
| `session [N]` | Single session breakdown with first/second half trends (0=most recent) |
| `sessions` | List all recorded sessions across all days |
| `search` | ANN similarity search for neurally similar historical moments |
| `compare` | A/B session comparison with metric deltas and trend analysis |
| `sleep [N]` | Sleep stage classification (Wake/N1/N2/N3/REM) with analysis |
| `label "text"` | Create a timestamped annotation at the current moment |
| `search-labels "query"` | Semantic vector search over past labels |
| `interactive "query"` | Cross-modal 4-layer graph search (text → EXG → labels) |
| `listen` | Real-time event streaming (default 5s, set `--seconds N`) |
| `umap` | 3D UMAP projection of session embeddings |
| `calibrate` | Open calibration window and start a profile |
| `timer` | Launch focus timer (Pomodoro/Deep Work/Short Focus presets) |
| `notify "title" "body"` | Send an OS notification via the NeuroSkill app |
| `raw '{json}'` | Raw JSON passthrough to the server |

### Global Flags
| Flag | Description |
|------|-------------|
| `--json` | Raw JSON output (no ANSI, pipe-safe) |
| `--full` | Human summary + colorized JSON |
| `--port <N>` | Override server port (default: auto-discover, usually 8375) |
| `--ws` | Force WebSocket transport |
| `--http` | Force HTTP transport |
| `--k <N>` | Nearest neighbors count (search, search-labels) |
| `--seconds <N>` | Duration for listen (default: 5) |
| `--trends` | Show per-session metric trends (sessions) |
| `--dot` | Graphviz DOT output (interactive) |

---

## 1. Checking Current State

### Get Live Metrics
```bash
npx neuroskill status --json
```

**Always use `--json`** for reliable parsing. The default output is colorized
human-readable text.

### Key Fields in the Response

The `scores` object contains all live metrics (0–1 scale unless noted):

```jsonc
{
  "scores": {
    "focus": 0.70,           // β / (α + θ) — sustained attention
    "relaxation": 0.40,      // α / (β + θ) — calm wakefulness
    "engagement": 0.60,      // active mental investment
    "meditation": 0.52,      // alpha + stillness + HRV coherence
    "mood": 0.55,            // composite from FAA, TAR, BAR
    "cognitive_load": 0.33,  // frontal θ / temporal α · f(FAA, TBR)
    "drowsiness": 0.10,      // TAR + TBR + falling spectral centroid
    "hr": 68.2,              // heart rate in bpm (from PPG)
    "snr": 14.3,             // signal-to-noise ratio in dB
    "stillness": 0.88,       // 0–1; 1 = perfectly still
    "faa": 0.042,            // Frontal Alpha Asymmetry (+ = approach)
    "tar": 0.56,             // Theta/Alpha Ratio
    "bar": 0.53,             // Beta/Alpha Ratio
    "tbr": 1.06,             // Theta/Beta Ratio (ADHD proxy)
    "apf": 10.1,             // Alpha Peak Frequency in Hz
    "coherence": 0.614,      // inter-hemispheric coherence
    "bands": {
      "rel_delta": 0.28, "rel_theta": 0.18,
      "rel_alpha": 0.32, "rel_beta": 0.17, "rel_gamma": 0.05
    }
  }
}
```

Also includes: `device` (state, battery, firmware), `signal_quality` (per-electrode 0–1),
`session` (duration, epochs), `embeddings`, `labels`, `sleep` summary, and `history`.

### Interpreting the Output

Parse the JSON and translate metrics into natural language. Never report raw
numbers alone — always give them meaning:

**DO:**
> "Your focus is solid right now at 0.70 — that's flow state territory. Heart
> rate is steady at 68 bpm and your FAA is positive, which suggests good
> approach motivation. Great time to tackle something complex."

**DON'T:**
> "Focus: 0.70, Relaxation: 0.40, HR: 68"

Key interpretation thresholds (see `references/metrics.md` for the full guide):
- **Focus > 0.70** → flow state territory, protect it
- **Focus < 0.40** → suggest a break or protocol
- **Drowsiness > 0.60** → fatigue warning, micro-sleep risk
- **Relaxation < 0.30** → stress intervention needed
- **Cognitive Load > 0.70 sustained** → mind dump or break
- **TBR > 1.5** → theta-dominant, reduced executive control
- **FAA < 0** → withdrawal/negative affect — consider FAA rebalancing
- **SNR < 3 dB** → unreliable signal, suggest electrode repositioning

---

## 2. Session Analysis

### Single Session Breakdown
```bash
npx neuroskill session --json         # most recent session
npx neuroskill session 1 --json       # previous session
npx neuroskill session 0 --json | jq '{focus: .metrics.focus, trend: .trends.focus}'
```

Returns full metrics with **first-half vs second-half trends** (`"up"`, `"down"`, `"flat"`).
Use this to describe how a session evolved:

> "Your focus started at 0.64 and climbed to 0.76 by the end — a clear upward trend.
> Cognitive load dropped from 0.38 to 0.28, suggesting the task became more automatic
> as you settled in."

### List All Sessions
```bash
npx neuroskill sessions --json
npx neuroskill sessions --trends      # show per-session metric trends
```

---

## 3. Historical Search

### Neural Similarity Search
```bash
npx neuroskill search --json                    # auto: last session, k=5
npx neuroskill search --k 10 --json             # 10 nearest neighbors
npx neuroskill search --start <UTC> --end <UTC> --json
```

Finds moments in history that are neurally similar using HNSW approximate
nearest-neighbor search over 128-D ZUNA embeddings. Returns distance statistics,
temporal distribution (hour of day), and top matching days.

Use this when the user asks:
- "When was I last in a state like this?"
- "Find my best focus sessions"
- "When do I usually crash in the afternoon?"

### Semantic Label Search
```bash
npx neuroskill search-labels "deep focus" --k 10 --json
npx neuroskill search-labels "stress" --json | jq '[.results[].EXG_metrics.tbr]'
```

Searches label text using vector embeddings (Xenova/bge-small-en-v1.5). Returns
matching labels with their associated EXG metrics at the time of labeling.

### Cross-Modal Graph Search
```bash
npx neuroskill interactive "deep focus" --json
npx neuroskill interactive "deep focus" --dot | dot -Tsvg > graph.svg
```

4-layer graph: query → text labels → EXG points → nearby labels. Use `--k-text`,
`--k-EXG`, `--reach <minutes>` to tune.

---

## 4. Session Comparison
```bash
npx neuroskill compare --json                   # auto: last 2 sessions
npx neuroskill compare --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

Returns metric deltas with absolute change, percentage change, and direction for
~50 metrics. Also includes `insights.improved[]` and `insights.declined[]` arrays,
sleep staging for both sessions, and a UMAP job ID.

Interpret comparisons with context — mention trends, not just deltas:
> "Yesterday you had two strong focus blocks (10am and 2pm). Today you've had one
> starting around 11am that's still going. Your overall engagement is higher today
> but there have been more stress spikes — your stress index jumped 15% and
> FAA dipped negative more often."

```bash
# Sort metrics by improvement percentage
npx neuroskill compare --json | jq '.insights.deltas | to_entries | sort_by(.value.pct) | reverse'
```

---

## 5. Sleep Data
```bash
npx neuroskill sleep --json                     # last 24 hours
npx neuroskill sleep 0 --json                   # most recent sleep session
npx neuroskill sleep --start <UTC> --end <UTC> --json
```

Returns epoch-by-epoch sleep staging (5-second windows) with analysis:
- **Stage codes**: 0=Wake, 1=N1, 2=N2, 3=N3 (deep), 4=REM
- **Analysis**: efficiency_pct, onset_latency_min, rem_latency_min, bout counts
- **Healthy targets**: N3 15–25%, REM 20–25%, efficiency >85%, onset <20 min

```bash
npx neuroskill sleep --json | jq '.summary | {n3: .n3_epochs, rem: .rem_epochs}'
npx neuroskill sleep --json | jq '.analysis.efficiency_pct'
```

Use this when the user mentions sleep, tiredness, or recovery.

---

## 6. Labeling Moments
```bash
npx neuroskill label "breakthrough"
npx neuroskill label "studying algorithms"
npx neuroskill label "post-meditation"
npx neuroskill label --json "focus block start"   # returns label_id
```

Auto-label moments when:
- User reports a breakthrough or insight
- User starts a new task type (e.g., "switching to code review")
- User completes a significant protocol
- User asks you to mark the current moment
- A notable state transition occurs (entering/leaving flow)

Labels are stored in a database and indexed for later retrieval via `search-labels`
and `interactive` commands.

---

## 7. Real-Time Streaming
```bash
npx neuroskill listen --seconds 30 --json
npx neuroskill listen --seconds 5 --json | jq '[.[] | select(.event == "scores")]'
```

Streams live WebSocket events (EXG, PPG, IMU, scores, labels) for the specified
duration. Requires WebSocket connection (not available with `--http`).

Use this for continuous monitoring scenarios or to observe metric changes in real-time
during a protocol.

---

## 8. UMAP Visualization
```bash
npx neuroskill umap --json                      # auto: last 2 sessions
npx neuroskill umap --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

GPU-accelerated 3D UMAP projection of ZUNA embeddings. The `separation_score`
indicates how neurally distinct two sessions are:
- **> 1.5** → Sessions are neurally distinct (different brain states)
- **< 0.5** → Similar brain states across both sessions

---

## 9. Proactive State Awareness

### Session Start Check
At the beginning of a session, optionally run a status check if the user mentions
they're wearing their device or asks about their state:
```bash
npx neuroskill status --json
```

Inject a brief state summary:
> "Quick check-in: focus is building at 0.62, relaxation is good at 0.55, and your
> FAA is positive — approach motivation is engaged. Looks like a solid start."

### When to Proactively Mention State

Mention cognitive state **only** when:
- User explicitly asks ("How am I doing?", "Check my focus")
- User reports difficulty concentrating, stress, or fatigue
- A critical threshold is crossed (drowsiness > 0.70, focus < 0.30 sustained)
- User is about to do something cognitively demanding and asks for readiness

**Do NOT** interrupt flow state to report metrics. If focus > 0.75, protect the
session — silence is the correct response.

---

## 10. Suggesting Protocols

When metrics indicate a need, suggest a protocol from `references/protocols.md`.
Always ask before starting — never interrupt flow state:

> "Your focus has been declining for the past 15 minutes and TBR is climbing past
> 1.5 — signs of theta dominance and mental fatigue. Want me to walk you through
> a Theta-Beta Neurofeedback Anchor? It's a 90-second exercise that uses rhythmic
> counting and breath to suppress theta and lift beta."

Key triggers:
- **Focus < 0.40, TBR > 1.5** → Theta-Beta Neurofeedback Anchor or Box Breathing
- **Relaxation < 0.30, stress_index high** → Cardiac Coherence or 4-7-8 Breathing
- **Cognitive Load > 0.70 sustained** → Cognitive Load Offload (mind dump)
- **Drowsiness > 0.60** → Ultradian Reset or Wake Reset
- **FAA < 0 (negative)** → FAA Rebalancing
- **Flow State (focus > 0.75, engagement > 0.70)** → Do NOT interrupt
- **High stillness + headache_index** → Neck Release Sequence
- **Low RMSSD (< 25ms)** → Vagal Toning

---

## 11. Additional Tools

### Focus Timer
```bash
npx neuroskill timer --json
```
Launches the Focus Timer window with Pomodoro (25/5), Deep Work (50/10), or
Short Focus (15/5) presets.

### Calibration
```bash
npx neuroskill calibrate
npx neuroskill calibrate --profile "Eyes Open"
```
Opens the calibration window. Useful when signal quality is poor or the user
wants to establish a personalized baseline.

### OS Notifications
```bash
npx neuroskill notify "Break Time" "Your focus has been declining for 20 minutes"
```

### Raw JSON Passthrough
```bash
npx neuroskill raw '{"command":"status"}' --json
```
For any server command not yet mapped to a CLI subcommand.

---

## Error Handling

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `npx neuroskill status` hangs | NeuroSkill app not running | Open NeuroSkill desktop app |
| `device.state: "disconnected"` | BCI device not connected | Check Bluetooth, device battery |
| All scores return 0 | Poor electrode contact | Reposition headband, moisten electrodes |
| `signal_quality` values < 0.7 | Loose electrodes | Adjust fit, clean electrode contacts |
| SNR < 3 dB | Noisy signal | Minimize head movement, check environment |
| `command not found: npx` | Node.js not installed | Install Node.js 20+ |

---

## Example Interactions

**"How am I doing right now?"**
```bash
npx neuroskill status --json
```
→ Interpret scores naturally, mentioning focus, relaxation, mood, and any notable
  ratios (FAA, TBR). Suggest an action only if metrics indicate a need.

**"I can't concentrate"**
```bash
npx neuroskill status --json
```
→ Check if metrics confirm it (high theta, low beta, rising TBR, high drowsiness).
→ If confirmed, suggest an appropriate protocol from `references/protocols.md`.
→ If metrics look fine, the issue may be motivational rather than neurological.

**"Compare my focus today vs yesterday"**
```bash
npx neuroskill compare --json
```
→ Interpret trends, not just numbers. Mention what improved, what declined, and
  possible causes.

**"When was I last in a flow state?"**
```bash
npx neuroskill search-labels "flow" --json
npx neuroskill search --json
```
→ Report timestamps, associated metrics, and what the user was doing (from labels).

**"How did I sleep?"**
```bash
npx neuroskill sleep --json
```
→ Report sleep architecture (N3%, REM%, efficiency), compare to healthy targets,
  and note any issues (high wake epochs, low REM).

**"Mark this moment — I just had a breakthrough"**
```bash
npx neuroskill label "breakthrough"
```
→ Confirm label saved. Optionally note the current metrics to remember the state.

---

## References

- [NeuroSkill Paper — arXiv:2603.03212](https://arxiv.org/abs/2603.03212) (Kosmyna & Hauptmann, MIT Media Lab)
- [NeuroSkill Desktop App](https://github.com/NeuroSkill-com/skill) (GPLv3)
- [NeuroLoop CLI Companion](https://github.com/NeuroSkill-com/neuroloop) (GPLv3)
- [MIT Media Lab Project](https://www.media.mit.edu/projects/neuroskill/overview/)
