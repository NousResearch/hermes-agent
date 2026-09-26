# NeuroSkill WebSocket & HTTP API Reference

NeuroSkill runs a local server (default port **8375**) discoverable via mDNS
(`_skill._tcp`). It exposes both WebSocket and HTTP endpoints.

---

## Server Discovery

```bash
# Auto-discovery (built into the CLI — usually just works)
npx neuroskill status --json

# Manual port discovery
NEURO_PORT=$(lsof -i -n -P | grep neuroskill | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
echo "NeuroSkill on port: $NEURO_PORT"
```

The CLI auto-discovers the port. Use `--port <N>` to override.

---

## HTTP REST Endpoints

### Universal Command Tunnel
```bash
# POST / — accepts any command as JSON
curl -s -X POST http://127.0.0.1:8375/ \
  -H "Content-Type: application/json" \
  -d '{"command":"status"}'
```

### Convenience Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/status` | System status |
| GET | `/v1/sessions` | List sessions |
| POST | `/v1/label` | Create label |
| POST | `/v1/search` | ANN search |
| POST | `/v1/compare` | A/B comparison |
| POST | `/v1/sleep` | Sleep staging |
| POST | `/v1/notify` | OS notification |
| POST | `/v1/say` | Text-to-speech |
| POST | `/v1/calibrate` | Open calibration |
| POST | `/v1/timer` | Open focus timer |
| GET | `/v1/dnd` | Get DND status |
| POST | `/v1/dnd` | Force DND on/off |
| GET | `/v1/calibrations` | List calibration profiles |
| POST | `/v1/calibrations` | Create profile |
| GET | `/v1/calibrations/{id}` | Get profile |
| PATCH | `/v1/calibrations/{id}` | Update profile |
| DELETE | `/v1/calibrations/{id}` | Delete profile |

---

## WebSocket Events (Broadcast)

Connect to `ws://127.0.0.1:8375/` to receive real-time events:

### EXG (Raw EEG Samples)
```json
{"event": "EXG", "electrode": 0, "samples": [12.3, -4.1, ...], "timestamp": 1740412800.512}
```

### PPG (Photoplethysmography)
```json
{"event": "PPG", "channel": 0, "samples": [...], "timestamp": 1740412800.512}
```

### IMU (Inertial Measurement Unit)
```json
{"event": "IMU", "ax": 0.01, "ay": -0.02, "az": 9.81, "gx": 0.1, "gy": -0.05, "gz": 0.02}
```

### Scores (Computed Metrics)
```json
{
  "event": "scores",
  "focus": 0.70, "relaxation": 0.40, "engagement": 0.60,
  "rel_delta": 0.28, "rel_theta": 0.18, "rel_alpha": 0.32,
  "rel_beta": 0.17, "hr": 68.2, "snr": 14.3
}
```

### EXG Bands (Spectral Analysis)
```json
{"event": "EXG-bands", "channels": [...], "faa": 0.12}
```

### Labels
```json
{"event": "label", "label_id": 42, "text": "meditation start", "created_at": 1740413100}
```

### Device Status
```json
{"event": "muse-status", "state": "connected"}
```

---

## JSON Response Formats

### `status`
```jsonc
{
  "command": "status", "ok": true,
  "device": {
    "state": "connected",     // "connected" | "connecting" | "disconnected"
    "name": "Muse-A1B2",
    "battery": 73,
    "firmware": "1.3.4",
    "EXG_samples": 195840,
    "ppg_samples": 30600,
    "imu_samples": 122400
  },
  "session": {
    "start_utc": 1740412800,
    "duration_secs": 1847,
    "n_epochs": 369
  },
  "signal_quality": {
    "tp9": 0.95, "af7": 0.88, "af8": 0.91, "tp10": 0.97
  },
  "scores": {
    "focus": 0.70, "relaxation": 0.40, "engagement": 0.60,
    "meditation": 0.52, "mood": 0.55, "cognitive_load": 0.33,
    "drowsiness": 0.10, "hr": 68.2, "snr": 14.3, "stillness": 0.88,
    "bands": { "rel_delta": 0.28, "rel_theta": 0.18, "rel_alpha": 0.32, "rel_beta": 0.17, "rel_gamma": 0.05 },
    "faa": 0.042, "tar": 0.56, "bar": 0.53, "tbr": 1.06,
    "apf": 10.1, "coherence": 0.614, "mu_suppression": 0.031
  },
  "embeddings": { "today": 342, "total": 14820, "recording_days": 31 },
  "labels": { "total": 58, "recent": [{"id": 42, "text": "meditation start", "created_at": 1740413100}] },
  "sleep": { "total_epochs": 1054, "wake_epochs": 134, "n1_epochs": 89, "n2_epochs": 421, "n3_epochs": 298, "rem_epochs": 112, "epoch_secs": 5 },
  "history": { "total_sessions": 63, "recording_days": 31, "current_streak_days": 7, "total_recording_hours": 94.2, "longest_session_min": 187, "avg_session_min": 89 }
}
```

### `sessions`
```jsonc
{
  "command": "sessions", "ok": true,
  "sessions": [
    { "day": "20260224", "start_utc": 1740412800, "end_utc": 1740415510, "n_epochs": 541 },
    { "day": "20260223", "start_utc": 1740380100, "end_utc": 1740382665, "n_epochs": 513 }
  ]
}
```

### `session` (single session breakdown)
```jsonc
{
  "ok": true,
  "metrics": { "focus": 0.70, "relaxation": 0.40, "n_epochs": 541 /* ... ~50 metrics */ },
  "first":   { "focus": 0.64 /* first-half averages */ },
  "second":  { "focus": 0.76 /* second-half averages */ },
  "trends":  { "focus": "up", "relaxation": "down" /* "up" | "down" | "flat" */ }
}
```

### `compare` (A/B comparison)
```jsonc
{
  "command": "compare", "ok": true,
  "insights": {
    "deltas": {
      "focus": { "a": 0.62, "b": 0.71, "abs": 0.09, "pct": 14.5, "direction": "up" },
      "relaxation": { "a": 0.45, "b": 0.38, "abs": -0.07, "pct": -15.6, "direction": "down" }
    },
    "improved": ["focus", "engagement"],
    "declined": ["relaxation"]
  },
  "sleep_a": { /* sleep summary for session A */ },
  "sleep_b": { /* sleep summary for session B */ },
  "umap": { "job_id": "abc123" }
}
```

### `search` (ANN similarity)
```jsonc
{
  "command": "search", "ok": true,
  "result": {
    "results": [{
      "neighbors": [{ "distance": 0.12, "metadata": {"device": "Muse-A1B2", "date": "20260223"} }]
    }],
    "analysis": {
      "distance_stats": { "mean": 0.15, "min": 0.08, "max": 0.42 },
      "temporal_distribution": { /* hour-of-day distribution */ },
      "top_days": [["20260223", 5], ["20260222", 3]]
    }
  }
}
```

### `sleep` (sleep staging)
```jsonc
{
  "command": "sleep", "ok": true,
  "summary": { "total_epochs": 1054, "wake_epochs": 134, "n1_epochs": 89, "n2_epochs": 421, "n3_epochs": 298, "rem_epochs": 112, "epoch_secs": 5 },
  "analysis": { "efficiency_pct": 87.3, "onset_latency_min": 12.5, "rem_latency_min": 65.0, "bouts": { /* wake/n3/rem bout counts and durations */ } },
  "epochs": [{ "utc": 1740380100, "stage": 0, "rel_delta": 0.15, "rel_theta": 0.22, "rel_alpha": 0.38, "rel_beta": 0.20 }]
}
```

### `label`
```json
{"command": "label", "ok": true, "label_id": 42}
```

### `search-labels` (semantic search)
```jsonc
{
  "command": "search-labels", "ok": true,
  "results": [{
    "text": "deep focus block",
    "EXG_metrics": { "focus": 0.82, "relaxation": 0.35, "engagement": 0.75, "hr": 65.0, "mood": 0.60 },
    "EXG_start": 1740412800, "EXG_end": 1740412805,
    "created_at": 1740412802,
    "similarity": 0.92
  }]
}
```

### `umap` (3D projection)
```jsonc
{
  "command": "umap", "ok": true,
  "result": {
    "points": [{ "x": 1.23, "y": -0.45, "z": 2.01, "session": "a", "utc": 1740412800 }],
    "analysis": {
      "separation_score": 1.84,
      "inter_cluster_distance": 2.31,
      "intra_spread_a": 0.82, "intra_spread_b": 0.94,
      "centroid_a": [1.23, -0.45, 2.01],
      "centroid_b": [-0.87, 1.34, -1.22]
    }
  }
}
```

---

## Useful `jq` Snippets

```bash
# Get just focus score
npx neuroskill status --json | jq '.scores.focus'

# Get all band powers
npx neuroskill status --json | jq '.scores.bands'

# Check device battery
npx neuroskill status --json | jq '.device.battery'

# Get signal quality
npx neuroskill status --json | jq '.signal_quality'

# Find improving metrics after a session
npx neuroskill session 0 --json | jq '[.trends | to_entries[] | select(.value == "up") | .key]'

# Sort comparison deltas by improvement
npx neuroskill compare --json | jq '.insights.deltas | to_entries | sort_by(.value.pct) | reverse'

# Get sleep efficiency
npx neuroskill sleep --json | jq '.analysis.efficiency_pct'

# Find closest neural match
npx neuroskill search --json | jq '[.result.results[].neighbors[]] | sort_by(.distance) | .[0]'

# Extract TBR from labeled stress moments
npx neuroskill search-labels "stress" --json | jq '[.results[].EXG_metrics.tbr]'

# Get session timestamps for manual compare
npx neuroskill sessions --json | jq '{start: .sessions[0].start_utc, end: .sessions[0].end_utc}'
```

---

## Data Storage

- **Local database**: `~/.skill/YYYYMMDD/` (SQLite + HNSW index)
- **ZUNA embeddings**: 128-D vectors, 5-second epochs
- **Labels**: Stored in SQLite, indexed with bge-small-en-v1.5 embeddings
- **All data is local** — nothing is sent to external servers
