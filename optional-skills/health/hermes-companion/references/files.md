# Hermes Companion files

The Hermes Companion iPhone app writes these files into the iCloud ubiquity container. The skill has no other source. Timestamps are ISO-8601 UTC with a `Z` suffix. Age is `now` in UTC minus that timestamp.

Default directory:

`~/Library/Mobile Documents/iCloud~com~hermes~HermesCompanion/Documents/`

## latest_location.json

Written only after `GPSPersistDecision` accepts a move. Refused indoor drift does not change the file or its modification time.

| Field | Meaning |
| --- | --- |
| `latitude`, `longitude` | Last accepted coordinate |
| `timestamp` | When that move was accepted, UTC |
| `horizontal_accuracy` | Meters |
| `speed_mps` | Speed at that fix. Negative means unknown |
| `movement_reason` | `moved`, `distance`, or `no_motion_reading`. Optional on older files |
| `motion_activity` | `stationary`, `walking`, `running`, `cycling`, `automotive`, `unknown` |
| `motion_confidence` | `high`, `medium`, `low` |
| `motion_timestamp` | When CoreMotion last reported, independent of the GPS write |
| `source` | What woke the app (`significant`, geofence, visit, standard) |
| `battery_level`, `battery_state` | Omitted from new writes. Older files may still have them. Ignore them. The skill does not report phone battery |
| `app_state` | `active`, `background`, or the wake state |

## latest_health.json

One deduplicated Apple Health snapshot. Sleep duration is the union of Deep, REM, and Core for the last clustered night, so overlapping Watch and iPhone samples are not added twice.

| Field | Meaning |
| --- | --- |
| `timestamp` | When the snapshot was written, UTC |
| `recovery_status` | `recovered`, `moderate`, or `fatigued` |
| `sleep` | Duration, stage summary, quality rating. Absent when no night was clustered |
| `workout` | Type, phase, duration, calories, `is_currently_active`, `minutes_since_completion` |
| `step_count_today` | Steps since local midnight |
| `active_calories_today` | Active energy since local midnight |
| `resting_heart_rate_bpm` | Resting heart rate |
| `heart_rate_variability_sdnn` | HRV SDNN in milliseconds |
| `conversational_context`, `sleep.summary`, `workout.summary` | Optional English sentences from older app builds. Do not say them. Use the numeric fields and reply in the user's language |

## places.json

A JSON list owned by the user, not by iCloud. Each object:

```json
{
  "id": "home",
  "name": "Home",
  "category": "home",
  "activity": "at home",
  "latitude": 0.0,
  "longitude": 0.0,
  "radius_meters": 120,
  "notes": ""
}
```

The nearest place whose radius contains the coordinate wins. No match stays `Unlisted place`.
