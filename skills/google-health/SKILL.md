---
triggers: ["my sleep data", "my step count", "my heart rate data", "my nutrition log", "my exercise history", "google health api"]
name: google-health
description: "Read and write the user's Google Health data (sleep, activity, heart-rate, nutrition, exercise) via health.googleapis.com — Fitbit-descended API."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [all]
metadata:
  hermes:
    tags: [Health, Google, Fitbit, sleep, nutrition, exercise, activity]
    related_skills: [gws-automation]
prerequisites:
  python: ["tools.gws_auth"]
---

# Google Health

Read and write the user's **Google Health** data (the Fitbit-descended API at
`health.googleapis.com/v4`, NOT the deprecated Google Fit `fitness.googleapis.com`).

## When to use

- "What was my sleep last night?", "how many steps today?", "my resting heart rate"
- "Log this food / meal", "record my workout / run"
- Any read/write of the user's Fitbit / Google Health data.

## The one rule that matters

**NEVER use `build_service` for the Health API.** The vault token is a bundled
grant (gmail + calendar + ... + googlehealth), and Google Health rejects any
token carrying non-health scopes (`DISALLOWED_OAUTH_SCOPES`). Always go through
the down-scoping helper in `tools.gws_auth`, which mints a health-only token:

```python
import sys; sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import google_health_request

# READ: yesterday's sleep (civil time = user's local clock; only >= and < allowed)
import datetime
today = datetime.date.today()
y = today - datetime.timedelta(days=1)
filt = f'sleep.interval.civil_end_time >= "{y}" AND sleep.interval.civil_end_time < "{today}"'
data = google_health_request("GET", "users/me/dataTypes/sleep/dataPoints",
                             params={"filter": filt})
```

`google_health_request(method, path, params=..., json_body=...)` returns parsed
JSON (`{}` on empty), raises `RuntimeError` on non-2xx. `service_name` defaults
to `"google-gmail"` (the account that consented health scopes).

Use `/opt/hermes/.venv/bin/python` and run via the `execute_code` / terminal tool
inside a session (identity comes from the session, like all GWS calls).

## Data types & scopes

Path pattern: `users/me/dataTypes/<TYPE>/dataPoints`. Reads need the bundle's
`.readonly` scope; creating/editing/deleting **our own** entries needs
`.writeonly` (writeonly does NOT grant reads).

| Bundle (scope) | Covers (TYPE) |
|---|---|
| `activity_and_fitness` | `exercise`, `steps`, `active-energy-burned`, `distance`, heart-rate-zones, floors, VO2 |
| `health_metrics_and_measurements` | `heart-rate`, HRV, `spo2`, `weight`, `body-fat`, respiratory-rate, temperature, blood-glucose, height |
| `nutrition` | `food`, `nutrition-log`, `hydration-log` |
| `sleep` | `sleep` |
| read-only extras | `ecg`, `irn`, `location` (GPS during exercise) |

## Reads

- **Filter field caveat:** the filter member is the data type's own message
  field. **Confirmed working: `sleep.interval.civil_end_time`.** Other types use
  their own member (e.g. the STEPS query 400'd on a guessed field). If unsure,
  check developers.google.com/health/filters, or omit `filter` and page results,
  then filter client-side by `interval`.
- **"Last night" pitfall:** sleep that ends this morning has `civil_end_time` =
  *today*, so a `< today` upper bound excludes it. For last night use
  `[today-1, today+1)` or bound on `today+1`.
- Optional `pageSize` (default 1440, max 10000).

### Sleep fields
`sleep.summary`: `minutesInSleepPeriod`, `minutesAsleep`, `minutesAwake`,
`minutesToFallAsleep`, `minutesAfterWakeUp`; `stagesSummary[]` = per-stage
`{type, minutes, count}`; raw `sleep.stages[]` = `{startTime, endTime, type}`
(AWAKE/LIGHT/DEEP/REM). **There is no numeric "sleep score"** — derive from
stages if the user wants one.

## Writes

Create = POST a DataPoint. The response is an LRO wrapper; save
`response.name` (needed for edit/delete). Update = `PATCH users/me/.../<name>`;
delete = `POST users/me/dataTypes/<TYPE>/dataPoints:batchDelete` with
`{"names": [...]}`.

### Log food (nutrition)
```python
from tools.gws_auth import google_health_request
import datetime
now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
res = google_health_request("POST", "users/me/dataTypes/nutrition-log/dataPoints",
    json_body={"nutritionLog": {
        "interval": {"startTime": iso(now - datetime.timedelta(minutes=30)), "endTime": iso(now)},
        "foodDisplayName": "Grilled chicken salad",
        "mealType": "LUNCH",                       # BREAKFAST/LUNCH/DINNER/SNACK
        "energy": {"kcal": 420},
        "totalCarbohydrate": {"grams": 12}, "totalFat": {"grams": 18},
        "nutrients": [{"type": "PROTEIN", "quantity": {"grams": 40}}],
    }})
name = res.get("response", {}).get("name")   # for later edit/delete
```
Only `interval` is schema-required; `foodDisplayName` is required for free-form
entries. For an identified food, use `"food": "users/me/dataTypes/food/dataPoints/<id>"`
instead of `foodDisplayName` (macros auto-populate).

### Log exercise (activity)
```python
google_health_request("POST", "users/me/dataTypes/exercise/dataPoints",
    json_body={"exercise": {
        "interval": {"startTime": "...Z", "startUtcOffset": "0s",
                     "endTime": "...Z", "endUtcOffset": "0s"},
        "exerciseType": "RUNNING",               # WALKING/BIKING/AEROBIC_WORKOUT/...
        "displayName": "Morning run",
        "activeDuration": "1800s",
        "metricsSummary": {"caloriesKcal": 380.0, "distanceMillimeters": 5000000.0, "steps": "6200"},
    }})
```
`metricsSummary` also accepts `averageHeartRateBeatsPerMinute`, `activeZoneMinutes`,
`averageSpeedMillimetersPerSecond`, `elevationGainMillimeters`. **Caveat:** writing
an exercise does NOT auto-update daily rollups — write `steps` /
`active-energy-burned` / `heart-rate` DataPoints separately if daily totals should move.

## Constraints

- All `googlehealth.*` are **restricted** scopes. Works for accounts on the GCP
  OAuth client's **Test Users** list (100-user cap); going public needs Google's
  restricted-scope (CASA) review.
- If a call raises "no googlehealth.* scopes granted", the account must
  re-consent via the connect-account flow (`get_auth_url` uses the full scope set).
