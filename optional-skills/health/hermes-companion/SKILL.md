---
name: hermes-companion
description: Requires the Companion iPhone app for place and health.
version: 1.2.0
author: danielmain, Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Location, Health, iPhone, iCloud, Apple]
    category: health
    related_skills: []
    config:
      - key: hermes-companion.places_file
        description: JSON file of named places with latitude, longitude, and radius
        default: "~/.hermes/hermes-companion/places.json"
        prompt: Places file path
      - key: hermes-companion.icloud_dir
        description: iCloud Documents directory where the iPhone writes latest_location.json and latest_health.json
        default: "~/Library/Mobile Documents/iCloud~com~hermes~HermesCompanion/Documents"
        prompt: iCloud container Documents path
---

# Hermes Companion

Read the user's live place, motion, sleep, workout, and recovery. The script prints facts. How to answer is in Language.

## Requires the iPhone app

This skill works only with the Hermes Companion iOS app installed on the user's iPhone. The app and this skill are documented at [hermescompanion.funktional.dev](https://hermescompanion.funktional.dev). Source: [github.com/danielmain/hermes-companion](https://github.com/danielmain/hermes-companion). The app writes `latest_location.json` and `latest_health.json` into the user's private iCloud container. This skill reads the copies macOS has already synced. It does not call a relay, open a port, or produce a coordinate on its own.

Until that app is installed, Location is set to Always, and iCloud has synced those two files, the script reports that no file exists. Say that the Hermes Companion iPhone app is required and has not synced yet. Do not invent a place, a motion state, or a health number.

## When to Use

- The user asks where they are, whether they are home, or what they are doing physically.
- A reply depends on being still, walking, driving, at the gym, or in transit.
- The user asks about sleep, a workout, steps, heart rate, recovery, or post-workout food.
- A check-in would be wrong without knowing if they are mid-workout or still at a known place.

Do not use this skill for a generic map, a route, or weather. Do not invent a place or a health number when the script has no file.

## Prerequisites

- The Hermes Companion iPhone app. Product page: [hermescompanion.funktional.dev](https://hermescompanion.funktional.dev). Source: [github.com/danielmain/hermes-companion](https://github.com/danielmain/hermes-companion). The skill has no data source other than that app.
- macOS, signed into the same Apple ID as the iPhone.
- Location set to Always, Motion & Fitness allowed, and Health access allowed.
- iCloud Drive has finished downloading `latest_location.json` and `latest_health.json`.

If the skill config block names `hermes-companion.places_file` or `hermes-companion.icloud_dir`, pass those paths as `--places` and `--icloud-dir`. Otherwise the script uses its defaults, including an existing Hermes profile `state/places.json` when exactly one profile has that file.

## How to Run

Run the bundled script with the `terminal` tool. `${HERMES_SKILL_DIR}` is the skill directory.

```bash
python3 ${HERMES_SKILL_DIR}/scripts/companion.py
python3 ${HERMES_SKILL_DIR}/scripts/companion.py --health
python3 ${HERMES_SKILL_DIR}/scripts/companion.py --context
python3 ${HERMES_SKILL_DIR}/scripts/companion.py --json
```

Known places:

```bash
python3 ${HERMES_SKILL_DIR}/scripts/companion.py --list
python3 ${HERMES_SKILL_DIR}/scripts/companion.py --add --name "Home" --category home --lat LAT --lon LON --radius 120
python3 ${HERMES_SKILL_DIR}/scripts/companion.py --remove home
```

Categories: `home`, `work`, `gym`, `cafe`, `outdoors`, `general`.

Save a place only when the user asks, using the coordinates from the latest script output.

## Quick Reference

| Question | Command | Fields to trust |
| --- | --- | --- |
| Where are they? | `companion.py` | `place_name`, `place_category`, `still_there` |
| Moving right now? | `companion.py` | `motion_activity`, `motion_fresh`, `is_moving_now` |
| How long in this place? | `companion.py` | `minutes_since_last_move` |
| Sleep, workout, recovery | `companion.py --health` | `sleep_duration`, `sleep_quality`, `workout_type`, `recovery_status` |
| Both | `companion.py --context` | the two blocks together |

`movement_reason` is why the phone accepted the last write: `moved`, `distance`, or `no_motion_reading`. `absent` means an older file from before that field existed.

## Language

This file is English because the model reads it. The user never sees it.

- Reply in the user's language, in the agent's own voice. Spanish, German, and every other language use these same fields.
- Keep a place name exactly as the user saved it (`Casa`, `Home`, `Arbeit`).
- `In Transit` and `Unlisted place` are codes. Translate them.
- Translate category, motion, sleep-quality, and recovery codes (`home`, `walking`, `automotive`, `excellent`, `fatigued`). A `workout_type` that is already a name in the user's language stays as written.
- `minutes_since_last_move`, durations, and heart-rate numbers stay numeric.
- Do not quote the script. Ignore `suggested_greeting`, `context_summary`, `suggested_openers`, `sleep_insight`, `workout_insight`, `nutrition_reminder`, `sleep.summary`, and `workout.summary` if a raw file still contains them.

## Procedure

1. Run `companion.py` for place and motion, `--health` for body metrics, or `--context` when the answer needs both. Completion: the command prints a `place_name:` or `recovery_status:` line, or an explicit "no file yet" line. On "no file yet", tell the user the Hermes Companion iPhone app has to be installed and synced, then stop.
2. Treat `place_name` as where they are now. `minutes_since_last_move` is how long the phone has kept that coordinate. A large number at a known place means they are still there. Say that in the present tense, in the user's language.
3. Treat `motion_activity` when `motion_fresh` is `yes` as what the body is doing this minute. Walking at a saved place with an old GPS age is still that place, walking around, not lost and not in transit.
4. Treat a fresh `recorded_at` as an accepted move. It does not say they arrived, left, or came back. Do not announce an arrival unless they said so.
5. For health, use only lines present in this run. `age_seconds` is when that snapshot was saved. Steps and calories are from that time. Sleep is for the morning, or a short night mentioned in the evening. Do not recap last night's hours in the afternoon. A workout in progress gets one short line. A workout finished within about 90 minutes can include how it felt and protein or water as care. Recovery `fatigued` is the only case for urging rest.
6. If `place_name` is `Unlisted place`, ask what to call it. Do not name it home, work, or a gym from the coordinates alone.

## Place and Motion Rules

The phone rewrites `latest_location.json` only when a move clears the persist gate:

- Fresh walking, running, cycling, or driving: at least the distance filter (default 30 m). Reason `moved`.
- Fresh stationary: only past 150 m (reason `distance`), or GPS speed above 1 m/s and still past the distance filter (reason `moved`). A shorter jump is indoor drift and is not written.
- No fresh motion sample: the distance filter alone. Reason `no_motion_reading`.

CoreMotion is refreshed on its own. Trust `is_moving_now` plus `motion_activity` for moving versus staying. Trust `place` for which place. Automotive motion is in transit even inside a saved radius. Walking, running, or cycling inside a saved radius stays at that place.

`is_stale` on older readers means a long stay, not a lost fix. This script prints `still_there: yes` for every accepted file.

Field notes live in `references/files.md`. Load that file only when a raw key is unclear.

## Pitfalls

- "No file yet" means the Hermes Companion iPhone app is missing, not yet allowed to run, or iCloud has not downloaded its files. Say that. Do not reuse a place from an earlier conversation as if it were a fresh reading.
- The simulator has no CoreMotion. `motion_activity: unknown` and `motion_fresh: no` are expected there.
- Motion & Fitness must be allowed or `motion_activity` stays `unknown`.
- Coordinates are for saving a place or when the user asks for them. Do not recite them in a normal reply.
- Ignore `battery_level` and `battery_state` if a file still has them. Phone battery is not part of this skill.
- Two Macs on the same Apple ID share the container. Read the local file; do not fetch it from the network.
- The script does not reverse-geocode. An unlisted coordinate stays unlisted until the user names it.

## Verification

Run `python3 ${HERMES_SKILL_DIR}/scripts/companion.py` again. The `place_name:` line matches the previous reading when they have not moved, and `source:` is a file under the iCloud container. For a code change, run `python3 ${HERMES_SKILL_DIR}/scripts/test_companion.py` and confirm `ok`.
