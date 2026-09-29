---
name: tuya-smart-control
description: Control Tuya smart home devices via natural language. Use when the user asks to control smart devices (turn on/off lights, AC, plugs, adjust brightness/temperature/mode), query device status or list devices, manage homes and rooms, rename devices, check weather by location, send notifications (SMS, voice call, email, or App push), view device data statistics (e.g. energy/power consumption), or capture snapshots/short videos from IPC cameras. Requires TUYA_API_KEY.
metadata: { "openclaw": { "version": "1.0.0", "emoji": "🏠", "requires": { "env": ["TUYA_API_KEY"], "pip": ["requests>=2.28.0"] }, "primaryEnv": "TUYA_API_KEY" } }
---

# Tuya Smart Home Device Control Skill

## Basic Information

- **Official Website**: https://www.tuya.com/
- **Source Code**: https://github.com/tuya/tuya-openclaw-skills
- **Authentication**: Via Header `Authorization: Bearer {Api-key}`
- **Credentials**: Read from environment variable `TUYA_API_KEY`. Base URL is auto-detected from API key prefix. See `references/api-conventions.md` for the prefix-to-region mapping table. You can override by setting `TUYA_BASE_URL`.
- **API Reference**: See individual files under `references/`
- **Python SDK**: See `scripts/tuya_api.py`

## Environment Variable Configuration

Set the following environment variable before use:

```bash
export TUYA_API_KEY="your-tuya-api-key"
# TUYA_BASE_URL is optional — auto-detected from API key prefix
# Override only if needed: export TUYA_BASE_URL="https://openapi.tuyaus.com"
```

The skill will not load if the `TUYA_API_KEY` environment variable is missing.

### How to Obtain API Key

**Important:** When a user asks where to get the API key, **always read `references/api-conventions.md`** for the official, up-to-date information.

Per that document:
- China Mainland users: Get from https://tuyasmart.com/
- International users: Get from https://tuya.ai/
- API key format: `sk-` prefix with 2-character region code (e.g., `sk-AY...` for China, `sk-AZ...` for US West)
- The region prefix determines the base URL automatically

## Usage

**Always prefer Method 1 (Command Line)** — single command, no boilerplate code. It handles authentication, URL resolution, JSON serialization, and error handling automatically.

### Method 1: Via Command Line (Recommended)

```bash
python3 {baseDir}/scripts/tuya_api.py <command> [params...]
# Examples:
python3 {baseDir}/scripts/tuya_api.py homes
python3 {baseDir}/scripts/tuya_api.py devices
python3 {baseDir}/scripts/tuya_api.py devices --home 5053559
python3 {baseDir}/scripts/tuya_api.py devices --room 123456
python3 {baseDir}/scripts/tuya_api.py device_detail <device_id>
python3 {baseDir}/scripts/tuya_api.py model <device_id>
python3 {baseDir}/scripts/tuya_api.py control <device_id> '{"switch_led":true}'
python3 {baseDir}/scripts/tuya_api.py rename <device_id> "New Name"
python3 {baseDir}/scripts/tuya_api.py weather 39.90 116.40
python3 {baseDir}/scripts/tuya_api.py sms "Your message"
python3 {baseDir}/scripts/tuya_api.py voice "Your message"
python3 {baseDir}/scripts/tuya_api.py mail "Subject" "Content"
python3 {baseDir}/scripts/tuya_api.py push "Subject" "Content"
python3 {baseDir}/scripts/tuya_api.py stats_config
python3 {baseDir}/scripts/tuya_api.py stats_data <dev_id> <dp_code> <type> <start> <end>
python3 {baseDir}/scripts/tuya_api.py ipc_pic_fetch <device_id> <consent> [pic_count] [home_id]
python3 {baseDir}/scripts/tuya_api.py ipc_video_fetch <device_id> <duration> <consent> [home_id]
```

CLI validation rules:
- `devices` supports only one scope flag at a time: `--home <id>` or `--room <id>`
- `control` requires `properties_json` to be a valid JSON object (not array/string)
- `weather` validates coordinate range: latitude `[-90, 90]`, longitude `[-180, 180]`
- `stats_data` validates `start`/`end` format `yyyyMMddHH` and max 24-hour window
- `ipc_pic_fetch` args: `<device_id> <consent> [pic_count] [home_id]` — consent `1` = decrypted URL
- `ipc_video_fetch` args: `<device_id> <duration> <consent> [home_id]` — duration in seconds (1-60)
- Use `python3 {baseDir}/scripts/tuya_api.py --help` for command help and examples

### Method 2: Via Python SDK

Use when you need to chain multiple API calls or do complex logic in a single script:

```python
import sys
sys.path.insert(0, "{baseDir}/scripts")
from tuya_api import TuyaAPI

api = TuyaAPI()

homes = api.get_homes()
devices = api.get_all_devices()
detail = api.get_device_detail("device_id_here")
result = api.issue_properties("device_id_here", {"switch_led": True, "bright_value": 500})
weather = api.get_weather(lat="39.90", lon="116.40")
# IPC cloud capture — take a snapshot and get decrypted URL
capture = api.ipc_ai_capture_pic_allocate_and_fetch("device_id_here", user_privacy_consent_accepted=True)
```

## Feature Overview

| Module | Capabilities | Reference |
|--------|-------------|-----------|
| Home Management | List all homes, list rooms in a home | `references/home-and-space.md` |
| Device Query | All devices, devices by home/room, single device detail (including current property states) | `references/device-query.md` |
| Device Control | Query device Thing Model, issue property commands | `references/device-control.md` |
| Device Management | Rename device | `references/device-management.md` |
| Weather Service | Current and forecast weather | `references/weather.md` |
| Notifications | SMS, voice call, email, App push | `references/notifications.md` |
| Data Statistics | Hourly statistics config query, statistics value query | `references/statistics.md` |
| IPC Cloud Capture | Cloud snapshot and short video capture for IPC cameras | `references/ipc-cloud-capture.md` |
| Error Handling | Error codes and recovery strategies | `references/error-handling.md` |
| API Conventions | Request/response format, data center mapping | `references/api-conventions.md` |

## Core Workflows

### Workflow 1: Device Control

When the user says things like "turn on the living room light" or "set the AC temperature to 26 degrees":

1. **Locate the device** — Find the target device based on the device name or location mentioned by the user. Follow this priority:
   - **Priority 1 — Room + category match**: If the user mentions a room (e.g. "living room AC"), first query the home list → room list to match the room, then list devices in that room and match by `category_name` or device `name`
   - **Priority 2 — Device name match**: If the user only mentions a device name (e.g. "AC"), call "List All Devices" API and match by `category_name` first, then by device `name` fuzzy match
   - **Priority 3 — Disambiguation**: If multiple devices match, list all candidates with their room information and ask the user to choose

2. **Get current state** — Call the "Get Single Device Detail" API
   - **If `result` is `null`**: the device does not exist or you have no permission — inform the user and stop
   - **If `online` is `false`**: the device is offline — tell the user "Device XX is currently offline, please check its power and network connection" and do not proceed further
   - Only continue when `result` is valid and `online` is `true`
   - The `properties` field contains current values of each functional property (e.g. switch state, brightness, temperature)

3. **Query capabilities** — Call the "Query Device Thing Model" API to get the device's supported property list
   - **Important**: The `result.model` field is a JSON **string** that must be parsed again (e.g. `json.loads(result["model"])`) to obtain the property definitions
   - Check each property's `accessMode`:
     - `ro` (read-only): cannot be controlled, only queried — inform the user "this property is read-only"
     - `wr` (write-only): can be controlled but current value cannot be read
     - `rw` (read-write): can be both controlled and queried

4. **Map the command** — Map the user's intent to Thing Model properties:
   - Turn on/off → find a bool-type switch property (e.g. `switch_led`, `switch`)
   - Adjust brightness → find a value-type brightness property
   - Adjust temperature → find a value-type temp property
   - If the device does not support the requested function, inform the user and list supported functions
   - **Relative adjustments** — When the user says "a bit brighter", "lower the temperature by 2 degrees", etc.:
     1. Read the current value from `properties` in the device detail (Step 2)
     2. Read `min`, `max`, `step` from the Thing Model `typeSpec` (Step 3)
     3. Calculate the target value:
        - Vague ("a bit", "a little") → current value ± (max - min) × 10%
        - Specific ("by 2 degrees", "by 100") → current value ± the specified amount
     4. Clamp the target value within [min, max] and round to the nearest `step`
   - **Validate value range**: Before issuing, confirm the target value is within the `typeSpec` min/max range

5. **Issue the command** — Call the "Issue Properties" API using the Python SDK: `api.issue_properties(device_id, {property_code: value})`
   - The SDK handles `properties` JSON string serialization automatically
   - If not using the SDK: the `properties` field must be a JSON **string**, not a JSON object. You must double-serialize: `{"properties": "{\"switch_led\":true}"}`

6. **Verify and return result** — After issuing the command:
   - Wait 1-2 seconds, then call "Get Device Detail" again to read the updated `properties`
   - Compare the target value with the actual value to confirm execution
   - If values match: inform the user the operation succeeded
   - If values differ: tell the user "command sent, but the device state has not updated yet — there may be a delay"

### Workflow 2: Rename Device

1. Locate the device using Workflow 1 Step 1 to obtain the device_id
2. Call the "Rename Device" API with the new name
3. Return the result

### Workflow 3: Notifications

1. Identify the message type: SMS / Voice / Email / App Push
2. Extract required parameters (message content; email and push also need a subject)
3. Call the corresponding API (all notification APIs are self-send — messages can only be sent to the current logged-in user)
4. Return the send result

### Workflow 4: Weather Query

1. **Obtain coordinates**:
   - First call Home List API and check the `latitude` / `longitude` fields
   - **Note**: the coordinate format is `{"Value": "30.3"}` — you must extract the `.Value` field (e.g. `home["latitude"]["Value"]`)
   - If the home has no location set, ask the user for their city and convert to coordinates (see common city coordinates in `references/weather.md`)
2. Determine which weather attributes to query (default: temperature, humidity, weather condition)
3. Call the weather query API
4. Translate the returned data into a human-readable description

### Workflow 5: Data Statistics

1. Locate the device (same as Workflow 1 Step 1)
2. Call the "Statistics Config Query" API to confirm whether the device has the corresponding statistics capability
3. If available, call the "Statistics Value Query" API
   - **Time inference**: Convert the user's natural language to `yyyyMMddHH` format:
     - "today" → start = today 00:00, end = current hour
     - "yesterday" → start = yesterday 00:00, end = yesterday 23:00
     - The time range cannot exceed 24 hours per request — for longer ranges, make multiple requests and aggregate
     - Format example: `2024010100` = January 1, 2024 00:00
4. Aggregate and return the results

### Workflow 6: Device Status Query

When the user asks "Is the living room light on?" or "What's the AC set to?":

1. Locate the device and get current state (same as Workflow 1 Steps 1-2; stop if device not found or offline)
2. Read `properties` values, cross-reference with the Thing Model property names/descriptions, and translate to natural language (e.g. `"switch_led": true` → "the light is currently on")

### Workflow 7: Multi-Device Batch Control

When the user says "Turn off all lights" or "Set all ACs to 26 degrees":

1. Call "List All Devices" API and filter matching devices by `category_name` or device `name` keyword
2. For each matching device: check `online` status (skip offline devices and note them), then execute Workflow 1 Steps 3-6
3. Aggregate results: report how many devices succeeded, which ones failed or were offline
4. Add a brief delay (0.5-1s) between requests to avoid rate limiting

### Workflow 8: IPC Cloud Capture

When the user asks to "take a photo with the camera" or "record a short video from the camera":

1. **Locate the IPC device** — same as Workflow 1 Step 1, filter by camera category
2. **Determine capture type**:
   - Snapshot → `PIC` (optional `pic_count`, 1-5)
   - Short video → `VIDEO` (optional `video_duration_seconds`, 1-60, default 10)
3. **Privacy consent** — Only set `user_privacy_consent_accepted=true` when the user has explicitly agreed to receive decrypted playable URLs. Default to `true` unless the user declines
4. **Execute capture** — Use the all-in-one helper methods:
   - For PIC: `api.ipc_ai_capture_pic_allocate_and_fetch(device_id, user_privacy_consent_accepted=True, pic_count=1)`
   - For VIDEO: `api.ipc_ai_capture_video_allocate_and_fetch(device_id, video_duration_seconds=5, user_privacy_consent_accepted=True)`
   - These methods handle the full allocate → wait → poll → retry flow automatically
5. **Return the result** — Extract the URL from the resolve result:
   - PIC with consent: `resolve["decrypt_image_url"]`
   - VIDEO with consent: `resolve["decrypt_video_url"]` (cover image may be null if still uploading)
   - If `status` is still `NOT_READY` after all retries, inform the user that the device may be slow to upload and suggest trying again later

### Workflow 9: IPC Visual Recognition

When the user asks "What's in front of my camera?", "Is there anyone at the door?", or "Describe what the camera sees":

1. **Capture a snapshot** — Follow Workflow 8 Steps 1-4 to take a PIC capture with `user_privacy_consent_accepted=True`
2. **Get the image URL** — Extract `resolve["decrypt_image_url"]` from the capture result. If the resolve failed or returned `NOT_READY`, inform the user and stop
3. **Download the image** — Fetch the image content from the decrypted URL
4. **Send to AI vision model** — Pass the image to the AI large model for visual understanding. Describe the image content in natural language based on the user's question:
   - General question ("What's there?") → describe the overall scene, objects, and people
   - Specific question ("Is there a package?", "Is anyone at the door?") → focus on answering the specific question
5. **Return the description** — Respond to the user with the visual analysis result in conversational language

## Important Notes

1. Device name matching uses fuzzy matching; when multiple results are found, ask the user to confirm
2. The statistics API time format is `yyyyMMddHH`, and the time range cannot exceed 24 hours per request
3. All four notification APIs are self-send only — messages can only be sent to the currently logged-in user
4. The weather query requires latitude and longitude; if unavailable from the Home API, ask for the user's city
5. Base URL is auto-detected from API key prefix. See `references/api-conventions.md` for details
6. If you encounter issues, visit https://github.com/tuya/tuya-openclaw-skills for announcements and troubleshooting
7. Never log or display the `TUYA_API_KEY` value in output
8. CLI exits with code `2` for usage/validation errors, and `1` for runtime/API/network errors

## Supported and Unsupported Operations

### Supported Property Types for Control

Only basic data type properties are currently supported for device control:

| Type | Description | Example |
|------|-------------|---------|
| bool | Boolean on/off | Turn light on/off, turn AC on/off, turn plug on/off |
| enum | Enumeration selection | Switch AC mode (auto/cold/hot), set fan speed (low/mid/high) |
| value (Integer) | Numeric value | Adjust brightness (0-1000), set temperature (16-30) |
| string | String value | Set device display text |

### Unsupported Operations

The following operations involve sensitive actions or complex data types and are **NOT supported**:

- **Lock control** — Unlock doors, lock/unlock smart locks (security-sensitive)
- **Live video streaming** — Pull real-time video streams or view camera live footage (cloud snapshot/short video capture IS supported — see Workflow 8)
- **Image operations** — Retrieve or push images from/to devices
- **Complex data type control** — Properties with `raw`, `bitmap`, `struct`, or `array` typeSpec are not supported for issuing commands
- **Firmware upgrades** — OTA firmware update operations
- **Device pairing/removal** — Adding new devices or removing existing devices

If the user requests any of these unsupported operations, clearly inform them that the operation is not available through this skill and suggest using the Tuya App directly.

## Data Egress Statement

**This skill sends data to the Tuya Open Platform**:

| Data Type | Sent To | Purpose | Required |
|-----------|---------|---------|----------|
| Api-key | User-configured base_url | API authentication | Required |
| Device ID | User-configured base_url | Device query and control | Required |
| Control commands | User-configured base_url | Device property issuance | Required |
