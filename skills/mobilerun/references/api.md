# Mobilerun API Reference

Full endpoint details for all Mobilerun APIs. For the runbook and quick reference, see [SKILL.md](../SKILL.md).

Base URL: `https://api.mobilerun.ai/v1`
Auth: `Authorization: Bearer <MOBILERUN_API_KEY>`

---

## Device Management

### Device States

| State | Meaning |
|-------|---------|
| `creating` | Being provisioned (cloud only) |
| `assigned` | Assigned but not yet ready |
| `ready` | Connected and accepting commands |
| `disconnected` | Portal app closed or network lost |
| `terminated` | Shut down (cloud only) |
| `maintenance` | Under maintenance (cloud only) |
| `migrating` | Moving between hosts (cloud only, 1-5 min) |

### List Devices — `GET /devices`

Query params: `state` (array), `type` (`dedicated_physical_device`, `dedicated_premium_device`), `name` (partial match), `page` (default: 1), `pageSize` (default: 20), `orderBy` (`id`, `createdAt`, `updatedAt`, `assignedAt`), `orderByDirection` (`asc`, `desc`)

Response: `{ items: DeviceInfo[], pagination: Meta }`

### Get Device — `GET /devices/{deviceId}`

### Device Count — `GET /devices/count`

### Provision — `POST /devices?deviceType=...`

Body: `{"name": "...", "apps": ["com.example.app"]}`

Cloud & Physical need a proxy for internet. Personal Phones don't.

### Wait for Ready — `GET /devices/{deviceId}/wait`

Blocks until state transitions to `ready`.

### Terminate — `DELETE /devices/{deviceId}` (body: `{}`)

Personal devices can't be terminated — they disconnect when Portal closes.

### Time — `GET /devices/{deviceId}/time`

### Timezone — `GET /devices/{deviceId}/timezone` | `POST` with `{"timezone": "Europe/Berlin"}`

### Rename — `PUT /devices/{deviceId}/name` with `{"name": "new-name"}`

### Reboot — `POST /devices/{deviceId}/reboot`

Physical and Personal only. Not available on Cloud/VMOS.

---

## Screen Observation

### Screenshot — `GET /devices/{deviceId}/screenshot`

Query param: `hideOverlay` (default: false). Returns PNG binary.

### UI State — `GET /devices/{deviceId}/ui-state`

Query param: `filter` (default: false) — set `true` to filter non-interactive elements.

Returns `phone_state` (current app, keyboard, focused element), `device_context` (screen bounds, display metrics), `a11y_tree` (recursive element tree).

Key a11y_tree node fields: `text`, `contentDescription`, `resourceId`, `boundsInScreen` (`{left, top, right, bottom}`), `isClickable`, `isEditable`, `isScrollable`, `children`.

Tap target center: `x = (left + right) / 2`, `y = (top + bottom) / 2`

---

## Device Actions

### Tap — `POST /devices/{deviceId}/tap`

Body: `{"x": 540, "y": 960}`

### Swipe — `POST /devices/{deviceId}/swipe`

Body: `{"startX": 540, "startY": 1200, "endX": 540, "endY": 400, "duration": 300}`

Duration in ms (min: 10). Scroll down = high startY → low endY.

### Global Actions — `POST /devices/{deviceId}/global`

Body: `{"action": 2}` — 1=BACK, 2=HOME, 3=RECENT

### Type Text — `POST /devices/{deviceId}/keyboard`

Body: `{"text": "hello", "clear": false}` — supports Unicode. Focus an input field first.

### Press Key — `PUT /devices/{deviceId}/keyboard`

Body: `{"key": 66}` — 4=BACK, 61=TAB, 66=ENTER, 67=DEL, 112=FORWARD_DEL

### Clear Input — `DELETE /devices/{deviceId}/keyboard`

### Set Location — `POST /devices/{deviceId}/location`

Body: `{"latitude": 37.7749, "longitude": -122.4194}`

### Get Location — `GET /devices/{deviceId}/location`

### Overlay — `GET /devices/{deviceId}/overlay` | `POST` with `{"visible": false}`

### File Transfer

- List: `GET /devices/{id}/files?path=/sdcard/Download`
- Upload: `POST /devices/{id}/files?path=/sdcard/Download/file.txt` (body = file content)
- Download: `GET /devices/{id}/files/download?path=/sdcard/Download/file.txt`
- Delete: `DELETE /devices/{id}/files?path=/sdcard/Download/file.txt`

Path depends on device type — physical uses `/sdcard/Download/`, cloud devices use `/` as root.

### Connect Proxy — `POST /devices/{deviceId}/proxy`

Body: `{"host": "...", "port": 1080, "user": "...", "password": "..."}` (SOCKS5) or `{"wireguard": {...}}`. Optional: `name`, `smartIp`.

Check: `GET /devices/{deviceId}/proxy`. Disconnect: `DELETE /devices/{deviceId}/proxy`.

### eSIM — Physical and Personal Phones only

All Physical Phones are hosted in Germany — eSIM must support activation/roaming there.

- List: `GET /devices/{id}/esim`
- Download: `POST` with `{"smDpAddr": "...", "matchingId": "...", "enable": true}`
- Enable: `PUT` with `{"subId": 2}`
- Delete: `DELETE /devices/{id}/esim?subId=2`
- Status: `GET /devices/{id}/esim/status`
- APN list: `GET /devices/{id}/esim/apn`
- APN create: `POST /devices/{id}/esim/apn` with `{"name":"...","apn":"...","mcc":"...","mnc":"...","protocol":"IPV4V6","roamingProtocol":"IPV4V6","type":"default,supl","subId":2}`
- APN select: `PUT /devices/{id}/esim/apn` with `{"apnId": 1, "subId": 2}`
- Roaming: `PUT /devices/{id}/esim/roaming` with `{"enabled": true}`

---

## App Management

- List apps: `GET /devices/{id}/apps` (param: `includeSystemApps`)
- List packages: `GET /devices/{id}/packages` (param: `includeSystemPackages`)
- Install: `POST /devices/{id}/apps` with `{"packageName": "..."}` — from app library, prefer Play Store
- Start: `PUT /devices/{id}/apps/{pkg}` with `{}` (optional: `{"activity": "..."}`)
- Stop: `PATCH /devices/{id}/apps/{pkg}` with `{}`
- Uninstall: `DELETE /devices/{id}/apps/{pkg}` with `{}`

## App Library

- List: `GET /apps` (params: `page`, `pageSize`, `source`, `query`, `sortBy`, `order`)
- Get: `GET /apps/{id}`
- Upload: 3-step — `POST /apps/create-signed-upload-url` → PUT to R2 URL → `POST /apps/{id}/confirm-upload`
- Delete: `DELETE /apps/{id}`

---

## Tasks (AI Agent)

### Run Task — `POST /tasks`

Required: `task` (string), `deviceId` (UUID)

Optional: `llmModel` (default: `anthropic/claude-sonnet-4.6`), `apps`, `credentials` (`[{packageName, credentialNames[]}]`), `maxSteps` (default: 100), `reasoning` (default: true — set `false` unless user requests), `vision` (default: false), `temperature` (default: 0.5), `executionTimeout` (default: 1000s), `outputSchema` (JSON schema), `vpnCountry` (`US`,`BR`,`FR`,`DE`,`IN`,`JP`,`KR`,`ZA`), `continueOnFailure` (default: false)

Returns: `{"id": "uuid", "status": "queued", "streamUrl": "..."}`. Initial `queued` is normal — transitions to `running` shortly.

### Task Status — `GET /tasks/{id}/status`

Response: `status`, `succeeded`, `message`, `output`, `steps`, `lastResponse`

Statuses: `queued`, `created`, `running`, `paused`, `completed`, `failed`, `cancelled`

### Steer Task — `POST /tasks/{id}/message` with `{"message": "..."}`

### Cancel — `POST /tasks/{id}/cancel`

### Details — `GET /tasks/{id}`

### List — `GET /tasks` (params: `status`, `orderBy`, `orderByDirection`, `query`, `page`, `pageSize`)

### Screenshots — `GET /tasks/{id}/screenshots` | `GET /tasks/{id}/screenshots/{index}`

### UI States — `GET /tasks/{id}/ui_states` | `GET /tasks/{id}/ui_states/{index}`

### Trajectory — `GET /tasks/{id}/trajectory`

### Models — `GET /models`

### Device Tasks — `GET /devices/{deviceId}/tasks`

---

## Agents — `GET /agents`

Returns pre-configured agent profiles (Default, Instagram, TikTok, X) with model/settings.

## Credentials

- List all: `GET /credentials` (params: `page`, `pageSize`)
- Init package: `POST /credentials/packages` with `{"packageName": "..."}`
- List for package: `GET /credentials/packages/{pkg}`
- Create: `POST /credentials/packages/{pkg}`
- Get: `GET /credentials/packages/{pkg}/credentials/{name}`
- Delete: `DELETE /credentials/packages/{pkg}/credentials/{name}`
- Add field: `POST .../fields` with `{"fieldType": "email|username|password|api_token|phone_number|two_factor_secret", "value": "..."}`
- Update field: `PATCH .../fields/{fieldType}` with `{"value": "..."}`
- Delete field: `DELETE .../fields/{fieldType}`

## Proxy Configs

- List: `GET /proxies`
- Create: `POST /proxies` with `{"name":"...","host":"...","port":1080,"user":"...","password":"...","protocol":"socks5"}`
- Delete: `DELETE /proxies/{proxyId}`

## Webhooks

- List: `GET /hooks`
- Get: `GET /hooks/{id}`
- Subscribe: `POST /hooks/subscribe` with `{"targetUrl": "...", "events": ["completed","failed"]}`
- Edit: `POST /hooks/{id}/edit` with `{"events": [...], "state": "active"}`
- Unsubscribe: `POST /hooks/{id}/unsubscribe`
- Sample: `GET /hooks/sample`

## Feedback — `POST /feedback`

Body: `{"title": "..." (3-100 chars), "feedback": "..." (10-4000 chars), "rating": 1-5}`. Optional: `taskId`.

Auto-submit when tasks fail unexpectedly.
