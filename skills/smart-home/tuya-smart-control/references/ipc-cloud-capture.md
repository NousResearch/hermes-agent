# IPC AI — Cloud Capture

Cloud snapshot / short video capture for IPC (camera) devices.

> **Capture flow**: `allocate` requests the device to capture and upload → wait a few seconds → poll `resolve` until the media URL is ready. **Do not** call resolve immediately after allocate.

## 1. Allocate Cloud Capture

Request the device to take a snapshot or record a short video and upload it to cloud storage. Returns storage coordinates only — no media URL.

**Request**

```
POST /v1.0/end-user/ipc/{device_id}/capture/allocate
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| device_id | String | Yes | Device ID |

**Request Body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| capture_json | String | Yes | JSON string containing capture parameters |

**`capture_json` Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| device_id | String | Yes | Device ID (must match path parameter) |
| capture_type | String | Yes | `PIC` for snapshot, `VIDEO` for short video |
| pic_count | Integer | No | Number of snapshots. Server clamps to **1–5** |
| video_duration_seconds | Integer | No | Video duration in seconds. Default 10, server clamps to **1–60** |
| home_id | String | No | Home ID. If provided, must match the device owner |

**Request Example**

```json
{
  "capture_json": "{\"device_id\":\"6c95a7a3...\",\"capture_type\":\"PIC\",\"pic_count\":1}"
}
```

**Response**

```json
{
  "success": true,
  "result": {
    "success": true,
    "status": "ACCEPTED",
    "device_id": "6c95a7a3...",
    "capture_type": "PIC",
    "bucket": "ty-cn-storage30",
    "image_object_key": "/path/to/1234567890_0.jpg",
    "encryption_key": "abc123...",
    "upload_async_hint": "bucket/objectKey only mean upload slots were allocated; the device may still be uploading.",
    "pic_count_requested": 1,
    "pic_count_effective": 1,
    "pic_slots": [
      {
        "success": true,
        "image_object_key": "/path/to/1234567890_0.jpg",
        "encryption_key": "abc123..."
      }
    ]
  }
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| success | Boolean | Whether allocation succeeded |
| status | String | `ACCEPTED` or `REJECTED` |
| bucket | String | Cloud storage bucket name |
| image_object_key | String | Image object key (PIC) |
| video_object_key | String | Video object key (VIDEO) |
| cover_image_object_key | String | Video cover image key (VIDEO) |
| encryption_key | String | File encryption key |
| upload_async_hint | String | Reminder that file may still be uploading |
| pic_slots | List | Per-snapshot details (PIC only) |
| video_duration_seconds_effective | Integer | Actual clamped duration (VIDEO only) |

---

## 2. Resolve Capture Access URL

Poll this API after allocate to obtain accessible media URLs. Returns `NOT_READY` while the device is still uploading.

**Request**

```
POST /v1.0/end-user/ipc/{device_id}/capture/resolve
```

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| device_id | String | Yes | Device ID |

**Request Body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| resolve_json | String | Yes | JSON string containing resolve parameters |

**`resolve_json` Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| device_id | String | Yes | Device ID |
| capture_type | String | Yes | `PIC` or `VIDEO` (must match allocate) |
| bucket | String | Yes | Bucket from allocate response |
| image_object_key | String | PIC: Yes | Image object key from allocate |
| video_object_key | String | VIDEO: Yes | Video object key from allocate |
| cover_image_object_key | String | No | Cover image key (VIDEO only) |
| encryption_key | String | No | Encryption key from allocate |
| user_privacy_consent_accepted | Boolean | No | `true`: return decrypted playable URLs; `false`: return raw presigned URLs (encrypted) |
| home_id | String | No | Home ID |

**Request Example**

```json
{
  "resolve_json": "{\"device_id\":\"6c95a7a3...\",\"capture_type\":\"PIC\",\"bucket\":\"ty-cn-storage30\",\"image_object_key\":\"/path/to/1234567890_0.jpg\",\"encryption_key\":\"abc123...\",\"user_privacy_consent_accepted\":true}"
}
```

**Response (ready)**

```json
{
  "success": true,
  "result": {
    "status": "ACCEPTED",
    "decrypt_image_url": "https://...",
    "message_for_user": "ok"
  }
}
```

**Response (not ready — keep polling)**

```json
{
  "success": true,
  "result": {
    "status": "NOT_READY",
    "error_code": "OBJECT_NOT_READY",
    "message_for_user": "object not uploaded yet or empty (check contentLength)"
  }
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| status | String | `ACCEPTED` (URL ready), `NOT_READY` (keep polling) |
| decrypt_image_url | String | Decrypted playable image URL (when consent = true, PIC) |
| decrypt_video_url | String | Decrypted playable video URL (when consent = true, VIDEO) |
| decrypt_cover_image_url | String | Decrypted cover image URL (VIDEO). May be null if cover is not ready yet — check message_for_user |
| raw_presigned_image_url | String | Raw presigned URL (when consent = false, PIC) |
| raw_presigned_video_url | String | Raw presigned URL (when consent = false, VIDEO) |
| raw_presigned_cover_image_url | String | Raw presigned cover URL (VIDEO). May be null if cover is not ready yet |
| error_code | String | `OBJECT_NOT_READY` when file not yet uploaded |
| message_for_user | String | `ok` when all ready; `ok (cover image not ready yet)` when video ready but cover still uploading |

---

## 3. Fusion-level errors (`success: false` on the outer envelope)

Both capture endpoints are implemented by Hawkeye **`IIpcAiCaptureOpenFusionService`**.

Empty/invalid JSON or missing required fields cause Atop to **throw HawkeyeException**; business exceptions from the Biz layer also propagate as exceptions. All exceptions are caught and mapped to **`FusionResult.buildFail`** by **`HawkeyeFusionService`**. The outer **`code` / `msg`** are determined by the exception type and the Fusion / gateway layer (commonly numeric API error numbers such as **`1109`** for bad input).

---

## 4. Recommended Usage

1. **Capture**: call `allocate` → wait a few seconds → poll `resolve` until `status` is not `NOT_READY`.
2. **Video resolve**: when `status` is `ACCEPTED`, video URL is guaranteed. Cover image URL may be null if still uploading — check `message_for_user` for `"ok (cover image not ready yet)"`. You can call resolve again later to get the cover URL.
3. **Privacy**: only set `user_privacy_consent_accepted=true` when the user has explicitly agreed.
4. **Outer failures**: if `success` is `false`, read **`code`** and **`msg`** per §3. Parameter errors and Biz exceptions all produce outer `success: false` with a numeric error code.

---

## 5. Pic Capture: Wait and Retry

The device must finish capturing and uploading before resolve can return an image URL.

- **Initial wait**: sleep **2 seconds** before the first resolve call.
- **Polling**: call resolve every **~2 seconds** until an image URL appears or the timeout (default **30 seconds**) elapses.
- **Retries after timeout**: if polling times out without a URL, retry resolve up to **3 more times** at **3-second intervals**.

**Helper methods** (`scripts/tuya_api.py`):

| Method | Description |
|--------|-------------|
| `ipc_ai_capture_pic_resolve_with_wait(...)` | Wait, poll, and retry resolve given allocate results |
| `ipc_ai_capture_pic_allocate_and_fetch(...)` | Allocate PIC then automatically wait and resolve |

**CLI example**:

```bash
python3 scripts/tuya_api.py ipc_pic_fetch <device_id> 1
# consent=1 (decrypted URL); optional 3rd arg: pic_count; optional 4th: home_id
```

---

## 6. Video Capture: Wait and Retry

The device must finish recording and uploading before resolve can return a playable URL.

- **Minimum wait**: before the first resolve call, wait at least **`max(5, video_duration_seconds_effective) + 2` seconds**.
- **Polling**: call resolve every **~2 seconds** until a video URL appears or polling reaches the timeout (default **120 seconds**).
- **Retries after timeout**: if polling times out without a URL, retry resolve up to **3 more times** at **5-second intervals**.

**Helper methods** (`scripts/tuya_api.py`):

| Method | Description |
|--------|-------------|
| `ipc_ai_capture_video_resolve_with_wait(...)` | Wait, poll, and retry resolve given allocate results |
| `ipc_ai_capture_video_allocate_and_fetch(...)` | Allocate VIDEO then automatically wait and resolve |

**CLI example**:

```bash
python3 scripts/tuya_api.py ipc_video_fetch <device_id> 5 1
# 5-second video, consent=1 (decrypted URL); optional 4th arg: home_id
```
