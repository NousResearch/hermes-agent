# OpenRouter Video Generation (image-to-video)

Vertical/editorial video-from-still for fashion campaigns via the OpenRouter
`/api/v1/videos` async API. Same auth (OpenRouter key) and Drive staging as the
image-skill; the API is asynchronous (submit → poll → download), NOT
request/response like images.

## Key facts (verified 2026-08)

- **Endpoint**: `POST https://openrouter.ai/api/v1/videos` → returns
  `{id, polling_url, status:"pending"}`. Poll `GET polling_url` until
  `completed` / `failed` / `cancelled` / `expired`. Video takes ~1–3+ min.
- **Download**: once completed, GET `unsigned_urls[0]` (an
  `https://openrouter.ai/api/v1/videos/<id>/content?index=0` URL — needs the
  Authorization header). Save as `.mp4`.
- **Request body**:
  ```json
  {
    "model": "minimax/hailuo-3",
    "prompt": "...editorial camera/pic/hair/coat language...",
    "duration": 5,
    "aspect_ratio": "9:16",
    "resolution": "2K",
    "frame_images": [
      {"frame_type": "first_frame", "type": "image_url",
        "image_url": {"url": "https://.../public.png"}}
    ]
  }
  ```
  - `frame_images` entries MUST include `frame_type: "first_frame"|"last_frame"`
    or the API 400s ("Invalid option: expected one of first_frame|last_frame").
  - `duration`/`aspect_ratio`/`resolution` are NOT universally supported —
    check each model via `GET https://openrouter.ai/api/v1/videos/models`
    (`supported_durations`, `supported_aspect_ratios`, `supported_resolutions`,
    `supported_frame_images`, `allowed_passthrough_parameters`).
  - `size` (e.g. "1080x1920") is interchangeable with resolution+aspect_ratio.

- Discount cache: video models are NOT in the default `/api/v1/models` page
  and the page is paginated noisily. Use the dedicated `/api/v1/videos/models`
  endpoint to list them and their exact constraints.

## CRITICAL pitfall — real-person frames

**Seedance blocks people.** `bytedance/seedance-2.0` and `-fast` (and likely
other ByteDance/Seedance video models) REJECT any first/last frame that
contains a recognisable real person:

```
InputImageSensitiveContentDetected.PrivacyInformation
"The request failed because the input image 'content[1]' may contain real person."
```

This is a provider-side safety policy. It CANNOT be overridden — Seedance's
`allowed_passthrough_parameters` is only `['watermark','req_key']` (no
`personGeneration`), and trying `provider.options.parameters.personGeneration`
is silently ignored / still 400s. **Do not keep retrying Seedance with a
person frame** — it will fail every time and burn time.

Workarounds that DO accept a real-person first frame:
- `minimax/hailuo-3` (2K-only, durations 5–15s) — accepts people.
- `google/veo-3.1` / `-fast` / `-lite` — supports `personGeneration` and
  `negativePrompt` via `provider` passthrough (aspect ratio 9:16 supported).
- `kwaivgi/kling-*` — accepts people (has `negative_prompt`, `cfg_scale`).

## hailuo-3 (minimax) specific quirks

- **Only `2K` resolution is accepted.** Sending `"resolution":"720p"`,
  `"resolution":"1080p"`, or `"size":"1080x1920"` → 400
  (`Resolution 720p is not supported ... Supported resolutions: 2K` /
  `Unsupported resolution '1080p'. Supported values: 2K`). Use
  `"resolution":"2K"` and do NOT also set `size`.
- `supported_durations` = [5..15] → minimum 5s (can't do 3s).
- Accepts real people in the first frame (validated end-to-end, produced
  valid 5s 1440x2176 editoral clip at cost ~$0.65).

## Making a first-frame publicly fetchable

OpenRouter providers must fetch the frame via a stable HTTPS URL. A local file
or a non-public Drive share won't work. Reliable pattern:

1. Upload the image to Drive with `google_api.py`:
   `drive upload /local/img.png` → gives a file id.
2. Make it public: `drive share <FILE_ID> --type anyone --role reader`.
3. Use the direct-download URL as `image_url`:
   `https://drive.google.com/uc?export=download&id=<FILE_ID>`
   Verify it returns the image (`curl -sL <url> | file -`) before submitting.

## Workflow

1. Get the still (Drive download via `google_api.py`).
2. Inspect + describe it (vision) to write a specific editorial prompt
   (pose/turn, hand-on-collar, hair flip, coat/belt adjustments, camera drift,
   seamless studio bg).
3. Pick a model that accepts a person frame (not Seedance). With hailuo use
   `resolution:"2K"`.
4. Make the frame publicly fetchable (Drive share).
5. Submit, poll in a background process with `notify_on_complete`, download on
   completion, verify with ffprobe (dimensions/aspect/duration) and a vision
   spot-check of an extracted frame.
6. Upload `.mp4` to the target Drive folder.