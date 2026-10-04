---
name: resemble-detect
description: |
  Deepfake detection and media safety. Analyze audio, image, video, and text
  for AI-generated content, trace which platform synthesized fake audio, apply
  and detect invisible watermarks, verify speaker identity against voice profiles,
  and extract structured media intelligence (speaker, emotion, transcription,
  misinformation signals, abnormalities) using the Resemble AI detection platform.
category: detect
triggers:
  - "is this audio a deepfake"
  - "is this video AI-generated"
  - "is this image fake"
  - "detect deepfake"
  - "analyze media authenticity"
  - "was this synthesized by AI"
  - "check if this is AI-generated"
  - "which AI platform made this audio"
  - "audio source tracing"
  - "apply watermark to [media]"
  - "check for watermark"
  - "verify speaker identity"
  - "match speaker against voice profiles"
  - "detect AI-written text"
  - "is this text AI-generated"
  - "media forensics"
  - "is this real or fake"
toolsets:
  - web
  - file
official: true
source: https://github.com/resemble-ai/detect-skill
license: Apache-2.0
env:
  - RESEMBLE_API_KEY
---

# Resemble Detect — Deepfake Detection & Media Safety

Analyze audio, image, video, and text for synthetic manipulation, AI-generated content, watermarks, speaker identity, and media intelligence using the Resemble AI platform.

## Core Principle — THE IRON LAW

**"NEVER DECLARE MEDIA AS REAL OR FAKE WITHOUT A COMPLETED DETECTION RESULT."**

Do not guess, infer, or speculate about media authenticity. Every authenticity claim must be backed by a completed Resemble detect job with a returned `label`, `score`, and `status: "completed"`. If the detection is still `processing`, wait. If it `failed`, say so — do not substitute your own judgment.

## When to Use

Use this skill whenever the user's request involves any of these:

- Checking if audio, video, image, or text is AI-generated or manipulated
- Detecting deepfakes in any media format
- Verifying media authenticity or provenance
- Identifying which AI platform synthesized audio (source tracing)
- Applying or detecting watermarks on media
- Analyzing media for speaker info, emotion, transcription, or misinformation
- Asking natural-language questions about detection results
- Matching or verifying speaker identity against known voice profiles
- Detecting AI-generated or machine-written text
- Any mention of: "deepfake", "fake detection", "synthetic media", "voice verification", "watermark", "media forensics", "authenticity check", "source tracing", "is this real", "AI-written text", "text detection"

**Do NOT use** for text-to-speech generation, voice cloning, or speech-to-text transcription — those are separate Resemble capabilities.

## Capability Decision Tree

| User wants to...                                      | Use this                  | API endpoint               |
|-------------------------------------------------------|---------------------------|----------------------------|
| Check if media is AI-generated / deepfake             | **Deepfake Detection**    | `POST /detect`             |
| Know *which AI platform* made fake audio              | **Audio Source Tracing**  | `POST /detect` with flag   |
| Get speaker info, emotion, transcription from media   | **Intelligence**          | `POST /intelligence`       |
| Ask questions about a completed detection             | **Detect Intelligence**   | `POST /detects/{uuid}/intelligence` |
| Apply an invisible watermark to media                 | **Watermark Apply**       | `POST /watermark/apply`    |
| Check if media contains a watermark                   | **Watermark Detect**      | `POST /watermark/detect`   |
| Verify a speaker's identity against known profiles    | **Identity Search**       | `POST /identity/search`    |
| Check if text is AI-generated                         | **Text Detection**        | `POST /text_detect`        |
| Create a voice identity profile for future matching   | **Identity Create**       | `POST /identity`           |

When multiple capabilities apply (e.g., user wants deepfake detection AND intelligence), combine them in a single `POST /detect` call using the `intelligence: true` flag rather than making separate requests.

## Required Setup

- **API Key**: Bearer token from the Resemble AI dashboard, exposed as `RESEMBLE_API_KEY`
- **Base URL**: `https://app.resemble.ai/api/v2`
- **Auth Header**: `Authorization: Bearer <RESEMBLE_API_KEY>`
- **Media Requirement**: All media must be at a publicly accessible HTTPS URL

If the user provides a local file path instead of a URL, inform them the file must be hosted at a public HTTPS URL first. Do not attempt to upload local files to the API.

## MCP Tools Available

When the Resemble MCP server is connected, use these tools instead of raw API calls:

| Tool                      | Purpose                                           |
|---------------------------|---------------------------------------------------|
| `resemble_docs_lookup`    | Get comprehensive docs for any detect sub-topic   |
| `resemble_search`         | Search across all documentation                   |
| `resemble_api_endpoint`   | Get exact OpenAPI spec for any endpoint            |
| `resemble_api_search`     | Find endpoints by keyword                         |
| `resemble_get_page`       | Read specific documentation pages                 |
| `resemble_list_topics`    | List all available topics                         |

**Tool usage pattern**: Use `resemble_docs_lookup` with topic `"detect"` to get the full picture, then `resemble_api_endpoint` for exact request/response schemas before making API calls.

---

## Phase 1: Deepfake Detection

The core capability. Submit any audio, image, or video for AI-generated content analysis.

### Submit a Detection

```
POST /detect
Content-Type: application/json
Authorization: Bearer <API_KEY>

{
  "url": "https://example.com/media.mp4",
  "visualize": true,
  "intelligence": true,
  "audio_source_tracing": true
}
```

**Parameters:**

| Parameter              | Type    | Required | Description                                              |
|------------------------|---------|----------|----------------------------------------------------------|
| `url`                  | string  | Yes      | HTTPS URL to audio, image, or video file                 |
| `callback_url`         | string  | No       | Webhook URL for async completion notification             |
| `visualize`            | boolean | No       | Generate heatmap/visualization artifacts                  |
| `intelligence`         | boolean | No       | Run multimodal intelligence analysis alongside detection  |
| `audio_source_tracing` | boolean | No       | Identify which AI platform synthesized fake audio         |
| `frame_length`         | integer | No       | Audio/video analysis window size in seconds (1–4, default 2) |
| `start_region`         | number  | No       | Start of segment to analyze (seconds)                    |
| `end_region`           | number  | No       | End of segment to analyze (seconds)                      |
| `model_types`          | string  | No       | `"image"` or `"talking_head"` (for face-swap detection)  |
| `use_reverse_search`   | boolean | No       | Enable reverse image search (image only)                 |
| `use_ood_detector`     | boolean | No       | Enable out-of-distribution detection                     |
| `zero_retention_mode`  | boolean | No       | Auto-delete media after detection completes              |

**Supported formats:**
- Audio: WAV, MP3, OGG, M4A, FLAC
- Video: MP4, MOV, AVI, WMV
- Image: JPG, PNG, GIF, WEBP

### Poll for Results

Detection is asynchronous. Poll `GET /detect/{uuid}` until `status` is `"completed"` or `"failed"`.

```
GET /detect/{uuid}
Authorization: Bearer <API_KEY>
```

**Polling best practice:** Start at 2s intervals, back off to 5s, then 10s. Most detections complete within 10–60 seconds depending on media length.

### Reading Results by Media Type

**Audio results** — in `metrics`:
```json
{
  "label": "fake",
  "score": ["0.92", "0.88", "0.95"],
  "consistency": "0.91",
  "aggregated_score": "0.92",
  "image": "https://..."
}
```
- `label`: `"fake"` or `"real"` — the verdict
- `score`: Per-chunk prediction scores (array)
- `aggregated_score`: Overall confidence (0.0–1.0, higher = more likely synthetic)
- `consistency`: How consistent the prediction is across chunks
- `image`: Visualization heatmap URL (if `visualize: true`)

**Image results** — in `image_metrics`:
```json
{
  "type": "ImageAnalysis",
  "label": "fake",
  "score": 0.87,
  "image": "https://...",
  "ifl": { "score": 0.82, "heatmap": "https://..." },
  "reverse_image_search_sources": [
    { "url": "...", "title": "...", "verdict": "known_fake", "similarity": 0.95 }
  ]
}
```
- `label` / `score`: Verdict and confidence
- `ifl`: Invisible Frequency Layer analysis with heatmap
- `reverse_image_search_sources`: Known sources found online (if `use_reverse_search: true`)

**Video results** — in `video_metrics`:
```json
{
  "label": "fake",
  "score": 0.89,
  "certainty": 0.91,
  "children": [
    {
      "type": "VideoResult",
      "conclusion": "Fake",
      "score": 0.89,
      "timestamp": 2.5,
      "children": [...]
    }
  ]
}
```
- Hierarchical tree of frame-level and segment-level results
- Each child has `timestamp`, `score`, `certainty`, and may have nested `children`
- Video with audio track returns both `metrics` (audio) and `video_metrics` (visual)

### Interpreting Scores

| Score Range | Interpretation                                      |
|-------------|-----------------------------------------------------|
| 0.0 – 0.3   | Strong indication of authentic/real media           |
| 0.3 – 0.5   | Inconclusive — recommend additional analysis        |
| 0.5 – 0.7   | Likely synthetic — flag for review                  |
| 0.7 – 1.0   | High confidence synthetic/AI-generated              |

**Always present scores with context.** Say "The detection returned a score of 0.87, indicating high confidence that this audio is AI-generated" — never just "it's fake."

---

## Phase 2: Intelligence — Media Analysis

Analyze media for rich structured insights independent of or alongside detection.

### Standalone Intelligence

```
POST /intelligence
Content-Type: application/json
Authorization: Bearer <API_KEY>

{
  "url": "https://example.com/audio.mp3",
  "json": true
}
```

**Parameters:**

| Parameter      | Type    | Required | Description                                              |
|----------------|---------|----------|----------------------------------------------------------|
| `url`          | string  | One of   | HTTPS URL to media file                                  |
| `media_token`  | string  | One of   | Token from secure upload (alternative to URL)            |
| `detect_id`    | string  | No       | UUID of existing detect to associate                     |
| `media_type`   | string  | No       | `"audio"`, `"video"`, or `"image"` (auto-detected)      |
| `json`         | boolean | No       | Return structured fields (default: false for audio/video, true for image) |
| `callback_url` | string  | No       | Webhook for async mode                                   |

**Audio/Video structured response** (`json: true`):
- `speaker_info` — speaker description (age, gender)
- `language` / `dialect` — detected language
- `emotion` — detected emotional state
- `speaking_style` — conversational, formal, etc.
- `context` — inferred context of the speech
- `message` — content summary
- `abnormalities` — anomalies detected in the media
- `transcription` — full transcript
- `translation` — translation if non-English
- `misinformation` — misinformation analysis

**Image structured response:**
- `scene_description` — what the image shows
- `subjects` — people/objects identified
- `authenticity_analysis` — visual authenticity assessment
- `context_and_setting` — environment description
- `abnormalities` — visual anomalies
- `misinformation` — misinformation analysis

### Detect Intelligence — Ask Questions About Results

After a detection completes, ask natural-language questions about it:

```
POST /detects/{detect_uuid}/intelligence
Content-Type: application/json
Authorization: Bearer <API_KEY>

{
  "query": "How confident is the model that this audio is fake?"
}
```

This returns a question UUID. Poll `GET /detects/{detect_uuid}/intelligence/{question_uuid}` until `status` is `"completed"` to get the `answer`.

**Good questions to suggest:**
- "Summarize the detection results in plain language"
- "What specific indicators suggest this is AI-generated?"
- "How do the audio and video detection results differ?"
- "What is the confidence level and what does it mean?"
- "Are there any inconsistencies in the analysis?"

**Status flow:** `pending` → `processing` → `completed` (or `failed`)

**Prerequisite:** The detection must have `status: "completed"`. Submitting a question against a processing or failed detection returns a 422 error.

---

## Phase 3: Audio Source Tracing

When audio is detected as synthetic (`label: "fake"`), identify which AI platform generated it.

**Enable it** by setting `audio_source_tracing: true` in the `POST /detect` request.

**Result** appears in the detection response under `audio_source_tracing`:
```json
{
  "label": "elevenlabs",
  "error_message": null
}
```

Known source labels include: `resemble_ai`, `elevenlabs`, `real`, and others as the model expands.

**Important:** Source tracing only runs when audio is labeled as `"fake"`. If the audio is `"real"`, no source tracing result will appear.

**Standalone query:**
- `GET /audio_source_tracings` — list all source tracing reports
- `GET /audio_source_tracings/{uuid}` — get specific report

---

## Phase 4: Watermarking

Apply invisible watermarks to media for provenance tracking, or detect existing watermarks.

### Apply a Watermark

```
POST /watermark/apply
Content-Type: application/json
Authorization: Bearer <API_KEY>
Prefer: wait

{
  "url": "https://example.com/image.png",
  "strength": 0.3,
  "custom_message": "my-organization"
}
```

| Parameter        | Type   | Required | Description                                          |
|------------------|--------|----------|------------------------------------------------------|
| `url`            | string | Yes      | HTTPS URL to media file                              |
| `strength`       | number | No       | Watermark strength 0.0–1.0 (image/video only, default 0.2) |
| `custom_message` | string | No       | Custom message to embed (image/video only, default "resembleai") |

- Add `Prefer: wait` header for synchronous response
- Without it, poll `GET /watermark/apply/{uuid}/result`
- Response includes `watermarked_media` URL to download the watermarked file

### Detect a Watermark

```
POST /watermark/detect
Content-Type: application/json
Authorization: Bearer <API_KEY>
Prefer: wait

{
  "url": "https://example.com/suspect-image.png"
}
```

**Audio detection result:**
```json
{ "has_watermark": true, "confidence": 0.95 }
```

**Image/Video detection result:**
```json
{ "has_watermark": true }
```

---

## Phase 5: Identity — Speaker Verification (Beta)

Create voice identity profiles and match incoming audio against them.

> **Beta feature** — requires joining the preview program. Inform the user if they encounter access errors.

### Create an Identity Profile

```
POST /identity
Content-Type: application/json
Authorization: Bearer <API_KEY>

{
  "audio_url": "https://example.com/known-speaker.wav",
  "name": "Jane Doe"
}
```

### Search Against Known Identities

```
POST /identity/search
Content-Type: application/json
Authorization: Bearer <API_KEY>

{
  "audio_url": "https://example.com/unknown-speaker.wav",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "item": [
    { "uuid": "...", "name": "Jane Doe", "confidence": 0.92, "distance": 0.08 }
  ]
}
```

Lower `distance` = closer match. Higher `confidence` = stronger match.

---

## Phase 6: Text Detection

Detect whether text content is AI-generated or human-written.

> **Beta feature** — requires the `detect_beta_user` role or a billing plan that includes the `dfd_text` product.

### Submit a Text Detection

```
POST /text_detect
Content-Type: application/json
Authorization: Bearer <API_KEY>
```

Add the `Prefer: wait` header for a synchronous (blocking) response. Without it, the job runs asynchronously — poll or use a callback.

**Parameters:**

| Parameter      | Type    | Required | Description                                              |
|----------------|---------|----------|----------------------------------------------------------|
| `text`         | string  | Yes      | Text to analyze (max 100,000 characters)                 |
| `thinking`     | string  | No       | Always use `"low"` (default)                             |
| `threshold`    | float   | No       | Decision threshold 0.0–1.0 (default: 0.5)                |
| `callback_url` | string  | No       | Webhook URL for async completion notification            |
| `privacy_mode` | boolean | No       | If true, text content is not stored after analysis       |

**Response:**
```json
{
  "success": true,
  "item": {
    "uuid": "abc-123",
    "status": "completed",
    "prediction": "ai",
    "confidence": 0.91,
    "text_content": "This is some text to analyze.",
    "privacy_mode": false,
    "created_at": "...",
    "updated_at": "..."
  }
}
```

- `prediction`: `"ai"` or `"human"` — the verdict
- `confidence`: 0.0–1.0, higher = more confident in the prediction
- `status`: `"processing"`, `"completed"`, or `"failed"`

### Poll for Results

If you did not use `Prefer: wait`, poll until `status` is `"completed"` or `"failed"`:

```
GET /text_detect/{uuid}
Authorization: Bearer <API_KEY>
```

---

## Recommended Workflows

### Full Media Forensics (Most Thorough)

For a comprehensive analysis, combine all capabilities:

1. Submit detection with all flags enabled:
   ```json
   {
     "url": "https://example.com/suspect.mp4",
     "visualize": true,
     "intelligence": true,
     "audio_source_tracing": true,
     "use_reverse_search": true
   }
   ```
2. Poll until `status: "completed"`
3. Read `metrics` / `image_metrics` / `video_metrics` for the verdict
4. Read `intelligence.description` for structured media analysis
5. If audio labeled `"fake"`, check `audio_source_tracing.label` for the source platform
6. Ask follow-up questions via Detect Intelligence if anything needs clarification
7. Check for watermarks via `POST /watermark/detect` if provenance is relevant

### Quick Authenticity Check (Fastest)

For a fast pass/fail:

1. Submit minimal detection: `{ "url": "..." }`
2. Poll until complete
3. Check `label` and `aggregated_score` (audio) or `label` and `score` (image/video)
4. Report result with score context

### Provenance Pipeline (Content Creators)

For creators who want to prove their content is authentic:

1. Apply watermark to original content: `POST /watermark/apply`
2. Distribute watermarked media
3. Later, verify provenance: `POST /watermark/detect` against any copy

---

## Red Flags — Stop and Reassess

- **Declaring authenticity without a detection result** — Never say media is real or fake based on visual/auditory inspection alone
- **Ignoring the score and reporting only the label** — A `"fake"` label with score 0.51 means something very different from score 0.95
- **Submitting local file paths to the API** — The API requires publicly accessible HTTPS URLs (does not apply to text detection)
- **Sending text longer than 100,000 characters to text detection** — Split into chunks or inform the user of the limit
- **Polling too aggressively** — Start at 2s intervals, back off exponentially; do not loop at <1s
- **Asking Detect Intelligence questions before detection completes** — Results in 422 error
- **Expecting source tracing on "real" audio** — Source tracing only runs on audio labeled `"fake"`
- **Treating beta features (Identity, Text Detection) as production-ready** — Warn users about beta status
- **Ignoring `zero_retention_mode` for sensitive media** — Always suggest this flag when the user indicates the media is sensitive or private
- **Making multiple separate API calls when flags can combine** — Use `intelligence: true` and `audio_source_tracing: true` on the detection call instead of separate requests

## Response Presentation Guidelines

When presenting results to users:

1. **Lead with the verdict** — "The detection indicates this audio is likely AI-generated (score: 0.87)"
2. **Provide score context** — Use the score interpretation table above
3. **Mention limitations** — Detection is probabilistic, not absolute proof
4. **Include actionable next steps** — Suggest intelligence queries, source tracing, or watermark checks as appropriate
5. **For inconclusive results (0.3–0.5)** — Explicitly state the result is inconclusive and recommend additional analysis with different parameters or manual review
6. **Never present detection as legal evidence** — Detection results are analytical tools, not forensic certifications

## Error Handling

| Error     | Cause                                    | Resolution                                      |
|-----------|------------------------------------------|--------------------------------------------------|
| 400       | Invalid request body or missing `url`    | Check required parameters                        |
| 401       | Invalid or missing API key               | Verify `RESEMBLE_API_KEY`                        |
| 404       | Detection UUID not found                 | Verify the UUID from the creation response       |
| 422       | Detection not completed (for Intelligence) | Wait for detection to reach `completed` status |
| 429       | Rate limited                             | Back off and retry with exponential delay        |
| 500       | Server error                             | Retry once, then report to user                  |

## Privacy & Compliance Notes

- **Zero retention mode**: Set `zero_retention_mode: true` to auto-delete media after analysis. The URL is redacted and `media_deleted` is set to true post-completion.
- **Text privacy mode**: Set `privacy_mode: true` on text detection to prevent text content from being stored after analysis.
- **Data handling**: Media URLs and text content are stored by default. For GDPR/compliance-sensitive workflows, enable zero retention (media) or privacy mode (text).
- **Callback security**: If using `callback_url`, ensure the endpoint is HTTPS and authenticated on the receiving end.

## Source

- **GitHub**: https://github.com/resemble-ai/detect-skill
- **Platform**: https://resemble.ai
- **API Docs**: https://docs.resemble.ai
- **License**: Apache-2.0
