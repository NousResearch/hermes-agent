---
name: youtube-transcript-extract
description: >
  Extract full metadata and timestamped transcripts from YouTube videos via the
  public InnerTube API + oEmbed. No API keys required. Returns video metadata,
  human-authored or auto-generated captions, and timestamped segments.
  No side effects: never likes, comments, subscribes, or watches videos.
category: browser
version: 1.0.0
---

# YouTube Transcript Extraction

## Target
- URL: https://www.youtube.com/watch?v=<VIDEO_ID>
- Method: Fetch (curl + web_extract)
- Bot detection: Low (uses API endpoints, not UI scraping)

## Capabilities
- Extract video title, channel, duration
- Extract full timestamped transcript
- Detect if captions are auto-generated or human-authored
- Handle all YouTube URL formats (watch, short, embed, youtu.be)

## Anti-Detection
- Bot detection level: Low
- CAPTCHA risk: None (API endpoints)
- Rate limit: ~30 requests/hour per IP

## Input
Accept any of:
- `https://www.youtube.com/watch?v=<ID>`
- `https://youtu.be/<ID>`
- `https://www.youtube.com/shorts/<ID>`
- `https://www.youtube.com/embed/<ID>`
- Bare 11-character video ID (regex: `[A-Za-z0-9_-]{11}`)

## Workflow

### Step 1: Normalize to Video ID
Strip all query parameters except `v=` from URL forms. Reject if no valid 11-char ID.

### Step 2: Fetch Basic Metadata via oEmbed
```bash
curl -s "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=<VIDEO_ID>&format=json"
```
Returns: `title`, `author_name`, `author_url`, `thumbnail_url`
- 404 → video private/deleted → stop
- 401 → video exists but embedding disabled → continue to InnerTube

### Step 3: Fetch Caption Tracks via InnerTube
```bash
curl -s -X POST "https://www.youtube.com/youtubei/v1/player?prettyPrint=false" \
  -H "Content-Type: application/json" \
  -H "Origin: https://www.youtube.com" \
  -d '{
    "context": {
      "client": {
        "clientName": "ANDROID",
        "clientVersion": "19.09.37",
        "androidSdkVersion": 30,
        "hl": "en",
        "gl": "US"
      }
    },
    "videoId": "<VIDEO_ID>"
  }'
```

Key response fields:
- `playabilityStatus.status` — OK / ERROR / UNPLAYABLE
- `videoDetails.title` — video title
- `videoDetails.author` — channel name
- `videoDetails.lengthSeconds` — duration
- `captions.playerCaptionsTracklistRenderer.captionTracks` — available captions
  - `baseUrl` — URL to fetch transcript
  - `name.simpleText` — language name
  - `vssId` — ".en" for human, "a.en" for auto-generated
  - `kind` — "asr" if auto-generated
  - `languageCode` — e.g., "en"

If ANDROID fails, retry with `IOS` client (swap `clientName` to `"IOS"`, `clientVersion` to `"19.09.3"`).

### Step 4: Fetch Transcript
```
curl -s "<captionTracks[0].baseUrl>"
```
Returns XML with `<text start="seconds" dur="duration">text</text>` segments.

### Step 5: Parse and Return
```json
{
  "success": true,
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "channel": "Rick Astley",
  "duration_seconds": 213,
  "captions_available": true,
  "is_auto_generated": false,
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "duration": 5.2,
      "text": "We're no strangers to love"
    },
    {
      "start": 5.2,
      "duration": 4.8,
      "text": "You know the rules and so do I"
    }
  ],
  "full_transcript": "We're no strangers to love You know the rules and so do I..."
}
```

## Error Handling

| Condition | Response |
|-----------|----------|
| Video private/deleted | `success: false, reason: "video_unavailable"` |
| No captions available | `success: true, captions_available: false` |
| Live stream (no replay) | `success: false, reason: "live_stream_no_captions"` |
| InnerTube both clients fail | `success: false, reason: "inner_api_failure"` |

## Integration Points

- **Content pipeline**: Transcript → summary → blog post → social thread
- **Research**: Transcript → key insights → note in Mnemosyne
- **Job hunt**: Transcript of tech talks/company presentations → prep material
- **Football**: Transcript of match analysis → content for MatchdayMaestro

## Pitfalls

- InnerTube ANDROID client may require periodic clientVersion updates
- Some videos have embedding disabled (oEmbed 401) — proceed to InnerTube
- Auto-generated captions have lower accuracy than human-authored
- Max 30+ req/hr per IP before YouTube rate blocks
- Live streams return no captions until VOD is processed
- Always fetch caption in original language first; translation quality varies

## Verification

1. Test with public video (standard upload)
2. Test with auto-generated captions video
3. Test with no-captions video
4. Test with private/deleted video
5. Verify timestamp accuracy matches video length
