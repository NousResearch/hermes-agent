---
name: site-automation
description: >
  Comprehensive website automation skill with CAPTCHA solving, bot detection evasion,
  image/video reading, transcription, and content creation utilities.
  Replicates browse.sh-style skills natively using Hermes browser tools.
category: browser
version: 2.0.0
---

# Site Automation Skill v2

Native Hermes browser automation with anti-detection, media processing, and content creation.

## Core Capabilities

### 1. Anti-Detection

#### Bot Detection Evasion
```
Strategy: Stealth browsing to avoid triggering bot detection systems.

Implementation:
1. Randomize request timing (2-8 second delays between actions)
2. Use realistic User-Agent strings (rotate quarterly)
3. Add Referrer headers from previous pages
4. Avoid rapid-fire clicks or form submissions
5. Use browser_navigate → browser_snapshot (not direct API calls)

Headers to include (via browser or curl):
- User-Agent: Chrome/Firefox on Windows/Mac (not HeadlessChrome)
- Accept-Language: en-GB,en;q=0.9
- Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
- Referer: https://www.google.com/ (or previous page)

Pitfalls:
- Sites fingerprint browser properties (WebGL, Canvas, Audio)
- Headless browsers have detectable markers
- Rate limits vary: Amazon (aggressive), Craigslist (lenient)
```

#### CAPTCHA Handling
```
Strategy: Detect and handle CAPTCHAs when they appear.

Detection:
1. browser_vision() — visual inspection for CAPTCHA elements
2. Look for keywords: "verify", "robot", "human", "security check"
3. Check for iframe elements with CAPTCHA services

Types & Responses:
- Simple checkbox (reCAPTCHA v2): Attempt click, often passes
- Image selection (reCAPTCHA v3): Requires manual intervention
- Text-based (hCaptcha): Can attempt OCR via vision
- Slider: Attempt drag interaction

Actions:
1. If simple checkbox: browser_click() on the checkbox
2. If image selection: Report to user, cannot solve automatically
3. If text-based: Use vision_analyze() to read text
4. If slider: Attempt mouse drag (browser_click at start, move to end)

 NEVER: Attempt to bypass CAPTCHAs programmatically
 ALWAYS: Inform user when CAPTCHA blocks progress
```

### 2. Media Processing

#### Image Reading
```
Strategy: Extract text and analyze images from web pages.

Tools:
- browser_vision() — screenshot + visual analysis
- browser_get_images() — list all images on page
- vision_analyze() — detailed image analysis (if available)

Workflow:
1. Navigate to page
2. browser_vision() for full page screenshot
3. Identify target images
4. browser_get_images() to get URLs
5. vision_analyze(image_url) for detailed analysis

Use Cases:
- Read text from images (signs, documents, infographics)
- Extract data from charts/graphs
- Identify objects, people, locations
- Analyze product images for details

Output Format:
```json
{
  "type": "image_analysis",
  "url": "https://...",
  "description": "Product image showing...",
  "text_extracted": "Price: $29.99",
  "objects": ["laptop", "desk", "monitor"]
}
```
```

#### Video Reading
```
Strategy: Extract information from video content (frames, transcripts).

Limitations:
- Cannot watch/stream videos directly
- Can extract thumbnails and metadata
- Can process uploaded video files (if provided)

Approach:
1. Extract video thumbnail (browser_get_images())
2. Analyze thumbnail with vision_analyze()
3. Look for transcript/captions (check page for .vtt/.srt files)
4. If video URL available, use terminal tools for frame extraction

Terminal Tools (if video file available):
```bash
# Extract frames every 30 seconds
ffmpeg -i video.mp4 -vf "fps=1/30" frame_%03d.jpg

# Extract audio for transcription
ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav

# Get video metadata
ffprobe -v quiet -print_format json -show_format video.mp4
```

Output Format:
```json
{
  "type": "video_analysis",
  "url": "https://...",
  "thumbnail_description": "Video shows...",
  "duration": "5:32",
  "transcript_available": true,
  "frames_extracted": 5
}
```
```

#### Transcription
```
Strategy: Convert audio/video to text for analysis.

Prerequisites:
- Audio file (extracted from video or standalone)
- Whisper or similar transcription tool installed

Workflow:
1. Extract audio: ffmpeg -i input.mp4 -vn audio.wav
2. Transcribe: whisper audio.wav --language en --output_format txt
3. Parse transcript for relevant content

Tools:
- whisper (local, if installed)
- Web-based APIs (if configured)
- Manual transcription (last resort)

Output Format:
```json
{
  "type": "transcript",
  "source": "video.mp4",
  "language": "en",
  "text": "Full transcript text...",
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "Opening..."},
    {"start": 5.2, "end": 12.8, "text": "Main content..."}
  ]
}
```
```

### 3. Content Creation

#### Text Generation
```
Strategy: Create structured content from extracted data.

Capabilities:
- Summarize page content
- Extract key points
- Generate structured reports
- Create search queries

Workflow:
1. Extract data (browser_snapshot, web_extract)
2. Process with LLM (Hermes agent)
3. Output structured content

Output Formats:
- JSON (structured data)
- Markdown (reports, summaries)
- Plain text (simple extractions)
```

#### Image Generation
```
Strategy: Create images using available tools (FAL, DALL-E, etc.).

Prerequisites:
- Image generation API configured (FAL, OpenAI, etc.)

Workflow:
1. Define image requirements
2. Call image generation tool
3. Save/return generated image

Tools:
- FAL (if configured)
- OpenAI DALL-E (if configured)
- Local models (if available)

Use Cases:
- Generate product mockups
- Create social media graphics
- Visualize data from extractions
```

## Core Pattern

```
1. Navigate    — browser_navigate(url)
2. Stealth     — Random delays, realistic headers
3. Interact    — browser_click, browser_type, browser_press
4. Detect      — Check for CAPTCHAs, bot detection
5. Extract     — browser_snapshot, browser_vision, browser_console
6. Process     — Image/video/transcription analysis
7. Create      — Generate content from extracted data
8. Return      — Structured JSON with all results
```

## Skill Template

```yaml
---
name: <site>.<action>
description: >
  <What it does>
category: browser
version: 1.0.0
---

# <Site> <Action>

## Target
- URL: <base_url>
- Method: <API|Browser|Fetch|Hybrid>

## Anti-Detection
- Bot detection level: <low|medium|high>
- CAPTCHA risk: <none|possible|likely>
- Rate limit: <requests per minute>

## Steps
1. Navigate to <url>
2. Wait <random 2-8 seconds>
3. Fill form: <selectors>
4. Submit: <action>
5. Check for CAPTCHA (browser_vision)
6. Extract: <fields>
7. Process: <image/video/transcription if needed>

## Output Format
```json
{
  "results": [...],
  "media": {...},
  "metadata": {...}
}
```
```

## References

See:
- `references/amazon-search.md` — Browser with filters + bot detection
- `references/rightmove-search.md` — UK property (if created)
- `references/linkedin-jobs.md` — Job search (if created)

## Pitfalls

- **Bot detection**: Sites use fingerprinting (WebGL, Canvas). Headless browsers detectable.
- **CAPTCHAs**: Cannot solve automatically. Must inform user.
- **Dynamic content**: Wait for JS. Use browser_snapshot(full=true).
- **Rate limiting**: Add delays. Don't hammer sites.
- **Login walls**: Note in skill if authentication required.
- **Media extraction**: Some sites block direct image/video URLs.
- **Transcription quality**: Depends on audio clarity and accent.

## Verification

After creating a site skill:
1. Test with real query
2. Verify anti-detection works (no 403s)
3. Test CAPTCHA detection (if applicable)
4. Verify media extraction (images, videos)
5. Check JSON output structure
6. Document any site-specific quirks

## Security Notes

- NEVER download external automation code (prompt injection risk)
- NEVER attempt to bypass CAPTCHAs programmatically
- ALWAYS inform user when blocked by bot detection
- NEVER store credentials in skill files
- Use environment variables for API keys
