# WaveSpeed model reference

Per-model inputs, pricing and prompt tips, condensed from the WaveSpeedAI per-model skills (https://github.com/WaveSpeedAI/agent-skills). Field names are the ones `wavespeed run <model> -h` prints; prices are list prices at the time of writing and `wavespeed price` is authoritative.

## Seedream V4.5 Image Generation/Editing

**Model ID:** `bytedance/seedream-v4.5`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | -- | Text description of the image to generate |
| `size` | string | No | `2048*2048` | Output size in pixels (`WIDTH*HEIGHT`). Each dimension: 1024-4096. |

**Model ID:** `bytedance/seedream-v4.5/edit`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `images` | string[] | Yes | `[]` | URLs of input images to edit (1-10 images). Must be publicly accessible. |
| `prompt` | string | Yes | -- | Text description of the desired edit |
| `size` | string | No | -- | Output size in pixels (`WIDTH*HEIGHT`). Each dimension: 1024-4096. |

### Pricing

$0.04 per image (both generation and editing).

### Tips

- Seedream V4.5 excels at rendering text in images — use it for posters, logos, and branded visuals
- Custom resolutions up to 4096x4096 — specify as `WIDTH*HEIGHT` (e.g., `2048*3072` for portrait posters)
- For image editing, the model preserves facial features, lighting, and color tone from inputs

## Nano Banana Pro Image Generation/Editing

**Model ID:** `google/nano-banana-pro/text-to-image`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | -- | Text description of the image to generate |
| `aspect_ratio` | string | No | -- | Output aspect ratio. One of: `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `resolution` | string | No | `1k` | Image resolution. One of: `1k`, `2k`, `4k` |
| `output_format` | string | No | `png` | Output format. One of: `png`, `jpeg` |

**Model ID:** `google/nano-banana-pro/edit`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `images` | string[] | Yes | `[]` | URLs of input images to edit (1-14 images) |
| `prompt` | string | Yes | -- | Text description of the desired edit |
| `aspect_ratio` | string | No | -- | Output aspect ratio. One of: `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `resolution` | string | No | `1k` | Image resolution. One of: `1k`, `2k`, `4k` |
| `output_format` | string | No | `png` | Output format. One of: `png`, `jpeg` |

### Aspect Ratio Options

| Aspect Ratio | Use Case |
|-------------|----------|
| `1:1` | Square — social media posts, profile pictures |
| `3:2` | Landscape — standard photography |
| `2:3` | Portrait — standard photography |
| `3:4` | Portrait — social media, product images |
| `4:3` | Landscape — presentations, web content |
| `4:5` | Portrait — Instagram posts |
| `5:4` | Landscape — print, web banners |
| `9:16` | Vertical — mobile wallpapers, stories |
| `16:9` | Widescreen — desktop wallpapers, video thumbnails |
| `21:9` | Ultra-wide — cinematic, panoramic |

### Resolution and Pricing

| Resolution | Cost |
|------------|------|
| 1k | $0.14 per image |
| 2k | $0.14 per image |
| 4k | $0.24 per image |

### Prompt Tips

- Be specific and descriptive: "A red vintage Porsche 911 on a winding mountain road at golden hour" vs "a car"
- Include style keywords: "digital art", "oil painting", "photorealistic", "watercolor", "cinematic"
- For edits, clearly describe the desired change: "Replace the sky with a dramatic sunset"
- For multi-image edits, reference images by position: "Apply the style from the second image to the first image"
- Leverage multilingual text rendering: the model supports on-image text with automatic translation
- Use camera-style control language: "shallow depth of field", "wide angle", "top-down view"

## Nano Banana 2 Image Generation/Editing

**Model ID:** `google/nano-banana-2/text-to-image`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | -- | Text description of the image to generate |
| `aspect_ratio` | string | No | -- | Output aspect ratio. One of: `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, `1:4`, `4:1`, `1:8`, `8:1` |
| `resolution` | string | No | `1k` | Image resolution. One of: `1k`, `2k`, `4k` |
| `output_format` | string | No | `png` | Output format. One of: `png`, `jpeg` |

**Model ID:** `google/nano-banana-2/edit`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `images` | string[] | Yes | `[]` | URLs of input images to edit (1-14 images) |
| `prompt` | string | Yes | -- | Text description of the desired edit |
| `aspect_ratio` | string | No | -- | Output aspect ratio. One of: `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, `1:4`, `4:1`, `1:8`, `8:1` |
| `resolution` | string | No | `1k` | Image resolution. One of: `1k`, `2k`, `4k` |
| `output_format` | string | No | `png` | Output format. One of: `png`, `jpeg` |

### Aspect Ratio Options

| Aspect Ratio | Use Case |
|-------------|----------|
| `1:1` | Square — social media posts, profile pictures |
| `3:2` | Landscape — standard photography |
| `2:3` | Portrait — standard photography |
| `3:4` | Portrait — social media, product images |
| `4:3` | Landscape — presentations, web content |
| `4:5` | Portrait — Instagram posts |
| `5:4` | Landscape — print, web banners |
| `9:16` | Vertical — mobile wallpapers, stories |
| `16:9` | Widescreen — desktop wallpapers, video thumbnails |
| `21:9` | Ultra-wide — cinematic, panoramic |
| `1:4` | Ultra-tall — vertical banners, tall infographics |
| `4:1` | Ultra-wide — horizontal banners, letterbox |
| `1:8` | Extreme vertical — scrolling banners, tower ads |
| `8:1` | Extreme horizontal — panoramic strips, timelines |

### Resolution and Pricing

| Resolution | Cost |
|------------|------|
| 1k | $0.08 per image |
| 2k | $0.12 per image |
| 4k | $0.16 per image |

### Prompt Tips

- Be specific and descriptive: "A red vintage Porsche 911 on a winding mountain road at golden hour" vs "a car"
- Include style keywords: "digital art", "oil painting", "photorealistic", "watercolor", "cinematic"
- For edits, clearly describe the desired change: "Replace the sky with a dramatic sunset"
- For multi-image edits, reference images by position: "Apply the style from the second image to the first image"
- Leverage multilingual text rendering: the model supports on-image text with automatic translation
- Use camera-style control language: "shallow depth of field", "wide angle", "top-down view"

## Seedance V1.5 Pro Video Generation

**Model ID:** `bytedance/seedance-v1.5-pro/text-to-video`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | -- | Text description of the scene, style, actions, camera motion, and mood |
| `aspect_ratio` | string | No | `16:9` | Aspect ratio. One of: `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16` |
| `duration` | integer | No | `5` | Video duration in seconds. Range: 4-12. Use `-1` for smart duration (model selects). |
| `resolution` | string | No | `720p` | Video resolution. One of: `480p`, `720p`, `1080p` |
| `generate_audio` | boolean | No | `true` | Generate accompanying audio |
| `camera_fixed` | boolean | No | `false` | Keep camera fixed (true) or allow prompt-driven camera motion (false) |
| `seed` | integer | No | `-1` | Random seed (-1 for random). Range: -1 to 2147483647 |

**Model ID:** `bytedance/seedance-v1.5-pro/image-to-video`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the source image to animate |
| `prompt` | string | Yes | -- | Text description of the desired motion/animation |
| `last_image` | string | No | -- | URL of an optional end-frame reference image |
| `aspect_ratio` | string | No | -- | Aspect ratio. One of: `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16` |
| `duration` | integer | No | `5` | Video duration in seconds. Range: 4-12 |
| `resolution` | string | No | `720p` | Video resolution. One of: `480p`, `720p`, `1080p` |
| `generate_audio` | boolean | No | `true` | Generate accompanying audio |
| `camera_fixed` | boolean | No | `false` | Keep camera fixed (true) or allow prompt-driven camera motion (false) |
| `seed` | integer | No | `-1` | Random seed (-1 for random). Range: -1 to 2147483647 |

### Pricing

| Resolution | Duration | Audio | Cost |
|------------|----------|-------|------|
| 480p | 5s | No | $0.06 |
| 480p | 5s | Yes | $0.12 |
| 720p | 5s | No | $0.13 |
| 720p | 5s | Yes | $0.26 |
| 480p | 10s | Yes | $0.24 |
| 720p | 10s | Yes | $0.52 |

### Prompt Tips

- Describe scene, style, subject actions, camera motion, and mood in your prompt
- Use `camera_fixed: true` for stable tripod-style shots
- Use `camera_fixed: false` and describe camera motion: "slow pan left", "tracking shot", "zoom in"
- Set `generate_audio: false` when you plan to add your own audio track
- Use smart duration (`duration: -1`) to let the model choose the best length for text-to-video

## Veo 3.1 Fast Video Generation

**Model ID:** `google/veo3.1-fast/text-to-video`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | -- | Text description of the video to generate |
| `aspect_ratio` | string | No | `16:9` | Aspect ratio. One of: `16:9`, `9:16` |
| `duration` | integer | No | `8` | Duration in seconds. One of: `4`, `6`, `8` |
| `resolution` | string | No | `1080p` | Video resolution. One of: `720p`, `1080p`, `4k` |
| `generate_audio` | boolean | No | `true` | Generate accompanying audio |
| `negative_prompt` | string | No | -- | Text describing unwanted elements |
| `seed` | integer | No | -- | Random seed for reproducibility. Range: -1 to 2147483647 |

**Model ID:** `google/veo3.1-fast/image-to-video`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the source image (clear, high-quality still image) |
| `prompt` | string | Yes | -- | Text description of the desired motion/animation |
| `last_image` | string | No | -- | URL of an end-frame reference image |
| `aspect_ratio` | string | No | `16:9` | Aspect ratio. One of: `16:9`, `9:16` |
| `duration` | integer | No | `8` | Duration in seconds. One of: `4`, `6`, `8` |
| `resolution` | string | No | `1080p` | Video resolution. One of: `720p`, `1080p`, `4k` |
| `generate_audio` | boolean | No | `true` | Generate accompanying audio |
| `negative_prompt` | string | No | -- | Text describing unwanted elements |
| `seed` | integer | No | -- | Random seed for reproducibility. Range: -1 to 2147483647 |

**Model ID:** `google/veo3.1-fast/video-extend`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video` | string | Yes | -- | URL of the Veo-generated video to extend. Max 141 seconds. |
| `prompt` | string | No | -- | Text guidance for the extension |
| `resolution` | string | No | `1080p` | Video resolution. One of: `720p`, `1080p` |
| `negative_prompt` | string | No | -- | Text describing unwanted elements |
| `seed` | integer | No | -- | Random seed for reproducibility. Range: -1 to 2147483647 |

### Constraints

- Input video **must be Veo-generated** (will not work with arbitrary videos)
- Each run adds **+7 seconds** to the video
- Maximum **20 extensions** in a chain
- Maximum final video length: **148 seconds**
- Output is a single MP4 (original + extension appended)
- Aspect ratio and resolution are inherited from the input video

### Pricing

### Prompt Tips

- Be specific about scene, style, subject actions, camera motion, and mood
- Use `negative_prompt` to avoid artifacts: "blurry, low quality, distorted"
- For image-to-video, use a clear, high-quality still image as input
- For video extend, describe what should happen next in the scene
- Video extend chains enable building longer narratives — up to 148 seconds total

## Wan 2.6 Video Generation

**Model ID:** `alibaba/wan-2.6/text-to-video`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | -- | Text description of the video to generate |
| `negative_prompt` | string | No | -- | Text description of what to avoid in the video |
| `audio` | string | No | -- | Audio URL to guide generation |
| `size` | string | No | `1280*720` | Output size in pixels. One of: `1280*720`, `720*1280`, `1920*1080`, `1080*1920` |
| `duration` | integer | No | `5` | Video duration in seconds. One of: `5`, `10`, `15` |
| `shot_type` | string | No | `single` | Shot type. One of: `single`, `multi` |
| `enable_prompt_expansion` | boolean | No | `false` | Enable prompt optimizer for enhanced prompts |
| `seed` | integer | No | `-1` | Random seed (-1 for random). Range: -1 to 2147483647 |

**Model ID:** `alibaba/wan-2.6/image-to-video`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the source image to animate |
| `prompt` | string | Yes | -- | Text description of the desired motion/animation |
| `negative_prompt` | string | No | -- | Text description of what to avoid in the video |
| `audio` | string | No | -- | Audio URL to guide generation |
| `resolution` | string | No | `720p` | Output resolution. One of: `720p`, `1080p` |
| `duration` | integer | No | `5` | Video duration in seconds. One of: `5`, `10`, `15` |
| `shot_type` | string | No | `single` | Shot type. One of: `single`, `multi` |
| `enable_prompt_expansion` | boolean | No | `false` | Enable prompt optimizer for enhanced prompts |
| `seed` | integer | No | `-1` | Random seed (-1 for random). Range: -1 to 2147483647 |

### Resolution Options (Image-to-Video)

| Resolution | Use Case |
|------------|----------|
| `720p` | Standard quality, faster generation |
| `1080p` | Full HD, higher quality |

### Pricing

| Resolution | 5 seconds | 10 seconds | 15 seconds |
|------------|-----------|------------|------------|
| 720p | $0.50 | $1.00 | $1.50 |
| 1080p | $0.75 | $1.50 | $2.25 |

### Prompt Tips

- Be specific about motion and action: "A bird takes flight from a branch" vs "a bird"
- Include camera movement: "slow pan left", "zoom in", "tracking shot"
- Describe temporal progression: "transitioning from day to night", "flowers slowly blooming"
- Use `negative_prompt` to avoid artifacts: "blurry, low quality, distorted, static"
- Enable `enable_prompt_expansion` for automatic prompt enhancement
- For `multi` shot type, describe distinct scenes for more dynamic videos

## Wan 2.2 Animate

**Model ID:** `wavespeed-ai/wan-2.2/animate`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the character image to animate |
| `video` | string | Yes | -- | URL of the driving video providing motion reference |
| `prompt` | string | No | -- | Text prompt for additional guidance |
| `mode` | string | No | `animate` | Operation mode. `animate`: image character moves like video subject. `replace`: video subject is swapped with image character. |
| `resolution` | string | No | `480p` | Output resolution. One of: `480p`, `720p` |
| `seed` | integer | No | `-1` | Random seed (-1 for random). Range: -1 to 2147483647 |

### Pricing

| Resolution | Cost per 5 seconds |
|------------|--------------------|
| 480p | $0.20 |
| 720p | $0.40 |

Output duration is 5-120 seconds. Minimum charge is 5 seconds. Per-second rate: $0.04/s (480p), $0.08/s (720p).

### Tips

- Match composition and pose between the input image and driving video for best results
- Use the same or similar aspect ratio between image and video
- Avoid heavy occlusion by hands, microphones, or props in the input media
- Start with `480p` for prototyping, then move to `720p` for production quality
- **Animate mode**: best when you want the image character to perform the motions from the video
- **Replace mode**: best when you want to keep the video's scene and motion but swap in a different character

## InfiniteTalk

**Model ID:** `wavespeed-ai/infinitetalk`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the portrait image to animate |
| `audio` | string | Yes | -- | URL of the audio to drive the animation |
| `mask_image` | string | No | -- | URL of a mask image to specify which person to animate. **Warning:** The mask should only cover the regions to animate — do not upload the full image as `mask_image`, or the result may render as fully black. |
| `prompt` | string | No | -- | Text prompt for additional guidance. Keep it short; English recommended to avoid noisy results. |
| `resolution` | string | No | `480p` | Output resolution. One of: `480p`, `720p` |
| `seed` | integer | No | `-1` | Random seed (-1 for random). Range: -1 to 2147483647 |

### Resolution and Pricing

| Resolution | Cost per 5 seconds | Rate per second | Max length |
|------------|--------------------|-----------------| -----------|
| 480p | $0.15 | $0.03/s | 10 minutes |
| 720p | $0.30 | $0.06/s | 10 minutes |

Minimum charge is 5 seconds. Video length is determined by the audio duration (up to 10 minutes).

### Tips

- Use a clear, front-facing portrait for best results
- Audio quality matters — use clean speech recordings with minimal background noise
- Keep prompts short and in English to avoid noisy or unexpected results
- For group photos, always provide a `mask_image` to target the correct face
- 480p is faster to generate; use 720p when higher quality is needed
- Processing time is approximately 10-30 seconds of wall time per 1 second of video

## MiniMax Speech 2.6 Turbo

**Model ID:** `minimax/speech-2.6-turbo`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | -- | Text to convert to speech. Max 10,000 characters. Use `<#x#>` between words to insert pauses (0.01-99.99 seconds). |
| `voice_id` | string | Yes | -- | Voice preset ID. See [Voice IDs](#voice-ids) below. |
| `speed` | number | No | `1` | Speech speed. Range: 0.50-2.00 |
| `volume` | number | No | `1` | Speech volume. Range: 0.10-10.00 |
| `pitch` | number | No | `0` | Speech pitch. Range: -12 to 12 |
| `emotion` | string | No | `happy` | Emotional tone. One of: `happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`, `neutral` |
| `english_normalization` | boolean | No | `false` | Improve English number reading normalization |
| `sample_rate` | integer | No | -- | Sample rate in Hz. One of: `8000`, `16000`, `22050`, `24000`, `32000`, `44100` |
| `bitrate` | integer | No | -- | Bitrate in bps. One of: `32000`, `64000`, `128000`, `256000` |
| `channel` | string | No | -- | Audio channels. `1` (mono) or `2` (stereo) |
| `format` | string | No | -- | Output format. One of: `mp3`, `wav`, `pcm`, `flac` |
| `language_boost` | string | No | -- | Enhance recognition for a specific language. See [Supported Languages](#supported-languages). |

### Voice IDs

### English Voices (Popular)

| Voice ID | Description |
|----------|-------------|
| `English_CalmWoman` | Calm female voice |
| `English_Trustworth_Man` | Trustworthy male voice |
| `English_expressive_narrator` | Expressive narrator |
| `English_radiant_girl` | Radiant girl voice |
| `English_magnetic_voiced_man` | Magnetic male voice |
| `English_CaptivatingStoryteller` | Storyteller voice |
| `English_Upbeat_Woman` | Upbeat female voice |
| `English_GentleTeacher` | Gentle teacher voice |
| `English_PlayfulGirl` | Playful girl voice |
| `English_ManWithDeepVoice` | Deep male voice |
| `English_ConfidentWoman` | Confident female voice |
| `English_Comedian` | Comedic voice |
| `English_SereneWoman` | Serene female voice |
| `English_WiseScholar` | Scholarly voice |
| `English_Cute_Girl` | Cute girl voice |
| `English_Sharp_Commentator` | Sharp commentator |
| `English_Lucky_Robot` | Robot voice |

### General Voices

`Wise_Woman`, `Friendly_Person`, `Inspirational_girl`, `Deep_Voice_Man`, `Calm_Woman`, `Casual_Guy`, `Lively_Girl`, `Patient_Man`, `Young_Knight`, `Determined_Man`, `Lovely_Girl`, `Decent_Boy`, `Imposing_Manner`, `Elegant_Man`, `Abbess`, `Sweet_Girl_2`, `Exuberant_Girl`

### Special Voices

`whisper_man`, `whisper_woman_1`, `angry_pirate_1`, `massive_kind_troll`, `movie_trailer_deep`, `peace_and_ease`

### Pricing

$0.06 per 1,000 characters.

## Image Upscaler

**Model ID:** `wavespeed-ai/image-upscaler`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the image to upscale |
| `target_resolution` | string | No | `4k` | Target resolution. One of: `2k`, `4k`, `8k` |
| `output_format` | string | No | `jpeg` | Output format. One of: `jpeg`, `png`, `webp` |

### Pricing

$0.01 per image (all resolutions).

## Ultimate Video Upscaler

**Model ID:** `wavespeed-ai/ultimate-video-upscaler`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video` | string | Yes | -- | URL of the video to upscale. Must be publicly accessible. |
| `target_resolution` | string | No | `1080p` | Target resolution. One of: `720p`, `1080p`, `2k`, `4k` |

### Pricing

| Target Resolution | Cost per 5 seconds |
|-------------------|--------------------|
| 720p | $0.10 |
| 1080p | $0.15 |
| 2K | $0.25 |
| 4K | $0.40 |

Minimum charge is 5 seconds. Videos up to 10 minutes supported. Processing time is approximately 10-30 seconds per 1 second of video.

## Face Swapper

**Model ID:** `wavespeed-ai/image-face-swap`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the image containing the face to replace |
| `face_image` | string | Yes | -- | URL of the reference face image to swap in |
| `target_index` | integer | No | `0` | Which face to replace (0 = largest face, 1-10 for others) |
| `output_format` | string | No | `jpeg` | Output format. One of: `jpeg`, `png`, `webp` |

**Model ID:** `wavespeed-ai/video-face-swap`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video` | string | Yes | -- | URL of the video containing the face to replace. Must be publicly accessible. Max 10 minutes. |
| `face_image` | string | Yes | -- | URL of the reference face image to swap in |
| `target_index` | integer | No | `0` | Which face to replace (0 = largest face, 1-10 for others) |

### Pricing

| Operation | Cost |
|-----------|------|
| Image face swap | $0.01 per image |
| Video face swap | $0.01 per second (minimum $0.05 / 5 seconds) |

Video face swap supports videos up to 10 minutes.

### Tips

- Use clear, front-facing portraits for the reference face for best results
- Consistent lighting between the target and reference face improves quality
- Anime or illustrated characters may produce lower quality output
- Use `target_index` to select specific faces when multiple people are present (0 = largest face)

### Responsible use

Face swapping manipulates a real person's likeness. Before calling either endpoint, confirm all of the following with the user. If any answer is no or unclear, do not run the model and explain why.

- **Consent**: the person whose face is being inserted, and any identifiable person in the target media, has agreed to this use. Do not use the face of a public figure, a colleague, an ex-partner, or anyone else without their explicit consent.
- **Rights to the media**: the user owns the target image or video or has permission from the owner to edit it.
- **No impersonation or deception**: the output will not be presented as a real, unedited recording, used for fraud, identity verification bypass, harassment, defamation, political manipulation, or to put words or actions on someone they did not say or do.
- **No sexual or intimate content**: never swap a face onto sexual, nude, or intimate material, regardless of consent claims.
- **No minors**: do not process images or videos of children.
- **Disclosure**: when the result will be shared, recommend labeling it as AI-edited.

WaveSpeed's [Terms of Service](https://wavespeed.ai/static/terms) prohibit non-consensual and deceptive likeness edits; requests that violate them are refused and accounts may be suspended.

## Watermark Remover

**Model ID:** `wavespeed-ai/image-watermark-remover`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | -- | URL of the image to process |
| `output_format` | string | No | `jpeg` | Output format. One of: `jpeg`, `png`, `webp` |

**Model ID:** `wavespeed-ai/video-watermark-remover`

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video` | string | Yes | -- | URL of the video to process. Must be publicly accessible. Max 10 minutes. |

### Pricing

| Operation | Cost |
|-----------|------|
| Image watermark removal | $0.012 per image |
| Video watermark removal | $0.01 per second (minimum $0.05 / 5 seconds) |

Video watermark removal supports videos up to 10 minutes. Processing time is approximately 5-20 seconds per 1 second of video.

### Responsible use

Watermarks and overlays are usually there to assert ownership or attribution. Before calling either endpoint, confirm the following with the user. If the answer is no or unclear, do not run the model and explain why.

- **Ownership or license**: the user created the media, holds the rights to it, or has a license that permits removing the mark (for example, a purchased stock asset whose license allows clean use, or their own export from a tool that stamps a logo).
- **Not someone else's mark**: do not remove a watermark, logo, credit, or copyright notice placed by a third party to identify their work. Stock-site preview watermarks, photographer credits, broadcaster logos, and platform attribution marks are all out of scope.
- **No misrepresentation**: the result will not be passed off as original, unlicensed, or unedited work, and will not be used to evade licensing fees or attribution requirements.
- **Legitimate overlays only**: typical valid uses are removing captions or subtitles the user added, cleaning timestamps from their own camera footage, or restoring an area under a logo they own.

WaveSpeed's [Terms of Service](https://wavespeed.ai/static/terms) prohibit using the service to infringe intellectual property; requests that violate them are refused and accounts may be suspended.

