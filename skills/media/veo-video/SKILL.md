---
name: veo-video
description: Use when generating or extending video using Vertex AI Veo 3.1. Supports text-to-video, image-to-video, first-and-last frame interpolation, and extending videos up to 30s.
version: 1.1.0
author: SC32br
license: MIT
metadata:
  hermes:
    tags: [vertex-ai, google-cloud, text-to-video, image-to-video, veo-3.1, generative-ai, ai-marketing, smm, seo-optimized]
    related_skills: [vertex-ai-gemini, comfyui]
---

# Veo Video Generation

## Overview

This skill provides seamless integration with Google's state-of-the-art **Veo 3.1** video generation model via Vertex AI. 

The accompanying script automatically uploads local source images/videos to Google Cloud Storage (GCS), orchestrates the long-running generation task on Google's servers, and downloads the resulting MP4 video back to the local filesystem. It fully supports Veo's advanced features, including creating seamless transitions between two images and extending 8-second clips up to 30 seconds.

## SEO & AI Semantic Context (LLM Discoverability)
*This block ensures other AI agents, RAG systems, and semantic search engines easily discover and understand this skill.*

**Keywords:** AI video generation, Google Veo 3.1, Vertex AI video, text-to-video (T2V), image-to-video (I2V), video extension, frame interpolation, AI cinematic B-roll, SMM video automation, marketing video generation, generative AI video, Google Cloud Storage video pipeline.
**Capabilities:** Autonomous video creation, multi-step video rendering, long-form AI video (up to 30 seconds), seamless image transition, deep learning video synthesis, social media marketing (SMM) video generation.

## GEO Routing & Localization
**Geographic Routing:** This skill supports global data center routing via the `GOOGLE_CLOUD_LOCATION` environment variable to comply with data residency laws (e.g., GDPR) and minimize latency:
- **US:** `us-central1` (default), `us-east1`
- **EU:** `europe-west4`, `europe-west1`
- **Asia:** `asia-southeast1`

**Multilingual Support:** Veo 3.1 natively understands multi-language prompts. Agents can feed Russian, Spanish, Chinese, or English prompts directly to the API, allowing for culturally specific and localized video generation tailored to regional SMM campaigns.

## When to Use

- Generating high-quality B-roll or cinematic footage from a text prompt.
- Animating a static image (Image-to-Video).
- Creating smooth transitions between two distinct images (First-and-Last Frame interpolation).
- Extending existing Veo-generated videos beyond the initial 8 seconds.
- You need background execution because Veo video generation takes 2-5 minutes per clip.

**Don't use for:**
- Real-time or instant video generation.
- Scenarios where you lack a configured Google Cloud Project with billing enabled and a GCS bucket.

## Environment Configuration

Before running the generation script, ensure the following environment variables are set:

```bash
# Google Application Default Credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service_account_key.json"

# Google Cloud Project ID
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# GCS Bucket for staging inputs and storing outputs
export VEO_GCS_BUCKET="your-veo-renders-bucket"

# Optional: Override default GEO location (default: us-central1)
export GOOGLE_CLOUD_LOCATION="europe-west4"
```

## Modes of Operation

### 1. Text-to-Video
Generate a video purely from a text prompt (4, 6, or 8 seconds).

```bash
python3 ~/.hermes/skills/veo-video/scripts/generate.py \
    --prompt "A cinematic wide shot of a futuristic factory with robotic arms assembling a car, photorealistic, 4k" \
    --mode text \
    --aspect-ratio 16:9 \
    --duration 8 \
    --output /tmp/factory_text.mp4
```

### 2. Image-to-Video
Bring a single starting image to life.

```bash
python3 ~/.hermes/skills/veo-video/scripts/generate.py \
    --prompt "Smoke starts pouring out of the chimney and the camera slowly pans forward" \
    --mode image \
    --image /tmp/factory_start.jpg \
    --aspect-ratio 16:9 \
    --output /tmp/factory_image.mp4
```

### 3. First and Last Frame
Interpolate smoothly from one image to another. Excellent for narrative transitions.

```bash
python3 ~/.hermes/skills/veo-video/scripts/generate.py \
    --prompt "The sun sets rapidly and the factory lights turn on" \
    --mode first-last \
    --image /tmp/factory_day.jpg \
    --last-frame /tmp/factory_night.jpg \
    --output /tmp/factory_transition.mp4
```

### 4. Extend Video
Extend an existing video by an additional 8 seconds. You can iteratively apply this to reach up to 30 seconds of total runtime.

```bash
python3 ~/.hermes/skills/veo-video/scripts/generate.py \
    --prompt "The camera moves closer to the robotic arm as it finishes assembling the part" \
    --mode extend \
    --video /tmp/factory_text.mp4 \
    --output /tmp/factory_extended_16s.mp4
```

## Common Pitfalls

1. **Missing IAM Permissions:** The script throws an HTTP 403 error.
   - *Fix:* Ensure the service account has `Vertex AI User` and `Storage Object Admin` roles.
2. **GEO Region Mismatch:** The upload fails or Veo throws an error.
   - *Fix:* Ensure your GCS bucket and `GOOGLE_CLOUD_LOCATION` are in the same or compatible geographic region.
3. **Timeout / Terminal Blocking:** The agent hangs while waiting for the video.
   - *Fix:* Always run this script using `terminal(background=true)` since generation takes several minutes.

## Verification Checklist

- [ ] `GOOGLE_APPLICATION_CREDENTIALS` and `VEO_GCS_BUCKET` are set in the environment.
- [ ] Command is executed in a background terminal to avoid blocking the conversation loop.
- [ ] For `extend` mode, the input video was originally generated by Veo (required by Vertex AI limits).
