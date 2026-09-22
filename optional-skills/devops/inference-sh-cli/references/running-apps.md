# Running Apps

## Basic Run

```bash
infsh app run user/app-name --input input.json
```

## Inline JSON

```bash
infsh app run falai/flux-dev-lora --input '{"prompt": "a sunset over mountains"}'
```

## Version Pinning

```bash
infsh app run user/app-name@1.0.0 --input input.json
```

## Local File Uploads

The CLI automatically uploads local files when you provide a file path instead of a URL. Any field that accepts a URL also accepts a local path:

```bash
# Upscale a local image
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}'

# Image-to-video from local file
infsh app run falai/wan-2-5-i2v --input '{"image": "./my-image.png", "prompt": "make it move"}'

# Avatar with local audio and image
infsh app run bytedance/omnihuman-1-5 --input '{"audio": "/path/to/speech.mp3", "image": "/path/to/face.jpg"}'

# Post tweet with local media
infsh app run x/post-create --input '{"text": "Check this out!", "media": "./screenshot.png"}'
```

Supported paths:
- Absolute paths: `/home/user/images/photo.jpg`
- Relative paths: `./image.png`, `../data/video.mp4`
- Home directory: `~/Pictures/photo.jpg`

## Generate Sample Input

Before running, generate a sample input file:

```bash
infsh app sample falai/flux-dev-lora
```

Save to file:

```bash
infsh app sample falai/flux-dev-lora --save input.json
```

Then edit `input.json` and run:

```bash
infsh app run falai/flux-dev-lora --input input.json
```

## Workflow Example

### Image Generation with FLUX

```bash
# 1. Get app details
infsh app get falai/flux-dev-lora

# 2. Generate sample input
infsh app sample falai/flux-dev-lora --save input.json

# 3. Edit input.json
# {
#   "prompt": "a cat astronaut floating in space",
#   "num_images": 1,
#   "image_size": "landscape_16_9"
# }

# 4. Run
infsh app run falai/flux-dev-lora --input input.json
```

### Video Generation with Veo

```bash
# 1. Generate sample
infsh app sample google/veo-3-1-fast --save input.json

# 2. Edit prompt
# {
#   "prompt": "A drone shot flying over a forest at sunset"
# }

# 3. Run
infsh app run google/veo-3-1-fast --input input.json
```

### Text-to-Speech

```bash
# Quick inline run
infsh app run falai/kokoro-tts --input '{"text": "Hello, this is a test."}'
```

## Task Tracking

When you run an app, the CLI shows the task ID:

```
Running falai/flux-dev-lora
Task ID: abc123def456
```

For long-running tasks, you can check status anytime:

```bash
# Check task status
infsh task get abc123def456

# Get result as JSON
infsh task get abc123def456 --json

# Save result to file
infsh task get abc123def456 --save result.json
```

### Run Without Waiting

For very long tasks, run in background:

```bash
# Submit and return immediately
infsh app run google/veo-3 --input input.json --no-wait

# Check later
infsh task get <task-id>
```

## Output

The CLI returns the app output directly. For file outputs (images, videos, audio), you'll receive URLs to download.

Example output:

```json
{
  "images": [
    {
      "url": "https://cloud.inference.sh/...",
      "content_type": "image/png"
    }
  ]
}
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "invalid input" | Schema mismatch | Check `infsh app get` for required fields |
| "app not found" | Wrong app name | Check `infsh app list --search` |
| "quota exceeded" | Out of credits | Check account balance |

## Documentation

- [Running Apps](https://inference.sh/docs/apps/running) - Complete running apps guide
- [Streaming Results](https://inference.sh/docs/api/sdk/streaming) - Real-time progress updates
- [Setup Parameters](https://inference.sh/docs/apps/setup-parameters) - Configuring app inputs
