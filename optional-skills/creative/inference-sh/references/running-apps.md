# Running Apps

## Basic Run

```bash
belt app run <app-id> --input '{"prompt": "your prompt"}'
```

## From File

```bash
belt app run <app-id> --input input.json
```

## Local File Uploads

The CLI automatically uploads local files when you provide a file path instead of a URL:

```bash
# Upscale a local image
belt app run topaz/image-upscaler --input '{"image": "/path/to/photo.jpg", "scale": 2}'

# Image-to-video from local file
belt app run wan/2.5-i2v --input '{"image": "./my-image.png", "prompt": "make it move"}'

# Avatar with local audio and image
belt app run omnihuman/1.5 --input '{"audio": "/path/to/speech.mp3", "image": "/path/to/face.jpg"}'

# Post tweet with local media
belt app run twitter/post-create --input '{"text": "Check this out!", "media": "./screenshot.png"}'
```

Supported paths: absolute (`/home/user/photo.jpg`), relative (`./image.png`), home (`~/Pictures/photo.jpg`).

## Generate Sample Input

Before running, generate a sample to see all available fields:

```bash
belt app sample seedream/4.5
belt app sample seedream/4.5 --save input.json
```

Edit the file, then run:

```bash
belt app run seedream/4.5 --input input.json
```

## Async Tasks

For long-running tasks (video, avatars), run in background:

```bash
# Submit and return immediately
belt app run veo/3.1 --input input.json --no-wait

# Check status later
belt task get <task-id>
```

## Output

Generated media returns as URLs:

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
| "invalid input" | Schema mismatch | Check `belt app get <id>` for required fields |
| "app not found" | Wrong app ID | Run `belt app store search <term>` |
| "quota exceeded" | Out of credits | Check account balance |
| Timeout | Long generation | Use `--no-wait` and poll with `belt task get` |
