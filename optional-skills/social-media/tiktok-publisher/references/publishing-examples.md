# Publishing Examples

Read this file when you need a concrete invocation pattern.

## Publish from a remote URL

```bash
python3 tiktok-publisher/scripts/publish_tiktok.py \
  --source "https://example.com/video.mp4" \
  --title "My TikTok Title" \
  --privacy-level "SELF_ONLY"
```

## Publish from a local file

```bash
python3 tiktok-publisher/scripts/publish_tiktok.py \
  --source "/path/to/video.mp4" \
  --title "Hello from local file" \
  --privacy-level "PUBLIC" \
  --wait-for-published
```

## Publish with explicit polling options

```bash
python3 tiktok-publisher/scripts/publish_tiktok.py \
  --source "https://example.com/video.mp4" \
  --title "Scheduled check" \
  --poll-interval 5000 \
  --poll-timeout 300000
```

## Check status

```bash
python3 tiktok-publisher/scripts/check_status.py \
  --publish-id "abc123"
```
