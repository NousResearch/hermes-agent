# Publishing Examples

Read this file when you need a concrete invocation pattern.

## Publish a single image from a URL

```bash
python3 instagram-publisher/scripts/publish_instagram.py \
  --type IMAGE \
  --url "https://example.com/image.jpg" \
  --caption "Hello World!"
```

## Publish a reel from a local file

```bash
python3 instagram-publisher/scripts/publish_instagram.py \
  --type REELS \
  --path "/path/to/video.mp4" \
  --caption "Check this out!" \
  --thumb-offset 1000
```

## Publish a mixed carousel

```bash
python3 instagram-publisher/scripts/publish_instagram.py \
  --type CAROUSEL \
  --items "/path/to/img1.jpg" "https://example.com/vid2.mp4" \
  --caption "My Carousel"
```

## Check status

```bash
python3 instagram-publisher/scripts/publish_instagram.py \
  --check-id "v_pub_file~123"
```
