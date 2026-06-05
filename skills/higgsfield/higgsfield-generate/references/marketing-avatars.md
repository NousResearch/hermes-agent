# Avatars

## Preset vs Custom

| | Preset | Custom |
|---|---|---|
| Source | Curated by Higgsfield | User-uploaded |
| Cost | None for selection | Cost of upload |
| Diversity | Limited but professional | Unlimited |
| Use when | Generic ad, fast turnaround | Brand-specific face, founder, employee |

## Listing presets

```bash
higgsfield marketing-studio avatars list
higgsfield marketing-studio avatars list --json | jq '.[] | select(.gender=="female")'
```

Filter by `name`, `gender`, etc. on the JSON output.

## Creating a custom avatar

```bash
ID=$(higgsfield upload create founder.png)
URL=$(higgsfield upload create founder.png --json | jq -r .url)   # if you need cloudfront URL
higgsfield marketing-studio avatars create --name "Founder" --image $ID --image-url $URL
```

`--image-url` is the cloudfront URL from the upload. Required by the API.

## Passing to video

```bash
AVATARS_JSON=$(mktemp)
printf '[{"id":"<avatar_id>","type":"preset"}]' > "$AVATARS_JSON"

higgsfield generate create marketing_studio_video \
  --avatars @"$AVATARS_JSON" \
  ... \
  --wait
```

`type` is `preset` for curated, `custom` for user-created.
`--avatars` expects a JSON array, so pass it via `@/path/to/file.json`.

For UGC modes, an avatar is optional if the brief clearly mentions a person and no specific presenter was requested; the backend can synthesize a Soul Character automatically.
