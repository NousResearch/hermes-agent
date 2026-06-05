# Marketing Studio Ad References

Ad references are reusable inspiration videos a user wants to model new ads after — typically tied to a specific avatar and/or product. The backend processes the input video and stores a reusable reference the user can recall later.

## Inputs

Create an ad reference from one of two source types:

- `--video-input <upload_id>` — UUID returned by `higgsfield upload create <video_path> --video`. Use when the user has a local video file or already uploaded one.
- `--job <job_id>` — UUID of a previously generated video job (`higgsfield generate list --json`). Use when the user wants to reuse one of their own generated clips as a reference.

Exactly one of these two flags is required.

Optional binding flags (each accepts at most one id):

- `--avatar <avatar_id>` — link the reference to a marketing-studio avatar (preset or custom). One avatar max per reference.
- `--product <product_id>` — link the reference to a marketing-studio product. One product max per reference.

## Sources accepted

There are exactly two supported inputs:

- A **local video file** — pass via `higgsfield upload create <path> --video`, then use the returned `upload_id` with `--video-input`.
- A **prior video generation job** from this account — pass the `job_id` with `--job`.

If the user supplies anything else (a URL, a streaming link, an external reference), ask for a local video file. Do not attempt to fetch or convert other inputs.

## Constraints

- **Mutually exclusive with hook/setting at generation time.** When the user has selected an ad reference for the ad, do **not** also pass `--hook_id` or `--setting_id` to `marketing_studio_video`. Pick one path: either reference-driven (ad_reference) or composed-from-blocks (hook + setting). Mixing is not supported.
- **One avatar, one product per reference.** Both `--avatar` and `--product` accept a single id. To bind multiple, create separate references.

## Create

```bash
# From an uploaded local video
UPLOAD_ID=$(higgsfield upload create reel.mp4 --video --json | jq -r .id)
REF_ID=$(higgsfield marketing-studio ad-references create --video-input $UPLOAD_ID --json | jq -r .id)

# From a previous generation job
JOB_ID="b1a2c3d4-..."
higgsfield marketing-studio ad-references create --job $JOB_ID --json

# Bind to an avatar and product at creation time
higgsfield marketing-studio ad-references create \
  --video-input $UPLOAD_ID \
  --avatar <avatar_id> \
  --product <product_id> \
  --json
```

The backend kicks off processing asynchronously. Newly created references start in `status: queued` then move to `in_progress` and finally `completed` (or `failed`).

## Discover

```bash
higgsfield marketing-studio ad-references list
higgsfield marketing-studio ad-references list --json
higgsfield marketing-studio ad-references list --size 50 --cursor <cursor>
```

Aliases: `ad-refs`, `adrefs`.

The response shape is:

- `items`: each item has `id`, `status`, `source_platform`, `video_input_id`, `job_id`, `video_s3_url` (the processed reference URL when ready), `video_thumbnail_url`, `avatar_id`, `product_id`, `created_at`, `updated_at`.
- `total_count`: total references on the account.
- `cursor`: pagination cursor (created_at unix timestamp); `null` when there are no more pages.

## Inspect

```bash
higgsfield marketing-studio ad-references get <id>
higgsfield marketing-studio ad-references get <id> --json
```

Use this to check `status`, read `fail_reason` when `status: failed`, or grab `video_s3_url` once `status: completed`.

## Polling for completion

`create` returns immediately with `status: queued`. The reference is **not** usable for generation until `status: completed`. There is no built-in `--wait` flag, so poll explicitly:

```bash
REF_ID=$(higgsfield marketing-studio ad-references create --video-input $UPLOAD_ID --json | jq -r .id)
while :; do
  STATUS=$(higgsfield marketing-studio ad-references get $REF_ID --json | jq -r .status)
  case "$STATUS" in
    completed) break ;;
    failed)
      REASON=$(higgsfield marketing-studio ad-references get $REF_ID --json | jq -r .fail_reason)
      echo "Ad reference failed: $REASON" >&2
      exit 1
      ;;
    *) sleep 5 ;;
  esac
done
```

Always wait for `completed` before passing the reference id to a generation step.
