# Notion File Uploads

Sources:

- `https://developers.notion.com/reference/file-object.md`
- `https://developers.notion.com/reference/file-upload.md`
- `https://developers.notion.com/reference/create-file.md`
- `https://developers.notion.com/reference/upload-file.md`
- `https://developers.notion.com/reference/complete-file-upload.md`
- `https://developers.notion.com/reference/list-file-uploads.md`
- `https://developers.notion.com/reference/retrieve-file-upload.md`
- `https://developers.notion.com/guides/data-apis/working-with-files-and-media.md`
- `https://developers.notion.com/cli/guides/file-uploads.md`

## File object types

`file`:

- Notion-hosted file already attached in workspace.
- Response includes signed URL and `expiry_time`.
- URL expires after about one hour; re-fetch object for a fresh URL.

`external`:

- Stable public HTTPS URL hosted outside Notion.
- No Notion file storage or URL expiry.

`file_upload`:

- File uploaded through File Upload API.
- Reference by File Upload ID after status is `uploaded`.
- Attach to page/block/page icon/page cover/database files property.

## CLI fast path

Use `ntn` if available:

```bash
ntn files create < ./photo.png
ntn files create --external-url https://example.com/file.pdf --filename file.pdf
ntn files get FILE_UPLOAD_ID
ntn files list
```

For scripts, `ntn files create --plain` prints tab-separated output with ID first. `--json` exists but prefer structured text unless an existing program requires JSON.

## HTTP direct upload: <=20 MB

### 1. Create upload object

```bash
curl -sS -X POST "https://api.notion.com/v1/file_uploads" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11" \
  -H "Content-Type: application/json" \
  -d '{"filename":"photo.png","content_type":"image/png"}'
```

Response includes `id`, status `pending`, and `upload_url`.

### 2. Send bytes

This endpoint uses multipart/form-data field `file`:

```bash
curl -sS -X POST "https://api.notion.com/v1/file_uploads/${FILE_UPLOAD_ID}/send" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2026-03-11" \
  -F "file=@photo.png"
```

Do not use a PUT to a presigned URL; current official docs use `POST /send` multipart.

### 3. Attach the file

Example image block:

```json
{
  "object": "block",
  "type": "image",
  "image": {
    "type": "file_upload",
    "file_upload": {"id": "FILE_UPLOAD_ID"},
    "caption": []
  }
}
```

Attach within one hour of creation/upload. Once attached successfully, the upload becomes permanent/reusable in the workspace and no longer has an expiry time.

## Multi-part upload: >20 MB

Use when file is over 20 MiB and workspace plan supports the size.

1. Create File Upload with `mode: "multi_part"`, `number_of_parts`, filename/content type.
2. Split into 5-20 MiB parts; final part may be under 5 MiB. Docs recommend 10 MiB parts.
3. Send each part to `/send` with form fields `file` and `part_number`.
4. Complete with `POST /v1/file_uploads/{file_upload_id}/complete`.
5. Attach within one hour.

Parts may be sent concurrently/out of order, but rate limits still apply and completion validates full size.

Doc conflict to watch: create schema allows `number_of_parts` up to 10,000, while send schema lists `part_number` max 1,000. Test current behavior before designing >1,000-part uploads.

## External URL import

Create with:

```json
{
  "mode": "external_url",
  "external_url": "https://example.com/file.pdf",
  "filename": "file.pdf"
}
```

Rules:

- URL must be public HTTPS.
- Server should expose `Content-Type` and `Content-Length` for Notion validation.
- Import is asynchronous; poll `GET /v1/file_uploads/{id}` or listen to file-upload webhooks if available.
- Final status is `uploaded` or `failed`.
- Failed imports cannot be attached or reused; create a new upload.

## Size and lifecycle limits

- Free workspace file limit: 5 MiB.
- Paid workspace file limit: 5 GiB.
- Direct single-part upload: <=20 MiB.
- Larger files use multi-part.
- Filename max: 900 bytes including extension.
- Status enum: `pending`, `uploaded`, `expired`, `failed`.
- Unattached uploads expire after about one hour.
- There is no public API to delete/revoke a created File Upload.
- Signed download URLs expire after about one hour.

Bot user response can expose `workspace_limits.max_file_upload_size_in_bytes`; use it to avoid predictable validation errors.

## Attach targets

Uploaded files can be attached to:

- media blocks: file, image, pdf, audio, video;
- data-source/page `files` properties;
- page icon;
- page cover.

The file type must match context. For example: no PDF as page icon, no video file in image block.

## Common errors

- `400 validation_error`: content too large, invalid MIME/extension, invalid filename, expired/pending state mismatch.
- `409 conflict_error`: rare third-party storage downtime while sending contents; retry later.
- `404 object_not_found`: upload ID not found for token/connection.
- `429 rate_limited`: back off per `Retry-After`.

## Operational pitfalls

- Do not cache signed file URLs.
- Do not retry upload creates after ambiguous failure without dedupe; new File Upload IDs may be created.
- If uploading large files concurrently, throttle to Notion's 3 req/s average limit.
- External imports depend on third-party server HEAD/GET behavior; failure may not be Notion's fault.
