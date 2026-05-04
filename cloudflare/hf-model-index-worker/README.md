# Hugging Face Model Index Worker

Cloudflare Worker for Hermes Android model discovery. It keeps a tiny signed
JSON catalog in Workers KV and stores repo metadata in D1. Model weights stay on
Hugging Face; Hermes downloads them directly on-device through Android
DownloadManager.

## Endpoints

- `GET /models.json` serves the signed model catalog consumed by the Android app.
- `POST /hf-webhook` accepts Hugging Face model repo webhook events.
- `POST /admin/scan` rescans watched orgs and rebuilds the catalog.
- `GET /admin/scan?token=...` is the HTTP(S) fallback cron path for services
  that cannot send POST webhooks.
- `GET /admin/status?token=...` reports scan/index state.

## Storage

This project uses the metadata-only free-limit-friendly path:

- D1 `hf-model-index` stores repo rows and scan state.
- KV `MODEL_KV` stores the latest signed `models.json` document.
- R2 is not required because model binaries are not mirrored. On the current
  account Wrangler reported that R2 is not enabled, so bucket creation is not
  available until R2 is enabled in the Cloudflare dashboard.

## Setup

```bash
npm install
npm run generate-keys
npm run migrate:remote
type .private-jwk.json | wrangler secret put PRIVATE_JWK
```

Set production secrets:

```bash
wrangler secret put ADMIN_SECRET
wrangler secret put HF_WEBHOOK_SECRET
```

`HF_TOKEN` is optional for public repo polling but can be added for higher
Hugging Face API allowance:

```bash
wrangler secret put HF_TOKEN
```

Deploy:

```bash
npm run deploy
```

Manual fallback scan:

```bash
curl "https://<worker>.workers.dev/admin/scan?token=<ADMIN_SECRET>"
```
