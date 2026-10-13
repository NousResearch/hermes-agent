---
name: livepair
description: Generate images and video via the LivePair API.
version: 1.0.0
author: [livepairai]
license: MIT
platforms: [macos, linux, windows]
required_environment_variables:
  - name: LIVEPAIR_API_KEY
    prompt: "Enter your LivePair API key (lp_…)"
    help: "Create a prepaid key at https://livepairai.com/settings — top up from $5; generation is billed per result."
    required_for: generation
metadata:
  hermes:
    tags:
      - livepair
      - image-generation
      - video-generation
      - seedream
      - qwen-image
      - wan-video
      - pay-per-result
    related_skills: [comfyui]
    category: creative
---

# LivePair Skill

Generate images and video through the LivePair hosted API — no GPU and
no subscription; a prepaid balance is debited per finished result and
failed jobs refund automatically. This skill does not cover LivePair's
chat/text models or its MCP server — it is the media-generation path.

## When to Use

- Text-to-image, image edit (i2i), upscale, or background removal.
- Text-to-video or image-to-video clips up to the model's duration list.
- You need a one-off render inside an agent run without standing up
  ComfyUI or a GPU (for local workflows use the `comfyui` skill instead).

## Prerequisites

- `LIVEPAIR_API_KEY` set to an `lp_…` key (Settings → top up from $5;
  image ≈ $0.04–0.16, 5 s 720 p video ≈ $0.40–0.80 depending on model).
- Python 3 for `scripts/livepair.py` (stdlib only, no pip installs).

## How to Run

Everything goes through `terminal` calling `scripts/livepair.py`; all
subcommands print JSON.

## Quick Reference

```bash
python scripts/livepair.py models --kind image      # catalog + prices
python scripts/livepair.py models --kind video
python scripts/livepair.py balance                  # prepaid USD left
python scripts/livepair.py generate \
  --modelId seedream-5.0 --prompt "…" \
  --aspectRatio 16:9 --resolution 1k --wait --out render.png
python scripts/livepair.py generate \
  --modelId wan-3.0-i2v --prompt "…" \
  --image ./frame.png --duration 5 --wait --out clip.mp4
python scripts/livepair.py status <jobId> --out result.bin
```

## Procedure

1. **Quote first.** Run `models --kind <image|video>` and pick a `id`;
   `priceUsd` is the per-call image price, and video bills
   per-second × resolution (`resolutions[].priceUsd` × `duration`).
   Never hardcode ids or prices — the catalog is live.
2. **Generate.** `generate --modelId <id> --prompt "…" --wait --out
   <file>`. Image fields: `aspectRatio`, `resolution`, `imageSize`.
   Video fields: `duration` (seconds), `aspectRatio`, `resolution`.
   For `i2i`/`tool` modes pass `--image <https-url-or-local-file>`
   (local files ≤ 8 MB become data URIs).
3. **Poll if needed.** The API may answer synchronously
   (`status:"done"`, `url`) or return `jobId`; `--wait` polls every 3 s
   for up to 10 min. `status <jobId>` resumes an earlier job.
4. **Persist.** Output URLs auto-delete after 48 h — always pass
   `--out` (or download) for anything that must survive.

## Pitfalls

- **`402` = empty prepaid balance** — top up in Settings; it is not a
  rate limit. `401` = bad/missing key. `429` = actual rate limit — back
  off and retry.
- **Failed jobs refund** — a `status:"failed"` poll result is normal;
  the debit is reversed. Surface the `error` field to the user.
- **Tool modes take no creative prompt** — `img-upscale` and
  `img-remove-bg` need `image` only; billing is at floor margin.
- **`private: true` catalog entries** render on the private model
  line — their output must not be published to public feeds.
- **Keyless x402 exists but is opt-in** — wallets settle a `402`
  challenge in USDC on Base; don't drive that rail unless the operator
  asked for wallet billing. The `lp_` key is the normal path.

## Verification

- `python scripts/livepair.py balance` → JSON with a USD balance (proves
  key + reachability without spending).
- `models --kind image` → non-empty list containing `priceUsd` fields.
- A real render ends with `"status": "done"` and a non-zero `--out`
  file on disk.
