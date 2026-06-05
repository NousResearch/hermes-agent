# Marketing Studio Brand Kits

A brand kit captures a brand's identity (name, logo, hero images, colours, fonts, tone, products) and is reused across image generations to keep visuals on-brand.

A kit is created by handing in a website URL — it is being fetched and fields are being filled.

## Discover

```bash
higgsfield marketing-studio brand-kits list
higgsfield marketing-studio brand-kits list --json
higgsfield marketing-studio brand-kits list --size 50 --cursor <cursor>
```

Aliases: `brand-kit`, `bk`.

Each item carries `id`, `status` (`queued | in_progress | completed | failed | canceled`), `data.brand_name`, `data.logo`, `data.tagline`, `data.business_overview`, `data.industry`, `created_at`, `updated_at`, and a `latest_progress` snapshot when the kit is mid-fetch.

Use the `cursor` field for the next page (it's `null` when there are no more pages).

## Create from a URL

A brand kit is created by handing in a website URL.

```bash
# Fire and forget — returns the id, fetch runs in the background
KIT_ID=$(higgsfield marketing-studio brand-kits fetch --url https://drinkolipop.com --json | jq -r .id)

# Block until the kit is completed or failed (typical 30–90s)
higgsfield marketing-studio brand-kits fetch --url https://drinkolipop.com --wait
higgsfield marketing-studio brand-kits fetch --url https://drinkolipop.com --wait --timeout 180s
```

`--wait` polls and renders the latest step on stderr until terminal status. On failure, the command exits non-zero and surfaces the kit's `error` field.

The only valid input is a real website URL.

### Failed kits

A kit that ends in `status: "failed"` is terminal. Ask the user for a different URL, or skip the brand kit and generate without one (`dtc-ads generate` works without `--brand-kit-id`).

## Inspect

```bash
higgsfield marketing-studio brand-kits get <id>
higgsfield marketing-studio brand-kits get <id> --json
```

The server checks completion **on its side** before applying a kit to a generation. So if `dtc-ads generate` returns an error mentioning the kit isn't ready, just wait and retry — don't try to bypass.

## Use in generation

Pass `--brand-kit-id <id>` to `higgsfield marketing-studio dtc-ads generate`. See `marketing-dtc-ads.md` for the full flow.
