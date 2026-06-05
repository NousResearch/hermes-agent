# DTC Ads Engine

DTC Ads Engine is a flexible ad-image generation surface for DTC brands. It composes a prompt with a brand kit and a picked ad format, optionally folds in reference media, an avatar, and a product, and produces a branded image.

CLI command: `higgsfield marketing-studio dtc-ads generate` (aliases: `dtc`, `ads`).

## End-to-end flow

There is no fixed order, but **picking an ad format is mandatory** — both the CLI and the server reject calls without `format_id`. There is no auto-default. Everything else (brand kit, avatar, product, media) is optional and only applied when the user explicitly provides them.

1. **Ask the user for an ad format.** Run `higgsfield marketing-studio ad-formats list --json` and let them pick by `name`. Do not auto-pick from the user's phrasing — the catalogue is small and the choice is creative, not technical.
2. **Pick or create a brand kit (optional).** See `marketing-brand-kits.md`.
3. **Offer optional inputs.** Suggest the user can attach an avatar (`--avatar`), a product (`--product`), or reference media (`--media`) if it would suit the brief.
4. **Generate.** With `--wait`, the command polls until the job completes and prints the result URL.

## Discover ad formats

```bash
higgsfield marketing-studio ad-formats list
higgsfield marketing-studio ad-formats list --type headline
higgsfield marketing-studio ad-formats list --json
```

Each item exposes `id`, `name`, `type`, `priority`, and an optional `media.thumbnail_url`. Filter by `--type` (client-side) when the user mentions a specific category — `headline`, `bullet-points`, `us-vs-them`, etc.

## Generate

```bash
# Bare minimum (format id is required)
higgsfield marketing-studio dtc-ads generate \
  --prompt "Bold hero shot on marble" \
  --format-id <format_uuid> \
  --wait

# With a brand kit
higgsfield marketing-studio dtc-ads generate \
  --prompt "Bold hero shot on marble" \
  --format-id <format_uuid> \
  --brand-kit-id <kit_uuid> \
  --wait

# With reference media + avatar + product
higgsfield marketing-studio dtc-ads generate \
  --prompt "Founder unboxing the product" \
  --format-id <format_uuid> \
  --brand-kit-id <kit_uuid> \
  --media <upload_id> \
  --avatar <avatar_id> \
  --product <product_id> \
  --aspect-ratio 9:16 \
  --resolution 2k \
  --quality medium \
  --batch-size 2 \
  --wait
```

Flags:

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--prompt` | yes | — | The brief. Required at the CLI level (or supply via `--from-file`). |
| `--format-id` | **yes** | — | Pick from `ad-formats list`. The CLI rejects calls without it. |
| `--brand-kit-id` | no | — | The kit must be `status: "completed"`. |
| `--aspect-ratio` | no | `1:1` | `1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9, 27:16, 16:27, 9:8, 8:9, 4:9, 9:4, auto` |
| `--resolution` | no | `1k` | `1k, 2k, 4k` |
| `--quality` | no | `low` | `low, medium, high` |
| `--batch-size` | no | `1` | 1..20 |
| `--media` | no | — | Repeatable, ≤14. Format: `<media_input_id>[:role]` (default role `image`). |
| `--avatar` | no | — | Max 1. Format: `<avatar_id>[:type]` (default type `preset`; use `custom` for user-uploaded avatars). |
| `--product` | no | — | Max 1. Product UUID from `marketing-studio products list`. |
| `--folder-id` | no | — | Folder placement. |
| `--from-file` | no | — | JSON file with the params shape. Flags merge over file values (flags win). |
| `--cost-only` | no | `false` | Print credit cost; do not create a job. |
| `--wait` | no | `false` | Poll until terminal status, then print the result URL. |
| `--timeout` | no | `5m` | Wait timeout. |

## UX rules for the agent

- **Always ask for an ad format.** `--format-id` is hard-required. Show the user the list (use `--type` to narrow down only when they hinted at a category) and let them pick by name.
- **Suggest avatar, product, or reference media** when the brief calls for them ("show the founder", "feature the product", "match this look") — but only attach what the user actually picks.
- **Ask the user about output settings.** Aspect ratio (1:1, 9:16, 16:9, …), resolution (1k/2k/4k), quality (low/medium/high), and batch size all materially change the output and cost — confirm them with the user instead of silently defaulting.
- **No auto-retry on failure.** If the job fails, surface the reason and let the user adjust prompt or parameters.

## Errors

- `Flag --prompt is required` — no prompt supplied.
- `Flag --format-id is required` — call rejected client-side. Run `ad-formats list` and pass an id.
- `--media accepts at most 14 entries.` / `--avatar accepts at most 1 entry.` / `--product accepts at most 1 entry.` — trim the input.
