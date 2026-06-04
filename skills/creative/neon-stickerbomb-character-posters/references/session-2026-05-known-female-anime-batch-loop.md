# 2026-05 known female anime neon batch loop

## Trigger

Use these notes when Nick asks for repeated batches such as `使用skill生成4张知名女性动漫角色`, `生成4张kof女性动漫角色`, then later selects items with `发布 1 3 4`.

## What worked

- Use `neon-stickerbomb-character-posters` for prompt construction and `codex-ops` direct CLIProxyAPI `/v1/responses` when the native `image_generate` route is unreliable.
- Generate one image per request, sequentially, with `gpt-image-2` `1024x1536` `quality: medium`.
- Save a manifest before generation and update it after each item completes. This is now required for numbered publication selection.
- Use character batches with visibly different archetypes, not just different names:
  - circular spell / fan / motion ring
  - side-profile split with foreground lens/scope
  - floating diagonal with tail/cloak/glider trails
  - low-angle sword/rifle/duelist diagonal
  - dutch-angle weapon/action poster
  - foreground prop depth poster
- For known female anime characters, keep prompts covered/non-explicit by default and preserve recognizability through concise identity cues rather than exposed-fashion cues.

## Batch-selection rule

When Nick says `发布 1 3 4`, map the numbers to the most recent manifest from the latest generated batch, not to Discord attachments, Eagle ordering, or frontend ordering.

## Operational latency

Direct CLIProxyAPI image calls may show no process output for several minutes even with `python3 -u`; wait in 180s chunks. Observed per-image latency in this session ranged from roughly 90s to 325s.

## Manifest fields to preserve

Each item should include:

- `index`
- `title`
- `semantic`
- `slug`
- `prompt`
- `status`
- `path` once saved
- `elapsed_seconds`

The manifest should include:

- `batch_id`
- `skill: neon-stickerbomb-character-posters`
- `folder: neon`
- `model: gpt-image-2`

## Pitfalls

- Do not end after writing the script; start it, wait for completion, and report actual file paths.
- Do not rely on session logs for prompt recovery: tool arguments are often truncated.
- Do not publish by the visible order in the Discord response if a newer batch ran afterward; use the latest relevant manifest path.
- Do not generate minor variants of the same pose in multi-character batches; pre-assign different layout skeletons before generation.
