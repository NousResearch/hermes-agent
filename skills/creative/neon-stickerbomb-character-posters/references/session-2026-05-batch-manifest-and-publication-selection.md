# 2026-05 neon batch manifest + numbered publication selection notes

## Trigger

Use these notes when Nick asks for a multi-image neon character batch and may later say things like `发布1 3 4` or `发布到 image2skill`.

## Learning

For direct CLIProxyAPI `/v1/responses` image batches, the generation script should save a small sidecar manifest immediately, not only print `DONE ... saved ...` lines. Later publication depends on exact mapping from visible result numbers to:

- local image path
- semantic filename/title
- full prompt text
- model name
- intended Eagle folder
- tags / skill name

Without a manifest, publication scripts have to reconstruct prompts manually from the generation script or conversation, which is slower and error-prone.

## Recommended batch manifest shape

Write JSON under `~/.hermes/profiles/jea/state/`, for example:

```json
{
  "batch_id": "neon_known_female_20260505_1554",
  "skill": "neon-stickerbomb-character-posters",
  "folder": "neon",
  "model": "gpt-image-2",
  "items": [
    {
      "index": 1,
      "title": "Nico Robin / 罗宾",
      "semantic": "Nico-Robin_罗宾_neon-stickerbomb-cyber-pop-archaeology-poster",
      "path": "/absolute/path.png",
      "prompt": "full prompt text",
      "status": "done",
      "elapsed_seconds": 188.6
    }
  ]
}
```

Update each item as soon as its image is saved so partial batches still preserve completed outputs.

## Publication mapping rule

When Nick says `发布 1 3 4`, map those numbers to the most recent generated batch manifest, not to Eagle order or Discord message history. Then publish only the selected indices.

For `image2skill` publication:

1. Import selected files to Eagle `neon` folder if not already imported.
2. Use clean creative-library metadata:
   - `name`: `gpt-image-2`
   - `url` / website field: semantic filename
   - `annotation`: prompt text only
   - tags: `neon`, `image2skill`, `neon-stickerbomb-character-posters`, `gpt-image-2`, `model:gpt-image-2`, `prompt-saved`, `filename:<semantic>`, `synced-from-hermes`
3. Rebuild `/Users/nick/.hermes/profiles/jea/outputs/image2skill-redesign/rebuild_image2skill.py`.
4. Verify frontend counts, newest ordering, title presence, and `node --check` on the extracted inline script.

## Pitfall

Do not treat `发布到 image2skill` as Discord delivery, even if a Discord channel named `#image2skill` exists. It means Eagle-backed frontend publication unless Nick explicitly asks for Discord channel posting.
