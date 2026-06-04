# 2026-05 neon image generation + image2skill publication notes

## What happened

Nick generated multiple glossy neon cyber-pop sticker-bomb character posters, then selected specific result numbers to publish to image2skill. A key workflow correction occurred: “发布到 image2skill” means publish via the Eagle-backed image2skill frontend, not send attachments to Discord `#image2skill`.

## Working generation path

When Hermes `image_generate` / OpenAI Codex image auth returns `401 token_invalidated`, `429 usage_limit_reached`, or the configured CLIProxyAPI provider still hits a bad `/v1/images/generations` route, the successful path is direct CLIProxyAPI `POST /v1/responses` with required `image_generation` tool:

- base: `http://127.0.0.1:8317/v1`
- endpoint: `/responses`
- tool: `{ "type": "image_generation", "model": "gpt-image-2", "size": "1024x1536", "quality": "medium", "output_format": "png" }`
- one request per image
- use long timeouts; observed 4-image batch took roughly 12 minutes in one run
- run Python unbuffered (`python3 -u`) or use `flush=True` for progress visibility

This direct path is already documented more fully in `codex-ops/references/cliproxyapi-gptimage2-batch-generation.md`; keep this file focused on neon/image2skill workflow implications.

## Publication workflow after generation

1. If Nick says “发布 1 2”, map numbers to the most recent generated outputs in the assistant response.
2. If Nick says “发布到 image2skill”, treat it as frontend publication, not Discord posting.
3. Import selected images to the relevant Eagle folder (normally `neon`) if they are not already there.
4. Use clean creative-library metadata:
   - `name`: official model name, e.g. `gpt-image-2`
   - `url`/website field: semantic filename, e.g. `Yoruichi_夜一_neon-stickerbomb-cyber-pop-lightning-cat-dash-poster`
   - `annotation`: pure prompt text only when recoverable
   - tags: `neon`, `image2skill`, `neon-stickerbomb-character-posters`, `gpt-image-2`, `model:gpt-image-2`, `prompt-saved`, `filename:<semantic>`, `synced-from-hermes`
5. Rebuild the frontend with `/Users/nick/.hermes/profiles/jea/outputs/image2skill-redesign/rebuild_image2skill.py`.
6. Verify:
   - Eagle item(s) are in folder `neon` and tagged `image2skill`
   - rebuilt `image2skill-redesign.html` includes the selected title/semantic names
   - `node --check` on extracted inline script passes
   - frontend count increased and newest ordering surfaces the newly published items

## Pitfall

Do not confuse the Discord channel name `#image2skill` with the Eagle-backed frontend named image2skill. Sending media to the channel does not publish the image to the frontend.
