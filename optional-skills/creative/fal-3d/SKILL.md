---
name: fal-3d
description: Generate 3D meshes (GLB) from text or an image via fal.ai.
version: 1.0.0
author: Hermes Agent (agent-tools-scout)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [3d, mesh, glb, text-to-3d, image-to-3d, fal, tripo, meshy, hunyuan, trellis, game-assets, generation]
    category: creative
    related_skills: [dream-loop, unreal-mcp]
required_environment_variables:
  - name: FAL_KEY
    prompt: fal.ai API key
    help: "Create one at https://fal.ai/dashboard/keys — the same key the image_generate / video_generate FAL backends use."
    required_for: "every generation call (--list and --dry-run work without it)"
---

# fal-3d — hosted text-to-3D and image-to-3D

One script, one key, five hosted 3D generators. Use this when the user wants an
actual mesh file (GLB by default; FBX/OBJ/USDZ URLs when the model returns them)
for a game, a Blender/Unreal scene, AR, or a 3D print. Cloud generation takes
30 s to a few minutes and costs $0.22–1.50 per model; no local GPU.

## When to Use

- "Make me a 3D model of a treasure chest / cartoon fox / this mug in the photo"
- Game or scene assets for `unreal-mcp`, `dream-loop`, Blender, Godot, three.js
- Turn a product photo or a generated image (`image_generate`) into a mesh
- Something printable: geometry-only, then hand the GLB/OBJ to a slicer
- The user has a fal.ai account (`FAL_KEY`) — the same key the `image_generate`
  and `video_generate` FAL backends use

Do NOT use for editing an existing mesh (retexture, rig, remesh — Meshy's own CLI
or Blender do that) or for scenes built from primitives (write the Blender/three.js
code instead).

## Models

| id | text | image | faces knob | pick it when |
|----|------|-------|------------|--------------|
| `hunyuan-3.1-rapid` (default) | yes | yes | no | fastest and cheapest; $0.225 per model (+$0.15 `--pbr`); prompt ≤ 200 chars; OBJ+MTL out, GLB when available |
| `hunyuan-3.1-pro` | yes | yes | 40k–1.5M | dense high-poly mesh; $0.375; `--no-texture` gives a white geometry-only model |
| `tripo-p2` | yes | yes | ≤ 25k | production PBR assets, `--quad` topology, `--texture-quality`, seedable, `--negative-prompt`; $1.00–1.30 |
| `meshy-7.1` | yes | yes | 100–300k | game-ready assets, `--quad`, `--pbr`, seedable (text); $0.80 untextured / $1.20 textured; also returns FBX + USDZ |
| `trellis-2` | no | yes | 5k–2M vertices | open Microsoft model, image only, GPU-time billing; good for a single clean product shot |

Prices come from fal's model pages at authoring time and change; `--dry-run` never
bills. Every endpoint is queue-based: the script blocks until the mesh is ready.

## Procedure

1. Locate the script (skills install under `~/.hermes/skills/`; profiles under
   `~/.hermes/profiles/<name>/skills/`):
   ```bash
   SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/fal-3d/SKILL.md' 2>/dev/null | head -1)")
   [ -f "$SKILL_DIR/SKILL.md" ] || echo "fal-3d is not installed: hermes skills install official/creative/fal-3d"
   ```
2. Confirm prerequisites once per machine. Use the interpreter Hermes runs on
   (`~/.hermes/hermes-agent/.venv/bin/python` on a source install; a bare `python`
   on a PEP 668 host may be the wrong one):
   ```bash
   python -c "import fal_client" 2>/dev/null || pip install fal-client==0.13.1
   test -n "$FAL_KEY" || echo "FAL_KEY missing — https://fal.ai/dashboard/keys"
   ```
   Hermes sessions already export the `FAL_KEY` saved during setup (`hermes setup`
   or `hermes config set`), so the `terminal` tool sees it without extra steps; a
   plain shell outside Hermes needs `export FAL_KEY=...` first.
3. Write the prompt as an object description, not a scene: one subject, material,
   style, and view-independent details ("low-poly wooden treasure chest, iron
   bands, closed lid, stylized game asset"). For image-to-3D use a single object on
   a plain background, front or three-quarter view; a busy photo yields a blob.
   Generate the reference with `image_generate` first when the user only has words
   but wants a specific look.
4. Preview the payload, then generate:
   ```bash
   python "$SKILL_DIR/scripts/fal_3d.py" --model hunyuan-3.1-rapid \
     --prompt "low-poly wooden treasure chest, iron bands, closed lid" --dry-run
   python "$SKILL_DIR/scripts/fal_3d.py" --model hunyuan-3.1-rapid \
     --prompt "low-poly wooden treasure chest, iron bands, closed lid" -o chest.glb
   ```
   Image-to-3D from a local file (uploaded to fal's CDN first) or a URL:
   ```bash
   python "$SKILL_DIR/scripts/fal_3d.py" --model tripo-p2 --image ./mug.png \
     --quad --faces 20000 --pbr -o mug.glb
   ```
   Cheap geometry for printing (no textures):
   ```bash
   python "$SKILL_DIR/scripts/fal_3d.py" --model meshy-7.1 \
     --prompt "chess knight piece, smooth, solid base" --no-texture -o knight.glb
   ```
5. The script prints JSON with `output`, `url`, `format`, `preview_url`, `seed` and
   `other_formats` (FBX/OBJ/USDZ URLs when the model returned them). Return the
   mesh with `MEDIA:<absolute path of chest.glb>`, show the `preview_url` render so
   the user can judge without opening a viewer, and mention the seed when present.
6. Iterate by changing one thing at a time (prompt wording, model, seed, `--faces`).
   Escalate cost only when the cheap tier fails: `hunyuan-3.1-rapid` → `hunyuan-3.1-pro`
   → `meshy-7.1` / `tripo-p2`. For downstream tooling, `--quad` (tripo, meshy) gives
   cleaner topology for sculpting and rigging; triangles are fine for engines.

## Pitfalls

- **`--faces` is clamped, never rejected** — 5,000 on `hunyuan-3.1-pro` becomes
  40,000 (its floor); 999,999 on `tripo-p2` becomes 25,000. Models with no face knob
  (`hunyuan-3.1-rapid`) drop it and the script prints a `note: ... ignored` line on
  stderr. `--list` shows each range.
- **`--seed` is only honoured where declared**: tripo-p2, meshy-7.1 text-to-3D,
  trellis-2. Hunyuan has no seed knob; Meshy image-to-3D has none either.
- **`--no-texture` maps per model**: tripo `texture=false, pbr=false`; meshy
  `mode=preview` / `should_texture=false`; hunyuan pro `generate_type=Geometry`;
  hunyuan rapid `enable_geometry=true`. Trellis always textures.
- **Hunyuan rapid returns OBJ + MTL + a texture PNG**, not always a GLB; the script
  saves whatever the primary mesh is and fixes the extension when `-o` has none.
  Pass `-o thing.glb` only for models that return GLB, or check `format` in the JSON.
- **`trellis-2` has no text-to-3D endpoint**; the script exits with a message to pass
  `--image`. Generate a reference with `image_generate` first.
- **Prompt length limits**: hunyuan rapid 200 chars, meshy 600, tripo/hunyuan pro 1024.
  Over-long prompts are rejected by the endpoint (HTTP 422), not truncated.
- **Local `--image` files are uploaded to fal's public CDN** (`fal_client.upload_file`);
  say so when the reference is a private photo, or pass a URL you already host.
- **`fal_client` errors with "User is locked. Reason: Exhausted balance"** means the
  fal account needs a top-up, not a code problem. Surface the message.
- The script only forwards keys each endpoint's OpenAPI schema declares; `--quad`
  on hunyuan is dropped with a stderr note, not an error.

## Verification

- `python fal_3d.py --list` prints all five models with both endpoints.
- `--dry-run` shows the exact endpoint + payload before any billable call, and
  `<upload:/abs/path>` in place of a local image instead of uploading it.
- After generation the output file starts with the `glTF` magic (`head -c 4 out.glb`)
  for GLB, or `# ` / `v ` lines for OBJ; `preview_url` opens to a rendered PNG.
