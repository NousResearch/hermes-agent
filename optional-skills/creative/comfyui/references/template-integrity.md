# ComfyUI Workflow-Template Integrity

> **Authored by [@purzbeats](https://github.com/purzbeats)** — adapted from
> [purzbeats/hermes-agent-comfyui-helper](https://github.com/purzbeats/hermes-agent-comfyui-helper).
> Use this reference when converting workflows from the official
> `comfyui-workflow-templates` package (editor format) into API format for
> submission via `/api/prompt`. The conversion has subtle gotchas that cause
> hard-to-diagnose validation errors if you don't follow these rules.

## Background

The official ComfyUI template package (`comfyui-workflow-templates`, currently
v0.9.69) is installed inside the ComfyUI venv at a path like:

```
<comfy-install>/.venv/lib/python3.*/site-packages/comfyui_workflow_templates_*/templates/
```

The exact path depends on how ComfyUI was installed (comfy-cli default,
Comfy Desktop, manual venv, etc.). Find it once with:

```bash
comfy --workspace <ws> run-python -c "import comfyui_workflow_templates, pathlib; print(pathlib.Path(comfyui_workflow_templates.__file__).parent / 'templates')"
```

Templates ship in **editor format** — `nodes` / `links` arrays inside
`data['definitions']['subgraphs'][0]`. They must be converted to **API
format** (a `node_id -> {class_type, inputs}` mapping) before submission.

---

## RULE #1: Use templates AS CLOSE TO ORIGINAL AS POSSIBLE

- **Never strip, simplify, or "minimize" nodes** from a template.
- Full template architecture (dual-pass pipelines, LoRA chains, distilled
  sigmas, conditioning paths) is intentional — removing any part breaks quality.
- If an image-dependent path exists but the task is text-to-video, **leave
  it wired with the bypass toggle enabled** — don't remove the nodes.
- Only change: prompt text, seed, and dimensions (when explicitly requested).

## RULE #2: Server validation errors are the source of truth

When a workflow submission fails, the server response looks like:

```json
{
  "node_errors": {
    "238": {
      "errors": [{
        "message": "Required input is missing",
        "details": "width",
        "extra_info": { "input_name": "resize_type.width" }
      }]
    }
  }
}
```

**The `extra_info.input_name` field tells you EXACTLY what JSON key the server
wants. Use it literally.** If it says `"values.a"` or `"resize_type.width"`,
those are the actual key names in the JSON object. Do not "simplify" them to
flat names based on assumptions about what the field "should" be called.

## RULE #3: Don't rebuild from scratch — patch the failing nodes

Every regeneration from the template reintroduces the same bugs. Instead:

1. Submit the workflow once.
2. Read the server error details for exact key names.
3. Use targeted patch/fix calls against the workflow file on disk.
4. Resubmit and check if errors resolved.

---

## Reroute nodes: bypass, don't delete

Most servers (local, Cloud) don't have a `Reroute` node type. When converting
a template:

1. Find what feeds into the Reroute by looking at links where
   `target_id` = the Reroute node ID.
2. Replace all inputs referencing the Reroute with
   `[source_node_id, source_slot]`.
3. Delete the Reroute node from the API mapping.

**Real example — LTX 2.3 t2v template:**

- Reroute node 255 receives VAE from `CheckpointLoaderSimple 236` slot 2.
- Three nodes reference Reroute 255 for their VAE input:
  `LTXVImgToVideoInplace` (230), `LTXVLatentUpsampler` (253),
  `VAEDecodeTiled` (251).
- Fix: replace all occurrences of `vae: ["255", 0]` with `vae: ["236", 2]`.
- `CheckpointLoaderSimple` slot 2 = VAE (not slot 0 = MODEL).

| | |
|---|---|
| ❌ Wrong  | `vae: ["236", 0]` → `MODELV mismatch input_type(VAE)` |
| ✅ Correct | `vae: ["236", 2]` |

---

## Dynamic template nodes: dotted key names are correct

### ComfyMathExpression (COMFY_AUTOGROW_V3)

```json
{
  "class_type": "ComfyMathExpression",
  "inputs": {
    "expression": "a/2",
    "values.a": ["257", 0]
  }
}
```

- `values` is a `COMFY_AUTOGROW_V3` template.
- Input names in links are `values.a`, `values.b`, etc.
- **Keep the dotted format as JSON keys.**
- Do NOT convert to `{"values": {"a": ...}}` or flatten to just `"a"`.

### ResizeImageMaskNode (COMFY_DYNAMICCOMBO_V3)

```json
{
  "class_type": "ResizeImageMaskNode",
  "inputs": {
    "input": ["276", 0],
    "scale_method": "lanczos",
    "resize_type": "scale dimensions",
    "resize_type.width": 1920,
    "resize_type.height": 1088,
    "resize_type.crop": "center"
  }
}
```

- `resize_type` is a `COMFY_DYNAMICCOMBO_V3`.
- Mode-specific fields: `resize_type.width`, `resize_type.height`, `resize_type.crop`.
- `scale_method` options: `"nearest-exact"`, `"bilinear"`, `"area"`, `"bicubic"`, `"lanczos"`.
- **Keep the dotted format as JSON keys.**
- Do NOT flatten `resize_type.width` to just `"width"`.

---

## Conversion recipe

1. Load template from the installed package path.
2. Parse `data['definitions']['subgraphs'][0]`.
3. For each node (skip Reroute):
   - Resolve linked inputs from `sg['links']` dict.
   - Map `widgets_values` to input field names.
   - Keep all dotted key names as-is from the template.
4. Bypass Reroute: trace source, replace references.
5. Change only: prompt text, seed values, and user-requested parameters.
6. Add `SaveVideo` terminal node if template uses only `CreateVideo`.
7. Submit → read errors → patch specific nodes → resubmit.

## What to NEVER change in a template

| Element | Why |
|---------|-----|
| Node topology | Graph is designed for the specific model |
| Sigmas values | Tuned for the model/sampler combination |
| LoRA/distilled paths | Required for quality, even if they look unused |
| Model parameters (cfg, steps, shifts) | Model-specific |
| Conditioning chains (zero-out, crop guides) | Required for correct conditioning |
| Pass-through wiring | Don't remove nodes, bypass them |

---

## Cloud compatibility (verified May 2025)

The full LTX 2.3 T2V template (`video_ltx2_3_t2v.json`) runs **without
modification** on Comfy Cloud.

**Confirmed working on Cloud (all custom nodes available):**
`ComfyMathExpression`, `ResizeImageMaskNode`, `ResizeImagesByLongerEdge`,
`PrimitiveInt`, `PrimitiveStringMultiline`, `PrimitiveBoolean`, `SaveVideo`,
`LTXVCropGuides`, `LTXVImgToVideoInplace`, `LTXVConcatAVLatent`,
`LTXVSeparateAVLatent`, `LTXVLatentUpsampler`, `LTXVAudioVAELoader`,
`LTXVAudioVAEDecode`, `LTXVEmptyLatentAudio`, `LTXVPreprocess`,
`LTXVConditioning`, `ManualSigmas`, `LTXAVTextEncoderLoader`, plus all core
nodes.

**Cloud vs Local for LTX 2.3 (768x512):**

- Cloud: ~39s per video (4x faster).
- Local (RTX 5090): ~160s per video.
- `example.png` placeholder works on Cloud for bypassed image-dependent paths.
- Submission format is **identical** between local and Cloud:
  `{"prompt": wf, "extra_data": {}}` to `/api/prompt`.
- Free tier = 1 concurrent job.

**Cloud submission pitfalls:**

- `/api/object_info/<node>` returns 404 on free tier — can't query node
  schemas remotely, but the workflow runs fine anyway. Always probe
  `object_info` locally before building workflows.
- Cloud is ~4x faster — prefer Cloud for batch runs unless local is needed
  for debugging.
- Cloud `/api/view` returns **302 redirect to signed GCS URL** — use
  `curl -s -L` to follow and download. Python `urllib` fails with 401
  (forwards auth headers to GCS CDN).
- `COMFY_CLOUD_API_KEY` is only in the terminal/bash env, not in the Python
  sandbox. Use subprocess or terminal scripts for Cloud API calls.
- Cloud free tier processes jobs **sequentially** (1 at a time). Submit all,
  then poll history.
- LTX 2.3 at **1920x1080 OOMs locally** (even RTX 5090) — upscaler pass
  exceeds VRAM. Prefer Cloud for 1080p; use 1280x720 locally (~90s/video).

---

## FFmpeg stitch settings (Discord-compatible)

Generated ComfyUI videos often use `yuv444p` pixel format which does NOT work
on Discord. Re-encode with:

```bash
ffmpeg -y -i input.mp4 \
  -c:v libx264 -profile:v main -preset medium -crf 13 -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  output_discord.mp4
```

Key settings:

- `-pix_fmt yuv420p` — **required for Discord**, ComfyUI outputs `yuv444p` by default.
- `-crf 13` — high quality without massive file size (default 23 is too lossy).
- `-profile:v main` — widely compatible.

For multi-video crossfade stitching, chain `xfade` (video) and `acrossfade`
(audio):

```bash
ffmpeg -y -i a.mp4 -i b.mp4 -i c.mp4 \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=1:offset=3.04[v1];[v1][2:v]xfade=transition=fade:duration=1:offset=6.08[vout];[0:a][1:a]acrossfade=duration=1:c1=tri:c2=tri[a1];[a1][2:a]acrossfade=duration=1:c1=tri:c2=tri[aout]" \
  -map "[vout]" -map "[aout]" \
  -c:v libx264 -profile:v main -crf 13 -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  output.mp4
```

Offset for xfade #N = `(N+1) × duration - N × overlap`.
