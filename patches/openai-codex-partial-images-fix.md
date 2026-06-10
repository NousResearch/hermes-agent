# Fix: openai-codex image-gen — sharper images via partial-frame handling

**File:** `plugins/image_gen/openai-codex/__init__.py`
**Patch:** `openai-codex-partial-images-fix.patch` (apply with `git apply` from repo root)
**Date:** 2026-06-08
**Status:** validated against the live SEO Hermes backend (see Investigation); deployed to the VM.
Pending upstream PR + Ansible role update.

## Symptom

Images from the Codex-OAuth backend (`gpt-image-2` via the Responses `image_generation`
tool) were often soft / half-rendered — worst on organic textures, complex poses, hands, and
multi-character scenes. Flat/voxel/glossy subjects looked crisp; photoreal-ish organic looked
mushy.

## Investigation (what actually happens on this backend)

The ChatGPT/Codex `backend-api/codex` Responses proxy, in this deployment, delivers the image
**through `partial_image_b64` streaming frames** — the final `image_generation_call.result`
is frequently empty or absent. Proof: setting `partial_images: 0` made *every* generation
fail with `Codex response contained no image_generation_call result` (~95-170s), i.e. with
partials disabled there was no image at all.

So pre-fix we were almost always saving a **partial (intermediate diffusion) frame** as the
final PNG. With `partial_images: 1` you get a single, early preview frame:

- Simple subjects (flat 2D, voxel, glossy plastic) converge fast → the 1st partial already
  looks ≈ final → crisp.
- Complex organic / faces / hands converge slowly → the 1st partial is still mid-diffusion →
  soft.

That single mechanism explains the whole "flat-sharp / organic-soft" split observed while
authoring the quiz deck. (Model anatomy limits — extra fingers, fused limbs — are a *separate*
issue and are addressed with prompt framing, not here.)

## Fix

1. **`partial_images: 3`** (was `1`). The API streams progressively higher-quality previews;
   requesting the max means the **last** partial is far closer to the finished image →
   markedly sharper output even when no true final `result` arrives.
2. **`_extract_image_b64` returns `(final_b64, partial_b64)` separately**, tracking the
   *latest* partial (the sharpest one) distinctly from the final result.
3. **`_collect_image_b64` returns `final_b64 or partial_b64`** — always prefer a true final
   result if the backend ever sends one; otherwise fall back to the sharpest partial. This
   also fixes the original ordering bug where a partial could overwrite a real final.

Net: best available image every time (true final when present, else the sharpest partial),
and never an early low-quality frame when a better one exists.

## Deploy (manual, applied to the SEO VM)

```bash
# deployed plugin path on hermes-mojapteczka-seo:
#   /home/hermes/.hermes/hermes-agent/plugins/image_gen/openai-codex/__init__.py
cp __init__.py __init__.py.bak.2026-06-08
# apply the 3 edits (git apply <patch>, or hand-edit), then reload the gateway:
systemctl --user restart hermes-gateway.service   # serves 127.0.0.1:8642
```

Note: the deployed VM copy is an older revision (no `reference_images` support, ~76 lines
shorter), so the `.patch` line offsets won't match — apply the three logical edits by content,
not by line number.

## Follow-ups

- **Upstream PR:** include a unit test feeding a synthetic SSE stream (3 partials, optional
  final) and asserting: returns final if present, else the *last* partial; never an earlier
  partial.
- **Ansible:** ship the patched plugin (or template the `image_gen` tool config with
  `partial_images: 3`) in the hermes role so re-provisioning keeps the fix.
- Consider logging `revised_prompt` from the stream (the Responses API may auto-rewrite the
  prompt) to debug prompt-adherence drift.
- Open question: why is the final `image_generation_call.result` usually empty on this proxy?
  If a config/headers tweak makes it return true finals, that would beat any partial.
