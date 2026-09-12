---
name: codex-vision
description: "Use Codex CLI for image analysis and generation — vision tasks that the Hermes vision_analyze tool can't handle. Covers MCP setup, exec workflow, image generation with the built-in image_gen tool, and iCloud File Provider placeholder pitfalls."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags:
      - codex
      - vision
      - image-generation
      - gpt-image
      - creative
      - generative-ai
      - mcp
    related_skills: [codex, comfyui, stable-diffusion-image-generation]
    category: creative
---

# Codex Vision — Image Analysis and Generation via Codex CLI

## Overview

Use the Codex CLI (`codex exec`) for vision tasks: analyzing images that the Hermes `vision_analyze` tool can't handle, and generating new images via Codex's built-in `image_gen` tool (GPT-image-2).

This skill goes beyond basic vision analysis. It covers:

- **Image analysis** — Detailed descriptions, QA, and understanding of local images
- **Image generation** — Generate new images using Codex's built-in `image_gen` tool
- **Identity-preserve generation** — Use reference images to maintain consistent identity across generated portraits
- **MCP server setup** — Register Codex as an MCP server in Hermes for code tasks
- **Hermes integration** — Proper handling of `codex exec` from non-TTY Hermes terminal sessions

**Auth is ChatGPT OAuth, not API key.** Codex uses ChatGPT Plus OAuth tokens stored in `~/.codex/auth.json` with `"auth_mode": "chatgpt"` and `"OPENAI_API_KEY": null`. This means you must have an active ChatGPT Plus subscription and run `codex auth login` at least once.

---

## When to Use

- Analyzing an image in detail (who/what, style, mood, colors, composition)
- Generating a new image from a text prompt
- Generating an image with identity preservation from a reference photo
- Generating portraits, editorial characters, maps, or scene illustrations
- QA of previously generated images for identity consistency or artifacts

**Don't use for:**
- Quick image lookups (use `vision_analyze` if the image is a URL)
- Image generation that doesn't need Codex's built-in tool (use `image_generate` / FAL for style-diverse generation)
- Video or multi-frame content

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Codex CLI** | `npm install -g @openai/codex` |
| **ChatGPT Plus** | Required for image generation (OAuth, not API key) |
| **Auth** | `codex auth login` completed — valid `~/.codex/auth.json` |
| **macOS `sips`** | For format conversion on macOS. On Linux, use `convert` from ImageMagick |
| **Hermes Agent** | The skill is designed for the Hermes agent framework |

---

## MCP Setup (Code Tasks)

Add Codex as an MCP server to Hermes so its code tools are available natively:

```bash
hermes mcp add codex --command codex --args mcp-server
```

**What you get:** `codex` (run a coding session) and `codex-reply` (continue a thread).

**Important:** The MCP server does **NOT** expose image generation. Use `codex exec` for all vision and image work.

**Auth:** If tokens are invalidated, re-auth with `codex auth login`.

---

## Image Analysis

### Quick Analysis

```bash
cd /path/to/images && codex exec \
  "Describe this image in detail — who/what is depicted, style, mood, colors, composition" \
  --image <filename.jpg> \
  --skip-git-repo-check
```

- `--skip-git-repo-check` is required when the working directory is not a git repo
- Codex auto-selects the configured model (typically GPT-4o or higher for vision)

### Cross-Directory Analysis

```bash
codex exec \
  "Describe this image in detail" \
  --image /full/path/to/image.jpg \
  --skip-git-repo-check
```

### Auth Failures

```
401 Unauthorized
Your access token could not be refreshed because your refresh token was already used.
```

→ User needs to re-authenticate: `codex auth login`

**MCP is not a bypass:** `mcp_codex_codex` uses the same auth path. When CLI OAuth is invalidated, MCP sessions fail with the same error.

---

## Image Generation

### Critical: Writable Sandbox Required

Codex runs in a read-only sandbox by default. For image generation (which writes to `$CODEX_HOME/generated_images/`), you **must** use `-s workspace-write`:

```bash
codex exec --skip-git-repo-check -s workspace-write --add-dir /target/dir \
  "Generate a portrait. Save to /target/dir/output.jpg"
```

If the target output directory differs from the workdir, include it via `--add-dir`.

### Basic Generation

```bash
cd /target/directory

codex exec --skip-git-repo-check -s workspace-write \
  --add-dir /target/directory \
  "Generate a photorealistic portrait of a woman in a Pacific Northwest forest at dawn. Save to /target/directory/output.jpg"
```

### Non-TTY Prompt Handling (Hermes Terminal)

In Hermes terminal sessions (non-TTY), `codex exec` may silently drop a positional prompt. **Always pipe via stdin** for reliable delivery:

```bash
printf '%s' "Generate a portrait. Save to /target/output.jpg" | \
  codex exec --skip-git-repo-check -s workspace-write \
    --add-dir /target/directory \
    -
```

This is critical for image generation runs triggered from Telegram, Discord, or other non-interactive Hermes channels.

### Output Path Resolution

Codex writes generated images to `$CODEX_HOME/generated_images/<session_id>/ig_XXXX.png` first, then you copy/convert to the desired final path:

```bash
# 1. Generate
codex exec --skip-git-repo-check -s workspace-write --add-dir /target/dir \
  "Generate a portrait. Save to /target/dir/output.jpg"

# 2. Find the latest session's output
SESSION=$(ls -lt "$HOME/.codex/generated_images/" | head -1 | awk '{print $NF}')

# 3. Convert (macOS sips)
sips -s format jpeg "$HOME/.codex/generated_images/$SESSION/ig_*.png" --out /target/dir/output.jpg
```

**Timeout:** Image generation takes 2–5 minutes. Use `timeout=300` in Hermes terminal calls.

---

## Identity-Preserve Generation

Use a reference image to guide identity (face, build, style) in generated output. The `image_gen` tool treats references as style/identity guidance — there is no explicit face-locking parameter, so results are non-deterministic.

### Verbose Prompt (Quality, Slow)

```bash
codex exec --skip-git-repo-check -s workspace-write \
  --add-dir /path/to/references \
  -i reference.jpg \
  --add-dir /path/to/output \
  "Use case: identity-preserve
Asset type: portrait
Primary request: A portrait of a woman with long curly hair on a cabin deck at dawn. Soft morning light, mist, mountains in the background.
Input images: Image 1: canonical reference — preserve exact face, face shape, hair, build, skin tone
Style: photorealistic, natural morning light, warm tones
Constraints: lock face identity from reference. Change only scene and clothing. No text, no watermark.
Output: save to /path/to/output/result.jpg"
```

### Brief Prompt (Fast, Iterative)

```bash
codex exec --skip-git-repo-check -s workspace-write \
  --add-dir /path/to/references \
  -i reference.jpg \
  --add-dir /path/to/output \
  "Generate a portrait: woman, curly hair, slim, on a morning deck with coffee, natural light. Reference: reference.jpg. Save to /path/to/output/result.jpg"
```

### Key Flags Reference

| Flag | Purpose |
|---|---|
| `--add-dir <path>` | Whitelist a directory for read/write access in the sandbox |
| `-i <file>` or `--image <file>` | Attach a reference image (must be in an `--add-dir` directory) |
| `-s workspace-write` | Enable writable sandbox (required for image generation) |
| `--skip-git-repo-check` | Skip git repo validation (required outside git repos) |
| `-` | Read prompt from stdin (required in non-TTY sessions) |

### Prompt Style Guide

| Use case | Prompt style | Expected time |
|---|---|---|
| Iterative exploration, pose variety | Brief single-line | ~60s |
| Final canonical portraits | Verbose structured | 2–3 min |
| Time-critical generation | Brief only | ~60s |
| Multi-person generation | Verbose or brief (test both) | Varies |

**Rule:** Start with brief prompts for iteration. Use verbose structured for final assets. If a verbose prompt times out (300s), retry with a condensed version.

---

## Time-Aware Scene Generation

Match the scene to actual local time for authentic lighting:

```bash
date '+%Y-%m-%d %H:%M %Z'
```

| Time of Day | Lighting & Scene Context |
|---|---|
| Morning (6–11 AM) | Soft sun, mist, golden/cream light, warm drinks, waking scenes |
| Midday (11 AM–5 PM) | Clear light, warm indoor sun, afternoon clarity |
| Evening (5–8 PM) | Golden hour, sunset, last warm light, shadows |
| Night (8 PM–6 AM) | Stars, darkness, candlelight, rain at night, artificial light |

Include the actual local time in the prompt for Codex to generate appropriate lighting and atmosphere.

---

## Messenger Delivery Workflow

For images requested over Telegram, Discord, or other messaging platforms:

1. Generate via `codex exec` with reference attached (if identity-preserve)
2. Copy/convert the output PNG to a stable project path
3. Verify with `file` / `ls -lh`
4. Run a quick `vision_analyze` QA pass for identity consistency, scene match, artifacts
5. Deliver with `MEDIA:/absolute/path/to/file` in the final response

Keep the reply concise: one grounded sentence about the scene + the `MEDIA:` line + any major QA caveat only if one exists.

---

## Common Pitfalls

1. **Verbose prompts timeout (300s).** Use brief prompts for iteration. Condense verbose prompts if timeout occurs.
2. **iCloud placeholders.** Files synced to iCloud may be "dataless" — correct file size but no readable pixel data. Always verify with `file <path>`.
3. **Forgot writable sandbox.** Default read-only sandbox prevents Codex from writing generated images. Always use `-s workspace-write`.
4. **Forgot `--add-dir`.** Without `--add-dir /target/dir`, Codex cannot write output to that directory.
5. **Auth token expiry.** `401 Unauthorized` means re-auth via `codex auth login`.
6. **`keep` database error.** Non-fatal. Codex proceeds without the reflection step. Ignore.
7. **Forgot `--skip-git-repo-check`.** Always include when outside a git repo.
8. **Non-TTY prompt dropped.** Pipe via `printf '...' | codex exec -` for reliable delivery in Hermes terminal sessions.
9. **Output in wrong directory.** Codex writes to `$CODEX_HOME/generated_images/<session_id>/` first, not the target directory. Copy/convert afterward.
10. **`vision_analyze` not available.** If Hermes `vision_analyze` returns errors on image input, use `codex exec` with `--image` flag for vision QA instead.

---

## Verification Checklist

- [ ] Codex CLI is installed and `codex auth login` has been run
- [ ] ChatGPT Plus subscription is active (required for image generation)
- [ ] Reference images are local files, not iCloud placeholders (verify with `file <path>`)
- [ ] All output directories are whitelisted with `--add-dir`
- [ ] `-s workspace-write` is set for any generation task
- [ ] `--skip-git-repo-check` is included when outside a git repo
- [ ] Prompts are piped via `printf '...' | codex exec -` in non-TTY sessions
- [ ] `timeout=300` is set for image generation terminal calls
- [ ] Output was copied from `$CODEX_HOME/generated_images/<session_id>/` to the final destination
- [ ] Generated image passes identity/scene QA before delivery
