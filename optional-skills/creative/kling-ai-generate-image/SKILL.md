---
name: kling-ai-generate-image
description: Generate images through the Kling AI MCP.
version: 1.0.3
author: William (@Wlain), KLING AI Pte Ltd; Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Kling-AI, MCP]
    category: creative
    related_skills: [kling-ai]
---

# Kling AI Image Skill

Turn the creative request into one precise, user-confirmed remote image generation. Follow the Hermes OAuth, billing confirmation, single-submission, status, and result rules in the [core Kling Skill](../kling-ai/SKILL.md).

## When to Use

Use for Kling image generation and controlled variations; use the core Skill for account, asset-library, or status-only requests.

## Prerequisites

Install the core `kling-ai` Skill alongside this Skill and authorize the single `Plugin-Hermes-kling-ai` server at `https://kling.ai/mcp/plugin` through Hermes-native OAuth. Follow the core connection and billing contract.

## How to Run

Read [prompt construction](references/prompt-construction.md) for reference images, exact copy, branded assets, or controlled variants. Read [scene patterns](references/scene-patterns.md) for product, advertising, portrait, poster, or cover work. For Element reuse or local media, also read the core [asset workflows](../kling-ai/SKILL.md#how-to-run).

## Quick Reference

| Intent | Action |
|---|---|
| Generate | Select the live image tool and confirm billable settings |
| Reuse assets | Read the core Skill and resolve references first |
| Check status | Query the known task number without resubmitting |

## Procedure

1. Use `text_to_image` for a new image. Use `image_to_image` when source media controls identity, product structure, composition, editable content, or style. An attachment alone does not determine the mode.
2. Ask only for missing information that materially changes the result: subject, destination and aspect ratio, required copy, reference roles, protected facts, or output count.
3. Create or reuse the UUIDv7 `taskTraceId` for this objective, call `who_am_i`, and select an explicitly compatible model and live arguments.
4. Upload local images first and use only the live input names and returned URLs. Never silently degrade to text-to-image after an upload failure.
5. Build one prompt covering subject, action, environment, composition, lighting, color, materials, camera language, protected facts, and necessary exclusions.
6. Show the final mode, model, prompt summary, resolution, aspect ratio, output count, and reference roles. Wait for explicit user confirmation.
7. Call the image generation tool exactly once and preserve the task number. If its MCP App mounts, let that one App self-refresh and do not call `query_tasks` from the model. If no App mounts, follow the core Skill's headless polling fallback; never resubmit.
8. When Hermes renders the MCP App, do not add Markdown media or duplicate links. When no App is rendered, provide one primary image link.

## Pitfalls

- Text-to-image does not use Elements.
- Image-to-image may use one or more images with explicit roles. State the purpose of every image when multiple references are present.
- Before using an Element subject, follow the core asset workflow, call `element_get`, and choose only an image-to-image model whose live schema explicitly supports `elements`. Bind the exact ID in both the prompt and structured `elements` argument.
- For a variant, lock every fact that the user did not ask to change. Change one major dimension per confirmed generation.
- A status request uses only `query_tasks`; it does not create an image.

### Quality defaults

- Prefer live-supported `2k` for normal delivery, `4k` for commercial work, advertising, fine materials, or crop-heavy post-production, and `1k` only for drafts, speed, or credit savings. Never lower a higher live default.
- Infer ratio from destination: `1:1` for square social or product media, `4:5` for portrait feeds, `9:16` for stories or vertical covers, and `16:9` for banners or thumbnails.
- Generate one image unless the user requests more.
- Unless the user explicitly requests text inside the generated image, prefer a clean image with intentional copy-safe space.
- Treat brand names, logos, faces, product structure, and user-provided copy as protected facts. Never invent performance, price, certification, ingredients, awards, effects, or statistics.

Before submission, internally check focal subject, visual hierarchy, safe area, lighting, reference roles, and conflicting instructions. Claim visual QA only when the result was actually inspected.

## Verification

Confirm reference roles, live-schema compatibility, and final billing approval before the single submission. Report success only when the original task finishes with usable primary media; a preview cover is insufficient for video.
