---
name: kling-ai
description: Create and monitor Kling image and video generations.
version: 1.0.2
author: William (@Wlain), KLING AI Pte Ltd; Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Kling-AI, Image-Generation, Video-Generation, MCP]
    category: creative
    related_skills: []
---

# Kling AI Skill

Create images and videos through Kling AI's OAuth-protected remote MCP server. The Skill guides Hermes through safe generation, uploads, task monitoring, and result handoff without bundling a local MCP runtime.

## When to Use

- The user asks to create or transform an image or video with Kling AI.
- A generation needs an uploaded image, first frame, last frame, or other media input.
- The user asks for Kling credit information or the status of an existing task.
- A prior submission returned a `generationId` or `taskTraceId` that must be monitored.

Do not use this Skill for unrelated media providers or for editing media locally.

## Prerequisites

- Hermes Agent has one MCP server named `Plugin-Hermes-kling-ai`.
- The server points to the Kling AI Global endpoint `https://kling.ai/mcp`.
- The server uses Hermes-native OAuth. Never request an API key or expose credentials, cookies, authorization headers, private account fields, or signed URLs in logs.
- For the repository package, preserve `X-Kling-Integration: Plugin-Hermes` and `supports_parallel_tool_calls: false` from `mcp.config.yaml`. The official MCP catalog schema cannot currently encode these two optional fields.

Hermes owns OAuth dynamic registration and uses its native client identity. The integration header is telemetry-only and must not affect authorization, billing, or rollout.

## How to Run

Discover the live tools exposed by `Plugin-Hermes-kling-ai` and use their current schemas. Hermes normally exposes them as `mcp__Plugin_Hermes_kling_ai__<tool>` after sanitizing the server key; never infer model names, enums, or required fields from this Skill.

Read [references/tool-workflows.md](references/tool-workflows.md) before a generation call. Read [references/troubleshooting.md](references/troubleshooting.md) only after an authorization, schema, upload, or provider failure.

## Quick Reference

| Intent | Action |
|---|---|
| Authorize | `hermes mcp login Plugin-Hermes-kling-ai` |
| Check connection | `hermes mcp test Plugin-Hermes-kling-ai` |
| Generate | Confirm final billable settings, then call one live generation tool once |
| Monitor | Call the live `query_tasks` tool once |
| Ambiguous timeout | Query existing tasks; never submit again blindly |
| Present result | Render the returned MCP App resource when supported, otherwise use the same response's text fallback |

## Procedure

1. Identify the requested generation or read-only operation.
2. Ask only for missing creative requirements that materially affect the result.
3. Upload attached or local media first when the live workflow requires it, and preserve the returned provider reference exactly.
4. Treat generation as a credit-consuming write action. Show the final workflow, model, duration and resolution or aspect ratio, then obtain explicit confirmation immediately before submission unless the current user message explicitly authorizes immediate submission with final settings.
5. Call the selected generation tool at most once per approved intent. Never automatically retry a failed, timed-out, or ambiguous submission.
6. Preserve and report the exact `generationId` and any `taskTraceId` returned by the provider.
7. If status checking is needed, call the live `query_tasks` tool. Do not invent a local result or claim that a card will refresh itself.
8. If Hermes supports the returned MCP App resource, let it render that resource. Otherwise report the same call's text fallback and one link to the primary output when available; do not synthesize or duplicate media.

Use these defaults only when the user did not specify alternatives and the live schema supports them:

- Video resolution: `720p`
- Video duration: `5` seconds
- Text-to-video aspect ratio: `16:9`
- Image-to-video aspect ratio: derive from the first frame unless required

## Pitfalls

- **Legacy endpoint:** remove an older non-Global connection before adding the Global endpoint, then complete OAuth again.
- **Blind retries:** a lost response does not prove the credit-consuming request failed. Query existing tasks first.
- **Stale schema assumptions:** refresh the live schema and revise only unsupported fields.
- **Parallel generation calls:** keep parallel tool calls disabled because submissions and reads may share account/task state.
- **Invented presentation:** use only the resource or text fallback from the original provider response.

## Verification

- `hermes mcp list` shows exactly one `Plugin-Hermes-kling-ai` server.
- `hermes mcp test Plugin-Hermes-kling-ai` succeeds after OAuth authorization.
- A generation is submitted only after approval and only once.
- The result preserves `generationId`, `taskTraceId` when present, and the provider's text/resource fallback.
- On provider failure, report the provider message and preserve identifiers without resubmitting.
