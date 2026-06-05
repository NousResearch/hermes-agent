# SitioUno Derived Commercial Agent VM Runbook

This document is the repo-side pointer for configuring commercial derived agents cloned from Zeus.

Canonical long-form manual lives in Jean's wiki:

- `/home/jean/wiki/concepts/derived-commercial-agent-vm-runbook.md`
- Obsidian link: `[[derived-commercial-agent-vm-runbook]]`

## Core rules

- One commercial agent = one single-tenant VM.
- Zeus is the prototype/orchestrator; derived agents do not inherit Zeus memories.
- Infisical is the source of truth for secrets, with one project/identity per agent.
- Agent core is local: Agent Core SQL plus local modules/tools/skills.
- Twenty/Odoo/ERP/email/payment/social publishing are optional adapters, not core.
- WhatsApp and Telegram are the primary UX; dashboards/workspaces are private secondary surfaces.
- `*-sandbox.kidu.app` is the public delivery sandbox/placeholder surface, not the private workspace/dashboard.
- Dashboard/private workspace stay behind Tailscale/VPN and password.

## Mandatory local core modules

A derived commercial agent should have these local schemas/tools:

- `agent_core`
- `crm`
- `sales`
- `schedule/calendar` (Nettu)
- `factory` where progress/project tracking is used
- `signature`
- `accounting`

## Mandatory chat business toolsets

`platform_toolsets.telegram` and `platform_toolsets.whatsapp` must include the channel adapter plus the business toolsets:

- `calendar`
- `crm`
- `sales`
- `factory`
- `file`
- `web`
- `vision`
- `video`
- `terminal` only scoped to the agent's own VM
- `todo`
- `skills`
- `session_search`
- `memory`
- `cronjob`
- `messaging`
- `tts`
- `image_gen`
- `minimax_cli` when present

## Current Bael-derived lessons

Bael showed the important separation:

- Private workspace: `100.83.146.125:8080`, nginx Basic Auth.
- Public delivery sandbox: `bael-sandbox.kidu.app -> 100.83.146.125:9323`, `/api/* -> :9324`.
- Business chat toolsets for Telegram/WhatsApp must include CRM/Sales/Calendar/Signature/Accounting/Factory/Video, not just the platform adapter.

Do not route public sandbox domains to private workspaces.

## Implementation note

CRM/Sales/Factory/Calendar, Signature Core, and Accounting Lite Core now exist as repo/runtime modules in the Zeus feature branch. For new agents, deploy the branch/commit that contains:

- `tools/signature_tool.py`
- `tools/accounting_tool.py`
- `db/modules/signature/`
- `db/modules/accounting/`
- `toolsets.py` entries for `signature` and `accounting`

Video analysis has two layers:

- `video_analyze`: requires a real vision/video-capable provider. Bael uses Gemini via `auxiliary.vision.provider=gemini` and `model=gemini-3-flash-preview`.
- `video-intel-pipeline`: local skill/runtime pipeline for yt-dlp + ffmpeg + faster-whisper + hybrid search.

Video generation is separate:

- `video_generate` requires a configured provider/plugin such as FAL with `FAL_KEY`. Do not mark it complete without a real provider credential and smoke test.

Do not treat skills alone as sufficient; the agent must have schemas plus registered tools plus smoke tests.
