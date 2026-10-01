# Fork divergence scope

This is the agreed boundary for changes to Hermes. It supersedes broader
contracts in the original employee implementation. Extend native owners; do not
add alternative runtime paths. Tests should exercise native behavior except for
the explicit differences below.

| Area | Decision and reason | Where / boundary |
| --- | --- | --- |
| Main prompt | Native baseline with guide/connection/file-keeping pointers and the unchanged full responsibility section in the skills-index slot. Native memory guidance stays; no added Hindsight paragraph. | `agent/system_prompt.py`; `employee_prompt.py` supplies knowledge pointers and the responsibility roster. Native identity/SOUL, task guidance and environment context remain. |
| Model tools | Keep the selected employee surface; avoid paying for unwanted tools. Configured MCP tools remain eligible. | `agent/employee_policy.py`, `model_tools.py`, `toolsets.py`. This is a model-tool filter, not a ban on native administrator commands. |
| Skills | Disabled in agent loading, invocation, sync and curator. | Small gates in native skill owners. Native implementation retained for upstream merges. |
| Responsibilities | Keep charters, state, references, scripts, archives and discovery. Consolidate schedules and webhooks into this one divergence area. | `responsibilities/`; native cron and webhook integration. Native administration remains, with file-owned jobs protected against conflicting edits. |
| File tools | Keep responsibility-specific validation, limits and feedback. Everything else uses native read/write/patch behavior. | `tools/file_*`, `responsibilities/files.py`. Guide reads use native pagination and budgets. |
| Guides | Native Hermes self-reference copied as a guide with necessary runtime edits, plus responsibility authoring, file keeping and connection setup. | `guides/employee`, `guides/responsibility-authoring`, `guides/file-keeping`, `guides/connections`. |
| Connections | Document every service connection, verification and operating procedure; scripts/references alongside. | Top-level connection guide, native service subguides and prompt pointer; manuals copy working access patterns, not setup instructions. Native service CLI/MCP/authentication. No connection-management tool. |
| Filesystem | Native Hermes layout and access, with profile-local filing roots. | `$HERMES_HOME/{documents,repos,responsibilities,connections}`. Native initialization adds documents/repos; no migrations, sandbox or custom temp/cache layout. |
| Authored/personal memory | Source-product memory tool description and personal/shared targets; native store operations, limits/configuration and approval mechanics, except person selection/loading and current-turn placement. Background review can consolidate as agreed below. | `agent/people.py`, memory tool/store, inline executor. Current-person context follows user content before recall. Existing person paths retained. |
| Context/replay | Native serialization, persistence, provider replay and token accounting. | Native string sidecars; native durable text-part handling for multimodal input. Only personal-context composition changes; Codex app-server receives current-turn personal/recall/plugin context at dispatch without changing native stored rows or history seeding. |
| Hindsight | Keep provider integration and employee policy; same-message recall and only `recall` exposed. | `plugins/memory/hindsight`; native memory-provider lifecycle, context wrapper and provider selection. Deployment selects Hindsight. |
| Background learning | Adapt native combined-review guidance to three distinct destinations: memory, connection manuals and responsibilities. Keep proactive learning, the memory-turn trigger and person attribution. Review stays due until a local fork starts. No sends, delegation or automation declaration edits. | `agent/background_review.py`, `employee_review.py`, review-scoped file checks. |
| Messaging tool | Keep send/list interface and target discovery. | `tools/employee_messaging.py`, schema/registration. Delivery, mirroring and adapters remain native; additional sends carry the interim marker so they cannot seal an active final-answer stream. |
| Conversation framing | Native sender framing, session IDs, transcript mirroring and delivery history. | Removed compact handles and deferred delivery queue. |
| Browser | Select native Browser Use through deployment settings. No changed lifecycle or provider implementation. | Reverted cloud resolver, browser CLI/provider and fork profile provisioning. |
| Video | Keep OpenRouter/Gemini video route. | `tools/vision_tools.py` and deployment settings. |
| Runtime defaults | Keep medium reasoning, 85% compression, three tail user messages, approvals off, shared group sessions and quiet Telegram/restart/transcript settings. | Native defaults and settings; operators can override. |
| Transcription | Local Whisper; bundle its default model in the Railway image. | Native transcription; Docker dependency/model cache and deployment settings. |
| Configuration | Native mutable configuration and credential handling. | Removed fork config/auth/guide write prohibitions. Review-only write scope remains. |
| CLI/admin APIs | Native commands, personality, memory configuration and endpoints. | Reverted command filtering, authoring/API rejection and custom doctor behavior. Skills remain disabled in agent runtime. |
| Dashboard/TUI/desktop | Hide selected native controls. Add the approved hosted settings UI over native config, auth and pairing owners. | [Hosted settings](../specs/hosted-settings.md): focused API, shared-admin rotation, chat-scoped silent topics and private Hindsight updates. Native administration remains available. |
| Deployment | Keep Railway, Codex inference and Hindsight provisioning. Review live deployment after code decisions settle. | `deploy/railway`, Docker/s6, packaging. No live deployment yet. |
| Repository workflow | Keep scoped AGENTS guidance, review script, specifications and this ledger. | `AGENTS.md`, `.codex`, `docs/`. |
| Tests | Prefer native tests; add focused coverage for deliberate differences. | Revert tests changed solely for removed restrictions. |

## Merge rule

Every runtime change must belong to an approved row. Restore unrelated code to
native Hermes. Prefer configuration for provider/default choices. Preserve the
native prompt cache lifecycle; never reload a warm prompt or rewrite history.
The agreed folder layout does not authorize moving or deleting existing data.

## Native reference

Restorations use upstream commit
`6e69a8933adda7dbbff7cf3009a259a4524477e9`, the fork's native starting point.
Compare restored paths with `git diff --exit-code <commit> -- <paths>`;
unchanged means identical Git content, including whitespace. Mixed files keep
only the exceptions in the table. This pass does not update the upstream base
or promise that later upstream changes will never conflict.
