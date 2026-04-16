# Hermes Microsoft Teams Adapter Implementation Plan

> For Hermes: implement as a native gateway platform adapter with OpenClaw msteams feature parity as the benchmark.

**Goal:** Add native Microsoft Teams bot support to Hermes with architecture, config, tests, and docs aligned to Hermes conventions, then prepare the work for an upstream PR.

**Architecture:** Build a first-class `gateway/platforms/msteams.py` adapter plus minimal helper modules for Bot Framework ingress, Graph-backed outbound/mentions/media, and adapter-local state. Wire it through Hermes platform/config/runtime surfaces exactly the way official docs require. After each implementation round, check parity against OpenClaw msteams capabilities and fill gaps until complete.

**Tech Stack:** Python, aiohttp webhook server, Hermes gateway platform adapter model, Microsoft Bot Framework / Graph HTTP APIs, pytest fixtures.

---

## Source-of-truth parity target

OpenClaw references audited before implementation:
- `/Users/moatable/.nvm/versions/node/v24.12.0/lib/node_modules/openclaw/docs/channels/msteams.md`
- `/Users/moatable/.nvm/versions/node/v24.12.0/lib/node_modules/openclaw/dist/plugin-sdk/src/config/types.msteams.d.ts`

Parity areas to keep checking each round:
1. Inbound scopes: DM, group chat, team channel, thread reply
2. Outbound: text, mentions, media/files, cards/polls where feasible
3. Policies: DM policy, group policy, allowlists, mention gating
4. Routing: session mapping, reply style (`thread` vs `top-level`)
5. Auth/config: Azure bot creds, webhook config, Graph enrichments
6. Attachments/history: Graph-dependent gaps explicitly tracked
7. Local dev + testing flow: webhook, tunnel, Web Chat, Teams E2E

---

## Task 1: Add platform enum + config plumbing

**Objective:** Introduce `Platform.MSTEAMS` and make Hermes treat it as a first-class configurable platform.

**Files:**
- Modify: `gateway/config.py`
- Modify: `hermes_cli/platforms.py`
- Modify: `hermes_cli/gateway.py`
- Modify: `hermes_cli/status.py`
- Modify: `hermes_cli/config.py`
- Modify: `toolsets.py`

**Steps:**
1. Add `MSTEAMS = "msteams"` to `Platform` enum.
2. Extend config/env loading for Teams credentials and webhook options.
3. Add connected-platform detection logic for Teams.
4. Register `hermes-msteams` toolset and expose it via CLI platform registry.
5. Add Teams entries to gateway setup/status surfaces.
6. Add/adjust tests for platform registration and config loading.

**Verification:**
- Targeted pytest for gateway/config and CLI platform registry passes.

---

## Task 2: Implement minimal native adapter skeleton

**Objective:** Create a Hermes-native Teams adapter that can accept inbound Bot Framework activities and send outbound text replies.

**Files:**
- Create: `gateway/platforms/msteams.py`
- Create: `gateway/platforms/msteams_graph.py`
- Create: `gateway/platforms/msteams_mentions.py`
- Create: `gateway/platforms/msteams_state.py`
- Modify: `gateway/run.py`
- Modify: `agent/prompt_builder.py`

**Steps:**
1. Implement adapter class inheriting `BasePlatformAdapter`.
2. Add aiohttp webhook server for `/api/messages` style ingress.
3. Normalize Bot Framework activities into `MessageEvent` + `SessionSource`.
4. Implement outbound `send()` for DM/group/channel reply routing.
5. Add adapter factory branch in `gateway/run.py`.
6. Add platform prompt hint.
7. Add initial unit tests for event normalization and send routing.

**Verification:**
- Targeted pytest for adapter construction/event normalization passes.
- Local POST to webhook endpoint creates a normalized message event.

---

## Task 3: Wire authorization + delivery surfaces

**Objective:** Make Teams fully participate in Hermes gateway authorization, cron delivery, and send-message tooling.

**Files:**
- Modify: `gateway/run.py`
- Modify: `cron/scheduler.py`
- Modify: `tools/send_message_tool.py`
- Modify: `gateway/channel_directory.py` (if needed)
- Modify: `gateway/session.py` (if identity extensions needed)
- Modify: `tools/environments/local.py`

**Steps:**
1. Add Teams allowlist / allow-all env mappings.
2. Add Teams to adapter factory, update-safe platforms, and delivery maps.
3. Add one-shot send support for Teams in `send_message_tool.py`.
4. Blocklist Teams secrets from subprocess env leakage.
5. Add targeted tests for these registrations.

**Verification:**
- Cron/send-message tests recognize `msteams`.
- Authorization tests pass for Teams env vars.

---

## Task 4: Implement OpenClaw parity features in rounds

**Objective:** Iteratively reach feature parity with OpenClaw msteams.

**Files:**
- Modify adapter/helpers/tests/docs as needed.

**Rounds:**
1. Mentions + mention gating
2. Group policy / DM policy / allowlists
3. Thread/top-level reply style + session mapping
4. Chunking / typing / long-response handling
5. Attachments/media baseline
6. Graph-backed enrichments (`member-info`, history, richer mention resolution) where practical
7. Track any explicit non-parity items with rationale if upstream scope should defer them

**Verification each round:**
- Update parity checklist in docs
- Run targeted tests
- Compare implemented capabilities against OpenClaw source-of-truth list

---

## Task 5: Docs + PR readiness

**Objective:** Make the implementation reviewable and upstreamable.

**Files:**
- Create: `website/docs/user-guide/messaging/msteams.md`
- Modify: `website/docs/user-guide/messaging/index.md`
- Modify: `website/docs/reference/environment-variables.md`
- Modify: `website/docs/reference/toolsets-reference.md`
- Modify: `website/docs/developer-guide/architecture.md`
- Modify: `website/docs/developer-guide/gateway-internals.md`
- Add/modify tests under `tests/gateway/`, `tests/hermes_cli/`, `tests/tools/`, `tests/cron/`

**Steps:**
1. Document setup, Azure bot config, local tunnel workflow, and E2E validation.
2. Document current parity + any deliberate gaps.
3. Run targeted tests, then the full test suite.
4. Prepare a PR summary tying implementation directly to issue #9512.

**Verification:**
- `python -m pytest tests/ -o 'addopts=' -q`
- Docs mention Teams consistently across user and developer guides.

---

## Immediate execution order
1. Task 1
2. Task 2
3. Task 3
4. Task 4 (iterative)
5. Task 5

## Definition of done
- Hermes has a native `msteams` platform adapter
- OpenClaw msteams capability parity has been checked after every implementation round
- Remaining gaps, if any, are explicit and justified
- Tests pass
- The branch is in a state that can be proposed upstream as a PR for issue #9512
