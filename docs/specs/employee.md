# Employee behavior on native Hermes

Status: implementation started. Native runtime defaults are implemented; the
remaining work packages and live deployment checks are pending.
This is the entry point for the specification. Linked documents own detailed
contracts; they must remain consistent with the decisions summarized here.
Implementation proposals and unverified integration requirements are labelled.

## Purpose

This Hermes fork is an employee that owns work over time: it maintains standing
instructions and handoff state, uses services, executes scheduled/event-driven
work, reports verified outcomes, and incorporates corrections into future work.

Preserve the established employee model-visible behavior while retaining native
Hermes runtime owners. Integrate directly; do not import a hosted runtime and
build local adapters for it. Keep upstream merges manageable through focused
modules, small integration points and explicit downstream intent.

Self-hosted now means a Railway deployment, not necessarily a laptop. The product
has no hosted billing, account system, credential broker or custom cloud control
plane. One Hermes profile is one employee's knowledge boundary, with native
profile-scoped state and execution.

## Contract map

| Area | Settled direction | Authoritative detail |
| --- | --- | --- |
| Tools | `browser_exec`, `memory`, `recall`, `send_message`, shared work tools and `video_analyze`; remove skills, standalone schedule, connection-management and extra tools | [Tool surface](tool-surface.md) |
| Prompt | Configurable name; preserve employee wording except concrete local/runtime adaptations | [System prompt](system-prompt.md) |
| Knowledge | Responsibility packages, agent-owned service manuals, authored memory and shipped guides | [Guides](guides.md), [layout](local-layout.md) |
| Responsibility files | Exact established limits, read feedback, validation and actionable errors | [Tool surface](tool-surface.md#file-tools) |
| Automation | Employee file-authoring/run contracts through native cron and webhook ingress | [Execution](responsibility-execution.md) |
| Personal memory | Stable person records; append fresh current-person context after the current user message; freeze historical API bytes | [Identity](person-identity.md), [memory](tool-surface.md#authored-memory-and-personal-profiles) |
| Learning | Unified employee review prompt with native review mechanics and declaration-write restrictions | [Background learning](background-learning.md) |
| Conversations | Native framing plus person labels, compact session handles and confirmed outbound-delivery context | [Framing](conversation-framing.md) |
| Native behavior | Keep attachments, delegation, steering, Telegram adapter, service lifecycle and missed-run handling | [Tools](tool-surface.md#additional-native-behavior-decisions), [lifecycle](local-layout.md#runtime-lifecycle) |
| Administration | Native dashboard and Config editor; no custom group toggle or hosted-dashboard port | [Runtime defaults](runtime-defaults.md) |
| Services | Railway, Browser Use Cloud, local Whisper, self-hosted Hindsight; Codex, OpenRouter and Parallel as specified | [Deployment](deployment.md) |

## Preservation rules

“Same as the employee” governs user/model-visible behavior, not wholesale source
replacement. Explicit later decisions to keep native behavior take precedence:
attachments, delegation, steering and display must not inherit incompatible
reference prompt instructions. Other tool feedback leans toward the employee
where it matches the actual selected behavior. File budgets, personal-memory
placement and messaging parity have explicit contracts, not stylistic latitude.

Guides and references retain wording except necessary adaptations. The approved
shorter [connections guide](reference/connections-guide.md) is the deliberate
exception. It must be available from the spec without relying on gitignored
workspace notes. Hindsight has a [configuration snapshot](reference/hindsight-config.json);
endpoint and authentication substitutions are explicit, not silent retuning.

## Configuration and administration

Fixed product rules are authoritative in code: the employee tool surface,
knowledge organization, memory/context mechanics and review doctrine. Native
configuration still exists for operational preferences and deployment inputs.
The agent cannot change those product rules merely by editing configuration.
This is not a promise to hide files from an unrestricted terminal or prevent it
from editing source code; the strict two-folder sandbox proposal was withdrawn.

Use native dashboard controls for keys and settings, and its Config editor for
Telegram group/topic policy. No bespoke frontend or account-management product
is required. Display settings remain adjustable native settings. Provider keys,
Telegram credentials and owner identity are deployment inputs; fixed defaults
should minimize setup. Never hardcode secret values.

The native dashboard's public-access/auth deployment must be verified for Railway.
A proposed shared username/password must not be treated as an already-validated
public deployment. Preserve native auth constraints and do not remove them to
make hosting convenient.

## Implementation sequence

These are implementation work packages, not claims that code already exists.
Read the applicable area guides before editing each owner.

| Step | Change and likely native owners | Completion evidence |
| --- | --- | --- |
| 0. Validate provider routes | `hermes_cli/auth_codex*`, native provider/runtime paths, Hindsight inference integration, `plugins/browser/browser_use/`, image provider | Real Codex main-model request, Hindsight retain/recall/reflect and image generation; token refresh works. Browser reconnect preserves the selected cloud profile. No unapproved API-key fallback. |
| 1. Runtime policy and deployment foundation | `hermes_cli/config*`, native construction/toolset resolution, Docker startup, dashboard configuration, Railway service definitions | Fixed rules cannot be overridden through config; allowed preferences still work. Native gateway and dashboard start together; state survives redeployment. |
| 2. Knowledge and prompt | New focused responsibility modules; `tools/file_tools*`; `agent/prompt_builder.py`, `agent/system_prompt.py`; shipped `guides/` | Real create/read/patch/archive/restore; exact budget feedback; manual creation and persistence; removed skill/connection tools absent from actual schema; prompt diff accounted for. |
| 3. People and authored memory | `gateway/session*`, `tools/memory_tool*`, `agent/turn_context.py`, existing local persistence | Same person across DM/group; collisions distinguished; explicit local-owner link; fresh personal context in correct position; historical request replay unchanged. |
| 4. Responsibility schedules | `cron/job_definition.py`, job store, scheduler prompt/tick/script owners; file reconciliation | One file produces one job; last-good malformed-edit handling; fresh state; guard behavior; one-shot stays complete; archive stops work. |
| 5. Messaging and conversation context | `tools/send_message_*`, registry/toolsets; `gateway/run_turn*`, history assembly, session search/storage | Model can discover/send to valid targets; short handles resolve; confirmed out-of-session delivery appears once as new context at next turn; no cache-breaking transcript mirroring. |
| 6. Learning | `agent/background_review.py`, `agent/review_engine.py`, lifecycle triggers and write validation | One unified review prompt, native cadence; correct person/knowledge target; no sends/delegation/automation edits; repeated review improves existing records. |
| 7. Responsibility webhooks | `gateway/platforms/webhook*`, native subscription storage plus declaration reconciliation | File creates usable route; stable identity and lifecycle; authenticated ingress; source-compatible buffering/deduplication; correct model context and reporting. |
| 8. Integrated deployment | Native dashboard, container supervision, durable volumes, Hindsight config/bank reconciliation | Configure keys and group overrides through native administration; restart/redeploy/restore exercise; complete employee work scenario below. |

Resolve step 0 before depending on unproven subscription routes. Later work can
proceed in independent slices, but a full deployment is not complete without
those receipts. Prefer existing schemas and stores; settle new person/route
persistence by tracing native owners, not by creating a second hosted database.

## End-to-end acceptance

A permitted Telegram user assigns an area. The employee creates a responsibility,
uses a verified service and records its manual, produces an artifact, saves a
concise handoff and schedules its next action. A runtime restart does not lose
state or duplicate the job. The scheduled result is delivered, and a reply to
that result sees the delivered content. A correction updates the right record;
the next run uses it. Review consolidates knowledge without changing automation.
Archiving the responsibility stops its schedules and webhook ingress.

Repeat relevant paths with two Hermes homes (A → B → A) and two speakers,
including colliding display names. Verify profile isolation without claiming
that ordinary host filesystem access is sandboxed. Test actual model requests
and native tool/cron/gateway paths, not only mocked functions or source strings.
Python checks use `scripts/run_tests.sh`; UI behavior belongs in the owning TS
suite. No runtime suite is required for this documentation-only pass.

## Remaining engineering checks

These do not reopen agreed product choices:

- Codex compatibility and entitlement for Hindsight Luna and image generation;
  ownership of refresh credentials, concurrency and rate-limit recovery.
- Durable Browser Use cloud profile provisioning/reuse, including the current
  native provider version. Cloud is selected; exact profile setup is not yet proven.
- Hindsight pinned-image defaults, bank reconciliation and exact config parity.
- Railway service startup, volume paths/ownership, database backup/restore,
  authenticated dashboard access and public webhook routing.
- One authoritative source for each secret so Railway variables cannot silently
  override a key the user changes in the dashboard.
- Native MCP setup remains usable after removing `manage_connections`.
- Selected model capability/context sizes; choose an auxiliary vision route only
  if the selected main model requires one. No speculative fallback provider.

Material incompatibilities are reported with the specific failing contract.
Do not silently replace the selected service, weaken behavior, or import a
hosted compatibility layer to hide them.

## Upstream maintenance

Keep mechanical extraction separate from behavior changes. Reuse current native
fixes rather than replacing newer files with older reference copies. Add concise
[downstream intent](../downstream/README.md) entries when runtime divergences
land, with merge rules and behavior checks. The ledger records implemented guidance and runtime-default changes; future
work packages get entries when their behavior lands. Retire fork code
when upstream provides equivalent behavior.
