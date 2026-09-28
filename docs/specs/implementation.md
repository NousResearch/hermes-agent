# Employee implementation map

The runtime is implemented directly in native owners, with focused modules for
employee contracts. Server deployment is a separate acceptance step: no Railway
project, account entitlement or production login has been tested here.

Runtime scope is Linux (including WSL2) and macOS. Native Windows is not supported
by this fork: the preserved responsibility filesystem relies on POSIX directory
handles and relative file operations. Upstream's native Windows support does not
extend to these employee additions. A Windows implementation and real Windows
acceptance tests are separate work from the Linux deployment specified here.

| Surface | Before | Implemented behavior / owner |
| --- | --- | --- |
| System prompt | Native persona, skill index and global personal profile | Employee wording, configured name/instructions, frozen manual/responsibility listings; `agent/employee_prompt.py` and native `system_prompt.py` |
| Tools | Native broad default toolsets | Employee allowlist and configured MCP; `model_tools.py`, `toolsets.py`, native registry |
| File operations | Generic file feedback | Source responsibility budgets, validation, gauges and warnings on native reads/writes/patches; `responsibilities/`, `tools/file_*` |
| Guides | Skill-based instructions | Shipped guides read fully by file tools; profile/work paths rendered; product files read-only |
| Authored memory | Global `USER.md` in system prefix | Fixed shared store plus stable person files; fresh profile after current user input, before recalled memory and hooks; `agent/people.py` |
| Replay | String sidecar; injected multimodal text stored as user content | String and list sidecars use native SQLite content encoding; clean transcript stays separate |
| Background memory | Configurable plugin surface | Bundled Hindsight with employee policy; model sees `recall`, automatic retain/prefetch uses source mechanics |
| Review | Native memory/skill review variants | Unified employee review prompt on native lifecycle, knowledge-only writes; no messages, delegation or declaration changes |
| Schedules | Model cron tool | Declaration reconciliation into native cron, fresh package context, guarded cadence, completed one-shots preserved |
| Webhooks | Native static/dynamic subscriptions | Responsibility declarations on native HTTP listener, signatures/handshakes, durable bounded queues and stable stream conversations |
| Messaging | Tool unregistered; transcript mirroring | Exposed text send/list, native delivery and authorization, next-turn confirmed delivery context |
| Session references | Full internal IDs | Compact suffix handles; native store resolves uniquely, collisions fail closed |
| Browser | Local/cloud/managed selection | Direct Browser Use Cloud, durable profile, native CDP/session lifecycle |
| Deployment | Generic native image | First-boot employee config, native dashboard/gateway, private Codex endpoint, pinned Hindsight policy and bank reconciler |

## Wording preservation

The checked-in [guide diff](reference/guide-adaptations.diff) compares source
wording with the shipped guides. Path/name substitutions are normalized out so
necessary runtime changes are visible. The connections guide uses the explicitly
approved shorter version. Other changes remove concrete contradictions with
native attachments, delegation, steering, filesystem persistence and
administration. Credential references use native `env:NAME` secrets instead of
the hosted credential store. Guides never direct the model to the source repo.

The employee identity drops the source product/company names. Working, memory,
file-keeping and review doctrine are copied; only paths and concrete native
contracts change. Native environment/project hints remain native. Shared memory
stays in the frozen system prefix; personal memory never enters that prefix.
`employee.instructions` replaces source organization-specific persona text.

## Configuration

Product rules are code-owned. Operational values remain native settings, with
[deployment seed configuration](../../deploy/railway/config.yaml) written once.
The native Config editor controls `employee.name`, `employee.instructions`,
`employee.owner`, `employee.identity_links`, main model and channel policy.
Use native secret management; [deployment instructions](../../deploy/railway/README.md)
define which secrets belong in Railway and which in the profile.

## Evidence and limits

Local integration checks exercise real native file/cron stores, actual SQLite
replay, person and browser profile isolation, authenticated HTTP webhook ingress,
actual Hindsight client requests to a local service, and private Codex protocol
translation against a local SSE server. They do not prove real model entitlement,
cloud browser account access, or Railway routing/backup behavior. The pinned
Hindsight image builds and starts with a disposable pgvector database; migrations
and the health endpoint pass. Those checks are listed in the deployment
acceptance procedure and require the future server.

## Surface cleanup

The fixed policy in `agent/employee_policy.py` also governs native slash-command
registration, skill discovery/loading/seeding, automatic curator/sync work and
Kanban workers. Upstream implementations remain in place for merges; their
employee entry points are closed. Dashboard Skills routes are unmounted.

Cron's dashboard keeps inspection, pause/resume and manual execution. Native
create/edit/delete and blueprint authoring are rejected; edit responsibility
files instead. The CLI retains cron operations but rejects those authoring
subcommands. Skill/bundles/sync/curator/Kanban CLI groups are absent.

Legacy personality commands, overlays and the SOUL editor are unavailable.
Use `employee.name` and `employee.instructions` in the native Config editor.
Memory administration exposes only Hindsight, reports per-person memory sizes,
and permits only shared-memory reset; it does not erase people through the old
USER.md reset control. Individual files remain under `memory/people/`.

Hindsight uses current-turn recall (`recall_sync: true`): each substantive message
queries memory before its reply, including the first turn. This deliberately
replaces previous-message background recall and adds recall latency. Four-turn
retention batches, native trivial-message skipping, context placement and warm
conversation caching stay unchanged. The real-client timing test checks the
employee policy as well as the retained native asynchronous mode. Native
project/environment prompt additions remain unchanged.

Retired slash commands and their aliases are rejected at CLI, gateway and TUI dispatch, including automation suggestions. Memory setup offers only Hindsight.

Dedicated TUI/desktop `skills.manage` and `cron.manage` RPCs enforce the same exclusions; TUI command entries and desktop routine create/delete controls are hidden.

Kanban is excluded from dashboard plugin discovery, the plugin hub, and API mounting, so dashboard dispatch cannot bypass gateway worker policy. Profile creation rejects retired skill-install payloads.
