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
