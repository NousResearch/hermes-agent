# Implementation boundaries

The [scope ledger](../downstream/scope.md) maps every approved area to its owner.
Native restores use the exact Git blobs from upstream starting commit
`6e69a8933adda7dbbff7cf3009a259a4524477e9`; remaining hunks must implement a listed
exception. No upstream update is included in this narrowing pass.

Restored native areas: context serialization/persistence, provider replay/token
accounting, message mirroring, session ID resolution, browser CLI/provider/cloud
resolution, CLI dispatch, administration endpoints, configuration access and
SOUL/identity behavior. Prompt assembly retains only knowledge pointers,
responsibility discovery and removal of global user-memory injection.

Retained implementation: responsibility packages on native cron/webhook owners;
responsibility-aware file validation; person-aware native memory storage;
Hindsight plugin using native memory lifecycle; unified knowledge review;
send/list messaging; fixed model tool filter and disabled skills; runtime
settings; video route; deployment and repository workflow.

Client controls are hidden without denying native backend calls. Hindsight's
config.yaml schema read/write support remains so native administration can
configure the added provider; provider choice is not forced in core.

Personal context is appended through native string sidecars or native
multimodal text parts. Shared memory configuration and store operations remain
native. Background consolidation is a deliberate review exception.

Browser Use is selected in Railway config. There is no fork-specific persistent
browser profile manager. Whisper's native default model is baked into the image
and copied to the native cache on first boot without replacing existing state.

File keeping uses profile-local documents/repos alongside responsibilities and
connections, with a shipped guide and prompt pointer. Deployment review follows. Local checks do not prove real Codex
entitlement, Browser Use account behavior, public ingress or backup/restore.
Responsibility filesystem code still requires POSIX (Linux/WSL2 or macOS).
