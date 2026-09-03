# Process-Edge Authority Architecture

Status: Phase G implementation contract
Architecture owner: #91911
Class tracker: #83565

## Decision

The closed class is not merely “credentials inherited by subprocesses.” It is:

> Ambient process authority crossing a principal, profile, role, or trust boundary.

An environment variable can encode a provider credential, gateway ownership,
Kanban lifecycle authority, session identity, profile provenance, loader
control, or routing state. The same name can be required on one edge and an
authority leak on another. Therefore a global denylist or `inherit_credentials`
boolean cannot represent the security decision.

The control-plane rule from #91911 applies directly:

> Coordinates select. Current proof objects authorize.

For process creation, the executable and environment are coordinates. A typed,
current process-edge contract authorizes what crosses.

## Invariants

1. Every non-terminal credential-bearing or full-profile child declares an
   immutable process intent and expected principal.
2. The live parent environment is never copied by an ordinary call site.
3. Gateway, session, Kanban, delegation, and other control-plane authority is
   denied unless the exact edge declares a grant.
4. Model-driving CLIs receive model authentication, not every tool or messaging
   credential owned by the active profile.
5. Trusted Hermes execution children may receive profile tool capabilities, but
   not gateway transport or parent lifecycle authority.
6. Vault CLIs receive a positive operational allowlist plus their own exact auth
   family and closed stdin.
7. Profile-scoped execution fails closed when multiplexing is active without an
   installed profile scope.
8. Caller overrides pass through the same policy and cannot reintroduce stripped
   authority after construction.
9. Spawn receipts contain policy identity and environment key names, never
   values.
10. The model-authored terminal path remains sanitize-by-default at the shared
    `_popen_bash` boundary.

## Typed intents

| Intent | Principal | Environment contract |
|---|---|---|
| `MODEL_DRIVER` | model-driving external CLI | safe OS baseline, exact provider auth, explicit edge grants |
| `TRUSTED_HERMES_CHILD` | Hermes runtime | active profile view, explicit tool grants, no parent role authority |
| `INTERACTIVE_HERMES_PTY` | Hermes runtime | trusted-child contract with PTY stdin |
| `VAULT_CLI` | external credential tool | safe/network baseline plus BWS or OP auth family only |
| `SECRET_HELPER` | operator-configured helper | active profile view with transport/lifecycle authority removed |
| `CHECKPOINT_GIT` | local infrastructure | local OS baseline plus exact internal Git variables |
| `CONTAINER_CONTROL` | local infrastructure | sanitized non-provider control environment |
| `CONTAINER_IMAGE_BUILD` | local infrastructure | sanitized control environment plus six registry-auth variables |
| `PROBE` | local infrastructure | minimal baseline, closed stdin, no credentials |

## Explicit authority continuation

Kanban authority demonstrates why an edge contract is necessary:

- dispatcher → real worker: allowed;
- worker → Codex → `hermes-tools`: allowed by an explicit Kanban grant;
- worker → arbitrary nested `hermes chat`: denied;
- gateway → ordinary descendant: gateway ownership denied.

The grant is attached to the edge, not inferred from the presence of
`HERMES_KANBAN_*` in the parent environment.

## Current implementation boundary

`tools.child_process_authority` is the Phase G narrow waist. During migration it
uses the existing terminal sanitizer as an internal implementation primitive,
but production callers may no longer select
`inherit_credentials=True`, `scrub_secrets=False`, or remerge
`os.environ` themselves.

The first implementation slice migrates:

- compute-host, CLI-exec, Codex, Copilot ACP, and dashboard PTY children;
- CLI TUI, dashboard machine reroute, gateway restart watcher, dashboard setup,
  and slash-worker Hermes runtime children;
- Codex, repository-build, PID, and libc probes;
- TUI `shell.exec` and quick-command terminal children through the established
  sanitized terminal boundary;
- Bitwarden and 1Password fetch/probe children;
- operator command-secret helpers;
- checkpoint Git;
- Apptainer/Singularity lifecycle and image-build children; and
- the shared Docker/SSH/Singularity terminal `_popen_bash` boundary, including
  explicit stripping of Bitwarden and 1Password bootstrap authority.

An AST gate rejects recurrence of the retired ambient-authority escape hatches.
Every production module that imports the typed broker must also pass explicit
`env` and `stdin` arguments at each direct subprocess boundary; omission is a
CI failure rather than an implicit request for ambient inheritance.

## Interlocks

- #77027: shared model-authored terminal boundary; its unique current semantics
  are folded into `_popen_bash`.
- #91293: profile provenance, persistent runtime, and arbitrary-name
  cross-profile isolation; this remains the profile-boundary implementation
  owner and should compose with this broker rather than duplicate it.
- #92633: inherited gateway marker is not gateway ownership.
- #92416: nested children must not inherit Kanban lifecycle ownership.
- #92309: the exact worker → `hermes-tools` edge requires bounded authority
  continuation.
- #70372: Desktop/TypeScript process policy must consume the same intent/grant
  model in its own survivor.
- #91362: fixed-command workers are the reference for literal argv, executable
  validation, and credential-free command children.
- #77467: vault CLI full-environment sink; the `VAULT_CLI` intent is its
  canonical implementation.

## Remaining class-wide phases

The Python Phase G seam is necessary but not the final repository-wide closure.
The following are required before #83565 itself closes:

1. generate Python and TypeScript policy artifacts from one reviewed manifest;
2. route Desktop, Node bridges, plugin sidecars, installers, and descendants
   through typed intents;
3. structurally ban direct production spawn APIs outside approved broker
   modules in Python and JS/TS;
4. replace original provider secrets with audience-bound broker grants wherever
   lower-trust children do not require the raw credential;
5. add randomized child/grandchild canaries and mutation tests across every
   registered intent;
6. emit value-free runtime receipts at production spawn boundaries; and
7. execute the final independent sweep against one post-merge upstream `main`
   SHA with the boundary check enforced as required CI.

## Closure predicate

Close #83565 only when one immutable upstream `main` SHA proves:

- no production child environment is derived from live ambient state outside
  the broker;
- no production API exposes an untyped full-inheritance switch;
- every child edge declares principal, intent, profile, executable identity,
  grants, stdin policy, and descendant policy;
- each authority has provenance and an explicit permitted-edge set;
- Python and TypeScript policy hashes agree;
- canary and mutation witnesses fail when the boundary is weakened;
- no relevant regression remains expected-failure;
- all superseded work identifies the upstream commit that absorbed its unique
  behavior and attribution; and
- the boundary status is actually required on `main`.
