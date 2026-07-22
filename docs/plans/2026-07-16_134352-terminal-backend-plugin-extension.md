# Hermes Agent Extensible Terminal Backend Architecture Plan

## Goal

Allow third-party Hermes plugins to register custom terminal backends and select one through `terminal.backend` as the shared underlying execution environment for terminal, file, process, code execution, and related capabilities. Third parties must not need to modify Hermes Agent source code or expose additional tools.

This plan defines only the extension mechanism, runtime boundaries, and migration path. It does not design any specific backend implementation or prematurely decompose the work into individual source files or line-level changes.

## Current Architecture

### Key constraints of the existing terminal backend design

- `BaseEnvironment` already provides a stable execution protocol: a backend produces pollable and terminable process handles and releases resources, while the base class uniformly handles shell snapshots, cwd continuity, stdin, timeouts, interrupts, and output collection.
- Backend selection and construction currently use a centralized hard-coded branch. Adding a backend requires changing that branch and also accounting for backend-name sets scattered across configuration parsing, container path handling, remote-environment prompting, approval policy, and UI options.
- Terminal, file, and code execution share the same environment instances, but file and code execution reuse them by importing the terminal implementation's internal caches, locks, and constructor. Environment creation is therefore not yet an independent, stable host service.
- Environment instances are created and cached per task/session, cleaned up after inactivity, and cleaned up again on process exit. Plugins should therefore register backend definitions or factories, not global `BaseEnvironment` instances.

### Comparison with specialized plugin types

| Type | Discovery and registration | Selection and lifecycle | Implications for terminal backends |
|---|---|---|---|
| Model provider | Lazy discovery; modules register profiles in a central registry; lookup by name or alias | Profiles hold almost no runtime resources; overrides are allowed | Reuse named lookup, lazy discovery, and a metadata registry; do not reuse the permissive override policy |
| Gateway platform | `PluginContext` registers a descriptor containing a factory, availability checks, and configuration/display capabilities; deferred loading is supported | Adapters are instantiated from platform configuration; the registry does not hold active connections | **Closest match**: register descriptors and factories rather than instances, and keep capabilities and setup metadata in the same extension contract as the factory |
| Memory provider | Independent directory scanning with a simulated collector; configuration selects one provider | A single instance has initialize/shutdown and per-turn lifecycle hooks | Reuse explicit single-provider selection and a complete lifecycle; do not create another independent scanner or source-text heuristic |
| Context engine | Independent loader with a simulated `PluginContext`; only one engine may be active | Configuration loads a single instance | Specialized collectors tend to diverge from the general plugin system; terminal backends should integrate directly with the standard `PluginContext` |
| Image-generation backend | General plugin loading registers provider instances in a specialized registry | Configuration selects the active instance, with simple fallback rules | Reuse a specialized registry and explicit fallback semantics; do not register shared instances because terminal environments must be created and reclaimed per task |

Use a combined model: **platform-style descriptors and factories, context/memory-style exclusive configuration selection, model/platform-style lazy discovery, and the existing environment instance lifecycle**.

## Target Architecture

### 1. Establish a Terminal Backend Registry as the narrow waist

Introduce a host-owned terminal backend registry. The registry stores only immutable backend definitions, never active environments.

Each backend definition must describe at least:

- A stable canonical backend name and a human-readable label.
- A factory that creates `BaseEnvironment` instances.
- A side-effect-free availability/dependency check.
- Backend capabilities used for host decisions instead of name-based checks.
- A configuration schema, installation guidance, and diagnostic metadata.
- Source information such as built-in versus plugin and the owning plugin, for status display and conflict diagnosis.

The registry supports lookup by name, listing, diagnostics, and test reset. Creation, caching, and cleanup of active environments remain outside the registry.

### 2. Register backends through the standard PluginContext

Add terminal backend registration to the standard plugin context. A third-party plugin submits a backend definition from its existing `register(ctx)` function. No additional entry-point group, directory scanner, or source-text heuristic is needed.

Loading semantics:

- User-installed backend plugins remain gated by the existing `plugins.enabled` trust boundary. Installation alone must not execute code.
- General plugin discovery must complete idempotently before backend lookup across CLI, gateway, cron, TUI/API, and direct tool-call startup paths.
- Registration declares only a descriptor and factory. It must not connect to a remote service, create a sandbox, or initialize a backend.
- Plugin removal, disablement, or configuration changes do not hot-swap existing environments in-process. Changes take effect on the next clean startup. Tests may use an explicit reset.

`plugin.yaml` continues to use the general `kind: backend`. The registration API actually invoked by the plugin determines whether it is a terminal backend; no competing manifest kind is introduced.

### 3. Register built-in and plugin backends in the same registry

Register existing built-in backends as host-owned definitions in the same registry while preserving their current names, defaults, construction behavior, and error semantics.

Selection rules:

1. When `terminal.backend` is explicitly configured, select only a registered backend with that name. If it is unregistered or unavailable, return a precise error and do not silently switch to local.
2. When no backend is configured, continue to default to local for compatibility.
3. Compare backend names through one normalization rule while retaining the canonical name in the registry for display.
4. Reject duplicate names by default, especially attempts by third parties to silently override built-in backends. If overrides are ever required, they must use a separate explicit high-trust authorization rather than model-provider-style last-writer-wins behavior.

### 4. Promote environment instance management to a host service

Create an independent Environment Manager as the single shared environment entry point for terminal, file, process, code execution, and other callers. It is responsible for:

- Resolving the effective backend, cwd, timeout, and task overrides for a task/session.
- Creating environments through registry factories under a per-task creation lock.
- Maintaining active instances, last activity, idle cleanup, and process-exit cleanup.
- Notifying dependent layers to invalidate derived caches such as file-operation wrappers after environment reclamation.
- Returning the same environment instance to all underlying tools for the same task.
- Converting factory or initialization failures into consistent diagnosable errors carrying backend and plugin identity.

File and code execution then stop depending on terminal-tool private globals and private constructors. Any plugin backend that satisfies the environment contract automatically serves every tool that depends on that environment.

### 5. Define a stable Factory Request instead of expanding positional arguments

The host passes a structured creation request to a factory in three layers:

- Common runtime parameters such as backend name, task ID, cwd, timeout, profile, and Hermes home.
- Host-resolved common terminal configuration and task overrides.
- Namespaced options for the selected backend.

Plugin-specific configuration should live under `terminal.backends.<backend-name>`, while common fields remain at the top level of `terminal`. Adding a backend then does not require adding vendor-specific keys to the core environment-variable bridge or a hard-coded parser table.

Configuration principles:

- The factory receives only the selected backend's configuration slice, not the entire Hermes configuration.
- Secrets continue to use Hermes secret/environment mechanisms and are never copied into logs or status output.
- Backend-definition schema and availability metadata drive setup flows, dashboard selection, status, and error guidance.
- An explicitly configured backend with missing dependencies or credentials does not fall back, so the user receives precise remediation guidance.

### 6. Replace backend-name sets with a capability model

Gradually replace core checks of the form "if the backend name is in this set" with registry capability queries. At minimum, capabilities cover:

- Execution locality: host-local or remote.
- Filesystem/path semantics: host filesystem, isolated POSIX filesystem, or shared/mounted host filesystem.
- Cwd policy: whether host cwd is accepted and whether sandbox cwd mapping is required.
- Image/resource model: whether image, CPU, memory, disk, and similar common sandbox parameters are accepted.
- Optional process, PTY, and file-transfer support.
- Persistence/reuse properties and host-access state.

Capabilities drive prompting, path conversion, configuration presentation, and tool adaptation, but must never allow a plugin to grant itself additional security privileges.

### 7. Keep security boundaries under host control

Terminal backends are a higher-risk extension point than ordinary providers because factories run in the Hermes process and backends carry command and file access.

Therefore:

- User plugins must be explicitly enabled. Installation does not imply trust.
- Custom backends use a conservative security policy by default: assume possible host access and keep dangerous-command approval enabled.
- A plugin's `isolated` declaration may affect display or path compatibility, but cannot by itself bypass approval.
- Security-sensitive capabilities such as approval bypass, host mounts, and built-in backend overrides must be host-verifiable or explicitly authorized by user policy.
- The registry validates names, factory callability, and definition structure during registration. A factory result must be a `BaseEnvironment` instance before it enters the active cache.
- Plugin failures are isolated at discovery, registration, and factory boundaries and include source identity in diagnostics. They must not trigger fallback to the more dangerous local backend.

### 8. Lifecycle contract

Keep `BaseEnvironment` as the core runtime protocol and define these stages:

1. **Discovery**: load enabled plugins and register backend definitions without creating backend resources.
2. **Resolution**: read `terminal.backend`, resolve the definition, and check availability and configuration.
3. **Creation**: under a per-task lock, the Environment Manager invokes the factory, which returns a ready-to-use `BaseEnvironment`.
4. **Use**: terminal, file, process, and code execution share the same environment and continue to rely on BaseEnvironment's unified execution semantics.
5. **Cleanup**: the Environment Manager calls `cleanup()` exactly once on idle expiry, explicit session teardown, or process exit. Cleanup failures are logged without blocking reclamation of other environments.

Existing backends may continue to connect, synchronize, and call `init_session()` during construction. Initialization can be made explicit later, but it is not a prerequisite for third-party registration; avoiding it keeps this architecture change from simultaneously rewriting every backend lifecycle.

## Incremental Migration with `EXP_BACKEND`

The migration architecture has two layers:

- **Temporary façade/router**: selects the legacy or new runtime according to `EXP_BACKEND`; remove it after the refactor.
- **Permanent Registry and Environment Manager**: provide plugin extension, environment lifecycle, and shared environments across tools; retain them after full cutover.

The final cutover removes only the feature-flag router and legacy adapter. Terminal, file, process, and code execution ultimately depend directly on the new Registry/Environment Manager API.

### Single migration façade

Gradually route all backend-related calls through one host-owned runtime façade. It exposes domain operations rather than leaking the legacy implementation's dictionaries, locks, or private factory. Its minimum responsibilities are:

- Get or create an environment for a task.
- Query effective backend identity, active environments, and host-trusted capabilities.
- Register, read, and clear task overrides.
- Mark activity and perform task-level and global cleanup.
- Invalidate derived caches such as file-operation wrappers with the environment lifecycle.
- Provide a read-only snapshot for status and diagnostics.

The migration provides two implementations:

- **Legacy runtime adapter**: a thin wrapper over existing functions, caches, and locks, with no behavioral rewrite; used when `EXP_BACKEND` is not enabled.
- **Registry runtime adapter**: uses the new Registry, Factory Request, Environment Manager, and capability model; used only when `EXP_BACKEND=1`.

Callers must not inspect the selected adapter and branch independently, or bypass the façade to access legacy globals. The feature flag may appear only at the composition root, not throughout terminal, file, process, or code execution.

### Flag semantics

- `EXP_BACKEND=1`: use the registry runtime for the entire process.
- Unset or `0`: use the legacy runtime for the entire process.
- Any other value: log one clear warning and use legacy, preventing a typo from accidentally enabling the experimental execution path.
- Read and freeze the runtime selection on first process initialization. Later environment-variable changes do not switch a running process or mix legacy and new environments in one process.
- Startup logs, status, and dump output identify the active runtime as `legacy` or `experimental-registry` for feedback and rollback diagnosis.
- `EXP_BACKEND=1` controls only the host runtime. It does not change `terminal.backend`, which continues to select local, Docker, or a third-party backend.

During migration, third-party terminal backends are available only with `EXP_BACKEND=1`. If the legacy runtime receives a third-party backend name, preserve legacy unknown-backend failure semantics and add guidance that third-party backends require `EXP_BACKEND=1`. Never fall back to local.

### Validation strategy: no dual execution

Terminal backends create containers or remote sandboxes, execute commands, and write files. Migration validation must not execute legacy and new paths simultaneously, avoiding duplicate side effects, resource leaks, and security risk.

Validation uses:

- The same side-effect-free contract tests run independently against both runtimes.
- Fake backends to compare creation, caching, cleanup, and capability decisions.
- Separate test processes and isolated workspaces for real backends.
- Comparison only of pure metadata such as resolution results and diagnostic snapshots, never shadow execution of user commands.

### Compatibility and rollback boundaries

- Freeze the legacy path during migration. Apart from necessary bug and security fixes, do not evolve it structurally; evaluate whether every required fix must also be applied to the new runtime.
- Environments created by the new path cannot be adopted by the legacy path. Rollback requires restarting the process without `EXP_BACKEND=1`, with each process performing its own normal cleanup.
- Both runtimes continue to use the same external backend configuration semantics, task IDs, and cwd/session record format to avoid user-visible drift after switching.
- Do not declare private registry-runtime checkpoint/cache formats as stable interfaces. Migration does not promise reuse of active instances across runtimes.
- Keep legacy as the default until every migration acceptance gate passes. Do not change the default midway through the refactor.

## Implementation Phases

### Phase A: Establish the legacy characterization baseline

- Inventory all callers that directly read private terminal backend state, invoke the private factory, or duplicate environment-creation logic.
- Add characterization tests for existing backend selection, task overrides, concurrent first creation, cwd continuity, file/code sharing, idle cleanup, and global cleanup.
- Record observable legacy semantics and known defects. Tests should protect compatibility, not the internal dictionary and lock structure.

### Phase B: Introduce the façade with legacy runtime only

- Define the migration façade's minimal domain API and delegate it to the legacy adapter.
- Migrate terminal, file, process, code execution, cleanup, task override, and status callers to the façade one by one.
- During this phase, return only the legacy adapter regardless of `EXP_BACKEND`, ensuring that introducing the boundary alone does not change production behavior.
- Add architecture guard tests that prevent new callers from importing legacy globals or the private factory.

### Phase C: Freeze feature-flag and dual-path composition semantics

- Read `EXP_BACKEND` once at the composition root, select the legacy or registry adapter, and record the runtime mode.
- Build a dual-mode contract-test harness with explicit cases for unset, invalid, `0`, and `1` values.
- Verify that changing the environment variable after process initialization does not switch the runtime and that no per-call dynamic routing exists.

### Phase D: Implement the registry runtime skeleton

- Define the backend definition, factory request, capability, availability, and error contracts.
- Establish the terminal backend registry, Environment Manager, and `PluginContext` registration API.
- Use a fake backend to lock down: registration does not instantiate; explicit failures do not fall back; concurrent calls create once; factories must return `BaseEnvironment`; and cleanup runs exactly once.
- Ensure plugin discovery completes before backend resolution across CLI, gateway, cron, TUI/API, and direct tool-call paths.

### Phase E: Migrate built-in backends one at a time

- Migrate local first to establish the smallest end-to-end experimental path, then migrate other built-ins in dependency-complexity order.
- For each backend, run the same characterization and contract scenarios with both `EXP_BACKEND=0` and `EXP_BACKEND=1`.
- Preserve names, defaults, connection behavior, persistence, cwd, task overrides, cleanup, and user-visible errors.
- Keep the legacy hard-coded factory during this phase, serving only `EXP_BACKEND!=1`.

### Phase F: Switch capability and peripheral consumers

- Expose unified capability queries through the façade. The legacy adapter synthesizes equivalent capabilities from existing names/configuration, while the new adapter reads registry definitions.
- Migrate remote, container, path, prompt, and approval decisions to façade capabilities so peripheral code does not know the runtime mode.
- Keep security capabilities under host policy; neither the experimental path nor plugin declarations may loosen approval.
- Make setup, dashboard, status, dump, and diagnostics present the experimental registry dynamically while clearly identifying the active runtime mode.

### Phase G: Validate plugin configuration end to end

- Support namespaced backend options and ensure every startup surface receives the same effective configuration.
- Validate with a minimal external plugin that registers only a backend, no tools or hooks. With `EXP_BACKEND=1`, terminal, file, and code execution must use the same plugin environment instance.
- With the flag disabled, verify that selecting a third-party backend fails clearly with experimental-mode guidance and does not fall back to local.
- Cover disabled plugins, name conflicts, missing dependencies, factory exceptions, invalid return types, and cleanup exceptions.

### Phase H: Expand experimental coverage and define cutover gates

- Run backend-related CI tests against both legacy and new runtimes, with at least one full test job using the experimental runtime.
- Gradually enable `EXP_BACKEND=1` for developers, controlled gateway/cron deployments, and real built-in backend integration environments.
- Require all cutover gates: every built-in backend migrated; direct private imports eliminated; dual-mode contracts pass; approval, cleanup, and concurrency behavior match; third-party plugin tests pass; and rollback is documented.
- Keep legacy as the default until all gates pass.

### Phase I: Cut over fully and remove the migration layer

- For one release cycle, explicitly set `EXP_BACKEND=1` across target deployments while the code default remains legacy, so removing the environment variable provides rollback. Observe real runtime behavior and resource-cleanup metrics.
- After the final gates pass, remove the legacy path in one explicit release and make the new runtime the only implementation. Do not introduce an additional intermediate state that merely flips the flag's default in the same release.
- Remove the `EXP_BACKEND` router, legacy adapter, legacy hard-coded factory, and compatibility state used only by the old path.
- Make callers depend directly on the permanent Registry/Environment Manager API without reintroducing backend-specific branching.
- Remove the dual-mode test matrix while retaining the new runtime contracts, all built-in backend regressions, and third-party plugin tests.
- Update the Specialized plugin types documentation and remove experimental-mode guidance.

## Validation Matrix

### Registry and discovery

- Enabled entry-point plugins can register backends; disabled plugins do not execute registration code.
- Every startup surface completes idempotent discovery before first backend resolution.
- Duplicate and invalid names, invalid factories, and non-`BaseEnvironment` return values produce deterministic diagnostics.

### Selection and configuration

- No explicit configuration selects local.
- Explicit built-in and plugin backend selections resolve exactly.
- Explicit unknown or unavailable backend selections fail without fallback.
- `terminal.backends.<name>` is passed only to the corresponding factory, and secrets never enter logs.

### Shared runtime

- Terminal, file, process, and code execution share one instance for the same task.
- Different tasks preserve existing isolation and reuse rules.
- Concurrent first calls create only one instance.
- Idle cleanup, session teardown, and process exit never duplicate cleanup. Calls after cleanup rebuild correctly and restore cwd.

### Dual-runtime migration

- Unset, `0`, and invalid `EXP_BACKEND` values use only legacy; `1` uses only the registry runtime.
- Runtime mode freezes on first initialization, so one process never creates both legacy and new active environments.
- The same façade contract scenarios run against both runtimes and align on observable results. Command and file-write scenarios use separate isolated processes, not dual execution.
- With the flag disabled, every existing backend and configuration behaves unchanged. Third-party backends fail with experimental-mode guidance and never fall back.
- Status, dump, and logs identify the current runtime mode, effective backend, and backend source.

### Capabilities and security

- Remote, path, and prompt behavior follows capabilities without adding third-party names to hard-coded sets.
- A custom backend declaring itself isolated does not automatically bypass dangerous-command approval.
- Explicit configuration failures never silently degrade to local.
- Existing approval and host-mount behavior for built-in backends remains unchanged.

### Regression commands

During implementation, run at least:

```bash
EXP_BACKEND=0 uv run pytest -q tests/tools tests/hermes_cli
EXP_BACKEND=1 uv run pytest -q tests/tools tests/hermes_cli
EXP_BACKEND=0 uv run pytest -q
EXP_BACKEND=1 uv run pytest -q
```

If running the full suite twice is too expensive, every PR must run backend contract/tool tests in both modes, and at least one CI job must run the full suite with `EXP_BACKEND=1`. The full legacy suite remains the default required check until final cutover. Backend integration tests requiring external credentials or runtimes remain gated by integration markers or explicit environment variables; default tests must not depend on Docker, SSH, Modal, Daytona, or third-party services.

## Risks and Mitigations

- **Plugin discovery occurs too late**: the Environment Manager explicitly triggers idempotent discovery before first resolution instead of assuming a CLI import already performed it.
- **Peripheral code still branches on backend names**: maintain a capability-consumer inventory and validate end to end with a test backend whose name is unknown to core code.
- **Plugins falsify isolation metadata**: security decisions do not directly trust plugin declarations; defaults remain conservative, and relaxation requires host or user-policy authorization.
- **Multiple tools continue duplicating creation logic**: consolidate the Environment Manager before exposing a stable plugin contract, so third parties never depend on private terminal state.
- **Backend configuration schemas expand core indefinitely**: namespace plugin-specific fields and provide schema/setup metadata through backend definitions.
- **Built-in backend overrides introduce supply-chain risk**: reject duplicate names by default and treat override as a separate high-trust capability, not ordinary registry behavior.
- **Plugin load failures trigger dangerous fallback**: explicit backend selection fails closed and never falls back to local.
- **Feature-flag branches spread throughout the codebase**: only the composition root reads `EXP_BACKEND`; architecture tests prevent business callers from checking it directly.
- **One process mixes both runtimes**: freeze runtime selection on first initialization; runtime switching is unsupported and rollback requires restart.
- **The legacy adapter becomes permanent baggage**: set direct-private-import elimination and final deletion gates for each phase. New features land only in the target runtime, while legacy is frozen except for bug and security fixes.
- **Dual-path testing costs grow and behavior drifts**: share one runtime contract suite, execute real side-effect scenarios in separate processes, and establish parity with each migrated built-in rather than reconciling everything at the end.
- **Commands are dual-executed for comparison**: explicitly prohibit shadow execution and compare identical inputs only for pure parsing and metadata.

## Non-goals

- Do not change the terminal, file, process, or code execution tool schemas.
- Do not implement Coder or any other specific backend in this work.
- Do not require backend plugins to expose additional tools, hooks, or slash commands.
- Do not rewrite BaseEnvironment's existing shell snapshot or execution protocol.
- Do not support in-process hot unloading or lossless switching of active environments.
- Do not support hot runtime switching after changing `EXP_BACKEND` in a running process.
- Do not retain `EXP_BACKEND`, the migration façade, or the legacy adapter long term; they exist only for the incremental refactor.
- Do not isolate third-party backends in separate processes. Python plugins remain trusted in-process extensions whose security depends on explicit enablement and conservative host policy.
