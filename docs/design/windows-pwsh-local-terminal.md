# Native PowerShell for the Windows local terminal backend

Status: investigation and scope-alignment draft. This document records the observed implementation seams on `origin/main` at the start of the feature work. It is not an accepted maintainer design and does not authorize changing remote backend semantics.

Feature-branch status: `terminal.shell: bash | pwsh`, the unified resolver,
selection-aware prompt/tool guidance, and native Windows synchronous PowerShell
foreground execution are implemented and covered by focused tests. The
PowerShell contract in this branch persists environment variables and cwd,
establishes UTF-8 explicitly, preserves PowerShell/native exit status, reuses
the shared foreground timeout and process-tree teardown, and keeps background
and PTY execution fail-closed. Resume-time shell identity and PowerShell state
beyond environment variables/cwd remain explicitly out of scope.

Related upstream work: issue #36929, PR #15641, PR #41326, and closed PR #60803.

## Objective and first-increment boundary

The first increment would add an opt-in `terminal.shell: pwsh` setting for the native Windows local backend. The current Bash behavior would remain the default. WSL, SSH, Docker, Singularity, Modal, Daytona, other remote or container backends, additional shells such as `cmd.exe`, and selecting PowerShell on non-Windows hosts remain outside this increment.

In this branch, `terminal.shell: pwsh` enables synchronous foreground execution only. Local background and terminal-tool PTY entry points consume the same resolved shell identity but deliberately fail closed; they never fall back to Bash. Full PowerShell background/PTY argv support remains a later increment, not a capability of this one. The model-facing environment hint and terminal schema describe this foreground-only boundary. Shell selection and tool metadata are frozen when the Hermes process starts, so changing `terminal.shell` requires restarting Hermes; opening another session inside the same long-lived process does not refresh the registered schema. Resume-time identity persistence remains an open design question until the existing session/config policy is traced end to end.

No new user-facing environment variable is proposed. Hermes already projects `terminal.*` configuration into child-process environment variables through `hermes_cli.config.apply_terminal_config_to_env()`. If an internal projection is needed for process boundaries, it must remain an implementation detail sourced from `config.yaml`, not a second public configuration authority.

## Observed implementation

### Foreground execution and shell state

`tools/environments/base.py` currently defines a Bash execution contract for every environment backend. `BaseEnvironment.init_session()` creates a `.sh` snapshot, invokes a login Bash process, captures exported variables, functions, aliases, and shell options, and publishes the snapshot atomically. `BaseEnvironment.execute()` later generates a Bash wrapper that sources that snapshot, changes directory with Bash syntax, executes the requested command, captures `pwd -P` through a marker, rewrites the snapshot, and returns the command status.

The coupling is semantic rather than merely executable-specific. The implementation uses `source`, `export -p`, `declare -f`, `alias -p`, `shopt`, `set +e`, `set +u`, `mktemp`, `mv`, `rm`, `builtin cd`, `$?`, and `pwd -P`. Replacing only `bash.exe` with `pwsh.exe` would therefore pass invalid wrappers to PowerShell and would not provide persistent environment or working-directory behavior.

`tools/environments/local.py::LocalEnvironment._run_bash()` supplies the native process lifecycle for the local backend. It launches a new shell process for each foreground command, combines stderr with stdout, exposes stdin only when requested, applies Windows hidden-window flags, enforces timeout handling through the shared environment lifecycle, and decodes output as UTF-8. This process-launch layer is reusable, but its current argv construction and method contract are Bash-specific.

### Background and PTY execution

Local background and PTY commands do not use `BaseEnvironment.execute()`. They go through `tools/process_registry.py::ProcessRegistry.spawn_local()`. That path independently resolves a shell with `_find_shell()`, rewrites the Bash-specific `A && B &` construct, and builds `[shell, "-lic", "set +m; <command>"]` for both pipe and PTY modes. On Windows, `_find_shell()` deliberately delegates to `_find_bash()`.

PowerShell support cannot silently reuse the Bash background path. This branch therefore passes the already-resolved shell identity into `ProcessRegistry.spawn_local()` and rejects PowerShell background/PTY requests before any Bash rewrite, argv construction, session registration, or process spawn. A later increment may add a dialect-specific PowerShell argv builder while keeping process registration, output readers, checkpointing, stdin delivery, completion notification, and Windows process-tree teardown shared.

### Configuration and prompt identity

`hermes_cli/config_defaults.py` contains the `terminal` defaults, including Bash-specific `shell_init_files` and `auto_source_bashrc`. `hermes_cli/config.py::apply_terminal_config_to_env()` is the existing bridge from `config.yaml` into child processes. The new setting should enter through this path, with `bash` as the default and `pwsh` accepted only for the native Windows local backend.

`agent/prompt_builder.py::build_environment_hints()` currently appends a fixed Windows-local Bash hint. The hint is built as part of the stable system prompt. It must become selection-aware before a PowerShell command can be generated reliably. The shell choice should be resolved before prompt construction and reused by terminal execution, rather than independently loaded at each call.

The current source does not yet establish where a shell identity should be persisted across application restart and session resume. The first implementation should not silently claim resume stability without tracing the session creation, persistence, and resume paths and deciding how old sessions without a recorded shell are handled.

### Safety and approval

`tools/approval.py` is not purely POSIX today. It already contains Windows and PowerShell coverage for destructive deletion, encoded commands, remote-content execution through `Invoke-Expression`, forced process and service termination, disk and volume destruction, registry deletion, ACL changes, backup deletion, boot configuration changes, and Windows credential paths. `tests/tools/test_approval_windows.py` exercises both dangerous and benign examples.

The remaining task is therefore an audit and gap analysis, not a new safety subsystem. The native PowerShell-default path must verify that bare cmdlets, aliases, PowerShell quoting, command separators, script blocks, encoded or obfuscated forms, and referenced `.ps1` files cannot bypass approval merely because existing normalization assumes POSIX syntax. Unsupported or ambiguous destructive constructs must require approval or be rejected rather than fail open. Benign PowerShell commands must not all become approval prompts.

## Runtime probes on the development host

The Windows development host has PowerShell 7.6.3 available as `pwsh`. A direct non-interactive pipe probe showed that the process inherited a `gb2312` console output encoding on this host. Chinese and accented output was emitted as non-UTF-8 bytes and decoded incorrectly by a UTF-8-only parent. Setting `[Console]::OutputEncoding` to a UTF-8 encoding before command execution produced valid UTF-8 bytes. The adapter must establish encoding explicitly instead of assuming that PowerShell 7 always emits UTF-8 when stdout is redirected.

A native process failure also requires explicit status propagation. After `cmd.exe /c "exit 7"`, PowerShell exposed `$LASTEXITCODE` as 7, but the surrounding PowerShell process returned success when the wrapper subsequently exited normally. Explicitly exiting with `$LASTEXITCODE` returned 7. The PowerShell wrapper must define the intended precedence between a terminating PowerShell error, a non-terminating error, and the most recent native program exit code, then preserve that result while performing state capture and cleanup.

## Smallest viable architecture seam

The implementation should separate dialect generation from process lifecycle without turning the whole environment system into a universal multi-shell framework. A small immutable shell selection is sufficient for this increment. The Bash path preserves current command behavior. The current PowerShell adapter owns executable discovery, foreground argv, initialization/state capture, per-command wrapping, CWD extraction, status preservation, encoding setup, and the model-facing foreground-only description. Background and PTY argv are explicitly deferred.

`BaseEnvironment` and `LocalEnvironment` should continue to own backend lifecycle, timeout, stdin, output collection, cleanup, and remote-environment behavior. Remote implementations should continue using the existing Bash adapter by default. `ProcessRegistry.spawn_local()` should receive the already-resolved local shell selection rather than resolving another shell independently. The Bash-only compound-background rewrite should be selected by the Bash adapter, not applied unconditionally.

This code slice establishes configuration validation and a single shell resolver, makes the Windows prompt and terminal schema consume the resolved identity, implements PowerShell foreground state reconstruction, and makes background/PTY requests fail closed under that same identity. It does not route PowerShell background or PTY argv because those execution paths are outside the current delivery boundary.

## PowerShell state contract to prove

The PowerShell path needs a deterministic initialization and per-command wrapper. It must capture environment variables without persisting per-session variables that Hermes intentionally refreshes on every invocation. It must restore the configured or session-owned working directory using native Windows paths. It should avoid treating arbitrary functions, aliases, modules, jobs, drives, and runspace state as automatically serializable session state unless the compatibility contract explicitly includes them.

This is intentionally narrower than emulating a permanent interactive PowerShell runspace. Hermes currently creates a new Bash process per foreground command and reconstructs selected state; the PowerShell implementation should preserve that process model. Environment variables and CWD are required. Any additional state, such as functions or aliases, should be justified by existing user-visible Bash behavior and by a safe deterministic serialization design.

Temporary state files must be written atomically and with restrictive permissions where Windows permits. The snapshot format must not be executable PowerShell source assembled from untrusted values unless every value is safely serialized. A structured data format consumed by a fixed wrapper may be safer for environment state than emitting assignment statements.

## Verification plan

The initial tests should pin the unchanged default and reject invalid combinations. They should prove that omitted `terminal.shell` selects Bash, `pwsh` is accepted only on native Windows with the local backend, remote backends retain Bash semantics, and prompt guidance matches the selected shell. Existing Bash snapshot and process-registry tests must remain unchanged and passing.

Native `windows_only` tests should then exercise foreground execution with paths containing spaces, single and double quotes, dollar signs, backticks, multiline commands, Unicode paths and output, stdin, explicit PowerShell errors, native executable exit codes, timeout, cancellation, and process-tree termination. Consecutive calls must prove CWD and environment persistence without leaking session-context variables between sessions.

Current background and PTY tests prove that the same frozen shell selection is used and that PowerShell requests are rejected before Bash-specific `-lic`, `set +m`, session registration, or process spawn. A later background/PTY increment must add native argv, UTF-8, stdin, descendant termination, and no-duplicate-fallback tests when those capabilities are implemented.

Safety tests should add PowerShell-default cases only where the current Windows approval suite does not already establish the invariant. The test set should include both dangerous constructs that must not fail open and benign constructs that must remain usable without unnecessary approval.

## Open questions before a merge-ready implementation

Maintainer direction is still needed on whether this should extend one of the existing `terminal.shell` pull requests or be submitted independently while preserving applicable contributor work. The session-resume identity rule needs an explicit project-level answer. PowerShell state compatibility beyond environment variables and CWD must be bounded. The expected behavior when `pwsh` is configured but unavailable must be chosen: a clear startup/configuration error is safer than silently switching dialects underneath the prompt.

The local Windows feature should stop and be re-scoped if it requires changing remote backend shell semantics, introducing a general shell plugin system, adding `cmd.exe`, or migrating existing session storage solely to satisfy an unconfirmed resume policy.
