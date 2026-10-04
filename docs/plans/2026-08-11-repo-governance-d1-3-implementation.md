# D1.3 Repository Identity Implementation Note

## Status and scope

This is the bounded D1.3 source implementation note for the Option-B-R1 recovery slice. The implementation remains request-only and non-activating: it does not create repository-incarnation markers, enroll repositories, alter configuration or databases, or perform Git/GitHub/deployment mutations.

The normative executable inputs are the frozen D1.3-I3 candidate at:

`/Users/ykliu/.hermes/profiles/dev/artifacts/repo-governance/2026-08-11-repo-governance-d1-3-i3-candidate`

The source preserves the frozen request grammar, public-result validation and framing, R6 evaluation vectors and precedence, coherent-reseal adversary seam, exact Darwin command/environment contract, and native ABI lifecycle registry.

## Public boundary

`resolve_repository_identity(request)` accepts exactly one closed request object. Request and expected-binding validation run before the Darwin observer is imported or invoked. Caller-supplied observations, descriptors, PIDs, callbacks, state, environment, or authority are not public inputs.

The private `_build_test_resolver` seam exists only to prove reject-before-observer behavior and to execute frozen complete-state contracts without widening the production API.

## Darwin descriptor-relative launch

Darwin Git discovery uses only the five frozen `/usr/bin/git rev-parse` argv tuples and the four frozen environment entries. `_spawn_git_at_fd`:

1. requires an approved argv tuple;
2. creates CLOEXEC, nonblocking stdout/stderr pipes;
3. configures `posix_spawn_file_actions_addfchdir` with the already-open effective-workdir descriptor;
4. duplicates only the child pipe write ends onto stdout/stderr and closes the enumerated pipe descriptors in the child actions;
5. invokes exact `/usr/bin/git` through `posix_spawn` without `git -C`, pathname-CWD fallback, shell execution, `preexec_fn`, `pass_fds`, `/dev/fd`, or a fork-before-exec CWD shim;
6. closes parent write ends, concurrently drains both read ends under the absolute monotonic deadline and bounded output caps;
7. waits for the exact returned child PID and returns the exact drained stdout/stderr bytes, exit code, exact-reap assertion, and empty remaining-FD set;
8. fail-closed cleanup closes every still-owned descriptor, destroys initialized file actions, and terminates/reaps only the exact spawned child when an exceptional branch still owns it.

This bounded recovery corrects the historical pending Darwin RED by retaining the drained bytearrays after EOF descriptor removal rather than reconstructing output from the emptied descriptor map. The A1 fork fixture is marked only with the local live-system bypass required for signaling its own exact child. Cleanup now begins at the operation deadline with exact-PID `SIGTERM`, allows an absolute 250,000,000 ns grace interval, conditionally sends exact-PID `SIGKILL` only while the child remains alive, and retries EINTR while performing only bounded `waitpid(pid, WNOHANG)` exact-child reaping.

## Native ABI and lifecycle seam

`_run_native_abi_probe_for_test` executes the frozen ctypes native probe with the current pinned interpreter and `PYTHONDONTWRITEBYTECODE=1`. It requires the exact 18-row registry and classifies a row as passing only when created descriptors equal closed descriptors, created children equal reaped children, child ownership is false at branch end, and residue is zero. The frozen profile covers Darwin `poll`, `read`, `close`, `kill`, and exact-PID `waitpid` ABI/lifecycle behavior, including actual EINTR, terminal exit/signal, WNOHANG-running, and ECHILD-after-reap cases.

The probe is test evidence only; it is not a runtime enrollment, marker, or activation path.

## Fail-closed production posture

On non-Darwin platforms, observation reports `PLATFORM_UNSUPPORTED`. On Darwin, this bounded source slice does not create or repair a marker and therefore remains fail-closed at `MARKER_MISSING`. The frozen complete-state evaluator remains the deterministic content-based contract surface for derivation, precedence, drift, and binding tests. It accepts coherent fresh-copy reseals by recomputing the expected primitive/hash transformation from content; it has no object-identity registry, hidden expected result, global cache, or test-only evaluator bypass.

## Verification ownership boundary

Recovery-writer checks may be narrowly focused and use only:

`/Users/ykliu/Projects/hermes-agent-repo-governance-d1/.d1-venv/bin/python`

with resolved executable `/Users/ykliu/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11`, pytest `9.0.2`, and `PYTHONDONTWRITEBYTECODE=1`.

The final exact Darwin target, both-test-files command, root fresh bounded suite, and unrelated-CWD bounded suite are intentionally not run by the recovery writer; they remain parent-owned final-byte verification gates. No freeze, independent review, hard-stop, commit, push, merge, deployment, or activation claim is made here.
