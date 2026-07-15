# Task plan: issue #39 test-runtime isolation

## Contract

Canonical `scripts/run_tests.sh` must run tests from the current linked worktree,
reject or safely normalize inherited import-path poisoning, keep duration/cache/
bytecode artifacts out of the primary checkout, and return the underlying suite
status. Scope is limited to the canonical runner, its focused regression tests,
and the runtime contract documentation.

## Ordered slices

1. **Trace and RED** — document the runner contract, add subprocess tests that
   reproduce import provenance, source-tree bytecode, primary-root duration
   writes, success exit status, failure exit status, and unsafe inherited import
   configuration. Verify each test fails for the current runner.
2. **GREEN: environment and import boundary** — make the runner derive the
   current worktree root, launch with a sanitized environment and explicit
   worktree import root, and make child pytest processes inherit that exact
   import contract.
3. **GREEN: artifact boundary** — route `compileall` bytecode to an isolated
   temporary cache and route duration state to a worktree-local/temporary
   runtime directory without arbitrary inherited-path writes.
4. **Review and integration** — run focused GREEN tests, expanded offline
   runner tests, diff/static checks, independent security/code review, then
   commit only after all evidence is recorded.

## Acceptance criteria

- `hermes_cli` resolves from the current worktree even when inherited
  `PYTHONPATH` names the primary checkout or an editable-install location.
- Test bytecode, pytest cache, and duration state do not write under the primary
  checkout; the chosen cache location is explicit, local, and safe.
- A passing suite exits 0; a failing suite remains non-zero.
- Unsafe inherited import configuration is removed or causes a clear fail-closed
  error; it cannot select an arbitrary path.
- Production worker/dispatcher behavior and dependencies remain unchanged.

## Evidence gates

- RED focused suite with the pre-fix failure output.
- GREEN focused suite and expanded offline suite using `scripts/run_tests.sh`.
- `git diff --check`.
- Static syntax/compile check with bytecode redirected to temporary storage.
- Security/code review covering environment poisoning, symlinks/path traversal,
  inherited `PYTHONPATH`, arbitrary writes, and exit-code propagation.
- Main-checkout status unchanged except for its pre-existing untracked files.

## Security/code review verdict

Independent review after implementation found no Critical or Required findings.

- **Environment poisoning:** inherited environment is rebuilt with `env -i`;
  `PYTHONPATH` is allowlisted with the selected worktree first and
  `PYTHONNOUSERSITE=1` disables user-site shadowing.
- **Symlink/path traversal:** repository identity comes from Git, not a caller
  path variable; the candidate is accepted only when the expected runner exists
  under that repository. Runtime output paths are not caller-configurable.
- **Arbitrary writes:** bytecode uses a `mktemp` directory and cleanup is
  bounded to that generated path; duration/cache state is rooted at the
  selected worktree. No primary checkout or production path is selected.
- **Exit propagation:** the wrapper captures and returns the child status after
  cleanup, and the failure probe proves a failing suite stays non-zero.
- **Scope/performance:** only canonical test-runner resolution and artifact
  handling changed; no worker/dispatcher, provider, dependency, or production
  path changed. Precompile cost remains bounded to the existing tracked-file
  pass.

Review verdict: **approve for local commit** after the final GREEN evidence.
