# Validation

## Deterministic checks

From a Hermes source checkout, run:

```sh
scripts/run_tests.sh tests/skills/test_omp_session_supervision_skill.py tests/skills/test_omp_session_supervision_integration.py -q
scripts/run_tests.sh tests/skills/test_authoring_standards.py -q
node --experimental-strip-types --test tests/skills/test_omp_session_supervision_extension.mjs
```

The Python suite exercises enrollment, exact owner matching across platforms and
profiles, unsafe paths/files, observer exclusion, cursor/epoch continuity,
bounded frames, launch intent, command quoting and optional launch settings.
Its installation test fetches the actual official-skill bundle into an isolated
directory and executes the installed CLI in a venv without pip or Hermes installed.

The integration suite connects the real Python observer to the real TypeScript
extension over a Unix socket. Only the OMP lifecycle-hook source is synthetic.
It waits for the actual socket hello before emitting a new completion. It checks
first completion, timeout/re-arm without replay, original-session revocation
and terminal journal state for Discord and Slack owners. It also runs the Node
extension suite, covering continuation, sanitized errors, approval events,
retention gaps, wrong-epoch subscriber isolation, FIFO rejection and unsafe
symlink startup. No network, tmux, model or credentials are required.

Node 22.6+ with type stripping is required for extension integration. Environments
without it skip those tests explicitly; a skipped integration is not a pass.
The skill is Linux-only; unrelated OS test jobs skip it before POSIX imports.

## Actual native delivery

Synthetic tests do not establish a live gateway wake. From a real originating
conversation, use the installed helper to prepare an empty-workspace, tools-disabled
canary, arm it through the native background terminal tool, then launch. Let the
process completion wake that conversation. Re-arm before a second harmless turn
and verify a higher sequence without replay. Check the actual installed artifact;
an older build's successful canary does not qualify a new installation.

The protocol intentionally cannot manufacture a native delivery context or
verify platform acknowledgment. Record socket receipt, native wake and visible
message separately. Reset isolation belongs to the host's native completion
boundary; run its gateway completion/session-binding regressions when changing
the integration or upgrading Hermes. Do not erase cursors to recover ambiguous
delivery, and never adopt, restart or terminate unrelated workers.
