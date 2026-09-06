import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import { buildExactTerminateScript, parseTerminateScriptOutput } from './windows-process-terminate'

/**
 * 2026-09-05: the force-release refused the gateway's venv trampoline with
 * `TREE_ASSIGN_FAILED win32=5` — AssignProcessToJobObject is refused for a
 * process that already lives in a job whose hierarchy does not admit ours —
 * and reported the whole termination as failed while the holder kept the venv
 * locked. Such a holder is still exact and authenticated: it stays suspended
 * and is terminated by handle. Source-contract tests; the real-PowerShell
 * boundary suites cover the contained path.
 */
describe('degraded containment for holders already inside a foreign job', () => {
  const script = buildExactTerminateScript(4242, 1_700_000_000, 500, { installRoot: 'C:\\hermes' })

  it('falls back to by-handle termination on ERROR_ACCESS_DENIED for a job member', () => {
    assert.match(script, /ProcessIsInAnyJob/)
    assert.match(script, /TerminateProcessAndWait/)
    assert.match(script, /\(-\$assignCode\) -eq 5 -and \[HermesForceReleaseNative\]::ProcessIsInAnyJob\(\$processHandle\) -eq 1/)
    // Other assignment failures still fail the boundary.
    assert.match(script, /throw \('TREE_ASSIGN_FAILED pid=' \+ \$currentPid/)
  })

  it('only terminates through the job when something was actually contained', () => {
    assert.match(script, /if \(\$contained\.Count -gt 0\) \{\s*\$terminateCode = \[HermesForceReleaseNative\]::TerminateJobAndWait/)
    assert.match(script, /foreach \(\$degradedPid in \$degraded\)/)
    assert.match(script, /CONTAINMENT_DEGRADED pids=/)
  })

  it('keeps degraded members out of the job set so a failed boundary resumes them', () => {
    assert.match(script, /if \(Assign-ContainedProcess \$rootHandle \$pidTarget\) \{ \[void\]\$contained\.Add\(\$pidTarget\) \}/)
    assert.match(script, /if \(Assign-ContainedProcess \$childHandle \$childPid\) \{ \[void\]\$contained\.Add\(\$childPid\) \}/)
  })

  it('still parses as terminated when containment was degraded', () => {
    const result = parseTerminateScriptOutput('TERMINATED\nCONTAINMENT_DEGRADED pids=58196', 0)

    assert.deepEqual(result, { kind: 'terminated' })
  })
})
