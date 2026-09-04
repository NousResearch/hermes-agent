import { describe, expect, it } from 'vitest'

import { createRunGit, type RunGitOptions } from './run-git'

/**
 * Regression tests for the bounded git runner (issue #95788).
 *
 * The bug: runGit resolved only on the child's `exit` event with no timeout,
 * so a blackholed git call (TCP up, HTTP never answered) hung the update
 * check on "Looking for updates…" forever. These tests prove the hard
 * timeout settles the promise with a `timedOut` result well before a wedged
 * child would finish, and that fast commands still resolve normally.
 */

/** Path to a tiny executable "git" used by every test. */
const GIT_BIN = __dirname + '/fixtures/fake-git-bin/git'

/** Timeout the run and measure how long the promise actually took. */
async function timed<T>(timeoutMs: number | undefined, args: string[], options: Omit<RunGitOptions, 'timeoutMs'> = {}) {
  const runGit = createRunGit(() => GIT_BIN)
  const started = Date.now()
  const result = await runGit(args, { ...options, timeoutMs })

  return { result, elapsed: Date.now() - started }
}

describe('runGit timeout bound (regression #95788)', () => {
  it('resolves with timedOut when the child would block forever', async () => {
    const { result, elapsed } = await timed(400, ['block'])

    // The promise must settle on the timeout, not hang forever.
    expect(result.timedOut).toBe(true)
    expect(result.code).toBeNull()
    // Fast completion of the bound, not the child (which would never finish).
    expect(elapsed).toBeLessThan(5000)
    // A clear message flows to the fetch-failed path via stderr.
    expect(result.stderr).toContain('timed out after 400ms')
  })

  it('still resolves normally with its exit code for a fast command', async () => {
    const { result } = await timed(2000, ['ok'])

    expect(result.timedOut).toBe(false)
    expect(result.code).toBe(0)
    expect(result.stdout).toContain('ok-stdout')
    expect(result.stderr).toBe('')
  })

  it('propagates a non-zero exit code for a fast failing command', async () => {
    const { result } = await timed(2000, ['fail'])

    expect(result.timedOut).toBe(false)
    expect(result.code).toBe(7)
    expect(result.stderr).toContain('fail-stderr')
  })
})
