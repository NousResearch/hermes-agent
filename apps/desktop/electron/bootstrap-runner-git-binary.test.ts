import assert from 'node:assert/strict'
import type * as ChildProcess from 'node:child_process'

import { test, vi } from 'vitest'

const execFileSyncMock = vi.fn((..._args: unknown[]) => 'abcdef1234567890\n')

vi.mock('node:child_process', async importOriginal => ({
  ...(await importOriginal<typeof ChildProcess>()),
  execFileSync: execFileSyncMock
}))

test('resolveCheckoutHead runs the probed git binary, not the first PATH hit (#114718)', async () => {
  // The bootstrap marker pin used a bare `git`, which on the reported Mac is
  // an Intel-only /usr/local/bin/git that fails at spawn, silently dropping
  // the real pin. main.ts now passes the binary it already probed.
  const { resolveCheckoutHead } = await import('./bootstrap-runner')

  assert.equal(resolveCheckoutHead('/repo', { gitBinary: '/usr/bin/git' }), 'abcdef1234567890')
  assert.equal(execFileSyncMock.mock.calls[0]?.[0], '/usr/bin/git')
})
