import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isPidOnlyStartMarker, processStartMarker } from './backend-claim'

// --- real probe: drives the actual OS helper (PowerShell on the Windows lane) ---

test('processStartMarker resolves a real marker for the current process', async () => {
  const marker = await processStartMarker(process.pid)

  assert.match(marker, /^(linux|win|winms|ps):.+/)
})

test('a missing PID is classified as ESRCH so reapOrphans can drop the record', async () => {
  // Largest PIDs are bounded well below this on every supported platform.
  // Windows Get-Process / macOS `ps -p` used to surface exit code 1, which
  // the identity matchers treated as "unknown" and kept forever. The native
  // gate throws ESRCH — the errno those catch blocks already map to gone.
  await assert.rejects(processStartMarker(2 ** 30 + 12345), (error: NodeJS.ErrnoException) => error?.code === 'ESRCH')
})

// --- PID-only marker helpers --------------------------------------------------

test('isPidOnlyStartMarker recognizes degraded PID-only ownership records', () => {
  assert.equal(isPidOnlyStartMarker('pid-only:4242'), true)
  assert.equal(isPidOnlyStartMarker('linux:12345'), false)
  assert.equal(isPidOnlyStartMarker(undefined), false)
})
