import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import {
  claimDecision,
  createBackendOutputTail,
  DEFAULT_OUTPUT_TAIL_LIMIT,
  isPidOnlyStartMarker,
  pidOnlyStartMarker,
  probeStartMarker,
  processLiveness,
  processStartMarker
} from './backend-claim'

// --- claimDecision: the #93608 policy ---------------------------------------

test('probe success claims with the full start marker (unchanged behavior)', () => {
  const decision = claimDecision(true, { ok: true, startMarker: 'linux:12345' })

  assert.deepEqual(decision, { action: 'claim', startMarker: 'linux:12345' })
})

test('probe success claims even when the child already exited (ownership records the incarnation)', () => {
  // The claim itself must not invent a failure: the exit handler owns cleanup.
  const decision = claimDecision(false, { ok: true, startMarker: 'win:99' })

  assert.deepEqual(decision, { action: 'claim', startMarker: 'win:99' })
})

test('probe failure on a LIVE child degrades to PID-only identity — never kills a healthy backend (#93608)', () => {
  const decision = claimDecision(true, { ok: false, reason: 'powershell.exe timed out after 30000ms' })

  assert.equal(decision.action, 'degrade')
  assert.match((decision as { reason: string }).reason, /timed out/)
})

test('probe failure on a DEAD child fails closed so the caller can attach the stderr tail', () => {
  const decision = claimDecision(false, { ok: false, reason: 'Get-Process: no process with ID 4242' })

  assert.equal(decision.action, 'fail')
  assert.match((decision as { reason: string }).reason, /4242/)
})

// --- probeStartMarker: throw → value ----------------------------------------

test('probeStartMarker converts a probe throw into { ok: false, reason }', async () => {
  const probe = await probeStartMarker(4242, async () => {
    throw new Error('PowerShell 5.1 cold start exceeded budget')
  })

  assert.deepEqual(probe, { ok: false, reason: 'PowerShell 5.1 cold start exceeded budget' })
})

test('probeStartMarker passes a successful marker through', async () => {
  const probe = await probeStartMarker(4242, async pid => `linux:${pid}`)

  assert.deepEqual(probe, { ok: true, startMarker: 'linux:4242' })
})

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

test('processLiveness rejects dead PIDs without launching a marker probe', () => {
  const dead = processLiveness(4242, () => {
    const error = new Error('missing') as NodeJS.ErrnoException

    error.code = 'ESRCH'
    throw error
  })

  assert.equal(dead, false)
})

test('processLiveness treats permission refusal as alive and unknown errors as inconclusive', () => {
  const failing = (code: string) => {
    const error = new Error(code) as NodeJS.ErrnoException

    error.code = code
    throw error
  }

  assert.equal(processLiveness(1, () => failing('EPERM')), true)
  assert.equal(processLiveness(1, () => failing('EIO')), undefined)
})

// --- PID-only marker helpers --------------------------------------------------

test('pidOnlyStartMarker round-trips through isPidOnlyStartMarker', () => {
  const marker = pidOnlyStartMarker(4242)

  assert.equal(marker, 'pid-only:4242')
  assert.equal(isPidOnlyStartMarker(marker), true)
  assert.equal(isPidOnlyStartMarker('linux:12345'), false)
  assert.equal(isPidOnlyStartMarker(undefined), false)
})

// --- output tail ring buffer ----------------------------------------------------

test('output tail keeps only the most recent bytes once past the limit', () => {
  const tail = createBackendOutputTail(16)

  tail.append('0123456789')
  tail.append('abcdefghij')

  assert.equal(tail.text(), '456789abcdefghij')
  assert.equal(tail.text().length, 16)
})

test('output tail default limit is ~8KB', () => {
  const tail = createBackendOutputTail()

  tail.append('x'.repeat(DEFAULT_OUTPUT_TAIL_LIMIT + 500))

  assert.equal(tail.text().length, DEFAULT_OUTPUT_TAIL_LIMIT)
  assert.equal(DEFAULT_OUTPUT_TAIL_LIMIT, 8192)
})

test('output tail interleaves stdout and stderr attached from spawn time', () => {
  const child = { stderr: new EventEmitter(), stdout: new EventEmitter() }
  const tail = createBackendOutputTail(64)

  tail.attach(child)
  child.stdout.emit('data', Buffer.from('booting\n'))
  child.stderr.emit('data', Buffer.from("ModuleNotFoundError: No module named 'hermes_cli'\n"))

  assert.match(tail.text(), /booting/)
  assert.match(tail.text(), /ModuleNotFoundError/)
})

test('describe() is empty when nothing was captured, formatted when output exists', () => {
  const tail = createBackendOutputTail(64)

  assert.equal(tail.describe(), '')

  tail.append('Traceback (most recent call last):\n')
  assert.match(tail.describe(), /^\nRecent backend output:\nTraceback/)
})

test('attach tolerates a child with missing stdio streams', () => {
  const tail = createBackendOutputTail(64)

  tail.attach({ stderr: null, stdout: null })
  assert.equal(tail.text(), '')
})
