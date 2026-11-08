import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isRetryableRemoteBootFailure, isSshClientFailedBootFailure } from './backend-start-failure'
import {
  isSshConfigProbeTimeout,
  isSshConfigProbeTimeoutError,
  SSH_CONFIG_PROBE_TIMEOUT_LATCH,
  SshConfigProbe
} from './ssh-config-probe'

const TIMEOUT_MS = 10_000
// A genuine hang is killed right at the timeout; a sleep-killed probe fires
// on wake, long after it started (wall clock includes the sleep).
const AWAKE_TIMEOUT_MS = TIMEOUT_MS
const SLEPT_MS = 17 * 60_000

// Shapes Node's execFile produces for each failure mode.
function execFileTimeout(): Error {
  return Object.assign(new Error('Command failed: ssh -G -- box'), { killed: true, signal: 'SIGTERM', code: null })
}

function execTextTimeout(): Error {
  return Object.assign(new Error('ssh timed out after 10000ms'), { timedOut: true })
}

function missingBinary(): Error {
  return Object.assign(new Error('spawn ssh ENOENT'), { code: 'ENOENT', killed: false, signal: null })
}

function badConfig(): Error {
  return Object.assign(new Error('Command failed: ssh -G -- box\n/Users/x/.ssh/config line 3: Bad configuration option'), {
    code: 255,
    killed: false,
    signal: null
  })
}

type Step = { output: string; ms?: number } | { error: Error; ms: number }

// A fake clock plus an exec that advances it by each step's duration.
function harness() {
  let clock = 0
  const probe = new SshConfigProbe({ now: () => clock })

  function exec(steps: Step[]) {
    let calls = 0

    const run = async () => {
      const step = steps[calls++]
      clock += step.ms ?? 5

      if ('error' in step) {
        throw step.error
      }

      return step.output
    }

    return { run, calls: () => calls }
  }

  return { probe, exec }
}

const ok: Step = { output: 'config' }
const hang: Step = { error: execFileTimeout(), ms: AWAKE_TIMEOUT_MS }
const slept: Step = { error: execFileTimeout(), ms: SLEPT_MS }

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise
  } catch (error) {
    return error
  }

  throw new Error('expected rejection')
}

test('classifies timeouts and kills as timeouts, real client failures as not', () => {
  assert.equal(isSshConfigProbeTimeout(execFileTimeout()), true)
  assert.equal(isSshConfigProbeTimeout(execTextTimeout()), true)
  assert.equal(isSshConfigProbeTimeout(missingBinary()), false)
  assert.equal(isSshConfigProbeTimeout(badConfig()), false)
  assert.equal(isSshConfigProbeTimeout(new Error('boom')), false)
  assert.equal(isSshConfigProbeTimeout(null), false)
})

test('wake from sleep: one timeout is retried in place and succeeds', async () => {
  const { probe, exec } = harness()
  const e = exec([slept, ok])

  assert.equal(await probe.run('ssh', TIMEOUT_MS, e.run), 'config')
  assert.equal(e.calls(), 2)
  assert.equal(probe.timeouts, 0)
})

test('two timeouts in one round are retryable, not latched', async () => {
  const { probe, exec } = harness()
  const error = await rejection(probe.run('ssh', TIMEOUT_MS, exec([hang, { error: execTextTimeout(), ms: TIMEOUT_MS }]).run))

  assert.equal(isSshConfigProbeTimeoutError(error), true)
  assert.equal(isSshClientFailedBootFailure(error), false)
  assert.equal(
    isRetryableRemoteBootFailure({
      attemptedRemote: true,
      isReauth: false,
      isSshClientFailed: isSshClientFailedBootFailure(error)
    }),
    true
  )
})

test('a missing ssh binary stays terminal and is not retried', async () => {
  const { probe, exec } = harness()
  const e = exec([{ error: missingBinary(), ms: 5 }, ok])
  const error = await rejection(probe.run('ssh', TIMEOUT_MS, e.run))

  assert.equal(isSshClientFailedBootFailure(error), true)
  assert.equal(e.calls(), 1)
})

test('a non-zero exit (bad ssh_config) stays terminal', async () => {
  const { probe, exec } = harness()
  const error = await rejection(probe.run('ssh', TIMEOUT_MS, exec([{ error: badConfig(), ms: 5 }]).run))

  assert.equal(isSshClientFailedBootFailure(error), true)
})

test(`${SSH_CONFIG_PROBE_TIMEOUT_LATCH} consecutive awake timeouts latch as a hanging client`, async () => {
  const { probe, exec } = harness()
  const first = await rejection(probe.run('ssh', TIMEOUT_MS, exec([hang, hang]).run))

  assert.equal(isSshConfigProbeTimeoutError(first), true)

  const e = exec([hang, ok])
  const second = await rejection(probe.run('ssh', TIMEOUT_MS, e.run))

  assert.equal(isSshClientFailedBootFailure(second), true)
  assert.equal(e.calls(), 1)
})

test('a night of dark wakes: sleep-killed timeouts never count toward the latch', async () => {
  const { probe, exec } = harness()

  // Ten boot retries in a row, every probe killed by sleep, no success between.
  for (let wake = 0; wake < 10; wake++) {
    const error = await rejection(probe.run('ssh', TIMEOUT_MS, exec([slept, slept]).run))

    assert.equal(isSshConfigProbeTimeoutError(error), true)
    assert.equal(isSshClientFailedBootFailure(error), false)
    assert.equal(probe.timeouts, 0)
  }

  assert.equal(await probe.run('ssh', TIMEOUT_MS, exec([ok]).run), 'config')
})

test('sleep-killed timeouts neither count nor clear a run of genuine ones', async () => {
  const { probe, exec } = harness()

  await rejection(probe.run('ssh', TIMEOUT_MS, exec([hang, slept]).run))
  assert.equal(probe.timeouts, 1)
})

test('a success resets the timeout run', async () => {
  const { probe, exec } = harness()

  for (let day = 0; day < 5; day++) {
    await rejection(probe.run('ssh', TIMEOUT_MS, exec([hang, hang]).run))
    assert.equal(probe.timeouts, 2)
    assert.equal(await probe.run('ssh', TIMEOUT_MS, exec([ok]).run), 'config')
    assert.equal(probe.timeouts, 0)
  }
})

test('reset() (system resume) clears a run of genuine timeouts', async () => {
  const { probe, exec } = harness()

  await rejection(probe.run('ssh', TIMEOUT_MS, exec([hang, hang]).run))
  assert.equal(probe.timeouts, 2)
  probe.reset()
  assert.equal(probe.timeouts, 0)

  const error = await rejection(probe.run('ssh', TIMEOUT_MS, exec([hang, hang]).run))

  assert.equal(isSshConfigProbeTimeoutError(error), true)
})
