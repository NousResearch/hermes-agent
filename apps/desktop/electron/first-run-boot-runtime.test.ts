import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { createFirstRunBootRuntime } from './first-run-boot-runtime'

function fixture() {
  const send = vi.fn()
  const webContents = { isDestroyed: () => false, send }
  let window: any = { isDestroyed: () => false, webContents }
  let reauthMessage: string | null = null
  const log = vi.fn()

  const runtime = createFirstRunBootRuntime({
    activeRoot: '/tmp/hermes-agent',
    fakeMode: false,
    fakeStepMs: 0,
    getMainWindow: () => window,
    getRemoteReauthFailure: () => reauthMessage,
    log,
    platform: 'linux'
  })

  return {
    log,
    runtime,
    send,
    setReauthMessage: (value: string | null) => {
      reauthMessage = value
    },
    setWindow: (value: any) => {
      window = value
    }
  }
}

test('boot progress stays monotonic unless explicitly reset, and the live reauth latch keeps its overlay', () => {
  const { runtime, send, setReauthMessage, setWindow } = fixture()

  assert.equal(runtime.getBootProgressState().phase, 'idle')
  runtime.updateBootProgress({ message: 'Resolving backend', phase: 'backend.resolve', progress: 28, running: true })
  assert.equal(runtime.getBootProgressState().progress, 28)
  assert.equal(send.mock.lastCall?.[0], 'hermes:boot-progress')
  assert.equal(send.mock.lastCall?.[1], runtime.getBootProgressState())

  runtime.updateBootProgress({ progress: 7 })
  assert.equal(runtime.getBootProgressState().progress, 28)
  runtime.updateBootProgress({ progress: 7 }, { allowDecrease: true })
  assert.equal(runtime.getBootProgressState().progress, 7)

  const latch = 'Your remote gateway session has expired. Sign in again.'
  setReauthMessage(latch)
  const beforeHeldUpdate = runtime.getBootProgressState()
  const beforeHeldSends = send.mock.calls.length
  runtime.updateBootProgress({ error: null, phase: 'backend.spawn', progress: 84 })
  assert.equal(runtime.getBootProgressState(), beforeHeldUpdate)
  assert.equal(send.mock.calls.length, beforeHeldSends)

  runtime.updateBootProgress({ error: latch, retryable: false })
  assert.equal(runtime.getBootProgressState().error, latch)
  assert.equal(runtime.getBootProgressState().retryable, false)

  setReauthMessage(null)
  setWindow({ isDestroyed: () => true, webContents: { isDestroyed: () => false, send } })
  const beforeMissingWindow = send.mock.calls.length
  runtime.updateBootProgress({ error: null }, { allowDecrease: true })
  assert.equal(send.mock.calls.length, beforeMissingWindow)
})

test('only a bootstrap-needed backend prompts; local confirmation resumes its one waiter', async () => {
  const { runtime, send } = fixture()
  const external = { kind: 'external', platform: 'linux' }

  assert.equal(await runtime.waitForFirstRunSetupChoice(external), 'continue-local')
  assert.equal(send.mock.calls.length, 0)
  assert.equal(runtime.getBootProgressState().phase, 'idle')

  const backend = { kind: 'bootstrap-needed', platform: 'linux', activeRoot: '/tmp/hermes-agent' }
  const pending = runtime.waitForFirstRunSetupChoice(backend)

  assert.equal(runtime.getFirstRunSetupGate().hasWaiter(), true)
  assert.deepEqual(runtime.getBootstrapState().setupChoice, {
    platform: 'linux',
    activeRoot: '/tmp/hermes-agent'
  })
  assert.equal(runtime.getBootProgressState().phase, 'bootstrap.choice')
  assert.ok(send.mock.calls.some(([channel]) => channel === 'hermes:bootstrap:event'))

  runtime.continueFirstRunLocalBootstrap()
  assert.equal(await pending, 'continue-local')
  assert.equal(runtime.getFirstRunSetupGate().hasWaiter(), false)
  assert.equal(await runtime.waitForFirstRunSetupChoice(backend), 'continue-local')
})

test('remote apply dismisses a waiting choice without entering local bootstrap; snapshot log is bounded', async () => {
  const { runtime, send } = fixture()
  const backend = { kind: 'bootstrap-needed', platform: 'linux', activeRoot: '/tmp/hermes-agent' }
  const pending = runtime.waitForFirstRunSetupChoice(backend)

  assert.equal(runtime.abandonFirstRunSetupChoiceForRemoteApply(), true)
  assert.equal(await pending, 'remote-applied')
  assert.equal(runtime.getBootstrapState().setupChoice, null)
  assert.equal(runtime.getBootstrapState().active, false)
  assert.ok(
    send.mock.calls.some(([channel, event]) => channel === 'hermes:bootstrap:event' && event.type === 'dismissed')
  )
  assert.equal(runtime.abandonFirstRunSetupChoiceForRemoteApply(), false)

  runtime.broadcastBootstrapEvent({ type: 'manifest', stages: [{ name: 'install' }] })
  runtime.broadcastBootstrapEvent({ type: 'stage', name: 'install', state: 'running' })

  for (let index = 0; index < 505; index += 1) {
    runtime.broadcastBootstrapEvent({ type: 'log', stage: 'install', line: `line ${index}` })
  }

  assert.equal(runtime.getBootstrapState().log.length, 500)
  assert.equal(runtime.getBootstrapState().log[0].line, 'line 5')
  assert.equal((runtime.getBootstrapState().stages as Record<string, { state: string }>).install.state, 'running')
  runtime.resetBootstrapSnapshot()
  assert.equal(runtime.getBootstrapState().manifest, null)
  assert.equal(runtime.getBootstrapState().log.length, 0)
})
