import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { RemoteUpdateTarget } from './managed-ssh-update'
import {
  createManagedSshUpdateService,
  type ManagedSshUpdateServiceDependencies,
  type ManagedSshUpdateSource,
  type ManagedSshUpdateScope
} from './managed-ssh-update-service'

const CORRELATION = '12345678-1234-4678-9234-567812345678'
const OTHER_CORRELATION = '22345678-1234-4678-9234-567812345678'

interface TestScope extends ManagedSshUpdateScope {
  state?: object | null
}

interface TestSource extends ManagedSshUpdateSource {
  label: string
}

function source(id = 'homelab', kind: 'ssh' | 'url' = 'ssh'): TestSource {
  return { id, kind, label: id }
}

function target(): RemoteUpdateTarget {
  return {
    ssh: { exec: async () => '' },
    platform: 'Linux',
    hermesPath: '~/.local/bin/hermes',
    hermesHome: '~/.hermes'
  }
}

function deps(
  overrides: Partial<ManagedSshUpdateServiceDependencies<TestSource, TestScope>> = {}
): ManagedSshUpdateServiceDependencies<TestSource, TestScope> {
  const defaultSource = source()

  return {
    resolveSource: id => (id === defaultSource.id ? defaultSource : null),
    readRecoveryRecords: () => [],
    captureScopes: async () => [],
    openTransport: async () => ({ target: target(), close: async () => {} }),
    targetFromState: () => target(),
    executeRemoteUpdate: async (_target, correlation, context) => {
      await context.onLaunchProved()
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    },
    preflightRemote: async () => {},
    awaitRestoreClearance: async () => {},
    drainScope: async () => {},
    closeTransports: async () => {},
    restoreScope: async () => {},
    prepareRecovery: async () => {},
    completeRecovery: async () => {},
    restoreRecoveryScope: async () => {},
    ...overrides
  }
}

test('service admission deduplicates duplicate claims and refuses foreign sources', async () => {
  let release!: () => void
  let executions = 0
  const pending = new Promise<void>(resolve => {
    release = resolve
  })
  const service = createManagedSshUpdateService(
    deps({
      executeRemoteUpdate: async (_target, correlation, context) => {
        executions += 1
        await pending
        await context.onLaunchProved()
        return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
      }
    })
  )

  const first = service.request('homelab')
  const second = service.request('homelab')

  assert.strictEqual(first, second)
  assert.equal(service.activeUpdates.size, 1)
  assert.equal(executions, 0)

  const foreign = await service.request('missing')
  assert.equal(foreign.outcome, 'refused')
  assert.equal(executions, 0)

  release()
  const result = await first
  assert.equal(result.ok, true)
  assert.equal(executions, 1)
  assert.equal(service.activeUpdates.size, 0)
  assert.equal(service.gate.owner('homelab'), null)
})

test('durable ownership fences new admission and only the exact owner can release it', () => {
  const service = createManagedSshUpdateService(
    deps({
      readRecoveryRecords: () => [
        {
          connectionId: 'homelab',
          correlationId: CORRELATION,
          phase: 'prepared',
          scopes: [],
          source: source()
        }
      ]
    })
  )

  assert.equal(service.gate.claim('homelab', OTHER_CORRELATION), false)
  assert.equal(service.gate.owner('homelab'), CORRELATION)
  service.gate.release('homelab', OTHER_CORRELATION)
  assert.equal(service.gate.owner('homelab'), CORRELATION)
  assert.throws(() => service.gate.assertCanDial('homelab'), /paused/)
  assert.throws(() => service.gate.assertCanMutate('homelab'), /edited or removed/)
})

test('service captures, restores, closes owned transport, and releases admission in order', async () => {
  const events: string[] = []
  const scopes: TestScope[] = [
    { key: 'primary', profile: 'default', primary: true, state: {} },
    { key: 'conn:homelab::research', profile: 'research', registryScoped: true, state: {} },
    { key: 'research', profile: 'research', state: {} }
  ]
  const service = createManagedSshUpdateService(
    deps({
      captureScopes: async () => {
        events.push('capture')
        return scopes
      },
      preflightRemote: async () => events.push('preflight'),
      prepareRecovery: async () => events.push('prepare-recovery'),
      drainScope: async scope => events.push(`drain:${scope.profile}`),
      executeRemoteUpdate: async (_target, correlation, context) => {
        events.push('launch')
        await context.onLaunchProved()
        return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
      },
      awaitRestoreClearance: async () => events.push('clearance'),
      closeTransports: async () => events.push('close-transports'),
      restoreScope: async scope => events.push(`restore:${scope.profile}`),
      completeRecovery: async () => events.push('complete-recovery')
    })
  )

  const result = await service.request('homelab')

  assert.equal(result.ok, true)
  assert.deepEqual(events, [
    'capture',
    'preflight',
    'prepare-recovery',
    'drain:default',
    'drain:research',
    'drain:research',
    'launch',
    'clearance',
    'close-transports',
    'restore:default',
    'restore:research',
    'restore:research',
    'complete-recovery'
  ])
})

test('primary restoration is serialized and exact-owner scoped', async () => {
  const service = createManagedSshUpdateService(deps())
  let release!: () => void
  const pending = new Promise<void>(resolve => {
    release = resolve
  })

  const first = service.restorePrimary(source(), 'default', CORRELATION, async () => {
    await pending
    return 'restored'
  })

  await Promise.resolve()
  await assert.rejects(
    service.restorePrimary(source('other'), 'default', OTHER_CORRELATION, async () => 'wrong'),
    /Another managed SSH primary restore is already in progress/
  )

  release()
  assert.equal(await first, 'restored')
  assert.equal(service.primaryRestoreOwnerForProfile('default'), null)
})
