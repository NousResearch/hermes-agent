import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { ManagedSshUpdateIntent, RemoteUpdateTarget } from './managed-ssh-update'
import {
  createManagedSshUpdateService,
  type ManagedSshUpdateServiceDependencies,
  type ManagedSshUpdateSource,
  type ManagedSshUpdateScope
} from './managed-ssh-update-service'

const CORRELATION = '12345678-1234-4678-9234-567812345678'
const OTHER_CORRELATION = '22345678-1234-4678-9234-567812345678'
const PINNED_INTENT: ManagedSshUpdateIntent = {
  targetSha: 'abcdef0123456789abcdef0123456789abcdef01',
  source: {
    repositoryRoot: '/srv/hermes-agent',
    originUrl: 'https://github.com/NousResearch/hermes-agent.git',
    resolvedRef: 'refs/remotes/origin/main',
    targetSha: 'abcdef0123456789abcdef0123456789abcdef01',
    assuranceProfile: 'managed-ssh-review-v1',
    assuranceEvidenceSha256: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
    assuranceGeneration: 7
  }
}

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
  const foreignService = createManagedSshUpdateService(
    deps({ resolveSource: id => (id === 'cloud' ? source('cloud', 'url') : null) })
  )
  const foreignKind = await foreignService.request('cloud')
  assert.equal(foreignKind.outcome, 'refused')
  assert.match(foreignKind.error || '', /registered Desktop-managed SSH/)
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

test('coordinator forwarding requires a capability bound to one exact pinned mutation', async () => {
  let mutations = 0
  let forwarded: ManagedSshUpdateIntent | undefined
  const service = createManagedSshUpdateService(
    deps({
      executeRemoteUpdate: async (_target, correlation, context) => {
        mutations += 1
        forwarded = context.intent
        await context.onLaunchProved()
        return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
      }
    })
  )
  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT)

  const first = await service.request('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    mode: 'coordinator',
    launchCapability: capability
  })
  const reused = await service.request('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    mode: 'coordinator',
    launchCapability: capability
  })

  assert.equal(first.ok, true)
  assert.deepEqual(forwarded, PINNED_INTENT)
  assert.equal(reused.outcome, 'update-failed')
  assert.match(reused.error || '', /already been consumed/)
  assert.equal(mutations, 1)
})

test('a coordinator update cannot reach mutation transport without its matching capability', async () => {
  let mutations = 0
  const service = createManagedSshUpdateService(
    deps({
      executeRemoteUpdate: async (_target, correlation, context) => {
        mutations += 1
        await context.onLaunchProved()
        return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
      }
    })
  )

  const result = await service.request('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    mode: 'coordinator'
  })

  assert.equal(result.ok, false)
  assert.match(result.error || '', /single-use launch capability/)
  assert.equal(mutations, 0)
})

test('preparation shares admission with updates, records its receipt, and does not launch an update', async () => {
  let releasePreparation!: () => void
  let launched = 0
  const events: string[] = []
  const pendingPreparation = new Promise<void>(resolve => {
    releasePreparation = resolve
  })
  const service = createManagedSshUpdateService(
    deps({
      executeRemoteUpdate: async (_target, correlation, context) => {
        launched += 1
        await context.onLaunchProved()
        return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
      },
      prepareRemote: async (_source, correlation) => {
        events.push('prepare')
        await pendingPreparation
        return { kind: 'preparation', correlationId: correlation }
      },
      recordPreparationReceipt: async () => events.push('record'),
      refreshEligibility: async () => events.push('refresh')
    })
  )

  const preparation = service.prepare('homelab', { correlationId: CORRELATION })
  await Promise.resolve()
  const blockedUpdate = await service.request('homelab', { correlationId: OTHER_CORRELATION })

  assert.equal(blockedUpdate.outcome, 'refused')
  assert.match(blockedUpdate.error || '', /already in progress/)
  assert.equal(launched, 0)
  releasePreparation()
  assert.deepEqual(await preparation, {
    connectionId: 'homelab',
    correlationId: CORRELATION,
    ok: true,
    outcome: 'prepared',
    receipt: { kind: 'preparation', correlationId: CORRELATION }
  })
  assert.deepEqual(events, ['prepare', 'record', 'refresh'])
  assert.equal(launched, 0)
})

test('preparation refuses a frozen target because preparation must precede target review', async () => {
  const service = createManagedSshUpdateService(deps({ prepareRemote: async () => ({ kind: 'preparation', correlationId: CORRELATION }) }))

  const result = await service.prepare('homelab', { correlationId: CORRELATION, intent: PINNED_INTENT })

  assert.equal(result.outcome, 'refused')
  assert.match(result.error || '', /must not include a pinned target/)
})

test('recovery wins admission over a duplicate update and releases only its own correlation', async () => {
  let recoveryRelease!: () => void
  const recoveryPending = new Promise<void>(resolve => { recoveryRelease = resolve })
  const service = createManagedSshUpdateService(
    deps({
      readRecoveryRecords: () => [{ connectionId: 'homelab', correlationId: CORRELATION, phase: 'prepared', scopes: [], source: source() }],
      awaitRestoreClearance: async () => { await recoveryPending }
    })
  )

  const recovery = service.resumeRecoveries()
  await Promise.resolve()
  const update = await service.request('homelab')
  assert.equal(update.outcome, 'refused')
  assert.equal(service.gate.owner('homelab'), CORRELATION)
  recoveryRelease()
  await recovery
  assert.equal(service.gate.owner('homelab'), null)
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
