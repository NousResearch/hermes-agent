import assert from 'node:assert/strict'

import { test } from 'vitest'

import { ManagedConnectionUpdateGate, type ManagedSshUpdateIntent, type RemoteUpdateTarget } from './managed-ssh-update'
import {
  createManagedSshUpdateService,
  type ManagedSshUpdateScope,
  type ManagedSshUpdateServiceDependencies,
  type ManagedSshUpdateSource
} from './managed-ssh-update-service'

const CORRELATION = '12345678-1234-4678-9234-567812345678'
const OTHER_CORRELATION = '22345678-1234-4678-9234-567812345678'

const PINNED_INTENT: ManagedSshUpdateIntent = {
  targetSha: 'abcdef0123456789abcdef0123456789abcdef01',
  expectedInstallId: 'a'.repeat(32),
  expectedCurrentSha: '0123456789abcdef0123456789abcdef01234567',
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

const EXPECTED_SOURCE = {
  installId: 'a'.repeat(32),
  installationFingerprint: 'b'.repeat(64),
  sourceFingerprint: 'c'.repeat(64)
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
    resolveInstallationId: async () => EXPECTED_SOURCE.installId,
    readRecoveryRecords: () => [],
    captureScopes: async () => [],
    openTransport: async () => ({ target: target(), close: async () => {} }),
    targetFromState: () => target(),
    executeRemoteUpdate: async (_target, correlation, context) => {
      await context.beforeLaunchDispatch()

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

test('two connection aliases cannot launch concurrent updates against one installation', async () => {
  let release!: () => void
  let launched = 0
  let preparations = 0
  const pending = new Promise<void>(resolve => {release = resolve})
  const service = createManagedSshUpdateService(deps({
    resolveSource: id => ['homelab', 'alias'].includes(id) ? source(id) : null,
    verifyCoordinatorSource: async () => {},
    prepareRemote: async (_source, correlationId) => {
      preparations += 1
      return { kind: 'preparation', correlationId }
    },
    executeRemoteUpdate: async (_target, correlation, context) => {
      launched += 1
      await context.beforeLaunchDispatch()
      await pending
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
  const first = service.requestCoordinator('homelab', {
    correlationId: CORRELATION, intent: PINNED_INTENT,
    expectedSource: EXPECTED_SOURCE, launchCapability: capability
  })
  assert.equal(first.admitted, true)

  const aliasCapability = service.issueLaunchCapability('alias', OTHER_CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
  const alias = service.requestCoordinator('alias', {
    correlationId: OTHER_CORRELATION, intent: PINNED_INTENT,
    expectedSource: EXPECTED_SOURCE, launchCapability: aliasCapability
  })
  assert.equal(alias.admitted, false)
  assert.match(alias.reason, /installation.*in progress/i)
  assert.equal((await alias.operation).outcome, 'refused')
  assert.equal((await service.request('alias', { correlationId: OTHER_CORRELATION })).outcome, 'refused')
  assert.equal((await service.prepare('alias', { correlationId: OTHER_CORRELATION })).outcome, 'refused')
  assert.equal(preparations, 0)

  release()
  assert.equal((await first.operation).ok, true)
  assert.equal(launched, 1)
})

test('a legacy update owns its installation before a coordinator alias can enter', async () => {
  let release!: () => void
  let started!: () => void
  let launched = 0
  const pending = new Promise<void>(resolve => {release = resolve})
  const entered = new Promise<void>(resolve => {started = resolve})
  const service = createManagedSshUpdateService(deps({
    resolveSource: id => ['homelab', 'alias'].includes(id) ? source(id) : null,
    verifyCoordinatorSource: async () => {},
    executeRemoteUpdate: async (_target, correlation, context) => {
      launched += 1
      await context.beforeLaunchDispatch()
      started()
      await pending
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  const first = service.request('homelab', { correlationId: CORRELATION })
  await entered
  try {
    const capability = service.issueLaunchCapability('alias', OTHER_CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
    const alias = service.requestCoordinator('alias', {
      correlationId: OTHER_CORRELATION, intent: PINNED_INTENT,
      expectedSource: EXPECTED_SOURCE, launchCapability: capability
    })
    assert.equal(alias.admitted, false)
    assert.equal((await alias.operation).outcome, 'refused')
    assert.equal(launched, 1)
  } finally {
    release()
    await first
  }
})

test('durable recovery under one alias fences another alias of the same installation', async () => {
  let launched = 0
  const service = createManagedSshUpdateService(deps({
    resolveSource: id => ['homelab', 'alias'].includes(id) ? source(id) : null,
    readRecoveryRecords: () => [{
      connectionId: 'homelab', correlationId: CORRELATION, installationId: EXPECTED_SOURCE.installId,
      phase: 'launching', scopes: [], source: source('homelab')
    }],
    verifyCoordinatorSource: async () => {},
    executeRemoteUpdate: async (_target, correlation) => {
      launched += 1
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  const capability = service.issueLaunchCapability('alias', OTHER_CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
  const coordinator = service.requestCoordinator('alias', {
    correlationId: OTHER_CORRELATION, intent: PINNED_INTENT,
    expectedSource: EXPECTED_SOURCE, launchCapability: capability
  })
  assert.equal(coordinator.admitted, false)
  assert.equal((await coordinator.operation).outcome, 'refused')
  assert.equal((await service.request('alias', { correlationId: OTHER_CORRELATION })).outcome, 'refused')
  assert.equal(launched, 0)
})

test('a recovery record without proven installation identity fences new aliases', async () => {
  let launched = 0
  const service = createManagedSshUpdateService(deps({
    resolveSource: id => ['homelab', 'alias'].includes(id) ? source(id) : null,
    readRecoveryRecords: () => [{
      connectionId: 'homelab', correlationId: CORRELATION,
      phase: 'launching', scopes: [], source: source('homelab')
    }],
    executeRemoteUpdate: async (_target, correlation) => {
      launched += 1
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  assert.equal((await service.request('alias', { correlationId: OTHER_CORRELATION })).outcome, 'refused')
  assert.equal(launched, 0)
})

test('missing or malformed remote installation identity refuses update and preparation before mutation', async () => {
  let updates = 0
  let preparations = 0
  const service = createManagedSshUpdateService(deps({
    resolveInstallationId: async () => null,
    prepareRemote: async (_source, correlationId) => {
      preparations += 1
      return { kind: 'preparation', correlationId }
    },
    executeRemoteUpdate: async (_target, correlation) => {
      updates += 1
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  assert.equal((await service.request('homelab', { correlationId: CORRELATION })).outcome, 'refused')
  assert.equal((await service.prepare('homelab', { correlationId: OTHER_CORRELATION })).outcome, 'refused')
  assert.equal(updates, 0)
  assert.equal(preparations, 0)
  assert.equal(service.gate.owner('homelab'), null)
})

test('coordinator refuses a changed remote installation before transport mutation', async () => {
  let mutations = 0
  const service = createManagedSshUpdateService(deps({
    resolveInstallationId: async () => 'd'.repeat(32),
    verifyCoordinatorSource: async () => {},
    executeRemoteUpdate: async (_target, correlation) => {
      mutations += 1
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
  const admission = service.requestCoordinator('homelab', {
    correlationId: CORRELATION, intent: PINNED_INTENT,
    expectedSource: EXPECTED_SOURCE, launchCapability: capability
  })

  assert.equal(admission.admitted, true)
  const result = await admission.operation
  assert.equal(result.outcome, 'refused')
  assert.match(result.error || '', /reviewed installation/)
  assert.equal(mutations, 0)
  assert.equal(service.gate.owner('homelab'), null)
})

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
        await context.beforeLaunchDispatch()

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

test('fleet service uses the exact legacy gate and operation maps for admission', async () => {
  const gate = new ManagedConnectionUpdateGate()
  const activeUpdates = new Map<string, Promise<any>>()
  const activeRecoveries = new Map<string, Promise<void>>()
  const primaryRestoreOwners = new Map<string, { correlationId: string; profile: string; source: TestSource }>()

  const service = createManagedSshUpdateService(deps({
    gate, activeUpdates, activeRecoveries, primaryRestoreOwners
  }))

  assert.strictEqual(service.gate, gate)
  assert.strictEqual(service.activeUpdates, activeUpdates)
  assert.strictEqual(service.activeRecoveries, activeRecoveries)
  assert.equal(gate.claim('homelab', CORRELATION), true)
  const blocked = await service.request('homelab', { correlationId: OTHER_CORRELATION })
  assert.equal(blocked.outcome, 'refused')
  assert.equal(gate.owner('homelab'), CORRELATION)
  gate.release('homelab', CORRELATION)
})

test('coordinator never reuses an active scope SSH socket for a reviewed source', async () => {
  const staleTarget = target()
  const selectedTarget = target()
  let opened = 0
  let inspected: RemoteUpdateTarget | null = null
  let mutated: RemoteUpdateTarget | null = null
  const service = createManagedSshUpdateService(deps({
    captureScopes: async () => [{ key: 'primary', profile: 'default', state: { ssh: staleTarget.ssh } }],
    targetFromState: () => staleTarget,
    openTransport: async () => {
      opened += 1

      return { target: selectedTarget, close: async () => {} }
    },
    verifyCoordinatorSource: async (_source, selected) => {inspected = selected},
    executeRemoteUpdate: async (selected, correlation, context) => {
      mutated = selected
      await context.beforeLaunchDispatch()

      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))
  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
  const admission = service.requestCoordinator('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    expectedSource: EXPECTED_SOURCE,
    launchCapability: capability
  })

  assert.equal(admission.admitted, true)
  assert.equal((await admission.operation).ok, true)
  assert.equal(opened, 1)
  assert.strictEqual(inspected, selectedTarget)
  assert.strictEqual(mutated, selectedTarget)
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
      verifyCoordinatorSource: async () => {},
      executeRemoteUpdate: async (_target, correlation, context) => {
        mutations += 1
        forwarded = context.intent
        await context.beforeLaunchDispatch()

        return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
      }
    })
  )

  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)

  const first = await service.request('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    mode: 'coordinator',
    launchCapability: capability,
    expectedSource: EXPECTED_SOURCE
  })

  const reused = await service.request('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    mode: 'coordinator',
    launchCapability: capability,
    expectedSource: EXPECTED_SOURCE
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
      verifyCoordinatorSource: async () => {},
      executeRemoteUpdate: async (_target, correlation, context) => {
        mutations += 1
        await context.beforeLaunchDispatch()

        return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
      }
    })
  )

  const result = await service.request('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    mode: 'coordinator',
    expectedSource: EXPECTED_SOURCE
  })

  assert.equal(result.ok, false)
  assert.match(result.error || '', /single-use launch capability/)
  assert.equal(mutations, 0)
})

test('a route edit after rollout start cannot dispatch to the newly selected installation', async () => {
  let mutations = 0
  let verificationCalls = 0
  const selectedRoute = source('homelab')
  const selectedTarget = target()

  const service = createManagedSshUpdateService(Object.assign(deps({
    resolveSource: () => selectedRoute,
    openTransport: async () => ({ target: selectedTarget, close: async () => {} }),
    executeRemoteUpdate: async (_target, correlation) => {
      mutations += 1

      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }), {
    verifyCoordinatorSource: async (resolved: TestSource, actualTarget: RemoteUpdateTarget, expected: typeof EXPECTED_SOURCE) => {
      verificationCalls += 1
      assert.deepEqual(expected, { ...EXPECTED_SOURCE, expectedCurrentSha: PINNED_INTENT.expectedCurrentSha })
      assert.strictEqual(actualTarget, selectedTarget)

      if (resolved.label !== 'reviewed-route') {throw new Error('coordinator-source-binding-mismatch')}
    }
  }))

  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
  selectedRoute.label = 'edited-route'

  const admission = service.requestCoordinator('homelab', Object.assign({
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    launchCapability: capability
  }, { expectedSource: EXPECTED_SOURCE }))

  const result = await admission.operation

  assert.equal(admission.admitted, true)
  assert.equal(result.ok, false)
  assert.match(result.error || '', /coordinator-source-binding-mismatch/)
  assert.equal(verificationCalls, 1)
  assert.equal(mutations, 0)
})

test('coordinator admission fails closed without a trusted source verifier', async () => {
  let mutations = 0

  const service = createManagedSshUpdateService(deps({
    executeRemoteUpdate: async (_target, correlation) => {
      mutations += 1

      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)

  const admission = service.requestCoordinator('homelab', Object.assign({
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    launchCapability: capability
  }, { expectedSource: EXPECTED_SOURCE }))

  const result = await admission.operation

  assert.equal(admission.admitted, false)
  assert.equal(result.outcome, 'refused')
  assert.match(result.error || '', /source verifier/i)
  assert.equal(mutations, 0)
})

test('coordinator capability cannot authorize a different source binding', async () => {
  let mutations = 0

  const service = createManagedSshUpdateService(deps({
    verifyCoordinatorSource: async () => {},
    executeRemoteUpdate: async (_target, correlation) => {
      mutations += 1

      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)

  const admission = service.requestCoordinator('homelab', {
    correlationId: CORRELATION,
    intent: PINNED_INTENT,
    launchCapability: capability,
    expectedSource: { ...EXPECTED_SOURCE, sourceFingerprint: 'f'.repeat(64) }
  })

  const result = await admission.operation

  assert.equal(admission.admitted, true)
  assert.equal(result.ok, false)
  assert.match(result.error || '', /different update transaction/)
  assert.equal(mutations, 0)
})

test('coordinator capability cannot authorize a different reviewed current commit', async () => {
  let mutations = 0
  const service = createManagedSshUpdateService(deps({
    verifyCoordinatorSource: async () => {},
    executeRemoteUpdate: async (_target, correlation) => {
      mutations += 1
      return { exitCode: 0, receipt: { correlationId: correlation, outcome: 'success' } }
    }
  }))

  const capability = service.issueLaunchCapability('homelab', CORRELATION, PINNED_INTENT, EXPECTED_SOURCE)
  const changedIntent = { ...PINNED_INTENT, expectedCurrentSha: 'f'.repeat(40) }
  const admission = service.requestCoordinator('homelab', {
    correlationId: CORRELATION,
    intent: changedIntent,
    launchCapability: capability,
    expectedSource: EXPECTED_SOURCE
  })

  assert.equal(admission.admitted, true)
  assert.equal((await admission.operation).outcome, 'update-failed')
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
        await context.beforeLaunchDispatch()

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
  let recoveryRecords = [{ connectionId: 'homelab', correlationId: CORRELATION, phase: 'prepared', scopes: [], source: source() }]

  const service = createManagedSshUpdateService(
    deps({
      readRecoveryRecords: () => recoveryRecords,
      awaitRestoreClearance: async () => { await recoveryPending },
      completeRecovery: async () => {
        recoveryRecords = []
      }
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
  let persistedInstallationId: string | null = null

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
      prepareRecovery: async (_source, _correlation, _scopes, installationId) => {
        persistedInstallationId = installationId
        events.push('prepare-recovery')
      },
      drainScope: async scope => events.push(`drain:${scope.profile}`),
      executeRemoteUpdate: async (_target, correlation, context) => {
        events.push('launch')
        await context.beforeLaunchDispatch()

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
  assert.equal(persistedInstallationId, EXPECTED_SOURCE.installId)
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
