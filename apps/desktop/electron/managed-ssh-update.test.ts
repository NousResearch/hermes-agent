import assert from 'node:assert/strict'
import { exec as execCallback } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import {
  buildPosixManagedUpdateLaunch,
  buildRemoteUpdateObservationCommand,
  buildWindowsManagedUpdateLaunch,
  fenceManagedSshBootstrapPublication,
  ManagedConnectionUpdateGate,
  managedSshRecoveryScopes,
  managedSshScopeRole,
  managedSshTokenPersistencePlan,
  parseRemoteUpdateObservation,
  RECEIPT_GRACE_MS,
  recoverManagedSshScopes,
  refusedManagedSshUpdate,
  runManagedSshUpdate,
  waitForManagedRemoteClearance,
  waitForManagedRemoteUpdate,
  waitForManagedSshBootstrapFence,
  waitForManagedUpdateOperations
} from './managed-ssh-update'
import { createBootstrapCoordinator } from './ssh-bootstrap-coordinator'

const CORRELATION = '12345678-1234-4678-9234-567812345678'
const exec = promisify(execCallback)

function observation(over: Record<string, unknown> = {}) {
  return JSON.stringify({
    marker: 'absent',
    launchIntent: 'absent',
    exitCode: null,
    receipt: null,
    coordinatorReady: null,
    ...over
  })
}

test('ManagedConnectionUpdateGate blocks new dials but admits the exact restoring transaction', () => {
  const gate = new ManagedConnectionUpdateGate()

  assert.equal(gate.claim('homelab', CORRELATION), true)
  assert.equal(gate.claim('homelab', '22345678-1234-4678-9234-567812345678'), false)
  assert.throws(() => gate.assertCanDial('homelab'), /paused/)
  assert.doesNotThrow(() => gate.assertCanDial('homelab', CORRELATION))
  gate.release('homelab', '22345678-1234-4678-9234-567812345678')
  assert.equal(gate.owner('homelab'), CORRELATION)
  gate.release('homelab', CORRELATION)
  assert.doesNotThrow(() => gate.assertCanDial('homelab'))
})

test('durable recovery claim survives memory release and fences ordinary dials, edits, and removal', () => {
  let durable: string | null = CORRELATION
  const gate = new ManagedConnectionUpdateGate(id => (id === 'homelab' ? durable : null))

  assert.equal(gate.claim('homelab', CORRELATION), true)
  gate.release('homelab', CORRELATION)
  assert.equal(gate.owner('homelab'), CORRELATION)
  assert.throws(() => gate.assertCanDial('homelab'), /paused/)
  assert.throws(() => gate.assertCanMutate('homelab'), /edited or removed/)
  assert.doesNotThrow(() => gate.assertCanDial('homelab', CORRELATION))
  assert.equal(gate.claim('homelab', '22345678-1234-4678-9234-567812345678'), false)

  durable = null
  assert.doesNotThrow(() => gate.assertCanDial('homelab'))
  assert.doesNotThrow(() => gate.assertCanMutate('homelab'))
})

test('inactive SSH crash recovery keeps ordinary dials fenced until positive clearance', async () => {
  let durableOwner: string | null = CORRELATION
  const relaunchedGate = new ManagedConnectionUpdateGate(id => (id === 'homelab' ? durableOwner : null))
  let releaseClearance!: () => void
  const clearance = new Promise<void>(resolve => {
    releaseClearance = resolve
  })
  let restoreCalls = 0
  let journalCleared = false
  const recovery = recoverManagedSshScopes({
    scopes: [] as Array<{ profile: string }>,
    awaitClearance: () => clearance,
    restoreScope: async () => {
      restoreCalls += 1
    },
    completeRecovery: async () => {
      journalCleared = true
      durableOwner = null
    }
  })

  await Promise.resolve()
  assert.throws(() => relaunchedGate.assertCanDial('homelab'), /paused/)
  assert.equal(journalCleared, false)
  releaseClearance()
  const results = await recovery

  assert.deepEqual(results, [])
  assert.equal(restoreCalls, 0)
  assert.equal(journalCleared, true)
  assert.doesNotThrow(() => relaunchedGate.assertCanDial('homelab'))
})

test('managed update joins a pre-claim bootstrap until its final gate check rolls back the serve', async () => {
  const gate = new ManagedConnectionUpdateGate()
  const coordinator = createBootstrapCoordinator()
  let releaseLifecycle!: () => void
  const lifecycle = new Promise<void>(resolve => {
    releaseLifecycle = resolve
  })
  let serveLive = false
  let rollbackComplete = false
  const startPromise = coordinator.start(
    '',
    'fingerprint',
    async () => {
      await lifecycle
      serveLive = true
      await fenceManagedSshBootstrapPublication({
        assertCanPublish: () => gate.assertCanDial('homelab'),
        publish: () => {},
        rollback: async () => {
          serveLive = false
          rollbackComplete = true
        }
      })
    },
    { managedScope: 'primary', registryConnectionId: 'homelab' }
  )
  // The fence rethrows managed-update-in-progress after rolling back — that
  // rejection propagating out of start() is the CONTRACT under test, not an
  // accident. Swallow it here so vitest doesn't flag the floating promise as
  // an unhandled rejection while the assertions below verify the rollback.
  startPromise.catch(() => {})
  const barrier = waitForManagedSshBootstrapFence(coordinator.active, 'homelab')
  let barrierComplete = false
  void barrier.then(() => {
    barrierComplete = true
  })

  assert.equal(gate.claim('homelab', CORRELATION), true)
  await Promise.resolve()
  assert.equal(barrierComplete, false)
  releaseLifecycle()
  await barrier

  assert.equal(rollbackComplete, true)
  assert.equal(serveLive, false)
  assert.equal(barrierComplete, true)
})

test('bootstrap publication stays in the same turn as its final gate assertion', async () => {
  const gate = new ManagedConnectionUpdateGate()
  let published = false
  let rolledBack = false

  queueMicrotask(() => {
    gate.claim('homelab', CORRELATION)
  })
  await fenceManagedSshBootstrapPublication({
    assertCanPublish: () => gate.assertCanDial('homelab'),
    publish: () => {
      published = true
    },
    rollback: async () => {
      rolledBack = true
    }
  })

  assert.equal(published, true)
  assert.equal(rolledBack, false)
  assert.equal(gate.owner('homelab'), CORRELATION)
})

test('registry-qualified primary token adoption stays reusable across consecutive launches', () => {
  const legacy: Record<string, string> = {}
  const registry: Record<string, string> = {}

  for (const servedToken of ['served-on-first-launch', 'served-on-second-launch']) {
    const plan = managedSshTokenPersistencePlan('profile', 'homelab')

    assert.equal(plan.legacySource, 'profile')
    assert.equal(plan.registryConnectionId, 'homelab')
    legacy[plan.legacySource!] = servedToken
    registry[plan.registryConnectionId] = servedToken
    assert.equal(legacy.profile, registry.homelab)
  }

  assert.equal(legacy.profile, 'served-on-second-launch')
  assert.equal(registry.homelab, 'served-on-second-launch')
  assert.deepEqual(managedSshTokenPersistencePlan('registry:homelab'), {
    legacySource: null,
    registryConnectionId: 'homelab'
  })
})

test('actual primary and matching bare/composite pools keep distinct managed scope roles', () => {
  const base = { connectionId: 'homelab', prefix: 'conn:homelab::' }

  assert.equal(
    managedSshScopeRole({
      ...base,
      key: '',
      state: { primaryRegistryScope: true, registryConnectionId: 'homelab' }
    }),
    'primary'
  )
  assert.equal(
    managedSshScopeRole({
      ...base,
      key: 'research',
      routeConnectionId: 'homelab',
      state: { primaryRegistryScope: false, registryConnectionId: 'homelab' }
    }),
    'pool'
  )
  assert.equal(managedSshScopeRole({ ...base, key: 'conn:homelab::research' }), 'pool')
})

test('durable recovery preserves every same-profile primary, registry, and legacy scope', () => {
  const scopes = managedSshRecoveryScopes(
    [
      { key: '', profile: 'default', primary: true },
      { key: 'conn:homelab::default', profile: 'default' },
      { key: 'default', profile: 'default' }
    ],
    'conn:homelab::'
  )

  assert.deepEqual(scopes, [
    { key: '', kind: 'primary', profile: 'default' },
    { key: 'conn:homelab::default', kind: 'registry', profile: 'default' },
    { key: 'default', kind: 'legacy', profile: 'default' }
  ])
})

test('update-all deduplicates the same recovery scope and keeps primary precedence', () => {
  assert.deepEqual(
    managedSshRecoveryScopes(
      [
        { key: 'conn:homelab::default', profile: 'default' },
        { key: 'conn:homelab::default', profile: 'default', primary: true },
        { key: 'conn:homelab::default', profile: 'default' }
      ],
      'conn:homelab::'
    ),
    [{ key: 'conn:homelab::default', kind: 'primary', profile: 'default' }]
  )
})

test('POSIX managed launcher is detached, correlation-scoped, and never publishes handoff exit 75', () => {
  const command = buildPosixManagedUpdateLaunch(
    {
      ssh: { exec: async () => '' },
      platform: 'Linux',
      hermesPath: '~/.local/bin/hermes',
      hermesHome: '~/.hermes'
    },
    CORRELATION
  )

  assert.match(command, /setsid/)
  assert.match(command, /update --yes/)
  assert.doesNotMatch(command, /update --yes --gateway/)
  assert.match(command, new RegExp(`HERMES_UPDATE_CORRELATION_ID=.*${CORRELATION}`))
  assert.match(command, /\[ "\$rc" -ne 75 \]/)
  assert.match(command, new RegExp(`\\.update_exit_code\\.${CORRELATION}`))
  assert.match(command, new RegExp(`\\.update_launch_intent\\.${CORRELATION}`))
  assert.match(command, /while \[ ! -e/)
})

test('POSIX managed launcher executes the updater command and atomically publishes its status', async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), 'hermes-managed-launch-'))

  try {
    const command = buildPosixManagedUpdateLaunch(
      {
        ssh: { exec: async () => '' },
        platform: 'Linux',
        hermesPath: '/bin/true',
        hermesHome: home
      },
      CORRELATION
    )
    const { stdout } = await exec(command, { shell: '/bin/sh' })
    const statusPath = path.join(home, `.update_exit_code.${CORRELATION}`)
    let status = ''

    for (let attempt = 0; attempt < 50 && !status; attempt += 1) {
      try {
        status = await readFile(statusPath, 'utf8')
      } catch {
        await new Promise(resolve => setTimeout(resolve, 10))
      }
    }

    assert.match(stdout, /MANAGED_UPDATE_STARTED/)
    assert.equal(status, '0')
  } finally {
    await rm(home, { force: true, recursive: true })
  }
})

test('Windows managed launcher starts a hidden child and leaves exit 75 to the external coordinator', () => {
  const command = buildWindowsManagedUpdateLaunch(
    {
      ssh: { exec: async () => '' },
      platform: 'Windows',
      hermesPath: 'C:\\Hermes\\hermes.exe',
      hermesHome: 'C:\\Users\\alice\\.hermes',
      pythonPath: 'C:\\Hermes\\python.exe'
    },
    CORRELATION
  )
  const outer = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
  const wrapperBase64 = outer.match(/"-EncodedCommand",'([^']+)'/)?.[1]

  assert.match(outer, /Start-Process/)
  assert.match(outer, /WindowStyle Hidden/)
  assert.ok(wrapperBase64, 'outer launcher carries a separately encoded detached wrapper')
  const wrapper = Buffer.from(wrapperBase64!, 'base64').toString('utf16le')

  assert.match(wrapper, /update --yes/)
  assert.doesNotMatch(wrapper, /update --yes --gateway/)
  assert.match(wrapper, /HERMES_UPDATE_WINDOWS_DETACHED/)
  assert.match(wrapper, /HERMES_UPDATE_TAURI_READY_PATH/)
  assert.match(wrapper, /HERMES_UPDATE_TAURI_OUTCOME_PATH/)
  assert.match(wrapper, /\$rc -ne 75/)
  assert.match(wrapper, /\$handoffAccepted=/)
  assert.match(wrapper, new RegExp(`update_launch_intent\\.${CORRELATION}`))
  assert.ok(wrapper.indexOf('update_launch_intent') < wrapper.indexOf('update --yes'))
  assert.match(outer, /launcher intent timed out/)
})

test('remote observation rejects a receipt for another correlation', () => {
  assert.throws(
    () =>
      parseRemoteUpdateObservation(
        observation({
          exitCode: 0,
          receipt: { correlationId: '22345678-1234-4678-9234-567812345678', outcome: 'success' }
        }),
        CORRELATION
      ),
    /did not match/
  )
})

test('POSIX observer reads the exact correlation receipt and terminal marker from disk', async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), 'hermes-managed-update-'))

  try {
    const receipts = path.join(home, 'logs', 'update_receipts')
    await mkdir(receipts, { recursive: true })
    await writeFile(path.join(home, `.update_exit_code.${CORRELATION}`), '0')
    await writeFile(
      path.join(receipts, `update_${CORRELATION}.json`),
      JSON.stringify({
        correlation_id: CORRELATION,
        outcome: 'success',
        started_at: '2026-08-23T00:00:00Z',
        finished_at: '2026-08-23T00:01:00Z',
        pre_update: { sha: 'old' },
        post_update: { sha: 'new' }
      })
    )
    const command = buildRemoteUpdateObservationCommand(
      {
        ssh: { exec: async () => '' },
        platform: 'Linux',
        hermesPath: '/opt/hermes/hermes',
        hermesHome: home
      },
      CORRELATION
    )
    const { stdout } = await exec(command, { shell: '/bin/sh' })
    const parsed = parseRemoteUpdateObservation(stdout, CORRELATION)

    assert.equal(parsed.marker, 'absent')
    assert.equal(parsed.exitCode, 0)
    assert.equal(parsed.receipt?.correlationId, CORRELATION)
    assert.equal(parsed.receipt?.preSha, 'old')
    assert.equal(parsed.receipt?.postSha, 'new')
  } finally {
    await rm(home, { force: true, recursive: true })
  }
})

test('managed observer unwraps a named profile home for the install-wide marker', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-managed-profile-marker-'))
  const profileHome = path.join(root, 'profiles', 'research')

  try {
    await mkdir(profileHome, { recursive: true })
    await writeFile(path.join(root, '.hermes-update-in-progress'), `${process.pid}\n1\n`)
    const command = buildRemoteUpdateObservationCommand(
      {
        ssh: { exec: async () => '' },
        platform: 'Linux',
        hermesPath: '/opt/hermes/hermes',
        hermesHome: profileHome
      },
      CORRELATION
    )
    const { stdout } = await exec(command, { shell: '/bin/sh' })
    const parsed = parseRemoteUpdateObservation(stdout, CORRELATION)

    assert.equal(parsed.marker, 'live')
    assert.equal(parsed.markerPid, process.pid)
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('Windows coordinator handoff is pending until its marker clears and correlated receipt is durable', async () => {
  const replies = [
    observation({ marker: 'live', markerPid: 44 }),
    observation({
      marker: 'live',
      markerPid: 88,
      coordinatorReady: { correlationId: CORRELATION, pid: 88 }
    }),
    observation({
      marker: 'absent',
      exitCode: null,
      receipt: {
        correlationId: CORRELATION,
        outcome: 'success',
        finishedAt: '2026-08-23T00:00:00Z'
      },
      coordinatorReady: { correlationId: CORRELATION, pid: 88 }
    })
  ]
  let calls = 0
  const target = {
    platform: 'Windows' as const,
    hermesPath: 'C:\\Hermes\\hermes.exe',
    hermesHome: 'C:\\Users\\alice\\.hermes',
    pythonPath: 'C:\\Hermes\\python.exe',
    ssh: {
      exec: async () => {
        const reply = replies[Math.min(calls, replies.length - 1)]
        calls += 1

        return reply
      }
    }
  }

  const proof = await waitForManagedRemoteUpdate(target, CORRELATION, {
    pollMs: 0,
    sleep: async () => {}
  })

  assert.equal(calls, 3)
  assert.equal(proof.exitCode, 0)
  assert.equal(proof.receipt.correlationId, CORRELATION)
})

test('terminal status without its durable receipt fails instead of claiming success', async () => {
  let now = 0
  const target = {
    platform: 'Linux' as const,
    hermesPath: '~/.local/bin/hermes',
    hermesHome: '~/.hermes',
    ssh: { exec: async () => observation({ marker: 'absent', exitCode: 0 }) }
  }

  await assert.rejects(
    waitForManagedRemoteUpdate(target, CORRELATION, {
      now: () => now,
      pollMs: 1,
      sleep: async () => {
        now += RECEIPT_GRACE_MS
      }
    }),
    /without a correlated durable receipt/
  )
})

test('live or malformed remote markers fail actionably at bounded update and recovery deadlines', async () => {
  for (const marker of ['live', 'malformed'] as const) {
    let now = 0
    const target = {
      platform: 'Linux' as const,
      hermesPath: '~/.local/bin/hermes',
      hermesHome: '~/.hermes',
      ssh: { exec: async () => observation({ marker, ...(marker === 'live' ? { markerPid: 44 } : {}) }) }
    }
    const clock = {
      timeoutMs: 5,
      pollMs: 1,
      now: () => now,
      sleep: async () => {
        now += 5
      }
    }

    await assert.rejects(waitForManagedRemoteUpdate(target, CORRELATION, clock), /services remain stopped/)
    now = 0
    await assert.rejects(waitForManagedRemoteClearance(target, CORRELATION, clock), /durable recovery record/)
  }
})

test('a journaled launch requires correlated terminal proof or an observed live-owner transition before restore', async () => {
  let now = 0
  const target = {
    platform: 'Linux' as const,
    hermesPath: '~/.local/bin/hermes',
    hermesHome: '~/.hermes',
    ssh: { exec: async () => observation({ marker: 'absent' }) }
  }

  await assert.rejects(
    waitForManagedRemoteClearance(target, CORRELATION, {
      requireTerminal: true,
      timeoutMs: 5,
      pollMs: 1,
      now: () => now,
      sleep: async () => {
        now += 5
      }
    }),
    /durable recovery record/
  )

  target.ssh.exec = async () =>
    observation({
      marker: 'absent',
      exitCode: 0,
      receipt: { correlationId: CORRELATION, outcome: 'success', finishedAt: '2026-08-23T00:00:00Z' }
    })
  await assert.doesNotReject(
    waitForManagedRemoteClearance(target, CORRELATION, { requireTerminal: true, timeoutMs: 0 })
  )
})

test('remote launch intent fences crash recovery even before the local journal records launch proof', async () => {
  let now = 0
  const target = {
    platform: 'Linux' as const,
    hermesPath: '~/.local/bin/hermes',
    hermesHome: '~/.hermes',
    ssh: { exec: async () => observation({ marker: 'absent', launchIntent: 'present' }) }
  }

  await assert.rejects(
    waitForManagedRemoteClearance(target, CORRELATION, {
      timeoutMs: 5,
      pollMs: 1,
      now: () => now,
      sleep: async () => {
        now += 5
      }
    }),
    /durable recovery record/
  )

  target.ssh.exec = async () => observation({ marker: 'absent', launchIntent: 'absent' })
  await assert.doesNotReject(waitForManagedRemoteClearance(target, CORRELATION, { timeoutMs: 0 }))
  target.ssh.exec = async () => observation({ marker: 'absent', launchIntent: 'dead' })
  await assert.doesNotReject(
    waitForManagedRemoteClearance(target, CORRELATION, { requireTerminal: true, timeoutMs: 0 })
  )
})

test('before-quit join observes operations added while an earlier update settles', async () => {
  let releaseFirst!: () => void
  let releaseSecond!: () => void
  const operations = new Set<Promise<void>>()
  const first = new Promise<void>(resolve => {
    releaseFirst = resolve
  })
  operations.add(first)
  const joined = waitForManagedUpdateOperations(() => operations)
  const second = new Promise<void>(resolve => {
    releaseSecond = resolve
  })

  operations.add(second)
  first.finally(() => operations.delete(first))
  second.finally(() => operations.delete(second))
  releaseFirst()
  await Promise.resolve()
  let settled = false
  joined.then(() => {
    settled = true
  })
  await Promise.resolve()
  assert.equal(settled, false)
  releaseSecond()
  await joined
  assert.equal(settled, true)
})

test('managed lifecycle restores every captured profile after update failure before releasing the gate', async () => {
  const events: string[] = []
  const scopes = [
    { key: 'conn:home::default', profile: 'default' },
    { key: 'conn:home::research', profile: 'research' }
  ]

  const result = await runManagedSshUpdate({
    connectionId: 'home',
    correlationId: CORRELATION,
    scopes,
    preflightRemote: async () => {
      events.push('preflight')
    },
    drainScope: async scope => {
      events.push(`drain:${scope.profile}`)
    },
    updateRemote: async () => {
      events.push('update')
      throw new Error('fetch failed')
    },
    awaitRestoreClearance: async () => {
      events.push('clear')
    },
    closeTransports: async () => {
      events.push('close')
    },
    restoreScope: async scope => {
      events.push(`restore:${scope.profile}`)
    },
    releaseGate: () => {
      events.push('release')
    }
  })

  assert.deepEqual(events, [
    'preflight',
    'drain:default',
    'drain:research',
    'update',
    'clear',
    'close',
    'restore:default',
    'restore:research',
    'release'
  ])
  assert.equal(result.ok, false)
  assert.equal(result.updateOk, false)
  assert.equal(result.restoreOk, true)
  assert.equal(result.outcome, 'update-failed')
  assert.deepEqual(
    result.scopes.map(scope => [scope.profile, scope.restored]),
    [
      ['default', true],
      ['research', true]
    ]
  )
})

test('inactive managed update journals before launch and clears only after remote clearance', async () => {
  const events: string[] = []
  const result = await runManagedSshUpdate({
    connectionId: 'homelab',
    correlationId: CORRELATION,
    scopes: [],
    preflightRemote: async () => {
      events.push('preflight')
    },
    prepareRecovery: async () => {
      events.push('journal')
    },
    drainScope: async () => {
      events.push('unexpected-drain')
    },
    updateRemote: async () => {
      events.push('update')

      return {
        exitCode: 0,
        receipt: { correlationId: CORRELATION, outcome: 'success' }
      }
    },
    awaitRestoreClearance: async () => {
      events.push('clearance')
    },
    closeTransports: async () => {
      events.push('close')
    },
    restoreScope: async () => {
      events.push('unexpected-restore')
    },
    completeRecovery: async () => {
      events.push('clear-journal')
    },
    releaseGate: () => {
      events.push('release')
    }
  })

  assert.deepEqual(events, ['preflight', 'journal', 'update', 'clearance', 'close', 'clear-journal', 'release'])
  assert.equal(result.ok, true)
  assert.deepEqual(result.scopes, [])
})

test('managed lifecycle attempts every drain and every restore when one ownership proof fails', async () => {
  const drained: string[] = []
  const restored: string[] = []
  let updateCalled = false

  const result = await runManagedSshUpdate({
    connectionId: 'home',
    correlationId: CORRELATION,
    scopes: [
      { key: 'a', profile: 'default' },
      { key: 'b', profile: 'research' }
    ],
    preflightRemote: async () => {},
    drainScope: async scope => {
      drained.push(scope.profile)
      if (scope.profile === 'default') {
        throw new Error('foreign owner')
      }
    },
    updateRemote: async () => {
      updateCalled = true

      return {
        exitCode: 0,
        receipt: { correlationId: CORRELATION, outcome: 'success' }
      }
    },
    awaitRestoreClearance: async () => {},
    closeTransports: async () => {},
    restoreScope: async scope => {
      restored.push(scope.profile)
    },
    releaseGate: () => {}
  })

  assert.deepEqual(drained, ['default', 'research'])
  assert.deepEqual(restored, ['default', 'research'])
  assert.equal(updateCalled, false)
  assert.match(result.error || '', /foreign owner/)
})

test('managed lifecycle journals before drain and leaves scopes stopped when clearance cannot be proved', async () => {
  const events: string[] = []
  const result = await runManagedSshUpdate({
    connectionId: 'home',
    correlationId: CORRELATION,
    scopes: [{ key: 'a', profile: 'default' }],
    preflightRemote: async () => {
      events.push('preflight')
    },
    prepareRecovery: async () => {
      events.push('journal')
    },
    drainScope: async () => {
      events.push('drain')
    },
    updateRemote: async () => {
      events.push('update')
      throw new Error('observer timed out')
    },
    awaitRestoreClearance: async () => {
      events.push('clear')
      throw new Error('marker unavailable; durable recovery retained')
    },
    closeTransports: async () => {
      events.push('close')
    },
    restoreScope: async () => {
      events.push('restore')
    },
    completeRecovery: async () => {
      events.push('clear-journal')
    },
    releaseGate: () => {
      events.push('release')
    }
  })

  assert.deepEqual(events, ['preflight', 'journal', 'drain', 'update', 'clear', 'close', 'release'])
  assert.equal(result.restoreOk, false)
  assert.equal(result.scopes[0].restored, false)
  assert.match(result.scopes[0].error || '', /durable recovery retained/)
})

test('journal cleanup failure prevents a false successful update result', async () => {
  const result = await runManagedSshUpdate({
    connectionId: 'home',
    correlationId: CORRELATION,
    scopes: [{ key: 'a', profile: 'default' }],
    preflightRemote: async () => {},
    prepareRecovery: async () => {},
    drainScope: async () => {},
    updateRemote: async () => ({
      exitCode: 0,
      receipt: { correlationId: CORRELATION, outcome: 'success' }
    }),
    awaitRestoreClearance: async () => {},
    closeTransports: async () => {},
    restoreScope: async () => {},
    completeRecovery: async () => {
      throw new Error('disk remained busy')
    },
    releaseGate: () => {}
  })

  assert.equal(result.updateOk, true)
  assert.equal(result.restoreOk, false)
  assert.equal(result.ok, false)
  assert.equal(result.outcome, 'restore-failed')
  assert.match(result.error || '', /recovery record/)
})

test('preflight refusal leaves a healthy primary scope untouched and releases without waiting', async () => {
  const events: string[] = []
  const result = await runManagedSshUpdate({
    connectionId: 'home',
    correlationId: CORRELATION,
    scopes: [{ key: '', primary: true, profile: 'default' }],
    preflightRemote: async () => {
      events.push('preflight')
      throw new Error('live foreign update')
    },
    drainScope: async () => {
      events.push('drain')
    },
    updateRemote: async () => {
      events.push('update')

      return { exitCode: 0, receipt: { correlationId: CORRELATION, outcome: 'success' } }
    },
    awaitRestoreClearance: async () => {
      events.push('clear')
    },
    closeTransports: async () => {
      events.push('close')
    },
    restoreScope: async () => {
      events.push('restore')
    },
    releaseGate: () => {
      events.push('release')
    }
  })

  assert.deepEqual(events, ['preflight', 'close', 'release'])
  assert.equal(result.updateOk, false)
  assert.equal(result.restoreOk, true)
})

test('refused result is structured and has no managed scopes to restore', () => {
  const result = refusedManagedSshUpdate('cloud', CORRELATION, 'not managed')

  assert.equal(result.outcome, 'refused')
  assert.equal(result.restoreOk, true)
  assert.deepEqual(result.scopes, [])
})
