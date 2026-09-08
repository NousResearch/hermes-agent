import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { UpdateMarkerClaim } from './update-marker'
import type { UpdateMutationPermit, UpdatePreflightOutcome } from './update-preflight'
import {
  applyWindowsUpdate,
  authenticateRecoveryUpdaterHandoff,
  readUpdateHandoffAck,
  type RecoveryUpdaterHandoffDeps,
  runRecoveryUpdaterHandoff,
  waitForAcknowledgedUpdaterClaim,
  type WindowsUpdateApplyDeps,
  windowsUpdateBlocksBackendStart,
  windowsUpdateIsBusy,
  type WindowsUpdateState
} from './windows-update-apply'

const CLAIM: UpdateMarkerClaim = { pid: 41, startedAt: 100 }
const LIVE_CLAIM = { ...CLAIM, kind: 'live' as const, ageMs: 0, overdue: false }
const PERMIT = {} as UpdateMutationPermit
const PREPARED = { kind: 'handoff' as const, transport: 'script', updateRoot: 'C:\\Hermes', branch: 'main' }

function clearPreflight(): UpdatePreflightOutcome {
  return { kind: 'clear', claim: CLAIM }
}

function deps(
  state: WindowsUpdateState,
  overrides: Partial<WindowsUpdateApplyDeps<string, { child: string }>> = {}
): WindowsUpdateApplyDeps<string, { child: string }> {
  return {
    state,
    hermesHome: 'C:\\Users\\u\\.hermes',
    prepare: async () => PREPARED,
    emitProgress: () => {},
    preflightStateDb: () => {},
    runPreflight: async () => clearPreflight(),
    stopSafeBlockers: async () => {},
    launch: () => ({ launch: { child: 'updater' }, updater: 'windows.ps1' }),
    observe: async () => ({ ok: true }),
    authenticate: async () => true,
    commit: () => {},
    waitForMarkerClearance: async () => 'clear',
    restoreBackends: async () => {},
    log: () => {},
    acquireMarker: () => ({ acquired: true, owner: LIVE_CLAIM }),
    releaseMarker: () => true,
    authorize: () => PERMIT,
    runAuthorized: (_permit, operation) => operation(),
    ...overrides
  }
}

test('the real apply entrypoint owns marker, preflight, immediate observation, authentication, and commit order', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }
  const events: string[] = []
  let releaseAuthentication!: () => void

  const authentication = new Promise<void>(resolve => {
    releaseAuthentication = resolve
  })

  const pending = applyWindowsUpdate(
    {},
    deps(state, {
      prepare: async () => {
        events.push('prepare')

        return PREPARED
      },
      acquireMarker: () => {
        events.push('marker')

        return { acquired: true, owner: LIVE_CLAIM }
      },
      preflightStateDb: () => {
        events.push('state-db')
      },
      runPreflight: async () => {
        events.push('preflight')

        return clearPreflight()
      },
      authorize: () => {
        events.push('authorize')

        return PERMIT
      },
      runAuthorized: (_permit, operation) => {
        events.push('authorized')

        return operation()
      },
      launch: () => {
        events.push('launch')

        return { launch: { child: 'updater' }, updater: 'windows.ps1' }
      },
      observe: async () => {
        events.push('observe')

        return { ok: true }
      },
      authenticate: async () => {
        events.push('authenticate')
        await authentication

        return true
      },
      commit: () => {
        events.push('commit')
      }
    })
  )

  await Promise.resolve()
  await Promise.resolve()
  assert.deepEqual(events, [
    'prepare',
    'marker',
    'state-db',
    'preflight',
    'authorize',
    'authorized',
    'launch',
    'observe',
    'authenticate'
  ])
  assert.equal(state.phase, 'updating')

  releaseAuthentication()
  assert.deepEqual(await pending, { ok: true, handedOff: true, updater: 'windows.ps1' })
  assert.equal(state.phase, 'idle')
  assert.equal(events.at(-1), 'commit')
})

test('abort restoration keeps retries excluded while allowing its backend start after marker clearance', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }
  let finishRestore!: () => void
  let restoreStarted!: () => void

  const restoring = new Promise<void>(resolve => {
    finishRestore = resolve
  })

  const started = new Promise<void>(resolve => {
    restoreStarted = resolve
  })

  let restores = 0

  const pending = applyWindowsUpdate(
    {},
    deps(state, {
      authenticate: async () => false,
      releaseMarker: () => true,
      restoreBackends: async () => {
        restores += 1
        assert.equal(state.phase, 'restoring')
        assert.equal(windowsUpdateBlocksBackendStart(state), false)
        assert.equal(windowsUpdateIsBusy(state), true)
        restoreStarted()
        await restoring
      }
    })
  )

  await started
  await assert.rejects(applyWindowsUpdate({}, deps(state)), /already in progress/)
  finishRestore()
  assert.deepEqual(await pending, {
    ok: false,
    error: 'update-handoff-unacknowledged',
    message: 'Update aborted: the updater did not acknowledge the handoff. Retry after any running update finishes.'
  })
  assert.equal(restores, 1)
  assert.equal(state.phase, 'idle')
})

// Both sides of the handoff record the spawn-to-ack gap: windows.ps1 logs its
// own seconds since HERMES_UPDATE_STARTED_AT, and this is the Desktop's half.
// The ack budget is a fixed 10 s, so a failing handoff is only diagnosable if
// at least one side says how long it actually waited.
test('the spawn-to-ack latency is logged whether or not the handoff lands', async () => {
  const lines: string[] = []

  const capture = (line: string) => {
    lines.push(line)
  }

  await applyWindowsUpdate({}, deps({ phase: 'idle' }, { log: capture }))
  assert.match(lines.at(-1) ?? '', /updater handoff acknowledged \d+ ms after spawn/)

  await applyWindowsUpdate({}, deps({ phase: 'idle' }, { log: capture, authenticate: async () => false }))
  assert.ok(
    lines.some(line => /updater handoff unacknowledged \d+ ms after spawn/.test(line)),
    'the failing path is the one the number is needed for'
  )
})

test('an aborted handoff reports the failure before it starts restoring', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }
  const events: string[] = []

  const result = await applyWindowsUpdate(
    {},
    deps(state, {
      authenticate: async () => false,
      emitProgress: progress => {
        events.push(`progress:${progress.stage}`)
      },
      // The script adopted the marker before the ack wait ran out, so the
      // Desktop cannot release it and falls back to polling for clearance.
      releaseMarker: () => false,
      waitForMarkerClearance: async () => {
        events.push('clearance')

        return 'clear'
      },
      restoreBackends: async () => {
        events.push('restore')
      }
    })
  )

  // The error came last, so the user sat on the 100% "this window will close
  // and Hermes will restart" card for the whole clearance poll -- up to 20
  // minutes -- with no backend behind it and no word about the failure.
  assert.deepEqual(events, ['progress:restart', 'progress:error', 'clearance', 'restore'])
  assert.equal(result.ok, false)
})

test('restoration cannot enter its backend-start phase until marker clearance is proven', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }
  let clearMarker!: () => void
  let clearanceStarted!: () => void

  const clearance = new Promise<'clear'>(resolve => {
    clearMarker = () => resolve('clear')
  })

  const started = new Promise<void>(resolve => {
    clearanceStarted = resolve
  })

  let restored = false

  const pending = applyWindowsUpdate(
    {},
    deps(state, {
      authenticate: async () => false,
      releaseMarker: () => false,
      waitForMarkerClearance: async () => {
        clearanceStarted()

        return clearance
      },
      restoreBackends: async () => {
        restored = true
      }
    })
  )

  await started
  assert.equal(state.phase, 'updating')
  assert.equal(windowsUpdateBlocksBackendStart(state), true)
  assert.equal(restored, false)
  clearMarker()
  await pending
  assert.equal(restored, true)
  assert.equal(state.phase, 'idle')
})

test('a restoration error still releases the UI busy state', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }

  await assert.rejects(
    applyWindowsUpdate(
      {},
      deps(state, {
        authenticate: async () => false,
        restoreBackends: async () => {
          throw new Error('backend restart failed')
        }
      })
    ),
    /backend restart failed/
  )

  assert.equal(state.phase, 'idle')
})

test('a blocked preflight can stop safe blockers once and must obtain a fresh clear permit', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }

  const empty = {
    blocked: true as const,
    processes: [],
    mcpBridges: [],
    desktopPluginServices: [],
    pausableGateways: 0
  }

  const outcomes: UpdatePreflightOutcome[] = [
    { kind: 'blocked', reason: 'holders', message: 'blocked', result: empty },
    clearPreflight()
  ]

  let stopped = 0
  let preflights = 0

  const result = await applyWindowsUpdate(
    { stopSafeBlockers: true },
    deps(state, {
      runPreflight: async () => {
        preflights += 1

        return outcomes.shift()!
      },
      stopSafeBlockers: async () => {
        stopped += 1
      }
    })
  )

  assert.equal(result.ok, true)
  assert.equal(preflights, 2)
  assert.equal(stopped, 1)
})

test('marker conflict never runs preflight and resets the busy phase', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }
  let preflight = false

  const result = await applyWindowsUpdate(
    {},
    deps(state, {
      acquireMarker: () => ({ acquired: false, owner: { pid: 99, startedAt: 200 }, message: 'owned elsewhere' }),
      runPreflight: async () => {
        preflight = true

        return clearPreflight()
      }
    })
  )

  assert.deepEqual(result, { ok: false, error: 'update-already-running', message: 'owned elsewhere' })
  assert.equal(preflight, false)
  assert.equal(state.phase, 'idle')
})


// ---------------------------------------------------------------------------
// Handoff authentication (#B4)
//
// The old check only required that SOME process the Desktop had not itself
// excluded wrote a plausible `<pid>\n<createdAt>` body into the marker inside
// a ten-second window. Any same-user process satisfied it, and the Desktop
// then quit with no update running at all.
// ---------------------------------------------------------------------------

const NONCE = 'a1b2c3d4e5f60718293a4b5c6d7e8f90'
const HOME = 'C:\\Users\\u\\.hermes'

function ackFixture(overrides: Record<string, unknown> = {}) {
  return { nonce: NONCE, pid: 500, createdAt: 1_723_330_000, ...overrides }
}

function liveMarker(pid: number, startedAt: number) {
  return { kind: 'live' as const, pid, startedAt, ageMs: 0, overdue: false }
}

function claimDeps(overrides: Record<string, unknown> = {}) {
  let clock = 0

  return {
    hermesHome: HOME,
    nonce: NONCE,
    excludedPids: [41],
    startedAfter: 1_723_329_999,
    timeoutMs: 500,
    pollMs: 50,
    now: () => clock,
    sleep: async (ms: number) => { clock += Math.max(ms, 50) },
    readMarker: () => liveMarker(500, 1_723_330_000),
    readAck: () => ackFixture(),
    queryCreatedAt: async () => 1_723_330_000,
    ...overrides
  }
}

test('an acknowledged claim from the spawned updater authenticates', async () => {
  assert.equal(await waitForAcknowledgedUpdaterClaim(claimDeps()), true)
})

test('a foreign same-user claimant with a fresh createdAt is rejected', async () => {
  // Exactly the accepted case before: not excluded, marker body internally
  // consistent, created inside the window. It never saw the nonce.
  assert.equal(
    await waitForAcknowledgedUpdaterClaim(claimDeps({ readAck: () => null })),
    false,
    'no ack at all must not authenticate'
  )

  assert.equal(
    await waitForAcknowledgedUpdaterClaim(claimDeps({ readAck: () => ackFixture({ nonce: 'f'.repeat(32) }) })),
    false,
    'an ack carrying somebody else\u2019s nonce must not authenticate'
  )
})

test('an ack that names a different pid than the marker is rejected', async () => {
  assert.equal(
    await waitForAcknowledgedUpdaterClaim(claimDeps({ readAck: () => ackFixture({ pid: 501 }) })),
    false
  )
})

test('an acknowledged claim whose pid was recycled is rejected', async () => {
  assert.equal(
    await waitForAcknowledgedUpdaterClaim(claimDeps({ queryCreatedAt: async () => 1_723_339_999 })),
    false
  )

  assert.equal(
    await waitForAcknowledgedUpdaterClaim(claimDeps({ queryCreatedAt: async () => null })),
    false,
    'an unprovable owner is not an authenticated one'
  )
})

test('a claim stamped before the spawn is rejected', async () => {
  assert.equal(
    await waitForAcknowledgedUpdaterClaim(claimDeps({ startedAfter: 1_723_330_500 })),
    false
  )
})

test('an excluded pid can never authenticate its own handoff', async () => {
  assert.equal(
    await waitForAcknowledgedUpdaterClaim(claimDeps({
      excludedPids: [41, 500],
      readMarker: () => liveMarker(500, 1_723_330_000)
    })),
    false
  )
})

test('readUpdateHandoffAck refuses partial and malformed sidecars', () => {
  const read = (body: string) => readUpdateHandoffAck(HOME, () => body)

  assert.deepEqual(read(`${NONCE}\n500\n1723330000\n`), { nonce: NONCE, pid: 500, createdAt: 1_723_330_000 })
  assert.equal(read(`${NONCE}\n500\n`), null, 'a torn ack is no ack')
  assert.equal(read(''), null)
  assert.equal(read(`${NONCE}\n-1\n1723330000\n`), null)
  assert.equal(read(`not-hex\n500\n1723330000\n`), null)
  assert.equal(
    readUpdateHandoffAck(HOME, () => { throw new Error('ENOENT') }),
    null,
    'a missing ack is no ack'
  )
})

// ---------------------------------------------------------------------------
// Bootstrap recovery (#B4). The old path wrote the child PID into the marker
// with transferUpdateMarkerIfOwnedBy and then "verified" a marker owned by
// that PID -- satisfied by construction.
// ---------------------------------------------------------------------------

function recoveryDeps(overrides: Record<string, unknown> = {}) {
  let clock = 0

  return {
    hermesHome: HOME,
    nonce: NONCE,
    childPid: 500,
    childCreatedAt: 1_723_330_000,
    desktopHoldsMarker: false,
    startedAfter: 1_723_329_999,
    isChildGenerationActive: () => true,
    timeoutMs: 300,
    pollMs: 50,
    now: () => clock,
    sleep: async (ms: number) => { clock += Math.max(ms, 50) },
    readMarker: () => null,
    readAck: () => null,
    queryCreatedAt: async () => 1_723_330_000,
    ...overrides
  }
}

test('bootstrap recovery rejects a child that never acknowledges or claims', async () => {
  assert.equal(await authenticateRecoveryUpdaterHandoff(recoveryDeps()), false)
})

test('bootstrap recovery accepts the child once it acknowledges with our nonce', async () => {
  assert.equal(
    await authenticateRecoveryUpdaterHandoff(recoveryDeps({ readAck: () => ackFixture() })),
    true
  )
})

test('bootstrap recovery accepts a marker the child claimed on its own', async () => {
  assert.equal(
    await authenticateRecoveryUpdaterHandoff(recoveryDeps({
      readMarker: () => liveMarker(500, 1_723_330_008)
    })),
    true,
    'the staged updater writes its own UpdateMarkerGuard claim, which we did not author'
  )
})

test('bootstrap recovery rejects a marker claimed by anyone but the child', async () => {
  assert.equal(
    await authenticateRecoveryUpdaterHandoff(recoveryDeps({
      readMarker: () => liveMarker(999, 1_723_330_008)
    })),
    false
  )
})

test('a marker the Desktop itself holds is never evidence about the child', async () => {
  // Repair keeps the marker under the Desktop's own claim, so the child
  // cannot possibly have written it. Only the spawn handle can speak here.
  assert.equal(
    await authenticateRecoveryUpdaterHandoff(recoveryDeps({
      desktopHoldsMarker: true,
      readMarker: () => liveMarker(500, 1_723_330_008),
      isChildGenerationActive: () => false
    })),
    false,
    'a dead child is not authenticated by a marker naming its pid'
  )

  assert.equal(
    await authenticateRecoveryUpdaterHandoff(recoveryDeps({
      desktopHoldsMarker: true,
      queryCreatedAt: async () => 1_723_339_999
    })),
    false,
    'a recycled pid is not the generation we spawned'
  )

  assert.equal(
    await authenticateRecoveryUpdaterHandoff(recoveryDeps({ desktopHoldsMarker: true })),
    true,
    'the live generation we started is the honest proof on the repair path'
  )
})

test('bootstrap recovery rejects a child whose generation was never captured', async () => {
  assert.equal(
    await authenticateRecoveryUpdaterHandoff(recoveryDeps({ childCreatedAt: null, readAck: () => ackFixture() })),
    false
  )
})

// ---------------------------------------------------------------------------
// The recovery handoff transaction. authenticateRecoveryUpdaterHandoff above
// decides whether the child is ours; this decides what the Desktop does with
// that answer, and in which order.
// ---------------------------------------------------------------------------

type FakeChild = { pid: number | null }

function handoffDeps(
  trace: string[],
  overrides: Partial<RecoveryUpdaterHandoffDeps<FakeChild, string>> = {}
): RecoveryUpdaterHandoffDeps<FakeChild, string> {
  return {
    hermesHome: HOME,
    nonce: NONCE,
    startedAfter: 1_723_329_999,
    repairClaim: CLAIM,
    spawn: () => {
      trace.push('spawn')

      return { pid: 500 }
    },
    observe: async () => {
      trace.push('observe')

      return { ok: true }
    },
    childPid: child => child.pid,
    captureCreatedAt: async () => {
      trace.push('capture')

      return 1_723_330_000
    },
    isChildGenerationActive: () => true,
    commit: () => {
      trace.push('commit')

      return 'handed-off'
    },
    restore: ({ markerTransferred, createdAt }) => {
      trace.push(`restore transferred=${markerTransferred} createdAt=${String(createdAt)}`)
    },
    authenticate: async () => {
      trace.push('authenticate')

      return true
    },
    transferMarker: async () => {
      trace.push('transfer')

      return true
    },
    authenticationError: 'The recovery updater did not acknowledge startup.',
    ...overrides
  }
}

test('the repair handoff authenticates before it hands over the marker', async () => {
  const trace: string[] = []
  const result = await runRecoveryUpdaterHandoff(handoffDeps(trace))

  assert.deepEqual(result, { ok: true, value: 'handed-off' })
  // #B4: the marker transfer used to run FIRST and then satisfy the check that
  // followed it. capture -> authenticate -> transfer is the whole fix.
  assert.deepEqual(
    trace.filter(step => step !== 'observe'),
    ['spawn', 'capture', 'authenticate', 'transfer', 'commit']
  )
})

test('a child that fails authentication never receives the marker', async () => {
  const trace: string[] = []

  const refuse = async () => {
    trace.push('authenticate')

    return false
  }

  const result = await runRecoveryUpdaterHandoff(handoffDeps(trace, { authenticate: refuse }))

  assert.equal(result.ok, false)
  assert.equal(result.ok === false && result.error, 'update-handoff-unacknowledged')
  assert.ok(!trace.includes('transfer'), 'the gate is never handed to an unauthenticated process')
  assert.ok(!trace.includes('commit'))
  assert.ok(trace.includes('restore transferred=false createdAt=1723330000'))
})

test('a refused marker transfer fails the handoff', async () => {
  const trace: string[] = []

  const refuse = async () => {
    trace.push('transfer')

    return false
  }

  const result = await runRecoveryUpdaterHandoff(handoffDeps(trace, { transferMarker: refuse }))

  // The Desktop's claim did not move, so the child cannot be allowed to
  // proceed as the update's owner.
  assert.equal(result.ok, false)
  assert.ok(!trace.includes('commit'))
  assert.ok(trace.includes('restore transferred=false createdAt=1723330000'))
})

test('the gentle path never transfers a marker it does not hold', async () => {
  const trace: string[] = []
  const result = await runRecoveryUpdaterHandoff(handoffDeps(trace, { repairClaim: null }))

  assert.deepEqual(result, { ok: true, value: 'handed-off' })
  assert.ok(!trace.includes('transfer'), 'the staged updater claims the marker itself')
})

test('the repair flag follows the claim the Desktop actually holds', async () => {
  const seen: boolean[] = []

  const spy = {
    authenticate: async (deps: { desktopHoldsMarker: boolean }) => {
      seen.push(deps.desktopHoldsMarker)

      return true
    }
  }

  await runRecoveryUpdaterHandoff(handoffDeps([], { ...spy, repairClaim: CLAIM }))
  await runRecoveryUpdaterHandoff(handoffDeps([], { ...spy, repairClaim: null }))

  assert.deepEqual(seen, [true, false])
})

test('a child with no pid or no captured generation is refused before authentication', async () => {
  const noPid: string[] = []

  const withoutPid = await runRecoveryUpdaterHandoff(
    handoffDeps(noPid, { spawn: () => ({ pid: null }) })
  )

  assert.equal(withoutPid.ok, false)
  assert.ok(!noPid.includes('authenticate'))
  assert.ok(noPid.includes('restore transferred=false createdAt=null'))

  const noGeneration: string[] = []

  const withoutGeneration = await runRecoveryUpdaterHandoff(
    handoffDeps(noGeneration, { captureCreatedAt: async () => null })
  )

  assert.equal(withoutGeneration.ok, false)
  assert.ok(!noGeneration.includes('authenticate'), 'an unread generation cannot be authenticated')
})

test('an updater that exits during the dwell is not a successful handoff', async () => {
  const trace: string[] = []

  const result = await runRecoveryUpdaterHandoff(
    handoffDeps(trace, { observe: async () => ({ ok: false, message: 'updater exited' }) })
  )

  assert.deepEqual(result, {
    ok: false,
    error: 'updater-spawn-failed',
    message: 'updater exited'
  })
  assert.ok(!trace.includes('commit'))
  // Authentication passed, so the marker did move; restore has to know that.
  assert.ok(trace.includes('restore transferred=true createdAt=1723330000'))
})

import { cancelWindowsUpdateWait } from './windows-update-apply'
import { formatWindowsHolderStopCommand } from './windows-update-force-release'

test('manual stop command includes only valid process identities and checks creation time', () => {
  const command = formatWindowsHolderStopCommand([12, 34, 12, -1, 0, NaN, 1.5, 2 ** 32].map(pid => ({ pid, createdAt: 100 })))!
  assert.equal(command.split('@{Id=').length - 1, 2)
  assert.ok(command.includes('@{Id=12;Started=100}'))
  assert.ok(command.includes('$null = $p.Handle'))
  assert.ok(command.indexOf('$p.StartTime') < command.indexOf('$p.Kill()'))
  assert.equal(formatWindowsHolderStopCommand([{ pid: 12, createdAt: NaN }]), null)
  assert.equal(formatWindowsHolderStopCommand([]), null)
})

test('waiting holds the update gate and reruns preflight before handoff', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }
  const events: string[] = []
  let preflights = 0

  const result = await applyWindowsUpdate({}, deps(state, {
    runPreflight: async (_prepared, claim) => {
      assert.equal(claim, LIVE_CLAIM)
      events.push('preflight')

      return ++preflights < 3 ? { kind: 'blocked', reason: 'holders', message: 'locked' } : clearPreflight()
    },
    waitForBlockers: async () => {
      assert.equal(state.phase, 'waiting')
      assert.equal(windowsUpdateBlocksBackendStart(state), true)
      assert.equal(windowsUpdateIsBusy(state), true)
      events.push('wait')

      return 'clear'
    },
    releaseMarker: () => { events.push('release');

 return true },
    authorize: () => { events.push('authorize');

 return PERMIT },
    commit: () => { events.push('commit') }
  }))

  assert.equal(result.ok, true)
  assert.deepEqual(events, ['preflight', 'wait', 'preflight', 'wait', 'preflight', 'authorize', 'commit'])
  assert.equal(state.phase, 'idle')
})

test('cancel while waiting prevents handoff even if the probe returns clear and restores before resolving', async () => {
  const state: WindowsUpdateState = { phase: 'idle' }
  const events: string[] = []

  const result = await applyWindowsUpdate({}, deps(state, {
    runPreflight: async () => ({ kind: 'blocked', reason: 'holders', message: 'locked' }),
    waitForBlockers: async () => {
      assert.equal(cancelWindowsUpdateWait(state), true)
      assert.equal(cancelWindowsUpdateWait(state), false)

      return 'clear'
    },
    authorize: () => { throw new Error('must not authorize') },
    releaseMarker: () => { events.push('release');

 return true },
    restoreBackends: async () => {
      assert.equal(state.phase, 'restoring')
      events.push('restore')
    }
  }))

  assert.equal(result.ok, false)
  assert.equal('error' in result && result.error, 'update-cancelled')
  assert.deepEqual(events, ['release', 'restore'])
  assert.equal(state.phase, 'idle')
  assert.equal(state.waitAbortController, undefined)
})

test('wait failures never authorize handoff and restore through the abort path', async () => {
  for (const outcome of ['claim-lost', 'probe-failed', 'throw'] as const) {
    const state: WindowsUpdateState = { phase: 'idle' }
    let restored = false

    const result = await applyWindowsUpdate({}, deps(state, {
      runPreflight: async () => ({ kind: 'blocked', reason: 'holders', message: 'locked' }),
      waitForBlockers: async () => {
        if (outcome === 'throw') { throw new Error('probe unavailable') }

        return outcome
      },
      authorize: () => { throw new Error('must not authorize') },
      restoreBackends: async () => { restored = true }
    }))

    assert.equal('error' in result && result.error, 'venv-probe-failed')
    assert.equal(restored, true)
    assert.equal(state.phase, 'idle')
    assert.equal(state.waitAbortController, undefined)
  }
})
