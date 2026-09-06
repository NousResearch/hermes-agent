import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { UpdateMarkerClaim } from './update-marker'
import type { UpdateMutationPermit, UpdatePreflightOutcome } from './update-preflight'
import {
  applyWindowsUpdate,
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
