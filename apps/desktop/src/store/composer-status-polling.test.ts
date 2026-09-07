import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $backgroundRunningSessionIds,
  $backgroundStatusBySession,
  clearAllSessionBackground,
  reconcileBackgroundProcesses,
  refreshBackgroundProcesses,
  resetSessionBackground,
  stopBackgroundProcess
} from './composer-status'
import { $gateway, requestGatewayForAgent, requestGatewayForProfile } from './gateway'
import { wipeSessionListsForGatewaySwitch } from './gateway-switch'
import { notifyError } from './notifications'
import { isSessionGone, resetBackgroundPollingGuard, resetRuntimeGoneHealing } from './runtime-gone'
import { $sessions } from './session'
import { healsByStoredId } from './session-gone-latch'
import { $sessionStates, $sessionTiles } from './session-states'

vi.mock(import('./gateway'), async importOriginal => ({
  ...(await importOriginal()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn()
}))

vi.mock('./native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))
vi.mock('./notifications', () => ({ notifyError: vi.fn() }))

const running = (id: string) => ({ command: '# Watching CI', session_id: id, status: 'running' })
const exited = (id: string) => ({ ...running(id), status: 'exited', exit_code: 0 })
const request = vi.fn()

beforeEach(() => {
  vi.useFakeTimers()
  clearAllSessionBackground()
  resetBackgroundPollingGuard()
  resetRuntimeGoneHealing()
  $sessions.set([])
  $sessionStates.set({})
  $sessionTiles.set([])
  vi.mocked(requestGatewayForAgent).mockReset().mockResolvedValue({ processes: [] })
  vi.mocked(requestGatewayForProfile).mockReset().mockResolvedValue({ processes: [] })
  request.mockReset().mockResolvedValue({ processes: [] })
  $gateway.set({ request } as never)
})

afterEach(() => {
  clearAllSessionBackground()
  resetBackgroundPollingGuard()
  resetRuntimeGoneHealing()
  $gateway.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
})

describe('shared background process refresh', () => {
  it.each(['snapshot', 'reset', 'wipe'] as const)(
    'ignores a stale rejected read after %s without latching or healing the current session',
    async invalidation => {
      const sid = `stale-${invalidation}`
      let reject!: (reason: unknown) => void
      request.mockReturnValueOnce(
        new Promise((_resolve, fail) => {
          reject = fail
        })
      )
      const pending = refreshBackgroundProcesses(sid)

      if (invalidation === 'snapshot') {
        reconcileBackgroundProcesses(sid, [running('current')])
      } else if (invalidation === 'reset') {
        resetSessionBackground(sid)
      } else {
        wipeSessionListsForGatewaySwitch()
      }

      $sessionStates.set({ [sid]: { storedSessionId: 'current-stored' } as never })
      $sessionTiles.set([{ storedSessionId: 'current-stored', runtimeId: sid }])
      healsByStoredId.set('current-stored', 2)
      reject(new Error('session not found'))
      await pending

      expect(isSessionGone(sid)).toBe(false)
      expect(healsByStoredId.get('current-stored')).toBe(2)
      expect($sessionTiles.get()[0]?.runtimeId).toBe(sid)
      await refreshBackgroundProcesses(sid)
      expect(request).toHaveBeenCalledTimes(2)
      expect(healsByStoredId.has('current-stored')).toBe(false)
    }
  )

  it.each(['snapshot', 'reset', 'wipe'] as const)(
    'does not refund the current recovery budget for a stale successful read after %s',
    async invalidation => {
      const sid = `stale-success-${invalidation}`
      let resolve!: (value: unknown) => void
      request.mockReturnValueOnce(
        new Promise(done => {
          resolve = done
        })
      )
      const pending = refreshBackgroundProcesses(sid)

      if (invalidation === 'snapshot') {
        reconcileBackgroundProcesses(sid, [])
      } else if (invalidation === 'reset') {
        resetSessionBackground(sid)
      } else {
        wipeSessionListsForGatewaySwitch()
      }

      $sessionStates.set({ [sid]: { storedSessionId: 'current-stored' } as never })
      healsByStoredId.set('current-stored', 2)
      resolve({ processes: [running('obsolete')] })
      await pending

      expect($backgroundStatusBySession.get()[sid]).toBeUndefined()
      expect(isSessionGone(sid)).toBe(false)
      expect(healsByStoredId.get('current-stored')).toBe(2)
    }
  )

  it('polls only known running work and stops requests after a gone-runtime rejection', async () => {
    $sessionStates.set({ idle: { storedSessionId: 'idle-stored' } as never })
    reconcileBackgroundProcesses('gone', [running('ci')])
    request.mockRejectedValue(new Error('session not found'))
    await vi.advanceTimersByTimeAsync(5_000)
    expect(request).toHaveBeenCalledExactlyOnceWith('process.list', { session_id: 'gone' })
    await vi.advanceTimersByTimeAsync(30_000)
    expect(request).toHaveBeenCalledTimes(1)
    expect(isSessionGone('gone')).toBe(true)
  })

  it('wipes cached rows, dismissals, timers and pending reads on a gateway switch', async () => {
    reconcileBackgroundProcesses('outgoing', [running('ci'), exited('finished')])
    let resolve!: (value: unknown) => void
    request.mockReturnValueOnce(
      new Promise(done => {
        resolve = done
      })
    )
    const pending = refreshBackgroundProcesses('outgoing')
    wipeSessionListsForGatewaySwitch()
    expect($backgroundStatusBySession.get()).toEqual({})
    resolve({ processes: [running('ci'), running('late')] })
    await pending
    request.mockClear()
    await vi.advanceTimersByTimeAsync(30_000)
    expect($backgroundStatusBySession.get()).toEqual({})
    expect(request).not.toHaveBeenCalled()
    // Old auto-dismiss timers must not dismiss a recycled id on the new backend.
    reconcileBackgroundProcesses('outgoing', [running('finished')])
    expect($backgroundStatusBySession.get().outgoing?.[0]?.id).toBe('finished')
  })

  it('does not let a late kill dismiss a recycled process on a new gateway', async () => {
    reconcileBackgroundProcesses('kill-switch', [running('ci')])
    let resolve!: (value: unknown) => void
    request.mockReturnValueOnce(
      new Promise(done => {
        resolve = done
      })
    )
    const pending = stopBackgroundProcess('kill-switch', 'ci')
    wipeSessionListsForGatewaySwitch()
    reconcileBackgroundProcesses('kill-switch', [running('ci')])
    resolve({})
    await pending
    expect($backgroundStatusBySession.get()['kill-switch']?.[0]?.state).toBe('running')
  })

  it('rewind kills only running processes through their owner and drops all rows', async () => {
    $sessions.set([{ id: 'reset-stored', profile: 'work' } as never])
    $sessionStates.set({ reset: { storedSessionId: 'reset-stored' } as never })
    reconcileBackgroundProcesses('reset', [running('ci'), exited('done')])
    resetSessionBackground('reset')
    expect(requestGatewayForProfile).toHaveBeenCalledExactlyOnceWith(
      'work',
      'process.kill',
      { process_id: 'ci', session_id: 'reset' },
      undefined,
      undefined
    )
    expect(request).not.toHaveBeenCalled()
    expect($backgroundStatusBySession.get().reset).toBeUndefined()
    reconcileBackgroundProcesses('reset', [running('ci'), exited('done')])
    expect($backgroundStatusBySession.get().reset).toBeUndefined()
  })

  it('retains a running row when there is no gateway to confirm a kill', async () => {
    reconcileBackgroundProcesses('disconnected', [running('ci')])
    $gateway.set(null)
    await stopBackgroundProcess('disconnected', 'ci')
    expect($backgroundStatusBySession.get().disconnected?.[0]?.state).toBe('running')
    expect(notifyError).toHaveBeenCalled()
  })

  it('routes an unmounted runtime through its stored session owner, not the active profile', async () => {
    $sessions.set([{ id: 'stored', profile: 'work' } as never])
    $sessionStates.set({ owned: { storedSessionId: 'stored' } as never })
    reconcileBackgroundProcesses('owned', [running('ci')])
    vi.mocked(requestGatewayForProfile).mockResolvedValue({ processes: [exited('ci')] })
    await vi.advanceTimersByTimeAsync(5_000)
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'work',
      'process.list',
      { session_id: 'owned' },
      undefined,
      undefined
    )
    expect(request).not.toHaveBeenCalled()
    expect($backgroundStatusBySession.get().owned?.[0]?.state).toBe('done')
  })

  it('preserves an exact tile connection route ahead of a row profile for reads and kills', async () => {
    $sessions.set([{ id: 'tile-stored', profile: 'other' } as never])
    $sessionTiles.set([
      {
        storedSessionId: 'tile-stored',
        runtimeId: 'tile-runtime',
        ownerRoute: { connectionId: 'remote-a', profile: 'work' }
      }
    ])
    reconcileBackgroundProcesses('tile-runtime', [running('ci')])
    await refreshBackgroundProcesses('tile-runtime')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-a', 'work', 'process.list', {
      session_id: 'tile-runtime'
    })
    reconcileBackgroundProcesses('tile-runtime', [running('ci')])
    await stopBackgroundProcess('tile-runtime', 'ci')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-a', 'work', 'process.kill', {
      session_id: 'tile-runtime',
      process_id: 'ci'
    })
    expect(request).not.toHaveBeenCalled()
    expect($backgroundStatusBySession.get()['tile-runtime']).toBeUndefined()
  })

  it('preserves an unmounted session registry connection for background reads and kills', async () => {
    $sessions.set([{ id: 'remote-stored', profile: 'work', connection_id: 'remote-a' } as never])
    $sessionStates.set({ 'remote-runtime': { storedSessionId: 'remote-stored' } as never })
    reconcileBackgroundProcesses('remote-runtime', [running('ci')])
    await refreshBackgroundProcesses('remote-runtime')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-a', 'work', 'process.list', {
      session_id: 'remote-runtime'
    })
    reconcileBackgroundProcesses('remote-runtime', [running('ci')])
    await stopBackgroundProcess('remote-runtime', 'ci')
    expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-a', 'work', 'process.kill', {
      session_id: 'remote-runtime',
      process_id: 'ci'
    })
    expect(requestGatewayForProfile).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })

  it('rechecks a tool completion that arrives during an older empty seed request', async () => {
    let resolve!: (value: unknown) => void
    request.mockReturnValueOnce(
      new Promise(done => {
        resolve = done
      })
    )
    request.mockResolvedValue({ processes: [running('new-process')] })
    const seed = refreshBackgroundProcesses('spawning')
    const toolCompletion = refreshBackgroundProcesses('spawning')
    expect(request).toHaveBeenCalledTimes(1)
    resolve({ processes: [] })
    await Promise.all([seed, toolCompletion])
    expect(request).toHaveBeenCalledTimes(2)
    expect($backgroundStatusBySession.get().spawning?.[0]?.state).toBe('running')
  })

  it('coalesces event, mount and polling requests while a snapshot is pending', async () => {
    reconcileBackgroundProcesses('coalesced', [running('ci')])
    request.mockResolvedValue({ processes: [exited('ci')] })
    let resolve!: (value: unknown) => void
    request.mockReturnValueOnce(
      new Promise(done => {
        resolve = done
      })
    )
    const first = refreshBackgroundProcesses('coalesced')
    const second = refreshBackgroundProcesses('coalesced')
    await vi.advanceTimersByTimeAsync(15_000)
    expect(request).toHaveBeenCalledTimes(1)
    resolve({ processes: [exited('ci')] })
    await Promise.all([first, second])
    expect($backgroundStatusBySession.get().coalesced?.[0]?.state).toBe('done')
  })

  it('does not let a late snapshot roll back a newer reconciliation', async () => {
    reconcileBackgroundProcesses('late', [running('ci')])
    let resolve!: (value: unknown) => void
    request.mockReturnValueOnce(
      new Promise(done => {
        resolve = done
      })
    )
    const pending = refreshBackgroundProcesses('late')
    reconcileBackgroundProcesses('late', [exited('ci')])
    resolve({ processes: [running('ci')] })
    await pending
    expect($backgroundStatusBySession.get().late?.[0]?.state).toBe('done')
  })

  it('fences a pre-reset snapshot including processes not yet seen by the renderer', async () => {
    let resolve!: (value: unknown) => void
    request.mockReturnValueOnce(
      new Promise(done => {
        resolve = done
      })
    )
    const pending = refreshBackgroundProcesses('rewound')
    resetSessionBackground('rewound')
    resolve({ processes: [running('unseen')] })
    await pending
    expect($backgroundStatusBySession.get().rewound).toBeUndefined()
  })
  it('clears silent exits without any composer or status subscriber mounted', async () => {
    reconcileBackgroundProcesses('unmounted', [running('ci')])
    request.mockResolvedValue({ processes: [exited('ci')] })

    await vi.advanceTimersByTimeAsync(5_000)

    expect(request).toHaveBeenCalledWith('process.list', { session_id: 'unmounted' })
    expect($backgroundStatusBySession.get().unmounted?.[0]?.state).toBe('done')
    expect($backgroundRunningSessionIds.get()).not.toContain('unmounted')
    request.mockClear()
    await vi.advanceTimersByTimeAsync(30_000)
    expect(request).not.toHaveBeenCalled()
  })
})
