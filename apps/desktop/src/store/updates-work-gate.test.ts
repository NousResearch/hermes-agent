import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopUpdateApplyResult, DesktopUpdateProgress } from '@/global'

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), dismissNotification: vi.fn() }))

const { atom } = await import('nanostores')

vi.mock('@/store/connections', () => ({ $connectionsRegistry: atom<unknown>(null) }))
vi.mock('@/store/gateway-reconnect', () => ({ reconnectGateway: vi.fn() }))

const { $updateApply, applyUpdates, cancelUpdateWhenIdle, resetUpdateApplyState, startUpdatePoller } =
  await import('./updates')

const CHAT = { kind: 'chats' as const, count: 1, titles: ['Refactor auth'] }

let apply: ReturnType<typeof vi.fn>
let cancelWhenIdle: ReturnType<typeof vi.fn>
let progress: ((payload: DesktopUpdateProgress) => void) | null = null

beforeEach(() => {
  resetUpdateApplyState()
  apply = vi.fn()
  cancelWhenIdle = vi.fn().mockResolvedValue({ cancelled: true })
  // `progress` survives tests: the poller subscribes once per module.
  ;(globalThis as unknown as { window: unknown }).window = {
    addEventListener: () => undefined,
    hermesDesktop: {
      updates: {
        apply,
        cancelWhenIdle,
        check: vi.fn().mockResolvedValue({ supported: true }),
        onProgress: (callback: (payload: DesktopUpdateProgress) => void) => {
          progress = callback

          return () => undefined
        }
      }
    }
  }
})

describe('in-app update while agent work runs', () => {
  it('lands on the choice, naming the work, when the gate answers active-work', async () => {
    apply.mockResolvedValue({ ok: false, error: 'active-work', waitingFor: [CHAT] } satisfies DesktopUpdateApplyResult)

    await applyUpdates()

    expect($updateApply.get()).toMatchObject({ applying: false, stage: 'busy', waitingFor: [CHAT] })
  })

  it('"update when idle" shows a cancellable wait that streams what it waits for', async () => {
    let resolve: (result: DesktopUpdateApplyResult) => void = () => undefined
    apply.mockReturnValue(new Promise(r => (resolve = r)))
    startUpdatePoller()

    const pending = applyUpdates({ mode: 'when-idle' })

    expect(apply).toHaveBeenCalledWith({ mode: 'when-idle' })
    expect($updateApply.get()).toMatchObject({ applying: false, stage: 'waiting' })

    progress?.({ stage: 'waiting', message: '', percent: null, error: null, at: 1, waitingFor: [CHAT], deadlineAt: 9 })
    expect($updateApply.get()).toMatchObject({ applying: false, stage: 'waiting', waitingFor: [CHAT], deadlineAt: 9 })

    await cancelUpdateWhenIdle()
    expect(cancelWhenIdle).toHaveBeenCalled()
    expect($updateApply.get().stage).toBe('idle')

    // The wait's late "cancelled" answer must not resurrect any state.
    resolve({ ok: false, error: 'cancelled' })
    await pending
    expect($updateApply.get().stage).toBe('idle')
  })

  it('the cancelling window stays closeable when its own cancel broadcast lands after the local reset', async () => {
    let resolve: (result: DesktopUpdateApplyResult) => void = () => undefined
    apply.mockReturnValue(new Promise(r => (resolve = r)))
    startUpdatePoller()

    const pending = applyUpdates({ mode: 'when-idle' })
    await cancelUpdateWhenIdle()

    // Main answers the cancel by telling every window the wait ended, this one included.
    progress?.({ stage: 'idle', message: '', percent: null, error: null, at: 2 })
    resolve({ ok: false, error: 'cancelled' })
    await pending

    expect($updateApply.get()).toMatchObject({ applying: false, stage: 'idle' })
  })

  it('a wait that ran out its bound ends on a closeable error, not a spinner', async () => {
    apply.mockResolvedValue({ ok: false, error: 'idle-timeout', waitingFor: [CHAT] })

    await applyUpdates({ mode: 'when-idle' })

    expect($updateApply.get()).toMatchObject({ applying: false, stage: 'error', error: 'idle-timeout' })
  })

  it('a window watching a wait another window started follows its cancel and its timeout', () => {
    startUpdatePoller()
    const waiting = { message: '', percent: null, error: null, at: 1, waitingFor: [CHAT], deadlineAt: 9 }

    // This window never called apply: it only sees the broadcast wait.
    progress?.({ stage: 'waiting', ...waiting })
    expect($updateApply.get().stage).toBe('waiting')

    progress?.({ stage: 'idle', message: '', percent: null, error: null, at: 2 })
    expect($updateApply.get()).toMatchObject({ applying: false, stage: 'idle' })

    progress?.({ stage: 'waiting', ...waiting })
    progress?.({
      stage: 'error',
      message: 'Hermes stayed busy',
      percent: null,
      error: 'idle-timeout',
      at: 3,
      waitingFor: [CHAT]
    })
    expect($updateApply.get()).toMatchObject({
      applying: false,
      stage: 'error',
      error: 'idle-timeout',
      waitingFor: [CHAT]
    })
  })
})
