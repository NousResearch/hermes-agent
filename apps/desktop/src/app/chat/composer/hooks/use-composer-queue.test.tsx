import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  isQueueParked,
  MAX_AUTO_DRAIN_ATTEMPTS,
  parkQueuedPrompts
} from '@/store/composer-queue'
import { setPlanMode, withPlanMode } from '@/store/plan-mode'
import { setSessionsLoading } from '@/store/session'

import type { QueueEditState } from '../composer-utils'
import type { ChatBarProps } from '../types'

import { useComposerQueue } from './use-composer-queue'

// The park ↔ drain contract at the hook level. The store tests pin the pure
// pieces (shouldAutoDrain, park bookkeeping); these pin the wiring — the
// auto-drain effect honoring the park, and send-now-while-busy lifting it so
// the settle drain still flows (the regression that sank the old blanket
// interrupt latch).

const SESSION_KEY = 'stored-session-queue-hook'

function renderQueueHook(
  overrides: {
    busy?: boolean
    onCancel?: () => void
    onSteer?: ChatBarProps['onSteer']
    /** Wrap the spy the way ChatBar wraps its send (e.g. withPlanMode). */
    wrapSubmit?: (submit: ChatBarProps['onSubmit']) => ChatBarProps['onSubmit']
  } = {}
) {
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const onCancel = overrides.onCancel ?? vi.fn()
  const onSteer = overrides.onSteer
  const queueEditRef: { current: QueueEditState | null } = { current: null }

  const hook = renderHook(
    ({ busy }: { busy: boolean }) =>
      useComposerQueue({
        activeQueueSessionKey: SESSION_KEY,
        attachments: [],
        busy,
        clearDraft: () => undefined,
        draftRef: { current: '' },
        focusInput: () => undefined,
        loadIntoComposer: () => undefined,
        onCancel,
        onSteer,
        onSubmit: overrides.wrapSubmit ? overrides.wrapSubmit(onSubmit) : onSubmit,
        queueEditRef,
        queueSessionKey: SESSION_KEY,
        sessionId: 'rt-session-queue-hook'
      }),
    { initialProps: { busy: overrides.busy ?? false } }
  )

  return { hook, onCancel, onSubmit }
}

describe('useComposerQueue park integration', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    setSessionsLoading(false)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    setSessionsLoading(true)
    setPlanMode(false)
  })

  it('drains through the plan-mode wrapper: /plan is sent, the queue keeps the raw text', async () => {
    setPlanMode(true)
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'refactor the parser' })
    const { hook, onSubmit: send } = renderQueueHook({ busy: true, wrapSubmit: withPlanMode })
    send.mockResolvedValue(false)

    await act(async () => {
      await hook.result.current.drainNextQueued()
    })

    expect(send).toHaveBeenLastCalledWith('/plan refactor the parser', expect.objectContaining({ fromQueue: true }))
    // Rejected: the entry stays as typed, for queue editing and the next attempt.
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['refactor the parser'])

    // The mode at send time decides: turned off before the retry, it goes out raw.
    setPlanMode(false)
    send.mockResolvedValue(true)
    await act(async () => {
      await hook.result.current.drainNextQueued()
    })

    expect(send).toHaveBeenLastCalledWith('refactor the parser', expect.objectContaining({ fromQueue: true }))
    expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
  })

  it('reschedules rejected foreground drains to a bounded stop and keeps manual recovery', async () => {
    vi.useFakeTimers()

    try {
      const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'recoverable' })!
      const { hook, onSubmit } = renderQueueHook({ busy: true })
      onSubmit.mockResolvedValue(false)
      hook.rerender({ busy: false })
      await act(async () => {
        await Promise.resolve()
      })

      for (let attempt = 1; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt++) {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(30_000)
        })
      }

      expect(onSubmit).toHaveBeenCalledTimes(MAX_AUTO_DRAIN_ATTEMPTS)
      await act(async () => {
        await vi.advanceTimersByTimeAsync(300_000)
      })
      expect(onSubmit).toHaveBeenCalledTimes(MAX_AUTO_DRAIN_ATTEMPTS)
      expect(getQueuedPrompts(SESSION_KEY).map(item => item.text)).toEqual(['recoverable'])
      onSubmit.mockResolvedValue(true)
      await act(async () => {
        await hook.result.current.sendQueuedNow(entry.id)
      })
      expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  it('cancels a pending retry on unmount without losing the queued entry', async () => {
    vi.useFakeTimers()

    try {
      enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'keep on disconnect' })
      const { hook, onSubmit } = renderQueueHook({ busy: true })
      onSubmit.mockRejectedValue(new Error('unavailable'))
      hook.rerender({ busy: false })
      await act(async () => {
        await Promise.resolve()
      })
      expect(vi.getTimerCount()).toBeGreaterThan(0)
      hook.unmount()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(300_000)
      })
      expect(onSubmit).toHaveBeenCalledTimes(1)
      expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('auto-drains an unparked queue once idle', async () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'flows' })

    const { onSubmit } = renderQueueHook()

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(0)
  })

  it('holds a parked queue at the idle settle (the Stop edge)', async () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'halted' })
    parkQueuedPrompts(SESSION_KEY)

    const { hook, onSubmit } = renderQueueHook({ busy: true })

    // The Stop settle: busy flips false with the park in place.
    hook.rerender({ busy: false })

    await act(async () => {
      await Promise.resolve()
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)
  })

  it('drainNextQueued sends a parked entry and lifts the park (manual resume)', async () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'resumed' })
    parkQueuedPrompts(SESSION_KEY)

    const { hook, onSubmit } = renderQueueHook()

    await act(async () => {
      await hook.result.current.drainNextQueued()
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(isQueueParked(SESSION_KEY)).toBe(false)
  })

  it('sendQueuedNow while busy unparks so the settle drain flows (no stale latch)', async () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'send me now' })
    parkQueuedPrompts(SESSION_KEY)

    const { hook, onCancel, onSubmit } = renderQueueHook({ busy: true })
    const target = getQueuedPrompts(SESSION_KEY).find(e => e.id !== first!.id)!

    act(() => {
      hook.result.current.sendQueuedNow(target.id)
    })

    // The interrupt fired and the park lifted — this interrupt exists to reach
    // the queue, not to halt it.
    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(isQueueParked(SESSION_KEY)).toBe(false)

    // Turn settles → the promoted entry drains.
    hook.rerender({ busy: false })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('send me now')
  })

  it('steerQueuedNow delivers via onSteer without cancelling and removes the entry', async () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'steer me' })
    const onSteer = vi.fn(async () => true)
    const { hook, onCancel, onSubmit } = renderQueueHook({ busy: true, onSteer })

    await act(async () => {
      expect(await hook.result.current.steerQueuedNow(entry!.id)).toBe(true)
    })

    expect(onSteer).toHaveBeenCalledWith('steer me')
    // A redirect rides the live turn: no interrupt, no submit.
    expect(onCancel).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(0)
  })

  it('a rejected steer leaves the entry queued so the settle drain still sends it', async () => {
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'kept on reject' })
    const onSteer = vi.fn(async () => false)
    const { hook, onSubmit } = renderQueueHook({ busy: true, onSteer })

    await act(async () => {
      expect(await hook.result.current.steerQueuedNow(entry!.id)).toBe(false)
    })

    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)

    // Turn settles → the surviving entry drains normally.
    hook.rerender({ busy: false })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('kept on reject')
  })

  it('steerQueuedNow refuses unsteerable entries (slash commands execute, never steer)', async () => {
    const slash = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: '/compress' })
    const onSteer = vi.fn(async () => true)

    // Busy, but a slash command never steers. (Idle needs no case of its own:
    // an idle session auto-drains its queue, so there is never an entry left
    // to steer — asserting that here would just re-test auto-drain.)
    const busy = renderQueueHook({ busy: true, onSteer })

    await act(async () => {
      expect(await busy.hook.result.current.steerQueuedNow(slash!.id)).toBe(false)
    })

    expect(onSteer).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)
  })

  it('a delivered steer lifts the park so the rest of the queue flows', async () => {
    const steerable = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'redirect' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'follows after' })
    parkQueuedPrompts(SESSION_KEY)

    const onSteer = vi.fn(async () => true)
    const { hook } = renderQueueHook({ busy: true, onSteer })

    await act(async () => {
      expect(await hook.result.current.steerQueuedNow(steerable!.id)).toBe(true)
    })

    expect(isQueueParked(SESSION_KEY)).toBe(false)
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)
  })

  it('does not auto-drain restored queues while the session list is still loading', async () => {
    setSessionsLoading(true)
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'wait for session list' })

    const { onSubmit } = renderQueueHook()

    await act(async () => {
      await Promise.resolve()
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)
  })

  it('auto-drains a restored queue once the session list finishes loading', async () => {
    setSessionsLoading(true)
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'send after load' })

    const { hook, onSubmit } = renderQueueHook()

    await act(async () => {
      await Promise.resolve()
    })
    expect(onSubmit).not.toHaveBeenCalled()

    setSessionsLoading(false)
    hook.rerender({ busy: false })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(0)
  })

  describe('deliverQueuedNow (double-Enter while busy)', () => {
    it('steers a text entry into the live turn instead of interrupting it', async () => {
      const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'fix the header too' })!
      const onSteer = vi.fn(async () => true)
      const { hook, onCancel, onSubmit } = renderQueueHook({ busy: true, onSteer })

      await act(async () => {
        expect(await hook.result.current.deliverQueuedNow(entry.id)).toBe(true)
      })

      expect(onSteer).toHaveBeenCalledWith('fix the header too')
      expect(onCancel).not.toHaveBeenCalled()
      expect(onSubmit).not.toHaveBeenCalled()
      expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(0)
    })

    it('falls back to send-now (interrupt) when the live turn refuses the steer', async () => {
      const other = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'older' })!
      const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'refused' })!
      const onSteer = vi.fn(async () => false)
      const { hook, onCancel } = renderQueueHook({ busy: true, onSteer })

      await act(async () => {
        await hook.result.current.deliverQueuedNow(entry.id)
      })

      expect(onSteer).toHaveBeenCalledTimes(1)
      expect(onCancel).toHaveBeenCalledTimes(1)
      expect(getQueuedPrompts(SESSION_KEY).map(e => e.id)).toEqual([entry.id, other.id])
    })

    it('interrupts directly for a payload a steer cannot carry', async () => {
      const entry = enqueueQueuedPrompt(SESSION_KEY, {
        attachments: [{ id: 'shot', kind: 'image', label: 'shot.png' }],
        text: 'look at this'
      })!

      const onSteer = vi.fn(async () => true)
      const { hook, onCancel } = renderQueueHook({ busy: true, onSteer })

      await act(async () => {
        await hook.result.current.deliverQueuedNow(entry.id)
      })

      expect(onSteer).not.toHaveBeenCalled()
      expect(onCancel).toHaveBeenCalledTimes(1)
    })

    it('a repeat Enter while the steer is in flight never escalates to an interrupt', async () => {
      const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'once' })!

      let accept: (value: boolean) => void = () => {}

      const onSteer = vi.fn(() => new Promise<boolean>(resolve => (accept = resolve)))
      const { hook, onCancel } = renderQueueHook({ busy: true, onSteer })

      let first: Promise<unknown> = Promise.resolve()

      await act(async () => {
        first = hook.result.current.deliverQueuedNow(entry.id)
        expect(await hook.result.current.deliverQueuedNow(entry.id)).toBe(true)
      })

      await act(async () => {
        accept(true)
        await first
      })

      expect(onSteer).toHaveBeenCalledTimes(1)
      expect(onCancel).not.toHaveBeenCalled()
    })
  })
})
