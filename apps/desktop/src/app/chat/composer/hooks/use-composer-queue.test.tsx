import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearComposerTerminalSelections, setComposerTerminalSelection } from '@/store/composer'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getFrozenQueuedTransport,
  getQueuedPrompts,
  isQueueParked,
  MAX_AUTO_DRAIN_ATTEMPTS,
  parkQueuedPrompts,
  resetFrozenQueuedTransportsForTests,
  simulateComposerQueueReloadForTests
} from '@/store/composer-queue'
import { $notifications, clearNotifications } from '@/store/notifications'
import { setSessionsLoading } from '@/store/session'

import type { QueueEditState } from '../composer-utils'
import type { ChatBarProps } from '../types'

import { useComposerQueue } from './use-composer-queue'

// The park ↔ drain contract at the hook level. The store tests pin the pure
// pieces (shouldAutoDrain, park bookkeeping); these pin the wiring — the
// auto-drain effect honoring the park, and per-entry Send now preserving the
// rest of the queue instead of silently resuming it.

const SESSION_KEY = 'stored-session-queue-hook'

function renderQueueHook(
  overrides: {
    busy?: boolean
    draft?: string
    onCancel?: ChatBarProps['onCancel']
    onSteer?: ChatBarProps['onSteer']
    onSubmit?: ChatBarProps['onSubmit']
  } = {}
) {
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(overrides.onSubmit ?? (async () => true))
  const onCancel = overrides.onCancel ?? vi.fn()
  const onSteer = overrides.onSteer
  const queueEditRef: { current: QueueEditState | null } = { current: null }
  const draftRef = { current: overrides.draft ?? '' }

  const hook = renderHook(
    ({ busy }: { busy: boolean }) =>
      useComposerQueue({
        activeQueueSessionKey: SESSION_KEY,
        attachments: [],
        busy,
        clearDraft: () => {
          draftRef.current = ''
        },
        draftRef,
        focusInput: () => undefined,
        loadIntoComposer: (text: string) => {
          draftRef.current = text
        },
        onCancel,
        onSteer,
        onSubmit,
        queueEditRef,
        queueSessionKey: SESSION_KEY,
        sessionId: 'rt-session-queue-hook'
      }),
    { initialProps: { busy: overrides.busy ?? false } }
  )

  return { draftRef, hook, onCancel, onSubmit }
}

describe('useComposerQueue park integration', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    resetFrozenQueuedTransportsForTests()
    clearNotifications()
    clearComposerTerminalSelections()
    clearNotifications()
    resetFrozenQueuedTransportsForTests()
    setSessionsLoading(false)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    resetFrozenQueuedTransportsForTests()
    clearNotifications()
    clearComposerTerminalSelections()
    clearNotifications()
    resetFrozenQueuedTransportsForTests()
    setSessionsLoading(true)
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

  it('sends only the selected entry after interrupt, leaving the other entries parked', async () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })!
    const target = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'send me now' })!

    const { hook, onCancel, onSubmit } = renderQueueHook({ busy: true })

    act(() => {
      hook.result.current.sendQueuedNow(target.id)
    })

    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(isQueueParked(SESSION_KEY)).toBe(true)
    // Manual Send now must not reorder or drain the first item.
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([first.id, target.id])

    hook.rerender({ busy: false })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('send me now')
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['first'])
    expect(isQueueParked(SESSION_KEY)).toBe(true)
  })


  it('explicit Stop after Send now cancels the pending delivery, even if already parked', async () => {
    const target = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'do not send' })!
    const { hook, onSubmit } = renderQueueHook({ busy: true })

    act(() => {
      void hook.result.current.sendQueuedNow(target.id)
      // Transcript Stop and composer Stop both park the session queue.
      parkQueuedPrompts(SESSION_KEY)
    })

    hook.rerender({ busy: false })
    await act(async () => {
      await Promise.resolve()
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([target.id])
    expect(isQueueParked(SESSION_KEY)).toBe(true)
  })

  it('does not dispatch before cancellation is acknowledged even if busy briefly becomes false', async () => {
    let acknowledgeCancel!: () => void
    const cancelling = new Promise<void>(resolve => {
      acknowledgeCancel = resolve
    })
    const target = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'after cancel' })!
    const { hook, onSubmit } = renderQueueHook({ busy: true, onCancel: () => cancelling })

    act(() => {
      void hook.result.current.sendQueuedNow(target.id)
    })
    hook.rerender({ busy: false })
    await act(async () => {
      await Promise.resolve()
    })
    expect(onSubmit).not.toHaveBeenCalled()

    await act(async () => {
      acknowledgeCancel()
      await cancelling
    })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('after cancel')
  })

  it('allows reselecting an entry while cancellation is pending', async () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first choice' })!
    const second = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'second choice' })!
    const { hook, onCancel, onSubmit } = renderQueueHook({ busy: true })

    act(() => {
      void hook.result.current.sendQueuedNow(first.id)
      void hook.result.current.sendQueuedNow(second.id)
    })
    expect(onCancel).toHaveBeenCalledTimes(1)

    hook.rerender({ busy: false })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('second choice')
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([first.id])
  })

  it('retries a selected send after an older drain unlocks without lifting the new park', async () => {
    let completeFirst!: (accepted: boolean) => void
    const firstResult = new Promise<boolean>(resolve => {
      completeFirst = resolve
    })
    let calls = 0
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })!
    const selected = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'selected' })!
    const trailing = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'trailing' })!
    parkQueuedPrompts(SESSION_KEY)

    const { hook, onSubmit } = renderQueueHook({
      onSubmit: () => {
        calls += 1
        return calls === 1 ? firstResult : Promise.resolve(true)
      }
    })

    await act(async () => {
      void hook.result.current.drainNextQueued()
      await Promise.resolve()
    })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))

    act(() => {
      void hook.result.current.sendQueuedNow(selected.id)
    })

    await act(async () => {
      completeFirst(true)
      await firstResult
    })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(2))

    expect(onSubmit.mock.calls.map(call => call[0])).toEqual(['first', 'selected'])
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([trailing.id])
    expect(getQueuedPrompts(SESSION_KEY).some(entry => entry.id === first.id)).toBe(false)
    expect(isQueueParked(SESSION_KEY)).toBe(true)
  })

  it('does not cancel a running turn for a stale queued id', async () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'real item' })
    const { hook, onCancel } = renderQueueHook({ busy: true })

    await act(async () => {
      expect(await hook.result.current.sendQueuedNow('removed-elsewhere')).toBe(false)
    })

    expect(onCancel).not.toHaveBeenCalled()
    expect(isQueueParked(SESSION_KEY)).toBe(false)
  })

  it('keeps an in-flight Send now after a hidden fallback enqueues', async () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })!
    const selected = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'selected' })!
    const { hook, onSubmit } = renderQueueHook({ busy: true })

    act(() => {
      void hook.result.current.sendQueuedNow(selected.id)
      enqueueQueuedPrompt(
        SESSION_KEY,
        { attachments: [], text: 'setup note', displayKind: 'hidden' },
        { resumeQueue: false }
      )
    })

    hook.rerender({ busy: false })
    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('selected')
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toContain(first.id)
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['first', 'setup note'])
    expect(isQueueParked(SESSION_KEY)).toBe(true)
  })

  it('keeps an interrupted manual send parked when cancellation rejects', async () => {
    const target = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'retained' })!
    const { hook, onSubmit } = renderQueueHook({
      busy: true,
      onCancel: () => Promise.reject(new Error('gateway unavailable'))
    })

    act(() => {
      void hook.result.current.sendQueuedNow(target.id)
    })
    await act(async () => {
      await Promise.resolve()
    })
    hook.rerender({ busy: false })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([target.id])
    expect(isQueueParked(SESSION_KEY)).toBe(true)
  })

  it('parks an initially unparked idle queue for an isolated send', async () => {
    setSessionsLoading(true)
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })!
    const selected = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'selected' })!
    const { hook, onSubmit } = renderQueueHook()

    act(() => {
      void hook.result.current.sendQueuedNow(selected.id)
    })
    expect(isQueueParked(SESSION_KEY)).toBe(true)
    setSessionsLoading(false)
    hook.rerender({ busy: false })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('selected')
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([first.id])
    expect(isQueueParked(SESSION_KEY)).toBe(true)
  })

  it('sends only the selected entry while idle without resuming a parked queue', async () => {
    const first = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first' })!
    const selected = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'selected' })!
    parkQueuedPrompts(SESSION_KEY)
    const { hook, onSubmit } = renderQueueHook()

    await act(async () => {
      await hook.result.current.sendQueuedNow(selected.id)
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('selected', expect.any(Object))
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([first.id])
    expect(isQueueParked(SESSION_KEY)).toBe(true)
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

  it('a delivered queued steer preserves a pre-existing park', async () => {
    const steerable = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'redirect' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'follows after' })
    parkQueuedPrompts(SESSION_KEY)

    const onSteer = vi.fn(async () => true)
    const { hook } = renderQueueHook({ busy: true, onSteer })

    await act(async () => {
      expect(await hook.result.current.steerQueuedNow(steerable!.id)).toBe(true)
    })

    expect(isQueueParked(SESSION_KEY)).toBe(true)
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

  it('freezes @terminal selection at enqueue so a later label collision cannot inject unrelated output', async () => {
    setComposerTerminalSelection('zsh:23-58', 'selection A')

    const { draftRef, hook, onSubmit } = renderQueueHook({
      busy: true,
      draft: 'look at @terminal:`zsh:23-58`'
    })

    act(() => {
      expect(hook.result.current.queueCurrentDraft()).toBe(true)
    })

    const queued = getQueuedPrompts(SESSION_KEY)

    expect(queued).toHaveLength(1)
    expect(queued[0]?.text).toBe('look at @terminal:`zsh:23-58`')
    expect(queued[0]?.displayText).toBe('look at @terminal:`zsh:23-58`')
    expect(queued[0]?.text).not.toContain('selection A')
    expect(getFrozenQueuedTransport(queued[0]!.id)).toContain('selection A')
    expect(draftRef.current).toBe('')

    setComposerTerminalSelection('zsh:23-58', 'selection B from another tab')

    await act(async () => {
      await hook.result.current.drainNextQueued()
    })

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toContain('selection A')
    expect(onSubmit.mock.calls[0]?.[0]).not.toContain('selection B')
    expect(onSubmit.mock.calls[0]?.[1]).toMatchObject({
      displayText: 'look at @terminal:`zsh:23-58`',
      fromQueue: true
    })
  })

  it('blocks drain after a simulated reload when the runtime terminal payload is gone', async () => {
    setComposerTerminalSelection('zsh:23-58', 'selection A')

    const { hook, onSubmit } = renderQueueHook({
      busy: true,
      draft: 'look at @terminal:`zsh:23-58`'
    })

    act(() => {
      expect(hook.result.current.queueCurrentDraft()).toBe(true)
    })

    simulateComposerQueueReloadForTests()

    await act(async () => {
      expect(await hook.result.current.drainNextQueued()).toBe(false)
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY)).toHaveLength(1)
    expect($notifications.get().some(n => n.message.includes('Re-select the lines'))).toBe(true)
  })
})
