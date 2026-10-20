import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
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
import { reconcilePendingSubmissions } from '@/store/pending-submissions'
import { $connection, setSessionsLoading } from '@/store/session'

import type { QueueEditState } from '../composer-utils'
import type { ChatBarProps } from '../types'

import { useComposerQueue } from './use-composer-queue'

// The park ↔ drain contract at the hook level. The store tests pin the pure
// pieces (shouldAutoDrain, park bookkeeping); these pin the wiring — the
// auto-drain effect honoring the park, and per-entry Send now preserving the
// rest of the queue instead of silently resuming it.

const SESSION_KEY = 'stored-session-queue-hook'

/** A reply the test settles by hand. */
function deferred() {
  let resolve!: (value: unknown) => void
  let reject!: (reason: unknown) => void

  const promise = new Promise<unknown>((onResolve, onReject) => {
    resolve = onResolve
    reject = onReject
  })

  return { promise, reject, resolve }
}

function renderQueueHook(
  overrides: {
    busy?: boolean
    draft?: string
    onCancel?: ChatBarProps['onCancel']
    onSteer?: ChatBarProps['onSteer']
    onSubmit?: ChatBarProps['onSubmit']
    replaceServerQueued?: (admissionId: string, text: string, submissionId: string) => Promise<unknown>
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
        replaceServerQueued: overrides.replaceServerQueued,
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

  it('admits native Queue through submit and retains an uncertain draft without a local replay', async () => {
    $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as HermesConnection)
    const draftRef = { current: 'durable queue' }
    const clearDraft = vi.fn(() => { draftRef.current = '' })
    const onSubmit = vi.fn<ChatBarProps['onSubmit']>().mockResolvedValue(false)

    const hook = renderHook(({ busy }) => useComposerQueue({
      activeQueueSessionKey: SESSION_KEY, attachments: [], busy, clearDraft, draftRef,
      focusInput: () => undefined, loadIntoComposer: () => undefined, onCancel: vi.fn(), onSteer: undefined,
      onSubmit, queueEditRef: { current: null }, queueSessionKey: SESSION_KEY, sessionId: 'rt-session-queue-hook'
    }), { initialProps: { busy: true } })

    try {
      await act(async () => { await hook.result.current.queueCurrentDraft() })
      expect(onSubmit).toHaveBeenCalledWith('durable queue', expect.objectContaining({ fromQueue: true, storedSessionId: SESSION_KEY }))
      expect(draftRef.current).toBe('durable queue')
      expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
      onSubmit.mockResolvedValue(true)
      await act(async () => { await hook.result.current.queueCurrentDraft() })
      expect(clearDraft).toHaveBeenCalledTimes(1)
      hook.rerender({ busy: false })
      await act(async () => { await Promise.resolve() })
      expect(onSubmit).toHaveBeenCalledTimes(2)
    } finally { $connection.set(null) }
  })

  it('does not clear the next session draft when a queue ACK arrives after navigation', async () => {
    $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as HermesConnection)
    let accept!: (value: boolean) => void
    const clearDraft = vi.fn()
    const draftRef = { current: 'same text' }

    const hook = renderHook(({ key }) => useComposerQueue({
      activeQueueSessionKey: key, attachments: [], busy: true, clearDraft, draftRef,
      focusInput: () => undefined, loadIntoComposer: () => undefined, onCancel: vi.fn(), onSteer: undefined,
      onSubmit: () => new Promise<boolean>(resolve => { accept = resolve }), queueEditRef: { current: null }, queueSessionKey: key, sessionId: key
    }), { initialProps: { key: 'outgoing' } })

    try {
      let pending: boolean | Promise<boolean> = false
      act(() => { pending = hook.result.current.queueCurrentDraft() })
      hook.rerender({ key: 'incoming' })
      await act(async () => { accept(true); await pending })
      expect(clearDraft).not.toHaveBeenCalled()
    } finally { $connection.set(null) }
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

      for (const [, options] of onSubmit.mock.calls) {
        expect(options).toMatchObject({ submission_id: entry.id })
      }

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

describe('useComposerQueue scope', () => {
  afterEach(() => {
    cleanup()
    window.localStorage.clear()
    $queuedPromptsBySession.set({})
  })

  it('leaving a session for a new chat does not move its queue there (#139190)', () => {
    enqueueQueuedPrompt('session-a', { attachments: [], text: 'for A' })

    type Scope = { activeQueueSessionKey: string; queueSessionKey: null | string; sessionId: null | string }

    const hook = renderHook(
      (scope: Scope) =>
        useComposerQueue({
          ...scope,
          attachments: [],
          busy: true,
          clearDraft: () => undefined,
          draftRef: { current: '' },
          focusInput: () => undefined,
          loadIntoComposer: () => undefined,
          onCancel: vi.fn(),
          onSteer: undefined,
          onSubmit: vi.fn(async () => true),
          queueEditRef: { current: null }
        }),
      { initialProps: { activeQueueSessionKey: 'session-a', queueSessionKey: 'session-a', sessionId: 'session-a' } as Scope }
    )

    expect(hook.result.current.queuedPrompts.map(entry => entry.text)).toEqual(['for A'])
    hook.rerender({ activeQueueSessionKey: '__new__:fresh', queueSessionKey: null, sessionId: null })

    expect(hook.result.current.queuedPrompts).toEqual([])
    expect(getQueuedPrompts('session-a').map(entry => entry.text)).toEqual(['for A'])
  })

  // RT-35: a server-queued row is an admission in the owner's FIFO. Saving its edit asks the owner
  // to replace it; the owner's pending fanout then repaints the card (no client-side edit).
  it('saving an edit of a server-queued row replaces its admission at the owner', async () => {
    reconcilePendingSubmissions(SESSION_KEY, [
      { admission_id: 'adm-1', status: 'queued', user: 'old words', controls: ['steer', 'replace'] }
    ])
    const replaceServerQueued = vi.fn(async () => ({}))
    const { draftRef, hook } = renderQueueHook({ busy: true, draft: 'my draft', replaceServerQueued })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    expect(draftRef.current).toBe('old words')
    draftRef.current = 'new words'
    act(() => {
      expect(hook.result.current.exitQueuedEdit('save')).toBe(true)
    })

    expect(replaceServerQueued).toHaveBeenCalledWith('adm-1', 'new words', expect.any(String))
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.text)).toEqual(['old words'])
    // The pre-edit draft comes back once the owner accepted the replacement.
    await waitFor(() => expect(draftRef.current).toBe('my draft'))
    expect(hook.result.current.queueEdit).toBeNull()
  })

  it('a refused replacement keeps the edited text as an unsent draft and says why', async () => {
    reconcilePendingSubmissions(SESSION_KEY, [
      { admission_id: 'adm-1', status: 'queued', user: 'old words', controls: ['steer', 'replace'] }
    ])
    clearNotifications()

    const replaceServerQueued = vi.fn(async () => {
      throw Object.assign(new Error('admission_started'), { data: { reason: 'admission_started' } })
    })

    const { draftRef, hook } = renderQueueHook({ busy: true, draft: 'my draft', replaceServerQueued })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    draftRef.current = 'new words'
    act(() => {
      expect(hook.result.current.exitQueuedEdit('save')).toBe(true)
    })

    await waitFor(() => expect(hook.result.current.queueEdit).toBeNull())
    // Nothing typed is lost: the edit, then the draft it interrupted.
    expect(draftRef.current).toBe('new words\n\nmy draft')
    expect($notifications.get().some(n => n.kind === 'warning')).toBe(true)
  })

  // Review: the composer stays editable while the owner answers a Save. That answer settles only the
  // save it belongs to and never overwrites text typed since.
  it.each([
    ['accepted', true],
    ['refused', false]
  ])('a %s Save reply keeps the text typed after Save', async (_label, accepted) => {
    reconcilePendingSubmissions(SESSION_KEY, [
      { admission_id: 'adm-1', status: 'queued', user: 'old words', controls: ['steer', 'replace'] }
    ])
    const reply = deferred()
    const { draftRef, hook } = renderQueueHook({ busy: true, draft: 'my draft', replaceServerQueued: () => reply.promise })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    draftRef.current = 'new words'
    act(() => {
      hook.result.current.exitQueuedEdit('save')
    })
    draftRef.current = 'typed after save'
    await act(async () => (accepted ? reply.resolve({}) : reply.reject({ data: { reason: 'admission_started' } })))

    expect(hook.result.current.queueEdit).toBeNull()
    expect(draftRef.current).toBe('typed after save\n\nmy draft')
  })

  // Review: a reply lost in transport leaves the outcome unknown (the owner may have committed the
  // replacement). The edit stays open under the same identity, so Save retries prompt.replace_queued
  // with it (the owner replays a committed one) instead of turning the text into a fresh send.
  it('an unconfirmed Save keeps the edit open and retries under the same identity', async () => {
    reconcilePendingSubmissions(SESSION_KEY, [
      { admission_id: 'adm-1', status: 'queued', user: 'old words', controls: ['steer', 'replace'] }
    ])
    clearNotifications()
    const replies = [deferred(), deferred(), deferred()]
    const replaceServerQueued = vi.fn((_admissionId: string, _text: string, _submissionId: string) =>
      replies[replaceServerQueued.mock.calls.length - 1]!.promise)
    const { draftRef, hook } = renderQueueHook({ busy: true, draft: 'my draft', replaceServerQueued })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    draftRef.current = 'new words'
    act(() => {
      hook.result.current.exitQueuedEdit('save')
    })
    await act(async () => replies[0]!.reject(new Error('Hermes gateway is not connected')))

    expect(hook.result.current.queueEdit?.entryId).toBe('adm-1')
    expect(draftRef.current).toBe('new words')
    expect($notifications.get().some(n => n.kind === 'warning')).toBe(true)

    // The replacement committed: the owner's snapshot lists it, and the old row is gone. The edit
    // stays open (no plain draft a later Enter would send again).
    act(() =>
      reconcilePendingSubmissions(SESSION_KEY, [{ admission_id: 'adm-2', status: 'queued', user: 'new words' }])
    )
    expect(hook.result.current.queueEdit?.entryId).toBe('adm-1')

    act(() => {
      expect(hook.result.current.exitQueuedEdit('save')).toBe(true)
    })
    const [first, second] = replaceServerQueued.mock.calls
    expect(second).toEqual(['adm-1', 'new words', first![2]])
    await act(async () => replies[1]!.resolve({ admission_id: 'adm-2' }))

    expect(hook.result.current.queueEdit).toBeNull()
    expect(draftRef.current).toBe('my draft')
  })

  it('a changed edit after an unconfirmed Save is a new identity', async () => {
    reconcilePendingSubmissions(SESSION_KEY, [
      { admission_id: 'adm-1', status: 'queued', user: 'old words', controls: ['steer', 'replace'] }
    ])
    const replaceServerQueued = vi.fn(async (_admissionId: string, _text: string, _submissionId: string) => {
      throw new Error('socket closed')
    })
    const { draftRef, hook } = renderQueueHook({ busy: true, replaceServerQueued })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    draftRef.current = 'new words'
    await act(async () => {
      hook.result.current.exitQueuedEdit('save')
    })
    draftRef.current = 'other words'
    await act(async () => {
      hook.result.current.exitQueuedEdit('save')
    })

    expect(replaceServerQueued).toHaveBeenCalledTimes(2)
    const [first, second] = replaceServerQueued.mock.calls
    expect(second![1]).toBe('other words')
    expect(second![2]).not.toBe(first![2])
  })

  it('a Save reply for an edit that was cancelled and reopened leaves the reopened edit alone', async () => {
    reconcilePendingSubmissions(SESSION_KEY, [
      { admission_id: 'adm-1', status: 'queued', user: 'old words', controls: ['steer', 'replace'] }
    ])
    const reply = deferred()
    const { draftRef, hook } = renderQueueHook({ busy: true, draft: 'my draft', replaceServerQueued: () => reply.promise })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    draftRef.current = 'new words'
    act(() => {
      hook.result.current.exitQueuedEdit('save')
    })
    act(() => {
      hook.result.current.exitQueuedEdit('cancel')
    })
    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    draftRef.current = 'reopened edit'
    await act(async () => reply.resolve({}))

    expect(hook.result.current.queueEdit?.entryId).toBe('adm-1')
    expect(draftRef.current).toBe('reopened edit')
  })

  it('a started-row snapshot arriving mid-edit keeps the edit as a draft', () => {
    reconcilePendingSubmissions(SESSION_KEY, [
      { admission_id: 'adm-1', status: 'queued', user: 'old words', controls: ['steer', 'replace'] }
    ])
    clearNotifications()
    const replaceServerQueued = vi.fn(async () => ({}))
    const { draftRef, hook } = renderQueueHook({ busy: true, replaceServerQueued })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))
    draftRef.current = 'half-typed edit'
    // The owner claimed the row: the next pending snapshot carries it as started.
    act(() =>
      reconcilePendingSubmissions(SESSION_KEY, [{ admission_id: 'adm-1', status: 'started', user: 'old words' }])
    )

    expect(hook.result.current.queueEdit).toBeNull()
    expect(draftRef.current).toBe('half-typed edit')
    expect(replaceServerQueued).not.toHaveBeenCalled()
    expect($notifications.get().some(n => n.kind === 'warning')).toBe(true)
  })

  it('a server-queued row whose owner serves no replace stays read-only', () => {
    reconcilePendingSubmissions(SESSION_KEY, [{ admission_id: 'adm-1', status: 'queued', user: 'old words' }])
    const { hook } = renderQueueHook({ busy: true, replaceServerQueued: vi.fn(async () => ({})) })

    act(() => hook.result.current.beginQueuedEdit(getQueuedPrompts(SESSION_KEY)[0]!))

    expect(hook.result.current.queueEdit).toBeNull()
  })
})
