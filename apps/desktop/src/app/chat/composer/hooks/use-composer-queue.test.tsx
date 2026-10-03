import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $salvagedEditNoticesBySession,
  announceSalvagedEdit,
  clearComposerTerminalSelections,
  type ComposerAttachment,
  dismissSalvagedEdit,
  getSalvagedEditNotice,
  setComposerTerminalSelection,
  undoSalvagedEdit
} from '@/store/composer'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getFrozenQueuedTransport,
  getQueuedPrompts,
  isQueueParked,
  MAX_AUTO_DRAIN_ATTEMPTS,
  parkQueuedPrompts,
  removeQueuedPrompt,
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
// auto-drain effect honoring the park, and send-now-while-busy lifting it so
// the settle drain still flows (the regression that sank the old blanket
// interrupt latch).

const SESSION_KEY = 'stored-session-queue-hook'

function renderQueueHook(
  overrides: { busy?: boolean; draft?: string; onCancel?: () => void; onSteer?: ChatBarProps['onSteer'] } = {}
) {
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const onCancel = overrides.onCancel ?? vi.fn()
  const onSteer = overrides.onSteer
  const queueEditRef: { current: QueueEditState | null } = { current: null }
  const draftRef = { current: overrides.draft ?? '' }
  const draftRefHook: { current: string } = { current: '' }

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
        readLiveText: () => draftRefHook.current,
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
    $salvagedEditNoticesBySession.set({})
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
    $salvagedEditNoticesBySession.set({})
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

  it('keeps a dirty in-progress queued edit when the turn settles in the background (#88621)', async () => {
    // The reporter's scenario: the user opens a queued prompt for in-place
    // editing and types for ~a minute; a background turn unwinds ("Operation
    // interrupted" / timeout / settle) and the queue starts flowing again.
    // The drain must not send the stale pre-edit text out from under the user,
    // and the edit must not be cancelled back to the pre-edit draft — either
    // way the typed minute is lost.
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'original words' })!

    const loadIntoComposer = vi.fn()
    const onSubmit = vi.fn(async () => true)
    const queueEditRef: { current: QueueEditState | null } = { current: null }
    const draftRef = { current: '' }

    const hook = renderHook(
      ({ busy }: { busy: boolean }) =>
        useComposerQueue({
          activeQueueSessionKey: SESSION_KEY,
          attachments: [],
          busy,
          clearDraft: () => undefined,
          draftRef,
          focusInput: () => undefined,
          loadIntoComposer,
          onCancel: vi.fn(),
          onSteer: undefined,
          onSubmit,
          queueEditRef,
          queueSessionKey: SESSION_KEY,
          readLiveText: () => draftRef.current,
          sessionId: 'rt-session-queue-hook'
        }),
      { initialProps: { busy: true } }
    )

    // Begin the in-place edit: the composer paints the entry text and records
    // the pre-edit draft snapshot.
    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })
    expect(loadIntoComposer).toHaveBeenCalledWith('original words', [])

    // The user types their replacement text for a minute. It lives in the
    // editor/draftRef — NOT in the queue entry (the edit is unsaved).
    draftRef.current = 'replacement text typed over a minute'

    // The background turn unwinds and settles: busy flips false with a dirty
    // edit buffer in the composer.
    hook.rerender({ busy: false })

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    // The dirty edit buffer must survive the settle: the drain held off the
    // entry under active edit (no submit of either text), the entry stays
    // queued, and the edit is not torn down.
    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts(SESSION_KEY).map(item => item.text)).toEqual(['original words'])
    expect(queueEditRef.current?.entryId).toBe(entry.id)
  })

  it('keeps the dirty edit buffer when the edited entry is drained out from under it (#88621)', async () => {
    // The reporter's loss: mid-edit on a queued prompt, a background or
    // cross-window drain removes the entry (the background drainer never
    // skips the id another surface is editing). The queue-edit cleanup then
    // tore the edit down and repainted the PRE-EDIT snapshot over the dirty
    // buffer, permanently destroying the typed minute. The teardown must
    // leave the composer untouched — no repaint, no focus call — and keep
    // the typed text recoverable through the salvage notice.
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'original words' })!

    const loadIntoComposer = vi.fn()
    const focusInput = vi.fn()
    const onSubmit = vi.fn(async () => true)
    const queueEditRef: { current: QueueEditState | null } = { current: null }
    const draftRef = { current: '' }

    const hook = renderHook(
      ({ busy }: { busy: boolean }) =>
        useComposerQueue({
          activeQueueSessionKey: SESSION_KEY,
          attachments: [],
          busy,
          clearDraft: () => undefined,
          draftRef,
          focusInput,
          loadIntoComposer,
          onCancel: vi.fn(),
          onSteer: undefined,
          onSubmit,
          queueEditRef,
          queueSessionKey: SESSION_KEY,
          readLiveText: () => draftRef.current,
          sessionId: 'rt-session-queue-hook'
        }),
      { initialProps: { busy: true } }
    )

    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })
    loadIntoComposer.mockClear()
    focusInput.mockClear()

    // The user's minute of typing: unsaved, lives only in the edit buffer.
    draftRef.current = 'replacement text typed over a minute'

    // The background event: the entry is drained/removed elsewhere.
    removeQueuedPrompt(SESSION_KEY, entry.id)

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    // The composer is untouched: the dirty buffer is still the draft, no
    // pre-edit repaint, no focus steal.
    expect(draftRef.current).toBe('replacement text typed over a minute')
    expect(loadIntoComposer).not.toHaveBeenCalled()
    expect(focusInput).not.toHaveBeenCalled()

    // The edit exited (its entry is gone) but the typed text is recoverable:
    // it stays in the editor AND the notice names it for an explicit put-back.
    expect(queueEditRef.current).toBeNull()
    const notice = getSalvagedEditNotice(SESSION_KEY)
    expect(notice?.currentText).toBe('replacement text typed over a minute')
    expect(notice?.undoText).toBe('replacement text typed over a minute')
  })

  it('offers the cancel of a dirty queued edit for undo instead of discarding it (#88621)', () => {
    // Esc / panel-delete on a dirty edit still restores the prior draft by
    // design — but the typed replacement must be recoverable, not destroyed.
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'original words' })!

    const loadIntoComposer = vi.fn()
    const queueEditRef: { current: QueueEditState | null } = { current: null }
    const draftRef = { current: '' }

    const hook = renderHook(
      () =>
        useComposerQueue({
          activeQueueSessionKey: SESSION_KEY,
          attachments: [],
          busy: false,
          clearDraft: () => undefined,
          draftRef,
          focusInput: () => undefined,
          loadIntoComposer,
          onCancel: vi.fn(),
          onSteer: undefined,
          onSubmit: vi.fn(async () => true),
          queueEditRef,
          queueSessionKey: SESSION_KEY,
          readLiveText: () => draftRef.current,
          sessionId: 'rt-session-queue-hook'
        })
    )

    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })
    draftRef.current = 'replacement text typed over a minute'

    act(() => {
      hook.result.current.exitQueuedEdit('cancel')
    })

    // Designed cancel semantics: the pre-edit draft is repainted…
    expect(loadIntoComposer).toHaveBeenLastCalledWith('', [])
    // …and the dirty buffer is kept recoverable behind the salvage notice.
    const notice = getSalvagedEditNotice(SESSION_KEY)
    expect(notice?.currentText).toBe('')
    expect(notice?.undoText).toBe('replacement text typed over a minute')

    // Undo while the composer still shows the repainted draft puts the typed
    // text back; after the user typed something new it only dismisses.
    expect(undoSalvagedEdit(SESSION_KEY, '')).toBe('replacement text typed over a minute')
    expect(getSalvagedEditNotice(SESSION_KEY)).toBeNull()
  })

  it('re-bases the dirty check on the entry reached by ArrowUp/ArrowDown (#88621)', () => {
    // Stepping to a neighbour repaints that entry's text; a buffer the user
    // has NOT touched since must read as clean against the new entry, or a
    // plain Esc publishes a spurious salvage notice for text nobody typed.
    const older = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'older words' })!
    const newer = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'newer words' })!

    const queueEditRef: { current: QueueEditState | null } = { current: null }
    const draftRef = { current: '' }

    const hook = renderHook(() =>
      useComposerQueue({
        activeQueueSessionKey: SESSION_KEY,
        attachments: [],
        busy: false,
        clearDraft: () => undefined,
        draftRef,
        focusInput: () => undefined,
        loadIntoComposer: (text: string) => {
          draftRef.current = text
        },
        onCancel: vi.fn(),
        onSteer: undefined,
        onSubmit: vi.fn(async () => true),
        queueEditRef,
        queueSessionKey: SESSION_KEY,
        readLiveText: () => draftRef.current,
        sessionId: 'rt-session-queue-hook'
      })
    )

    act(() => {
      hook.result.current.beginQueuedEdit(newer)
    })
    act(() => {
      hook.result.current.stepQueuedEdit(-1)
    })
    expect(queueEditRef.current?.entryId).toBe(older.id)
    expect(draftRef.current).toBe('older words')

    act(() => {
      hook.result.current.exitQueuedEdit('cancel')
    })

    expect(getSalvagedEditNotice(SESSION_KEY)).toBeNull()
  })

  it('reads the live editor before the vanished-entry dirty check (#88621)', async () => {
    // Input flushes the DOM into draftRef on a rAF; if the entry disappears
    // before that frame runs, draftRef still holds the pre-edit text while
    // the DOM shows the first typed burst. The teardown must read the LIVE
    // editor for the dirty decision and the salvage — reading draftRef alone
    // repaints the typed work away as "clean".
    const entry = enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'original words' })!

    const loadIntoComposer = vi.fn()
    const queueEditRef: { current: QueueEditState | null } = { current: null }
    const draftRef = { current: '' }
    // The live editor DOM is a frame ahead of draftRef: the user typed, the
    // rAF flush has not run yet.
    const liveEditorText = { current: 'original words plus the latest input' }

    const hook = renderHook(() =>
      useComposerQueue({
        activeQueueSessionKey: SESSION_KEY,
        attachments: [],
        busy: false,
        clearDraft: () => undefined,
        draftRef,
        focusInput: () => undefined,
        loadIntoComposer,
        onCancel: vi.fn(),
        onSteer: undefined,
        onSubmit: vi.fn(async () => true),
        queueEditRef,
        queueSessionKey: SESSION_KEY,
        readLiveText: () => liveEditorText.current,
        sessionId: 'rt-session-queue-hook'
      })
    )

    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })
    loadIntoComposer.mockClear()

    // Input landed in the DOM but not yet in draftRef; the entry vanishes
    // before the flush frame.
    removeQueuedPrompt(SESSION_KEY, entry.id)

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    // The dirty buffer (per the live read) survives: no pre-edit repaint…
    expect(loadIntoComposer).not.toHaveBeenCalled()
    expect(queueEditRef.current).toBeNull()
    // …and the notice carries the typed burst, not the pre-edit text.
    const notice = getSalvagedEditNotice(SESSION_KEY)
    expect(notice?.undoText).toBe('original words plus the latest input')
    expect(notice?.currentText).toBe('original words plus the latest input')
  })

  it('keeps each session\'s pending recovery independent across sessions (#88621)', () => {
    // Store-level contract: a cancel in session B must never erase session
    // A's pending recovery, and undo/dismiss consume only their own record.
    // (Two sessions losing an edit independently is the documented lifetime
    // of the session-scoped notice.)
    announceSalvagedEdit('session-a', '', 'unsaved replacement A')
    announceSalvagedEdit('session-b', '', 'unsaved replacement B')

    // Both records coexist.
    expect(getSalvagedEditNotice('session-a')?.undoText).toBe('unsaved replacement A')
    expect(getSalvagedEditNotice('session-b')?.undoText).toBe('unsaved replacement B')

    // Undo consumes only the matching session's record.
    expect(undoSalvagedEdit('session-a', '')).toBe('unsaved replacement A')
    expect(getSalvagedEditNotice('session-a')).toBeNull()
    expect(getSalvagedEditNotice('session-b')?.undoText).toBe('unsaved replacement B')

    // Dismissing B leaves nothing; A was already consumed.
    dismissSalvagedEdit('session-b')
    expect(getSalvagedEditNotice('session-b')).toBeNull()
    expect($salvagedEditNoticesBySession.get()).toEqual({})
  })
})


// ---------------------------------------------------------------------------
// #88621 review regressions — live-input reads at every queued-edit boundary
// (R3) and the payload-aware dirty decision (R2).
// ---------------------------------------------------------------------------

describe('useComposerQueue live-input reads (#88621 review R3/R2)', () => {
  const S = 'live-read-session'

  beforeEach(() => {
    window.localStorage.clear()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    $salvagedEditNoticesBySession.set({})
    setSessionsLoading(false)
  })

  afterEach(() => {
    cleanup()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    $salvagedEditNoticesBySession.set({})
    setSessionsLoading(true)
  })

  function renderLiveHook(overrides: {
    attachments?: ComposerAttachment[]
    liveText: { current: string }
    loadIntoComposer?: (text: string, attachments: ComposerAttachment[]) => void
  }) {
    const queueEditRef: { current: QueueEditState | null } = { current: null }
    const draftRef = { current: '' }

    const loadIntoComposer =
      overrides.loadIntoComposer ??
      ((text: string) => {
        overrides.liveText.current = text
        draftRef.current = text
      })

    const hook = renderHook(() =>
      useComposerQueue({
        activeQueueSessionKey: S,
        attachments: overrides.attachments ?? [],
        busy: false,
        clearDraft: () => undefined,
        draftRef,
        focusInput: () => undefined,
        loadIntoComposer,
        onCancel: vi.fn(),
        onSteer: undefined,
        onSubmit: vi.fn(async () => true),
        queueEditRef,
        queueSessionKey: S,
        readLiveText: () => overrides.liveText.current,
        sessionId: 'rt-live-read'
      })
    )

    return { hook, loadIntoComposer, queueEditRef }
  }

  it('R3: beginQueuedEdit snapshots the LIVE text as the pre-edit draft (the real assertion)', () => {
    const entry = enqueueQueuedPrompt(S, { attachments: [], text: 'queued words' })!
    const liveText = { current: 'final burst before the edit click' }

    const { hook, queueEditRef } = renderLiveHook({ liveText })

    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })

    // The dirty buffer repainted the entry text; the pre-edit snapshot must
    // hold the live final burst, not the stale '' the mirror still had.
    expect(queueEditRef.current?.draft).toBe('final burst before the edit click')
  })

  it('R3: Save commits the LIVE text, keeping a pending-frame burst', () => {
    const entry = enqueueQueuedPrompt(S, { attachments: [], text: 'original words' })!
    const liveText = { current: '' }

    const { hook } = renderLiveHook({ liveText })

    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })
    // loadIntoComposer seeded the live text with the entry text; the user
    // typed a replacement that the flush frame has not carried to draftRef.
    liveText.current = 'replacement queued words with final input'

    act(() => {
      hook.result.current.exitQueuedEdit('save')
    })

    expect(getQueuedPrompts(S).map(e => e.text)).toEqual(['replacement queued words with final input'])
  })

  it('R3: stepping with ArrowDown commits the LIVE text to the entry it leaves', () => {
    const older = enqueueQueuedPrompt(S, { attachments: [], text: 'older' })!
    const newer = enqueueQueuedPrompt(S, { attachments: [], text: 'newer' })!
    const liveText = { current: '' }

    const { hook } = renderLiveHook({ liveText })

    act(() => {
      hook.result.current.beginQueuedEdit(older)
    })
    liveText.current = 'typed over older, frame pending'

    act(() => {
      hook.result.current.stepQueuedEdit(1)
    })

    expect(getQueuedPrompts(S).map(e => e.text)).toEqual([
      'typed over older, frame pending',
      'newer'
    ])
  })

  it('R2: a chip added with unchanged text is dirty — Cancel publishes the salvage notice', () => {
    const entry = enqueueQueuedPrompt(S, { attachments: [], text: 'original words' })!
    const addedChip: ComposerAttachment = { id: 'file:new', kind: 'file', label: 'new.txt' }
    const liveText = { current: '' }

    const { hook } = renderLiveHook({ attachments: [addedChip], liveText })

    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })
    // loadIntoComposer seeded liveText with the entry text; text unchanged,
    // but the live attachment set gained a chip.

    act(() => {
      hook.result.current.exitQueuedEdit('cancel')
    })

    const notice = getSalvagedEditNotice(S)

    expect(notice?.undoText).toBe('original words')
    expect(notice?.undoAttachments).toEqual([addedChip])
  })

  it('R2: an untouched edit (text AND chips) stays clean — no spurious notice', () => {
    const chip: ComposerAttachment = { id: 'file:queued', kind: 'file', label: 'queued.txt' }
    const entry = enqueueQueuedPrompt(S, { attachments: [chip], text: 'original words' })!
    const liveText = { current: '' }

    const { hook } = renderLiveHook({ attachments: [chip], liveText })

    act(() => {
      hook.result.current.beginQueuedEdit(entry)
    })
    act(() => {
      hook.result.current.exitQueuedEdit('cancel')
    })

    expect(getSalvagedEditNotice(S)).toBeNull()
  })
})
