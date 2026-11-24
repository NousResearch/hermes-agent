// @vitest-environment jsdom
// While a profile's backend is in a planned transition (store/backend-transition.ts) the composer's
// other ways of putting a message on a backend hold like Enter and Send do
// (../backend-switch-hold.test.tsx):
//  - a request another surface sends through the composer (the review pane's ship button, a reply's
//    Continue, a preview widget, a plugin's composer.submit) is not taken while the profile moves to
//    another backend: sent, steered or queued now, it would run on the backend about to go away or
//    cross the change in the queue;
//  - a queued message is not delivered early from the queue panel (send now, steer now, the second
//    Enter): it waits for the transition's end like the rest of the queue.
import { act, cleanup, renderHook } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import type { BackendTransitionKind } from '@/store/backend-transition'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  resetFrozenQueuedTransportsForTests
} from '@/store/composer-queue'
import { setSessionsLoading } from '@/store/session'

import type { QueueEditState } from '../composer-utils'
import { requestComposerSubmit } from '../focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '../scope'
import type { ChatBarProps } from '../types'

import { useComposerQueue } from './use-composer-queue'
import { useComposerSubmit } from './use-composer-submit'

const SESSION_KEY = 'stored-work'

beforeEach(() => {
  window.localStorage.clear()
  $queuedPromptsBySession.set({})
  $parkedQueueSessions.set({})
  resetFrozenQueuedTransportsForTests()
  setSessionsLoading(false)
})

afterEach(() => {
  cleanup()
  $queuedPromptsBySession.set({})
  $parkedQueueSessions.set({})
  resetFrozenQueuedTransportsForTests()
  setSessionsLoading(true)
})

interface Chat {
  backendTransition: BackendTransitionKind | null
  busy: boolean
}

function renderSubmit(initial: Chat) {
  const onSteer = vi.fn(async () => true)
  const onSteerHidden = vi.fn(async () => true)
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const editor = window.document.createElement('div')
  // Stable across renders, as in ChatBar: the request listener re-binds only when what it reads changes.
  const activeQueueSessionKeyRef = { current: SESSION_KEY }

  const Wrapper = ({ children }: PropsWithChildren) => (
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: 'main' }}>
      <ComposerSurfaceProvider value="switch-surface">
        <PaneVisibleContext.Provider value={true}>
          <div data-composer-surface-id="switch-surface" data-composer-target="main">
            {children}
          </div>
        </PaneVisibleContext.Provider>
      </ComposerSurfaceProvider>
    </ComposerScopeProvider>
  )

  const hook = renderHook(
    ({ backendTransition, busy }: Chat) =>
      useComposerSubmit({
        activeQueueSessionKey: SESSION_KEY,
        activeQueueSessionKeyRef,
        attachments: [],
        backendTransition,
        busy,
        clearDraft: vi.fn(),
        disabled: false,
        draftRef: { current: 'an unsent draft' },
        draftScopeRef: { current: SESSION_KEY },
        drainNextQueued: vi.fn(async () => false),
        editorRef: { current: editor },
        exitQueuedEdit: vi.fn(() => false),
        focusInput: vi.fn(),
        inputDisabled: false,
        loadIntoComposer: vi.fn(),
        onCancel: vi.fn(),
        onSteer,
        onSteerHidden,
        onSubmit,
        queueCurrentDraft: vi.fn(() => true),
        queueEdit: null,
        queuedPrompts: [],
        sessionId: 'rt-work',
        setComposerText: vi.fn(),
        stashAt: vi.fn()
      }),
    { initialProps: initial, wrapper: Wrapper }
  )

  return { hook, onSteer, onSteerHidden, onSubmit }
}

const request = (text: string, displayKind?: 'hidden') =>
  act(async () => {
    expect(requestComposerSubmit(text, { target: 'main', ...(displayKind ? { displayKind } : {}) })).toBe(true)
  })

describe("a request sent through the composer while the profile's backend changes", () => {
  it.each(['starting', 'stopping'] as const)(
    '%s: it is not sent, steered or queued, and the same request goes once the profile is on its new backend',
    async backendTransition => {
      const idle = renderSubmit({ backendTransition, busy: false })

      await request('Commit this and open a pull request.')
      await request('[widget] a button was pressed', 'hidden')

      expect(idle.onSubmit).not.toHaveBeenCalled()
      expect(getQueuedPrompts(SESSION_KEY)).toEqual([])

      idle.hook.rerender({ backendTransition: null, busy: false })
      await request('Commit this and open a pull request.')

      expect(idle.onSubmit).toHaveBeenCalledTimes(1)
      expect(idle.onSubmit.mock.calls[0]?.[0]).toBe('Commit this and open a pull request.')
      cleanup()

      // Behind a running reply the request would steer that reply, or wait in the queue across the change.
      const busy = renderSubmit({ backendTransition, busy: true })

      await request('continue')
      await request('[widget] a button was pressed', 'hidden')

      expect(busy.onSteer).not.toHaveBeenCalled()
      expect(busy.onSteerHidden).not.toHaveBeenCalled()
      expect(busy.onSubmit).not.toHaveBeenCalled()
      expect(getQueuedPrompts(SESSION_KEY)).toEqual([])
    }
  )

  it('restarting: it is taken as before (the same backend comes back)', async () => {
    const { onSubmit } = renderSubmit({ backendTransition: 'restarting', busy: false })

    await request('Commit this and open a pull request.')

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })
})

function renderQueue(backendSwitching: boolean) {
  const onCancel = vi.fn()
  const onSteer = vi.fn(async () => true)
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const queueEditRef: { current: QueueEditState | null } = { current: null }

  const hook = renderHook(
    ({ switching }: { switching: boolean }) =>
      useComposerQueue({
        activeQueueSessionKey: SESSION_KEY,
        attachments: [],
        backendSwitching: switching,
        busy: true,
        clearDraft: () => undefined,
        draftRef: { current: '' },
        focusInput: () => undefined,
        loadIntoComposer: () => undefined,
        onCancel,
        onSteer,
        onSubmit,
        queueEditRef,
        queueSessionKey: SESSION_KEY,
        sessionId: 'rt-work'
      }),
    { initialProps: { switching: backendSwitching } }
  )

  return { hook, onCancel, onSteer, onSubmit }
}

describe("a queued message behind a running reply while the profile's backend is in a transition", () => {
  it('is not steered into the reply, nor sent by interrupting it, from the queue panel or a second Enter', async () => {
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'first queued' })
    enqueueQueuedPrompt(SESSION_KEY, { attachments: [], text: 'second queued' })
    const [first, second] = getQueuedPrompts(SESSION_KEY)
    const { hook, onCancel, onSteer, onSubmit } = renderQueue(true)

    await act(async () => {
      expect(await hook.result.current.steerQueuedNow(second!.id)).toBe(false)
      expect(await hook.result.current.sendQueuedNow(second!.id)).toBe(false)
      expect(await hook.result.current.deliverQueuedNow(second!.id)).toBe(false)
    })

    expect(onSteer).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    // Still queued, in the order they were queued.
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([first!.id, second!.id])

    // The transition has ended: the panel's actions work again.
    hook.rerender({ switching: false })
    await act(async () => {
      expect(await hook.result.current.steerQueuedNow(second!.id)).toBe(true)
    })

    expect(onSteer).toHaveBeenCalledExactlyOnceWith('second queued')
    expect(getQueuedPrompts(SESSION_KEY).map(entry => entry.id)).toEqual([first!.id])
  })
})
