import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import type { ClientSessionState } from '@/app/types'
import { type ComposerAttachment, mainComposerScope, stashSessionDraft } from '@/store/composer'
import { clearQueuedPrompts, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
import { $notifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $connection } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { requestComposerSubmit } from '../focus'
import { ComposerSurfaceProvider } from '../scope'

import receipts from './fixtures/synthetic-correction-receipts.json'
import { useComposerSubmit } from './use-composer-submit'

// Adapted from yuzilongleif-collab <yuzilong.leif@gmail.com>:
// https://github.com/NousResearch/hermes-agent/issues/141973#issuecomment-6482735120
// Synthetic transport fixtures, not gateway captures or model-consumption proof.
// Both consuming production hooks run; only the RPC boundary is controlled.
function rpcReply(name: keyof typeof receipts) {
  const frame = receipts[name]

  if (Array.isArray(frame)) {
    throw new Error('not a single reply')
  }

  if ('error' in frame) {
    throw new JsonRpcGatewayError(frame.error.message, frame.error)
  }

  return frame.result
}

function harness(name: keyof typeof receipts | 'legacy' | 'missing', delayed = false) {
  const selected = { current: 'stored-A' as string | null }
  const runtime = { current: 'rt-A' as string | null }
  const draftRef = { current: 'neutral follow-up' }
  const draftScopeRef = { current: 'stored-A' as string | null }
  const queueRef = { current: 'stored-A' as string | null }
  const editor = window.document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.textContent = draftRef.current
  const editorRef = { current: editor }
  let state = { messages: [], busy: true, awaitingResponse: true, streamId: null } as unknown as ClientSessionState
  let release!: () => void

  const gate = delayed
    ? new Promise<void>(resolve => {
        release = resolve
      })
    : Promise.resolve()

  const request = vi.fn(async () => {
    await gate

    return (name === 'legacy' ? { status: 'queued' } : name === 'missing' ? {} : rpcReply(name)) as never
  })

  const clearDraft = vi.fn(() => {
    draftRef.current = ''
    editor.textContent = ''
  })

  const load = vi.fn((text: string) => {
    draftRef.current = text
    editor.textContent = text
  })

  const submit = vi.fn(async () => true)

  const hook = renderHook(
    () => {
      const actions = usePromptActions({
        activeSessionId: runtime.current,
        activeSessionIdRef: runtime,
        branchCurrentSession: async () => true,
        busyRef: { current: true },
        createBackendSessionForSend: async () => 'rt-A',
        getRoutedStoredSessionId: () => selected.current,
        getRuntimeIdForStoredSession: () => runtime.current,
        getRouteToken: () => selected.current ?? '',
        handleSkinCommand: () => '',
        openMemoryGraph: () => {},
        refreshSessions: async () => {},
        requestGateway: request,
        resumeStoredSession: () => {},
        runtimeIdByStoredSessionIdRef: {
          current: new Map([
            ['stored-A', 'rt-A'],
            ['stored-B', 'rt-B']
          ])
        },
        selectedStoredSessionIdRef: selected,
        startFreshSessionDraft: () => {},
        sttEnabled: false,
        updateSessionState: (_id, updater) => {
          state = updater(state)

          return state
        }
      })

      return useComposerSubmit({
        activeQueueSessionKey: queueRef.current,
        activeQueueSessionKeyRef: queueRef,
        attachments: mainComposerScope.$attachments.get(),
        busy: true,
        busyInputMode: 'steer',
        clearDraft,
        disabled: false,
        draftScopeRef,
        draftRef,
        drainNextQueued: async () => false,
        editorRef,
        exitQueuedEdit: () => false,
        focusInput: () => {},
        inputDisabled: false,
        loadIntoComposer: load,
        onCancel: () => {},
        onSteer: actions.redirectPrompt,
        onSteerHidden: actions.injectHiddenPrompt,
        onSubmit: submit,
        queueCurrentDraft: () => {
          const accepted = enqueueQueuedPrompt(queueRef.current, {
            text: draftRef.current,
            attachments: mainComposerScope.$attachments.get()
          })

          if (accepted) {
            clearDraft()
          }

          return !!accepted
        },
        queueEdit: null,
        queuedPrompts: [],
        sessionId: runtime.current,
        setComposerText: text => {
          draftRef.current = text
        },
        stashAt: (scope, text = draftRef.current, attachments = []) => stashSessionDraft(scope, text, attachments)
      })
    },
    {
      wrapper: ({ children }) => (
        <ComposerSurfaceProvider value="correction-test">
          <div data-composer-surface-id="correction-test" data-composer-target="main">
            {children}
          </div>
        </ComposerSurfaceProvider>
      )
    }
  )

  return {
    hook,
    selected,
    runtime,
    draftRef,
    draftScopeRef,
    queueRef,
    editor,
    request,
    clearDraft,
    load,
    submit,
    get state() {
      return state
    },
    release: () => release(),
    type(text: string) {
      draftRef.current = text
      editor.textContent = text
    },
    switchToB() {
      stashSessionDraft('stored-A', draftRef.current, mainComposerScope.$attachments.get())
      selected.current = 'stored-B'
      runtime.current = 'rt-B'
      draftScopeRef.current = 'stored-B'
      queueRef.current = 'stored-B'
      this.type('B draft')
      hook.rerender()
    }
  }
}

afterEach(() => {
  cleanup()
  clearQueuedPrompts('stored-A')
  clearQueuedPrompts('stored-B')
  clearAllSessionStates()
  clearAllPrompts()
  mainComposerScope.clear()
  $notifications.set([])
  $connection.set(null)
  $activeGatewayProfile.set('default')
})

describe('synthetic correction replies → production prompt action → composer', () => {
  it('normal steer admits once, keeps boolean and does not claim model consumption', async () => {
    const h = harness('normal')
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).toHaveBeenCalledExactlyOnceWith('session.steer', {
      session_id: 'rt-A',
      text: 'neutral follow-up'
    })
    expect(h.draftRef.current).toBe('')
    expect(h.state.messages).toHaveLength(1)
    expect(getQueuedPrompts('stored-A')).toEqual([])
    expect($notifications.get()).toEqual([])
  })

  it('an explicitly queued compression receipt never creates a second queued send', async () => {
    const h = harness('compression')
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.draftRef.current).toBe('')
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.submit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-A')).toEqual([])
    expect($notifications.get()).toHaveLength(1)
    expect($notifications.get()[0].message).toMatch(/compact|压缩/i)
  })

  it('a known undelivered turn-ended receipt uses the existing local queue exactly once', async () => {
    const h = harness('turn_ended')
    await act(async () => h.hook.result.current.submitDraft())
    expect(getQueuedPrompts('stored-A').map(x => x.text)).toEqual(['neutral follow-up'])
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.state.messages).toEqual([])
    expect($notifications.get()).toHaveLength(1)
  })

  it.each(['agent_rejected', 'exception_after_accept', 'unsupported', 'missing'] as const)(
    '%s remains unknown, keeps draft and never resends',
    async name => {
      const h = harness(name)
      await act(async () => h.hook.result.current.submitDraft())
      expect(h.draftRef.current).toBe('neutral follow-up')
      expect(h.request).toHaveBeenCalledTimes(1)
      expect(h.submit).not.toHaveBeenCalled()
      expect(getQueuedPrompts('stored-A')).toEqual([])
      expect(h.state.messages).toEqual([])
      expect($notifications.get()).toHaveLength(1)
      expect($notifications.get()[0].message).toMatch(/unconfirmed|未确认/)
    }
  )

  it('retains legacy queued acceptance without replay or extra notice', async () => {
    const h = harness('legacy')
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.draftRef.current).toBe('')
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(getQueuedPrompts('stored-A')).toEqual([])
    expect(h.submit).not.toHaveBeenCalled()
  })

  it('attempts the running parent despite an async-completion queue backlog', async () => {
    enqueueQueuedPrompt('stored-A', { text: '[async completion] finished', attachments: [] })
    const h = harness('normal')
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).toHaveBeenCalledExactlyOnceWith('session.steer', {
      session_id: 'rt-A',
      text: 'neutral follow-up'
    })
    expect(getQueuedPrompts('stored-A').map(entry => entry.text)).toEqual(['[async completion] finished'])
    expect(h.submit).not.toHaveBeenCalled()
  })

  it('keeps the exact draft and refuses double Enter and Next during the RPC', async () => {
    const h = harness('normal', true)
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.draftRef.current).toBe('neutral follow-up')
    await act(async () => {
      h.hook.result.current.submitDraft()
      h.hook.result.current.queueDraft()
    })
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(getQueuedPrompts('stored-A')).toEqual([])
    await act(async () => h.release())
    expect(h.draftRef.current).toBe('')
  })

  it.each(['normal', 'turn_ended', 'exception_after_accept'] as const)(
    'late %s cannot overwrite a newly typed draft',
    async name => {
      const h = harness(name, true)
      await act(async () => h.hook.result.current.submitDraft())
      h.type('newer words')
      await act(async () => h.release())
      expect(h.draftRef.current).toBe('newer words')
      expect(h.load).not.toHaveBeenCalled()
    }
  )

  it.each([undefined, 'hidden'] as const)(
    'an uncertain external card submission (%s) retains exact text behind Copy without replay',
    async displayKind => {
      const h = harness('exception_after_accept')
      const writeText = vi.fn(async () => {})
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
      await act(async () => {
        expect(requestComposerSubmit('external exact text', { target: 'main', displayKind })).toBe(true)
      })
      expect(h.request).toHaveBeenCalledTimes(1)
      expect(h.draftRef.current).toBe('neutral follow-up')
      expect(h.submit).not.toHaveBeenCalled()
      expect(getQueuedPrompts('stored-A')).toEqual([])
      expect($notifications.get()).toHaveLength(1)
      expect($notifications.get()[0].persistent).toBe(true)
      await act(async () => $notifications.get()[0].action!.onClick())
      expect(writeText).toHaveBeenCalledExactlyOnceWith('external exact text')
    }
  )

  it.each(['compression', 'exception_after_accept', 'normal'] as const)(
    'late A %s cannot touch B, including its notice',
    async name => {
      const h = harness(name, true)
      await act(async () => h.hook.result.current.submitDraft())
      h.switchToB()
      await act(async () => h.release())
      expect(h.draftRef.current).toBe('B draft')
      expect(h.load).not.toHaveBeenCalled()
      expect($notifications.get()).toEqual([])
      expect(h.request).toHaveBeenCalledTimes(1)
      expect(getQueuedPrompts('stored-B')).toEqual([])
    }
  )

  it('captures A before the first microtask, even when navigation happens immediately', async () => {
    const h = harness('normal', true)
    act(() => h.hook.result.current.submitDraft())
    act(() => h.switchToB())
    await act(async () => h.release())
    expect(h.request).toHaveBeenCalledExactlyOnceWith('session.steer', {
      session_id: 'rt-A',
      text: 'neutral follow-up'
    })
    expect(h.draftRef.current).toBe('B draft')
  })

  it('same-text added attachment survives late success', async () => {
    const h = harness('normal', true)
    await act(async () => h.hook.result.current.submitDraft())
    mainComposerScope.add({
      id: 'attachment',
      kind: 'file',
      label: 'neutral.txt',
      path: '/fixture/neutral.txt',
      attachedSessionId: 'rt-A'
    } as ComposerAttachment)
    await act(async () => h.release())
    expect(h.draftRef.current).toBe('neutral follow-up')
    expect(mainComposerScope.$attachments.get()).toHaveLength(1)
  })

  it('A→B→A with C draft cannot be cleared by old A success', async () => {
    const h = harness('normal', true)
    await act(async () => h.hook.result.current.submitDraft())
    h.switchToB()
    h.selected.current = 'stored-A'
    h.runtime.current = 'rt-A'
    h.draftScopeRef.current = 'stored-A'
    h.queueRef.current = 'stored-A'
    h.type('C draft')
    h.hook.rerender()
    await act(async () => h.release())
    expect(h.draftRef.current).toBe('C draft')
  })

  it.each(['connection', 'profile'] as const)(
    'late reply across %s change does not clear or notify',
    async dimension => {
      const h = harness('compression', true)
      await act(async () => h.hook.result.current.submitDraft())

      if (dimension === 'profile') {
        $activeGatewayProfile.set('other')
      } else {
        $connection.set({ mode: 'remote', wsUrl: 'ws://fixture-b' } as never)
      }

      await act(async () => h.release())
      expect(h.draftRef.current).toBe('neutral follow-up')
      expect($notifications.get()).toEqual([])
    }
  )

  it('explicit Next retains the existing queue path without a steer or success notice', async () => {
    const h = harness('normal')
    await act(async () => h.hook.result.current.queueDraft())
    expect(h.request).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-A').map(entry => entry.text)).toEqual(['neutral follow-up'])
    expect($notifications.get()).toEqual([])
  })

  it('an attachment falls back with its actual payload intact', async () => {
    mainComposerScope.add({
      id: 'attachment',
      kind: 'file',
      label: 'neutral.txt',
      path: '/fixture/neutral.txt',
      attachedSessionId: 'rt-A'
    } as ComposerAttachment)
    const h = harness('normal')
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-A')[0].attachments).toMatchObject([
      { id: 'attachment', path: '/fixture/neutral.txt' }
    ])
  })

  it.each([undefined, 'hidden'] as const)(
    'review: late explicitly undelivered external %s remains recoverable after navigation',
    async displayKind => {
      const h = harness('turn_ended', true)
      await act(async () => {
        expect(requestComposerSubmit('external A exact text', { target: 'main', displayKind })).toBe(true)
      })
      h.switchToB()
      await act(async () => h.release())
      expect(h.request).toHaveBeenCalledTimes(1)
      expect(h.submit).not.toHaveBeenCalled()
      expect(h.draftRef.current).toBe('B draft')
      expect(getQueuedPrompts('stored-B')).toEqual([])
      const retainedInA = getQueuedPrompts('stored-A').some(x => x.text === 'external A exact text')
      const copyRecovery = $notifications.get().some(x => x.persistent && x.action)
      expect(retainedInA || copyRecovery).toBe(true)
    }
  )

  it('review: a distinct external request during another pending steer is not silently consumed', async () => {
    const h = harness('normal', true)
    await act(async () => h.hook.result.current.submitDraft())
    await act(async () => {
      expect(requestComposerSubmit('distinct external words', { target: 'main' })).toBe(true)
    })
    await act(async () => h.release())
    const queued = getQueuedPrompts('stored-A').some(x => x.text === 'distinct external words')
    const recovery = $notifications.get().some(x => x.persistent && x.action)
    expect(h.request.mock.calls.length > 1 || queued || recovery).toBe(true)
  })

  it.each([undefined, 'hidden'] as const)(
    'recovery: undelivered external %s offers exact Copy without writing into B',
    async displayKind => {
      const h = harness('turn_ended', true)
      const writeText = vi.fn(async () => {})
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
      await act(async () => {
        expect(requestComposerSubmit('external A exact text', { target: 'main', displayKind })).toBe(true)
      })
      h.switchToB()
      mainComposerScope.add({
        id: 'B-file',
        kind: 'file',
        label: 'B.txt',
        path: '/fixture/B.txt',
        attachedSessionId: 'rt-B'
      })
      await act(async () => h.release())
      expect(h.request).toHaveBeenCalledExactlyOnceWith(displayKind ? 'session.steer' : 'session.redirect', {
        session_id: 'rt-A',
        text: 'external A exact text'
      })
      expect(h.submit).not.toHaveBeenCalled()
      expect(getQueuedPrompts('stored-A')).toEqual([])
      expect(getQueuedPrompts('stored-B')).toEqual([])
      expect(h.draftRef.current).toBe('B draft')
      expect(mainComposerScope.$attachments.get().map(x => x.id)).toEqual(['B-file'])
      expect(h.load).not.toHaveBeenCalled()
      expect($notifications.get()).toHaveLength(1)
      expect($notifications.get()[0]).toMatchObject({ persistent: true, title: 'Message not sent' })
      await act(async () => $notifications.get()[0].action!.onClick())
      expect(writeText).toHaveBeenCalledExactlyOnceWith('external A exact text')
    }
  )

  it.each([
    [undefined, 'normal'],
    ['hidden', 'normal'],
    [undefined, 'exception_after_accept'],
    ['hidden', 'exception_after_accept']
  ] as const)(
    'recovery: distinct external %s stays copyable while the first %s steer owns admission',
    async (displayKind, firstReply) => {
      const h = harness(firstReply, true)
      const writeText = vi.fn(async () => {})
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
      await act(async () => h.hook.result.current.submitDraft())
      mainComposerScope.add({
        id: 'new-file',
        kind: 'file',
        label: 'new.txt',
        path: '/fixture/new.txt',
        attachedSessionId: 'rt-A'
      })
      await act(async () => {
        expect(requestComposerSubmit('distinct external words', { target: 'main', displayKind })).toBe(true)
        h.hook.result.current.submitDraft()
        h.hook.result.current.queueDraft()
      })
      expect(h.request).toHaveBeenCalledTimes(1)
      expect(h.submit).not.toHaveBeenCalled()
      expect(getQueuedPrompts('stored-A')).toEqual([])
      expect($notifications.get()).toHaveLength(1)
      expect($notifications.get()[0]).toMatchObject({ persistent: true, title: 'Message not sent' })
      const recovery = $notifications.get()[0]
      await act(async () => h.release())
      expect(h.request).toHaveBeenCalledTimes(1)
      expect(h.draftRef.current).toBe('neutral follow-up')
      expect(mainComposerScope.$attachments.get().map(x => x.id)).toEqual(['new-file'])
      expect(getQueuedPrompts('stored-A')).toEqual([])
      expect($notifications.get()).toContain(recovery)
      await act(async () => recovery.action!.onClick())
      expect(writeText).toHaveBeenCalledExactlyOnceWith('distinct external words')
    }
  )

  it.each([undefined, 'hidden'] as const)(
    'control: explicit undelivered external %s still queues once for the unchanged owner',
    async displayKind => {
      const h = harness('turn_ended')
      await act(async () => {
        expect(requestComposerSubmit('external fallback', { target: 'main', displayKind })).toBe(true)
      })
      expect(h.request).toHaveBeenCalledTimes(1)
      expect(h.submit).not.toHaveBeenCalled()
      expect(getQueuedPrompts('stored-A')).toMatchObject([
        { text: 'external fallback', attachments: [], ...(displayKind ? { displayKind } : {}) }
      ])
      expect(h.draftRef.current).toBe('neutral follow-up')
      expect($notifications.get()).toEqual([])
    }
  )

  it('blocked approval uses existing queue; it never answers or steers approval', async () => {
    setApprovalRequest({ sessionId: 'rt-A', command: 'neutral', description: 'fixture approval' })
    const h = harness('normal')
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-A')).toHaveLength(1)
  })
})
