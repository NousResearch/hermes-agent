import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { type Dispatch, type PropsWithChildren, type SetStateAction, useLayoutEffect, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { $clarifyRequests } from '@/store/clarify'
import type { ComposerAttachment } from '@/store/composer'
import { $gateway } from '@/store/gateway'
import {
  clearAllPrompts,
  hasBlockingPromptRequest,
  setApprovalRequest,
  setSecretRequest,
  setSudoRequest
} from '@/store/prompts'
import { $sessions } from '@/store/session'
import { $sessionStates, clearAllSessionStates } from '@/store/session-states'

import { type ComposerTarget, requestComposerSubmit } from '../focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerSubmit } from './use-composer-submit'

interface SubmitHarnessOptions {
  attachments?: ComposerAttachment[]
  busy?: boolean
  compacting?: boolean
  inputDisabled?: boolean
  scopeTarget?: ComposerTarget
  sessionId?: string
  sessionKey?: string | null
  submitOnHide?: boolean
  surfaceId?: string | null
  text?: string
  visible?: boolean
}

let surfaceSequence = 0

function renderSubmitHook({
  attachments = [],
  busy = false,
  compacting = false,
  inputDisabled = false,
  scopeTarget = 'main',
  sessionId = 'runtime-session',
  sessionKey = 'stored-session',
  submitOnHide = false,
  surfaceId,
  text = '',
  visible = true
}: SubmitHarnessOptions = {}) {
  const resolvedSurfaceId = surfaceId === undefined ? `test-surface-${++surfaceSequence}` : surfaceId
  const draftRef = { current: text }
  const editor = window.document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.textContent = text
  const editorRef = { current: editor }
  const onCancel = vi.fn()
  const onSteer = vi.fn(async () => true)
  const onSubmit = vi.fn(async () => true)
  const queueCurrentDraft = vi.fn(() => true)
  let updatePaneVisible: Dispatch<SetStateAction<boolean>> | undefined

  const clearDraft = vi.fn(() => {
    draftRef.current = ''
    editorRef.current!.textContent = ''
  })

  const Wrapper = ({ children }: PropsWithChildren) => {
    const [paneVisible, setPaneVisible] = useState(visible)
    updatePaneVisible = setPaneVisible

    useLayoutEffect(() => {
      if (submitOnHide && !paneVisible) {
        requestComposerSubmit('ship while hiding', { target: scopeTarget })
      }
    }, [paneVisible])

    return (
      <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target: scopeTarget }}>
        <ComposerSurfaceProvider value={resolvedSurfaceId}>
          <PaneVisibleContext.Provider value={paneVisible}>
            <div
              data-composer-surface-id={resolvedSurfaceId ?? undefined}
              data-composer-target={scopeTarget}
              data-pane-hidden={paneVisible ? undefined : ''}
            >
              {children}
            </div>
          </PaneVisibleContext.Provider>
        </ComposerSurfaceProvider>
      </ComposerScopeProvider>
    )
  }

  const hook = renderHook(
    () =>
      useComposerSubmit({
        activeQueueSessionKey: sessionKey,
        activeQueueSessionKeyRef: { current: sessionKey },
        attachments,
        busy,
        compacting,
        clearDraft,
        disabled: false,
        draftRef,
        drainNextQueued: vi.fn(async () => false),
        editorRef,
        exitQueuedEdit: vi.fn(() => false),
        focusInput: vi.fn(),
        inputDisabled,
        loadIntoComposer: vi.fn(),
        onCancel,
        onSteer,
        onSubmit,
        queueCurrentDraft,
        queueEdit: null,
        queuedPrompts: [],
        sessionId,
        setComposerText: vi.fn(),
        stashAt: vi.fn()
      }),
    { wrapper: Wrapper }
  )

  return {
    clearDraft,
    hook,
    onCancel,
    onSteer,
    onSubmit,
    queueCurrentDraft,
    composerSurfaceId: resolvedSurfaceId,
    setPaneVisible(nextVisible: boolean) {
      if (!updatePaneVisible) {
        throw new Error('Pane visibility setter was not initialized')
      }

      updatePaneVisible(nextVisible)
    }
  }
}

describe('useComposerSubmit external request routing', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('does not fan out a main ship across keep-alives or other projects', async () => {
    const visibleMain = renderSubmitHook({ sessionKey: 'session-a' })
    const hiddenMain = renderSubmitHook({ sessionKey: 'session-b', visible: false })
    const visibleTile = renderSubmitHook({ scopeTarget: 'tile:project-b', sessionKey: 'tile-session' })

    const hiddenTile = renderSubmitHook({
      scopeTarget: 'tile:project-c',
      sessionKey: 'other-tile',
      visible: false
    })

    expect(requestComposerSubmit('ship this branch', { target: 'main' })).toBe(true)

    await waitFor(() =>
      expect(visibleMain.onSubmit).toHaveBeenCalledWith('ship this branch', {
        composerScope: 'session-a'
      })
    )
    expect(visibleMain.onSubmit).toHaveBeenCalledTimes(1)
    expect(hiddenMain.onSubmit).not.toHaveBeenCalled()
    expect(visibleTile.onSubmit).not.toHaveBeenCalled()
    expect(hiddenTile.onSubmit).not.toHaveBeenCalled()
  })

  it('routes a tile-targeted submit to that tile only', async () => {
    const main = renderSubmitHook({ sessionKey: 'main-session' })
    const tile = renderSubmitHook({ scopeTarget: 'tile:project-b', sessionKey: 'tile-session' })

    expect(requestComposerSubmit('ship project B', { target: 'tile:project-b' })).toBe(true)

    await waitFor(() =>
      expect(tile.onSubmit).toHaveBeenCalledWith('ship project B', {
        composerScope: 'tile-session'
      })
    )
    expect(main.onSubmit).not.toHaveBeenCalled()
  })

  it('uses the captured surface id when two visible composers share a target', async () => {
    const first = renderSubmitHook({ sessionKey: 'session-first' })
    const second = renderSubmitHook({ sessionKey: 'session-second' })

    requestComposerSubmit('ship exactly one session', { surfaceId: second.composerSurfaceId, target: 'main' })

    await waitFor(() =>
      expect(second.onSubmit).toHaveBeenCalledWith('ship exactly one session', {
        composerScope: 'session-second'
      })
    )
    expect(first.onSubmit).not.toHaveBeenCalled()
  })

  it('submits to the session visible at click time even when the same click switches tabs', async () => {
    const hiddenA = renderSubmitHook({ sessionKey: 'session-a', visible: false })
    const visibleB = renderSubmitHook({ sessionKey: 'session-b' })

    act(() => {
      requestComposerSubmit('ship session B', { target: 'main' })
      visibleB.setPaneVisible(false)
      hiddenA.setPaneVisible(true)
    })

    await waitFor(() =>
      expect(visibleB.onSubmit).toHaveBeenCalledWith('ship session B', {
        composerScope: 'session-b'
      })
    )
    expect(hiddenA.onSubmit).not.toHaveBeenCalled()
  })

  it('does not fan out when visible composers do not have queue session keys yet', async () => {
    const firstNewSession = renderSubmitHook({ sessionKey: null })
    const secondNewSession = renderSubmitHook({ sessionKey: null })

    act(() => {
      requestComposerSubmit('ship the visible new session', { target: 'main' })
    })

    await waitFor(() => expect(firstNewSession.onSubmit).toHaveBeenCalledTimes(1))
    expect(secondNewSession.onSubmit).not.toHaveBeenCalled()
  })

  it('fails closed when the visible composer has no surface identity', () => {
    const unidentified = renderSubmitHook({ surfaceId: null })

    expect(requestComposerSubmit('do not broadcast this', { target: 'main' })).toBe(false)
    expect(unidentified.onSubmit).not.toHaveBeenCalled()
  })

  it('fails closed when a pinned origin surface is no longer visible', () => {
    const hidden = renderSubmitHook({ sessionKey: 'hidden-origin', visible: false })
    const visible = renderSubmitHook({ sessionKey: 'other-visible' })

    expect(
      requestComposerSubmit('do not send to a stale origin', {
        surfaceId: hidden.composerSurfaceId,
        target: 'main'
      })
    ).toBe(false)
    expect(hidden.onSubmit).not.toHaveBeenCalled()
    expect(visible.onSubmit).not.toHaveBeenCalled()
  })

  it('does not submit through a composer whose pane is hidden during the request', () => {
    const main = renderSubmitHook({ submitOnHide: true })

    act(() => main.setPaneVisible(false))

    expect(main.onSubmit).not.toHaveBeenCalled()
  })

  it('does not submit through a disabled composer', () => {
    const disabled = renderSubmitHook({ inputDisabled: true })

    requestComposerSubmit('do not send this', { target: 'main' })

    expect(disabled.onSubmit).not.toHaveBeenCalled()
  })
})

describe('useComposerSubmit busy-turn routing', () => {
  afterEach(() => {
    // Reset the store seeds used by the leftover-runtime tests unconditionally,
    // so a mid-test throw can't leak $sessions/$sessionStates into later tests.
    $sessions.set([])
    clearAllSessionStates()
    cleanup()
    vi.restoreAllMocks()
  })

  it('treats a payload mid-turn as send (steer), not stop', async () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: 'change course'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('change course'))
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('queues a plain-text follow-up while the active turn is compacting', () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      compacting: true,
      text: 'wait for the summary'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('runs slash commands immediately while busy', async () => {
    const { clearDraft, hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: '/compress preserve context'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('/compress preserve context', { composerScope: 'stored-session' })
    )
    expect(clearDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('queues an attachment-bearing follow-up while busy', () => {
    const attachment: ComposerAttachment = { id: 'doc', kind: 'file', label: 'notes.txt' }

    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      attachments: [attachment],
      busy: true,
      text: 'read this'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('stops an active turn only with an empty composer', () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ busy: true })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
  })

  it('does not steer a leftover busy runtime after the composer moved to another chat', async () => {
    $sessions.set([
      { id: 'stored-a', _lineage_root_id: 'stored-a' },
      { id: 'stored-b', _lineage_root_id: 'stored-b' }
    ] as never)
    $sessionStates.set({
      'runtime-a': { storedSessionId: 'stored-a', busy: true }
    } as never)

    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      sessionId: 'runtime-a',
      sessionKey: 'stored-b',
      text: 'how can I tell how much context is left in hermes desktop?'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('how can I tell how much context is left in hermes desktop?', {
        composerScope: 'stored-b'
      })
    )
    expect(onSteer).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()

    $sessions.set([])
    clearAllSessionStates()
  })

  it('still steers on a busy Enter when the runtime state has no stored id yet', async () => {
    // Runtime known busy but no storedSessionId attached — insufficient
    // evidence to declare a scope mismatch, keep the historical steer path.
    $sessions.set([{ id: 'stored-session', _lineage_root_id: 'stored-session' }] as never)
    $sessionStates.set({
      'runtime-session': { busy: true }
    } as never)

    const { hook, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: 'course correction mid-turn'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('course correction mid-turn'))
    expect(onSubmit).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()

    $sessions.set([])
    clearAllSessionStates()
  })

  it('still steers on a busy Enter when the session list has not loaded yet', async () => {
    // Same evidence gap as missing ids — refuse to demote a steer to a fresh
    // submit without proof the runtime belongs to a different scope.
    $sessions.set([] as never)
    $sessionStates.set({
      'runtime-session': { storedSessionId: 'stored-session', busy: true }
    } as never)

    const { hook, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: 'course correction mid-turn'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('course correction mid-turn'))
    expect(onSubmit).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()

    clearAllSessionStates()
  })

  it('submits a normal turn while idle', async () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ text: 'ordinary question' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('ordinary question', {
        attachments: [],
        composerScope: 'stored-session'
      })
    )
    expect(onSteer).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('threads the loaded composer scope through onSubmit for the #59305 submit-time guard', async () => {
    const { hook, onSubmit } = renderSubmitHook({ text: 'hello' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('hello', expect.objectContaining({ composerScope: 'stored-session' }))
    )
  })
})

describe('useComposerSubmit with a clarify parked on the session', () => {
  const gatewayRequest = vi.fn(async () => ({ ok: true }))

  const parkClarify = (sessionId: string) => {
    $clarifyRequests.set({
      [sessionId]: {
        requestId: `req-${sessionId}`,
        question: 'which one?',
        choices: ['a', 'b'],
        multiSelect: false,
        sessionId
      }
    })
    $gateway.set({ request: gatewayRequest } as unknown as ReturnType<typeof $gateway.get>)
  }

  afterEach(() => {
    cleanup()
    gatewayRequest.mockClear()
    $clarifyRequests.set({})
    $gateway.set(null)
    vi.restoreAllMocks()
  })

  it('skips the question and still sends the typed message on an idle session', async () => {
    parkClarify('runtime-session')
    const { hook, onSubmit } = renderSubmitHook({ text: 'actually do this instead' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(gatewayRequest).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'req-runtime-session',
        answer: ''
      })
    )
    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('actually do this instead', expect.objectContaining({ attachments: [] }))
    )
    expect($clarifyRequests.get()['runtime-session']).toBeUndefined()
  })

  it('skips the question before steering a busy turn', async () => {
    parkClarify('runtime-session')
    const { hook, onSteer } = renderSubmitHook({ busy: true, text: 'change course' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('change course'))
    expect(gatewayRequest).toHaveBeenCalledWith('clarify.respond', { request_id: 'req-runtime-session', answer: '' })
  })

  it('leaves the question alone for an empty Enter (Stop, not an answer)', () => {
    parkClarify('runtime-session')
    const { hook, onCancel } = renderSubmitHook({ busy: true })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(gatewayRequest).not.toHaveBeenCalled()
    expect($clarifyRequests.get()['runtime-session']).toBeDefined()
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it("leaves another session's question alone", async () => {
    parkClarify('other-session')
    const { hook, onSubmit } = renderSubmitHook({ text: 'unrelated message' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    expect(gatewayRequest).not.toHaveBeenCalled()
    expect($clarifyRequests.get()['other-session']).toBeDefined()
  })
})

describe('useComposerSubmit with a blocking prompt parked on the session', () => {
  // Typing cannot answer approval/sudo/secret prompts, so the busy submit must
  // route text to the QUEUE — a steer would sit undelivered behind the blocked
  // tool batch, and interrupting to force it through resolves the prompt empty
  // and ends the turn as "Operation interrupted." with the message lost.
  afterEach(() => {
    cleanup()
    clearAllPrompts()
    vi.restoreAllMocks()
  })

  it('queues a busy text follow-up instead of steering while an approval is pending', () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous', sessionId: 'runtime-session' })

    const { hook, onCancel, onSteer, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: 'and also fix the padding'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('queues while a sudo prompt is pending', () => {
    setSudoRequest({ requestId: 'sudo-1', sessionId: 'runtime-session' })

    const { hook, onSteer, queueCurrentDraft } = renderSubmitHook({ busy: true, text: 'next thing' })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
  })

  it('queues while a secret prompt is pending', () => {
    setSecretRequest({ envVar: 'API_KEY', prompt: 'key?', requestId: 'sec-1', sessionId: 'runtime-session' })

    const { hook, onSteer, queueCurrentDraft } = renderSubmitHook({ busy: true, text: 'next thing' })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
  })

  it('still runs slash commands inline', async () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous', sessionId: 'runtime-session' })

    const { hook, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ busy: true, text: '/status' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('/status', { composerScope: 'stored-session' }))
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onSteer).not.toHaveBeenCalled()
  })

  it("ignores another session's blocking prompt and still steers", async () => {
    setApprovalRequest({ command: 'ls', description: 'other', sessionId: 'other-session' })

    const { hook, onSteer, queueCurrentDraft } = renderSubmitHook({ busy: true, text: 'change course' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('change course'))
    expect(queueCurrentDraft).not.toHaveBeenCalled()
  })

  it('leaves the prompt pending — queueing must not resolve or dismiss it', () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous', sessionId: 'runtime-session' })

    const { hook } = renderSubmitHook({ busy: true, text: 'follow-up' })

    act(() => {
      hook.result.current.submitDraft()
    })

    // The approval card is still the turn's owner; only its own buttons answer it.
    expect(hasBlockingPromptRequest('runtime-session')).toBe(true)
  })
})
