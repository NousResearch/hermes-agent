import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { type Dispatch, type PropsWithChildren, type SetStateAction, useLayoutEffect, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useSubmitPrompt } from '@/app/session/hooks/use-prompt-actions/submit'
import { registerRecoveredRuntime } from '@/app/session/hooks/use-prompt-actions/single-flight-resume'
import type { GatewayRequest, SubmitTextOptions } from '@/app/session/hooks/use-prompt-actions/utils'
import type { Translations } from '@/i18n'
import { createClientSessionState } from '@/lib/chat-runtime'
import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { $clarifyRequests } from '@/store/clarify'
import type { ComposerAttachment } from '@/store/composer'
import { $gateway } from '@/store/gateway'
import { $parkedQueueSessions, $queuedPromptsBySession, enqueueQueuedPrompt } from '@/store/composer-queue'
import {
  clearAllPrompts,
  hasBlockingPromptRequest,
  setApprovalRequest,
  setSecretRequest,
  setSudoRequest
} from '@/store/prompts'

import { type ComposerTarget, requestComposerSubmit } from '../focus'
import { ComposerScopeProvider, ComposerSurfaceProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useComposerSubmit } from './use-composer-submit'
import { useComposerQueue } from './use-composer-queue'

interface SubmitHarnessOptions {
  attachments?: ComposerAttachment[]
  submit?: (text: string, options?: SubmitTextOptions) => Promise<boolean>
  busy?: boolean
  compacting?: boolean
  inputDisabled?: boolean
  scopeTarget?: ComposerTarget
  sessionKey?: string | null
  submitOnHide?: boolean
  surfaceId?: string | null
  text?: string
  visible?: boolean
}

let surfaceSequence = 0

function renderSubmitHook({
  attachments = [],
  submit,
  busy = false,
  compacting = false,
  inputDisabled = false,
  scopeTarget = 'main',
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
  const onSubmit = vi.fn(submit ?? (async () => true))
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
        sessionId: 'runtime-session',
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

function renderWireSubmit() {
  const requestGateway = vi.fn<GatewayRequest>().mockResolvedValue({})
  let state = createClientSessionState('stored-session')
  const wire = renderHook(() =>
    useSubmitPrompt({
      activeSessionIdRef: { current: 'runtime-session' },
      busyRef: { current: false },
      copy: {} as Translations['desktop'],
      createBackendSessionForSend: vi.fn(async () => null),
      getRoutedStoredSessionId: () => 'stored-session',
      getRuntimeIdForStoredSession: () => 'runtime-session',
      getRouteToken: () => 'stored-session',
      onRuntimeRecovered: vi.fn(),
      requestGateway: requestGateway as GatewayRequest,
      runtimeIdByStoredSessionIdRef: { current: new Map([['stored-session', 'runtime-session']]) },
      resumeStoredSession: vi.fn(),
      selectedStoredSessionIdRef: { current: 'stored-session' },
      syncAttachmentsForSubmit: async (sessionId, attachments) => ({ sessionId, attachments }),
      updateSessionState: (_id, updater) => (state = updater(state)),
      scope: {
        readAttachments: () => [],
        removeAttachments: vi.fn(),
        setAwaitingResponse: vi.fn(),
        setBusy: vi.fn(),
        setMessages: vi.fn()
      }
    })
  )
  return { requestGateway, submit: wire.result.current }
}

describe('composer producer to prompt.submit provenance', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('preserves raw deliberate text before path and attachment enrichment', async () => {
    const wire = renderWireSubmit()
    const raw = '  inspect @apps/desktop/  '
    const composer = renderSubmitHook({
      text: raw,
      submit: wire.submit,
      attachments: [{ id: 'context', kind: 'file', label: 'context', refText: '@file:context.md' }]
    })
    act(() => composer.hook.result.current.submitDraft())
    await waitFor(() =>
      expect(wire.requestGateway).toHaveBeenCalledWith(
        'prompt.submit',
        expect.objectContaining({ input_provenance: { kind: 'desktop_composer', raw_text: raw } }),
        expect.any(Number)
      )
    )
    const params = wire.requestGateway.mock.calls.find(([method]) => method === 'prompt.submit')![1]!
    expect(params.text).toContain('@file:context.md')
    expect(params.text).not.toBe(raw)
  })

  it('does not promote generated external composer requests', async () => {
    const wire = renderWireSubmit()
    renderSubmitHook({ submit: wire.submit })
    act(() => {
      requestComposerSubmit('generated instruction', { target: 'main' })
    })
    await waitFor(() => expect(wire.requestGateway).toHaveBeenCalled())
    expect(wire.requestGateway.mock.calls[0][0]).toBe('prompt.submit')
    expect(wire.requestGateway.mock.calls[0][1]).not.toHaveProperty('input_provenance')
  })

  it('routes a hidden widget intent through the external composer producer without evidence', async () => {
    const wire = renderWireSubmit()
    renderSubmitHook({ submit: wire.submit })
    act(() => {
      requestComposerSubmit('widget intent', { target: 'main', displayKind: 'hidden' })
    })
    await waitFor(() => expect(wire.requestGateway).toHaveBeenCalled())
    expect(wire.requestGateway.mock.calls[0][1]).toMatchObject({ text: 'widget intent', display_kind: 'hidden' })
    expect(wire.requestGateway.mock.calls[0][1]).not.toHaveProperty('input_provenance')
  })

  it('drains a real queued entry into prompt.submit without composer evidence', async () => {
    const wire = renderWireSubmit()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    const entry = enqueueQueuedPrompt('stored-session', { text: 'queued instruction', attachments: [] })!
    const queue = renderHook(() =>
      useComposerQueue({
        activeQueueSessionKey: 'stored-session',
        attachments: [],
        busy: false,
        clearDraft: vi.fn(),
        draftRef: { current: '' },
        focusInput: vi.fn(),
        loadIntoComposer: vi.fn(),
        onCancel: vi.fn(),
        onSteer: undefined,
        onSubmit: wire.submit,
        queueEditRef: { current: null },
        queueSessionKey: 'stored-session',
        sessionId: 'runtime-session'
      })
    )
    try {
      await act(async () => {
        await queue.result.current.sendQueuedNow(entry.id)
      })
      expect(wire.requestGateway).toHaveBeenCalledWith(
        'prompt.submit',
        expect.objectContaining({ queued: true, text: 'queued instruction' }),
        expect.any(Number)
      )
      expect(wire.requestGateway.mock.calls[0][1]).not.toHaveProperty('input_provenance')
    } finally {
      queue.unmount()
      $queuedPromptsBySession.set({})
      $parkedQueueSessions.set({})
    }
  })

  it.each([
    ['widget', { displayText: 'Widget action' }],
    ['hidden', { displayKind: 'hidden' }],
    ['queue', { fromQueue: true }]
  ] satisfies [string, SubmitTextOptions][])('strips composer evidence from %s submissions', async (_kind, options) => {
    const wire = renderWireSubmit()
    const composer = renderSubmitHook({
      text: 'human text',
      submit: (text, evidence) => wire.submit(text, { ...evidence, ...options })
    })
    act(() => composer.hook.result.current.submitDraft())
    await waitFor(() => expect(wire.requestGateway).toHaveBeenCalled())
    expect(wire.requestGateway.mock.calls[0][1]).not.toHaveProperty('input_provenance')
  })

  it('omits evidence on a busy retry after a deliberate composer send', async () => {
    const wire = renderWireSubmit()
    wire.requestGateway.mockRejectedValueOnce(new Error('session busy'))
    const composer = renderSubmitHook({ text: 'human text', submit: wire.submit })
    act(() => composer.hook.result.current.submitDraft())
    await waitFor(() => expect(wire.requestGateway).toHaveBeenCalledTimes(2), { timeout: 5000 })
    expect(wire.requestGateway.mock.calls[0][1]).toHaveProperty('input_provenance.raw_text', 'human text')
    expect(wire.requestGateway.mock.calls[1][1]).not.toHaveProperty('input_provenance')
  })

  it('omits evidence when retrying a composer send on a recovered runtime', async () => {
    const wire = renderWireSubmit()
    registerRecoveredRuntime('stored-session', 'recovered-runtime')
    wire.requestGateway.mockRejectedValueOnce(new Error('session not found'))
    const composer = renderSubmitHook({ text: 'human text', submit: wire.submit })
    act(() => composer.hook.result.current.submitDraft())
    await waitFor(() => expect(wire.requestGateway).toHaveBeenCalledTimes(2))
    expect(wire.requestGateway.mock.calls[0][1]).toHaveProperty('input_provenance.raw_text', 'human text')
    expect(wire.requestGateway.mock.calls[1][1]).toMatchObject({ session_id: 'recovered-runtime' })
    expect(wire.requestGateway.mock.calls[1][1]).not.toHaveProperty('input_provenance')
  })

  it('defaults helper sends to no evidence', async () => {
    const wire = renderWireSubmit()
    await act(async () => {
      expect(await wire.submit('delegated task')).toBe(true)
    })
    expect(wire.requestGateway.mock.calls[0][1]).not.toHaveProperty('input_provenance')
  })
})

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

  it('captures the raw composer before reference expansion', async () => {
    const raw = '  inspect @apps/desktop/  '
    const { hook, onSubmit } = renderSubmitHook({ text: raw })
    act(() => hook.result.current.submitDraft())
    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    expect(onSubmit).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        inputProvenance: { kind: 'desktop_composer', raw_text: raw }
      })
    )
  })

  it('submits a normal turn while idle', async () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ text: 'ordinary question' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith('ordinary question', {
        attachments: [],
        inputProvenance: { kind: 'desktop_composer', raw_text: 'ordinary question' },
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
