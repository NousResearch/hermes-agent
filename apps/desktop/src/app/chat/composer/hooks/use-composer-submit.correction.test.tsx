import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import type { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import { Harness } from '@/app/session/hooks/use-prompt-actions/index-test-utils'
import type { HermesGateway } from '@/hermes'
import { notifyCorrectionRecovery } from '@/lib/correction-delivery'
import {
  clearSessionDraft,
  mainComposerScope,
  setComposerTerminalSelection,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'
import { type ComposerDraftRetry, matchingDraftRetry } from '@/store/composer-draft-retry'
import { clearQueuedPrompts, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
import { $gateway, ensureGatewayForProfile, setPrimaryGateway } from '@/store/gateway'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $connection } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { requestComposerSubmit } from '../focus'
import { ComposerSurfaceProvider } from '../scope'

import { useComposerSubmit } from './use-composer-submit'

// Adapted from yuzilongleif-collab <yuzilong.leif@gmail.com>:
// https://github.com/NousResearch/hermes-agent/issues/141973#issuecomment-6482735120
// Real consuming hooks; synthetic replies at the RPC boundary, not gateway captures.
function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(yes => {
    resolve = yes
  })

  return { promise, resolve }
}

function fixture(reply: unknown, held = false, busy = { current: true }, reconnect = false) {
  const gate = deferred<void>()
  const selected = { current: 'A' as string | null }
  const runtime = { current: 'rt-A' as string | null }
  const draft = { current: '  exact draft\n你好  ' }
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.textContent = draft.current

  const request = vi.fn(async () => {
    if (reconnect && request.mock.calls.length === 1) {
      gateway.connectionState = 'closed'
      throw new Error('connection closed')
    }

    if (held) {
      await gate.promise
    }

    if (reply instanceof Error) {
      throw reply
    }

    return reply as never
  })

  // Recording transport only. The real request hook publishes the fresh descriptor and replays.
  const gateway = {
    request,
    connectionState: 'open',
    connect: vi.fn(async () => {
      gateway.connectionState = 'open'
    })
  }

  if (reconnect) {
    setPrimaryGateway(gateway as unknown as HermesGateway, 'default')
    $gateway.set(gateway as unknown as HermesGateway)
  }

  const clear = vi.fn(() => {
    draft.current = ''
    editor.textContent = ''
  })

  const submit = vi.fn(async () => true)
  const retry = { current: undefined as ComposerDraftRetry | undefined }

  const queue = vi.fn((_id?: string): boolean | Promise<boolean> => {
    const entry = enqueueQueuedPrompt(selected.current, {
      text: draft.current,
      attachments: mainComposerScope.$attachments.get()
    })

    if (entry) {
      clear()
    }

    return !!entry
  })

  let actions!: ReturnType<typeof usePromptActions>

  function Wrapper({ children }: PropsWithChildren) {
    const { requestGateway } = useGatewayRequest()

    return (
      <Harness
        activeSessionIdRef={runtime}
        getRoutedStoredSessionId={() => selected.current}
        onReady={() => {}}
        refreshSessions={async () => {}}
        requestGateway={reconnect ? requestGateway : request}
        runtimeIdByStoredSessionIdRef={{
          current: new Map([
            ['A', 'rt-A'],
            ['B', 'rt-B']
          ])
        }}
        selectedStoredSessionIdRef={selected}
      >
        {value => {
          actions = value

          return (
            <ComposerSurfaceProvider value="correction-test">
              <div data-composer-surface-id="correction-test" data-composer-target="main">
                {children}
              </div>
            </ComposerSurfaceProvider>
          )
        }}
      </Harness>
    )
  }

  const hook = renderHook(
    () =>
      useComposerSubmit({
        activeQueueSessionKey: selected.current,
        activeQueueSessionKeyRef: selected,
        attachments: mainComposerScope.$attachments.get(),
        busy: busy.current,
        busyInputMode: 'steer',
        clearDraft: clear,
        disabled: false,
        draftScopeRef: selected,
        draftRef: draft,
        drainNextQueued: async () => false,
        editorRef: { current: editor },
        exitQueuedEdit: () => false,
        focusInput: () => {},
        inputDisabled: false,
        loadIntoComposer: vi.fn(),
        onCancel: () => {},
        onSteer: actions.redirectPrompt,
        onSteerHidden: actions.injectHiddenPrompt,
        onSubmit: submit,
        queueCurrentDraft: queue,
        queueEdit: null,
        queuedPrompts: [],
        rememberDraftRetry: (_scope, next) => {
          retry.current = next
        },
        sessionId: runtime.current,
        setComposerText: text => {
          draft.current = text
        },
        stashAt: (scope, text = '', attachments = []) => stashSessionDraft(scope, text, attachments),
        takeDraftRetry: (text, attachments) => {
          const taken = matchingDraftRetry(retry.current, text, attachments)
          retry.current = undefined

          return taken
        }
      }),
    { wrapper: Wrapper }
  )

  return {
    hook,
    gateway,
    retry,
    request,
    submit,
    queue,
    draft,
    selected,
    runtime,
    release: () => gate.resolve(),
    setBusy(value: boolean) {
      busy.current = value
      hook.rerender()
    },
    type(text: string) {
      draft.current = text
      editor.textContent = text
    },
    navigate() {
      selected.current = 'B'
      runtime.current = 'rt-B'
      this.type('B draft')
      hook.rerender()
    }
  }
}

async function copyRecovery(text: string, knownUnsent = false) {
  const writeText = vi.fn(async () => {})
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
  clearNotifications()

  const recovery = $notifications
    .get()
    .find(
      n => n.persistent && n.action && (knownUnsent ? /not sent|未发送/ : /unconfirmed|未确认/).test(n.message ?? '')
    )

  expect(recovery?.message).toMatch(knownUnsent ? /not sent|未发送/ : /unconfirmed|未确认/)
  await act(async () => recovery!.action!.onClick())
  expect(writeText).toHaveBeenCalledExactlyOnceWith(text)
}

const lost = new Error('ack lost')

const undelivered = new JsonRpcGatewayError('execution_not_ready', {
  code: 4001,
  data: { reason: 'execution_not_ready' }
})

const outcomes = [undefined, {}, { status: 'rejected' }, lost]

afterEach(() => {
  cleanup()

  for (const key of ['A', 'B']) {
    clearQueuedPrompts(key)
    clearSessionDraft(key)
  }

  clearAllSessionStates()
  clearAllPrompts()
  mainComposerScope.clear()
  $notifications.set([])
  $connection.set(null)
  $activeGatewayProfile.set('default')
  setPrimaryGateway(null)
  $gateway.set(null)
  vi.unstubAllGlobals()
})

it.each(outcomes)('uncertain %j retains exact draft without a new turn', async reply => {
  const h = fixture(reply)
  await act(async () => h.hook.result.current.submitDraft())
  expect(h.request).toHaveBeenCalledTimes(1)
  expect(h.draft.current).toBe('  exact draft\n你好  ')
  expect(h.submit).not.toHaveBeenCalled()
  expect(getQueuedPrompts('A')).toEqual([])
})

it.each([undefined, 'hidden'] as const)('external %s uncertainty retains its own exact Copy', async displayKind => {
  const h = fixture(lost)
  await act(async () => {
    expect(requestComposerSubmit('external\n你好', { target: 'main', displayKind })).toBe(true)
  })
  expect(h.request).toHaveBeenCalledTimes(1)
  expect(h.draft.current).toBe('  exact draft\n你好  ')
  expect(h.submit).not.toHaveBeenCalled()
  expect(getQueuedPrompts('A')).toEqual([])
  await copyRecovery('external\n你好')
})

it.each(
  ([undefined, 'hidden'] as const).flatMap(displayKind =>
    ['navigation', 'profile', 'connection'].map(change => ({ displayKind, change }))
  )
)('F1 external $displayKind known-unsent after $change never writes another owner', async ({ displayKind, change }) => {
  const h = fixture({ status: 'rejected', delivered: false }, true)
  act(() => {
    requestComposerSubmit('external A\n你好', { target: 'main', displayKind })
  })

  if (change === 'navigation') {
    act(() => h.navigate())
  }

  if (change === 'profile') {
    $activeGatewayProfile.set('other')
  }

  if (change === 'connection') {
    $connection.set({ mode: 'remote', wsUrl: 'ws://other' } as never)
  }

  mainComposerScope.add({ id: 'B-file', kind: 'url', label: 'B file' })
  await act(async () => h.release())
  expect(h.request).toHaveBeenCalledExactlyOnceWith(displayKind ? 'session.steer' : 'session.redirect', {
    session_id: 'rt-A',
    text: 'external A\n你好'
  })
  expect(h.draft.current).toBe(change === 'navigation' ? 'B draft' : '  exact draft\n你好  ')
  expect(mainComposerScope.$attachments.get()).toHaveLength(1)
  expect(h.submit).not.toHaveBeenCalled()
  expect(getQueuedPrompts('A')).toEqual([])
  expect(getQueuedPrompts('B')).toEqual([])
  await copyRecovery('external A\n你好', true)
})

it.each(
  ([undefined, 'hidden'] as const).flatMap(displayKind => [false, true].map(loseAck => ({ displayKind, loseAck })))
)(
  'F2 distinct external $displayKind during pending ordinary steer stays copyable (lost ACK: $loseAck)',
  async ({ displayKind, loseAck }) => {
    const h = fixture(loseAck ? lost : { status: 'queued' }, true)
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.draft.current).toBe('  exact draft\n你好  ')
    act(() => {
      requestComposerSubmit('distinct\n你好', { target: 'main', displayKind })
      h.hook.result.current.queueDraft()
      h.hook.result.current.submitDraft()
    })
    await act(async () => h.release())
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.submit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('A')).toEqual([])
    await copyRecovery('distinct\n你好', true)
  }
)

it.each(['new-text', 'attachment', 'navigation', 'profile', 'connection'] as const)(
  'late acceptance preserves %s',
  async change => {
    const h = fixture({ status: 'queued' }, true)
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).toHaveBeenCalledTimes(1)

    if (change === 'new-text') {
      h.type('newer draft')
    }

    if (change === 'attachment') {
      mainComposerScope.add({ id: 'new', kind: 'url', label: 'new' })
    }

    if (change === 'navigation') {
      act(() => h.navigate())
    }

    if (change === 'profile') {
      $activeGatewayProfile.set('other')
    }

    if (change === 'connection') {
      $connection.set({ mode: 'remote', wsUrl: 'ws://fixture' } as never)
    }

    await act(async () => h.release())
    expect(h.request).toHaveBeenCalledExactlyOnceWith('session.steer', {
      session_id: 'rt-A',
      text: 'exact draft\n你好'
    })
    expect(h.draft.current).toBe(
      change === 'navigation' ? 'B draft' : change === 'new-text' ? 'newer draft' : '  exact draft\n你好  '
    )

    if (change === 'attachment') {
      expect(mainComposerScope.$attachments.get()).toHaveLength(1)
    }
  }
)

it('captures the original destination synchronously before immediate navigation', async () => {
  const h = fixture({ status: 'queued' }, true)
  act(() => h.hook.result.current.submitDraft())
  act(() => h.navigate())
  await act(async () => h.release())
  expect(h.request).toHaveBeenCalledExactlyOnceWith('session.steer', { session_id: 'rt-A', text: 'exact draft\n你好' })
  expect(h.draft.current).toBe('B draft')
})

it.each([{ status: 'queued' }, { status: 'redirected' }])('accepted %j clears only the owned draft', async reply => {
  const h = fixture(reply)
  await act(async () => h.hook.result.current.submitDraft())
  expect(h.draft.current).toBe('')
  expect(getQueuedPrompts('A')).toEqual([])
})

it.each([{ status: 'rejected', delivered: false }, undelivered])(
  'proven undelivered %j queues exactly once',
  async reply => {
    const h = fixture(reply)
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.draft.current).toBe('')
    expect(getQueuedPrompts('A').map(x => x.text)).toEqual(['exact draft\n你好'])
  }
)

it.each([{ status: 'queued' }, { status: 'rejected' }])(
  'Enter after the turn ends does not resend a pending %j steer',
  async reply => {
    const busy = { current: true }
    const h = fixture(reply, true, busy)
    await act(async () => h.hook.result.current.submitDraft())
    busy.current = false
    h.hook.rerender()
    await act(async () => h.hook.result.current.submitDraft())
    await act(async () => h.release())
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.submit).not.toHaveBeenCalled()
  }
)

// Adapted from BearHuddleston, PR #149707 discussion_r4488989564.
it.each([{ status: 'queued' }, { status: 'rejected' }])(
  'a second Steer click while a %j steer is pending sends nothing',
  async reply => {
    const h = fixture(reply, true)
    await act(async () => h.hook.result.current.steerDraft('steer'))
    await act(async () => h.hook.result.current.steerDraft('steer'))
    await act(async () => h.release())
    expect(h.request).toHaveBeenCalledTimes(1)
  }
)

it.each(['navigation', 'profile', 'connection'] as const)(
  'known-undelivered steer after %s queues nothing',
  async change => {
    const h = fixture({ status: 'rejected', delivered: false }, true)
    await act(async () => h.hook.result.current.submitDraft())

    if (change === 'navigation') {
      act(() => h.navigate())
    }

    if (change === 'profile') {
      $activeGatewayProfile.set('other')
    }

    if (change === 'connection') {
      $connection.set({ mode: 'remote', wsUrl: 'ws://other' } as never)
    }

    await act(async () => h.release())
    expect(getQueuedPrompts('A')).toEqual([])
    expect(getQueuedPrompts('B')).toEqual([])
  }
)

it.each([true, false])('canonical known-safe fallback holds Enter/Next until acceptance=%s', async accepted => {
  $connection.set({ mode: 'remote', wsUrl: 'ws://fixture/?native_dial=browser' } as never)
  const h = fixture({ status: 'rejected', delivered: false })
  const fallback = deferred<boolean>()
  h.submit.mockImplementation(() => fallback.promise)
  await act(async () => h.hook.result.current.submitDraft())
  expect(h.submit).toHaveBeenCalledTimes(1)
  expect(h.draft.current).toBe('  exact draft\n你好  ')
  await act(async () => {
    h.hook.result.current.submitDraft()
    h.hook.result.current.queueDraft()
  })
  expect(h.request).toHaveBeenCalledTimes(1)
  expect(h.queue).not.toHaveBeenCalled()
  await act(async () => fallback.resolve(accepted))
  expect(h.draft.current).toBe(accepted ? '' : '  exact draft\n你好  ')
  expect(getQueuedPrompts('A')).toEqual([])
  expect(h.submit).toHaveBeenCalledExactlyOnceWith(
    'exact draft\n你好',
    expect.objectContaining({ fromQueue: true, sessionId: 'rt-A', storedSessionId: 'A' })
  )
})

it('an unconfirmed canonical fallback admission keeps its identity for the retry', async () => {
  $connection.set({ mode: 'remote', wsUrl: 'ws://fixture/?native_dial=browser' } as never)
  const h = fixture({ status: 'rejected', delivered: false })
  h.submit.mockRejectedValueOnce(new Error('request timed out: prompt.submit'))
  await act(async () => h.hook.result.current.submitDraft())
  expect(h.draft.current).toBe('  exact draft\n你好  ')
  const [, sent] = h.submit.mock.calls[0] as unknown as [string, { submission_id: string }]
  await act(async () => h.hook.result.current.submitDraft())
  expect(h.request).toHaveBeenCalledTimes(1)
  expect(h.queue).toHaveBeenCalledExactlyOnceWith(sent.submission_id)
})

it.each(['profile', 'connection'] as const)(
  'canonical accepted fallback cannot clear another %s stash',
  async dimension => {
    $connection.set({ mode: 'remote', wsUrl: 'ws://fixture/?native_dial=browser' } as never)
    const h = fixture({ status: 'rejected', delivered: false })
    const fallback = deferred<boolean>()
    h.submit.mockImplementation(() => fallback.promise)
    await act(async () => h.hook.result.current.submitDraft())
    expect(h.submit).toHaveBeenCalledTimes(1)

    if (dimension === 'profile') {
      $activeGatewayProfile.set('other')
    } else {
      $connection.set({ mode: 'remote', wsUrl: 'ws://other' } as never)
    }

    stashSessionDraft('A', 'exact draft\n你好', [])
    await act(async () => fallback.resolve(true))
    expect(takeSessionDraft('A').text).toBe('exact draft\n你好')
    expect(h.draft.current).toBe('  exact draft\n你好  ')
  }
)

it.each(['next', 'attachment', 'approval', 'backlog'] as const)('preserves existing %s routing', async control => {
  const h = fixture({ status: 'queued' })

  if (control === 'attachment') {
    mainComposerScope.add({ id: 'file', kind: 'url', label: 'file' })
    h.hook.rerender()
  }

  if (control === 'approval') {
    setApprovalRequest({ sessionId: 'rt-A', command: 'fixture', description: 'fixture' })
  }

  if (control === 'backlog') {
    enqueueQueuedPrompt('A', { text: 'prior completion', attachments: [] })
  }

  await act(async () => (control === 'next' ? h.hook.result.current.queueDraft() : h.hook.result.current.submitDraft()))
  expect(h.request).toHaveBeenCalledTimes(control === 'backlog' ? 1 : 0)
  expect(getQueuedPrompts('A')).toHaveLength(1)

  if (control === 'attachment') {
    expect(getQueuedPrompts('A')[0].attachments).toHaveLength(1)
  }
})

// Adolanium #6678917819: denied Copy retries must retain one notice and the only copy.
it.each([undefined, 'existing-correction-notice'])('Copy failure reuses notice %s', async noticeId => {
  const text = '  external exact\n你好  '
  const writeText = vi.fn().mockRejectedValue(new Error('clipboard denied'))
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
  const id = notifyCorrectionRecovery(text, true, noticeId)

  for (let attempt = 0; attempt < 3; attempt++) {
    await act(async () =>
      $notifications
        .get()
        .find(n => n.id === id)!
        .action!.onClick()
    )
    expect($notifications.get()).toHaveLength(1)
    expect($notifications.get()[0]).toMatchObject({ id, persistent: true })
  }

  writeText.mockResolvedValue(undefined)
  await act(async () => $notifications.get()[0].action!.onClick())
  expect(writeText.mock.calls.map(([payload]) => payload)).toEqual([text, text, text, text])
})

// Adolanium #6678917819: taking a retry identity does not finish its admission.
it.each([true, false].flatMap(busy => [true, false].map(settle => ({ busy, settle }))))(
  'retry admission holds Enter/Next/Steer while busy=$busy, settle=$settle',
  async ({ busy, settle }) => {
    $connection.set({ mode: 'remote', wsUrl: 'ws://fixture/?native_dial=browser' } as never)
    const h = fixture({ status: 'rejected', delivered: false })
    const original = 'look at @terminal:`zsh:23-58`'
    setComposerTerminalSelection('zsh:23-58', 'selected terminal lines')
    h.type(original)
    h.submit.mockRejectedValueOnce(new Error('request timed out: prompt.submit'))
    await act(async () => h.hook.result.current.submitDraft())
    const retry = h.retry.current!
    expect(retry).toMatchObject({ text: original, id: expect.any(String) })
    const admission = deferred<boolean>()
    h.queue.mockImplementationOnce(() => admission.promise)
    await act(async () => h.hook.result.current.steerDraft('steer'))
    expect(h.retry.current).toBeUndefined()
    // The live turn can finish while the canonical retry is still waiting.
    h.setBusy(busy)
    await act(async () => {
      h.hook.result.current.submitDraft()
      h.hook.result.current.queueDraft()
      h.hook.result.current.steerDraft('steer')
    })
    expect(h.request).toHaveBeenCalledTimes(1)
    expect(h.submit).toHaveBeenCalledTimes(1)
    expect(h.queue).toHaveBeenCalledExactlyOnceWith(retry.id)
    await act(async () => admission.resolve(settle === true))

    if (settle !== true) {
      expect(h.retry.current).toEqual(retry)
      await act(async () => h.hook.result.current.steerDraft('steer'))
      expect(h.queue).toHaveBeenLastCalledWith(retry.id)
    }
  }
)

it('retry scope A/B/A holds each admission until its own settlement', async () => {
  const h = fixture({ status: 'queued' })
  const a = deferred<boolean>()
  const b = deferred<boolean>()
  h.retry.current = { id: 'retry-A', text: h.draft.current, attachmentIds: [] }
  h.queue.mockImplementationOnce(() => a.promise).mockImplementationOnce(() => b.promise)
  await act(async () => h.hook.result.current.steerDraft('steer'))
  act(() => h.navigate())
  h.retry.current = { id: 'retry-B', text: h.draft.current, attachmentIds: [] }
  await act(async () => h.hook.result.current.submitDraft())
  expect(h.queue.mock.calls).toEqual([['retry-A'], ['retry-B']])
  act(() => {
    h.selected.current = 'A'
    h.runtime.current = 'rt-A'
    h.type('  exact draft\n你好  ')
    h.hook.rerender()
  })
  await act(async () => h.hook.result.current.steerDraft('steer'))
  expect(h.request).not.toHaveBeenCalled()
  expect(h.queue).toHaveBeenCalledTimes(2)
  await act(async () => a.resolve(true))
  act(() => h.navigate())
  await act(async () => {
    h.hook.result.current.submitDraft()
    h.hook.result.current.queueDraft()
    h.hook.result.current.steerDraft('steer')
  })
  expect(h.queue).toHaveBeenCalledTimes(2)
  expect(h.request).not.toHaveBeenCalled()
  expect(h.draft.current).toBe('B draft')
  await act(async () => b.resolve(false))
  expect(h.retry.current).toEqual({ id: 'retry-B', text: 'B draft', attachmentIds: [] })
  await act(async () => h.hook.result.current.steerDraft('steer'))
  expect(h.queue).toHaveBeenLastCalledWith('retry-B')
})

it('an accepted steer after the route is re-activated keeps the draft', async () => {
  const h = fixture({ status: 'queued' }, true)
  await act(async () => h.hook.result.current.steerDraft('steer'))
  await act(async () => ensureGatewayForProfile('default'))
  await act(async () => h.release())
  expect(h.request).toHaveBeenCalledTimes(1)
  expect(h.draft.current).toBe('  exact draft\n你好  ')
})

// Adolanium #6678917819: exercise composer -> prompt action -> real request-hook reconnect.
it.each(
  (['steer', 'interrupt'] as const).flatMap(mode =>
    ['unchanged', 'new-text', 'attachment', 'navigation', 'profile', 'connection', 'same-id-endpoint'].map(change => ({
      mode,
      change
    }))
  )
)('accepted $mode reconnect clears only owned draft after $change', async ({ mode, change }) => {
  const original = {
    mode: 'remote' as const,
    connectionId: 'fixture',
    profile: 'default',
    baseUrl: 'http://127.0.0.1:5151',
    wsUrl: 'ws://127.0.0.1:5151/api/ws',
    token: 'synthetic',
    authMode: 'token' as const,
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null,
    logs: []
  }

  const refreshed = { ...original }
  const getConnection = vi.fn(async () => refreshed)
  vi.stubGlobal('hermesDesktop', {
    getConnection,
    getGatewayWsUrl: vi.fn(async () => ({ ok: true, wsUrl: refreshed.wsUrl }))
  })
  $connection.set(original)
  const h = fixture({ status: mode === 'steer' ? 'queued' : 'redirected' }, true, { current: true }, true)
  await act(async () => h.hook.result.current.steerDraft(mode))
  await waitFor(() => expect(h.request).toHaveBeenCalledTimes(2))
  expect(getConnection).toHaveBeenCalledExactlyOnceWith()
  expect($connection.get()).toBe(refreshed)
  expect($connection.get()).not.toBe(original)

  if (change === 'new-text') {
    h.type('newer draft')
  }

  if (change === 'attachment') {
    mainComposerScope.add({ id: 'new', kind: 'url', label: 'new' })
  }

  if (change === 'navigation') {
    act(() => h.navigate())
  }

  if (change === 'profile') {
    $activeGatewayProfile.set('other')
  }

  if (change === 'connection') {
    $connection.set({ ...refreshed, connectionId: 'other' })
  }

  if (change === 'same-id-endpoint') {
    $connection.set({ ...refreshed, baseUrl: 'http://127.0.0.1:5152', wsUrl: 'ws://127.0.0.1:5152/api/ws' })
  }

  await act(async () => h.release())
  expect(h.draft.current).toBe(
    change === 'unchanged'
      ? ''
      : change === 'navigation'
        ? 'B draft'
        : change === 'new-text'
          ? 'newer draft'
          : '  exact draft\n你好  '
  )
  expect(mainComposerScope.$attachments.get()).toHaveLength(change === 'attachment' ? 1 : 0)
  expect(h.queue).not.toHaveBeenCalled()
  expect(h.submit).not.toHaveBeenCalled()
})

it('a retry admission releases only its own scope when it settles', async () => {
  const h = fixture({ status: 'queued' })
  const a = deferred<boolean>()
  const b = deferred<boolean>()
  h.retry.current = { id: 'retry-A', text: h.draft.current, attachmentIds: [] }
  h.queue.mockImplementationOnce(() => a.promise).mockImplementationOnce(() => b.promise)
  await act(async () => h.hook.result.current.steerDraft('steer'))
  act(() => h.navigate())
  h.retry.current = { id: 'retry-B', text: h.draft.current, attachmentIds: [] }
  await act(async () => h.hook.result.current.submitDraft())
  // A settles while B is on screen: B's admission is still pending.
  await act(async () => a.resolve(true))
  await act(async () => h.hook.result.current.steerDraft('steer'))
  expect(h.request).not.toHaveBeenCalled()
  // B's refusal gives its id back and releases B, so Steer retries it.
  await act(async () => b.resolve(false))
  await act(async () => h.hook.result.current.steerDraft('steer'))
  expect(h.queue.mock.calls).toEqual([['retry-A'], ['retry-B'], ['retry-B']])
})
