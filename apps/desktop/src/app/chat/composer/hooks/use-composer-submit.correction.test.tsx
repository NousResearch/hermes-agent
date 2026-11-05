import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup, renderHook } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import { Harness } from '@/app/session/hooks/use-prompt-actions/index-test-utils'
import { clearSessionDraft, mainComposerScope, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { clearQueuedPrompts, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
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

function fixture(reply: unknown, held = false) {
  const gate = deferred<void>()
  const selected = { current: 'A' as string | null }
  const runtime = { current: 'rt-A' as string | null }
  const draft = { current: '  exact draft\n你好  ' }
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.textContent = draft.current

  const request = vi.fn(async () => {
    if (held) {
      await gate.promise
    }

    if (reply instanceof Error) {
      throw reply
    }

    return reply as never
  })

  const clear = vi.fn(() => {
    draft.current = ''
    editor.textContent = ''
  })

  const submit = vi.fn(async () => true)

  const queue = vi.fn(() => {
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
    return (
      <Harness
        activeSessionIdRef={runtime}
        getRoutedStoredSessionId={() => selected.current}
        onReady={() => {}}
        refreshSessions={async () => {}}
        requestGateway={request}
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
        busy: true,
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
        sessionId: runtime.current,
        setComposerText: text => {
          draft.current = text
        },
        stashAt: (scope, text = '', attachments = []) => stashSessionDraft(scope, text, attachments)
      }),
    { wrapper: Wrapper }
  )

  return {
    hook,
    request,
    submit,
    queue,
    draft,
    selected,
    runtime,
    release: () => gate.resolve(),
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
