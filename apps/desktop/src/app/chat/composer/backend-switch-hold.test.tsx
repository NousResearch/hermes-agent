// @vitest-environment jsdom
// A profile whose backend is away on purpose (a plugin starting, stopping or restarting it:
// store/backend-transition.ts) must not take a message on the backend that is about to go away.
//  - Starting / stopping: the profile moves to another backend. Nothing is sent or queued: Enter
//    and Send do nothing, Send is off, the message stays in the composer, and when the move is
//    done the composer works again with the message still there. Nothing goes by itself.
//  - Restarting: the same backend and its sessions come back. A message waits in the chat's
//    local queue (the queued row, not a thread row), also while the connection is down, and goes
//    once when the backend is back.
// Slash commands, other profiles' chats and chats without a transition are not affected.
import { AssistantRuntimeProvider, type ThreadMessageLike, useExternalStoreRuntime } from '@assistant-ui/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useState } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { I18nProvider } from '@/i18n'
import { clearProfileBackend, setProfileBackend } from '@/sdk/profile-backend'
import {
  backendTransitionFor,
  beginBackendTransition,
  endBackendTransition,
  resetBackendTransitionsForTests
} from '@/store/backend-transition'
import { announceNewSessionDraftKey, mainComposerScope } from '@/store/composer'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  resetFrozenQueuedTransportsForTests
} from '@/store/composer-queue'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, setGatewayState, setSessionsLoading } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarProps, ChatBarState } from './types'

import { ChatBar } from './index'

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

type Submit = ChatBarProps['onSubmit']

interface View {
  busy?: boolean
  disabled?: boolean
  freshDraftKey?: string
  onSubmit: Submit
  profile?: string
  queueSessionKey?: null | string
  sessionId?: null | string
}

function Harness({
  busy = false,
  disabled = false,
  freshDraftKey,
  onSubmit,
  profile = 'work',
  queueSessionKey,
  sessionId
}: View) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))

  return (
    <QueryClientProvider client={client}>
      <AssistantRuntimeProvider runtime={runtime}>
        <MemoryRouter>
          <I18nProvider configClient={null} initialLocale="en">
            <ChatBar
              busy={busy}
              disabled={disabled}
              freshDraftKey={freshDraftKey}
              gateway={null}
              onCancel={vi.fn()}
              onSteer={vi.fn(async () => true)}
              onSubmit={onSubmit}
              profile={profile}
              queueSessionKey={queueSessionKey}
              sessionId={sessionId}
              state={state}
            />
          </I18nProvider>
        </MemoryRouter>
      </AssistantRuntimeProvider>
    </QueryClientProvider>
  )
}

const settle = (ms = 0) => act(() => new Promise<void>(resolve => window.setTimeout(resolve, ms)))

function caretToEnd(editor: HTMLElement) {
  const range = window.document.createRange()
  range.selectNodeContents(editor)
  range.collapse(false)
  const selection = window.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
}

const EXISTING_CHAT = { queueSessionKey: 'stored-work', sessionId: 'rt-work' }

function mount(view: Omit<View, 'onSubmit'>) {
  const onSubmit = vi.fn<Submit>(async () => true)
  const rendered = render(<Harness onSubmit={onSubmit} {...view} />)
  const editor = rendered.container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!
  // jsdom does not make contenteditable focusable like Chromium does.
  editor.tabIndex = 0

  const rerender = (next: Omit<View, 'onSubmit'>) =>
    act(() => rendered.rerender(<Harness onSubmit={onSubmit} {...view} {...next} />))

  // The Send button (the composer form's submit button; shown once the composer holds a message).
  const send = () => rendered.container.querySelector<HTMLButtonElement>('button[type="submit"]')

  // What a click on Send does: jsdom does not submit a form from a button click.
  const clickSend = async () => {
    await act(async () => {
      fireEvent.submit(editor.closest('form')!)
    })
    await settle(50)
  }

  return { clickSend, editor, onSubmit, rerender, send }
}

/** Type a message, as the user does, and let the composer take it in. */
async function type(editor: HTMLElement, text: string) {
  await act(async () => {
    editor.focus()
    editor.textContent = text
    caretToEnd(editor)
    fireEvent.input(editor)
  })
  await settle(50)
}

async function pressEnter(editor: HTMLElement) {
  await act(async () => {
    fireEvent.keyDown(editor, { key: 'Enter' })
  })
  await settle(50)
}

async function typeAndEnter(editor: HTMLElement, text: string) {
  await type(editor, text)
  await pressEnter(editor)
}

const placeholder = (editor: HTMLElement) => editor.getAttribute('data-placeholder')

beforeEach(() => {
  window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
  resetFrozenQueuedTransportsForTests()
  setSessionsLoading(false)
  setGatewayState('open')
  // This computer's backend owns its chats' queues (the gateway FIFO): nothing may be put there
  // while the profile leaves that backend.
  $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as HermesConnection)
})

afterEach(() => {
  cleanup()
  resetBackendTransitionsForTests()
  $activeGatewayProfile.set('default')
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  $queuedPromptsBySession.set({})
  resetFrozenQueuedTransportsForTests()
  mainComposerScope.clear()
  announceNewSessionDraftKey(null)
  clearAllSessionStates()
  $connection.set(null)
  clearNotifications()
  setGatewayState('idle')
  setSessionsLoading(true)
  window.localStorage.clear()
})

describe('a message while the profile backend starts or stops', () => {
  it('starting: Enter and Send do nothing, the message stays in the composer, nothing goes by itself', async () => {
    beginBackendTransition('work', 'starting', { source: 'switch' })
    // This computer's backend still serves the chat while the plugin's starts: the connection is open.
    const { clickSend, editor, onSubmit, send } = mount(EXISTING_CHAT)

    expect(placeholder(editor)).toBe('Starting sandbox…')

    await type(editor, 'hey can you read updog.com for me')
    expect(send()?.disabled).toBe(true)

    await pressEnter(editor)
    await clickSend()

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-work')).toEqual([])
    expect(composerPlainText(editor)).toBe('hey can you read updog.com for me')

    // The plugin's backend is up and the window's socket to it is open: the transition ends.
    act(() => endBackendTransition('work', { source: 'switch' }))
    await settle(150)

    expect(onSubmit).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe('hey can you read updog.com for me')
    expect(send()?.disabled).toBe(false)
    expect(placeholder(editor)).not.toBe('Starting sandbox…')

    await pressEnter(editor)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe('hey can you read updog.com for me')
    expect(onSubmit.mock.calls[0]?.[1]?.fromQueue).toBeUndefined()
    expect(composerPlainText(editor)).toBe('')
  })

  it('stopping: the same, while the backend still answers and after the connection dropped', async () => {
    beginBackendTransition('work', 'stopping', { source: 'switch' })
    const { clickSend, editor, onSubmit, rerender, send } = mount(EXISTING_CHAT)

    expect(placeholder(editor)).toBe('Stopping sandbox…')

    await typeAndEnter(editor, 'and the sandbox is off now?')
    expect(send()?.disabled).toBe(true)
    expect(onSubmit).not.toHaveBeenCalled()

    // The plugin stopped its backend: the window's socket to it drops.
    setGatewayState('connecting')
    await rerender({ ...EXISTING_CHAT, disabled: true })
    await pressEnter(editor)
    await clickSend()

    // This computer's backend took the profile back; the transition has not ended yet.
    setGatewayState('open')
    await rerender({ ...EXISTING_CHAT, disabled: false })
    await pressEnter(editor)

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-work')).toEqual([])
    expect(composerPlainText(editor)).toBe('and the sandbox is off now?')

    act(() => endBackendTransition('work', { source: 'switch' }))
    await settle(150)

    expect(onSubmit).not.toHaveBeenCalled()
    expect(send()?.disabled).toBe(false)

    await clickSend()

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe('and the sandbox is off now?')
    expect(onSubmit.mock.calls[0]?.[1]?.fromQueue).toBeUndefined()
  })

  it("a new chat's first message stays in the composer too", async () => {
    beginBackendTransition('work', 'starting', { source: 'switch' })
    const { editor, onSubmit } = mount({ freshDraftKey: '__new__:start', queueSessionKey: null, sessionId: null })

    await typeAndEnter(editor, 'hey can you read updog.com for me')

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('__new__:start')).toEqual([])
    expect(composerPlainText(editor)).toBe('hey can you read updog.com for me')

    act(() => endBackendTransition('work', { source: 'switch' }))
    await settle(150)
    expect(onSubmit).not.toHaveBeenCalled()

    await pressEnter(editor)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[1]?.fromQueue).toBeUndefined()
  })

  it('a reply still running: neither Enter nor Ctrl+Enter queues the message', async () => {
    beginBackendTransition('work', 'stopping', { source: 'switch' })
    const { editor, onSubmit } = mount({ ...EXISTING_CHAT, busy: true })

    await typeAndEnter(editor, 'and then summarize it')
    // Ctrl+Enter queues a follow-up behind a running reply; not across a backend change.
    await act(async () => {
      fireEvent.keyDown(editor, { ctrlKey: true, key: 'Enter' })
    })
    await settle(50)

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-work')).toEqual([])
    expect(composerPlainText(editor)).toBe('and then summarize it')
  })

  it('a slash command still runs at once', async () => {
    beginBackendTransition('work', 'starting', { source: 'switch' })
    const { editor, onSubmit } = mount(EXISTING_CHAT)

    await typeAndEnter(editor, '/status')

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe('/status')
    expect(getQueuedPrompts('stored-work')).toEqual([])
    expect(composerPlainText(editor)).toBe('')
  })

  it('a message queued earlier is not sent to the backend going away', async () => {
    enqueueQueuedPrompt('stored-work', { attachments: [], text: 'queued before the switch' })
    beginBackendTransition('work', 'starting', { source: 'switch' })
    const { editor, onSubmit } = mount(EXISTING_CHAT)

    // An empty Enter sends the next queued message; not now.
    await pressEnter(editor)
    await settle(100)

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-work').map(entry => entry.text)).toEqual(['queued before the switch'])

    act(() => endBackendTransition('work', { source: 'switch' }))

    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
  })

  it('sends at once without a transition', async () => {
    const { editor, onSubmit, send } = mount(EXISTING_CHAT)

    await type(editor, 'hello')
    expect(send()?.disabled).toBe(false)
    await pressEnter(editor)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[1]?.fromQueue).toBeUndefined()
    expect(getQueuedPrompts('stored-work')).toEqual([])
  })

  it("sends at once in another profile's chat while this one starts", async () => {
    beginBackendTransition('work', 'starting', { source: 'switch' })

    const { editor, onSubmit, send } = mount({
      profile: 'default',
      queueSessionKey: 'stored-default',
      sessionId: 'rt-default'
    })

    expect(placeholder(editor)).not.toBe('Starting sandbox…')

    await type(editor, 'hello from default')
    expect(send()?.disabled).toBe(false)
    await pressEnter(editor)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(getQueuedPrompts('stored-default')).toEqual([])
  })
})

describe('a message while the profile backend restarts', () => {
  it('waits in the local queue with the connection down, then goes once when the backend is back', async () => {
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    setGatewayState('connecting')
    const { editor, onSubmit, rerender, send } = mount({ ...EXISTING_CHAT, disabled: true })

    expect(placeholder(editor)).toBe('Restarting sandbox…')

    await type(editor, 'try again')
    expect(send()?.disabled).toBe(false)
    await pressEnter(editor)

    expect(onSubmit).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe('')
    expect(getQueuedPrompts('stored-work').map(entry => [entry.text, entry.serverStatus ?? null])).toEqual([
      ['try again', null]
    ])

    // The socket opens before the transition has ended: still held.
    setGatewayState('open')
    await rerender({ ...EXISTING_CHAT, disabled: false })
    await settle(100)
    expect(onSubmit).not.toHaveBeenCalled()

    act(() => endBackendTransition('work', { source: 'launcher' }))

    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('try again')
    // At once, naming the chat's runtime id from before the restart: the send recovers the session
    // (use-prompt-actions/submit-restarted-legacy-backend.test.tsx).
    expect(onSubmit.mock.calls[0]?.[1]).toMatchObject({
      fromQueue: true,
      sessionId: 'rt-work',
      storedSessionId: 'stored-work'
    })
    expect(getQueuedPrompts('stored-work')).toEqual([])

    await settle(150)
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it("a new chat's first message waits too, and is sent as that chat's first message", async () => {
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    setGatewayState('connecting')
    const fresh = { freshDraftKey: '__new__:restart', queueSessionKey: null, sessionId: null }
    const { editor, onSubmit, rerender } = mount({ ...fresh, disabled: true })

    await typeAndEnter(editor, 'a first message')

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('__new__:restart').map(entry => entry.text)).toEqual(['a first message'])

    setGatewayState('open')
    await rerender({ ...fresh, disabled: false })
    act(() => endBackendTransition('work', { source: 'launcher' }))

    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    const options = onSubmit.mock.calls[0]?.[1]
    // No stored session to resume yet: the send creates the chat.
    expect(options?.storedSessionId).toBeUndefined()
    expect(options?.sessionId).toBeUndefined()
    expect(getQueuedPrompts('__new__:restart')).toEqual([])
  })

  it("a new chat's held first message goes once, also when the chat gets its stored id during that send", async () => {
    // The send creates the chat. session.create announces the stored id and the composer moves onto it, taking
    // the new chat's queue along (use-composer-draft's announced handoff), while the send is still in flight:
    // the message has to leave the queue it is in by then, or it goes a second time once the reply has ended.
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    setGatewayState('connecting')
    const fresh = { freshDraftKey: '__new__:restart', queueSessionKey: null, sessionId: null }
    const { editor, onSubmit, rerender } = mount({ ...fresh, disabled: true })

    let accept: (value: boolean) => void = () => {}
    onSubmit.mockImplementationOnce(() => new Promise<boolean>(resolve => (accept = resolve)))

    await typeAndEnter(editor, 'a first message')
    setGatewayState('open')
    await rerender({ ...fresh, disabled: false })
    act(() => endBackendTransition('work', { source: 'launcher' }))
    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))

    act(() => announceNewSessionDraftKey('stored-new'))
    await rerender({ ...fresh, disabled: false, queueSessionKey: 'stored-new', sessionId: 'rt-new' })
    await act(async () => accept(true))
    await settle(1500)

    expect(getQueuedPrompts('stored-new')).toEqual([])
    expect(getQueuedPrompts('__new__:restart')).toEqual([])
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('a slash command still runs at once', async () => {
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    const { editor, onSubmit } = mount(EXISTING_CHAT)

    await typeAndEnter(editor, '/status')

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe('/status')
    expect(getQueuedPrompts('stored-work')).toEqual([])
  })

  it('an empty Enter does not push the held message out early', async () => {
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    const { editor, onSubmit } = mount(EXISTING_CHAT)

    await typeAndEnter(editor, 'held for the restart')
    await typeAndEnter(editor, '')

    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-work').map(entry => entry.text)).toEqual(['held for the restart'])
  })

  // A restart can take 10-20 s. Holding must not count as failed sends: the queue's retry budget
  // (MAX_AUTO_DRAIN_ATTEMPTS, about 4.5 s of retries) would run out and leave the message stuck with an error.
  it('a restart longer than the queue retry budget still delivers the message, with no stuck notice', async () => {
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    const { editor, onSubmit } = mount(EXISTING_CHAT)

    await typeAndEnter(editor, 'a slow restart')

    // In steps, so React applies each queue retry the moment its timer fires, as it does in the app.
    for (let waited = 0; waited < 6_000; waited += 250) {
      await settle(250)
    }

    act(() => endBackendTransition('work', { source: 'launcher' }))

    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect($notifications.get().some(notice => notice.id === 'composer-queue-stuck')).toBe(false)
  }, 20_000)
})

// The switch as the plugin SDK runs it (sdk/profile-backend.ts), with the window showing the profile.
describe('a message during the plugin switch, end to end', () => {
  function installBridge(overrides: Record<string, ReturnType<typeof vi.fn>>) {
    const bridge = {
      clearProfileBackend: vi.fn(async () => ({ applied: true })),
      confirmClearProfileBackend: vi.fn(async () => ({ confirmed: true, grant: 'g-clear' })),
      confirmSetProfileBackend: vi.fn(async () => ({ confirmed: true, grant: 'g-set' })),
      getConnectionConfig: vi.fn(async () => ({ launcher: null, mode: 'local', remoteUrl: '' })),
      setProfileBackend: vi.fn(async () => ({ applied: true })),
      ...overrides
    }

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

    return bridge
  }

  beforeEach(() => {
    $activeGatewayProfile.set('work')
  })

  it("switch on: a message typed while the plugin's backend starts is sent only by the user, to that backend", async () => {
    let started: (address: { token: string; url: string }) => void = () => undefined

    // Main applies the route and the window redials the profile onto the plugin's backend.
    const bridge = installBridge({
      setProfileBackend: vi.fn(async () => {
        setGatewayState('connecting')

        return { applied: true }
      })
    })

    const switching = setProfileBackend({
      plugin: 'acme',
      prepare: () => new Promise(resolve => (started = resolve)),
      profile: 'work'
    })

    await vi.waitFor(() => expect(backendTransitionFor('work')?.kind).toBe('starting'))
    // This computer's backend still serves the chat while the plugin's starts: the message must not go there.
    const { editor, onSubmit, rerender, send } = mount(EXISTING_CHAT)

    await typeAndEnter(editor, 'hey can you read updog.com for me')
    expect(onSubmit).not.toHaveBeenCalled()

    started({ token: 'tok', url: 'http://127.0.0.1:47911' })
    await act(async () => {
      await switching
    })
    // Applied, but the socket to the new backend is not open yet.
    await rerender({ ...EXISTING_CHAT, disabled: true })
    await pressEnter(editor)
    expect(onSubmit).not.toHaveBeenCalled()

    act(() => setGatewayState('open'))
    await rerender({ ...EXISTING_CHAT, disabled: false })
    await settle(150)

    expect(backendTransitionFor('work')).toBeNull()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-work')).toEqual([])
    expect(composerPlainText(editor)).toBe('hey can you read updog.com for me')
    expect(send()?.disabled).toBe(false)

    await pressEnter(editor)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit.mock.calls[0]?.[0]).toBe('hey can you read updog.com for me')
    expect(bridge.setProfileBackend.mock.invocationCallOrder[0]).toBeLessThan(onSubmit.mock.invocationCallOrder[0]!)
  })

  it("switch off: a message typed while the plugin's backend stops is sent only by the user, afterwards", async () => {
    let stopped: () => void = () => undefined

    const bridge = installBridge({
      clearProfileBackend: vi.fn(async () => {
        setGatewayState('connecting')

        return { applied: true }
      })
    })

    const switching = clearProfileBackend({
      beforeApply: () => new Promise<void>(resolve => (stopped = resolve)),
      plugin: 'acme',
      profile: 'work'
    })

    await vi.waitFor(() => expect(backendTransitionFor('work')?.kind).toBe('stopping'))
    // The plugin's backend still answers until the plugin stops it: the message must not go there.
    const { editor, onSubmit, rerender } = mount(EXISTING_CHAT)

    await typeAndEnter(editor, 'and the sandbox is off now?')
    expect(onSubmit).not.toHaveBeenCalled()

    stopped()
    await act(async () => {
      await switching
    })

    act(() => setGatewayState('open'))
    await rerender({ ...EXISTING_CHAT, disabled: false })
    await settle(150)

    expect(backendTransitionFor('work')).toBeNull()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-work')).toEqual([])
    expect(composerPlainText(editor)).toBe('and the sandbox is off now?')

    await pressEnter(editor)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(bridge.clearProfileBackend.mock.invocationCallOrder[0]).toBeLessThan(onSubmit.mock.invocationCallOrder[0]!)
    expect(onSubmit.mock.calls[0]?.[1]?.fromQueue).toBeUndefined()
  })
})
