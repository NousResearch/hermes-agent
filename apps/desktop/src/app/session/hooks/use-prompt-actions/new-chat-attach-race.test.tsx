import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'

import { usePromptActions } from '.'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

/**
 * Regression: new chat + file attachment + text was silently never sent.
 *
 * The wiring (contrib/wiring.tsx) derives the routed session id / route token
 * from useLocation() and only refreshes the backing refs on RE-RENDER. After
 * createBackendSessionForSend's navigate(), that flush lands asynchronously:
 * microtask-speed submits (text-only / folder) finish the post-create drift
 * check BEFORE the flush, but a file attachment awaits the file.attach WS
 * round-trip, the flush lands mid-await, and a raw-token drift guard reads
 * our own re-home as a user switch → abortForSessionSwitch → silent false →
 * the composer restores the draft and prompt.submit never fires.
 *
 * This test models that exact timing: the route flush and the file.attach
 * response are both macrotasks, the flush queued first.
 */

interface HarnessHandle {
  submitText: (text: string, options?: { attachments?: ComposerAttachment[] }) => Promise<boolean>
}

function Harness({
  activeSessionIdRef,
  createBackendSessionForSend,
  getRoutedStoredSessionId,
  getRouteToken,
  onReady,
  requestGateway,
  selectedStoredSessionIdRef
}: {
  activeSessionIdRef: MutableRefObject<null | string>
  createBackendSessionForSend: () => Promise<null | string>
  getRoutedStoredSessionId: () => null | string
  getRouteToken: () => string
  onReady: (handle: HarnessHandle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  selectedStoredSessionIdRef: MutableRefObject<null | string>
}) {
  const stateRef = useRef({
    awaitingResponse: false,
    busy: false,
    interrupted: true,
    messages: []
  } as never)

  const actions = usePromptActions({
    activeSessionId: null,
    activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef: { current: false },
    createBackendSessionForSend,
    getRoutedStoredSessionId,
    getRuntimeIdForStoredSession: () => null,
    getRouteToken,
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: () => undefined,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState: (_sessionId, updater) => {
      const next = updater(stateRef.current) as never
      stateRef.current = next

      return next
    }
  })

  useEffect(() => {
    onReady({
      submitText: (...args: Parameters<typeof actions.submitText>) =>
        act(async () => actions.submitText(...args)) as Promise<boolean>
    })
  }, [actions.submitText, onReady])

  return null
}

describe('new-chat submit route-flush race', () => {
  afterEach(() => cleanup())

  function setup() {
    const calls: { method: string }[] = []
    // The route ref pair starts on the new-chat route and flushes to the
    // created session's route one macrotask after create returns — exactly
    // like React re-rendering after navigate().
    let routeFlushed = false
    const activeSessionIdRef: MutableRefObject<null | string> = { current: null }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }

    const requestGateway = vi.fn(async (method: string) => {
      calls.push({ method })

      if (method === 'file.attach') {
        // WS round-trip: resolves on a macrotask, AFTER the route flush.
        await new Promise(resolve => setTimeout(resolve, 0))

        return { attached: true, ref_text: '@file:.hermes/desktop-attachments/a.txt', uploaded: false } as never
      }

      return {} as never
    })

    const createBackendSessionForSend = vi.fn(async () => {
      activeSessionIdRef.current = 'rt-new-chat'
      selectedStoredSessionIdRef.current = 'stored-new-chat'
      setTimeout(() => {
        routeFlushed = true
      }, 0)

      return 'rt-new-chat'
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => (routeFlushed ? 'stored-new-chat' : null)}
        getRouteToken={() => (routeFlushed ? '/stored-new-chat::' : '/::')}
        onReady={h => (handle = h)}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
      />
    )

    return { calls, getHandle: () => handle }
  }

  it('text-only first send of a new chat survives (flush lands after the drift check)', async () => {
    const { calls, getHandle } = setup()
    await waitFor(() => expect(getHandle()).not.toBeNull())

    expect(await getHandle()!.submitText('hello new chat')).toBe(true)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(true)
  })

  it('file attachment first send of a new chat still reaches prompt.submit', async () => {
    const { calls, getHandle } = setup()
    await waitFor(() => expect(getHandle()).not.toBeNull())

    const attachments: ComposerAttachment[] = [
      {
        id: 'file:a',
        kind: 'file',
        label: 'a.txt',
        path: '/tmp/a.txt',
        refText: '@file:a.txt'
      }
    ]

    expect(await getHandle()!.submitText('describe this file', { attachments })).toBe(true)
    expect(calls.some(c => c.method === 'prompt.submit')).toBe(true)
  })
})
