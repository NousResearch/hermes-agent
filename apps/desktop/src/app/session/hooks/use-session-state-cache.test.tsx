import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $busy, $messages } from '@/store/session'

import { useSessionStateCache } from './use-session-state-cache'

type SessionStateCacheApi = ReturnType<typeof useSessionStateCache>

function assistantMessage(id: string, text: string): ChatMessage {
  return {
    id,
    parts: [assistantTextPart(text)],
    role: 'assistant'
  }
}

function SessionStateCacheHarness({
  onReady,
  setAwaitingResponse,
  setBusy,
  setMessages
}: {
  onReady: (api: SessionStateCacheApi) => void
  setAwaitingResponse: (awaiting: boolean) => void
  setBusy: (busy: boolean) => void
  setMessages: (messages: ChatMessage[]) => void
}) {
  const busyRef = useRef(false)

  const api = useSessionStateCache({
    activeSessionId: 'session-1',
    busyRef,
    selectedStoredSessionId: null,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  useEffect(() => {
    onReady(api)
  }, [api, onReady])

  return null
}

async function waitForRaf() {
  await act(async () => {
    await new Promise(resolve => window.setTimeout(resolve, 0))
  })
}

describe('useSessionStateCache', () => {
  beforeEach(() => {
    $busy.set(false)
    $messages.set([])
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(performance.now()), 0)
    )
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
  })

  afterEach(() => {
    cleanup()
    $busy.set(false)
    $messages.set([])
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('does not publish messages when an active-session flush is equivalent to the current view', async () => {
    const messages = [assistantMessage('assistant-1', 'Already rendered response')]
    const setAwaitingResponse = vi.fn()
    const setBusy = vi.fn()
    const setMessages = vi.fn()
    let api: SessionStateCacheApi | undefined

    $messages.set(messages)

    render(
      <SessionStateCacheHarness
        onReady={nextApi => {
          api = nextApi
        }}
        setAwaitingResponse={setAwaitingResponse}
        setBusy={setBusy}
        setMessages={setMessages}
      />
    )

    if (!api) {
      throw new Error('Session state cache harness did not initialize')
    }

    const sessionCacheApi = api as SessionStateCacheApi

    act(() => {
      sessionCacheApi.syncSessionStateToView('session-1', {
        ...createClientSessionState(null, messages),
        awaitingResponse: true,
        busy: true
      })
    })
    await waitForRaf()

    expect(setBusy).toHaveBeenCalledWith(true)
    expect(setAwaitingResponse).toHaveBeenCalledWith(true)
    expect(setMessages).not.toHaveBeenCalled()
  })
})
