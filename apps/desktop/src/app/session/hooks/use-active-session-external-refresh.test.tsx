import { cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { SessionInfo } from '@/types/hermes'

import { sameSessionRefreshRows, useActiveSessionExternalRefresh } from './use-active-session-external-refresh'

interface HarnessProps {
  activeSessionId: string | null
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: string | null,
    runtimeSessionId?: string | null
  ) => Promise<void>
  messagingSessions: SessionInfo[]
  selectedStoredSessionId: string | null
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  sessions: SessionInfo[]
}

function Harness(props: HarnessProps) {
  useActiveSessionExternalRefresh(props)

  return null
}

function session(over: Partial<SessionInfo>): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    id: 'session-1',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...over
  }
}

function clientSessionState(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return {
    awaitingResponse: false,
    branch: '',
    busy: false,
    cwd: '',
    fast: false,
    interrupted: false,
    messages: [],
    model: '',
    needsInput: false,
    pendingBranchGroup: null,
    personality: '',
    provider: '',
    reasoningEffort: '',
    sawAssistantPayload: false,
    serviceTier: '',
    storedSessionId: 'session-1',
    streamId: null,
    turnStartedAt: null,
    yolo: false,
    ...over
  }
}

function message(id: string): ClientSessionState['messages'][number] {
  return { id, parts: [{ text: id, type: 'text' }], role: 'user' }
}

describe('useActiveSessionExternalRefresh', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('treats externally appended messaging metadata as a changed row', () => {
    const before = [
      session({ id: 'signal-1', last_active: 10, message_count: 2, preview: 'old', source: 'signal', title: 'setup' })
    ]

    const after = [
      session({ id: 'signal-1', last_active: 20, message_count: 4, preview: 'new', source: 'signal', title: 'setup' })
    ]

    expect(sameSessionRefreshRows(before, after)).toBe(false)
  })

  it('hydrates the open messaging session when its refreshed row changes', () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-1', clientSessionState({ storedSessionId: 'signal-1' })]])
    }

    const { rerender } = render(
      <Harness
        activeSessionId="runtime-1"
        hydrateFromStoredSession={hydrateFromStoredSession}
        messagingSessions={[
          session({ id: 'signal-1', last_active: 10, message_count: 2, preview: 'old', source: 'signal' })
        ]}
        selectedStoredSessionId="signal-1"
        sessions={[]}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    expect(hydrateFromStoredSession).not.toHaveBeenCalled()

    rerender(
      <Harness
        activeSessionId="runtime-1"
        hydrateFromStoredSession={hydrateFromStoredSession}
        messagingSessions={[
          session({ id: 'signal-1', last_active: 20, message_count: 4, preview: 'new', source: 'signal' })
        ]}
        selectedStoredSessionId="signal-1"
        sessions={[]}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    expect(hydrateFromStoredSession).toHaveBeenCalledTimes(1)
    expect(hydrateFromStoredSession).toHaveBeenCalledWith(1, 'signal-1', 'runtime-1')
  })

  it('does not hydrate a busy local runtime from an external metadata refresh', () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-1', clientSessionState({ busy: true, storedSessionId: 'signal-1' })]])
    }

    const { rerender } = render(
      <Harness
        activeSessionId="runtime-1"
        hydrateFromStoredSession={hydrateFromStoredSession}
        messagingSessions={[session({ id: 'signal-1', last_active: 10, message_count: 2, source: 'signal' })]}
        selectedStoredSessionId="signal-1"
        sessions={[]}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    rerender(
      <Harness
        activeSessionId="runtime-1"
        hydrateFromStoredSession={hydrateFromStoredSession}
        messagingSessions={[session({ id: 'signal-1', last_active: 20, message_count: 4, source: 'signal' })]}
        selectedStoredSessionId="signal-1"
        sessions={[]}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })

  it('does not hydrate when refreshed metadata does not add persisted messages', () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)

    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([
        ['runtime-1', clientSessionState({ messages: [message('m1'), message('m2')], storedSessionId: 'signal-1' })]
      ])
    }

    const { rerender } = render(
      <Harness
        activeSessionId="runtime-1"
        hydrateFromStoredSession={hydrateFromStoredSession}
        messagingSessions={[session({ id: 'signal-1', last_active: 10, message_count: 2, source: 'signal' })]}
        selectedStoredSessionId="signal-1"
        sessions={[]}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    rerender(
      <Harness
        activeSessionId="runtime-1"
        hydrateFromStoredSession={hydrateFromStoredSession}
        messagingSessions={[
          session({ id: 'signal-1', last_active: 20, message_count: 2, source: 'signal', title: 'renamed' })
        ]}
        selectedStoredSessionId="signal-1"
        sessions={[]}
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )

    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })
})
