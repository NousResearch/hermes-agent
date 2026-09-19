import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { MESSAGE_PARTS_COMPONENTS } from '@/components/assistant-ui/thread/message-parts'
import { I18nProvider } from '@/i18n'
import { $connectionRequests, type ConnectionRequest, setConnectionRequest } from '@/store/connection-request'
import { $gateway } from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

// jsdom has no preload bridge, so isOnboardingEnabled() is false here: the card must not depend on it.
// Runtime id (events, operation store) and stored id (owner hints) differ in the app.
const SESSION_ID = 'runtime-1'
const STORED_ID = 'stored-1'
const OWNER = { connectionId: 'connection-1', profile: 'default' }

const REQUEST: ConnectionRequest = {
  deadlineAt: 1_800_000_000,
  opId: 'operation-1',
  toolCallId: 'connector-call-1',
  sessionId: SESSION_ID,
  settled: false,
  settledBy: null,
  targets: [
    {
      action: 'connect',
      connectUrl: 'https://connect.example/gmail',
      detail: '',
      kind: 'connector',
      name: 'gmail',
      state: 'initiated',
      tools: []
    },
    {
      action: 'connect',
      connectUrl: 'https://connect.example/notion',
      detail: '',
      kind: 'connector',
      name: 'notion',
      state: 'initiated',
      tools: []
    }
  ]
}

function props(): ToolCallMessagePartProps {
  const args = { action: 'connect', connectors: ['gmail', 'notion'] }

  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId: 'connector-call-1',
    toolName: 'manage_connections',
    type: 'tool-call'
  }
}

function view(sessionId: string, storedId: string): SessionView {
  return {
    $awaitingResponse: atom(false),
    $busy: atom(false),
    $cwd: atom(''),
    $fast: atom(false),
    $lastVisibleIsUser: atom(false),
    $messages: atom([]),
    $messagesEmpty: atom(false),
    $model: atom(''),
    $provider: atom(''),
    $reasoningEffort: atom(''),
    $reasoningEffortWire: atom(''),
    $runtimeId: atom(sessionId),
    $storedId: atom(storedId),
    $turnStartedAt: atom(null),
    kind: 'primary'
  }
}

afterEach(() => {
  cleanup()
  $connectionRequests.set({})
  $gateway.set(null)
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.clearAllMocks()
})

describe('manage_connections routing outside guided onboarding', () => {
  it('renders the operation card for a plain chat session', async () => {
    const Fallback = MESSAGE_PARTS_COMPONENTS.tools.Fallback
    setSessionOwnerHint(STORED_ID, OWNER)
    setConnectionRequest(REQUEST)
    // SAFETY: the card reads only `request` off the client in this test.
    $gateway.set({ request: vi.fn().mockResolvedValue({ status: 'ok' }) } as never)

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <SessionViewProvider value={view(SESSION_ID, STORED_ID)}>
          <Fallback {...props()} />
        </SessionViewProvider>
      </I18nProvider>
    )

    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: 'Connect' })).toHaveLength(2)
    })
    expect(screen.queryByText(/running manage connections/i)).toBeNull()
  })
})
