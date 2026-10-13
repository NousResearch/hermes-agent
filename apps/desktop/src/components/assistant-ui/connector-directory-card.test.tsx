import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { ConnectorDirectoryCard } from '@/components/assistant-ui/connector-directory-card'
import { ConnectorTool } from '@/components/assistant-ui/connector-tool'
import { I18nProvider } from '@/i18n'
import { $connectionRequests } from '@/store/connection-request'
import { setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

const SESSION_ID = 'session-1'

const row = (connector: string, connected: boolean) => ({
  connected,
  connection_status: connected ? 'active' : null,
  connector,
  enabled: true,
  gateway_disabled_tools: [],
  status_reason: null
})

const REPLIES = new Map<string, object>([
  ['connectors.accounts', { accounts: [] }],
  ['connectors.catalog', { connectors: [] }],
  [
    'connectors.list',
    { available: true, connectors: [row('gmail', true), row('linear', false), row('notion', false), row('slack', false)] }
  ],
  ['connectors.policy.get', { layers: [] }]
])

function gateway() {
  const request = vi.fn(async (method: string) => REPLIES.get(method) ?? {})
  setSessionOwnerHint(SESSION_ID, { connectionId: 'connection-1', profile: 'default' })
  // SAFETY: the card and its queries call only `request` on the primary socket.
  setPrimaryGateway({ request } as never)
  setPrimaryGatewayConnectionId('connection-1')

  return request
}

function view(): SessionView {
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
    $reasoningEffortPending: atom(false),
    $reasoningEffortWire: atom(''),
    $runtimeId: atom(SESSION_ID),
    $storedId: atom(SESSION_ID),
    $turnStartedAt: atom(null),
    kind: 'primary'
  }
}

function inChat(children: ReactNode) {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <I18nProvider configClient={null} initialLocale="en">
        <SessionViewProvider value={view()}>{children}</SessionViewProvider>
      </I18nProvider>
    </QueryClientProvider>
  )
}

// Each app is a toggle tile; its first line is the app name.
const tileOrder = (container: HTMLElement) =>
  [...container.querySelectorAll('button[aria-pressed]')].map(tile => tile.querySelector('.block')?.textContent)

afterEach(() => {
  cleanup()
  $connectionRequests.set({})
  setPrimaryGateway(null)
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.clearAllMocks()
})

describe('ConnectorDirectoryCard', () => {
  it('pins Suggested rows first in the given order, then the most used apps, and Continue reports the result', async () => {
    gateway()
    const onContinue = vi.fn()

    const { container } = inChat(<ConnectorDirectoryCard onContinue={onContinue} suggested={['slack', 'linear']} />)

    await waitFor(() => expect(tileOrder(container)).toEqual(['Slack', 'Linear', 'Gmail', 'Notion']))
    expect(screen.getByText('Suggested')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))

    expect(onContinue).toHaveBeenCalledWith({ connected: ['gmail'], newly_connected: [], suggested: ['slack', 'linear'] })
  })

  it('collapses to one receipt line naming what the user connected once the call returns', () => {
    const args = { action: 'display', connectors: ['slack'] }

    const props: ToolCallMessagePartProps = {
      addResult: vi.fn(),
      args,
      argsText: JSON.stringify(args),
      isError: false,
      respondToApproval: vi.fn(),
      result: JSON.stringify({ connected: ['gmail', 'slack'], newly_connected: ['slack'], suggested: ['slack'] }),
      resume: vi.fn(),
      status: { type: 'complete' },
      toolCallId: 'display-call-1',
      toolName: 'manage_connections',
      type: 'tool-call'
    }

    const { container } = inChat(<ConnectorTool {...props} />)

    expect(screen.getByText('Connected: Slack')).toBeTruthy()
    expect(tileOrder(container)).toEqual([])
  })
})
