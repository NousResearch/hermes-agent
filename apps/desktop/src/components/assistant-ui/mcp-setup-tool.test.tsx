import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { McpSetupPending, McpSetupTool } from '@/components/assistant-ui/mcp-setup-tool'
import { I18nProvider } from '@/i18n'
import { $connectionRequests, type ConnectionRequest, setConnectionRequest } from '@/store/connection-request'

const SESSION_ID = 'session-1'

const REQUEST: ConnectionRequest = {
  deadlineAt: 1_800_000_000,
  opId: 'operation-1',
  toolCallId: 'mcp-call-1',
  sessionId: SESSION_ID,
  settled: false,
  settledBy: null,
  targets: [
    { action: 'install', connectUrl: null, detail: '', kind: 'mcp', name: 'linear', state: 'pending', tools: [] },
    { action: 'install', connectUrl: null, detail: '', kind: 'mcp', name: 'postgres', state: 'pending', tools: [] }
  ]
}

const ARGS = {
  action: 'install',
  connectors: [
    { mcp: true, name: 'linear' },
    { mcp: true, name: 'postgres' }
  ]
}

function props(result?: ToolCallMessagePartProps['result']): ToolCallMessagePartProps {
  return {
    addResult: vi.fn(),
    args: ARGS,
    argsText: JSON.stringify(ARGS),
    isError: false,
    respondToApproval: vi.fn(),
    result,
    resume: vi.fn(),
    status: result === undefined ? { type: 'running' } : { type: 'complete' },
    toolCallId: 'mcp-call-1',
    toolName: 'manage_connections',
    type: 'tool-call'
  }
}

function view(sessionId: string): SessionView {
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
    $runtimeId: atom(sessionId),
    $storedId: atom(sessionId),
    $turnStartedAt: atom(null),
    kind: 'primary'
  }
}

// The live gate (message still running) is assistant-ui state; the pending card renders below it.
function renderTool(result?: ToolCallMessagePartProps['result']) {
  const Card = result === undefined ? McpSetupPending : McpSetupTool

  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SessionViewProvider value={view(SESSION_ID)}>
        <Card {...props(result)} />
      </SessionViewProvider>
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  $connectionRequests.set({})
  vi.clearAllMocks()
})

describe('the MCP setup card', () => {
  it('shows one row per target with its verb, and Continue below', () => {
    setConnectionRequest(REQUEST)

    renderTool()

    expect(screen.getByText('Add MCP servers')).toBeTruthy()
    expect(screen.getByText('Linear', { selector: 'span' })).toBeTruthy()
    expect(screen.getByText('Postgres', { selector: 'span' })).toBeTruthy()
    expect(screen.getAllByRole('button', { name: 'Install' })).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'Continue' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Not now' })).toBeNull()
  })

  it('lists every target once settled, in the same three words as the connector card', () => {
    renderTool({
      settled_by: 'continue',
      status: 'settled',
      targets: [
        { action: 'install', kind: 'mcp', name: 'linear', state: 'connected', tools: ['a', 'b'] },
        { action: 'install', detail: 'catalog write failed', kind: 'mcp', name: 'postgres', state: 'not_connected' }
      ]
    })

    expect(screen.getByText('Installed Linear · 2 tools')).toBeTruthy()
    expect(screen.getByText('Not connected')).toBeTruthy()
    expect(screen.queryByText(/catalog write failed/)).toBeNull()
    expect(screen.queryAllByRole('button').filter(button => !button.hasAttribute('disabled'))).toHaveLength(0)
  })
})
