import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { McpSetupPending, McpSetupTool } from '@/components/assistant-ui/mcp-setup-tool'
import { I18nProvider } from '@/i18n'
import {
  $connectionRequests,
  type ConnectionRequest,
  type ConnectionTarget,
  setConnectionRequest
} from '@/store/connection-request'
import { $gateway, setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { setSessionOwnerHint } from '@/store/session'

const SESSION_ID = 'session-1'

const LINEAR: ConnectionTarget = {
  action: 'install',
  connectUrl: null,
  connectionId: '',
  detail: '',
  discoveryError: null,
  instructions: null,
  kind: 'mcp',
  name: 'linear',
  requiredEnv: [],
  state: 'pending',
  tools: []
}

const REQUEST: ConnectionRequest = {
  deadlineAt: 1_800_000_000,
  opId: 'operation-1',
  seq: 0,
  toolCallId: 'mcp-call-1',
  sessionId: SESSION_ID,
  settled: false,
  settledBy: null,
  targets: [LINEAR, { ...LINEAR, name: 'postgres' }]
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
    $reasoningEffortWire: atom(''),
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
  $gateway.set(null)
  setPrimaryGateway(null)
  vi.clearAllMocks()
})

describe('the MCP setup card', () => {
  it('opens required details from the row action and sends the approved environment', async () => {
    const rpc = vi.fn().mockResolvedValue({ status: 'ok', settled: false })

    const target = {
      ...LINEAR,
      instructions: 'Create a Linear API key.',
      requiredEnv: [{ default: 'workspace', name: 'LINEAR_TEAM', prompt: 'Team', required: true, secret: false }]
    }

    setSessionOwnerHint(SESSION_ID, { connectionId: 'local', profile: 'default' })
    // SAFETY: the card calls only `request`; the rest of the client is never touched in this test.
    setPrimaryGateway({ request: rpc } as never)
    setPrimaryGatewayConnectionId('local')
    setConnectionRequest({ ...REQUEST, targets: [target] })

    renderTool()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))

    expect(screen.getByText('Set up Linear')).toBeTruthy()
    expect(screen.getByText('Create a Linear API key.')).toBeTruthy()
    expect(screen.getByLabelText('Team').getAttribute('value')).toBe('workspace')
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
    await waitFor(() => expect(rpc).toHaveBeenCalledTimes(1))
    expect(rpc).toHaveBeenCalledWith('connection.respond', {
      op_id: 'operation-1',
      owner: { session_id: SESSION_ID, type: 'session' },
      result: { targets: [{ env: { LINEAR_TEAM: 'workspace' }, name: 'linear', status: 'approved' }] }
    })
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
