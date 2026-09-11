import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import type * as Gateway from '@/store/gateway'
import { setSessionOwnerHint, setSessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { ConnectorTool } from './connector-tool'
import { assistantMessage, stubThreadEnvironment, ThreadRuntime, userMessage } from './test-utils'
import { Thread } from './thread'

const request = vi.hoisted(() => vi.fn())
vi.mock('@/store/gateway', async original => ({
  ...(await original<typeof Gateway>()),
  requestGatewayForAgent: request
}))
stubThreadEnvironment()
afterEach(() => {
  cleanup()
  request.mockReset()
  vi.unstubAllGlobals()
})

it.each([undefined, false, true])(
  'only renders and refreshes owning-session controls when the launch flag is true (%s)',
  async guestOnboardingEnabled => {
    stubThreadEnvironment()
    vi.stubGlobal('hermesDesktop', { ...window.hermesDesktop, guestOnboardingEnabled })
    request.mockResolvedValue({
      available: true,
      connectors: [{ connector: 'gmail', enabled: true, connected: false }]
    })
    setSessionOwnerHint('connector-render-stored', { connectionId: 'source-a', profile: 'default' })

    const tool = {
      type: 'tool-call' as const,
      toolCallId: 'connector-render-call',
      toolName: 'manage_connections',
      args: { action: 'status', connectors: ['gmail'] },
      argsText: '',
      result: { connectors: [{ connector: 'gmail' }] }
    }

    const message = assistantMessage()

    if (message.role !== 'assistant') {
      throw new Error('Expected an assistant message fixture')
    }

    const assistant = {
      ...message,
      content: [
        {
          type: 'tool-call' as const,
          toolCallId: 'before',
          toolName: 'read_file',
          args: { path: 'readme' },
          argsText: '',
          result: { content: 'test' }
        },
        tool,
        {
          type: 'tool-call' as const,
          toolCallId: 'after',
          toolName: 'read_file',
          args: { path: 'notes' },
          argsText: '',
          result: { content: 'test' }
        }
      ]
    }

    const view = {
      ...PRIMARY_SESSION_VIEW,
      $runtimeId: atom<string | null>('connector-render-runtime'),
      $storedId: atom<string | null>('connector-render-stored'),
      $busy: atom(false),
      $messages: atom([{ id: assistant.id, role: 'assistant' as const, parts: assistant.content }])
    }

    render(
      <SessionViewProvider value={view}>
        <ThreadRuntime messages={[userMessage(), assistant]}>
          <Thread />
        </ThreadRuntime>
      </SessionViewProvider>
    )

    if (guestOnboardingEnabled !== true) {
      expect(screen.queryByRole('button', { name: 'Connect' })).toBeNull()
      expect(request).not.toHaveBeenCalled()

      return
    }

    await waitFor(() => expect(screen.getByRole('button', { name: 'Connect' })).toBeTruthy())
    expect(request).toHaveBeenCalledWith(
      'source-a',
      'default',
      'connectors.list',
      { session_id: 'connector-render-runtime' },
      45000
    )
    expect(request.mock.calls.every(([, , method]) => method === 'connectors.list')).toBe(true)
  }
)

it('offers controls for a profile-owned session without a registry connection id', async () => {
  const storedId = 'profile-only-connector'
  setSessions([makeSessionInfo({ id: storedId, profile: 'guide-profile', title: 'Guide' })])
  request.mockResolvedValue({ available: true, connectors: [{ connector: 'gmail', enabled: true, connected: false }] })

  const view = {
    ...PRIMARY_SESSION_VIEW,
    $runtimeId: atom<string | null>('profile-runtime'),
    $storedId: atom<string | null>(storedId)
  }

  render(
    <SessionViewProvider value={view}>
      <ConnectorTool
        addResult={vi.fn()}
        args={{ action: 'status', connectors: ['gmail'] }}
        argsText=""
        respondToApproval={vi.fn()}
        result={{ connectors: [{ connector: 'gmail' }] }}
        resume={vi.fn()}
        status={{ type: 'complete' }}
        toolCallId="profile-call"
        toolName="manage_connections"
        type="tool-call"
      />
    </SessionViewProvider>
  )

  await waitFor(() => expect(screen.getByRole('button', { name: 'Connect' })).toBeTruthy())
  expect(request).toHaveBeenCalledWith(
    null,
    'guide-profile',
    'connectors.list',
    { session_id: 'profile-runtime' },
    45000
  )
})
