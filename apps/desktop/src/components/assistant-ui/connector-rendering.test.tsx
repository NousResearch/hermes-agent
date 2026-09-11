import type { ThreadMessage } from '@assistant-ui/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { setSessionOwnerHint } from '@/store/session'

import { assistantMessage, stubThreadEnvironment, ThreadRuntime, userMessage } from './test-utils'
import { Thread } from './thread'

const request = vi.hoisted(() => vi.fn())
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
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

    const assistant = {
      ...assistantMessage(),
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
        <ThreadRuntime messages={[userMessage(), assistant as ThreadMessage]}>
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
