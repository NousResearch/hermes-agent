import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import type { OAuthPollResponse, ToolsetConfig } from '@/types/hermes'

import { ToolsetConfigPanel } from './toolset-config-panel'

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(() => 'oauth-notification'),
  dismissNotification: vi.fn(),
  notifyError: vi.fn()
}))

const provider = 'OpenAI Codex OAuth'

const config: ToolsetConfig = {
  name: 'stt',
  has_category: true,
  active_provider: null,
  providers: [
    {
      name: provider,
      badge: 'subscription',
      tag: 'ChatGPT/Codex dictation',
      env_vars: [],
      post_setup: null,
      auth_provider: 'openai-codex',
      requires_nous_auth: false,
      is_active: false,
      status: 'needs_auth'
    }
  ]
}

const api = vi.fn<(request: HermesApiRequest) => Promise<unknown>>()
const poll = vi.fn<(request: HermesApiRequest) => Promise<OAuthPollResponse>>()

function panel(connectionId: string) {
  return (
    <MemoryRouter>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ToolsetConfigPanel profile={{ connectionId, profile: 'coder' }} toolset="stt" />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  setApiRequestConnection('ambient-gateway')
  setApiRequestProfile('coder')
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api, openExternal: vi.fn().mockResolvedValue(undefined) }
  })
  poll.mockImplementation(async request => ({ status: 'approved', session_id: `${request.connectionId}-session` }))
  api.mockImplementation(async request => {
    if (request.path === '/api/tools/toolsets/stt/config') {
      return config
    }

    if (request.path.endsWith('/start')) {
      return {
        flow: 'device_code',
        session_id: `${request.connectionId}-session`,
        user_code: 'CODEX-1234',
        verification_url: 'https://auth.openai.com/device',
        poll_interval: 1,
        expires_in: 900
      }
    }

    if (request.path.includes('/poll/')) {
      return poll(request)
    }

    if (request.path === '/api/tools/toolsets/stt/provider' || request.method === 'DELETE') {
      return { ok: true, name: 'stt', provider }
    }

    throw new Error(`Unexpected request: ${request.path}`)
  })
})

afterEach(() => {
  cleanup()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.clearAllMocks()
})

describe('toolset OAuth gateway ownership through the real API helpers', () => {
  it.each(['local', 'remote-gateway'])(
    'authenticates and selects on %s despite an identical ambient profile',
    async connectionId => {
      render(panel(connectionId))
      fireEvent.click(await screen.findByRole('button', { name: /Use this backend/ }))

      await waitFor(
        () =>
          expect(api).toHaveBeenCalledWith({
            connectionId,
            profile: 'coder',
            path: '/api/tools/toolsets/stt/provider',
            method: 'PUT',
            body: { provider }
          }),
        { timeout: 3000 }
      )
      expect(api).toHaveBeenCalledWith({
        connectionId,
        profile: 'coder',
        path: '/api/providers/oauth/openai-codex/start',
        method: 'POST',
        body: { activate_provider: false }
      })
      expect(poll).toHaveBeenCalledWith({
        connectionId,
        profile: 'coder',
        path: `/api/providers/oauth/openai-codex/poll/${connectionId}-session`
      })
      expect(
        api.mock.calls.every(([request]) => request.connectionId === connectionId && request.profile === 'coder')
      ).toBe(true)
    }
  )

  it('cancels each owner on a same-name gateway switch and unmount, ignoring both late approvals', async () => {
    const completions = new Map<string, (response: OAuthPollResponse) => void>()

    poll.mockImplementation(request => new Promise(resolve => completions.set(request.connectionId!, resolve)))
    const rendered = render(panel('gateway-a'))

    fireEvent.click(await screen.findByRole('button', { name: /Use this backend/ }))
    await waitFor(() => expect(completions.has('gateway-a')).toBe(true), { timeout: 3000 })
    rendered.rerender(panel('gateway-b'))
    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        connectionId: 'gateway-a',
        profile: 'coder',
        path: '/api/providers/oauth/sessions/gateway-a-session',
        method: 'DELETE'
      })
    )

    fireEvent.click(await screen.findByRole('button', { name: /Use this backend/ }))
    await waitFor(() => expect(completions.has('gateway-b')).toBe(true), { timeout: 3000 })
    rendered.unmount()
    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        connectionId: 'gateway-b',
        profile: 'coder',
        path: '/api/providers/oauth/sessions/gateway-b-session',
        method: 'DELETE'
      })
    )

    await act(async () => {
      for (const [connectionId, complete] of completions) {
        complete({ status: 'approved', session_id: `${connectionId}-session` })
      }
    })
    expect(api.mock.calls.some(([request]) => request.path === '/api/tools/toolsets/stt/provider')).toBe(false)
    expect(api.mock.calls.some(([request]) => request.connectionId === 'ambient-gateway')).toBe(false)
  })
})
