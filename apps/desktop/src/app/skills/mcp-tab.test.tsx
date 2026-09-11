import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileScope } from '@/hermes'
import { probeCache } from '@/lib/mcp-probe-cache'
import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'

// Replace only the native editor, keeping the real pane, query cache, profile
// subscriptions, controls and API scope resolver. Its keyed/uncontrolled draft
// contract is the same as CodeMirror's; a mock config hook would hide the race.
vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({ initialValue, onChange }: { initialValue: string; onChange: (text: string) => void }) => (
    <textarea aria-label="mcp.json" defaultValue={initialValue} onChange={event => onChange(event.target.value)} />
  )
}))
vi.mock('@/components/chat/log-tail', () => ({ LogTail: () => null }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const api = vi.fn()
let failReads = false

const configs: Record<string, Record<string, unknown>> = {
  a: { mcp_servers: { old: { command: 'old-server', enabled: false } } },
  b: { mcp_servers: {} },
  pinned: { mcp_servers: {} }
}

beforeEach(() => {
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false } })
  probeCache.clear()
  failReads = false
  vi.stubGlobal('hermesDesktop', { api })
  api.mockImplementation(async (request: { path: string; method?: string; profile?: string }) => {
    if (request.path === '/api/config') {
      if (failReads) {
        throw new Error('Backend unavailable')
      }

      return configs[request.profile ?? 'a'] ?? {}
    }

    if (request.path === '/api/mcp/servers') {
      return { ok: true }
    }

    if (request.path === '/api/mcp/catalog') {
      return { entries: [] }
    }

    return { tools: [], lines: [], ok: true }
  })
  $activeGatewayProfile.set('a')
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  probeCache.clear()
  $activeGatewayProfile.set('default')
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

async function mount(profile?: ProfileScope) {
  const { McpTab } = await import('./mcp-tab')

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <McpTab gateway={null} profile={profile} />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

const editor = () => screen.getByRole('textbox', { name: 'mcp.json' }) as HTMLTextAreaElement
const writes = () => api.mock.calls.map(([request]) => request).filter(request => request.method === 'PUT')

describe('MCP profile ownership', { timeout: 60_000 }, () => {
  it('keeps a pinned draft editable when the foreground profile changes and its refetch fails', async () => {
    const scope = { connectionId: 'local', profile: 'pinned' }
    await mount(scope)
    await screen.findByRole('button', { name: 'New server' })
    const draft = JSON.stringify({ mcpServers: { unity: { url: 'http://localhost:8080/mcp' } } })
    fireEvent.change(editor(), { target: { value: draft } })

    failReads = true
    await act(async () => {
      $activeGatewayProfile.set('b')
    })
    await waitFor(() =>
      expect(queryClient.getQueryState(['hermes-config-record', 'local::pinned'])?.status).toBe('error')
    )
    expect(editor().value).toBe(draft)
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(writes()).toHaveLength(1))
    expect(writes()[0]).toMatchObject({
      ...scope,
      body: { servers: { unity: { url: 'http://localhost:8080/mcp' } } }
    })
  })

  it('never unlocks an ambient profile with another profile’s cached server map after a failed read', async () => {
    await mount()
    await screen.findByRole('textbox', { name: 'mcp.json' })
    await waitFor(() => expect(editor().value).toContain('old-server'))

    failReads = true
    await act(async () => {
      $activeGatewayProfile.set('b')
    })
    await screen.findByText('Backend unavailable')
    expect(screen.queryByRole('textbox', { name: 'mcp.json' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'New server' })).toBeNull()
    expect(writes()).toHaveLength(0)

    failReads = false
    fireEvent.click(screen.getByRole('button', { name: 'Reload MCP' }))
    await screen.findByRole('button', { name: 'New server' })
    expect(editor().value).not.toContain('old-server')
    fireEvent.click(screen.getByRole('button', { name: 'New server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(writes()).toHaveLength(1))
    expect(writes()[0].profile).toBe('b')
    expect(writes()[0].body.servers).not.toHaveProperty('old')
    expect(writes()[0].body.servers).toHaveProperty('my-server')
  })
})
