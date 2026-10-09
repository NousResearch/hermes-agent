// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { createElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getMcpCatalog, saveMcpServers, setMcpServerEnabledFor } from '@/hermes'
import { notifyError } from '@/store/notifications'
import { queryClient } from '@/lib/query-client'

import { useMcpServers } from './use-mcp-servers'

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  runProbe: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getHermesConfigRecord: mocks.getConfig,
  getMcpCatalog: vi.fn().mockResolvedValue({ entries: [] }),
  peekConfigReadOrigin: vi.fn(() => null),
  profileScopeKey: vi.fn(() => 'test-scope'),
  retainConfigReadOrigin: vi.fn((record: unknown) => record),
  saveMcpServers: vi.fn().mockResolvedValue({ ok: true }),
  setMcpServerEnabledFor: vi.fn().mockResolvedValue({ ok: true })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        mcp: {
          authenticatedMessage: () => '',
          authenticatedTitle: '',
          invalidJson: '',
          reloadFailed: 'Reload failed',
          removeFailed: '',
          savedMessage: () => '',
          savedTitle: '',
          saveFailed: 'Could not save MCP servers'
        }
      }
    }
  })
}))

vi.mock('@/lib/mcp-dashboard-oauth', () => ({
  completeMcpDesktopOAuth: vi.fn()
}))

vi.mock('@/lib/mcp-cost', () => ({
  estimateServerTokens: vi.fn(() => null),
  serverUsageCount: vi.fn(() => null)
}))

vi.mock('@/lib/mcp-probe-cache', () => ({
  NEEDS_AUTH_RE: /needs auth/i,
  probeCache: { get: vi.fn(), set: vi.fn() },
  probeKey: vi.fn(() => 'probe')
}))

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom(null)
}))

vi.mock('@/store/free-tier', () => ({
  $freeTierStatus: atom(null)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom(null),
  normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default'
}))

vi.mock('@/store/session', () => ({
  $activeSessionId: atom(null)
}))

vi.mock('./use-mcp-draft', () => ({
  useMcpDraft: () => ({
    cursor: 0,
    dirty: false,
    draft: '',
    patchDraft: vi.fn(),
    reset: vi.fn(),
    resetDraft: vi.fn(),
    setCursor: vi.fn()
  })
}))

vi.mock('./use-mcp-probes', () => ({
  useMcpProbes: () => ({
    costFor: () => ({ tokens: null, uses: null }),
    probes: {},
    resetForProfileSwitch: vi.fn(),
    retainProbes: vi.fn(),
    runProbe: mocks.runProbe,
    setProbe: vi.fn(),
    toolCounts: {},
    usageByServer: {}
  })
}))

afterEach(() => {
  cleanup()
  queryClient.clear()
  window.localStorage.clear()
  vi.clearAllMocks()
})

const wrapper = ({ children }: { children: React.ReactNode }) =>
  createElement(QueryClientProvider, { client: queryClient }, children)

const CONFIG = { mcp_servers: { alpha: { command: 'npx', args: ['-y', 'alpha'] } } }

async function renderController() {
  mocks.getConfig.mockResolvedValue(CONFIG)

  const { result } = renderHook(() => useMcpServers({ gateway: null }), { wrapper })

  await waitFor(() => expect(result.current.configLoading).toBe(false))
  await waitFor(() => expect(result.current.profilePending).toBe(false))

  return result
}

describe('setServerEnabled', () => {
  it('disables through the per-server endpoint, not the whole-map save', async () => {
    const { result } = await renderController()

    await act(async () => {
      await result.current.setServerEnabled('alpha', false)
    })

    expect(setMcpServerEnabledFor).toHaveBeenCalledWith('alpha', false, undefined)
    expect(saveMcpServers).not.toHaveBeenCalled()
    expect(result.current.servers.alpha.enabled).toBe(false)
    expect(mocks.runProbe).not.toHaveBeenCalled()
  })

  it('re-enables by dropping the enabled key and probes the server', async () => {
    const { result } = await renderController()

    await act(async () => {
      await result.current.setServerEnabled('alpha', false)
    })
    await act(async () => {
      await result.current.setServerEnabled('alpha', true)
    })

    expect(setMcpServerEnabledFor).toHaveBeenLastCalledWith('alpha', true, undefined)
    expect('enabled' in result.current.servers.alpha).toBe(false)
    expect(mocks.runProbe).toHaveBeenCalledWith('alpha')
  })

  it('does not paint the cache or stay silent when the backend rejects the toggle', async () => {
    const { result } = await renderController()

    vi.mocked(setMcpServerEnabledFor).mockResolvedValueOnce({ ok: false })

    await act(async () => {
      await result.current.setServerEnabled('alpha', false)
    })

    expect(notifyError).toHaveBeenCalledTimes(1)
    expect('enabled' in result.current.servers.alpha).toBe(false)
    expect(saveMcpServers).not.toHaveBeenCalled()
  })

  it('skips servers the config does not know', async () => {
    const { result } = await renderController()

    await act(async () => {
      await result.current.setServerEnabled('ghost', true)
    })

    expect(setMcpServerEnabledFor).not.toHaveBeenCalled()
  })
})
