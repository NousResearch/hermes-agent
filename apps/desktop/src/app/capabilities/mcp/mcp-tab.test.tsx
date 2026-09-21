// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { PropsWithChildren } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type * as HermesApi from '@/hermes'
import { probeCache, probeKey } from '@/lib/mcp-probe-cache'

const storeMocks = vi.hoisted(() => {
  function atom<T>(initial: T) {
    let value = initial
    const listeners = new Set<(value: T) => void>()

    return {
      get: () => value,
      get value() {
        return value
      },
      set: (next: T) => {
        value = next
        listeners.forEach(listener => listener(value))
      },
      listen: (listener: (value: T) => void) => {
        listeners.add(listener)

        return () => listeners.delete(listener)
      }
    }
  }

  return {
    activeProfile: atom('default'),
    activeSessionId: atom<null | string>(null),
    connection: atom<null | { baseUrl: string; mode: 'local' | 'remote'; profile?: string }>({
      baseUrl: 'https://connection-a.example',
      mode: 'remote',
      profile: 'default'
    })
  }
})

const configRecord = vi.hoisted(() => ({
  mcp_servers: { shared: { enabled: true, url: 'https://mcp.example.test/shared' } }
}))

const testMcpServer = vi.fn()

const mcpText = new Proxy<Record<string, unknown>>(
  {
    authenticatedMessage: () => '',
    authenticatedTitle: 'Authenticated',
    capabilitySummary: (count: number) => `tools:${count}`,
    catalogEnvPrompt: () => '',
    catalogInstallFailed: () => '',
    catalogInstallStarted: () => '',
    costTokens: () => '',
    disableTool: () => '',
    enableTool: () => '',
    importConfirm: 'Import',
    importConfirmMany: () => '',
    importNoMatch: 'No match',
    removeFailed: () => '',
    savedMessage: () => '',
    savedTitle: 'Saved',
    usage30d: () => ''
  },
  {
    get: (target, property: string) => target[property] ?? property
  }
)

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: vi.fn().mockResolvedValue({ running: false, exit_code: 0 }),
  getLogs: vi.fn().mockResolvedValue({ lines: [] }),
  getMcpCatalog: vi.fn().mockResolvedValue({ entries: [] }),
  getUsageAnalytics: vi.fn().mockResolvedValue({ tools: [] }),
  installMcpCatalogEntry: vi.fn().mockResolvedValue({}),
  saveMcpServers: vi.fn().mockResolvedValue({ ok: true }),
  testMcpServer
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { save: 'Save', saving: 'Saving' },
      settings: { mcp: mcpText },
      skills: { loading: 'Loading' }
    }
  })
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: storeMocks.activeProfile,
  normalizeProfileKey: (profile: string | null | undefined) => profile?.trim() || 'default'
}))

vi.mock('@/store/session', () => ({
  $activeSessionId: storeMocks.activeSessionId,
  $connection: storeMocks.connection
}))

vi.mock('../../hooks/use-config-record', () => ({
  hermesConfigCacheWriter: () => vi.fn(),
  useHermesConfigRecord: () => ({
    data: configRecord,
    dataUpdatedAt: 1,
    error: null,
    errorUpdatedAt: 0,
    isError: false,
    isLoading: false,
    refetch: vi.fn()
  })
}))

vi.mock('../../hooks/use-on-profile-switch', () => ({ useOnProfileSwitch: () => undefined }))
vi.mock('../../settings/use-deep-link-highlight', () => ({ useDeepLinkHighlight: () => null }))

vi.mock('@/components/chat/code-editor', () => ({ JsonDocumentEditor: () => null }))
vi.mock('@/components/chat/json-document-editor', () => ({ JsonDocumentEditor: () => null }))
vi.mock('@/components/chat/log-tail', () => ({ LogTail: () => null }))
vi.mock('@/components/page-loader', () => ({ PageLoader: () => null }))
vi.mock('@/components/ui/avatar-chip', () => ({ AvatarChip: () => null }))
vi.mock('@/components/ui/button', () => ({
  Button: ({ children, ...props }: PropsWithChildren<Record<string, unknown>>) => <button {...props}>{children}</button>
}))
vi.mock('@/components/ui/codicon', () => ({ Codicon: () => null }))
vi.mock('@/components/ui/error-state', () => ({ ErrorBanner: () => null }))
vi.mock('@/components/ui/input', () => ({ Input: (props: Record<string, unknown>) => <input {...props} /> }))
vi.mock('@/components/ui/popover', () => ({
  Popover: ({ children }: PropsWithChildren) => <>{children}</>,
  PopoverContent: ({ children }: PropsWithChildren) => <>{children}</>,
  PopoverTrigger: ({ children }: PropsWithChildren) => <>{children}</>
}))
vi.mock('@/components/ui/switch', () => ({
  Switch: ({
    checked,
    onCheckedChange,
    ...props
  }: {
    checked?: boolean
    onCheckedChange?: (value: boolean) => void
  }) => (
    <input
      {...props}
      checked={checked}
      onChange={event => onCheckedChange?.(event.currentTarget.checked)}
      type="checkbox"
    />
  )
}))
vi.mock('@/components/ui/text-tab', () => ({
  TextTab: ({ children, ...props }: PropsWithChildren) => <button {...props}>{children}</button>
}))
vi.mock('@/components/ui/textarea', () => ({ Textarea: (props: Record<string, unknown>) => <textarea {...props} /> }))
vi.mock('@/components/ui/tooltip', () => ({ Tip: ({ children }: PropsWithChildren) => <>{children}</> }))

vi.mock('../../master-detail', () => ({
  DetailPane: ({ children }: PropsWithChildren) => <>{children}</>,
  ICON_BUTTON: '',
  MASTER_DETAIL_WIDE_COLS: '',
  MasterDetail: ({ children }: PropsWithChildren) => <>{children}</>
}))
vi.mock('../../overlays/panel', () => ({
  PanelAddButton: () => null,
  PanelEmpty: () => null
}))
vi.mock('../../settings/helpers', () => ({ prettyName: (value: string) => value }))

const { McpTab } = await import('./mcp-tab')

describe('McpTab owner changes', () => {
  beforeEach(() => {
    probeCache.clear()
    testMcpServer.mockReset()
    testMcpServer.mockResolvedValue({ ok: true, tools: [{ name: 'shared-tool', description: '' }] })
    setApiRequestConnection('connection-a')
    setApiRequestProfile('default')
    storeMocks.activeProfile.set('default')
    storeMocks.connection.set({ baseUrl: 'https://connection-a.example', mode: 'remote', profile: 'default' })
  })

  afterEach(() => {
    cleanup()
    probeCache.clear()
    setApiRequestConnection(null)
    setApiRequestProfile(null)
  })

  it('clears the ref synchronously before probing the same server on connection B', async () => {
    const server = { enabled: true, url: 'https://mcp.example.test/shared' }
    const cachedA = { ok: true, tools: [{ name: 'a-tool', description: '' }] }
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    probeCache.set(probeKey('shared', server, 'connection-a::default'), { at: Date.now(), result: cachedA })

    let view!: ReturnType<typeof render>
    await act(async () => {
      view = render(
        <QueryClientProvider client={queryClient}>
          <McpTab gateway={null} profile="default" />
        </QueryClientProvider>
      )
    })

    await waitFor(() => expect(window.document.body.textContent).toContain('tools:1'))
    await waitFor(() => expect(testMcpServer).not.toHaveBeenCalled())

    setApiRequestConnection('connection-b')

    await act(async () => {
      view.rerender(
        <QueryClientProvider client={queryClient}>
          <McpTab gateway={null} profile="default" />
        </QueryClientProvider>
      )
    })

    await waitFor(() => {
      expect(testMcpServer).toHaveBeenCalledWith('shared', {
        connectionId: 'connection-b',
        priority: 'foreground',
        profile: 'default'
      })
      expect(probeCache.has(probeKey('shared', server, 'connection-b::default'))).toBe(true)
    })

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })
  })
})
