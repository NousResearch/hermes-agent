/**
 * Shared harness for the New Bot dialog suites (create-agent-*.test.tsx).
 *
 * `vi.mock` must stay in each test file, so this module only supplies the
 * pieces: the hoisted mock bag, the SDK factory, and the render helpers. It
 * never imports `@hermes/plugin-sdk` itself, so the SDK factory can load it.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, vi } from 'vitest'

import type { RosterRow } from './types'

interface CapabilitiesViewProps {
  fixedConnection?: string
  fixedProfile?: string
}

export function createMocks() {
  return {
    activeConnection: { value: 'local' },
    connections: vi.fn(async () => [] as { id: string; label: string }[]),
    createCanonicalChat: vi.fn(async () => 'session-1'),
    deleteBot: vi.fn(async () => undefined),
    dropLeftoverProfilePrefs: vi.fn(),
    /** Flipped off to model a desktop build that predates the live surface. */
    hasCapabilitiesView: { value: true },
    notify: vi.fn(),
    notifyError: vi.fn(),
    request: vi.fn(),
    requestProfile: vi.fn(async (_route: unknown, _method: string, _params?: unknown) => ({})),
    saveBotMeta: vi.fn(),
    skillsView: [] as CapabilitiesViewProps[]
  }
}

export type CreateAgentMocks = ReturnType<typeof createMocks>

/** The `@hermes/plugin-sdk` mock: the real module with the host doors stubbed. */
export function sdkMock(original: Record<string, any>, mocks: CreateAgentMocks, translate: () => unknown) {
  const CapabilitiesViewStub = (props: CapabilitiesViewProps) => {
    mocks.skillsView.push(props)

    return null
  }

  // Builds that route `fixedConnection` get the live Capabilities tab for
  // remote targets too, pinned to the target machine's backend.
  CapabilitiesViewStub.supportsFixedConnection = true

  const mocked: Record<string, unknown> = {
    ...original,
    host: {
      ...original.host,
      state: {
        ...original.host.state,
        connectionId: { ...original.host.state.connectionId, get: () => mocks.activeConnection.value }
      },
      connections: mocks.connections,
      dropLeftoverProfilePrefs: mocks.dropLeftoverProfilePrefs,
      notify: mocks.notify,
      notifyError: mocks.notifyError,
      request: mocks.request,
      requestProfile: mocks.requestProfile
    },
    // The plugin bundle normally lands via `ctx.i18n.register` at load.
    usePluginI18n: translate
  }

  Object.defineProperty(mocked, 'CapabilitiesView', {
    configurable: true,
    enumerable: true,
    get: () => (mocks.hasCapabilitiesView.value ? CapabilitiesViewStub : undefined)
  })

  return mocked
}

/** A catalog entry that needs an API key — the only row that renders setup. */
export const catalog = { auth: null, installed: false, name: 'linear', requires: ['LINEAR_API_KEY'] }

const roster: RosterRow[] = [{ connectionId: 'local', name: 'default' }]

function withQueryClient(children: ReactNode) {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {children}
    </QueryClientProvider>
  )
}

/** The control under a `labeled(...)` caption — the label is presentational,
 *  so it carries no `for`/`id` pair to query by. */
export function controlUnder(caption: string) {
  return screen.getByText(caption).parentElement!.querySelector('[role="combobox"]') as HTMLElement
}

/** Dismisses every store notice, persistent ones included (timers and onDismiss too). */
export async function dismissAllNotifications() {
  const { $notifications, dismissNotification } = await import('@/store/notifications')

  for (const notice of $notifications.get()) {
    dismissNotification(notice.id)
  }
}

/** Registers the suite lifecycle and returns the dialog helpers bound to `mocks`. */
export function createAgentHarness(mocks: CreateAgentMocks) {
  // Each build re-imports the dialog's whole module graph.
  vi.setConfig({ testTimeout: 30_000 })

  beforeAll(async () => {
    // Radix Select/Dialog reach for APIs jsdom does not implement.
    Element.prototype.scrollIntoView = () => undefined
    Element.prototype.hasPointerCapture = () => false
    Element.prototype.releasePointerCapture = () => undefined
    Element.prototype.setPointerCapture = () => undefined
    // Warm the transform cache so the first test isn't charged for it.
    await import('./create-dialog')
  }, 120_000)

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.activeConnection.value = 'local'
    mocks.notify.mockReset()
    mocks.skillsView.length = 0
    mocks.connections.mockResolvedValue([])
    mocks.requestProfile.mockResolvedValue({})
    mocks.request.mockImplementation(async (method: string) => {
      switch (method) {
        case 'mcp.catalog':
          return { servers: [catalog] }

        case 'mcp.servers.test':
          return { ok: true }

        case 'profiles.describe':
          return { mcp_servers: [], skills: [], toolsets: [] }

        default:
          return {}
      }
    })
  })

  afterEach(async () => {
    cleanup()
    // A suite that routes notify to the real store leaves auto-dismiss timers
    // behind; one firing after this file's jsdom window is gone throws.
    await dismissAllNotifications()

    // Radix's focus scope dispatches its unmount event from a 0 ms timer; let it fire
    // here, or it lands in the next test file's jsdom window and throws there.
    await new Promise(resolve => setTimeout(resolve, 0))
  })

  async function renderDialog(hasCapabilitiesView: boolean, onConfigureModel?: (bot: RosterRow) => void) {
    mocks.hasCapabilitiesView.value = hasCapabilitiesView
    vi.resetModules()

    const { CreateAgentDialog } = await import('./create-dialog')

    const dialog = (
      <CreateAgentDialog onClose={() => undefined} onConfigureModel={onConfigureModel} open roster={roster} />
    )
    const view = render(withQueryClient(dialog))

    fireEvent.change(screen.getByPlaceholderText('inbox-triage'), { target: { value: 'inbox-triage' } })
    fireEvent.click(screen.getByRole('button', { name: /Advanced/ }))

    return view
  }

  function createCalls() {
    return mocks.request.mock.calls.filter(([method]) => method === 'profiles.create')
  }

  return { createCalls, renderDialog }
}
