import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/api/client'
import type { HermesConnection } from '@/global'
import { queryClient } from '@/lib/query-client'
import {
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} from '@/store/gateway'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import { $connection } from '@/store/session'
import { $gatewayRestarting, watchGatewayRestartOutcome } from '@/store/system-actions'

import { WebhooksView } from './index'

interface Request {
  body?: unknown
  connectionId?: string
  method?: string
  path: string
  profile?: string
}

const api = vi.fn<(request: Request) => Promise<unknown>>()

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
})

beforeEach(async () => {
  api.mockReset()
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false } })
  clearNotifications()
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('default')
  // Only socket transport and the Electron bridge are replaced. Registry
  // activation, scope stores, REST helpers and restart polling stay real.
  const connected = new WeakSet<HermesGateway>()
  vi.spyOn(HermesGateway.prototype, 'connect').mockImplementation(async function (this: HermesGateway) {
    connected.add(this)
  })
  vi.spyOn(HermesGateway.prototype, 'connectionState', 'get').mockImplementation(function (this: HermesGateway) {
    return connected.has(this) ? 'open' : 'closed'
  })
  vi.spyOn(HermesGateway.prototype, 'close').mockImplementation(function (this: HermesGateway) {
    connected.delete(this)
  })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api,
      getConnection: async () => connection(null, 'default'),
      getConnectionFor: async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        connection(connectionId, profile),
      touchBackend: async () => undefined
    }
  })
  configureGatewayRegistry({
    activeConnectionId: () => $connection.get()?.connectionId ?? null,
    onEvent: vi.fn(),
    onActiveRouteChanged: profile => {
      if ($activeGatewayProfile.get() !== profile) {
        $activeGatewayProfile.set(profile)
      }
    }
  })
  await activatePrimary(null)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  clearNotifications()
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('default')
  closeSecondaryGateways()
  setPrimaryGateway(null)
  $connection.set(null)
  vi.restoreAllMocks()
  vi.useRealTimers()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

function renderWebhooks() {
  return render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <WebhooksView onClose={vi.fn()} />
      </QueryClientProvider>
    </StrictMode>
  )
}

function connection(connectionId: null | string, profile: string): HermesConnection {
  return {
    baseUrl: 'https://example.test',
    ...(connectionId ? { connectionId } : {}),
    isFullscreen: false,
    logs: [],
    nativeOverlayWidth: 0,
    profile,
    registryScoped: Boolean(connectionId),
    token: '',
    windowButtonPosition: null,
    wsUrl: 'wss://example.test'
  }
}

async function activatePrimary(connectionId: null | string) {
  await ensureGatewayForProfile('default')
  const gateway = new HermesGateway()
  await gateway.connect('wss://example.test')
  setPrimaryGateway(gateway, 'default')
  setPrimaryGatewayConnectionId(connectionId)
  $connection.set(connection(connectionId, 'default'))
  await ensureGatewayForProfile('default')
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

it('keeps drafts and one-time secrets with their backend profile under StrictMode', async () => {
  const pending = deferred<{ secret: string; url: string }>()
  let creates = 0
  api.mockImplementation(async request => {
    if (request.path === '/api/webhooks' && request.method === 'POST') {
      creates += 1

      return creates === 2 ? pending.promise : { secret: `secret-${creates}`, url: 'https://example.test/created' }
    }

    return { enabled: true, subscriptions: [] }
  })
  await activatePrimary('host-a')
  $showAllProfiles.set(true)
  renderWebhooks()

  fireEvent.click(await screen.findByRole('button', { name: 'New subscription' }))
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'first' } })
  fireEvent.click(screen.getByRole('button', { name: 'Create' }))
  expect(await screen.findByText('secret-1')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Done' }))

  fireEvent.click(screen.getByRole('button', { name: 'New subscription' }))
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'pending-draft' } })
  fireEvent.click(screen.getByRole('button', { name: 'Create' }))
  await waitFor(() => expect(creates).toBe(2))
  clearNotifications()

  await act(async () => {
    expect(await ensureGatewayForAgent('host-b', 'default')).toBe(true)
  })
  await act(async () => {
    pending.resolve({ secret: 'previous-profile-secret', url: 'https://example.test/previous' })
    await pending.promise
  })

  await waitFor(() => expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'host-b' })))
  expect(screen.queryByText('previous-profile-secret')).toBeNull()
  expect(screen.queryByRole('dialog')).toBeNull()
  expect($notifications.get()).toEqual([])
  expect($showAllProfiles.get()).toBe(true)
  expect(queryClient.getQueryData(['webhooks', 'host-b::default'])).toEqual({ enabled: true, subscriptions: [] })
  fireEvent.click(screen.getByRole('button', { name: 'New subscription' }))
  expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('')
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'new-host-draft' } })
  fireEvent.click(screen.getByRole('button', { name: 'Create' }))
  expect(await screen.findByText('secret-3')).toBeTruthy()
  const secondaryDescriptor = $connection.get()
  await act(async () => {
    await ensureGatewayForProfile('default')
  })
  // The primary fast path changes the socket, but publishes neither a new
  // descriptor nor a different profile name. Its old secret must still clear.
  expect($connection.get()).toBe(secondaryDescriptor)
  expect($activeGatewayProfile.get()).toBe('default')
  expect(screen.queryByText('secret-3')).toBeNull()
  expect(screen.queryByRole('dialog')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'New subscription' }))
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'primary-draft' } })
  await act(async () => {
    expect(await ensureGatewayForAgent('host-b', 'worker')).toBe(true)
  })
  await waitFor(() => expect(api).toHaveBeenCalledWith(expect.objectContaining({ profile: 'worker' })))
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(queryClient.getQueryData(['webhooks', 'host-b::worker'])).toEqual({ enabled: true, subscriptions: [] })
})

it('keeps restart polls with their legacy or registry owner across a same-profile host switch', async () => {
  let polls = 0
  api.mockImplementation(async request => {
    if (request.path === '/api/webhooks/enable') {
      return { restart_started: false, restart_error: 'manual restart required' }
    }

    if (request.path === '/api/gateway/restart') {
      return { name: 'gateway-restart' }
    }

    if (request.path.startsWith('/api/actions/')) {
      polls += 1

      return { running: polls === 1, exit_code: polls === 1 ? null : 1 }
    }

    return { enabled: false, subscriptions: [] }
  })

  for (const connectionId of [null, 'host-a']) {
    polls = 0
    api.mockClear()
    await activatePrimary(connectionId)
    const view = renderWebhooks()
    fireEvent.click(await screen.findByRole('button', { name: 'Enable webhooks' }))
    const restart = await screen.findByRole('button', { name: 'Restart gateway' })
    clearNotifications()
    vi.useFakeTimers()

    await act(async () => {
      fireEvent.click(restart)
      await vi.advanceTimersByTimeAsync(1200)
    })
    expect(polls).toBe(1)
    expect($gatewayRestarting.get()).toBe(true)
    await act(async () => {
      expect(await ensureGatewayForAgent('host-b', 'default')).toBe(true)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1200)
    })

    expect(polls).toBe(2)

    const restartRequests = api.mock.calls
      .map(([request]) => request)
      .filter(request => request.path === '/api/gateway/restart' || request.path.startsWith('/api/actions/'))

    for (const request of restartRequests) {
      expect(request.profile).toBe('default')
      // Untagged legacy requests must stay untagged, preserving remote overrides.
      expect(request.connectionId).toBe(connectionId ?? undefined)
    }

    expect($gatewayRestarting.get()).toBe(false)
    expect($notifications.get()).toEqual([])
    expect(screen.queryByRole('button', { name: 'Restart gateway' })).toBeNull()
    view.unmount()
    queryClient.clear()
    vi.useRealTimers()
  }

  // The same ownership applies when the backend started the restart itself.
  vi.useFakeTimers()
  const watching = watchGatewayRestartOutcome()
  await act(async () => {
    await ensureGatewayForProfile('default')
  })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1200)
    expect(await watching).toBe(false)
  })
  expect(api.mock.calls.at(-1)?.[0]).toEqual(expect.objectContaining({ connectionId: 'host-b', profile: 'default' }))
  expect($gatewayRestarting.get()).toBe(false)
  expect($notifications.get()).toEqual([])
})
