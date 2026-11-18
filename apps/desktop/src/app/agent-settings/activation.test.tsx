import type * as ReactQuery from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { renameProfile } from '@/hermes'
import type * as HermesApi from '@/hermes'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { selectConnection } from '@/store/connections'
import type * as ConnectionsStore from '@/store/connections'
import type * as FleetRosterStore from '@/store/fleet-roster'
import type * as GatewayStore from '@/store/gateway'
import { $activeGatewayProfile, requestFreshSession } from '@/store/profile'
import {
  getRememberedSessionId,
  getSessionOwnerHints,
  setRememberedSessionId,
  setSessionOwnerHint
} from '@/store/session'
import { deferred } from '@/test/deferred'

import { useFreshSessionRequest } from '../contrib/wiring-effects'

import { AgentSettingsView } from './index'

vi.mock('@tanstack/react-query', async importOriginal => ({
  ...(await importOriginal<typeof ReactQuery>()),
  useQuery: ({ queryKey }: { queryKey: readonly unknown[] }) =>
    queryKey.includes('profiles')
      ? {
          data: {
            profiles: ['research', 'renamed'].map(name => ({ name, path: `/tmp/${name}`, is_default: false }))
          }
        }
      : {}
}))
vi.mock('../hooks/use-config-record', () => ({ useHermesConfigRecord: () => ({}) }))
vi.mock('../capabilities/skills/skills-data', () => ({ useSkillsQuery: () => ({}) }))
vi.mock('../capabilities/toolsets/toolsets-data', () => ({ useToolsetsQuery: () => ({}) }))
vi.mock('@/store/connections', async importOriginal => ({
  ...(await importOriginal<typeof ConnectionsStore>()),
  selectConnection: vi.fn()
}))
vi.mock('@/store/fleet-roster', async importOriginal => ({
  ...(await importOriginal<typeof FleetRosterStore>()),
  refreshFleetRoster: vi.fn(async () => undefined)
}))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  activeGatewayConnectionId: () => 'local',
  retireLocalProfileGateways: vi.fn(() => $activeGatewayProfile.set('default'))
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  renameProfile: vi.fn()
}))

const destination = '/settings?agentProfile=research&connection=local&panel=settings&tab=agent%3Ageneral'
// Another machine's agent still waits for "Open agent"; this window's machine renders in place.
const remoteDestination = destination.replace('connection=local', 'connection=remote')

function Harness() {
  const location = useLocation()
  const navigate = useNavigate()
  useFreshSessionRequest(options => {
    if (!options?.preserveRoute) {
      navigate('/')
    }
  })
  const params = new URLSearchParams(location.search)
  const profile = params.get('agentProfile') ?? 'default'
  const connectionId = params.get('connection') ?? 'local'

  return (
    <>
      <output data-testid="route">{location.pathname + location.search}</output>
      <button onClick={() => navigate('/elsewhere')} type="button">
        Leave settings
      </button>
      <button
        onClick={() => navigate(location.pathname + location.search, { replace: true, state: { refreshed: true } })}
        type="button"
      >
        Refresh route state
      </button>
      {location.pathname === '/settings' && (
        <AgentSettingsView key={profile} onClose={() => navigate('/')} owner={{ connectionId, profile }} />
      )}
    </>
  )
}

function mount(entry = destination) {
  render(
    <MemoryRouter initialEntries={[entry]}>
      <Harness />
    </MemoryRouter>
  )
}

beforeEach(() => {
  localStorage.clear()
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $activeGatewayProfile.set('default')
  $connectionsRegistry.set({
    connections: [
      { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
      { id: 'remote', kind: 'remote', label: 'Remote', tokenPreview: null, tokenSet: false }
    ],
    primary: 'local',
    secureTokenStorage: true,
    version: 2
  })
  vi.mocked(selectConnection).mockImplementation(async (connection, options) => {
    requestFreshSession({ preserveRoute: options?.preserveRoute })
    $apiRequestScope.setKey('connectionId', connection)
    $activeGatewayProfile.set(options?.profile ?? 'default')
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $connectionsRegistry.set(null)
  $apiRequestScope.set({ connectionId: null, profile: null })
  $activeGatewayProfile.set('default')
})

it.each([false, true])('keeps the requested panel without overriding later navigation (leave=%s)', async leave => {
  const activation = deferred<void>()
  vi.mocked(selectConnection).mockImplementation(async (connection, options) => {
    await activation.promise
    requestFreshSession({ preserveRoute: options?.preserveRoute })
    $apiRequestScope.setKey('connectionId', connection)
    $activeGatewayProfile.set(options?.profile ?? 'default')
  })
  mount(remoteDestination)
  fireEvent.click(screen.getByRole('button', { name: 'Open agent' }))

  if (leave) {
    fireEvent.click(screen.getByRole('button', { name: 'Leave settings' }))
  }

  await act(async () => {
    activation.resolve()
    await activation.promise
  })
  expect(screen.getByTestId('route').textContent).toBe(leave ? '/elsewhere' : remoteDestination)

  if (!leave) {
    expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()
    expect(screen.getByRole('button', { name: /^Rename: / })).toBeTruthy()
  }

  // A later ordinary new-chat request must not inherit settings preservation.
  act(() => requestFreshSession())
  expect(screen.getByTestId('route').textContent).toBe('/')
})

it.each(['stay', 'leave', 'refresh'] as const)(
  'finishes a local rename after gateway retirement without stealing navigation (%s)',
  async navigation => {
    $activeGatewayProfile.set('research')
    setRememberedSessionId('rename-session', 'research')
    setSessionOwnerHint('rename-session', { connectionId: 'local', profile: 'research' })
    setSessionOwnerHint('remote-session', { connectionId: 'remote', profile: 'research' })
    const renamed = deferred<{ name: string; ok: boolean; path: string }>()
    vi.mocked(renameProfile).mockReturnValueOnce(renamed.promise)
    mount()
    fireEvent.click(screen.getByRole('button', { name: /^Rename: / }))
    fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Rename' }))
    await waitFor(() => expect(renameProfile).toHaveBeenCalled())
    // Retirement moved the window to the default agent; the page keeps showing its owner.
    expect($activeGatewayProfile.get()).toBe('default')
    expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()

    // The rename dialog is still open over the page, so the harness controls sit behind it.
    if (navigation === 'leave') {
      fireEvent.click(screen.getByRole('button', { hidden: true, name: 'Leave settings' }))
    } else if (navigation === 'refresh') {
      fireEvent.click(screen.getByRole('button', { hidden: true, name: 'Refresh route state' }))
    }

    await act(async () => {
      renamed.resolve({ name: 'renamed', ok: true, path: '/tmp/renamed' })
      await renamed.promise
    })
    expect(getRememberedSessionId('research')).toBeNull()
    expect(getRememberedSessionId('renamed')).toBe('rename-session')
    expect(getSessionOwnerHints('rename-session')).toEqual([{ connectionId: 'local', profile: 'renamed' }])
    expect(getSessionOwnerHints('remote-session')).toEqual([{ connectionId: 'remote', profile: 'research' }])
    expect(screen.getByTestId('route').textContent).toBe(
      navigation === 'leave' ? '/elsewhere' : destination.replace('agentProfile=research', 'agentProfile=renamed')
    )

    if (navigation === 'leave') {
      expect(selectConnection).not.toHaveBeenCalled()
    } else {
      expect($activeGatewayProfile.get()).toBe('renamed')
      expect(screen.getByRole('button', { name: /^Rename: / })).toBeTruthy()
    }
  }
)

it('follows a renamed agent the window was only viewing without activating it', async () => {
  vi.mocked(renameProfile).mockResolvedValueOnce({ name: 'renamed', ok: true, path: '/tmp/renamed' })
  mount()
  fireEvent.click(screen.getByRole('button', { name: /^Rename: / }))
  fireEvent.change(screen.getByLabelText(/new name/i), { target: { value: 'renamed' } })
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Rename' }))
  await waitFor(() =>
    expect(screen.getByTestId('route').textContent).toBe(destination.replace('agentProfile=research', 'agentProfile=renamed'))
  )
  expect(selectConnection).not.toHaveBeenCalled()
  expect($activeGatewayProfile.get()).toBe('default')
})
