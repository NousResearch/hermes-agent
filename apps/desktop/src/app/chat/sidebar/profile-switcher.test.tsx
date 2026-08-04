import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopCloudAgent } from '@/global'

class ResizeObserverMock {
  disconnect() {}
  observe() {}
  unobserve() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverMock)

const cloudTargets = atom<{ agent: DesktopCloudAgent; org: string | null }[]>([])
const targetsLoading = atom(false)
const starredIds = atom<string[]>([])
const switching = atom<string | null>(null)
const modeSwitching = atom<string | null>(null)
const gatewayConfig = atom<any>(null)
const connection = atom<any>(null)
const refreshGatewaySwitcher = vi.fn()
const setCloudAgentStarred = vi.fn()
const switchToCloudAgent = vi.fn()
const switchToGatewayMode = vi.fn()

vi.mock('@/store/gateway-switcher', () => ({
  $cloudAgentSwitching: switching,
  $cloudAgentTargets: cloudTargets,
  $cloudAgentTargetsLoading: targetsLoading,
  $gatewayConnectionConfig: gatewayConfig,
  $gatewayModeSwitching: modeSwitching,
  $starredCloudAgentIds: starredIds,
  cloudAgentIsActive: (target: { agent: DesktopCloudAgent }, baseUrl?: string) =>
    Boolean(baseUrl && baseUrl === target.agent.dashboardUrl),
  cloudAgentsStarredFirst: (targets: { agent: DesktopCloudAgent }[], ids: string[]) =>
    [...targets].sort((a, b) => Number(ids.includes(b.agent.id)) - Number(ids.includes(a.agent.id))),
  offerableRemoteUrl: (config: { savedRemoteUrl?: string } | null, targets: { agent: DesktopCloudAgent }[]) => {
    const url = config?.savedRemoteUrl || ''

    return targets.some(target => target.agent.dashboardUrl === url) ? '' : url
  },
  refreshGatewaySwitcher,
  setCloudAgentStarred,
  switchToCloudAgent,
  switchToGatewayMode
}))
vi.mock('@/store/session', () => ({ $connection: connection }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  $profileColors: atom({}),
  $profileCreateRequest: atom(0),
  $profileOrder: atom([]),
  $profiles: atom([]),
  $profileScope: atom('default'),
  ALL_PROFILES: '__all__',
  normalizeProfileKey: (name: string) => name || 'default',
  refreshActiveProfile: vi.fn(),
  selectProfile: vi.fn(),
  setProfileColor: vi.fn(),
  setProfileOrder: vi.fn(),
  setShowAllProfiles: vi.fn(),
  sortByProfileOrder: <T,>(items: T[]) => items
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/hermes', () => ({ getProfileSoul: vi.fn(), updateProfileSoul: vi.fn() }))

const agent = (id: string, name = id): DesktopCloudAgent => ({
  dashboardGatewayState: 'active',
  dashboardUrl: `https://${id}.example.com`,
  id,
  name,
  status: 'running'
})

beforeEach(() => {
  cloudTargets.set([
    { agent: agent('plain', 'Plain'), org: null },
    { agent: agent('starred', 'Starred'), org: 'team' }
  ])
  targetsLoading.set(false)
  starredIds.set(['starred'])
  switching.set(null)
  modeSwitching.set(null)
  gatewayConfig.set(null)
  connection.set(null)
  refreshGatewaySwitcher.mockReset().mockResolvedValue(undefined)
  setCloudAgentStarred.mockReset().mockResolvedValue(['starred'])
  switchToCloudAgent.mockReset().mockResolvedValue(undefined)
  switchToGatewayMode.mockReset().mockResolvedValue(undefined)
})

const openSwitcher = async () => {
  const { ProfileRail } = await import('./profile-switcher')
  render(
    <MemoryRouter>
      <ProfileRail />
    </MemoryRouter>
  )
  fireEvent.click(screen.getByRole('button', { name: 'Switch gateway' }))
}

describe('ProfileRail gateway switcher', () => {
  it('lists starred Cloud agents first and routes selection through the shared Cloud switch action', async () => {
    await openSwitcher()
    await waitFor(() => expect(refreshGatewaySwitcher).toHaveBeenCalled())
    expect(screen.getByText('Starred').compareDocumentPosition(screen.getByText('Plain'))).toBe(Node.DOCUMENT_POSITION_FOLLOWING)

    fireEvent.click(screen.getByRole('button', { name: 'Switch to Starred' }))
    await waitFor(() => expect(switchToCloudAgent).toHaveBeenCalledWith({ agent: agent('starred', 'Starred'), org: 'team' }))
  })

  it('toggles an agent star from the main switcher', async () => {
    await openSwitcher()
    fireEvent.click(await screen.findByRole('button', { name: 'Unstar Starred' }))

    await waitFor(() => expect(setCloudAgentStarred).toHaveBeenCalledWith('starred', false))
  })

  it('always offers the local gateway and switches through the shared mode action', async () => {
    await openSwitcher()

    fireEvent.click(screen.getByRole('button', { name: 'Switch to This device' }))
    await waitFor(() => expect(switchToGatewayMode).toHaveBeenCalledWith('local'))
  })

  it('offers configured remote and SSH targets under their own headings', async () => {
    gatewayConfig.set({
      envOverride: false,
      savedRemoteUrl: 'https://gw.example.com:9119/hermes',
      savedSshHost: 'workstation'
    })
    await openSwitcher()

    expect(screen.getByText('Remote')).toBeTruthy()
    expect(screen.getByText('SSH')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Switch to gw.example.com:9119' }))
    await waitFor(() => expect(switchToGatewayMode).toHaveBeenCalledWith('remote'))

    // A successful switch dismisses the menu; reopen it for the SSH row.
    fireEvent.click(screen.getByRole('button', { name: 'Switch gateway' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Switch to workstation' }))
    await waitFor(() => expect(switchToGatewayMode).toHaveBeenCalledWith('ssh'))
  })

  it('hides a Remote target that is really a discovered Cloud agent (phantom snapshot)', async () => {
    gatewayConfig.set({
      envOverride: false,
      savedRemoteUrl: 'https://starred.example.com',
      savedSshHost: ''
    })
    await openSwitcher()

    expect(screen.queryByText('Remote')).toBeNull()
  })

  it('hides remote and SSH sections when no target is configured', async () => {
    gatewayConfig.set({ envOverride: false, savedRemoteUrl: '', savedSshHost: '' })
    await openSwitcher()

    expect(screen.queryByText('Remote')).toBeNull()
    expect(screen.queryByText('SSH')).toBeNull()
  })

  it('keeps the ACTIVE entry enabled and emphasized instead of greying it out', async () => {
    connection.set({ baseUrl: 'https://starred.example.com', mode: 'remote', remoteKind: 'cloud' })
    await openSwitcher()

    const activeRow = screen.getByRole('button', { name: 'Switch to Starred' })
    expect(activeRow.hasAttribute('disabled')).toBe(false)
    expect(screen.getByText('Starred').className).toContain('font-semibold')

    // Clicking the live gateway is a no-op dismiss, not another switch.
    fireEvent.click(activeRow)
    expect(switchToCloudAgent).not.toHaveBeenCalled()
  })
})
