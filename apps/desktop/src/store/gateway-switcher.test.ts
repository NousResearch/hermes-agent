import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopCloudAgent, DesktopConnectionConfig } from '@/global'

const activeProfile = { get: vi.fn(() => 'default') }
const ensureGatewayProfile = vi.fn<(profile: string) => Promise<void>>()
const resetGatewayForProfile = vi.fn<(profile: string) => void>()
const notifyError = vi.fn()
const status = vi.fn()
const discover = vi.fn()
const starredAgents = vi.fn()
const setAgentStarred = vi.fn()
const agentSignIn = vi.fn()
const applyConnectionConfig = vi.fn<() => Promise<DesktopConnectionConfig>>()
const getConnectionConfig = vi.fn<() => Promise<DesktopConnectionConfig>>()

vi.mock('@/store/gateway', () => ({ resetGatewayForProfile }))
vi.mock('@/store/notifications', () => ({ notifyError }))
vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: activeProfile,
  ensureGatewayProfile,
  normalizeProfileKey: (value: string | null | undefined) => (value ?? '').trim() || 'default'
}))

const agent = (id: string, name = id): DesktopCloudAgent => ({
  dashboardGatewayState: 'active',
  dashboardUrl: `https://${id}.example.com`,
  id,
  name,
  status: 'running'
})

beforeEach(() => {
  activeProfile.get.mockReset().mockReturnValue('work')
  ensureGatewayProfile.mockReset().mockResolvedValue(undefined)
  resetGatewayForProfile.mockReset()
  notifyError.mockReset()
  status.mockReset().mockResolvedValue({ portalBaseUrl: 'https://portal.example', signedIn: true })
  discover.mockReset()
  starredAgents.mockReset().mockResolvedValue({ ids: ['starred'] })
  setAgentStarred.mockReset()
  agentSignIn.mockReset().mockResolvedValue({ baseUrl: 'https://starred.example', connected: true })
  applyConnectionConfig.mockReset().mockResolvedValue({} as DesktopConnectionConfig)
  getConnectionConfig.mockReset().mockResolvedValue({} as DesktopConnectionConfig)
  vi.stubGlobal('window', {
    hermesDesktop: {
      applyConnectionConfig,
      getConnectionConfig,
      cloud: { agentSignIn, discover, setAgentStarred, starredAgents, status }
    }
  })
})

describe('gateway switcher', () => {
  it('expands a multi-org discovery result into one target list', async () => {
    discover
      .mockResolvedValueOnce({
        needsOrgSelection: true,
        orgs: [
          { id: 'one', isPersonal: false, name: 'One', role: 'OWNER', slug: 'one' },
          { id: 'two', isPersonal: false, name: 'Two', role: 'MEMBER', slug: 'two' }
        ]
      })
      .mockResolvedValueOnce({ agents: [agent('one-agent')], org: { id: 'one', slug: 'one' } })
      .mockResolvedValueOnce({ agents: [agent('two-agent')], org: { id: 'two', slug: 'two' } })

    const { refreshCloudAgentTargets } = await import('./gateway-switcher')

    await expect(refreshCloudAgentTargets()).resolves.toEqual([
      { agent: agent('one-agent'), org: 'one' },
      { agent: agent('two-agent'), org: 'two' }
    ])
    expect(discover).toHaveBeenNthCalledWith(2, 'one')
    expect(discover).toHaveBeenNthCalledWith(3, 'two')
  })

  it('sorts starred targets first without changing the rest of the target identity', async () => {
    const { cloudAgentsStarredFirst } = await import('./gateway-switcher')

    const targets = [
      { agent: agent('second', 'Second'), org: null },
      { agent: agent('starred', 'Starred'), org: 'team' },
      { agent: agent('first', 'First'), org: null }
    ]

    expect(cloudAgentsStarredFirst(targets, ['starred']).map(target => target.agent.id)).toEqual(['starred', 'first', 'second'])
  })

  it('uses Cloud session cascade then the ordinary profile-scoped apply path', async () => {
    const { switchToCloudAgent } = await import('./gateway-switcher')
    const target = { agent: agent('starred'), org: 'team' }

    await switchToCloudAgent(target)

    expect(agentSignIn).toHaveBeenCalledWith('https://starred.example.com')
    expect(applyConnectionConfig).toHaveBeenCalledWith({
      cloudOrg: 'team',
      mode: 'cloud',
      profile: 'work',
      remoteAuthMode: 'oauth',
      remoteUrl: 'https://starred.example.com'
    })
    expect(resetGatewayForProfile).toHaveBeenCalledWith('work')
    expect(ensureGatewayProfile).toHaveBeenCalledWith('work')
  })

  it("applies to the GLOBAL scope for the 'default' profile so Settings can still switch back", async () => {
    // A 'default'-scoped apply would write a per-profile override that no UI
    // can see or clear — the "stuck on a cloud gateway" bug.
    activeProfile.get.mockReturnValue('default')
    const { switchToCloudAgent } = await import('./gateway-switcher')

    await switchToCloudAgent({ agent: agent('starred'), org: null })

    expect(applyConnectionConfig).toHaveBeenCalledWith(
      expect.objectContaining({ mode: 'cloud', profile: undefined })
    )
    expect(resetGatewayForProfile).toHaveBeenCalledWith('default')
    expect(ensureGatewayProfile).toHaveBeenCalledWith('default')
  })

  it('switches to local on the active profile scope (default → global)', async () => {
    activeProfile.get.mockReturnValue('default')
    const { switchToGatewayMode } = await import('./gateway-switcher')

    await switchToGatewayMode('local')

    expect(applyConnectionConfig).toHaveBeenCalledWith({ mode: 'local', profile: undefined })
    expect(ensureGatewayProfile).toHaveBeenCalledWith('default')
  })

  it('clears a named profile override when switching that profile to local', async () => {
    const { switchToGatewayMode } = await import('./gateway-switcher')

    await switchToGatewayMode('local')

    expect(applyConnectionConfig).toHaveBeenCalledWith({ mode: 'local', profile: 'work' })
  })

  it('switches remote/ssh on the global scope, re-adopting the saved target', async () => {
    const { switchToGatewayMode } = await import('./gateway-switcher')

    await switchToGatewayMode('remote')
    await switchToGatewayMode('ssh')

    expect(applyConnectionConfig).toHaveBeenNthCalledWith(1, { mode: 'remote', profile: undefined })
    expect(applyConnectionConfig).toHaveBeenNthCalledWith(2, { mode: 'ssh', profile: undefined })
  })

  it('surfaces a failed mode switch and clears the switching marker', async () => {
    applyConnectionConfig.mockRejectedValue(new Error('nope'))
    const { $gatewayModeSwitching, switchToGatewayMode } = await import('./gateway-switcher')

    await expect(switchToGatewayMode('remote')).rejects.toThrow('nope')
    expect(notifyError).toHaveBeenCalled()
    expect($gatewayModeSwitching.get()).toBeNull()
  })

  it('marks the connected agent active by URL even when provenance was recorded as plain remote', async () => {
    const { cloudAgentIsActive } = await import('./gateway-switcher')
    const target = { agent: agent('starred'), org: null }

    // Kind-agnostic: a cloud instance connected through an older flow is
    // stored as mode 'remote', but it is still the gateway we're talking to.
    expect(cloudAgentIsActive(target, 'https://starred.example.com/')).toBe(true)
    expect(cloudAgentIsActive(target, 'https://other.example.com')).toBe(false)
    expect(cloudAgentIsActive(target, undefined)).toBe(false)
  })

  it('suppresses a saved remote target that duplicates a discovered Cloud agent', async () => {
    const { offerableRemoteUrl } = await import('./gateway-switcher')
    const targets = [{ agent: agent('starred'), org: null }]
    const config = (savedRemoteUrl: string) => ({ savedRemoteUrl }) as never

    expect(offerableRemoteUrl(config('https://starred.example.com/'), targets)).toBe('')
    expect(offerableRemoteUrl(config('https://my-own-gateway.example.com'), targets)).toBe(
      'https://my-own-gateway.example.com'
    )
    expect(offerableRemoteUrl(null, targets)).toBe('')
  })

  it('updates stars only from Electron-owned state', async () => {
    setAgentStarred.mockResolvedValue({ ids: ['starred', 'new'] })
    const { $starredCloudAgentIds, setCloudAgentStarred } = await import('./gateway-switcher')

    await expect(setCloudAgentStarred('new', true)).resolves.toEqual(['starred', 'new'])
    expect($starredCloudAgentIds.get()).toEqual(['starred', 'new'])
  })
})
