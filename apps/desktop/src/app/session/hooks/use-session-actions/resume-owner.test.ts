import { beforeEach, describe, expect, it, vi } from 'vitest'

// An ownerless resume (no captured route, no `connection_id` on the row, no
// remembered hint) used to fall through to the BARE profile door. That door
// resolves the name against the primary route, and `default` is the primary
// profile of every source, so a resume that ran while a registry connection was
// in front re-homed the window onto this machine's local gateway: the pick that
// put it there was undone and the sidebar followed it back to the local
// profile's sessions. These tests pin the rule the other resume paths already
// follow — an ownerless row stays on the source the window is looking at.

const deps = vi.hoisted(() => ({
  activeConnectionId: null as null | string,
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined),
  openGatewayForAgent: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined),
  showAllProfiles: false
}))

vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => deps.activeConnectionId,
  isActivePrimary: vi.fn(() => true),
  openGatewayForAgent: deps.openGatewayForAgent,
  openGatewayForProfile: deps.openGatewayForProfile
}))

vi.mock('@/store/profile', () => ({
  $showAllProfiles: { get: () => deps.showAllProfiles },
  ensureGatewayAgent: deps.ensureGatewayAgent,
  ensureGatewayProfile: deps.ensureGatewayProfile,
  normalizeProfileKey: (value: null | string | undefined) => String(value ?? '').trim() || 'default'
}))

vi.mock('@/store/session', () => ({ setSessionOwnerHint: vi.fn() }))

const { openResumeGateway } = await import('./resume-owner')

beforeEach(() => {
  deps.activeConnectionId = null
  deps.showAllProfiles = false
  vi.clearAllMocks()
})

describe('openResumeGateway keeps an ownerless resume on the visible source', () => {
  it('stays on the registry connection in front instead of the bare profile door', async () => {
    deps.activeConnectionId = 'nailde'

    await openResumeGateway('', undefined, 'default')

    expect(deps.ensureGatewayAgent).toHaveBeenCalledWith('nailde', 'default')
    expect(deps.ensureGatewayProfile).not.toHaveBeenCalled()
  })

  it('keeps the row profile when the ownerless resume names another one', async () => {
    deps.activeConnectionId = 'nailde'

    await openResumeGateway('', { connectionId: '', profile: 'orchestrator' }, undefined)

    expect(deps.ensureGatewayAgent).toHaveBeenCalledWith('nailde', 'orchestrator')
    expect(deps.ensureGatewayProfile).not.toHaveBeenCalled()
  })

  it('keeps the local profile door for the local source', async () => {
    deps.activeConnectionId = 'local'

    await openResumeGateway('', undefined, 'default')

    expect(deps.ensureGatewayProfile).toHaveBeenCalledWith('default')
    expect(deps.ensureGatewayAgent).not.toHaveBeenCalled()
  })

  it('keeps the local profile door while no source is published', async () => {
    await openResumeGateway('', undefined, 'research')

    expect(deps.ensureGatewayProfile).toHaveBeenCalledWith('research')
    expect(deps.ensureGatewayAgent).not.toHaveBeenCalled()
  })

  it('still honours a resolved owner over the visible source', async () => {
    deps.activeConnectionId = 'nailde'

    await openResumeGateway('homelab', undefined, 'default')

    expect(deps.ensureGatewayAgent).toHaveBeenCalledWith('homelab', 'default')
    expect(deps.ensureGatewayProfile).not.toHaveBeenCalled()
  })
})
