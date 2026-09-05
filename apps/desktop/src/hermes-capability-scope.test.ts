import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getHermesConfigRecord,
  getMcpCatalog,
  getSkillContent,
  getSkills,
  getTerminalBackends,
  getToolsets,
  getUsageAnalytics,
  installSkillFromHub,
  profileScopeKey,
  saveMcpServers,
  selectTerminalBackend,
  setApiRequestConnection,
  setApiRequestProfile,
  setSkillEnabled,
  setToolsetEnabled
} from './hermes'

// Contract: the Capabilities surface (skills / toolsets / MCP / hub / config)
// can be scoped to a (connection, profile) pair — a profile belongs to ONE
// gateway, so its skills/tools/MCP must be read from and written to THAT
// machine's backend. Three shapes:
//   - no scope         → ambient profile + ambient registry connection tag
//   - string scope     → explicit profile, ambient connection tag
//   - object scope     → explicit (connection, profile) pin; 'local' pins the
//                        local pool and DROPS the ambient connection tag
describe('capability helpers are connection-scoped', () => {
  const api = vi.fn(async (_req: { connectionId?: string; path: string; profile?: string }) => ({}) as never)

  beforeEach(() => {
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = { api }
    api.mockClear()
  })

  afterEach(() => {
    setApiRequestProfile(null)
    setApiRequestConnection(null)
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  const last = () => api.mock.calls.at(-1)?.[0] as { connectionId?: string; profile?: string }

  it('omits both scopes when none are active (single-source users unaffected)', () => {
    void getSkills()
    expect(last()).not.toHaveProperty('profile')
    expect(last()).not.toHaveProperty('connectionId')
  })

  it('carries the ambient registry connection on the legacy string/ambient path', () => {
    // A window activated onto a registered remote gateway must read THAT
    // machine's skills — not the local pool's (the ambient-tag half).
    setApiRequestProfile('research')
    setApiRequestConnection('gw-tailscale')

    void getSkills()
    void getToolsets()
    void getHermesConfigRecord()
    void getMcpCatalog()
    void getUsageAnalytics(30)

    for (const call of api.mock.calls) {
      expect((call[0] as { connectionId?: string }).connectionId).toBe('gw-tailscale')
      expect(call[0].profile).toBe('research')
    }
  })

  it('string scopes override the profile but keep the ambient connection', () => {
    setApiRequestProfile('research')
    setApiRequestConnection('gw-tailscale')

    void getSkills('coder')

    expect(last().profile).toBe('coder')
    expect(last().connectionId).toBe('gw-tailscale')
  })

  it('routes terminal backend reads and writes through the rendered capability scope', () => {
    // The terminal picker used to bypass capabilityScoped() and fall back to
    // this stale ambient profile, silently writing another profile's config.
    setApiRequestProfile('stale-sidebar-profile')
    setApiRequestConnection('gw-tailscale')

    void getTerminalBackends('research')
    expect(last()).toMatchObject({
      connectionId: 'gw-tailscale',
      path: '/api/tools/terminal/backends',
      profile: 'research'
    })

    void selectTerminalBackend('docker', { connectionId: 'homelab', profile: 'research' })
    expect(last()).toMatchObject({
      body: { backend: 'docker' },
      connectionId: 'homelab',
      method: 'PUT',
      path: '/api/tools/terminal/backend',
      profile: 'research'
    })
  })

  it('object scopes pin every read and write to the named connection', () => {
    void getSkills({ connectionId: 'homelab', profile: 'inbox-bot' })
    void getToolsets({ connectionId: 'homelab', profile: 'inbox-bot' })
    void getSkillContent('arxiv', { connectionId: 'homelab', profile: 'inbox-bot' })
    void setSkillEnabled('arxiv', false, { connectionId: 'homelab', profile: 'inbox-bot' })
    void setToolsetEnabled('browser', true, { connectionId: 'homelab', profile: 'inbox-bot' })
    void saveMcpServers({}, { connectionId: 'homelab', profile: 'inbox-bot' })
    void installSkillFromHub('official/research/arxiv', { connectionId: 'homelab', profile: 'inbox-bot' })

    for (const call of api.mock.calls) {
      expect((call[0] as { connectionId?: string }).connectionId).toBe('homelab')
      expect(call[0].profile).toBe('inbox-bot')
    }
  })

  it("a 'local' pin carries an explicit connectionId even while a remote gateway is active", () => {
    setApiRequestProfile('research')
    setApiRequestConnection('gw-tailscale')

    void getSkills({ connectionId: 'local', profile: 'coder' })

    // The explicit pin must survive to Electron main: its registry resolver
    // owns 'local' (forced-local pooled child). Omitting the key here let the
    // ambient tag — or, worse, a remote registry PRIMARY on the v1 fallback
    // route — absorb a "This device" pick (v0.20.6 regression, #91564 rung).
    expect(last().profile).toBe('coder')
    expect(last().connectionId).toBe('local')
  })

  it('profileScopeKey keeps legacy keys byte-identical and namespaces every explicit pin', () => {
    expect(profileScopeKey()).toBe('default')
    expect(profileScopeKey(null)).toBe('default')
    expect(profileScopeKey('coder')).toBe('coder')
    // A 'local' pin and the ambient path can resolve to DIFFERENT backends
    // when the registry primary is remote — they must not share a cache row.
    expect(profileScopeKey({ connectionId: 'local', profile: 'coder' })).toBe('local::coder')
    expect(profileScopeKey({ connectionId: 'homelab', profile: 'coder' })).toBe('homelab::coder')
    expect(profileScopeKey({ connectionId: 'homelab' })).toBe('homelab::default')
  })
})
