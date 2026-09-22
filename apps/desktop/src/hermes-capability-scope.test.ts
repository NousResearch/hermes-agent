import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { approvePairing, getMessagingPlatforms } from './api/messaging'
import { getAuxiliaryModels, getGlobalModelInfo } from './api/models'
import { getOfficialSkills, getSkillHubSources } from './api/skills'
import { getToolsetConfig } from './api/toolsets'
import {
  getHermesConfigRecord,
  getMcpCatalog,
  getSkillContent,
  getSkills,
  getToolsets,
  getUsageAnalytics,
  installSkillFromHub,
  profileScopeKey,
  saveMcpServers,
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

  const last = () => api.mock.calls.at(-1)?.[0] as { connectionId?: string; profile?: string; priority?: string }

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

  it('marks an explicitly scoped Settings / Capabilities read as foreground (#111651)', () => {
    // A scope-selector pick is a visible user action: its cold dial must take
    // the pool's reserved foreground slot instead of queueing behind hydration.
    getHermesConfigRecord('coder')
    expect(last()).toMatchObject({ profile: 'coder', priority: 'foreground' })

    void getSkills('coder')
    expect(last()).toMatchObject({ profile: 'coder', priority: 'foreground' })

    // The Model page fires these alongside the config record for the same
    // scope; an untagged sibling would queue as background work again.
    void getGlobalModelInfo('coder')
    expect(last()).toMatchObject({ profile: 'coder', priority: 'foreground' })

    void getAuxiliaryModels('coder')
    expect(last()).toMatchObject({ profile: 'coder', priority: 'foreground' })
  })

  it('every explicitly scoped api/ helper dials foreground, not only the Settings pages (#111651)', () => {
    // The class rule lives in the scope helpers themselves, so a helper in
    // any api/ module inherits it — Capabilities hub/toolset-config reads and
    // the Messaging page were left queueing as background work when the rule
    // was spread per call site.
    void getOfficialSkills('coder')
    expect(last()).toMatchObject({ profile: 'coder', priority: 'foreground' })

    void getSkillHubSources('coder')
    expect(last()).toMatchObject({ profile: 'coder', priority: 'foreground' })

    void getToolsetConfig('browser', { connectionId: 'homelab', profile: 'coder' })
    expect(last()).toMatchObject({ connectionId: 'homelab', profile: 'coder', priority: 'foreground' })

    void getMessagingPlatforms('coder')
    expect(last()).toMatchObject({ profile: 'coder', priority: 'foreground' })

    // `null` deliberately targets the primary — that backend is always warm,
    // so it stays untagged like the ambient path.
    void getOfficialSkills(null)
    expect(last()).not.toHaveProperty('priority')
  })

  it('a foreground tag never leaks into a request body that carries the profile', () => {
    void approvePairing('telegram', 'req-1', 'coder')

    const call = api.mock.calls.at(-1)?.[0] as { body?: Record<string, unknown>; priority?: string }

    expect(call.priority).toBe('foreground')
    expect(call.body).toEqual({ platform: 'telegram', request_id: 'req-1', profile: 'coder' })
  })

  it('keeps ambient config reads unprioritized for background hydration', () => {
    getHermesConfigRecord()
    expect(last()).not.toHaveProperty('priority')

    void getGlobalModelInfo()
    expect(last()).not.toHaveProperty('priority')
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

  it('the toolset toggle body carries the profile, not just the query scope', () => {
    // web_routers/tools.py scopes this write by body.profile before the
    // ?profile= query. When the Electron bridge classifies the PUT as
    // unscopable on a multiplex host the query never lands — without the
    // body copy the toggle silently writes the launch profile's config
    // (#118633). Mirrors the pairing approve contract above.
    void setToolsetEnabled('browser', false, 'research')

    const stringScoped = api.mock.calls.at(-1)?.[0] as { body?: Record<string, unknown> }
    expect(stringScoped.body).toEqual({ enabled: false, profile: 'research' })

    void setToolsetEnabled('browser', true, { connectionId: 'homelab', profile: 'inbox-bot' })

    const objectScoped = api.mock.calls.at(-1)?.[0] as { body?: Record<string, unknown> }
    expect(objectScoped.body).toEqual({ enabled: true, profile: 'inbox-bot' })

    // Ambient callers (no explicit scope) keep a body free of the key, same
    // as every other ambient request.
    setApiRequestProfile(null)
    void setToolsetEnabled('browser', true)

    const ambient = api.mock.calls.at(-1)?.[0] as { body?: Record<string, unknown> }
    expect(ambient.body).toEqual({ enabled: true, profile: undefined })
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
