import { describe, expect, it } from 'vitest'

import {
  AGENT_SETTINGS_GROUPS,
  agentConnectorsHref,
  agentSettingsSearch,
  isAgentConnectorsLink,
  resolveAgentSettingsSection
} from './settings-routes'

describe('agent settings routes', () => {
  it('keeps the owner when moving between config, capability and runtime controls', () => {
    const original = '?agentProfile=research&connection=remote-a&panel=capabilities&tab=toolsets&field=old'
    const query = new URLSearchParams(agentSettingsSearch(original, 'model'))
    expect(query.get('agentProfile')).toBe('research')
    expect(query.get('connection')).toBe('remote-a')
    expect(query.has('field')).toBe(false)
    expect(resolveAgentSettingsSection(query)).toBe('model')
    const runtime = new URLSearchParams(agentSettingsSearch(query.toString(), 'limits'))
    expect(runtime.get('agentProfile')).toBe('research')
    expect(runtime.get('connection')).toBe('remote-a')
    expect(resolveAgentSettingsSection(runtime)).toBe('limits')
  })

  it('resolves existing deep links without admitting global or nonexistent config pages', () => {
    expect(resolveAgentSettingsSection(new URLSearchParams('tab=config:model&page=fallbacks'))).toBe('model')
    expect(resolveAgentSettingsSection(new URLSearchParams('panel=capabilities&tab=toolsets'))).toBe('tools')
    expect(resolveAgentSettingsSection(new URLSearchParams('panel=messaging'))).toBe('messaging')
    // The Overview's Capabilities heading names the panel without a page: it opens Tools, not General.
    expect(resolveAgentSettingsSection(new URLSearchParams('agentProfile=ops&panel=capabilities'))).toBe('tools')
    expect(resolveAgentSettingsSection(new URLSearchParams('agentProfile=ops&panel=settings'))).toBe('general')

    for (const route of [
      'tab=config:appearance',
      'tab=billing',
      'tab=config:model&page=unknown',
      'tab=agent:limits&page=desktop'
    ]) {
      expect(resolveAgentSettingsSection(new URLSearchParams(route))).toBeNull()
    }
  })

  it.each([
    ['tools', 'panel=capabilities&tab=toolsets'],
    ['skills', 'panel=capabilities&tab=skills'],
    ['plugins', 'panel=capabilities&tab=plugins'],
    ['providers', 'panel=settings&tab=providers'],
    ['limits', 'panel=settings&tab=config:advanced&page=terminal'],
    ['tools-advanced', 'panel=settings&tab=config:advanced&page=tools'],
    ['browser', 'panel=settings&tab=vault&page=sources']
  ])('opens %s from the link shape app Settings and Capabilities write', (section, link) => {
    expect(resolveAgentSettingsSection(new URLSearchParams(`agentProfile=research&${link}`))).toBe(section)
  })

  it.each(['connectors', 'mcp', 'agent:connectors'])('has no Connectors page: an old %s link is a link to Capabilities for that agent', tab => {
    const params = new URLSearchParams(`agentProfile=research&connection=homelab&panel=capabilities&tab=${tab}`)

    expect(resolveAgentSettingsSection(params)).toBeNull()
    expect(isAgentConnectorsLink(params)).toBe(true)
    expect(agentConnectorsHref({ connectionId: 'homelab', profile: 'research' })).toBe(
      '/capabilities?tab=connectors&agentProfile=research&connection=homelab'
    )
    expect(AGENT_SETTINGS_GROUPS.flatMap(([, sections]) => sections)).not.toContain('connectors')
  })

  it('round-trips every menu section through the link it writes', () => {
    for (const [, sections] of AGENT_SETTINGS_GROUPS) {
      for (const section of sections) {
        const link = new URLSearchParams(agentSettingsSearch('?agentProfile=research&field=old', section))

        expect(link.get('agentProfile')).toBe('research')
        expect(resolveAgentSettingsSection(link)).toBe(section)
      }
    }
  })
})
