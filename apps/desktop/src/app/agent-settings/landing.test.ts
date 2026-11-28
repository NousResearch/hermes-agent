import { describe, expect, it } from 'vitest'

import { agentPageHref, pickLandingAgent } from './landing'

const agents = [
  { connectionId: null, profile: 'default' },
  { connectionId: null, profile: 'ops' },
  { connectionId: 'homelab', profile: 'ops' }
]

const isDefault = (agent: { profile: string }) => agent.profile === 'default'

describe('pickLandingAgent', () => {
  it('opens the only agent when there is one', () => {
    expect(pickLandingAgent([agents[1]!], null, isDefault)).toBe(agents[1])
  })

  it('reopens the agent last viewed in this window, matched on profile and machine', () => {
    expect(pickLandingAgent(agents, { connectionId: 'homelab', profile: 'ops' }, isDefault)).toBe(agents[2])
  })

  it('falls back to the default agent when the last viewed one was deleted', () => {
    expect(pickLandingAgent(agents, { connectionId: null, profile: 'gone' }, isDefault)).toBe(agents[0])
  })

  it('falls back to the first agent when none is the default, and to nothing with no agents', () => {
    expect(pickLandingAgent(agents.slice(1), null, isDefault)).toBe(agents[1])
    expect(pickLandingAgent([], null, isDefault)).toBeNull()
  })
})

describe('agentPageHref', () => {
  it('names the machine only for a remote agent', () => {
    expect(agentPageHref({ connectionId: null, profile: 'ops' })).toBe('/settings?agentProfile=ops')
    expect(agentPageHref({ connectionId: 'homelab', profile: 'ops' })).toBe(
      '/settings?agentProfile=ops&connection=homelab'
    )
  })
})
