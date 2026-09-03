import { describe, expect, it } from 'vitest'

import { quickEntryAgentVisual } from './quick-entry-agent-visual'

describe('quickEntryAgentVisual', () => {
  it('maps the configured Hermes profiles to distinct poses', () => {
    const profiles = ['default', 'gary', 'jarvis', 'repokeeper', 'sabiska', 'warren']
    const poses = profiles.map(profile => quickEntryAgentVisual(profile).pose)

    expect(new Set(poses).size).toBe(profiles.length)
  })

  it('normalizes profile names and falls back to the commander', () => {
    expect(quickEntryAgentVisual(' JARVIS ')).toEqual(quickEntryAgentVisual('jarvis'))
    expect(quickEntryAgentVisual('future-agent')).toEqual(quickEntryAgentVisual('default'))
  })
})
