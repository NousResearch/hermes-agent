import { describe, expect, it } from 'vitest'

import { movedSettingsTabRedirect } from './moved-tabs'

describe('movedSettingsTabRedirect', () => {
  it('sends the retired Settings plugin/MCP tabs to Capabilities, keeping the row selector', () => {
    expect(movedSettingsTabRedirect('?tab=plugins')).toBe('/capabilities?tab=plugins')
    expect(movedSettingsTabRedirect('?tab=plugins&plugin=demo%2Fplugin')).toBe(
      '/capabilities?tab=plugins&plugin=demo%2Fplugin'
    )
    expect(movedSettingsTabRedirect('?tab=mcp&server=github')).toBe('/capabilities?tab=connectors&server=github')
  })

  it('leaves live Settings tabs alone', () => {
    expect(movedSettingsTabRedirect('?tab=providers&pview=keys')).toBeNull()
    expect(movedSettingsTabRedirect('')).toBeNull()
  })
})
