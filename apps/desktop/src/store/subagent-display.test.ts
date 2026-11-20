import { describe, expect, it } from 'vitest'

import {
  $subagentOpenMode,
  normalizeSubagentOpenMode,
  setSubagentOpenModeFromConfig,
  showSubagentsConfigValue,
  subagentSidebarMode
} from './subagent-display'

describe('sessions.show_subagents', () => {
  it('reads booleans as before and "active" as the running-only mode', () => {
    expect(subagentSidebarMode(false)).toBe('none')
    expect(subagentSidebarMode(undefined)).toBe('none')
    expect(subagentSidebarMode('false')).toBe('none')
    expect(subagentSidebarMode('bogus')).toBe('none')
    expect(subagentSidebarMode('active')).toBe('active')
    expect(subagentSidebarMode(' Active ')).toBe('active')
    expect(subagentSidebarMode(true)).toBe('all')
    expect(subagentSidebarMode('true')).toBe('all')
    expect(subagentSidebarMode(1)).toBe('all')
  })

  it('writes booleans for none/all and the string for active', () => {
    expect(showSubagentsConfigValue('none')).toBe(false)
    expect(showSubagentsConfigValue('active')).toBe('active')
    expect(showSubagentsConfigValue('all')).toBe(true)
  })
})

describe('desktop.subagent_open', () => {
  it('reads only "tab" as tab; anything else keeps the window default', () => {
    expect(normalizeSubagentOpenMode('tab')).toBe('tab')
    expect(normalizeSubagentOpenMode(' Tab ')).toBe('tab')
    expect(normalizeSubagentOpenMode('window')).toBe('window')
    expect(normalizeSubagentOpenMode(undefined)).toBe('window')
    expect(normalizeSubagentOpenMode(true)).toBe('window')
  })

  it('publishes the config value', () => {
    setSubagentOpenModeFromConfig('tab')
    expect($subagentOpenMode.get()).toBe('tab')
    setSubagentOpenModeFromConfig(undefined)
    expect($subagentOpenMode.get()).toBe('window')
  })
})
