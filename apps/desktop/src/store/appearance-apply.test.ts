import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { registerLayoutPresets } from '@/app/contrib/layout-presets'
import { $activePresetId } from '@/components/pane-shell/tree/store'
import { $pendingSkinApply } from '@/themes/backend-sync'
import { $pendingModeApply } from '@/themes/request'

import { applyAppearance } from './appearance-apply'
import { $interfaceMode, setInterfaceMode } from './interface-mode'

describe('applyAppearance', () => {
  beforeAll(() => {
    registerLayoutPresets()
  })

  afterEach(() => {
    $pendingSkinApply.set(null)
    $pendingModeApply.set(null)
  })

  it('applies every value and reports the ids the agent reads', () => {
    setInterfaceMode('simple')

    expect(applyAppearance({ accent: 'violet', layout: 'terminal-deck', mode: 'dark' })).toEqual({
      applied: { accent: 'violet', layout: 'terminal-deck', mode: 'dark' },
      success: true
    })
    expect($pendingSkinApply.get()).toBe('nous-accent-8a2be2')
    expect($pendingModeApply.get()).toBe('dark')
    expect($activePresetId.get()).toBe('terminal-deck')
    expect($interfaceMode.get()).toBe('advanced')
  })

  it('applies nothing when one value is unknown, and names what exists', () => {
    const answer = applyAppearance({ accent: 'violet', layout: 'cockpit' })

    expect(answer.success).toBe(false)
    expect(!answer.success && answer.available).toEqual(expect.arrayContaining(['sidebar-left', 'terminal-deck']))
    expect($pendingSkinApply.get()).toBeNull()
  })
})
