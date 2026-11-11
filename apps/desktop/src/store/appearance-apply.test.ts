import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { registerLayoutPresets } from '@/app/contrib/layout-presets'
import { group } from '@/components/pane-shell/tree/model'
import { applyLayoutPreset, snapshotChatLayout } from '@/components/pane-shell/tree/presets'
import { $activePresetId, $layoutTree, $userPlacedPanes } from '@/components/pane-shell/tree/store'
import { $pendingSkinApply } from '@/themes/backend-sync'
import { $pendingModeApply } from '@/themes/request'

import { applyAppearance } from './appearance-apply'
import { $interfaceMode, setInterfaceMode } from './interface-mode'
import { $paneStates } from './panes'
import { $shellChoice } from './shell-variant'

describe('applyAppearance', () => {
  beforeAll(() => {
    registerLayoutPresets()
  })

  afterEach(() => {
    $pendingSkinApply.set(null)
    $pendingModeApply.set(null)
    $shellChoice.set(null)
  })

  // An unpicked skin resolves to the shell's own default, so the accent is saved on that skin.
  it.each([
    { shell: 'classic', choice: false, saved: 'nous-accent-8a2be2' },
    { shell: 'redesign', choice: null, saved: 'portal-accent-8a2be2' }
  ])('applies every value and reports the ids the agent reads ($shell)', ({ choice, saved }) => {
    $shellChoice.set(choice)
    setInterfaceMode('simple')

    expect(applyAppearance({ accent: 'violet', layout: 'terminal-deck', mode: 'dark' })).toEqual({
      applied: { accent: 'violet', layout: 'terminal-deck', mode: 'dark' },
      success: true
    })
    expect($pendingSkinApply.get()).toBe(saved)
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

  it('restores the arrangement and pane preferences when a layout preview is cancelled', () => {
    setInterfaceMode('advanced')
    applyLayoutPreset('mine', group(['workspace']))
    $paneStates.set({ terminal: { open: false, heightOverride: 240 }, files: { open: true, widthOverride: 320 } })
    $userPlacedPanes.set(new Set(['terminal']))
    const tree = $layoutTree.get()
    const panes = $paneStates.get()
    const placed = $userPlacedPanes.get()
    const preview = snapshotChatLayout()

    applyLayoutPreset('preview', group(['sessions', 'workspace']))
    $paneStates.set({ terminal: { open: true, heightOverride: 180 } })
    $userPlacedPanes.set(new Set(['files']))
    preview.revert()

    expect($layoutTree.get()).toBe(tree)
    expect($paneStates.get()).toBe(panes)
    expect($userPlacedPanes.get()).toBe(placed)
    expect($activePresetId.get()).toBe('mine')
    expect($interfaceMode.get()).toBe('advanced')
  })

  it("keeps the outgoing mode's own layout when committing a preview switches modes", () => {
    setInterfaceMode('advanced')
    applyLayoutPreset('mine', group(['workspace']))
    const mine = $layoutTree.get()
    const preview = snapshotChatLayout()

    applyLayoutPreset('sidebar-left', group(['sessions', 'workspace']))

    expect(preview.keep(() => applyAppearance({ layout: 'sidebar-left' }).success)).toBe(true)
    expect($interfaceMode.get()).toBe('simple')
    expect($activePresetId.get()).toBe('sidebar-left')

    setInterfaceMode('advanced')
    expect($layoutTree.get()).toEqual(mine)
    expect($activePresetId.get()).toBe('mine')
  })
})
