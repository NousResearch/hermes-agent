import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerLayoutPresets } from '@/app/contrib/layout-presets'
import { group } from '@/components/pane-shell/tree/model'
import { applyLayoutPreset } from '@/components/pane-shell/tree/presets'
import { $activePresetId, $layoutTree } from '@/components/pane-shell/tree/store'
import { applyAppearance } from '@/store/appearance-apply'
import { $interfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { $chatOnboardingSolo, $introView } from '@/store/onboarding-intro'

import { openDemoLayout, snapshotChatLayout } from './assembly'
import { openIntro, skipIntro } from './intro'

const size = vi.fn()

beforeAll(() => {
  registerLayoutPresets()
})

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { chatOnboarding: { size } }
  })
  $introView.set('off')
  openDemoLayout()
  openIntro(false)
  size.mockClear()
})

describe('first-run layout beat', () => {
  it('keeps the intro through a preview and grows the window when a layout is committed', () => {
    setInterfaceMode('advanced')
    applyLayoutPreset('picked', group(['workspace']))
    expect($introView.get()).toBe('intro')
    expect(size).not.toHaveBeenCalled()

    expect(applyAppearance({ layout: 'sidebar-left' }).success).toBe(true)

    expect(size).toHaveBeenCalledWith('normal')
    expect($chatOnboardingSolo.get()).toBe(false)
    expect($introView.get()).toBe('ended')
    expect($interfaceMode.get()).toBe('simple')
    expect($activePresetId.get()).toBe('sidebar-left')
  })

  it("keeps the other mode's own layout when Continue switches modes after a preview", () => {
    skipIntro()
    setInterfaceMode('advanced')
    applyLayoutPreset('mine', group(['workspace']))
    const mine = $layoutTree.get()

    const preview = snapshotChatLayout()
    applyLayoutPreset('sidebar-left', group(['sessions', 'workspace']))
    preview.keep(() => applyAppearance({ layout: 'sidebar-left' }).success)
    expect($interfaceMode.get()).toBe('simple')

    setInterfaceMode('advanced')
    expect($layoutTree.get()).toEqual(mine)
  })

  it('keeps the Advanced layout the user had when Basic is picked over the demo', () => {
    skipIntro()
    setInterfaceMode('advanced')
    applyLayoutPreset('mine', group(['workspace']))
    const mine = $layoutTree.get()

    $introView.set('off')
    openDemoLayout()
    openIntro(false)
    expect($layoutTree.get()).not.toEqual(mine)

    expect(applyAppearance({ layout: 'sidebar-left' }).success).toBe(true)
    expect($interfaceMode.get()).toBe('simple')

    setInterfaceMode('advanced')
    expect($layoutTree.get()).toEqual(mine)
  })

  it('sizes the window normal on Skip setup', () => {
    skipIntro()

    expect(size).toHaveBeenCalledWith('normal')
    expect($chatOnboardingSolo.get()).toBe(false)
  })
})
